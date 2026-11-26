"""LoD2 3D buildings (measured walls + real roof planes) as a source next to OSM / Overture (M10).

Regions (``REGIONS``): ``bw`` Baden-Wuerttemberg (LGL CityGML 1.0 zips), ``nl`` 3DBAG (CityJSON 2.0 via OGC API),
``jp`` PLATEAU (CityGML 2.0, 3rd-mesh files), ``ch`` swissBUILDINGS3D 3.0 (CityGML 2.0, 2021 sheets; experimental).
Every adapter emits the same per-building record (see ``SCHEMA``) in the site frame (UTM metres minus origin).

Heights are NEVER absolute: vertical datums (DHHN2016 / NAP / LN02 / Tokyo-Bay MSL) differ from the DEM, so every
height is an offset above the building's own LoD2 ground z (= min z of its GroundSurfaces). The consumer puts the
building on ``base_z`` from the site DEM.

Parsing is streaming (``xml.etree.ElementTree.iterparse``, local names only so CityGML 1.0 and 2.0 both work).
Never blocks the OSM path: ``try_fetch`` / ``ensure`` turn any failure into a meta warning (same pattern as
``overture.py``).
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from .dem_sources import atomic_path

log = logging.getLogger("osmctx.lod2")

# browser-like UA: LGL answers 403 to the default python / curl agents
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 osm-context-archicad/0.1"
FILE = "lod2.json"
SCHEMA_VERSION = 1
# The builder (features.LOD2_*) decides what becomes a Roof / Morph / block top; the producer keeps its own
# classification (flat / pitched / fits) only as metadata and drops nothing but (near-)vertical surfaces
# (nz < 0.05). Constants below equal the builder's: flat < 1 deg, single-plane roof <= 75 deg, min 0.25 m2.
FLAT_TILT_DEG = 1.0          # roof surfaces flatter than this are "flat roofs" (= features.LOD2_FLAT_DEG)
MERGE_NORMAL_DEG = 2.0       # coplanar merge: normals closer than this ...
MERGE_OFFSET_M = 0.05        # ... and planes closer than this
TOUCH_TOL_M = 0.03           # ... and footprints touching within this
PLANAR_TOL_M = 0.05          # vertex distance to the Newell plane above which a surface is "not planar"
MAX_TILT_DEG = 75.0          # steeper surfaces do not fit a single-plane Roof (fits = False; = features.LOD2_VERTICAL_DEG)
MIN_ROOF_AREA_M2 = 0.25      # (= features.LOD2_MIN_PLANE_AREA)
PAGE_CAP = 50                # 3DBAG pages followed per fetch (50 features each)
PAGE_LIMIT = 50
TIME_BOX_S = {"bw": 120.0, "nl": 120.0, "jp": 900.0, "ch": 1800.0}

LGL_URL = "https://opengeodata.lgl-bw.de/data/lod2/LoD2_32_{e0}_{n0}_2_bw.zip"
BAG_API = "https://api.3dbag.nl/collections/pand/items"
PLATEAU_API = "https://api.plateauview.mlit.go.jp/datacatalog/citygml/r:{w},{s},{e},{n}?types=bldg"
SWISS_STAC = "https://data.geo.admin.ch/api/stac/v1/collections/ch.swisstopo.swissbuildings3d_3_0/items"

# Horizontal CRS used for the transform when the data's own CRS is compound / 3D (z is passed through untouched).
_HORIZ = {6697: 6668, 7415: 28992}

# region table. status: working = parsed against real data end to end; experimental = see ``note``.
REGIONS: dict[str, dict] = {
    "bw": {
        "name": "Baden-Wuerttemberg (LGL)", "source": "lod2:lgl-bw", "dataset": "LGL-BW LoD2",
        "src_epsg": 25832, "axis": "xy", "format": "CityGML 1.0", "status": "working", "note": "",
        "credit": "Datengrundlage: LGL, www.lgl-bw.de", "licence": "DL-DE BY 2.0", "credit_confirmed": True,
        "download_mb": 15, "large": False},
    "nl": {
        "name": "Netherlands (3DBAG)", "source": "lod2:3dbag", "dataset": "3DBAG",
        "src_epsg": 7415, "axis": "xy", "format": "CityJSON 2.0", "status": "working", "note": "",
        # TODO(confirm): exact credit wording (docs.3dbag.nl asks for the notice with the 3D Tiles).
        "credit": "3DBAG by tudelft3d and 3DGI", "licence": "CC BY 4.0", "credit_confirmed": False,
        "download_mb": 10, "large": False},
    "jp": {
        "name": "Japan (PLATEAU)", "source": "lod2:plateau", "dataset": "PLATEAU",
        "src_epsg": 6697, "axis": "yx", "format": "CityGML 2.0 (uro)", "status": "working", "note": "",
        # TODO(confirm): credit wording to confirm on the first PLATEAU build (MLIT terms).
        "credit": "3D city model PLATEAU, Ministry of Land, Infrastructure, Transport and Tourism (MLIT), Japan",
        "licence": "CC BY 4.0 / Government Standard Terms of Use v2.0", "credit_confirmed": False,
        "download_mb": 100, "large": True},
    "ch": {
        "name": "Switzerland / Liechtenstein (swissBUILDINGS3D 3.0)", "source": "lod2:swisstopo",
        "dataset": "swissBUILDINGS3D 3.0 Beta", "src_epsg": 2056, "axis": "xy", "format": "CityGML 2.0",
        "status": "experimental",
        "note": "CityGML exists only for the 2021 sheets (~169 MB zip, 1.6 GB XML); newer data is a national FileGDB",
        # TODO(confirm): swisstopo OGD wording ("open use, source required").
        "credit": "© swisstopo", "licence": "swisstopo OGD (open use, source must be named)",
        "credit_confirmed": False, "download_mb": 169, "large": True},
}

# per-building record (the contract with convert.py). All coordinates in the site frame (metres), heights are
# offsets above the building's own LoD2 ground z. Plain JSON only.
SCHEMA = {
    "id": "str, source native (DEBW_.., NL.IMBAG.Pand.x, gml:id/EGID, uro:buildingID)",
    "source": "lod2:lgl-bw|3dbag|swisstopo|plateau",
    "lod": "2 | 1 (LoD1 fallback: block only, parts[].roof_planes empty)",
    "crs_src": "EPSG:xxxx the data came in",
    "footprint": "[[x,y]] open ring, CCW, m", "holes": "[[[x,y]]]", "bbox": "[xmin,ymin,xmax,ymax]",
    "ground_z_src": "m, source datum, informational only",
    "eave_h": "m above ground (largest part)", "ridge_h": "m above ground (max over parts)",
    "height": "m above ground = measuredHeight if given else ridge_h",
    "parts": [{
        "id": "str", "footprint": "[[x,y]]", "holes": "[]", "min_h": "m (part ground above building ground)",
        "eave_h": "m", "top_h": "m", "flat_top_h": "m | null (highest flat roof)",
        "roof_planes": [{
            "polygon": "[[x,y]] CCW", "holes": "[]", "tilt": "rad", "azimuth": "rad, compass bearing of the downslope",
            "z0_h": "m above ground at the lowest vertex (pivot)", "pivot": "[[x,y],[x,y]] Roof-tool pivot line",
            "area": "m2 true surface area", "zmax_h": "m", "normal": "[nx,ny,nz] nz>0",
            "p0": "[x,y] lowest vertex; plane: z_h = z0_h - (nx (x-p0x) + ny (y-p0y)) / nz", "dev": "m max vertex distance to the plane", "planar": "bool", "fits": "bool"}],
        "flat_roofs": [{"polygon": "[[x,y]]", "holes": "[]", "z0_h": "m", "zmax_h": "m", "area": "m2",
                        "tilt": "rad", "normal": "[..]", "p0": "[x,y]", "dev": "m"}],
        "roof_raw": "int, roof polygons before the coplanar merge"}],
    "storeys": "int|null", "year": "int|null", "function": "str|null", "attrs": "dict raw codes",
    "licence": "str", "credit": "str",
}


class LoD2Error(RuntimeError):
    pass


class LargeDownload(LoD2Error):
    """A multi-ten-MB download was not allowed (``allow_large=False``)."""


# ------------------------------------------------------------------------------------------ small helpers
def _ln(tag: str) -> str:
    return tag.rpartition("}")[2]


def _r3(a) -> list:
    return np.round(np.asarray(a, float), 3).tolist()


def _ring_open(a) -> list:
    """Closed or open ndarray ring -> open [[x, y]] list (mm)."""
    a = np.asarray(a, float)[:, :2]
    if len(a) > 1 and float(np.abs(a[0, :2] - a[-1, :2]).max()) <= 1e-6:
        a = a[:-1]
    return _r3(a)


def _newell(p: np.ndarray) -> np.ndarray:
    q = np.roll(p, -1, axis=0)
    return np.array([
        np.sum((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])),
        np.sum((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])),
        np.sum((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])),
    ])


def _poly_xy(ext: np.ndarray, holes=()) -> Polygon | None:
    try:
        p = Polygon(ext[:, :2], [h[:, :2] for h in holes if len(h) >= 3])
    except Exception:  # noqa: BLE001
        return None
    if not p.is_valid:
        p = p.buffer(0)
    return _largest(p)


def _largest(g) -> Polygon | None:
    if g is None or g.is_empty:
        return None
    if isinstance(g, Polygon):
        return g
    polys = [x for x in getattr(g, "geoms", []) if isinstance(x, Polygon) and not x.is_empty]
    return max(polys, key=lambda x: x.area) if polys else None


def _poly_json(poly: Polygon) -> tuple[list, list]:
    poly = orient(poly, 1.0)
    return _ring_open(np.array(poly.exterior.coords)), [_ring_open(np.array(i.coords)) for i in poly.interiors]


def _utm_epsg(lon: float, lat: float) -> int:
    try:
        from .convert import utm_epsg   # the one helper the site frame uses
        return int(utm_epsg(lon, lat))
    except Exception:  # noqa: BLE001  (convert import failure must not break LoD2)
        zone = min(int((lon + 180.0) // 6.0) + 1, 60)
        return (32600 if lat >= 0 else 32700) + zone


class _Proj:
    """Raw coordinates (axis order of the source, z untouched) -> site frame (UTM metres minus origin)."""

    def __init__(self, src_epsg: int, out_epsg: int | None = None, origin=(0.0, 0.0), axis: str = "xy"):
        self.src_epsg, self.out_epsg, self.origin, self.axis = src_epsg, out_epsg, origin, axis
        self.tf = None
        src = _HORIZ.get(src_epsg, src_epsg)
        if out_epsg is not None and out_epsg != src:
            self.tf = Transformer.from_crs(f"EPSG:{src}", f"EPSG:{out_epsg}", always_xy=True)

    def xy(self, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw columns -> source-CRS (x, y) with axis order normalised (lon/lat for yx)."""
        return (a[:, 1], a[:, 0]) if self.axis == "yx" else (a[:, 0], a[:, 1])

    def __call__(self, a) -> np.ndarray:
        a = np.asarray(a, float)
        x, y = self.xy(a)
        if self.tf is not None:
            x, y = self.tf.transform(x, y)
        return np.column_stack([np.asarray(x) - self.origin[0], np.asarray(y) - self.origin[1], a[:, 2]])


# ------------------------------------------------------------------------------------------ planes
def _plane_record(poly: Polygon, n: np.ndarray, c: float, dev: float) -> dict | None:
    """Plane n . (x, y, z_h) = c over the XY polygon -> roof record (pitched or flat)."""
    poly = orient(poly, 1.0)
    ext = np.array(poly.exterior.coords)[:-1]
    if len(ext) < 3 or n[2] < 1e-6:
        return None
    z = (c - n[0] * ext[:, 0] - n[1] * ext[:, 1]) / n[2]
    i0 = int(np.argmin(z))
    tilt = math.acos(max(-1.0, min(1.0, float(n[2]))))
    area = poly.area / float(n[2])
    rec = {"polygon": _ring_open(ext), "holes": [_ring_open(np.array(i.coords)) for i in poly.interiors],
           "tilt": round(tilt, 5), "area": round(area, 3), "zmax_h": round(float(z.max()), 3),
           "normal": [round(float(v), 6) for v in n], "dev": round(float(dev), 4)}
    if math.degrees(tilt) < FLAT_TILT_DEG:
        cen = poly.centroid
        rec.update(p0=_r3([cen.x, cen.y]), z0_h=round(float((c - n[0] * cen.x - n[1] * cen.y) / n[2]), 3), flat=True)
        return rec
    nh = math.hypot(n[0], n[1])
    u = np.array([-n[0], -n[1]]) / nh                      # uphill, horizontal
    d = np.array([u[1], -u[0]])                            # pivot direction; left of d = uphill (Roof convention)
    p0 = ext[i0, :2]
    half = float(np.max(np.abs((ext[:, :2] - p0) @ d))) + 0.5
    az = math.atan2(n[0], n[1]) % (2 * math.pi)            # downslope compass bearing (cw from north)
    rec.update(azimuth=round(az, 5), p0=_r3(p0), z0_h=round(float(z[i0]), 3),
               pivot=[_r3(p0 - d * half), _r3(p0 + d * half)], planar=dev <= PLANAR_TOL_M,
               fits=bool(dev <= PLANAR_TOL_M and math.degrees(tilt) <= MAX_TILT_DEG and area >= MIN_ROOF_AREA_M2),
               flat=False)
    return rec


def _rec_poly(rec: dict) -> Polygon:
    return Polygon(rec["polygon"], rec.get("holes") or [])


def _plane_z(rec: dict, xy: np.ndarray) -> np.ndarray:
    """z_h of the record's plane at xy (N, 2). Anchored at (p0, z0_h): well-conditioned whatever the frame origin."""
    n, p0 = rec["normal"], rec["p0"]
    return rec["z0_h"] - (n[0] * (xy[:, 0] - p0[0]) + n[1] * (xy[:, 1] - p0[1])) / n[2]


def _plane_c(rec: dict) -> float:
    n, p0 = rec["normal"], rec["p0"]
    return n[0] * p0[0] + n[1] * p0[1] + n[2] * rec["z0_h"]


def merge_coplanar(planes: list[dict], normal_tol_deg: float = MERGE_NORMAL_DEG, offset_tol: float = MERGE_OFFSET_M,
                   touch_tol: float = TOUCH_TOL_M) -> list[dict]:
    """Merge adjacent roof polygons of ONE building whose normals differ < ``normal_tol_deg`` and whose planes lie
    within ``offset_tol`` metres of each other (cuts the Roof element count). Works on plane records (pitched or
    flat); the merged plane is the area-weighted refit, the polygon the union of the members. Order is stable."""
    n = len(planes)
    if n < 2:
        return list(planes)
    polys = [_rec_poly(p) for p in planes]
    nrm = [np.asarray(p["normal"], float) for p in planes]
    cos_tol = math.cos(math.radians(normal_tol_deg))
    ext_xy = [np.array(g.exterior.coords)[:-1] for g in polys]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def off_ok(a, b):   # all vertices of b within offset_tol of plane a
        xy = ext_xy[b]
        z = _plane_z(planes[b], xy)
        dist = np.abs(nrm[a][0] * xy[:, 0] + nrm[a][1] * xy[:, 1] + nrm[a][2] * z - _plane_c(planes[a]))
        return float(dist.max()) <= offset_tol

    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j) or float(nrm[i] @ nrm[j]) < cos_tol:
                continue
            if polys[i].distance(polys[j]) > touch_tol:
                continue
            if off_ok(i, j) and off_ok(j, i):
                parent[find(j)] = find(i)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out: list[tuple[int, dict]] = []
    for members in groups.values():
        if len(members) == 1:
            out.append((members[0], planes[members[0]]))
            continue
        w = np.array([max(polys[i].area, 1e-9) for i in members])
        nn = sum(wi * nrm[i] for wi, i in zip(w, members))
        nn = nn / np.linalg.norm(nn)
        cs = []
        for i in members:
            cen = polys[i].representative_point()
            z = float(_plane_z(planes[i], np.array([[cen.x, cen.y]]))[0])
            cs.append(nn[0] * cen.x + nn[1] * cen.y + nn[2] * z)
        c = float(np.average(cs, weights=w))
        dev = max(float(planes[i].get("dev", 0.0)) for i in members)
        u = unary_union([polys[i] for i in members])
        if not isinstance(u, Polygon):
            u = unary_union([polys[i].buffer(touch_tol, join_style=2) for i in members]).buffer(-touch_tol, join_style=2)
        parts = [u] if isinstance(u, Polygon) else [g for g in getattr(u, "geoms", []) if isinstance(g, Polygon)]
        for k, g in enumerate(sorted(parts, key=lambda x: -x.area)):
            rec = _plane_record(g, nn, c, dev)
            if rec is not None:
                out.append((members[0] + k * 1e-3, rec))
    out.sort(key=lambda t: t[0])
    return [r for _, r in out]


def _roof_records(roofs: list[tuple], ground_z: float) -> list[dict]:
    """[(ext Nx3, holes)] absolute z -> plane records (heights above ``ground_z``). Degenerate / vertical dropped."""
    recs = []
    for ext, holes in roofs:
        if len(ext) < 3:
            continue
        nv = _newell(ext)
        ln = float(np.linalg.norm(nv))
        if ln < 1e-6:
            continue
        nu = nv / ln
        if nu[2] < 0:
            nu = -nu
        if nu[2] < 0.05:        # (near) vertical "roof" polygon (e.g. canopy side faces): not a roof plane
            continue
        dev = float(np.max(np.abs((ext - ext.mean(axis=0)) @ nu)))
        poly = _poly_xy(ext, holes)
        if poly is None or poly.area < 1e-4:
            continue
        c_abs = float(nu @ ext.mean(axis=0))
        rec = _plane_record(poly, nu, c_abs - nu[2] * ground_z, dev)
        if rec is not None:
            recs.append(rec)
    return recs


# ------------------------------------------------------------------------------------------ record assembly
_KIND = {"RoofSurface": "roof", "WallSurface": "wall", "GroundSurface": "ground"}


def _classify(ext: np.ndarray, zmin: float) -> str:
    """Kind of an unlabelled solid face from its (outward) normal."""
    nv = _newell(ext)
    ln = float(np.linalg.norm(nv))
    if ln < 1e-9:
        return "other"
    nz = float(nv[2] / ln)
    if abs(nz) < 0.05:
        return "wall"
    if nz < 0 or (abs(nz) > 0.99 and float(ext[:, 2].mean()) <= zmin + 0.05):
        return "ground"
    return "roof"


def _part_geom(surfs: list[tuple]) -> dict | None:
    """surfs = [(kind, ext Nx3, holes)] in the site frame -> geometry summary of one part (absolute z)."""
    if not surfs:
        return None
    allz = np.concatenate([s[1][:, 2] for s in surfs])
    zmin = float(allz.min())
    fixed = []
    for kind, ext, holes in surfs:
        if kind == "unknown":
            kind = _classify(ext, zmin)
        fixed.append((kind, ext, holes))
    grounds = [s for s in fixed if s[0] == "ground"]
    roofs = [(s[1], s[2]) for s in fixed if s[0] == "roof"]
    foot_src = grounds or [s for s in fixed if s[0] == "roof"]
    polys = [p for p in (_poly_xy(s[1], s[2]) for s in foot_src) if p is not None and p.area > 1e-4]
    if not polys:
        return None
    foot = unary_union(polys)
    if not isinstance(foot, Polygon):
        foot = unary_union([g.buffer(0.01) for g in polys]).buffer(-0.01)
    foot = _largest(foot)
    if foot is None:
        return None
    gz = float(min(s[1][:, 2].min() for s in grounds)) if grounds else zmin
    return {"foot": foot, "ground_z": gz, "roofs": roofs, "zmax": float(allz.max()),
            "has_roof": bool(roofs), "n_ground": len(grounds)}


def _part_record(pid: str, g: dict, bground: float, merge: bool, measured: float | None, lod: int) -> dict:
    roofs = _roof_records(g["roofs"], bground)
    raw = len(roofs)
    if merge:
        roofs = merge_coplanar(roofs)
    pitched = [r for r in roofs if not r.get("flat")]
    flat = [r for r in roofs if r.get("flat")]
    min_h = max(0.0, g["ground_z"] - bground)
    if roofs:
        top = max(r["zmax_h"] for r in roofs)
        eave = min(r["z0_h"] for r in (pitched or flat))
    else:
        top = g["zmax"] - bground
        if measured and lod == 1 and measured > 0.5 and not g.get("solid"):
            top = min_h + measured
        eave = top
    eave = min(max(eave, min_h), top)
    fl, hl = _poly_json(g["foot"])
    for r in flat:
        for k in ("flat", "azimuth", "pivot", "planar", "fits"):
            r.pop(k, None)
    for r in pitched:
        r.pop("flat", None)
    return {"id": pid, "footprint": fl, "holes": hl, "min_h": round(min_h, 3), "eave_h": round(eave, 3),
            "top_h": round(top, 3), "flat_top_h": round(max(r["zmax_h"] for r in flat), 3) if flat else None,
            "roof_planes": pitched, "flat_roofs": flat, "roof_raw": raw}


def _to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _assemble(bid: str, units: list[dict], region: str, crs_src: str, attrs: dict, merge: bool = True,
              measured: float | None = None, storeys=None, year=None, function=None, lod: int = 2) -> dict | None:
    """units = [{"id", "surfs": [(kind, ext, holes)], "measured"}] (site frame) -> one record or None."""
    geoms = []
    for u in units:
        g = _part_geom(u["surfs"])
        if g is not None:
            g["solid"] = bool(u.get("solid", True))
            geoms.append((u, g))
    if not geoms:
        return None
    bground = min(g["ground_z"] for _, g in geoms)
    parts = [_part_record(u["id"], g, bground, merge, u.get("measured"), lod) for u, g in geoms]
    foots = unary_union([g["foot"] for _, g in geoms])
    foot = _largest(foots if isinstance(foots, Polygon) else unary_union([x.buffer(0.01) for _, g in geoms
                                                                         for x in [g["foot"]]]).buffer(-0.01))
    if foot is None:
        return None
    fl, hl = _poly_json(foot)
    big = max(range(len(parts)), key=lambda i: geoms[i][1]["foot"].area)
    ridge = max(p["top_h"] for p in parts)
    ms = measured if measured and measured > 0 else max([u.get("measured") or 0 for u, _ in geoms] + [0]) or None
    reg = REGIONS[region]
    b = foot.bounds
    return {"id": bid, "source": reg["source"], "lod": lod, "crs_src": crs_src, "footprint": fl, "holes": hl,
            "bbox": [round(v, 3) for v in b], "ground_z_src": round(bground, 3),
            "eave_h": parts[big]["eave_h"], "ridge_h": round(ridge, 3),
            "height": round(float(ms), 3) if ms else round(ridge, 3), "parts": parts,
            "storeys": _to_int(storeys), "year": _to_int(year), "function": function, "attrs": attrs,
            "licence": reg["licence"], "credit": reg["credit"]}


# ------------------------------------------------------------------------------------------ CityGML parser
def _ring_arr(lr) -> np.ndarray | None:
    for c in lr:
        nm = _ln(c.tag)
        if nm == "posList" and c.text:
            dim = int(c.get("srsDimension") or lr.get("srsDimension") or 3)
            a = np.array(c.text.split(), float)
            if dim < 2 or a.size % dim:
                return None
            a = a.reshape(-1, dim)
            if dim == 2:
                a = np.column_stack([a, np.zeros(len(a))])
            return a[:, :3]
    pts = [np.array(c.text.split(), float) for c in lr if _ln(c.tag) == "pos" and c.text]
    if pts:
        a = np.array([(p.tolist() + [0.0])[:3] for p in pts])
        return a
    return None


def _drop_close(a: np.ndarray) -> np.ndarray:
    return a[:-1] if len(a) > 1 and float(np.abs(a[0, :2] - a[-1, :2]).max()) <= 1e-6 else a


def _polys_of(el) -> list[tuple[np.ndarray, list]]:
    """Inline gml:Polygon descendants of `el` -> [(exterior Nx3, [hole Nx3])] raw coordinates."""
    out = []
    for p in el.iter():
        if _ln(p.tag) != "Polygon":
            continue
        ext, holes = None, []
        for c in p:
            nm = _ln(c.tag)
            if nm in ("exterior", "outerBoundaryIs"):
                for lr in c:
                    ext = _ring_arr(lr)
            elif nm in ("interior", "innerBoundaryIs"):
                for lr in c:
                    h = _ring_arr(lr)
                    if h is not None and len(_drop_close(h)) >= 3:
                        holes.append(_drop_close(h))
        if ext is None:
            continue
        ext = _drop_close(ext)
        if len(ext) >= 3:
            out.append((ext, holes))
    return out


def _gen_attrs(el, into: dict) -> None:
    name = el.get("name")
    for v in el:
        if _ln(v.tag) == "value" and name and v.text is not None and len(into) < 24:
            into[name] = v.text.strip()[:60]


class _Unit:
    """Raw content of one Building / BuildingPart element (source coordinates)."""

    def __init__(self, el):
        self.id = el.get("{http://www.opengis.net/gml}id") or ""
        self.lod2: list[tuple] = []      # (kind, ext, holes) from boundedBy semantic surfaces
        self.solid2: list[tuple] = []    # inline polygons of lod2Solid / lod2MultiSurface (kind unknown)
        self.solid1: list[tuple] = []    # lod1Solid / lod1MultiSurface
        self.lod0: list[tuple] = []      # lod0FootPrint / lod0RoofEdge
        self.measured: float | None = None
        self.attrs: dict = {}
        self.extra: dict = {}
        self.parts: list = []
        for c in el:
            nm = _ln(c.tag)
            if nm == "boundedBy":
                for s in c:
                    kind = _KIND.get(_ln(s.tag), "other")
                    for sub in s:
                        if _ln(sub.tag) in ("lod2MultiSurface", "lod2Geometry"):
                            self.lod2 += [(kind, e, h) for e, h in _polys_of(sub)]
            elif nm in ("lod2Solid", "lod2MultiSurface"):
                self.solid2 += [("unknown", e, h) for e, h in _polys_of(c)]
            elif nm in ("lod1Solid", "lod1MultiSurface"):
                self.solid1 += [("unknown", e, h) for e, h in _polys_of(c)]
            elif nm in ("lod0FootPrint", "lod0RoofEdge"):
                self.lod0 += [("ground", e, h) for e, h in _polys_of(c)]
            elif nm == "measuredHeight":
                try:
                    self.measured = float(c.text)
                except (TypeError, ValueError):
                    pass
            elif nm in ("roofType", "function", "usage", "class", "storeysAboveGround", "yearOfConstruction",
                        "storeysBelowGround"):
                if c.text and nm not in self.extra:
                    self.extra[nm] = c.text.strip()[:60]
            elif nm.endswith("Attribute") and c.get("name"):
                _gen_attrs(c, self.attrs)
            elif nm == "consistsOfBuildingPart":
                self.parts += [p for p in c if _ln(p.tag) == "BuildingPart"]
            elif "buildingIDAttribute" in nm:
                for x in c.iter():
                    if _ln(x.tag) == "buildingID" and x.text:
                        self.extra["buildingID"] = x.text.strip()

    def surfaces(self) -> tuple[list[tuple], int, bool]:
        """(surfaces in source coords, lod, is_solid) preferring semantic LoD2, then LoD2 solid, then LoD1."""
        if any(k != "other" for k, _, _ in self.lod2):
            return [s for s in self.lod2 if s[0] != "other"], 2, True
        if self.solid2:
            return self.solid2, 2, True
        if self.solid1:
            return self.solid1, 1, True
        if self.lod0:
            return self.lod0, 1, False
        return [], 2, True


def _unit_bounds(units: list[_Unit], proj: _Proj):
    xs, ys = [], []
    for u in units:
        for grp in (u.lod2, u.solid2, u.solid1, u.lod0):
            for _, e, _h in grp:
                x, y = proj.xy(e)
                xs.append((x.min(), x.max()))
                ys.append((y.min(), y.max()))
    if not xs:
        return None
    return (min(a for a, _ in xs), min(a for a, _ in ys), max(b for _, b in xs), max(b for _, b in ys))


def _rect_hit(b, rect) -> bool:
    return rect is None or b is None or not (b[2] < rect[0] or b[0] > rect[2] or b[3] < rect[1] or b[1] > rect[3])


def _open_src(src):
    if isinstance(src, (str, Path)):
        return open(src, "rb"), True
    if isinstance(src, (bytes, bytearray)):
        return io.BytesIO(src), True
    return src, False


def parse_citygml(src, region: str = "bw", *, out_epsg: int | None = None, origin=(0.0, 0.0), src_rect=None,
                  merge: bool = True, src_epsg: int | None = None, axis: str | None = None) -> list[dict]:
    """Stream a CityGML 1.0 / 2.0 file (path, bytes or binary file object) into per-building records.

    ``out_epsg``/``origin``: target projected CRS + origin to subtract (site frame); ``None`` keeps source x/y.
    ``src_rect``: (xmin, ymin, xmax, ymax) in the source CRS (x = lon for lat/lon data); buildings outside are
    skipped before any projection work. Building + BuildingPart, Roof/Wall/GroundSurface, lod2Solid /
    lod2MultiSurface; buildings with only LoD1 geometry come out with ``lod: 1``."""
    reg = REGIONS[region]
    proj = _Proj(src_epsg or reg["src_epsg"], out_epsg, origin, axis or reg["axis"])
    crs_src = f"EPSG:{proj.src_epsg}"
    fh, close = _open_src(src)
    out: list[dict] = []
    try:
        for _ev, el in ET.iterparse(fh, events=("end",)):
            nm = _ln(el.tag)
            if nm == "cityObjectMember":
                el.clear()
                continue
            if nm != "Building":
                continue
            try:
                rec = _building_from_el(el, proj, region, crs_src, src_rect, merge)
            except Exception as ex:  # noqa: BLE001  one bad building must not lose the tile
                log.warning("LoD2: skipped building %s (%s: %s)", el.get("{http://www.opengis.net/gml}id"),
                            type(ex).__name__, ex)
                rec = None
            if rec is not None:
                out.append(rec)
            el.clear()
    finally:
        if close:
            fh.close()
    return out


def _building_from_el(el, proj: _Proj, region: str, crs_src: str, src_rect, merge: bool) -> dict | None:
    top = _Unit(el)
    units = [top] + [_Unit(p) for p in top.parts]
    if not _rect_hit(_unit_bounds(units, proj), src_rect):
        return None
    pu, lod, measured = [], 2, top.measured
    for u in units:
        surfs, ulod, is_solid = u.surfaces()
        if not surfs:
            continue
        if ulod == 1:
            lod = 1
        pu.append({"id": u.id or top.id, "surfs": [(k, proj(e), [proj(h) for h in hs]) for k, e, hs in surfs],
                   "measured": u.measured, "solid": is_solid})
    if not pu:
        return None
    bid = top.extra.get("buildingID") or top.id
    ex = dict(top.extra)
    ex.pop("buildingID", None)
    attrs = {**top.attrs, **{k: v for k, v in ex.items()}}
    return _assemble(bid, pu, region, crs_src, attrs, merge, measured, ex.get("storeysAboveGround"),
                     ex.get("yearOfConstruction"), ex.get("function"), lod)


# ------------------------------------------------------------------------------------------ CityJSON parser (3DBAG)
_CJ_KIND = {"RoofSurface": "roof", "WallSurface": "wall", "GroundSurface": "ground"}
_CJ_ATTRS = ("b3_dak_type", "b3_h_maaiveld", "b3_h_nok", "b3_bouwlagen", "status", "b3_pw_datum", "b3_rmse_lod22",
             "b3_h_dak_min", "b3_h_dak_max")


def _lod_key(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return -1.0


def _cj_surfaces(geom: dict, V: np.ndarray) -> list[tuple]:
    t = geom.get("type")
    b = geom.get("boundaries") or []
    sem = geom.get("semantics") or {}
    surfs_def = sem.get("surfaces") or []
    vals = sem.get("values")
    if t == "Solid":
        shells, vals = b, (vals or [])
    elif t in ("MultiSurface", "CompositeSurface"):
        shells, vals = [b], ([vals] if vals is not None else [])
    else:
        return []
    out = []
    for si, shell in enumerate(shells):
        for fi, face in enumerate(shell):
            idx = None
            try:
                idx = vals[si][fi]
            except (IndexError, TypeError):
                pass
            typ = surfs_def[idx].get("type") if idx is not None and idx < len(surfs_def) else None
            kind = _CJ_KIND.get(typ, "unknown" if typ is None else "other")
            if kind == "other":
                continue
            rings = [_drop_close(V[np.asarray(r, int)]) for r in face if len(r) >= 3]
            if rings and len(rings[0]) >= 3:
                out.append((kind, rings[0], rings[1:]))
    return out


def parse_cityjson(features, meta: dict, region: str = "nl", *, out_epsg: int | None = None, origin=(0.0, 0.0),
                   src_rect=None, merge: bool = True) -> list[dict]:
    """3DBAG CityJSON(Feature)s -> records. ``meta`` = the page's ``metadata`` object (``transform`` scale /
    translate, applied as v * scale + translate). Picks the highest LoD 2.x geometry of each BuildingPart (LoD1.x
    fallback -> ``lod: 1``), semantics per face. ``features``: iterable of CityJSONFeature dicts."""
    reg = REGIONS[region]
    tr = meta.get("transform") or {}
    scale = np.array(tr.get("scale") or [1.0, 1.0, 1.0], float)
    transl = np.array(tr.get("translate") or [0.0, 0.0, 0.0], float)
    proj = _Proj(reg["src_epsg"], out_epsg, origin, reg["axis"])
    crs_src = f"EPSG:{reg['src_epsg']}"
    out = []
    for feat in features:
        verts = np.asarray(feat.get("vertices") or [], float)
        if verts.size == 0:
            continue
        V = proj(verts * scale + transl)
        cos = feat.get("CityObjects") or {}
        for oid, co in cos.items():
            if co.get("type") != "Building":
                continue
            units, lod = [], 2
            kids = [(k, cos[k]) for k in co.get("children") or [] if k in cos] or [(oid, co)]
            for kid_id, kid in kids:
                geoms = [g for g in kid.get("geometry") or [] if g.get("type") in ("Solid", "MultiSurface")
                         and _lod_key(g.get("lod")) >= 1.0]
                if not geoms:
                    continue
                best = max(geoms, key=lambda g: _lod_key(g.get("lod")))
                lod = min(lod, 2 if _lod_key(best.get("lod")) >= 2.0 else 1)
                surfs = _cj_surfaces(best, V)
                if surfs:
                    units.append({"id": kid_id, "surfs": surfs, "measured": None, "solid": True})
            if not units:
                continue
            if src_rect is not None:
                allxy = np.concatenate([s[1][:, :2] for u in units for s in u["surfs"]]) + np.array(origin)
                if allxy[:, 0].max() < src_rect[0] or allxy[:, 0].min() > src_rect[2] or \
                        allxy[:, 1].max() < src_rect[1] or allxy[:, 1].min() > src_rect[3]:
                    continue
            at = co.get("attributes") or {}
            attrs = {k: at[k] for k in _CJ_ATTRS if at.get(k) is not None}
            rec = _assemble(oid, units, region, crs_src, attrs, merge, None, at.get("b3_bouwlagen"),
                            at.get("oorspronkelijkbouwjaar"), None, lod)
            if rec is not None:
                out.append(rec)
    return out


# ------------------------------------------------------------------------------------------ HTTP / cache
def _get(url: str, **kw):
    """Single HTTP entry point (tests monkeypatch this)."""
    headers = {"User-Agent": UA, **kw.pop("headers", {})}
    return requests.get(url, headers=headers, **kw)


def _head(url: str, timeout: float = 15.0) -> int:
    return requests.head(url, headers={"User-Agent": UA}, timeout=timeout, allow_redirects=True).status_code


def _remaining(deadline: float | None, cap: float = 60.0) -> float:
    if deadline is None:
        return cap
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("LoD2 fetch exceeded its time box")
    return max(1.0, min(cap, left))


def _download(url: str, dest: Path, deadline: float | None, label: str, progress=None) -> int:
    """Stream `url` into `dest` (atomic; cached = kept). Logs progress every ~10 MB. -> bytes."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    r = _get(url, stream=True, timeout=(15, 60))
    try:
        r.raise_for_status()
        # Content-Length is the compressed size when the server gzips the body (PLATEAU does): then it says nothing
        total = 0 if r.headers.get("Content-Encoding") else int(r.headers.get("Content-Length") or 0)
        done, nxt, t0 = 0, 10 << 20, time.monotonic()
        log.info("LoD2: downloading %s%s", label, f" ({total / 1e6:.0f} MB)" if total else "")
        with atomic_path(dest) as tmp, open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(f"LoD2 download of {label} exceeded the time box ({done / 1e6:.0f} MB done)")
                f.write(chunk)
                done += len(chunk)
                if done >= nxt:
                    nxt += 10 << 20
                    msg = (f"{label}: {done / 1e6:.0f}" + (f"/{total / 1e6:.0f}" if total else "") +
                           f" MB, {done / 1e6 / max(1e-3, time.monotonic() - t0):.1f} MB/s")
                    log.info("LoD2: %s", msg)
                    if progress:
                        progress(msg)
    finally:
        r.close()
    return dest.stat().st_size


def _rect(bb: tuple, epsg: int, margin: float) -> tuple[float, float, float, float]:
    """(s, w, n, e) lon/lat -> (xmin, ymin, xmax, ymax) in `epsg` (corner envelope, + margin in that CRS's units)."""
    s, w, n, e = bb
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{_HORIZ.get(epsg, epsg)}", always_xy=True)
    xs, ys = tf.transform([w, e, w, e], [s, s, n, n])
    return min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin


def _site_rect(bb, out_epsg: int, origin, margin: float = 0.0):
    x0, y0, x1, y1 = _rect(bb, out_epsg, margin)
    return x0 - origin[0], y0 - origin[1], x1 - origin[0], y1 - origin[1]


def _dedupe(recs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in recs:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        out.append(r)
    return out


# ------------------------------------------------------------------------------------------ region detection
_BOX = {"bw": (47.5, 7.5, 49.8, 10.5), "ch": (45.8, 5.95, 47.85, 10.55), "nl": (50.7, 3.2, 53.6, 7.3),
        "jp": (24.0, 122.9, 45.6, 154.0)}
RD_ENVELOPE = (10000.0, 306250.0, 287760.0, 623690.0)   # EPSG:7415 (RD) x/y limits of the 3DBAG extent


def lgl_tiles(bb: tuple) -> list[tuple[int, int]]:
    """2 km LGL tiles (E0 odd km, N0 even km, UTM32 km) covering bb = (s, w, n, e) lon/lat."""
    x0, y0, x1, y1 = _rect(bb, 25832, 0.0)
    tiles = set()
    for ek in range(int(x0 // 1000), int(x1 // 1000) + 1):
        for nk in range(int(y0 // 1000), int(y1 // 1000) + 1):
            tiles.add((2 * ((ek - 1) // 2) + 1, 2 * (nk // 2)))
    return sorted(tiles)


def _inside(box, lat, lon) -> bool:
    return box[0] <= lat <= box[2] and box[1] <= lon <= box[3]


def detect_region(bb_lonlat: tuple, head=None) -> str | None:
    """Which LoD2 source covers bb = (south, west, north, east) lon/lat, judged at the bbox centre: ``bw`` (coarse
    box, then HEAD of the LGL tile 200/404), ``ch`` (CH + LI box), ``nl`` (box + RD envelope), ``jp`` (box); else
    None. `head(url) -> status` is injectable for tests. A failing HEAD counts as "not BW"."""
    s, w, n, e = bb_lonlat
    lat, lon = (s + n) / 2, (w + e) / 2
    if _inside(_BOX["bw"], lat, lon):
        try:
            e0, n0 = lgl_tiles((lat, lon, lat, lon))[0]
            if (head or _head)(LGL_URL.format(e0=e0, n0=n0)) == 200:
                return "bw"
        except Exception:  # noqa: BLE001
            pass
    if _inside(_BOX["ch"], lat, lon):
        return "ch"
    if _inside(_BOX["nl"], lat, lon):
        x, y = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True).transform(lon, lat)
        if RD_ENVELOPE[0] <= x <= RD_ENVELOPE[2] and RD_ENVELOPE[1] <= y <= RD_ENVELOPE[3]:
            return "nl"
    if _inside(_BOX["jp"], lat, lon):
        return "jp"
    return None


# ------------------------------------------------------------------------------------------ fetchers
def fetch_lgl(bb: tuple, cache_root: Path, *, out_epsg: int | None = None, origin=(0.0, 0.0),
              deadline: float | None = None, progress=None, merge: bool = True, **_) -> tuple[list[dict], dict]:
    """Baden-Wuerttemberg. Raw 2 km zips cached in ``cache_root/_lod2/bw/`` (shared across sites); only the 1 km GMLs
    intersecting `bb` are parsed (streamed out of the zip)."""
    root = Path(cache_root) / "_lod2" / "bw"
    root.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = _rect(bb, 25832, 0.0)
    cells = {(ek, nk) for ek in range(int(x0 // 1000), int(x1 // 1000) + 1)
             for nk in range(int(y0 // 1000), int(y1 // 1000) + 1)}
    src_rect = (x0 - 5, y0 - 5, x1 + 5, y1 + 5)
    recs, tiles, nbytes = [], [], 0
    for e0, n0 in lgl_tiles(bb):
        name = f"LoD2_32_{e0}_{n0}_2_bw.zip"
        path = root / name
        try:
            nbytes += _download(LGL_URL.format(e0=e0, n0=n0), path, deadline, name, progress)
        except requests.HTTPError as ex:
            if ex.response is not None and ex.response.status_code == 404:
                log.info("LoD2: no LGL tile %s (outside BW)", name)
                continue
            raise
        tiles.append(name)
        with zipfile.ZipFile(path) as zf:
            for zn in zf.namelist():
                low = zn.lower()
                if not low.endswith(".gml"):
                    continue
                parts = Path(zn).stem.split("_")      # LoD2_32_<E>_<N>_1_BW
                try:
                    cell = (int(parts[2]), int(parts[3]))
                except (IndexError, ValueError):
                    cell = None
                if cell is not None and cell not in cells:
                    continue
                _remaining(deadline)
                with zf.open(zn) as fh:
                    recs += parse_citygml(fh, "bw", out_epsg=out_epsg, origin=origin, merge=merge,
                                          src_rect=src_rect)
    if not tiles:
        raise LoD2Error("no LGL LoD2 tile for the bbox (outside Baden-Wuerttemberg?)")
    return recs, {"tiles": tiles, "bytes": nbytes, "dataset_version": "LGL LoD2 (yearly)"}


def _cached_page(url: str, root: Path, deadline: float | None) -> tuple[dict, bool]:
    path = root / (hashlib.sha1(url.encode()).hexdigest()[:20] + ".json")
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8")), True
        except ValueError:
            path.unlink(missing_ok=True)
    r = _get(url, timeout=(15, _remaining(deadline, 60.0)), headers={"Accept": "application/json"})
    r.raise_for_status()
    page = r.json()
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(page), encoding="utf-8")
    return page, False


def fetch_3dbag(bb: tuple, cache_root: Path, *, out_epsg: int | None = None, origin=(0.0, 0.0),
                deadline: float | None = None, progress=None, merge: bool = True, max_pages: int = PAGE_CAP,
                **_) -> tuple[list[dict], dict]:
    """Netherlands: OGC API Features, bbox in EPSG:7415 (RD metres, the API default), 50 features per page, follow
    ``next`` links (page cap ``max_pages``). Pages cached in ``cache_root/_lod2/3dbag/``."""
    root = Path(cache_root) / "_lod2" / "3dbag"
    root.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = _rect(bb, 28992, 0.0)
    url = f"{BAG_API}?bbox={x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f}&limit={PAGE_LIMIT}"
    recs, pages, cached, warns = [], 0, 0, []
    while url:
        if pages >= max_pages:
            warns.append(f"3DBAG page cap {max_pages} reached ({max_pages * PAGE_LIMIT} buildings): site truncated")
            break
        page, was_cached = _cached_page(url, root, deadline)
        pages += 1
        cached += was_cached
        recs += parse_cityjson(page.get("features") or [], page.get("metadata") or {}, "nl", out_epsg=out_epsg,
                               origin=origin, merge=merge)
        if progress:
            progress(f"3DBAG page {pages}: {len(recs)} buildings")
        url = next((lk.get("href") for lk in page.get("links") or [] if lk.get("rel") == "next"), None)
    return recs, {"pages": pages, "pages_cached": cached, "warnings": warns}


def mesh3_bounds(code: str) -> tuple[float, float, float, float]:
    """JIS X 0410 3rd mesh code (8 digits) -> (south, west, north, east) degrees (cell 30'' x 45'')."""
    p, q, r, s_, t, u = int(code[0:2]), int(code[2:4]), int(code[4]), int(code[5]), int(code[6]), int(code[7])
    s = p / 1.5 + r * (5 / 60) + t * (30 / 3600)
    w = 100 + q + s_ * (7.5 / 60) + u * (45 / 3600)
    return s, w, s + 30 / 3600, w + 45 / 3600


def plateau_files(catalog: dict, bb: tuple) -> list[dict]:
    """Catalog reply -> distinct bldg files whose 3rd-mesh cell intersects bb = (s, w, n, e)."""
    s, w, n, e = bb
    seen, out = set(), []
    for city in catalog.get("cities") or []:
        for f in (city.get("files") or {}).get("bldg") or []:
            cs, cw, cn, ce = mesh3_bounds(str(f["code"]))
            if cn < s or cs > n or ce < w or cw > e or f["url"] in seen:
                continue
            seen.add(f["url"])
            out.append({**f, "city": city.get("cityName"), "cityCode": city.get("cityCode")})
    return out


def fetch_plateau(bb: tuple, cache_root: Path, *, out_epsg: int | None = None, origin=(0.0, 0.0),
                  deadline: float | None = None, progress=None, merge: bool = True, allow_large: bool = False,
                  **_) -> tuple[list[dict], dict]:
    """Japan: datacatalog -> 3rd-mesh bldg CityGML files for the site (60-140 MB each, no Range support):
    cached raw in ``cache_root/_lod2/jp/``; needs ``allow_large=True`` unless already cached. EPSG:6697 lat, lon, h."""
    s, w, n, e = bb
    r = _get(PLATEAU_API.format(w=w, s=s, e=e, n=n), timeout=(15, _remaining(deadline, 30.0)))
    if r.status_code == 400:
        raise LoD2Error("PLATEAU catalog: bbox covers more than 50 cities")
    r.raise_for_status()
    files = plateau_files(r.json(), bb)
    if not files:
        raise LoD2Error("PLATEAU has no building data for the bbox")
    root = Path(cache_root) / "_lod2" / "jp"
    root.mkdir(parents=True, exist_ok=True)
    todo = [f for f in files if not (root / f"{f['cityCode']}_{f['code']}.gml").exists()]
    need = sum(int(f.get("fileSize") or 0) for f in todo)
    if todo and not allow_large:
        raise LargeDownload(f"PLATEAU needs {need / 1e6:.0f} MB of CityGML ({len(todo)} file(s)); "
                            "enable the large-download option")
    src_rect = _rect(bb, 6668, 0.0005)
    recs, nbytes, lod1 = [], 0, 0
    for f in files:
        path = root / f"{f['cityCode']}_{f['code']}.gml"
        nbytes += _download(f["url"], path, deadline, f"PLATEAU {f['code']}", progress)
        _remaining(deadline)
        got = parse_citygml(path, "jp", out_epsg=out_epsg, origin=origin, merge=merge, src_rect=src_rect)
        lod1 += sum(1 for g in got if g["lod"] == 1)
        recs += got
    return recs, {"tiles": [f["code"] for f in files], "bytes": nbytes, "lod1_only": lod1}


def swiss_items(stac: dict) -> list[dict]:
    """STAC reply -> [{id, href, datetime}] of sheets that have a CityGML asset (2021 sheets only), newest first."""
    out = []
    for it in stac.get("features") or []:
        for k, a in (it.get("assets") or {}).items():
            if k.endswith(".citygml.zip"):
                out.append({"id": it["id"], "href": a["href"], "datetime": (it.get("properties") or {}).get("datetime")})
    out.sort(key=lambda x: x["datetime"] or "", reverse=True)
    seen, uniq = set(), []
    for o in out:
        if o["href"] not in seen:
            seen.add(o["href"])
            uniq.append(o)
    return uniq


def fetch_swiss(bb: tuple, cache_root: Path, *, out_epsg: int | None = None, origin=(0.0, 0.0),
                deadline: float | None = None, progress=None, merge: bool = True, allow_large: bool = False,
                **_) -> tuple[list[dict], dict]:
    """Switzerland (EXPERIMENTAL): STAC items by bbox -> the 2021 CityGML sheet (~169 MB zip, one 1.6 GB GML), cached
    in ``cache_root/_lod2/ch/``; streamed through the parser with a bbox prefilter. Needs ``allow_large``."""
    s, w, n, e = bb
    r = _get(f"{SWISS_STAC}?bbox={w},{s},{e},{n}&limit=100", timeout=(15, _remaining(deadline, 30.0)))
    r.raise_for_status()
    items = swiss_items(r.json())
    if not items:
        raise LoD2Error("swissBUILDINGS3D: no CityGML sheet (only the 2021 sheets ship CityGML) for the bbox")
    root = Path(cache_root) / "_lod2" / "ch"
    root.mkdir(parents=True, exist_ok=True)
    it = items[0]
    path = root / Path(it["href"]).name
    if not path.exists() and not allow_large:
        raise LargeDownload("swissBUILDINGS3D CityGML sheet is ~169 MB (1.6 GB unzipped); enable the large-download option")
    nbytes = _download(it["href"], path, deadline, it["id"], progress)
    src_rect = _rect(bb, 2056, 5.0)
    with zipfile.ZipFile(path) as zf:
        name = max((i for i in zf.infolist() if i.filename.lower().endswith(".gml")), key=lambda i: i.file_size).filename
        with zf.open(name) as fh:
            recs = parse_citygml(fh, "ch", out_epsg=out_epsg, origin=origin, merge=merge, src_rect=src_rect)
    return recs, {"tiles": [it["id"]], "bytes": nbytes}


FETCHERS = {"bw": fetch_lgl, "nl": fetch_3dbag, "jp": fetch_plateau, "ch": fetch_swiss}


# ------------------------------------------------------------------------------------------ attribution
def credit_for(region: str) -> str:
    """One-line credit incl. licence for a region, e.g. ``Datengrundlage: LGL, www.lgl-bw.de (DL-DE BY 2.0)``."""
    r = REGIONS[region]
    return f"{r['credit']} ({r['licence']})"


def attribution_block(region: str) -> dict:
    r = REGIONS[region]
    return {"attribution": f"3D buildings (LoD2): {credit_for(region)}", "licence": r["licence"],
            "credit": r["credit"], "credit_confirmed": r["credit_confirmed"],
            "source_credits": [credit_for(region)]}


# ------------------------------------------------------------------------------------------ estimate (limits)
# priors per km2 (buildings) and roof planes per building after the coplanar merge; BW measured on Tuebingen
# (1070 buildings/km2, 5.5 raw roof surfaces per building -> ~4 merged)
_PRIOR = {"bw": (900, 4.0), "nl": (1100, 3.5), "ch": (600, 4.5), "jp": (2000, 2.5)}


def estimate(site_bbox: tuple, region: str | None = None, records: list[dict] | None = None) -> dict:
    """Expected LoD2 buildings / roof planes for bb = (s, w, n, e). Exact when parsed `records` are given, else a
    per-region density prior (basis ``prior``; upper-bound-ish for dense cores). Use for ``limits.estimate``."""
    if records is not None:
        planes = sum(len(p["roof_planes"]) for r in records for p in r["parts"])
        return {"region": region, "buildings": len(records), "roof_planes": planes, "basis": "records"}
    s, w, n, e = site_bbox
    km2 = max(0.0, (n - s) * 111.32) * max(0.0, (e - w) * 111.32 * math.cos(math.radians((s + n) / 2)))
    region = region or detect_region(site_bbox, head=lambda u: 404)
    if region not in _PRIOR:
        return {"region": region, "buildings": 0, "roof_planes": 0, "basis": "none"}
    dens, per = _PRIOR[region]
    nb = int(math.ceil(km2 * dens))
    return {"region": region, "buildings": nb, "roof_planes": int(math.ceil(nb * per)), "basis": "prior"}


# ------------------------------------------------------------------------------------------ orchestration
def load(path: Path) -> dict:
    """Parsed ``lod2.json`` (keys: region, crs, origin, buildings)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fetch_lod2(bb: tuple, cache_root: Path, out_path: Path, timeout: float | None = None, *, frame=None,
               region: str | None = None, allow_large: bool = False, merge: bool = True, progress=None) -> dict:
    """bb = (south, west, north, east) lon/lat. Writes `out_path` (``lod2.json``: records in the site frame) and returns
    the meta block ``{status, region, dataset, count, roof_planes, lod1, tiles, bytes, credit, warnings, ...}``.
    `frame` = the site ``convert.Frame`` (uses its ``epsg`` and ``E0``/``N0``); without it records are absolute UTM
    of the bbox centre. Raises LoD2Error / LargeDownload / TimeoutError (callers: use ``try_fetch`` / ``ensure``)."""
    t0 = time.monotonic()
    region = region or detect_region(bb)
    if region is None:
        raise LoD2Error("no LoD2 source covers this location (supported: DE-Baden-Wuerttemberg, NL, CH, JP)")
    box = TIME_BOX_S[region] if timeout is None else timeout
    deadline = t0 + box
    if frame is not None:
        epsg, origin = int(frame.epsg), (float(frame.E0), float(frame.N0))
    else:
        epsg, origin = _utm_epsg((bb[1] + bb[3]) / 2, (bb[0] + bb[2]) / 2), (0.0, 0.0)
    recs, info = FETCHERS[region](bb, Path(cache_root), out_epsg=epsg, origin=origin, deadline=deadline,
                                  progress=progress, merge=merge, allow_large=allow_large)
    rect = _site_rect(bb, epsg, origin)
    recs = [r for r in _dedupe(recs) if _rect_hit(tuple(r["bbox"]), rect)]
    warns = list(info.get("warnings") or [])
    if not recs:
        warns.append("no LoD2 buildings inside the bbox")
    reg = REGIONS[region]
    if reg["status"] != "working":
        warns.append(f"{reg['name']} LoD2 is {reg['status']}: {reg['note']}")
    Path(out_path).write_text(json.dumps({"version": SCHEMA_VERSION, "region": region, "crs": f"EPSG:{epsg}",
                                          "origin": list(origin), "buildings": recs}), encoding="utf-8")
    return {"status": "ok", "region": region, "dataset": reg["dataset"], "region_status": reg["status"],
            "count": len(recs), "roof_planes": sum(len(p["roof_planes"]) for r in recs for p in r["parts"]),
            "roof_raw": sum(p["roof_raw"] for r in recs for p in r["parts"]),
            "lod1": sum(1 for r in recs if r["lod"] == 1), "tiles": info.get("tiles"), "bytes": info.get("bytes"),
            **{k: info[k] for k in ("pages", "pages_cached", "lod1_only") if k in info},
            "crs": f"EPSG:{epsg}", "origin": list(origin), "elapsed_s": round(time.monotonic() - t0, 1),
            "fetched_at": datetime.now(timezone.utc).isoformat(), "warnings": warns, **attribution_block(region)}


def meta_bbox(meta: dict) -> tuple:
    """(south, west, north, east) LoD2 is fetched for = the OSM fetch bbox of a cache."""
    bb = meta.get("osm_bbox") or meta["bbox"]
    return bb["south"], bb["west"], bb["north"], bb["east"]


def needs_fetch(d: Path, meta: dict, force: bool = False) -> bool:
    l2 = meta.get("lod2") or {}
    return force or not (l2.get("status") == "ok" and (Path(d) / FILE).exists())


def try_fetch(bb: tuple, cache_root: Path, out_path: Path, timeout: float | None = None, **kw) -> dict:
    """fetch_lod2 that never raises: the meta block, ``{"status": "failed", ...}`` on any error. Touches no meta.json,
    so it can run beside the OSM / DEM / Overture fetches (record the block with ``apply``)."""
    try:
        return fetch_lod2(bb, cache_root, out_path, timeout, **kw)
    except Exception as ex:  # noqa: BLE001  never block the OSM path
        return {"status": "failed", "error": f"{type(ex).__name__}: {ex}",
                "fetched_at": datetime.now(timezone.utc).isoformat()}


def apply(meta: dict, l2: dict) -> None:
    """Record a ``try_fetch`` result in a meta dict: ``lod2`` block + warnings (failure or the block's own)."""
    warns = meta.setdefault("warnings", [])
    warns[:] = [x for x in warns if not x.startswith("LoD2")]
    if l2.get("status") == "failed":
        warns.append(f"LoD2 buildings not fetched ({l2['error'][:160]}): OSM / Overture buildings only")
    else:
        warns.extend(f"LoD2: {x}" for x in l2.get("warnings") or [])
    meta["lod2"] = l2


def ensure(d: Path, cache_root: Path | None = None, timeout: float | None = None, force: bool = False,
           frame=None, allow_large: bool = False) -> dict | None:
    """Fetch LoD2 buildings into cache dir `d` unless already there. Updates meta.json (``lod2`` block + warnings).
    Never raises. Returns the meta block."""
    d = Path(d)
    mp = d / "meta.json"
    meta = json.loads(mp.read_text(encoding="utf-8"))
    if not needs_fetch(d, meta, force):
        return meta["lod2"]
    l2 = try_fetch(meta_bbox(meta), Path(cache_root) if cache_root else d.parent, d / FILE, timeout,
                   frame=frame, allow_large=allow_large)
    apply(meta, l2)
    mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return l2
