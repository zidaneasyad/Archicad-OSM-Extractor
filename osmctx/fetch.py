"""Fetch OSM (Overpass) + a DEM for a boundary, cached.

DEM sources (``fetch(..., dem_source=...)``):
  "copernicus"   Copernicus GLO-30 (global, 1" grid, a SURFACE model incl. buildings/trees)
  "bw_dgm1"      LGL Baden-Wuerttemberg DGM1 (1 m bare-earth terrain model, open data dl-de/by-2-0)
  "file:<path>"  any user-supplied GeoTIFF terrain model (read on its own grid and CRS)
  "auto"         bw_dgm1 when the bbox lies in Baden-Wuerttemberg and the download works,
                 else copernicus (the converter then defaults to ground_mode="bare_earth_approx")

dem.tif is always written on the SOURCE pixel grid (bbox snapped outward to whole source pixels,
no resampling, written with the mosaic's own transform) in the source CRS (EPSG:4326 for GLO-30,
EPSG:25832 for DGM1, native for files). meta.json records dem_crs, attribution, licence, accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

from . import dem_sources as DS
from . import overpass as OV
from . import lod2 as L2
from . import overture as OVT
from .boundary import bbox, square_boundary, to_polygon

UA = DS.UA
BW_TILE_WORKERS = 4  # parallel 1 km tile downloads (LGL throttles per connection, ~25-40 kB/s each)
OVERPASS_URLS = OV.MIRRORS  # raced in parallel (osmctx/overpass.py), same list as the server's count path
DEM_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/"
           "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
           "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")
# fetch margin around the boundary bbox. A surface model gets a wider margin so the converter's
# bare-earth approximation has neighbouring ground cells to interpolate from; OSM is fetched over
# the same bbox as the DEM so building/forest masks cover the whole DEM.
DEM_MARGIN_M = 30.0
DSM_MARGIN_M = 90.0
CACHE_VERSION = 2  # v2: DEM on the source grid + dem_source in the cache key

BW_DGM1_ZIP = "https://opengeodata.lgl-bw.de/data/dgm/dgm1_32_{e}_{n}_2_bw.zip"
BW_NDOM1_ZIP = "https://opengeodata.lgl-bw.de/data/ndom1/ndom1_32_{e}_{n}_2_bw.zip"
BW_ENVELOPE_25832 = (388000.0, 5264000.0, 611000.0, 5520000.0)  # WCS DGM1 coverage envelope (approx. BW)
NODATA = DS.NODATA

DEM_INFO = {
    "copernicus": {
        "name": "Copernicus GLO-30 (copernicus-dem-30m.s3.amazonaws.com)",
        "attribution": "Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH "
                       "2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved",
        "licence": "Copernicus DEM licence (free use incl. commercial, attribution required)",
        "vertical_datum": "EGM2008",
        "accuracy": {"horizontal_m": 30.0, "vertical_m": 5.0, "kind": "DSM",
                     "note": "Copernicus GLO-30 is a SURFACE model (buildings, bridges, vegetation included), "
                             "~30 m grid (1 arc-second; coarser in longitude above 50 deg lat). Expect about "
                             "+-5 m in open terrain and +-10 m or worse in urban/forest areas and on steep slopes."},
    },
    "bw_dgm1": {
        "name": "LGL Baden-Wuerttemberg DGM1 (opengeodata.lgl-bw.de)",
        # LGL quote format (WCS capabilities AccessConstraints): holder + year of retrieval + licence
        "attribution": "LGL-BW ({year}) Datenlizenz Deutschland - Namensnennung - Version 2.0, www.lgl-bw.de",
        "licence": "dl-de/by-2-0 (https://www.govdata.de/dl-de/by-2-0)",
        "vertical_datum": "DHHN2016 (NHN, quasigeoid GCG2016)",
        "accuracy": {"horizontal_m": 1.0, "vertical_m": 0.5, "kind": "DTM",
                     "note": "LGL DGM1: bare-earth terrain model from airborne laser scanning, 1 m grid. "
                             "Tile metadata states 0.15 m accuracy (open terrain); allow +-0.5 m on steep, "
                             "vegetated or built-up ground."},
    },
    "file": {
        "name": "user file",
        "attribution": "user-supplied terrain model",
        "licence": "unknown (user-supplied)",
        "vertical_datum": "unknown",
        "accuracy": {"horizontal_m": None, "vertical_m": None, "kind": "DTM",
                     "note": "User-supplied GeoTIFF, assumed to be a bare-earth terrain model (DTM); "
                             "accuracy unknown - check the data provider's metadata."},
    },
}
DEM_INFO.update(DS.LAYERED_DEM_INFO)


def boundary_hash(boundary: dict) -> str:
    s = json.dumps(boundary, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _round_floats(o, nd: int = 6):
    """Normalise coordinates so UI-drawn (6 dp) and CLI (7 dp) boundaries share one cache key (~0.1 m)."""
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, list):
        return [_round_floats(v, nd) for v in o]
    if isinstance(o, dict):
        return {k: _round_floats(v, nd) for k, v in o.items()}
    return o


def cache_key(boundary: dict, dem_source: str) -> str:
    key = {"boundary": _round_floats(boundary), "dem_source": dem_source, "v": CACHE_VERSION}
    if dem_source.startswith("file:"):
        p = Path(dem_source[5:])
        if p.exists():
            st = p.stat()
            key["file"] = [str(p.resolve()), st.st_size, int(st.st_mtime)]
    return boundary_hash(key)


# 2 = + barrier=hedge, natural=shrubbery|scrub|shrub (M4). 3 = + natural=peak|volcano|cliff|sand,
# wetland=tidalflat|saltmarsh|mangrove, bridge/tunnel chains completed beyond the bbox (M4-A). Older caches get one
# OSM re-query (DEM/nDOM kept; GLO-30 abutment heights beyond the DEM go to dem_ext.tif).
# 4 = + green areas: leisure=park|garden, landuse=grass|meadow|plant_nursery|orchard|cemetery|village_green|
# recreation_ground, natural=grassland|heath (tree scatter + grass land cover).
# 5 = + more features (M6): railway lines/platforms, barrier=wall|retaining_wall|city_wall|fence|guard_rail,
# building:part, amenity=parking, leisure=pitch|playground, area:highway, place=square, man_made=pier|bridge, power.
OSM_QUERY_REV = 5


def build_query(bb: tuple) -> str:
    s, w, n, e = bb
    b = f"({s:.7f},{w:.7f},{n:.7f},{e:.7f})"
    return f"""[out:json][timeout:90];
(
  nwr["building"]{b};
  way["highway"]{b};
  node["natural"="tree"]{b};
  way["natural"="tree_row"]{b};
  nwr["natural"="wood"]{b};
  nwr["landuse"="forest"]{b};
  nwr["waterway"]{b};
  nwr["natural"="water"]{b};
  nwr["natural"="coastline"]{b};
  nwr["natural"="beach"]{b};
  nwr["natural"="wetland"]{b};
  way["barrier"="hedge"]{b};
  nwr["natural"="shrubbery"]{b};
  nwr["natural"="scrub"]{b};
  node["natural"="shrub"]{b};
  node["natural"~"^(peak|volcano)$"]{b};
  nwr["natural"="cliff"]{b};
  nwr["natural"="sand"]{b};
  nwr["wetland"~"^(tidalflat|saltmarsh|mangrove)$"]{b};
  nwr["leisure"~"^(park|garden)$"]{b};
  nwr["landuse"~"^(grass|meadow|plant_nursery|orchard|cemetery|village_green|recreation_ground)$"]{b};
  nwr["natural"~"^(grassland|heath)$"]{b};
  way["railway"~"^(rail|light_rail|tram|subway|narrow_gauge)$"]{b};
  nwr["railway"="platform"]{b};
  nwr["public_transport"="platform"]{b};
  way["barrier"~"^(wall|retaining_wall|city_wall|fence|guard_rail)$"]{b};
  nwr["building:part"]{b};
  nwr["amenity"="parking"]{b};
  nwr["leisure"~"^(pitch|playground)$"]{b};
  nwr["area:highway"]{b};
  nwr["place"="square"]{b};
  nwr["man_made"~"^(pier|bridge)$"]{b};
  way["power"~"^(line|minor_line)$"]{b};
  node["power"~"^(tower|pole)$"]{b};
);
out geom;
"""


def overpass(query: str) -> tuple[dict, str]:
    """Race all Overpass mirrors (osmctx.overpass, shared with the server's count path); the first valid answer
    whose data is <= 7 days old wins, a stale one only when no fresh mirror answers (see osm_freshness_warning)."""
    data, url, _info = OV.query(query, OVERPASS_URLS, timeout=150.0, rounds=2)
    return data, url


def osm_freshness_warning(data: dict, url: str | None) -> str | None:
    """Warning text when the answer's osm3s.timestamp_osm_base is older than OV.FRESH_DAYS."""
    return OV.stale_warning(OV.freshness(data), url)


# --------------------------------------------------------------------------- bridge / tunnel chains

CHAIN_MAX_ROUNDS = 6         # Overpass rounds following bridge/tunnel ways beyond the bbox
CHAIN_MAX_WAYS = 400         # safety cap on added ways
CHAIN_EXT_MAX_M = 2000.0     # dem_ext.tif covers chain ends at most this far beyond the DEM bbox
CHAIN_EXT_MARGIN_M = 60.0


def _chain_kind(t: dict) -> str | None:
    if "highway" not in t:
        return None
    if t.get("bridge", "no") not in ("no", "low_water_crossing"):
        return "bridge"
    if t.get("tunnel", "no") != "no":
        return "tunnel"
    return None


def dangling_chain_ends(elements: list) -> list[int]:
    """End node ids of highway bridge/tunnel ways that touch no other fetched highway way (= the chain was cut by
    the fetch bbox, the abutment / portal is still missing)."""
    use: Counter = Counter()
    for el in elements:
        if el.get("type") == "way" and "highway" in el.get("tags", {}):
            for n in set(el.get("nodes") or []):
                use[n] += 1
    out = []
    for el in elements:
        if el.get("type") != "way" or not _chain_kind(el.get("tags", {})):
            continue
        nodes = el.get("nodes") or []
        for n in (nodes[:1] + nodes[-1:]):
            if use[n] < 2 and n not in out:
                out.append(n)
    return out


def chain_query(node_ids: list[int]) -> str:
    ids = ",".join(str(int(n)) for n in node_ids)
    return f"""[out:json][timeout:90];
node(id:{ids});
way(bn)["highway"];
out geom;
"""


def complete_chains(data: dict, query_fn=None, max_rounds: int = CHAIN_MAX_ROUNDS) -> dict:
    """Follow highway bridge/tunnel ways cut by the bbox until every chain end meets a non-bridge road (abutment /
    portal), adding the connected ways (out geom) to data['elements'] in place. query_fn(q) -> (data, url) defaults
    to overpass(). Returns a record {rounds, added_ways, unresolved_ends, errors} for meta.json `bridge_chains`."""
    query_fn = query_fn or overpass
    els = data.setdefault("elements", [])
    have = {(e.get("type"), e.get("id")) for e in els}
    tried: set = set()
    rec = {"rounds": 0, "added_ways": [], "unresolved_ends": [], "errors": []}
    for _ in range(max_rounds):
        ends = [n for n in dangling_chain_ends(els) if n not in tried]
        if not ends or len(rec["added_ways"]) >= CHAIN_MAX_WAYS:
            break
        tried.update(ends)
        try:
            more, _url = query_fn(chain_query(ends))
        except Exception as ex:  # noqa: BLE001
            rec["errors"].append(f"{type(ex).__name__}: {str(ex)[:160]}")
            break
        rec["rounds"] += 1
        new = [e for e in more.get("elements", []) if (e.get("type"), e.get("id")) not in have]
        for e in new:
            have.add((e.get("type"), e.get("id")))
            els.append(e)
            rec["added_ways"].append(e.get("id"))
        if not new:
            break
    rec["unresolved_ends"] = dangling_chain_ends(els)
    rec["added_ways"] = rec["added_ways"][:CHAIN_MAX_WAYS]
    return rec


def chain_end_points(elements: list) -> list[tuple[float, float]]:
    """(lon, lat) of every end node of highway bridge/tunnel ways (abutments, portals, cut ends)."""
    pts = []
    for el in elements:
        if el.get("type") == "way" and _chain_kind(el.get("tags", {})):
            g = [p for p in (el.get("geometry") or []) if p]
            if g:
                pts += [(g[0]["lon"], g[0]["lat"]), (g[-1]["lon"], g[-1]["lat"])]
    return pts


def write_dem_ext(d: Path, elements: list, dem_bbox: dict) -> dict | None:
    """GLO-30 `dem_ext.tif` covering bridge/tunnel chain ends that lie outside the main DEM bbox (<= CHAIN_EXT_MAX_M
    beyond it), so abutment/portal heights are sampled, not clamped to the DEM edge. Never raises."""
    pts = chain_end_points(elements)
    s, w, n, e = dem_bbox["south"], dem_bbox["west"], dem_bbox["north"], dem_bbox["east"]
    out = [(lo, la) for lo, la in pts if not (w <= lo <= e and s <= la <= n)]
    if not out:
        return None
    lat0 = (s + n) / 2
    mlat = CHAIN_EXT_MAX_M / 111320.0
    mlon = CHAIN_EXT_MAX_M / (111320.0 * max(math.cos(math.radians(lat0)), 1e-6))
    keep = [(lo, la) for lo, la in out if w - mlon <= lo <= e + mlon and s - mlat <= la <= n + mlat]
    rec = {"source": "copernicus", "points_outside_dem": len(out), "points_covered": len(keep)}
    if not keep:
        rec["note"] = f"all {len(out)} chain ends are > {CHAIN_EXT_MAX_M:g} m beyond the DEM bbox"
        return rec
    glat = CHAIN_EXT_MARGIN_M / 111320.0
    glon = CHAIN_EXT_MARGIN_M / (111320.0 * max(math.cos(math.radians(lat0)), 1e-6))
    bb = (min([s] + [la for _, la in keep]) - glat, min([w] + [lo for lo, _ in keep]) - glon,
          max([n] + [la for _, la in keep]) + glat, max([e] + [lo for lo, _ in keep]) + glon)
    try:
        info = dem_from_sources(copernicus_tile_urls(bb), bb, Path(d) / "dem_ext.tif", missing_fill=0.0)
        rec.update({"bbox": {"south": bb[0], "west": bb[1], "north": bb[2], "east": bb[3]},
                    "shape": info["shape"], "warnings": info["warnings"],
                    "accuracy": "Copernicus GLO-30 DSM (~30 m); used only for bridge abutment / tunnel portal heights"})
    except Exception as ex:  # noqa: BLE001
        rec["error"] = f"{type(ex).__name__}: {ex}"
    return rec


# --------------------------------------------------------------------------- source-grid mosaic

def snap_bounds(bounds: tuple, transform) -> tuple:
    """Snap (w, s, e, n) OUTWARD to whole pixels of a north-up source grid. Returns (W, S, E, N)."""
    w, s, e, n = bounds
    dx, dy = transform.a, -transform.e
    x0, y0 = transform.c, transform.f
    eps = 1e-6
    cw = math.floor((w - x0) / dx + eps)
    ce = math.ceil((e - x0) / dx - eps)
    rn = math.floor((y0 - n) / dy + eps)
    rs = math.ceil((y0 - s) / dy - eps)
    ce = max(ce, cw + 2)
    rs = max(rs, rn + 2)
    return x0 + cw * dx, y0 - rs * dy, x0 + ce * dx, y0 - rn * dy


def _on_lattice(t, ref) -> bool:
    if not (math.isclose(t.a, ref.a, rel_tol=1e-9) and math.isclose(t.e, ref.e, rel_tol=1e-9)):
        return False
    fx = (t.c - ref.c) / ref.a
    fy = (t.f - ref.f) / ref.e
    return abs(fx - round(fx)) < 1e-4 and abs(fy - round(fy)) < 1e-4


def mosaic_on_source_grid(datasets: list, bounds: tuple) -> tuple[np.ndarray, object, list]:
    """Mosaic north-up rasters onto the grid of the finest dataset, covering `bounds` (in the datasets'
    CRS) snapped outward to whole source pixels. Pixels are copied, never resampled, when all sources
    share one lattice. Returns (float32 array with NaN for no data, Affine transform, warnings)."""
    import rasterio.enums
    from affine import Affine
    from rasterio.merge import merge
    from rasterio.windows import Window

    warnings: list[str] = []
    ref = min(datasets, key=lambda d: d.transform.a * -d.transform.e)
    for d in datasets:
        if d.transform.b != 0 or d.transform.d != 0 or d.transform.e >= 0:
            raise ValueError("only north-up, unrotated DEM rasters are supported")
    rt = ref.transform
    W, S, E, N = snap_bounds(bounds, rt)
    dx, dy = rt.a, -rt.e
    width, height = int(round((E - W) / dx)), int(round((N - S) / dy))
    transform = Affine(dx, 0.0, W, 0.0, -dy, N)
    if all(_on_lattice(d.transform, rt) for d in datasets):
        out = np.full((height, width), np.nan, dtype="float32")
        for d in datasets:
            t = d.transform
            col_off = int(round((W - t.c) / dx))
            row_off = int(round((t.f - N) / dy))
            c0, c1 = max(col_off, 0), min(col_off + width, d.width)
            r0, r1 = max(row_off, 0), min(row_off + height, d.height)
            if c0 >= c1 or r0 >= r1:
                continue
            blk = d.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype("float32")
            if d.nodata is not None:
                blk = np.where(blk == d.nodata, np.nan, blk)
            dst = out[r0 - row_off:r1 - row_off, c0 - col_off:c1 - col_off]
            np.copyto(dst, blk, where=np.isnan(dst))
        return out, transform, warnings
    # mixed grids (e.g. GLO-30 tiles either side of 50/60/70 deg lat): nearest onto the finest grid
    warnings.append("DEM tiles have different pixel grids; coarser tiles were resampled (nearest) "
                    "onto the finest grid")
    data, t = merge(datasets, bounds=(W, S, E, N), res=(dx, dy), nodata=NODATA,
                    resampling=rasterio.enums.Resampling.nearest)
    a = data[0].astype("float32")
    return np.where(a == NODATA, np.nan, a), t, warnings


def write_dem(path: Path, arr: np.ndarray, transform, crs, nodata_fill: float | None = None) -> None:
    """float32 GeoTIFF. NaN -> nodata_fill (no nodata tag) if given, else NODATA tagged as nodata."""
    DS.write_float32_tif(path, arr, transform, crs, nodata_fill=nodata_fill)


def _gdal_env():
    return DS.gdal_env(".tif")


def _json_dump(path: Path, obj, **kw) -> None:
    """Atomic JSON write (a half-written osm.json/meta.json would make the cache look complete but broken)."""
    with DS.atomic_path(path) as tmp:
        tmp.write_text(json.dumps(obj, **kw), encoding="utf-8")


# --------------------------------------------------------------------------- Copernicus GLO-30

def _tile_url(lat_floor: int, lon_floor: int) -> str:
    return DEM_URL.format(ns="N" if lat_floor >= 0 else "S", lat=abs(lat_floor),
                          ew="E" if lon_floor >= 0 else "W", lon=abs(lon_floor))


def copernicus_tile_urls(bb: tuple) -> list[str]:
    s, w, n, e = bb
    return [_tile_url(la, lo) for la in range(math.floor(s), math.floor(n - 1e-12) + 1)
            for lo in range(math.floor(w), math.floor(e - 1e-12) + 1)]


def dem_from_sources(sources: list, bb: tuple, out_path: Path, missing_fill: float | None = 0.0) -> dict:
    """Open GeoTIFF sources (paths or URLs, EPSG:4326 grids such as GLO-30 tiles), mosaic the WGS84 bbox
    (s, w, n, e) on the source grid and write out_path. Missing sources/cells -> missing_fill (0.0 = sea
    level for GLO-30 ocean tiles). Returns {"warnings", "crs", "shape", "transform"}."""
    import rasterio
    s, w, n, e = bb
    warnings: list[str] = []
    datasets = []
    with _gdal_env():
        try:
            for src in sources:
                src = str(src)
                try:
                    datasets.append(rasterio.open("/vsicurl/" + src if src.startswith("http") else src))
                except Exception as ex:  # 404 (ocean) / network
                    warnings.append(f"DEM tile missing ({src.rsplit('/', 1)[-1]}): filled {missing_fill} "
                                    f"({type(ex).__name__})")
            if datasets:
                arr, transform, w2 = mosaic_on_source_grid(datasets, (w, s, e, n))
                warnings += w2
                crs = datasets[0].crs
            else:
                from affine import Affine
                res = 1.0 / 3600.0
                W, S, E, N = snap_bounds((w, s, e, n), Affine(res, 0, -180 - res / 2, 0, -res, 90 + res / 2))
                arr = np.full((int(round((N - S) / res)), int(round((E - W) / res))), np.nan, "float32")
                transform = Affine(res, 0, W, 0, -res, N)
                crs = "EPSG:4326"
                warnings.append(f"No DEM tiles available: whole DEM is {missing_fill}")
        finally:
            for d in datasets:
                d.close()
    if np.isnan(arr).any() and datasets:
        warnings.append(f"DEM nodata/missing cells filled with {missing_fill}")
    write_dem(out_path, arr, transform, crs, nodata_fill=missing_fill)
    return {"warnings": warnings, "crs": str(crs), "shape": list(arr.shape), "transform": list(transform)[:6]}


# --------------------------------------------------------------------------- BW DGM1 (LGL open data)

class _HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests (enough for zipfile's central directory).
    readable/seekable/tell/seek are the file protocol zipfile.ZipFile needs (not dead code)."""

    def __init__(self, url: str, session: requests.Session, block: int = 65536):
        self.url, self.ses, self.block = url, session, block
        r = session.head(url, timeout=60, allow_redirects=True)
        if r.status_code == 404:
            raise FileNotFoundError(url)
        r.raise_for_status()
        self.size = int(r.headers["content-length"])
        self.pos = 0
        self._buf_start, self._buf = -1, b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def get_range(self, start: int, length: int, timeout: float = 600) -> bytes:
        with self.ses.get(self.url, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=timeout,
                          stream=True) as r:
            r.raise_for_status()
            if r.status_code != 206:
                raise RuntimeError(f"server ignored Range request ({r.status_code})")
            data = b"".join(r.iter_content(1 << 20))
        if len(data) != length:
            raise RuntimeError(f"short read {len(data)}/{length} from {self.url}")
        return data

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        bs = self._buf_start
        if not (bs <= self.pos and self.pos + n <= bs + len(self._buf)):
            ln = min(max(n, self.block), self.size - self.pos)
            self._buf, self._buf_start = self.get_range(self.pos, ln), self.pos
        o = self.pos - self._buf_start
        out = self._buf[o:o + n]
        self.pos += len(out)
        return out


def bw_zip_tile(e_km: int, n_km: int) -> tuple[int, int]:
    """LGL DGM1 download zips are 2x2 km with the SW corner at odd easting km / even northing km."""
    return e_km - ((e_km - 1) % 2), n_km - (n_km % 2)


def bw_subtiles(bounds_25832: tuple) -> list[tuple[int, int]]:
    """1 km sub-tiles (SW corner km) touching (w, s, e, n) in EPSG:25832."""
    w, s, e, n = bounds_25832
    return [(ek, nk) for nk in range(math.floor(s / 1000), math.floor((n - 1e-6) / 1000) + 1)
            for ek in range(math.floor(w / 1000), math.floor((e - 1e-6) / 1000) + 1)]


def parse_dgm1_xyz(data: bytes, e_km: int, n_km: int) -> np.ndarray:
    """LGL XYZ ('E N H' per 1 m cell centre, e.g. '504000.50 5374999.50 377.98') -> 1000x1000 float32,
    row 0 = north, NaN where a cell is missing."""
    v = np.array(data.split(), dtype="float64")
    if v.size % 3:
        raise ValueError("XYZ token count not a multiple of 3")
    v = v.reshape(-1, 3)
    col = np.floor(v[:, 0] - e_km * 1000.0).astype(int)
    row = np.floor((n_km + 1) * 1000.0 - v[:, 1]).astype(int)
    ok = (col >= 0) & (col < 1000) & (row >= 0) & (row < 1000)
    a = np.full((1000, 1000), np.nan, dtype="float32")
    a[row[ok], col[ok]] = v[ok, 2]
    return a


def _zip_member_bytes(rf, zi) -> bytes:
    """Decompressed bytes of one zip member via a single Range request."""
    hdr = rf.get_range(zi.header_offset, 30)
    if hdr[:4] != b"PK":
        raise RuntimeError("bad zip local header")
    nlen = int.from_bytes(hdr[26:28], "little")
    xlen = int.from_bytes(hdr[28:30], "little")
    raw = rf.get_range(zi.header_offset + 30 + nlen + xlen, zi.compress_size)
    if zi.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompressobj(-15).decompress(raw)
    elif zi.compress_type == zipfile.ZIP_STORED:
        data = raw
    else:
        raise RuntimeError(f"unsupported zip compression {zi.compress_type} in {url}")
    if zlib.crc32(data) & 0xFFFFFFFF != zi.CRC:
        raise RuntimeError(f"CRC mismatch for {zi.filename} in {url}")
    return data


def _bw_fetch_subtile(e_km: int, n_km: int, tile_dir: Path, session: requests.Session) -> tuple[Path | None, dict]:
    """Cached 1 km GeoTIFF (EPSG:25832, 1 m) of one DGM1 sub-tile; (None, info) if LGL has none."""
    from rasterio.transform import from_origin
    tif = tile_dir / f"dgm1_32_{e_km}_{n_km}.tif"
    info_p = tile_dir / f"dgm1_32_{e_km}_{n_km}.json"
    if info_p.exists():
        info = json.loads(info_p.read_text(encoding="utf-8"))
        if info.get("missing"):  # cached "not available"
            return None, info
        if tif.exists():
            return tif, info
    tile_dir.mkdir(parents=True, exist_ok=True)
    ze, zn = bw_zip_tile(e_km, n_km)
    url = BW_DGM1_ZIP.format(e=ze, n=zn)
    try:
        rf = _HttpRangeFile(url, session)
    except FileNotFoundError:
        info = {"missing": True, "url": url}
        _json_dump(info_p, info)
        return None, info
    z = zipfile.ZipFile(rf)
    prefix = f"dgm1_32_{e_km}_{n_km}_1_bw"
    members = {Path(i.filename).name: i for i in z.infolist()}
    xyz = [i for nm, i in members.items() if nm.startswith(prefix) and nm.endswith(".xyz")]
    if not xyz:
        info = {"missing": True, "url": url, "reason": f"no {prefix}*.xyz in zip"}
        _json_dump(info_p, info)
        return None, info
    zi = xyz[0]
    csv = [i for nm, i in members.items() if nm.startswith(prefix) and nm.endswith(".csv")]
    meta = {}
    if csv:
        lines = z.read(csv[0]).decode("latin-1").strip().splitlines()
        if len(lines) >= 2:
            meta = dict(zip(lines[0].split(";"), lines[1].split(";")))
    data = _zip_member_bytes(rf, zi)
    a = parse_dgm1_xyz(data, e_km, n_km)
    DS.write_float32_tif(tif, a, from_origin(e_km * 1000.0, (n_km + 1) * 1000.0, 1.0, 1.0), "EPSG:25832",
                         predictor=3, tag_area=False)
    info = {"missing": False, "url": url, "member": zi.filename, "n_nodata": int(np.isnan(a).sum()),
            "Aktualitaet": meta.get("Aktualitaet"), "Genauigkeit": meta.get("Genauigkeit"),
            "crs_h": meta.get("Koordinatenreferenzsystem_Hoehe")}
    _json_dump(info_p, info)
    return tif, info


def bbox_25832(bb: tuple) -> tuple:
    from rasterio.warp import transform_bounds
    s, w, n, e = bb
    return transform_bounds("EPSG:4326", "EPSG:25832", w, s, e, n, densify_pts=21)


def in_bw_envelope(bb: tuple) -> bool:
    w, s, e, n = bbox_25832(bb)
    W, S, E, N = BW_ENVELOPE_25832
    return w >= W and e <= E and s >= S and n <= N


def _mosaic_files(paths: list, bounds: tuple):
    """mosaic_on_source_grid over GeoTIFF paths; every dataset is closed even if one fails to open."""
    import rasterio
    with ExitStack() as st:
        return mosaic_on_source_grid([st.enter_context(rasterio.open(p)) for p in paths], bounds)


def fetch_bw_dgm1(bb: tuple, out_path: Path, tile_dir: Path) -> dict:
    """LGL DGM1 for the WGS84 bbox (s, w, n, e), written in EPSG:25832 on the 1 m source grid.
    Raises RuntimeError when no data; returns info incl. 'complete' (no nodata in the bbox)."""
    b = bbox_25832(bb)
    subs = bw_subtiles(b)
    tifs, infos, warnings = [], [], []
    with requests.Session() as ses:
        ses.headers["User-Agent"] = UA
        results = list(DS.parallel_fetch(lambda ek_nk: _bw_fetch_subtile(ek_nk[0], ek_nk[1], tile_dir, ses),
                                         subs, workers=BW_TILE_WORKERS))
    for (ek, nk), (tif, info) in zip(subs, results):
        infos.append({"tile": f"{ek}_{nk}", **info})
        if tif is None:
            warnings.append(f"DGM1 sub-tile {ek}_{nk} not available ({info.get('url')})")
        else:
            tifs.append(tif)
    if not tifs:
        raise RuntimeError("no DGM1 tiles available for this bbox")
    arr, transform, w2 = _mosaic_files(tifs, b)
    warnings += w2
    n_nan = int(np.isnan(arr).sum())
    if n_nan:
        warnings.append(f"DGM1 has {n_nan} nodata cells in the bbox")
    write_dem(out_path, arr, transform, "EPSG:25832")
    dates = sorted({i.get("Aktualitaet") for i in infos if i.get("Aktualitaet")})
    return {"warnings": warnings, "complete": n_nan == 0, "tiles": infos, "dates": dates,
            "shape": list(arr.shape), "transform": list(transform)[:6]}


# --------------------------------------------------------------------------- nDOM1 (tree heights)

def _bw_fetch_ndom_subtile(e_km: int, n_km: int, tile_dir: Path, session) -> Path | None:
    """Cached 1 km nDOM1 GeoTIFF (height above ground, 1 m, EPSG:25832) or None when LGL has none."""
    tif = tile_dir / f"ndom1_32_{e_km}_{n_km}.tif"
    miss = tile_dir / f"ndom1_32_{e_km}_{n_km}.missing"
    if tif.exists():
        return tif
    if miss.exists():
        return None
    tile_dir.mkdir(parents=True, exist_ok=True)
    ze, zn = bw_zip_tile(e_km, n_km)
    url = BW_NDOM1_ZIP.format(e=ze, n=zn)
    try:
        rf = _HttpRangeFile(url, session)
    except FileNotFoundError:
        miss.write_text(url, encoding="utf-8")
        return None
    z = zipfile.ZipFile(rf)
    name = f"ndom1_32_{e_km}_{n_km}_1_bw.tif"
    zi = next((i for i in z.infolist() if Path(i.filename).name == name), None)
    if zi is None:
        miss.write_text(f"{url}: no {name}", encoding="utf-8")
        return None
    with DS.atomic_path(tif) as tmp:
        tmp.write_bytes(_zip_member_bytes(rf, zi))
    return tif


def fetch_bw_ndom1(bb: tuple, out_path: Path, tile_dir: Path) -> dict:
    """LGL nDOM1 (normalised surface model = bDOM - DGM1 = height of objects above ground) for the WGS84
    bbox, written as `out_path` (EPSG:25832, 1 m source grid, NaN->nodata). Raises when no tile exists."""
    b = bbox_25832(bb)
    with requests.Session() as ses:
        ses.headers["User-Agent"] = UA
        found = list(DS.parallel_fetch(lambda ek_nk: _bw_fetch_ndom_subtile(ek_nk[0], ek_nk[1], tile_dir, ses),
                                       bw_subtiles(b), workers=BW_TILE_WORKERS))
    tifs = [t for t in found if t is not None]
    if not tifs:
        raise RuntimeError("no nDOM1 tiles available for this bbox")
    arr, transform, warnings = _mosaic_files(tifs, b)
    write_dem(out_path, arr, transform, "EPSG:25832")
    return {"warnings": warnings, "tiles": len(tifs), "shape": list(arr.shape)}


def _ndom_record(dem_bbox: dict, cache_d: Path, tile_root: Path, attribution: str | None) -> dict:
    """Fetch nDOM1 into cache_d/ndom.tif and return the meta.json `ndom` record. Never raises."""
    try:
        info = fetch_bw_ndom1((dem_bbox["south"], dem_bbox["west"], dem_bbox["north"], dem_bbox["east"]),
                              Path(cache_d) / "ndom.tif", tile_root)
        return {"available": True, "source": "LGL-BW nDOM1 (opengeodata.lgl-bw.de/data/ndom1, 1 m, = bDOM - DGM1)",
                "attribution": attribution, "tiles": info["tiles"], "warnings": info["warnings"]}
    except Exception as ex:  # noqa: BLE001
        return {"available": False, "reason": f"{type(ex).__name__}: {ex}"}


def ensure_ndom(cache_d: Path, tile_root: Path | None = None) -> dict:
    """Idempotent: for a DGM1-based cache add `ndom.tif` (+ meta.json `ndom` record). Never raises; the
    converter simply skips the measured-height step when the file is absent. Returns the meta record."""
    cache_d = Path(cache_d)
    mp = cache_d / "meta.json"
    meta = json.loads(mp.read_text(encoding="utf-8"))
    if meta.get("dem_source_id") != "bw_dgm1":
        return {"available": False, "reason": "nDOM1 only exists for the Baden-Wuerttemberg DGM1 source"}
    if (cache_d / "ndom.tif").exists() and meta.get("ndom", {}).get("available"):
        return meta["ndom"]
    tile_root = Path(tile_root) if tile_root else cache_d.parent / "_tiles" / "bw_ndom1"
    rec = _ndom_record(meta["dem_bbox"], cache_d, tile_root, meta.get("dem_attribution"))
    meta["ndom"] = rec
    _json_dump(mp, meta, indent=2)
    return rec


# --------------------------------------------------------------------------- user file

def fetch_file_dem(path: str, bb: tuple, out_path: Path) -> dict:
    """User GeoTIFF: clip the WGS84 bbox (transformed to the file CRS) on the file's own grid."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DEM file not found: {p}")
    s, w, n, e = bb
    with rasterio.open(p) as d:
        crs = d.crs
        pre: list[str] = []
        if crs is None:
            if p.name.lower().startswith("dgm1_32"):  # LGL BW XYZ/GeoTIFF without CRS
                crs = CRS.from_epsg(25832)
                pre.append(f"{p.name} has no CRS; assumed EPSG:25832 (LGL DGM1 naming)")
            else:
                raise ValueError(f"{p} has no CRS (GeoTIFF with CRS needed)")
        b = transform_bounds("EPSG:4326", crs, w, s, e, n, densify_pts=21)
        fb = d.bounds
        if b[2] <= fb.left or b[0] >= fb.right or b[3] <= fb.bottom or b[1] >= fb.top:
            raise ValueError(f"{p} does not overlap the site bbox")
        arr, transform, warnings = mosaic_on_source_grid([d], b)
        warnings = pre + warnings
    n_nan = int(np.isnan(arr).sum())
    if n_nan:
        warnings.append(f"user DEM does not fully cover the bbox: {n_nan} nodata cells")
    write_dem(out_path, arr, transform, crs)
    px = abs(transform.a) * (1.0 if crs.is_projected else 111320.0 * math.cos(math.radians((s + n) / 2)))
    return {"warnings": warnings, "crs": crs.to_string(), "pixel_m": round(px, 3), "shape": list(arr.shape)}


# --------------------------------------------------------------------------- main fetch

def _dem_meta(src_id: str, year: int, extra_note: str | None = None, **acc) -> dict:
    inf = DEM_INFO[src_id]
    a = dict(inf["accuracy"])
    a.update({k: v for k, v in acc.items() if v is not None})
    if extra_note:
        a["note"] = a["note"] + " " + extra_note
    return {"dem_source": inf["name"], "dem_source_id": src_id,
            "dem_attribution": inf["attribution"].format(year=year), "dem_licence": inf["licence"],
            "dem_vertical_datum": inf["vertical_datum"], "dem_accuracy": a}


def post_process_osm(d: Path, data: dict, url: str | None, meta: dict) -> None:
    """After an Overpass answer: data-age check, bridge/tunnel chain completion beyond the bbox, dem_ext.tif for
    chain ends outside the DEM. Updates `data` and `meta` in place (osm_base, osm_age_days, bridge_chains, dem_ext,
    warnings). Never raises."""
    fr = OV.freshness(data)
    meta["osm_base"], meta["osm_age_days"] = fr["osm_base"], fr["age_days"]
    warns = meta.setdefault("warnings", [])
    warns[:] = [w for w in warns if not w.startswith(("OSM data is ", "bridge/tunnel chain"))]
    sw = osm_freshness_warning(data, url)
    if sw:
        warns.append(sw)
    try:
        rec = complete_chains(data)
    except Exception as ex:  # noqa: BLE001
        rec = {"rounds": 0, "added_ways": [], "unresolved_ends": [], "errors": [f"{type(ex).__name__}: {ex}"]}
    meta["bridge_chains"] = {"rounds": rec["rounds"], "added_ways": len(rec["added_ways"]),
                             "unresolved_ends": len(rec["unresolved_ends"]), "errors": rec["errors"]}
    if rec["errors"]:
        warns.append("bridge/tunnel chain completion failed (" + "; ".join(rec["errors"])[:200]
                     + "): bridge ends cut by the bbox keep estimated heights")
    db = meta.get("dem_bbox")
    if db:
        ext = write_dem_ext(Path(d), data.get("elements", []), db)
        if ext is not None:
            meta["dem_ext"] = ext
        else:
            meta.pop("dem_ext", None)
            p = Path(d) / "dem_ext.tif"
            if p.exists():
                p.unlink()


def refresh_osm(d: Path) -> bool:
    """Re-run the Overpass query for a cache written by an older query revision (osm.json only; DEM/nDOM kept).
    Never raises: on failure the old osm.json stays and meta.json records `osm_refresh_error`."""
    d = Path(d)
    mp = d / "meta.json"
    meta = json.loads(mp.read_text(encoding="utf-8"))
    if meta.get("osm_query_rev", 1) >= OSM_QUERY_REV:
        return False
    ob = meta.get("osm_bbox") or meta["bbox"]
    try:
        data, url = overpass(build_query((ob["south"], ob["west"], ob["north"], ob["east"])))
        post_process_osm(d, data, url, meta)
        _json_dump(d / "osm.json", data)
        meta.update(osm_query_rev=OSM_QUERY_REV, overpass_url=url, osm_refreshed_at=datetime.now(timezone.utc).isoformat())
        meta.pop("osm_refresh_error", None)
        ok = True
    except Exception as ex:  # noqa: BLE001
        meta["osm_refresh_error"] = f"{type(ex).__name__}: {ex}"
        ok = False
    _json_dump(mp, meta, indent=2)
    return ok


def _predict_margin(dem_source: str, poly) -> float:
    """Margin of the DEM/OSM context bbox the DEM branches in _fetch_fresh end up with, so Overpass + Overture can
    start before the DEM download. Exact for every source except "auto" inside Baden-Wuerttemberg, where DGM1 can
    still fall back to Copernicus (wider margin); _fetch_fresh re-queries then."""
    if dem_source == "bw_dgm1" or dem_source.startswith("file:"):
        return DEM_MARGIN_M
    if dem_source == "auto":
        try:
            return DEM_MARGIN_M if in_bw_envelope(bbox(poly, DEM_MARGIN_M)) else DSM_MARGIN_M
        except Exception:  # noqa: BLE001
            return DSM_MARGIN_M
    if dem_source == "layered" or dem_source in DS.LAYERED_DEM_INFO:
        return DSM_MARGIN_M if dem_source in ("layered", "gedtm30", "fabdem") else DEM_MARGIN_M
    return DSM_MARGIN_M


def _lod2_fetch(bb, cache_dir: Path, out: Path, allow_large: bool):
    """LoD2 3D buildings (osmctx/lod2.py): None when no LoD2 source covers bb, else the meta block (never raises)."""
    try:
        region = L2.detect_region(bb)
    except Exception:  # noqa: BLE001
        region = None
    if region is None:
        return None
    return L2.try_fetch(bb, cache_dir, out, region=region, allow_large=allow_large)


def _fetch_cached(d: Path, cache_dir: Path, ndom: bool, overture: bool, lod2: bool = False,
                  lod2_allow_large: bool = False) -> Path:
    """Complete cache: re-query OSM if the query revision is old, add nDOM / Overture / LoD2 if missing. The Overture
    and LoD2 downloads touch no meta.json, so they overlap the Overpass refresh and the nDOM download; their records
    are written last."""
    m0 = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        ov_fut = (pool.submit(OVT.try_fetch, OVT.meta_bbox(m0), cache_dir, d / OVT.FILE)
                  if overture and OVT.needs_fetch(d, m0) else None)
        l2_fut = (pool.submit(_lod2_fetch, L2.meta_bbox(m0), cache_dir, d / L2.FILE, lod2_allow_large)
                  if lod2 and L2.needs_fetch(d, m0) else None)
        refresh_osm(d)
        if ndom and m0.get("dem_source_id") == "bw_dgm1" and "ndom" not in m0:
            ensure_ndom(d)
        l2_res = l2_fut.result() if l2_fut is not None else None
        if ov_fut is not None or l2_res is not None:
            mp = d / "meta.json"
            meta = json.loads(mp.read_text(encoding="utf-8"))
            if ov_fut is not None:
                OVT.apply(meta, ov_fut.result())
            if l2_res is not None:
                L2.apply(meta, l2_res)
            _json_dump(mp, meta, indent=2)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return d


def _fetch_fresh(boundary: dict, d: Path, cache_dir: Path, dem_source: str, ndom: bool, overture: bool,
                 lod2: bool = False, lod2_allow_large: bool = False) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    poly = to_polygon(boundary)
    bb_osm = bbox(poly)
    now = datetime.now(timezone.utc)
    warns: list[str] = []
    dem_extra: dict = {}
    used, fallback_reason = None, None
    dem_path = d / "dem.tif"

    def osm_job(bb):
        return overpass(build_query(bb))   # module-level lookup at call time (tests patch fetch.overpass)

    def overture_job(bb):
        return OVT.try_fetch(bb, cache_dir, d / OVT.FILE) if overture else None

    def lod2_job(bb):
        return _lod2_fetch(bb, cache_dir, d / L2.FILE, lod2_allow_large) if lod2 else None

    # OSM, Overture, LoD2 and nDOM do not depend on the DEM pixels, only on the context bbox: run them beside the DEM download
    pool = ThreadPoolExecutor(max_workers=6)
    try:
        bb_pred = bbox(poly, _predict_margin(dem_source, poly))
        osm_fut = pool.submit(osm_job, bb_pred)
        ov_fut = pool.submit(overture_job, bb_pred) if overture else None
        l2_fut = pool.submit(lod2_job, bb_pred) if lod2 else None

        if dem_source in ("auto", "bw_dgm1"):
            bb_dem = bbox(poly, DEM_MARGIN_M)
            try:
                if dem_source == "auto" and not in_bw_envelope(bb_dem):
                    raise RuntimeError("bbox outside Baden-Wuerttemberg")
                info = fetch_bw_dgm1(bb_dem, dem_path, cache_dir / "_tiles" / "bw_dgm1")
                if dem_source == "auto" and not info["complete"]:
                    raise RuntimeError("DGM1 does not fully cover the bbox (" + "; ".join(info["warnings"]) + ")")
                warns += info["warnings"]
                used = "bw_dgm1"
                dem_extra = {"dem_crs": "EPSG:25832", "dem_tiles": info["tiles"], "dem_dates": info["dates"]}
            except Exception as ex:
                if dem_source == "bw_dgm1":
                    raise
                fallback_reason = f"bw_dgm1 not used: {type(ex).__name__}: {ex}"
        if used is None and dem_source.startswith("file:"):
            bb_dem = bbox(poly, DEM_MARGIN_M)
            info = fetch_file_dem(dem_source[5:], bb_dem, dem_path)
            warns += info["warnings"]
            used = "file"
            dem_extra = {"dem_crs": info["crs"], "dem_file": str(Path(dem_source[5:]).resolve())}
        if used is None and (dem_source == "layered" or dem_source in DS.LAYERED_DEM_INFO):
            bb_dem = bbox(poly, DSM_MARGIN_M if dem_source in ("layered", "gedtm30", "fabdem") else DEM_MARGIN_M)

            def _bw_layered(b_arg, p_arg):
                if not in_bw_envelope(b_arg):
                    raise RuntimeError("bbox outside Baden-Wuerttemberg")
                res_bw = fetch_bw_dgm1(b_arg, p_arg, cache_dir / "_tiles" / "bw_dgm1")
                if not res_bw["complete"]:
                    raise RuntimeError("DGM1 does not fully cover the bbox (" + "; ".join(res_bw["warnings"]) + ")")
                return res_bw

            def _cop_layered(b_arg, p_arg):
                return dem_from_sources(copernicus_tile_urls(b_arg), b_arg, p_arg, missing_fill=0.0)

            lres = DS.fetch_layered_dem(
                bb_dem, dem_path, cache_dir, requested=dem_source,
                bw_fn=_bw_layered, copernicus_fn=_cop_layered,
            )
            used = lres["used"]
            warns += lres["warnings"]
            fallback_reason = lres.get("fallback_reason")
            dem_extra = {"dem_crs": lres["dem_crs"]}
            if lres.get("extra"):
                dem_extra["dem_source_detail"] = lres["extra"]
        if used is None:
            bb_dem = bbox(poly, DSM_MARGIN_M)
            info = dem_from_sources(copernicus_tile_urls(bb_dem), bb_dem, dem_path, missing_fill=0.0)
            warns += info["warnings"]
            used = "copernicus"
            dem_extra = {"dem_crs": "EPSG:4326"}

        bb_ctx = bb_dem  # OSM over the DEM extent so building/forest masks cover every DEM cell
        if bb_ctx != bb_pred:   # the "auto" guess was wrong: redo OSM + Overture + LoD2 for the real bbox
            if ov_fut is not None:
                ov_fut.result()   # same output file: let the stale run finish first
            if l2_fut is not None:
                l2_fut.result()
            osm_fut = pool.submit(osm_job, bb_ctx)
            ov_fut = pool.submit(overture_job, bb_ctx) if overture else None
            l2_fut = pool.submit(lod2_job, bb_ctx) if lod2 else None
        acc = {}
        if used == "file":
            acc["horizontal_m"] = info.get("pixel_m")
        dem_meta = _dem_meta(used, now.year, **acc)
        dem_bbox = {"south": bb_dem[0], "west": bb_dem[1], "north": bb_dem[2], "east": bb_dem[3]}
        ndom_fut = (pool.submit(_ndom_record, dem_bbox, d, cache_dir / "_tiles" / "bw_ndom1",
                                dem_meta["dem_attribution"]) if used == "bw_dgm1" and ndom else None)

        data, url = osm_fut.result()
        meta = {
            "boundary": boundary,
            "bbox": {"south": bb_osm[0], "west": bb_osm[1], "north": bb_osm[2], "east": bb_osm[3]},
            "osm_bbox": {"south": bb_ctx[0], "west": bb_ctx[1], "north": bb_ctx[2], "east": bb_ctx[3]},
            "dem_bbox": dem_bbox,
            "fetched_at": now.isoformat(),
            "dem_requested": dem_source,
            **dem_meta,
            **dem_extra,
            "overpass_url": url,
            "cache_version": CACHE_VERSION,
            "osm_query_rev": OSM_QUERY_REV,
            "warnings": warns,
        }
        if fallback_reason:
            meta["dem_fallback_reason"] = fallback_reason
        post_process_osm(d, data, url, meta)
        _json_dump(d / "osm.json", data)
        _json_dump(d / "meta.json", meta, indent=2)   # usable cache now; nDOM / Overture / LoD2 records follow
        l2_res = l2_fut.result() if l2_fut is not None else None
        if ndom_fut is not None or ov_fut is not None or l2_res is not None:
            if ndom_fut is not None:
                meta["ndom"] = ndom_fut.result()
            if ov_fut is not None:
                OVT.apply(meta, ov_fut.result())
            if l2_res is not None:
                L2.apply(meta, l2_res)
            _json_dump(d / "meta.json", meta, indent=2)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return d


def fetch(boundary: dict, cache_dir="cache", dem_source: str = "auto", ndom: bool = True,
          overture: bool = True, lod2: bool = True, lod2_allow_large: bool = False) -> Path:
    """Fetch OSM + DEM for `boundary` into cache/<key>/ (key = boundary + dem_source + cache version).
    ndom=False skips the nDOM1 (tree-height) download (super-light preset); an nDOM already in the cache stays.
    overture=True also fetches Overture Maps buildings (osmctx/overture.py, time-boxed, never raises; failure ->
    meta.json warning + overture.status 'failed'). OSM, Overture and nDOM run concurrently with the DEM download.
    lod2=True also fetches LoD2 3D buildings (osmctx/lod2.py) where a source covers the site (Baden-Wuerttemberg, NL, CH,
    JP; same contract: never raises, meta.json ``lod2`` block + warning on failure, never blocks OSM). Japan /
    Switzerland (~100-170 MB) only with lod2_allow_large=True. lod2.json holds absolute UTM of the bbox centre
    (origin 0,0); convert re-frames it."""
    if dem_source not in ("auto", *DEM_INFO) and not dem_source.startswith("file:"):
        raise ValueError(f"unknown dem_source {dem_source!r}")
    cache_dir = Path(cache_dir)
    d = cache_dir / cache_key(boundary, dem_source)
    if all((d / f).exists() for f in ("osm.json", "dem.tif", "meta.json")):
        return _fetch_cached(d, cache_dir, ndom, overture, lod2, lod2_allow_large)
    return _fetch_fresh(boundary, d, cache_dir, dem_source, ndom, overture, lod2, lod2_allow_large)


def count_categories(osm: dict) -> Counter:
    c: Counter = Counter()
    for el in osm.get("elements", []):
        t = el.get("tags", {})
        if "building" in t:
            c["building"] += 1
        if "highway" in t:
            c["highway"] += 1
        if t.get("natural") == "tree":
            c["tree"] += 1
        if t.get("natural") == "tree_row":
            c["tree_row"] += 1
        if t.get("natural") == "wood" or t.get("landuse") == "forest":
            c["wood/forest"] += 1
        if t.get("barrier") == "hedge":
            c["hedge"] += 1
        if t.get("natural") == "shrub":
            c["shrub"] += 1
        if t.get("natural") in ("shrubbery", "scrub"):
            c["shrubbery/scrub"] += 1
        if "waterway" in t:
            c["waterway"] += 1
        if t.get("natural") in ("water", "coastline", "beach", "wetland"):
            c["water/coast/beach/wetland"] += 1
        if t.get("leisure") in ("park", "garden"):
            c["park/garden"] += 1
        if t.get("landuse") in ("plant_nursery", "orchard", "cemetery"):
            c[t["landuse"]] += 1
        if t.get("landuse") in ("grass", "meadow", "village_green", "recreation_ground") or t.get("natural") in ("grassland", "heath"):
            c["grass/meadow/heath"] += 1
        if t.get("wetland") == "mangrove":
            c["mangrove"] += 1
        if t.get("railway") in ("rail", "light_rail", "tram", "subway", "narrow_gauge"):
            c["railway"] += 1
        if t.get("railway") == "platform" or t.get("public_transport") == "platform":
            c["platform"] += 1
        if t.get("barrier") in ("wall", "retaining_wall", "city_wall", "fence", "guard_rail"):
            c["wall/fence"] += 1
        if "building:part" in t:
            c["building:part"] += 1
        if t.get("amenity") == "parking" or t.get("leisure") in ("pitch", "playground") or "area:highway" in t or t.get("place") == "square":
            c["ground surface"] += 1
        if t.get("man_made") in ("pier", "bridge"):
            c["pier/bridge"] += 1
        if t.get("power") in ("line", "minor_line"):
            c["power line"] += 1
        if t.get("power") in ("tower", "pole"):
            c["power tower/pole"] += 1
    return c


def main():
    ap = argparse.ArgumentParser(description="Fetch OSM + DEM into cache")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--size", type=float, default=100)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--dem-source", default="auto",
                    help="auto | layered | id_demnas | mapterhorn | us_3dep | nl_ahn | fr_ign | uk_ea | "
                         "no_kartverket | arcticdem | rema | gedtm30 | fabdem | vt_vcgi | il_isgs | ky_kyfromabove | "
                         "cz_cuzk | hk_cedd | au_qld | ee_maaamet | ht_cnigs | jp_gsj | mx_inegi | nz_linz | "
                         "tz_msimbazi | bw_dgm1 | copernicus | file:<path>  (id_demnas, fabdem, nz_linz, tz_msimbazi "
                         "are registered but unavailable)")
    ap.add_argument("--no-overture", action="store_true", help="skip the Overture Maps buildings download")
    ap.add_argument("--no-lod2", action="store_true", help="skip the LoD2 3D buildings download")
    ap.add_argument("--lod2-allow-large", action="store_true", help="allow the large LoD2 downloads (Japan, Switzerland)")
    a = ap.parse_args()
    d = fetch(square_boundary(a.lat, a.lon, a.size), a.cache, a.dem_source, overture=not a.no_overture,
              lod2=not a.no_lod2, lod2_allow_large=a.lod2_allow_large)
    print("cache:", d.resolve())
    osm = json.loads((d / "osm.json").read_text(encoding="utf-8"))
    for k, v in sorted(count_categories(osm).items()):
        print(f"  {k}: {v}")
    import rasterio
    with rasterio.open(d / "dem.tif") as ds:
        arr = ds.read(1, masked=True)
        print(f"DEM {ds.crs} shape={arr.shape} res={ds.res} min={arr.min():.2f} max={arr.max():.2f}")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    print("source:", meta["dem_source"], "|", meta["dem_attribution"])
    print("accuracy:", json.dumps(meta["dem_accuracy"]))
    if meta.get("dem_fallback_reason"):
        print("fallback:", meta["dem_fallback_reason"])
    if meta.get("overture"):
        print("overture:", json.dumps(meta["overture"]))
    if meta.get("lod2"):
        print("lod2:", json.dumps({k: v for k, v in meta["lod2"].items() if k != "warnings"}))
    for w in meta.get("warnings", []):
        print("WARNING:", w)


if __name__ == "__main__":
    main()
