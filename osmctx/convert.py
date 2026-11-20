"""S2 converter: cache/<hash>/ (osm.json, dem.tif, meta.json) -> out/site.json (+ out/preview.svg).

Coordinates: UTM grid (EPSG 326xx/327xx) offsets from the boundary centroid.
x = E - E0 (grid east), y = N - N0 (grid north), z = absolute DEM height - origin.elevation.
Grid north differs from true north by the meridian convergence and UTM lengths differ from
ground lengths by the point scale factor; both are reported in report.info.crs.
"""
from __future__ import annotations

import argparse
import copy
import html
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import shapely
from pyproj import CRS, Proj, Transformer
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, shape
from shapely.geometry.polygon import orient
from shapely.ops import linemerge, polygonize, unary_union
from shapely.validation import make_valid

from . import features
from . import overture as ovt
from . import trees as treelib
from . import vegetation as veg
from .boundary import to_polygon

DEFAULT_SETTINGS = {
    "floor_height": 3.2,
    "default_levels": 3,
    "tree_height": 8.0,           # last-resort tree height (m) when nothing else is known
    "broadleaf_height": 12.0,     # leaf_type=broadleaved default height (m)
    "conifer_height": 15.0,       # leaf_type=needleleaved default height (m)
    "broadleaf_crown_ratio": 0.65,  # crown diameter / height
    "conifer_crown_ratio": 0.35,
    "use_genus_table": True,      # osmctx/trees.py typical genus/species sizes
    "grid_spacing": None,
    "contours": True,             # site-plan contour lines (builder; osmctx/contours.py). Off = no polylines
    "contour_interval": None,     # m; None = auto from relief (0.25|0.5|1|2|5|10); UI/builder setting, not used by the converter
    "terrain_min_spacing": None,  # m; auto grid never finer than this (super-light preset: 20). Ignored if grid_spacing set
    "road_classes_exclude": [],   # highway classes to skip (super-light: footway/path/cycleway/steps/service)
    "road_widths": {
        "motorway": 12, "trunk": 10, "primary": 8, "secondary": 7, "tertiary": 6,
        "residential": 5, "unclassified": 5, "living_street": 4, "service": 3.5,
        "pedestrian": 4, "footway": 2, "path": 2, "cycleway": 2, "steps": 2, "track": 3,
        "default": 4,
    },
    "include": {"terrain": True, "buildings": True, "roads": True, "trees": True, "water": True, "shrubs": True,
                "land_cover": True, "overture_buildings": True, **features.INCLUDE_DEFAULTS},
    # Overture Maps buildings fill gaps in OSM: kept only if < this share of their own area is covered by OSM
    # buildings (osmctx/overture.py conflate); heights Overture height -> num_floors x floor_height -> default
    "overture_overlap_max": ovt.OVERLAP_MAX,
    "crown_neighbour_cap": True,  # ratio-derived crowns may not exceed 1.15 x nearest-neighbour spacing
    # "dsm" = use the DEM as fetched; "bare_earth_approx" = mask DEM cells under buildings/wood/trees and
    # rebuild them from the remaining cells. None = auto: bare_earth_approx for surface models (DSM).
    "ground_mode": None,
}
DEFAULT_SETTINGS.update(veg.VEG_DEFAULTS)  # forest_scatter, shrub/hedge settings (osmctx/vegetation.py)
DEFAULT_SETTINGS.update(features.FEATURE_DEFAULTS)  # railways, walls, parts/roofs, ground, piers, power (osmctx/features.py)
GROUND_MODES = ("dsm", "bare_earth_approx")
GROUND_BUILDING_BUFFER = 2.0   # m, building footprints grown by this before masking DEM cells
GROUND_TREE_RADIUS = 3.0       # m, crown radius for natural=tree nodes / tree_row lines
GROUND_COVER_THRESHOLD = 0.25  # mask a DEM cell if >= this share of its area is covered
GROUND_SUBSAMPLES = 5          # k x k sub-points per DEM cell for the coverage estimate
GROUND_NEIGHBOURS = 12         # nearest unmasked cells used for the local plane fit
GROUND_MIN_REFS = 6            # fewer open-LAND reference cells -> bare_earth skipped (DSM kept) with a warning
GROUND_MAX_REF_DIST_CELLS = 3.0  # a masked cell with no land reference within this many cells keeps its DSM value
GROUND_MAX_LOWERING = 30.0     # m, never lower a DSM cell by more (canopy/buildings; cliffs are not "vegetation")
GROUND_BUILDING_LOWERING_FACTOR = 2.0  # building-covered cells: lowering <= this x the building height (and <= 30 m)
DESPIKE_M = 2.0                # DSM single-cell pits deeper than this below all 8 neighbours are filled
GROUND_COAST_LOW_M = 0.5      # with a coastline: DEM cells <= this (sea level / GLO-30 water mask) are no ground refs
SEA_LEVEL = 0.0                # sea surface z (abs) - EGM2008 for GLO-30 (~NHN for DGM1)
SEA_LINK_DIST = 2.0            # m, water within this distance of the sea is "sea-linked": z >= sea level
STEEP_FOOTPRINT_M = 3.0        # building footprint terrain spread above this -> flagged osmctx:steep_footprint
PEAK_TOLERANCE_M = 10.0        # |DEM - OSM ele| above max(this, 2 x DEM vertical accuracy) -> warning
STEEP_SLOPE_DEG = 45.0         # DEM slope above this counts as "steep" in report.dem_accuracy.slope
LAND_COVER_NATURAL = ("beach", "sand", "wetland")
LAND_COVER_WETLAND = ("tidalflat", "saltmarsh", "mangrove")
# (tag, value) -> land_cover kind for green areas: grass (lawn, village green, recreation ground, cemetery,
# natural=grassland), meadow, heath. Parks/gardens are NOT land cover (they overlap paths/buildings; trees only).
LAND_COVER_GREEN = {("landuse", "grass"): "grass", ("landuse", "village_green"): "grass",
                    ("landuse", "recreation_ground"): "grass", ("landuse", "cemetery"): "grass",
                    ("natural", "grassland"): "grass", ("landuse", "meadow"): "meadow", ("natural", "heath"): "heath"}

WATERWAY_WIDTHS = {"river": 20.0, "canal": 10.0, "stream": 3.0, "brook": 3.0, "tidal_channel": 5.0,
                   "ditch": 1.5, "drain": 1.5, "canoe_pass": 3.0}
RIVER_WATER_VALUES = {"river", "canal", "stream", "ditch", "drain", "rapids", "stream_pool", "riverbank"}
ROAD_SKIP_CLASSES = {"platform", "proposed", "construction", "abandoned", "razed", "disused",
                     "corridor", "elevator", "services", "rest_area", "bus_stop", "no"}
MAX_TERRAIN_POINTS = 2500
ROAD_MAX_SEG = 10.0          # m, road densification
ROAD_SMOOTH_HALFWIDTH = 15.0  # m, triangular kernel half-width for non-bridge z smoothing
BRIDGE_CLEARANCE_PER_LAYER = 5.5  # m per layer: estimated deck height above the DEM at a chain end cut by the data
BRIDGE_MIN_CLEARANCE = 5.5    # m, a higher-layer deck is >= this above a lower-layer road at every crossing
BRIDGE_WATER_CLEARANCE = 2.0  # m, a deck over an area water (river/lake/sea polygon) is >= this above the water
BRIDGE_PASSES = 4             # relaxation passes (chains that mix layers depend on each other)
RAMP_CAP = 0.08              # max grade a bridge/tunnel profile may use to reach a crossing (vehicle classes)
RAMP_CAP_SLOW = 0.12          # footway/path/cycleway/pedestrian/track/bridleway
RAMP_SLOW_CLASSES = {"footway", "path", "cycleway", "pedestrian", "track", "bridleway", "steps", "corridor"}
TREE_ROW_SPACING = 8.0
MIN_AREA = 2.0                # m2, sliver threshold

_NUM = r"(\d+(?:\.\d+)?)"
_RE_FEET = re.compile(_NUM + r"\s*(?:'|ft|feet|foot)\s*(?:" + _NUM + r"\s*(?:\"|''|in|inch|inches)?)?")
_RE_INCH = re.compile(_NUM + r"\s*(\"|in|inch|inches)")
_RE_METRIC = re.compile(_NUM + r"\s*(m|meter|meters|metre|metres|km|cm|mm|mi)?\.?")
_UNIT_M = {None: 1.0, "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0, "km": 1000.0,
           "cm": 0.01, "mm": 0.001, "mi": 1609.344}


# --------------------------------------------------------------------------- parsing

def parse_length(value, max_value: float = 1000.0) -> float | None:
    """OSM length tag -> metres. '12', '12 m', '12.5m', '12,5', "40'", "40'6\"", '40 ft', '3 km'.
    Returns None for empty/garbage/non-positive/absurd values."""
    if value is None:
        return None
    s = str(value).strip().lower().replace(",", ".")
    s = s.split(";")[0].strip().lstrip("~").strip()
    if not s:
        return None
    m = _RE_FEET.fullmatch(s)
    if m:
        v = float(m[1]) * 0.3048 + (float(m[2]) * 0.0254 if m[2] else 0.0)
    else:
        m = _RE_INCH.fullmatch(s)
        if m:
            v = float(m[1]) * 0.0254
        else:
            m = _RE_METRIC.fullmatch(s)
            if not m:
                return None
            v = float(m[1]) * _UNIT_M[m[2]]
    if not (v > 0) or v > max_value:
        return None
    return v


def parse_number(value) -> float | None:
    """First number in a tag value ('3', '3.5', '3;4', '3,5'). None if absent."""
    if value is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", "."))
    return float(m[0]) if m else None


def parse_layer(value) -> int:
    v = parse_number(value)
    return int(round(v)) if v is not None else 0


def building_height(tags: dict, s: dict) -> tuple[float, str]:
    h = parse_length(tags.get("height"))
    if h is not None:
        return h, "height"
    lv = parse_number(tags.get("building:levels"))
    if lv is not None and lv >= 0:
        rl = parse_number(tags.get("roof:levels"))
        h = lv * s["floor_height"] + (max(rl, 0.0) * s["floor_height"] * 0.5 if rl is not None else 0.0)
        if h > 0.5:
            return h, "levels"
    return s["default_levels"] * s["floor_height"], "default"


def road_width(tags: dict, s: dict) -> tuple[float, str, str]:
    """(width, width_source per CONTRACT 'tag'|'default', detail 'tag'|'class'|'default')."""
    for k in ("width", "est_width"):
        w = parse_length(tags.get(k), max_value=200.0)
        if w is not None:
            return w, "tag", "tag"
    cls = str(tags.get("highway", ""))
    rw = s["road_widths"]
    for c in (cls, cls[:-5] if cls.endswith("_link") else None):
        if c and c in rw and c != "default":
            return float(rw[c]), "default", "class"
    return float(rw.get("default", 4.0)), "default", "default"


# --------------------------------------------------------------------------- CRS / frame

def utm_epsg(lon: float, lat: float) -> int:
    """WGS84 UTM EPSG code (326xx north / 327xx south), or Polar Stereographic (3413/3031) at poles."""
    if lat > 84.0:
        return 3413
    if lat < -80.0:
        return 3031
    lon = ((lon + 180.0) % 360.0) - 180.0
    zone = min(int((lon + 180.0) // 6.0) + 1, 60)
    if 56.0 <= lat < 64.0 and 3.0 <= lon < 12.0:
        zone = 32
    elif 72.0 <= lat <= 84.0 and 0.0 <= lon < 42.0:
        zone = 31 if lon < 9 else 33 if lon < 21 else 35 if lon < 33 else 37
    return (32600 if lat >= 0 else 32700) + zone


class Frame:
    """WGS84 <-> local site metres (UTM grid offsets from the boundary centroid)."""

    def __init__(self, poly_wgs: Polygon):
        c = poly_wgs.centroid
        epsg = utm_epsg(c.x, c.y)
        for _ in range(2):  # zone from origin lon/lat; re-pick once if centroid lands in another zone
            self.epsg = epsg
            self.fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
            self.inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
            pu = shapely.transform(poly_wgs, lambda a: np.column_stack(self.fwd.transform(a[:, 0], a[:, 1])))
            cu = pu.centroid
            self.E0, self.N0 = float(cu.x), float(cu.y)
            self.lon0, self.lat0 = (float(v) for v in self.inv.transform(self.E0, self.N0))
            epsg2 = utm_epsg(self.lon0, self.lat0)
            if epsg2 == epsg:
                break
            epsg = epsg2
        self._crs_info: dict | None = None

    def to_local(self, lon, lat):
        e, n = self.fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(e) - self.E0, np.asarray(n) - self.N0

    def to_lonlat(self, x, y):
        lon, lat = self.inv.transform(np.asarray(x, float) + self.E0, np.asarray(y, float) + self.N0)
        return np.asarray(lon), np.asarray(lat)

    def geom_to_local(self, g):
        def f(a):
            x, y = self.to_local(a[:, 0], a[:, 1])
            return np.column_stack([x, y])
        return shapely.transform(g, f)

    def crs_info(self) -> dict:
        if self._crs_info is None:  # lon0/lat0 are fixed after __init__; pyproj CRS/Proj construction is slow
            crs = CRS.from_epsg(self.epsg)
            fac = Proj(crs).get_factors(self.lon0, self.lat0)
            self._crs_info = {"epsg": self.epsg, "name": crs.name,
                              "meridian_convergence_deg": round(float(fac.meridian_convergence), 5),
                              "point_scale_factor": round(float(fac.meridional_scale), 7),
                              "note": "x/y are UTM grid offsets: grid north = +y; true north is rotated by the "
                                      "meridian convergence; grid length = ground length * scale factor."}
        return dict(self._crs_info)


# --------------------------------------------------------------------------- DEM

class Dem:
    """Bilinear sampler; values are taken at pixel centres (GDAL corner-based transform)."""

    def __init__(self, path: Path):
        import rasterio
        with rasterio.open(path) as ds:
            a = ds.read(1).astype("float64")
            t = ds.transform
            nodata = ds.nodata
            crs = ds.crs
        if t.b != 0 or t.d != 0:
            raise ValueError("rotated DEM transforms not supported")
        self.warnings: list[str] = []
        bad = ~np.isfinite(a)
        if nodata is not None:
            bad |= a == nodata
        if bad.any():
            fill = float(np.nanmean(np.where(bad, np.nan, a))) if (~bad).any() else 0.0
            a = np.where(bad, fill, a)
            self.warnings.append(f"DEM has {int(bad.sum())} nodata cells, filled with mean {fill:.1f} m")
        self.a = a
        self.nrows, self.ncols = a.shape
        self.x0, self.dx, self.y0, self.dy = t.c, t.a, t.f, t.e
        self.to_dem = None
        self._from_dem = None   # inverse transformer (DEM CRS -> WGS84), built on first use
        if crs is not None and crs.to_epsg() != 4326:
            self.to_dem = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self.n_outside = 0

    def _native_xy(self, lon, lat):
        """lon/lat (WGS84) -> arrays in the DEM CRS (identity for a geographic DEM)."""
        lon = np.atleast_1d(np.asarray(lon, float))
        lat = np.atleast_1d(np.asarray(lat, float))
        if self.to_dem is not None:
            lon, lat = (np.asarray(v) for v in self.to_dem.transform(lon, lat))
        return lon, lat

    def _to_wgs84(self, x, y):
        """DEM-CRS coordinates -> lon/lat (identity for a geographic DEM)."""
        if self.to_dem is None:
            return x, y
        if self._from_dem is None:
            self._from_dem = Transformer.from_crs(self.to_dem.target_crs, "EPSG:4326", always_xy=True)
        return self._from_dem.transform(x, y)

    def sample(self, lon, lat) -> np.ndarray:
        lon, lat = self._native_xy(lon, lat)
        fc = (lon - self.x0) / self.dx - 0.5
        fr = (lat - self.y0) / self.dy - 0.5
        out = (fc < -0.51) | (fc > self.ncols - 0.49) | (fr < -0.51) | (fr > self.nrows - 0.49)
        self.n_outside += int(out.sum())
        fc = np.clip(fc, 0, self.ncols - 1)
        fr = np.clip(fr, 0, self.nrows - 1)
        c0 = np.minimum(np.floor(fc).astype(int), max(self.ncols - 2, 0))
        r0 = np.minimum(np.floor(fr).astype(int), max(self.nrows - 2, 0))
        c1 = np.minimum(c0 + 1, self.ncols - 1)
        r1 = np.minimum(r0 + 1, self.nrows - 1)
        tc = fc - c0
        tr = fr - r0
        a = self.a
        top = a[r0, c0] * (1 - tc) + a[r0, c1] * tc
        bot = a[r1, c0] * (1 - tc) + a[r1, c1] * tc
        return top * (1 - tr) + bot * tr

    def contains(self, lon, lat) -> np.ndarray:
        lon, lat = self._native_xy(lon, lat)
        fc = (lon - self.x0) / self.dx
        fr = (lat - self.y0) / self.dy
        return (fc >= 0) & (fc <= self.ncols) & (fr >= 0) & (fr <= self.nrows)

    def cell_centres(self):
        cols, rows = np.meshgrid(np.arange(self.ncols), np.arange(self.nrows))
        lon = self.x0 + (cols + 0.5) * self.dx
        lat = self.y0 + (rows + 0.5) * self.dy
        lon, lat = self._to_wgs84(lon, lat)
        return np.ravel(lon), np.ravel(lat), np.ravel(self.a)

    def subcell_lonlat(self, k: int):
        """k x k sub-points inside every cell -> lon, lat arrays shaped (nrows*ncols, k*k)."""
        off = (np.arange(k) + 0.5) / k
        cols, rows = np.meshgrid(np.arange(self.ncols), np.arange(self.nrows))
        ox, oy = np.meshgrid(off, off)
        X = self.x0 + (cols.ravel()[:, None] + ox.ravel()[None, :]) * self.dx
        Y = self.y0 + (rows.ravel()[:, None] + oy.ravel()[None, :]) * self.dy
        X, Y = self._to_wgs84(X, Y)
        return np.asarray(X), np.asarray(Y)

    def cell_size_m(self, lat: float) -> tuple[float, float]:
        if self.to_dem is not None:
            return abs(self.dx), abs(self.dy)
        return (abs(self.dx) * 111320.0 * math.cos(math.radians(lat)), abs(self.dy) * 110574.0)


class Site:
    def __init__(self, frame: Frame, dem: Dem):
        self.f = frame
        self.dem = dem
        self.elev0 = 0.0

    def z_abs(self, x, y) -> np.ndarray:
        lon, lat = self.f.to_lonlat(np.atleast_1d(x), np.atleast_1d(y))
        return self.dem.sample(lon, lat)


# --------------------------------------------------------------------------- geometry helpers

def densify(coords: np.ndarray, max_seg: float) -> tuple[np.ndarray, np.ndarray]:
    """Insert points so no segment > max_seg. Returns (points, indices of original vertices)."""
    coords = np.asarray(coords, float)
    out = [coords[:1]]
    idx = [0]
    for i in range(1, len(coords)):
        a, b = coords[i - 1], coords[i]
        n = max(1, int(math.ceil(float(np.hypot(*(b[:2] - a[:2]))) / max_seg - 1e-9)))
        t = (np.arange(1, n + 1) / n)[:, None]
        out.append(a + (b - a) * t)
        idx.append(idx[-1] + n)
    return np.vstack(out), np.asarray(idx)


def arclen(pts: np.ndarray) -> np.ndarray:
    d = np.hypot(*np.diff(pts[:, :2], axis=0).T) if len(pts) > 1 else np.zeros(0)
    return np.concatenate([[0.0], np.cumsum(d)])


def smooth_along(s: np.ndarray, z: np.ndarray, h: float) -> np.ndarray:
    """Triangular-kernel moving average along arc length, weighted by each point's share of length."""
    if len(z) < 3 or s[-1] <= 0:
        return z.copy()
    w = np.empty_like(s)
    w[1:-1] = (s[2:] - s[:-2]) / 2
    w[0] = (s[1] - s[0]) / 2
    w[-1] = (s[-1] - s[-2]) / 2
    w = w + 1e-9
    k = np.clip(1 - np.abs(s[:, None] - s[None, :]) / h, 0, None) * w[None, :]
    return (k @ z) / k.sum(axis=1)


def polygons_of(g) -> list[Polygon]:
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    if hasattr(g, "geoms"):
        return [p for sub in g.geoms for p in polygons_of(sub)]
    return []


def lines_of(g) -> list[LineString]:
    if g is None or g.is_empty:
        return []
    if isinstance(g, LineString):
        return [g]
    if hasattr(g, "geoms"):
        return [p for sub in g.geoms for p in lines_of(sub)]
    return []


def clean_polygons(g, min_area: float = MIN_AREA) -> tuple[list[Polygon], int]:
    """make_valid, drop slivers/tiny holes, remove repeated + collinear points, orient CCW/CW."""
    out, dropped = [], 0
    for p in polygons_of(make_valid(g)):
        p = shapely.remove_repeated_points(p, 0.01)
        p = p.simplify(0.01, preserve_topology=True)  # removes collinear vertices (<1 cm deviation)
        for q in polygons_of(make_valid(p)):
            if q.area < min_area:
                dropped += 1
                continue
            holes = [r for r in q.interiors if Polygon(r).area >= min_area]
            q = orient(Polygon(q.exterior, holes), sign=1.0)
            out.append(q)
    return out, dropped


def ring_xy(ring) -> list[list[float]]:
    c = list(ring.coords)
    if len(c) > 1 and c[0] == c[-1]:
        c = c[:-1]
    return [[round(x, 3), round(y, 3)] for x, y, *_ in c]


def assemble_rings(lines: list[LineString]) -> tuple[list[Polygon], int]:
    """Join way pieces into closed rings. Returns (ring polygons, n unclosed pieces dropped)."""
    if not lines:
        return [], 0
    merged = lines_of(linemerge(unary_union(lines))) if len(lines) > 1 else lines
    polys, bad = [], 0
    for ln in merged:
        c = list(ln.coords)
        if c[0] != c[-1]:
            if math.dist(c[0][:2], c[-1][:2]) < 1.0 and len(c) >= 3:
                c.append(c[0])
            else:
                bad += 1
                continue
        if len(c) >= 4:
            polys.append(Polygon(c))
    if not polys and lines:  # fallback: noded polygonize
        polys = list(polygonize(unary_union(lines)))
    return polys, bad


def clip_polyline_z(pts: np.ndarray, poly: Polygon) -> list[np.ndarray]:
    """Clip an (n,3) polyline to a polygon segment by segment, interpolating z at cut points."""
    runs: list[list[np.ndarray]] = []
    cur = None
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return []
    shapely.prepare(poly)
    a, b = pts[:-1], pts[1:]
    d = b[:, :2] - a[:, :2]
    L2 = (d * d).sum(axis=1)
    nonzero = L2 != 0
    segs = np.empty(len(L2), dtype=object)
    covered = np.zeros(len(L2), bool)
    if nonzero.any():   # one vectorised covers() over every segment instead of one LineString + predicate each
        segs[nonzero] = shapely.linestrings(np.stack([a[nonzero, :2], b[nonzero, :2]], axis=1))
        covered[nonzero] = shapely.covers(poly, segs[nonzero])
    for i in np.flatnonzero(nonzero):
        if covered[i]:
            pieces = [(0.0, 1.0)]
        else:
            pieces = []
            for part in lines_of(segs[i].intersection(poly)):
                ts = [float((np.asarray(p[:2]) - a[i, :2]) @ d[i] / L2[i]) for p in (part.coords[0], part.coords[-1])]
                t0, t1 = min(ts), max(ts)
                if t1 - t0 > 1e-9:
                    pieces.append((max(t0, 0.0), min(t1, 1.0)))
            pieces.sort()
        for t0, t1 in pieces:
            p0 = a[i] + (b[i] - a[i]) * t0
            p1 = a[i] + (b[i] - a[i]) * t1
            if cur is not None and np.hypot(*(cur[-1][:2] - p0[:2])) < 1e-6:
                cur.append(p1)
            else:
                cur = [p0, p1]
                runs.append(cur)
        if not pieces:
            cur = None
    return [np.asarray(r) for r in runs]


# --------------------------------------------------------------------------- OSM access

def osm_id(el) -> str:
    return f"{el['type']}/{el['id']}"


def way_lonlat(geom) -> list[tuple[float, float]]:
    return [(g["lon"], g["lat"]) for g in (geom or []) if g]


def way_nodes_lonlat(el) -> tuple[list, list]:
    geom = el.get("geometry") or []
    nodes = el.get("nodes") or [None] * len(geom)
    if len(nodes) != len(geom):
        nodes = [None] * len(geom)
    pairs = [(n, (g["lon"], g["lat"])) for n, g in zip(nodes, geom) if g]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def relation_area(el, frame: Frame) -> tuple[object, int]:
    """Multipolygon relation -> local geometry (even-odd of all rings), n unclosed pieces."""
    lines = []
    for m in el.get("members", []):
        if m.get("type") != "way" or m.get("role", "outer") not in ("outer", "inner", ""):
            continue
        ll = way_lonlat(m.get("geometry"))
        if len(ll) >= 2:
            lines.append(frame.geom_to_local(LineString(ll)))
    rings, bad = assemble_rings(lines)
    if not rings:
        return None, bad
    rings.sort(key=lambda p: -p.area)
    gs = [make_valid(r) for r in rings]
    while len(gs) > 1:   # symmetric difference is associative: pairwise tree, not one ever-growing accumulator (quadratic)
        nxt = [gs[i].symmetric_difference(gs[i + 1]) for i in range(0, len(gs) - 1, 2)]
        if len(gs) % 2:
            nxt.append(gs[-1])
        gs = nxt
    return gs[0], bad


def is_closed_way(ll) -> bool:
    return len(ll) >= 4 and ll[0] == ll[-1]


# --------------------------------------------------------------------------- DEM provenance

DEM_META_KEYS = ("dem_source", "dem_source_id", "dem_attribution", "dem_licence", "dem_vertical_datum", "dem_crs",
                 "dem_accuracy", "dem_requested", "dem_fallback_reason")


def dem_meta(meta: dict) -> dict:
    """DEM provenance + accuracy from meta.json (fetch v2); infers legacy M1 caches (GLO-30 only)."""
    if meta.get("dem_accuracy"):
        return {k: meta.get(k) for k in DEM_META_KEYS}
    src = meta.get("dem_source")
    if src and "GLO-30" in src:
        from .fetch import DEM_INFO
        inf = DEM_INFO["copernicus"]
        return {"dem_source": src, "dem_source_id": "copernicus", "dem_attribution": inf["attribution"],
                "dem_licence": inf["licence"], "dem_vertical_datum": inf["vertical_datum"], "dem_crs": "EPSG:4326",
                "dem_accuracy": dict(inf["accuracy"]), "dem_requested": None, "dem_fallback_reason": None}
    return {"dem_source": src, "dem_source_id": None, "dem_attribution": None, "dem_licence": None,
            "dem_vertical_datum": None, "dem_crs": None, "dem_requested": None, "dem_fallback_reason": None,
            "dem_accuracy": {"horizontal_m": None, "vertical_m": None, "kind": "unknown",
                             "note": "DEM source not recorded in meta.json; accuracy unknown."}}


def _local_plane(px, py, pz, qx, qy, cell: float, k: int = GROUND_NEIGHBOURS) -> float:
    """Value at (qx, qy) of an inverse-distance weighted least-squares plane through the k nearest
    samples; weighted mean when they cannot define a plane."""
    d2 = (px - qx) ** 2 + (py - qy) ** 2
    return _plane_at(px, py, pz, qx, qy, np.argsort(d2)[:k], d2, cell)


def _plane_at(px, py, pz, qx, qy, idx, d2, cell: float) -> float:
    """_local_plane with the k nearest sample indices `idx` (and all squared distances `d2`) already known."""
    w = 1.0 / (d2[idx] + cell * cell)
    if len(idx) >= 3:
        A = np.column_stack([np.ones(len(idx)), px[idx] - qx, py[idx] - qy]) * np.sqrt(w)[:, None]
        coef, _res, rank, _sv = np.linalg.lstsq(A, pz[idx] * np.sqrt(w), rcond=None)
        if rank == 3:
            return float(coef[0])
    return float((pz[idx] * w).sum() / w.sum())


# --------------------------------------------------------------------------- converter

def _merge_settings(settings: dict | None) -> dict:
    s = copy.deepcopy(DEFAULT_SETTINGS)
    for k, v in (settings or {}).items():
        if k in ("road_widths", "include", "tree_scatter") and isinstance(v, dict):
            s[k].update(v)
        elif k == "barrier_defaults" and isinstance(v, dict):
            s[k] = {kd: {**s[k].get(kd, {}), **(d or {})} for kd, d in {**s[k], **v}.items()}
        elif k == "tree_scatter_params" and isinstance(v, dict):
            s[k] = {t: {**(s[k].get(t) or {}), **(o or {})} for t, o in {**s[k], **v}.items()}
        elif v is not None or k == "grid_spacing":
            s[k] = v
    s["floor_height"] = float(s["floor_height"])
    s["default_levels"] = float(s["default_levels"])
    for k in ("tree_height", "broadleaf_height", "conifer_height", "broadleaf_crown_ratio", "conifer_crown_ratio"):
        s[k] = float(s[k])
    s["use_genus_table"] = bool(s["use_genus_table"])
    s["road_classes_exclude"] = sorted({str(c).strip().lower() for c in (s.get("road_classes_exclude") or []) if str(c).strip()})
    if s.get("terrain_min_spacing"):
        s["terrain_min_spacing"] = float(s["terrain_min_spacing"])
    veg.normalize_settings(s)
    features.normalize_settings(s)
    return s


def _near_any(px: np.ndarray, py: np.ndarray, nxy: np.ndarray, r: float) -> np.ndarray:
    """Per point (px, py): is any of the (n, 2) points `nxy` closer than r? Chunked so memory stays bounded."""
    if len(px) == 0 or len(nxy) == 0:
        return np.zeros(len(px), bool)
    step = max(1, 2_000_000 // len(nxy))
    return np.concatenate([
        np.hypot(nxy[None, :, 0] - px[i:i + step, None], nxy[None, :, 1] - py[i:i + step, None]).min(axis=1) < r
        for i in range(0, len(px), step)])


class _Crossings:
    """Crossings between the densified highway/rail ways of one roads() run. Line geometries and the STRtree are
    built once; the crossing geometry of a way pair (i, j) is computed once on demand - only the z of way j changes
    between the bridge relaxation passes and the final clearance check, the geometry does not."""

    def __init__(self, recs: list[dict], settings: dict):
        self.recs = recs
        self.s = settings
        self.lines = [LineString(r["dens"]) for r in recs]
        self.tree = shapely.STRtree(self.lines)
        self._pairs: dict = {}

    def _geometry(self, i: int, j: int) -> list:
        """[(s_i, s_j, half-width_j)] of the crossings of way i with way j. Contacts at shared nodes (junctions)
        are not crossings."""
        res = self._pairs.get((i, j))
        if res is None:
            r, o, lines = self.recs[i], self.recs[j], self.lines
            shared = set(r["nodes"]) & set(o["nodes"])
            sh_xy = [r["dens"][k] for n, k in zip(r["nodes"], r["oidx"]) if n in shared]
            ip = lines[i].intersection(lines[j])
            res = []
            for p in [g for g in getattr(ip, "geoms", [ip]) if g.geom_type == "Point"]:
                if any(math.hypot(p.x - a, p.y - b) < 0.5 for a, b in sh_xy):
                    continue
                res.append((lines[i].project(p), lines[j].project(p), road_width(o["t"], self.s)[0] / 2))
            self._pairs[(i, j)] = res
        return res

    def of(self, i: int, lower) -> list:
        """Crossings of way i with ways j for which lower(j) is true: [(s_i, z_j abs, half-width_j)]."""
        res = []
        for j in self.tree.query(self.lines[i], predicate="intersects"):
            j = int(j)
            if j == i or not lower(j) or self.recs[j].get("z") is None:
                continue
            o = self.recs[j]
            for si, sj, wj in self._geometry(i, j):
                res.append((si, float(np.interp(sj, o["s"], o["z"])), wj))
        return res


class Converter:
    def __init__(self, cache_dir, settings: dict | None):
        self.cache = Path(cache_dir)
        self.s = _merge_settings(settings)
        self.meta = json.loads((self.cache / "meta.json").read_text(encoding="utf-8"))
        self.osm = json.loads((self.cache / "osm.json").read_text(encoding="utf-8"))
        self.els = self.osm.get("elements", [])
        self.poly_wgs = to_polygon(self.meta["boundary"])
        self.f = Frame(self.poly_wgs)
        self.dem = Dem(self.cache / "dem.tif")
        # GLO-30 around bridge/tunnel chain ends beyond the DEM (fetch.write_dem_ext); abutment heights only
        self.dem_ext = Dem(self.cache / "dem_ext.tif") if (self.cache / "dem_ext.tif").exists() else None
        self.site = Site(self.f, self.dem)
        self.layer_info: dict = {}
        self.dem_info = dem_meta(self.meta)
        kind = self.dem_info["dem_accuracy"].get("kind")
        gm = self.s.get("ground_mode")
        if gm is None:
            gm = "bare_earth_approx" if kind == "DSM" else "dsm"
        elif gm not in GROUND_MODES:
            raise ValueError(f"ground_mode must be one of {GROUND_MODES} or None, got {gm!r}")
        self.ground_mode = gm
        self.ground_info = {"mode": gm, "requested": self.s.get("ground_mode") or "auto", "dem_kind": kind,
                            "applied": False}
        self.ground_warnings: list[str] = []
        bl = self.f.geom_to_local(self.poly_wgs)
        if isinstance(bl, MultiPolygon):
            bl = max(bl.geoms, key=lambda p: p.area)
        bl = orient(shapely.remove_repeated_points(bl, 0.01), 1.0)
        self.bpoly = bl
        shapely.prepare(self.bpoly)
        # lazily filled memo/state (set here so no getattr/hasattr probing is needed)
        self._ground_masks_cache = None   # (bld, wood, trees, water) local polygons, see _ground_mask_features
        self._bld_heights: list[float] = []
        self._cxy = None                  # DEM cell centres in local metres (geometry only: valid across despike/bare_earth)
        self._cell_in_b = None
        self._wg = None                   # _water_geoms result
        self._ov = None                   # _overture result
        self._area_memo: dict = {}        # id(osm element) -> _area_local result
        self._water_polys: list[dict] = []
        self._water_union_b = None
        self._water_tree = None
        self.rail_ways: list = []
        self.tree_xy: list = []
        self._ramp_viol: list = []
        self.counts: Counter = Counter()
        self.heights = Counter()
        self.widths = Counter()
        self.tree_heights = Counter()
        self.tree_crowns = Counter()
        self.scatter_report: dict = {}   # area type -> {areas, trees, area_m2} (estimated scattered trees)
        self.ndom_info: dict = {"available": False, "reason": "no ndom.tif in cache"}
        self.waterway_widths = Counter()
        self.warnings: list[str] = []
        self.notes: list[str] = []

    # ---- shared OSM area geometry ----------------------------------------------------
    def _area_local(self, el, t: dict):
        """Closed way / multipolygon relation -> (local geometry | None, n unclosed ring pieces, why-not). why-not is
        None, "unclosed" (open way), "empty" (relation without usable rings) or "other" (node / other relation).
        Memoised per element: buildings are needed by the ground mask, Overture conflation and buildings();
        callers keep their own counters/warnings."""
        r = self._area_memo.get(id(el))
        if r is None:
            if el["type"] == "way":
                ll = way_lonlat(el.get("geometry"))
                r = (self.f.geom_to_local(Polygon(ll)), 0, None) if is_closed_way(ll) else (None, 0, "unclosed")
            elif el["type"] == "relation" and t.get("type") == "multipolygon":
                g, bad = relation_area(el, self.f)
                r = (g, bad, None if g is not None else "empty")
            else:
                r = (None, 0, "other")
            self._area_memo[id(el)] = r
        return r

    # ---- ground (bare-earth approximation of a DSM) --------------------------------
    def _ground_mask_features(self):
        """Local polygons that raise a DSM: buildings (+buffer), wood/forest, tree crowns; plus water
        (DSM water surfaces are edited/flattened and are not land-ground references). Computed once (the
        element list, settings and Overture result are fixed for the Converter's life)."""
        if self._ground_masks_cache is not None:
            return self._ground_masks_cache
        bld, wood, trees, water = [], [], [], []
        node_slots, node_ll = [], []   # natural=tree nodes: projected in one batch, keep their list position
        bld_heights = []  # parallel to bld: OSM/default building height (lowering cap)
        for el in self.els:
            t = el.get("tags", {})
            is_b = ("building" in t and t.get("building") != "no") or "building:part" in t
            is_w = t.get("natural") == "wood" or t.get("landuse") == "forest"
            nat = t.get("natural")
            ww = t.get("waterway")
            is_water = (nat == "water" or ww == "riverbank") and not (is_b or is_w)
            if el["type"] == "way" and ww in WATERWAY_WIDTHS and t.get("tunnel", "no") == "no":
                ll = way_lonlat(el.get("geometry"))
                if len(ll) >= 2:
                    wd = parse_length(t.get("width"), max_value=5000.0) or WATERWAY_WIDTHS[ww]
                    water.append(self.f.geom_to_local(LineString(ll)).buffer(wd / 2, cap_style="flat"))
                continue
            if not (is_b or is_w or is_water or nat in ("tree", "tree_row")):
                continue
            if el["type"] == "node":
                if nat == "tree":
                    node_slots.append(len(trees))
                    trees.append(None)
                    node_ll.append((el["lon"], el["lat"]))
                continue
            if el["type"] == "way" and nat == "tree_row" and not (is_b or is_w):
                ll = way_lonlat(el.get("geometry"))
                if len(ll) >= 2:
                    trees.append(self.f.geom_to_local(LineString(ll)).buffer(GROUND_TREE_RADIUS))
                continue
            g, _, why = self._area_local(el, t)
            if why:
                continue
            g = make_valid(g)
            if is_b:
                bld.append(g.buffer(GROUND_BUILDING_BUFFER))
                bld_heights.append(building_height(t, self.s)[0])
            elif is_water:
                water.append(g)
            else:
                wood.append(g)
        if node_ll:
            nx, ny = self.f.to_local([a for a, _ in node_ll], [b for _, b in node_ll])
            for slot, c in zip(node_slots, shapely.buffer(shapely.points(nx, ny), GROUND_TREE_RADIUS, quad_segs=16)):
                trees[slot] = c
        for q, props in self._overture()["kept"]:   # Overture buildings raise the DSM too
            bld.append(q.buffer(GROUND_BUILDING_BUFFER))
            bld_heights.append(ovt.height_for(props, self.s["floor_height"], self.s["default_levels"])[0])
        self._bld_heights = bld_heights
        self._ground_masks_cache = (bld, wood, trees, water)
        return self._ground_masks_cache

    # ---- Overture Maps buildings (second source, conflated against OSM) ----------------
    def _osm_bld_polys(self) -> list[Polygon]:
        """Local polygons of ALL OSM buildings and building parts in the fetch area (also outside the boundary)."""
        out = []
        for el in self.els:
            t = el.get("tags", {})
            if not (("building" in t and t.get("building") != "no") or "building:part" in t):
                continue
            g, _, why = self._area_local(el, t)
            if why is None:
                out.extend(polygons_of(make_valid(g)))
        return out

    def _overture(self) -> dict:
        """{kept: [(local polygon, props)], stats, on, reason}: Overture buildings that fill gaps in OSM (cached)."""
        if self._ov is None:
            self._ov = self._conflate_overture()
        return self._ov

    def _conflate_overture(self) -> dict:
        ov = {"kept": [], "stats": {}, "on": False, "reason": None}
        inc = self.s["include"]
        path = self.cache / ovt.FILE
        if not inc.get("buildings", True) or not inc.get("overture_buildings", True):
            ov["reason"] = "off"
            return ov
        if not path.exists():
            err = (self.meta.get("overture") or {}).get("error") or "not fetched"
            ov["reason"] = f"no {ovt.FILE} in cache ({err})"
            return ov
        cands = []
        for ft in ovt.load(path):
            try:
                g = self.f.geom_to_local(shape(ft["geometry"]))
            except Exception:  # noqa: BLE001
                continue
            props = ft.get("properties") or {}
            cands.extend((q, props) for q in polygons_of(make_valid(g)))
        kept, stats = ovt.conflate(cands, self._osm_bld_polys(), float(self.s["overture_overlap_max"]))
        ov.update(on=True, kept=kept, stats=stats)
        return ov

    def despike(self):
        """Surface models only: single-cell pits below sea level and lower than all 8 neighbours by > DESPIKE_M are void/edit
        artefacts (GLO-30 shows e.g. -5.4 m cells on a river bank at Palembang); replace them by the neighbour
        median so they cannot drag origin.elevation or bank terrain below the water."""
        a = self.dem.a
        if min(a.shape) < 3:
            return
        p = np.pad(a, 1, mode="edge")
        nb = np.stack([p[1 + dr:1 + dr + a.shape[0], 1 + dc:1 + dc + a.shape[1]]
                       for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc])
        pit = (a < nb.min(axis=0) - DESPIKE_M) & (a < SEA_LEVEL)  # sub-sea-level pits only (void/edit artefacts)
        n = int(pit.sum())
        self.ground_info["dem_pits_filled"] = n
        if n:
            med = np.median(nb, axis=0)
            self.ground_info["dem_pits_max_m"] = round(float((med - a)[pit].max()), 2)
            a = np.where(pit, med, a)
            self.dem.a = a
            self.ground_warnings.append(f"{n} single-cell DEM pit(s) below sea level (> {DESPIKE_M:g} m below all neighbours, up to "
                                        f"{self.ground_info['dem_pits_max_m']:.1f} m) treated as surface-model artefacts "
                                        "and filled with the neighbour median")

    def bare_earth(self):
        """Mask DEM cells covered (>= GROUND_COVER_THRESHOLD) by buildings+2 m / wood / tree crowns and
        rebuild them from open LAND cells (weighted local plane fit of the nearest ones). Reference cells exclude
        water, the sea side of the coastline and (with a coastline) cells <= GROUND_COAST_LOW_M. The result is
        capped at the original value (a DSM is never below ground), at the lowest reference cell, and by a
        lowering limit (<= GROUND_MAX_LOWERING; building cells <= 2 x building height). A masked cell with no
        reference within GROUND_MAX_REF_DIST_CELLS keeps its DSM value; with < GROUND_MIN_REFS references the
        approximation is skipped (DSM kept) with a warning."""
        dem = self.dem
        raw = dem.a.ravel().copy()
        bld, wood, trees, water = self._ground_mask_features()
        sea = self._water_geoms()["sea_full"]
        gi = self.ground_info
        gi.update({"method": "osm_mask_local_plane", "building_buffer_m": GROUND_BUILDING_BUFFER,
                   "tree_crown_radius_m": GROUND_TREE_RADIUS, "cover_threshold": GROUND_COVER_THRESHOLD,
                   "max_lowering_m": GROUND_MAX_LOWERING, "min_land_refs": GROUND_MIN_REFS,
                   "sea_excluded": sea is not None,
                   "mask_features": {"buildings": len(bld), "wood_forest": len(wood), "trees": len(trees),
                                     "water": len(water)}})
        sub = None

        def cover(feats):
            nonlocal sub
            if not feats:
                return np.zeros(raw.size)
            if sub is None:
                lon, lat = dem.subcell_lonlat(GROUND_SUBSAMPLES)
                sub = self.f.to_local(lon, lat)
            mg = unary_union(feats)
            shapely.prepare(mg)
            return shapely.contains_xy(mg, *sub).mean(axis=1)
        masked = cover(bld + wood + trees) >= GROUND_COVER_THRESHOLD
        wet = (cover(water) >= 0.5) & ~masked  # kept as is, but not used as ground reference
        sea_c = cover([sea]) >= 0.5 if sea is not None else np.zeros(raw.size, bool)
        low_coast = (raw <= SEA_LEVEL + GROUND_COAST_LOW_M) if self._water_geoms()["has_coast"] \
            else np.zeros(raw.size, bool)
        cx, cy = self._cell_xy()
        ground = ~masked & ~wet & ~sea_c & ~low_coast
        cw, ch = dem.cell_size_m(self.f.lat0)
        cell = max(cw, ch)
        # lowering cap per cell: 30 m; building cells 2 x the tallest covering building (min 6 m)
        cap = np.full(raw.size, GROUND_MAX_LOWERING)
        if bld:
            hb = np.zeros(raw.size)
            ci, bi = shapely.STRtree(bld).query(shapely.points(cx, cy), predicate="within")
            np.maximum.at(hb, ci, np.asarray(self._bld_heights, float)[bi])
            veg_cov = cover(wood + trees) >= GROUND_COVER_THRESHOLD if (wood or trees) else np.zeros(raw.size, bool)
            only_b = (hb > 0) & ~veg_cov
            cap[only_b] = np.minimum(GROUND_MAX_LOWERING, np.maximum(6.0, GROUND_BUILDING_LOWERING_FACTOR * hb[only_b]))
        new = raw.copy()
        n_g = int(ground.sum())
        in_b = self._cells_in_boundary()
        n_far = 0
        if masked.any() and n_g < GROUND_MIN_REFS:
            gi.update({"applied": False, "method": "skipped_too_few_land_refs", "cells_total": int(raw.size),
                       "cells_masked": int(masked.sum()), "cells_masked_in_boundary": int((masked & in_b).sum()),
                       "cells_in_boundary": int(in_b.sum()), "cells_ground": n_g,
                       "cells_sea_or_coast_excluded": int(((sea_c | low_coast) & ~masked).sum()),
                       "lowering_m": {"max": 0.0, "mean_masked": 0.0}})
            self.ground_warnings.append(
                f"ground_mode=bare_earth_approx SKIPPED: {int(masked.sum())} DEM cells are covered by buildings/"
                f"vegetation but only {n_g} open LAND cell(s) remain as ground reference (sea, coast and water cells "
                f"are excluded; >= {GROUND_MIN_REFS} needed). The surface model (DSM) is used as is: building bases "
                "and terrain may sit on canopy/roofs by several metres.")
            return
        if masked.any():
            zlow = float(raw[ground].min())
            gx, gy, gz = cx[ground], cy[ground], raw[ground]
            thr2 = (GROUND_MAX_REF_DIST_CELLS * cell) ** 2
            todo = np.flatnonzero(masked)
            step = max(1, 4_000_000 // len(gx))   # masked cells per distance-matrix block (bounded memory)
            for c0 in range(0, len(todo), step):
                blk = todo[c0:c0 + step]
                d2 = (gx[None, :] - cx[blk, None]) ** 2 + (gy[None, :] - cy[blk, None]) ** 2
                near = d2.min(axis=1) <= thr2
                n_far += int((~near).sum())
                if not near.any():
                    continue
                d2 = d2[near]
                nearest = np.argsort(d2, axis=1)[:, :GROUND_NEIGHBOURS]
                for i, d2i, idx in zip(blk[near], d2, nearest):
                    est = _plane_at(gx, gy, gz, cx[i], cy[i], idx, d2i, cell)
                    new[i] = max(min(max(est, zlow), raw[i]), raw[i] - cap[i])
        low = raw - new
        n_capped = int((masked & (low >= cap - 1e-6) & (low > 0)).sum())
        gi.update({"applied": True, "cells_total": int(raw.size), "cells_masked": int(masked.sum()),
                   "cells_masked_in_boundary": int((masked & in_b).sum()),
                   "cells_in_boundary": int(in_b.sum()), "cells_ground": n_g,
                   "cells_water_kept": int((wet & ~ground).sum()),
                   "cells_sea_or_coast_excluded": int(((sea_c | low_coast) & ~masked).sum()),
                   "cells_no_ref_nearby": n_far, "cells_lowering_capped": n_capped,
                   "lowering_m": {"max": round(float(low.max()), 2) if low.size else 0.0,
                                  "mean_masked": round(float(low[masked].mean()), 2) if masked.any() else 0.0}})
        dem.a = new.reshape(dem.a.shape)
        if not masked.any():
            self.ground_warnings.append("ground_mode=bare_earth_approx: no DEM cell is covered by OSM buildings/"
                                        "vegetation, the surface model is used as is (unmapped objects stay in it).")
            return
        extra = ""
        if n_far:
            extra += f" {n_far} masked cell(s) had no land reference within {GROUND_MAX_REF_DIST_CELLS:g} cells and keep the DSM value."
        if n_capped:
            extra += f" {n_capped} cell(s) hit the lowering cap ({GROUND_MAX_LOWERING:g} m; buildings 2 x height)."
        self.ground_warnings.append(
            f"ground_mode=bare_earth_approx: the DEM is a surface model, so {int(masked.sum())} of {raw.size} DEM "
            f"cells (>= {GROUND_COVER_THRESHOLD:.0%} covered by OSM buildings+{GROUND_BUILDING_BUFFER:g} m, "
            f"wood/forest or tree crowns) were replaced by a local plane fit of the {n_g} open-land cells (water, sea and coast cells not used) "
            f"(lowered by up to {gi['lowering_m']['max']:.1f} m).{extra} This is an APPROXIMATION, not a terrain model: "
            "expect errors of several metres; unmapped buildings/vegetation and bridges are not removed. "
            "Use a real DTM (dem_source bw_dgm1 or file:<path>) where available.")

    # ---- terrain -------------------------------------------------------------------
    def _grid(self, s: float):
        ring = shapely.segmentize(self.bpoly.exterior, s)
        rxy = np.asarray(ring.coords)[:-1]
        minx, miny, maxx, maxy = self.bpoly.bounds
        gx = np.arange(math.ceil(minx / s) * s, maxx + 1e-9, s)
        gy = np.arange(math.ceil(miny / s) * s, maxy + 1e-9, s)
        X, Y = np.meshgrid(gx, gy)
        X, Y = X.ravel(), Y.ravel()
        inside = shapely.contains_xy(self.bpoly, X, Y)
        X, Y = X[inside], Y[inside]
        if len(X):
            far = shapely.distance(self.bpoly.exterior, shapely.points(X, Y)) > 0.3 * s
            X, Y = X[far], Y[far]
        return rxy, np.column_stack([X, Y])

    def terrain(self) -> dict:
        user = self.s.get("grid_spacing")
        if user:
            s = float(user)
            rxy, ixy = self._grid(s)
            if len(rxy) + len(ixy) > MAX_TERRAIN_POINTS:
                self.warnings.append(f"grid_spacing {s} m gives {len(rxy) + len(ixy)} terrain points (> {MAX_TERRAIN_POINTS})")
        else:
            s = max(5.0, float(self.s.get("terrain_min_spacing") or 0))
            rxy, ixy = self._grid(s)
            while len(rxy) + len(ixy) > MAX_TERRAIN_POINTS:
                s = round(s * 1.15, 2)
                rxy, ixy = self._grid(s)
        self.spacing = s
        self.ixy = ixy
        zr = self.site.z_abs(rxy[:, 0], rxy[:, 1])
        zi = self.site.z_abs(ixy[:, 0], ixy[:, 1]) if len(ixy) else np.zeros(0)
        # origin elevation: min of the bilinear LAND surface inside the boundary (samples + DEM nodes outside water,
        # so negative DEM artefacts in rivers/sea do not drag it below the water) and of the water surfaces
        cx, cy = self._cell_xy()
        val = self.dem.a.ravel()
        node_in = self._cells_in_boundary()
        self._prepare_water()
        wu = self._water_union_b

        def land(x, y):
            if wu is None or len(x) == 0:
                return np.ones(len(x), bool)
            return ~shapely.intersects_xy(wu, x, y)
        lr, li, ln = land(rxy[:, 0], rxy[:, 1]), land(ixy[:, 0], ixy[:, 1]), land(cx, cy)
        land_c = [a[m].min() for a, m in ((zr, lr), (zi, li), (val, node_in & ln)) if len(a) and m.any()]
        wz = [w["z_abs"] for w in self._water_polys]
        if land_c:
            self.site.elev0 = float(min(land_c + wz))
        else:  # all water: fall back to every sample
            cands = [zr.min()] + ([zi.min()] if len(zi) else []) + ([val[node_in].min()] if node_in.any() else [])
            self.site.elev0 = float(min(cands + wz))
        e0 = self.site.elev0
        pts = np.vstack([np.column_stack([rxy, zr - e0]), np.column_stack([ixy, zi - e0])]) if len(ixy) \
            else np.column_stack([rxy, zr - e0])
        self.counts["terrain_points"] = len(pts)
        self.terrain_info = {"spacing_m": s, "n_boundary_points": len(rxy), "n_interior_points": len(ixy),
                             "points_order": "terrain.points = boundary ring (same order as terrain.boundary) then interior grid",
                             "z_abs_min": round(float(pts[:, 2].min() + e0), 3),
                             "z_abs_max": round(float(pts[:, 2].max() + e0), 3)}
        return {"points": [[round(x, 3), round(y, 3), round(z, 3)] for x, y, z in pts],
                "boundary": [[round(x, 3), round(y, 3)] for x, y in rxy]}

    def z_rel(self, x, y) -> np.ndarray:
        return self.site.z_abs(x, y) - self.site.elev0

    def _building_record(self, p: Polygon, oid: str, t: dict, h: float, src: str, extra: dict | None = None) -> dict:
        ext = np.asarray(shapely.segmentize(p.exterior, 5.0).coords)
        c = p.centroid
        sx = np.append(ext[:, 0], c.x)
        sy = np.append(ext[:, 1], c.y)
        tags = dict(t)
        if t.get("building") in ("roof", "ruins", "construction", "collapsed"):
            tags["osmctx:flag"] = t["building"]
            self.counts[f"buildings_flagged_{t['building']}"] += 1
        zs = self.z_rel(sx, sy)
        base, spread = float(zs.min()), float(zs.max() - zs.min())
        if spread > STEEP_FOOTPRINT_M:
            tags["osmctx:steep_footprint"] = f"{spread:.1f}"
            self.counts["buildings_steep_footprint"] += 1
        wz = self._in_water_z(p)
        if wz is not None:
            tags["osmctx:in_water"] = "yes"
            self.counts["buildings_in_water"] += 1
            if base < wz:
                base = wz
                self.counts["buildings_raised_to_water"] += 1
        rec = {"osm_id": oid, "footprint": ring_xy(p.exterior),
               "holes": [ring_xy(r) for r in p.interiors],
               "base_z": round(base, 3), "base_spread": round(spread, 2),
               "height": round(h, 3), "height_source": src, "tags": tags}
        if extra:
            rec.update(extra)
        return rec

    def _overture_buildings(self, out: list[dict]) -> None:
        """Append the conflated Overture buildings (inside the boundary) to `out`. Records carry
        source "overture:<dataset>", estimated (ML-derived footprint) and height_source overture_height |
        overture_floors | default."""
        ov = self._overture()
        if not ov["on"]:
            return
        n = 0
        ds = Counter()
        for q, props in ov["kept"]:
            polys, dropped = clean_polygons(q)
            self.counts["building_slivers_dropped"] += dropped
            for pp in polys:
                if not self.bpoly.covers(pp.centroid):
                    continue
                h, src = ovt.height_for(props, self.s["floor_height"], self.s["default_levels"])
                self.heights["overture_default" if src == "default" else src] += 1
                tags = {"building": "yes", "osmctx:source": ovt.source_tag(props)}
                if props.get("class"):
                    tags["osmctx:overture_class"] = str(props["class"])
                out.append(self._building_record(pp, f"overture/{props.get('id')}", tags, h, src, {
                    "source": ovt.source_tag(props), "estimated": ovt.is_estimated(props)}))
                ds[ovt.primary_dataset(props)] += 1
                n += 1
        self.counts["buildings_overture"] = n
        ov["inside"] = n
        ov["by_dataset"] = dict(ds)

    # ---- buildings -----------------------------------------------------------------
    def buildings(self) -> list[dict]:
        out = []
        for el in self.els:
            t = el.get("tags", {})
            if "building" not in t and "building:part" not in t:
                continue
            if "building:part" in t:
                if not self.s["include"].get("building_parts", True):
                    self.counts["building_parts_skipped"] += 1
                continue
            if t.get("building") == "no":
                continue
            if el["type"] == "node":
                self.counts["building_nodes_skipped"] += 1
                continue
            g, bad, why = self._area_local(el, t)
            if bad:
                self.warnings.append(f"{osm_id(el)}: {bad} unclosed ring piece(s) dropped")
            if why:
                self.counts["building_ways_unclosed_skipped" if why == "unclosed" else "building_relations_skipped"] += 1
                continue
            polys, dropped = clean_polygons(g)
            self.counts["building_slivers_dropped"] += dropped
            for p in polys:
                if not self.bpoly.covers(p.centroid):
                    self.counts["buildings_outside"] += 1
                    continue
                h, src = building_height(t, self.s)
                self.heights[src] += 1
                out.append(self._building_record(p, osm_id(el), t, h, src))
        self.counts["buildings_osm"] = len(out)
        self._overture_buildings(out)
        self.counts["buildings"] = len(out)
        if self.counts["buildings_steep_footprint"]:
            top = max(b["base_spread"] for b in out)
            self.warnings.append(
                f"{self.counts['buildings_steep_footprint']} building(s) stand on ground that varies by more than "
                f"{STEEP_FOOTPRINT_M:g} m under the footprint (max {top:.1f} m; tag osmctx:steep_footprint): base_z = "
                "lowest terrain point, so the uphill side is buried; check against a survey (cliffs/slopes in a ~30 m "
                "DEM are smoothed)")
        if self.counts["buildings_in_water"]:
            self.warnings.append(
                f"{self.counts['buildings_in_water']} building(s) stand >= 50 % in water (stilt/pile houses, piers; tag "
                f"osmctx:in_water=yes); {self.counts['buildings_raised_to_water']} base(s) raised to the water surface")
        return out

    def _in_water_z(self, p: Polygon) -> float | None:
        """Relative surface z of the water covering >= 50 % of `p` (largest overlap), else None."""
        wu = self._water_union_b
        if wu is None or not wu.intersects(p) or p.intersection(wu).area < 0.5 * p.area:
            return None
        cand = sorted(int(i) for i in self._water_tree.query(p, predicate="intersects")) or range(len(self._water_polys))
        best = max((self._water_polys[i] for i in cand), key=lambda w: w["poly"].intersection(p).area)
        return best["z_abs"] - self.site.elev0

    # ---- roads ---------------------------------------------------------------------
    def roads(self) -> list[dict]:
        recs = []
        for el in self.els:
            t = el.get("tags", {})
            if "highway" not in t or el["type"] != "way":
                continue
            cls = str(t["highway"])
            if t.get("area") == "yes":
                self.counts["roads_skipped_area"] += 1
                continue
            if cls in ROAD_SKIP_CLASSES:
                self.counts[f"roads_skipped_{cls}"] += 1
                continue
            nodes, ll = way_nodes_lonlat(el)
            if len(ll) < 2:
                continue
            x, y = self.f.to_local([p[0] for p in ll], [p[1] for p in ll])
            xy = np.column_stack([x, y])
            if cls in self.s["road_classes_exclude"]:
                if LineString(xy).intersects(self.bpoly):  # only ways that would have been imported
                    self.counts[f"roads_skipped_excluded_{cls}"] += 1
                    self.counts["roads_skipped_excluded"] += 1
                continue
            keys = [n if n is not None else f"{round(a, 2)},{round(b, 2)}" for n, (a, b) in zip(nodes, xy)]
            bridge = t.get("bridge", "no") not in ("no", "low_water_crossing")
            tunnel = t.get("tunnel", "no") != "no" and not bridge
            dens, oidx = densify(xy, ROAD_MAX_SEG)
            recs.append({"el": el, "t": t, "cls": cls, "nodes": keys, "dens": dens, "oidx": oidx,
                         "s": arclen(dens), "bridge": bridge, "tunnel": tunnel,
                         "layer": parse_layer(t.get("layer")),
                         "kind": "bridge" if bridge else "tunnel" if tunnel else None})
        if self.s["include"].get("railways", True):
            recs += features.rail_records(self)   # railways share the road drape + bridge/tunnel logic
        # node usage across all eligible highway ways
        use: Counter = Counter()
        for r in recs:
            for n in set(r["nodes"]):
                use[n] += 1
        node_z: dict = {}
        xing = _Crossings(recs, self.s)
        self._drape(recs, use, node_z)          # provisional z of ordinary roads (lower-layer references)
        self._bridge_runs(recs, use, node_z, xing)    # tunnels + bridges: gradelines, layer ordering, ramps
        self._drape(recs, use, node_z)          # final z of ordinary roads, pinned to bridge/tunnel ends
        self._check_crossings(recs, xing)
        out = []
        self.rail_ways = []
        e0 = self.site.elev0
        for r in recs:
            pts3 = np.column_stack([r["dens"], r["z"] - e0])
            pieces = [p for p in clip_polyline_z(pts3, self.bpoly) if arclen(p)[-1] >= 0.5]
            if not pieces:
                self.counts["rail_ways_outside" if r.get("rail") else "road_ways_outside"] += 1
                continue
            if r.get("rail"):
                features.rail_out(self, r, pieces, self.rail_ways)
                continue
            w, wsrc, wdetail = road_width(r["t"], self.s)
            self.widths[wdetail] += 1
            self.counts["road_ways"] += 1
            if r["bridge"]:
                self.counts["bridge_ways"] += 1
            if r["tunnel"]:
                self.counts["tunnel_ways"] += 1
                self.warnings.append(f"{osm_id(r['el'])} is a tunnel ({r['t'].get('tunnel')}): kept with tunnel=true, "
                                     "z interpolated between portals (not draped); hide or delete if unwanted")
            out.extend({"osm_id": osm_id(r["el"]), "class": r["cls"], "width": round(w, 3),
                        "width_source": wsrc, "bridge": r["bridge"], "tunnel": r["tunnel"],
                        "layer": r["layer"],
                        "points": [[round(a, 3), round(b, 3), round(c, 3)] for a, b, c in p]} for p in pieces)
        self.counts["roads"] = len(out)
        return out

    def _drape(self, recs, use, node_z):
        """Ordinary roads: DEM z smoothed along the way, pinned to raw terrain (or bridge/tunnel end z) at ends and
        junctions."""
        for r in recs:
            if r["kind"]:
                continue
            if "_zs" not in r:   # DEM z and its smoothing do not depend on the pins: shared by both drape passes
                n_out = self.dem.n_outside
                r["_zraw"] = self.site.z_abs(r["dens"][:, 0], r["dens"][:, 1])
                r["_zs"] = smooth_along(r["s"], r["_zraw"], ROAD_SMOOTH_HALFWIDTH)
                r["_n_out"] = self.dem.n_outside - n_out
            else:
                self.dem.n_outside += r["_n_out"]   # keep report.warnings' "samples outside the DEM" count as before
            zraw, zs = r["_zraw"], r["_zs"]
            pin_s, pin_d = [], []
            last = len(r["nodes"]) - 1
            for k, (n, i) in enumerate(zip(r["nodes"], r["oidx"])):
                if k in (0, last) or use[n] >= 2 or n in node_z:
                    pin_s.append(r["s"][i])
                    pin_d.append(node_z.get(n, zraw[i]) - zs[i])
            order = np.argsort(pin_s)
            r["z"] = zs + np.interp(r["s"], np.asarray(pin_s)[order], np.asarray(pin_d)[order])

    def _end_z(self, x: float, y: float) -> tuple[float, str]:
        """Ground z (abs) at a bridge abutment / tunnel portal: main DEM, else dem_ext.tif (GLO-30 around chain ends
        beyond the DEM, written by fetch), else the DEM clamped to its edge."""
        lon, lat = self.f.to_lonlat([x], [y])
        if self.dem.contains(lon, lat).all():
            return float(self.dem.sample(lon, lat)[0]), "dem"
        if self.dem_ext is not None and self.dem_ext.contains(lon, lat).all():
            return float(self.dem_ext.sample(lon, lat)[0]), "dem_ext"
        return float(self.dem.sample(lon, lat)[0]), "clamped"

    def _chains(self, recs):
        """Contiguous bridge (or tunnel) ways joined end to end -> [(kind, [(rec index, reversed)], parts, Ltot)]."""
        endmap = defaultdict(list)
        idx = [i for i, r in enumerate(recs) if r["kind"]]
        for i in idx:
            r = recs[i]
            for n in {r["nodes"][0], r["nodes"][-1]}:
                endmap[(r["kind"], n)].append(i)

        def nb(i, n):
            lst = endmap.get((recs[i]["kind"], n), [])
            if len(lst) == 2 and i in lst:
                j = lst[0] if lst[1] == i else lst[1]
                return None if j == i else j
            return None

        def near(i, rev):
            return recs[i]["nodes"][-1] if rev else recs[i]["nodes"][0]

        def far(i, rev):
            return recs[i]["nodes"][0] if rev else recs[i]["nodes"][-1]

        used, out = set(), []
        for i0 in idx:
            if i0 in used:
                continue
            cur, rev, seen = i0, False, {i0}
            while True:  # walk back to the free start of the chain
                n = near(cur, rev)
                j = nb(cur, n)
                if j is None or j in seen:
                    break
                cur, rev = j, recs[j]["nodes"][0] == n
                seen.add(j)
            chain = [(cur, rev)]
            used.add(cur)
            while True:
                n = far(*chain[-1])
                j = nb(chain[-1][0], n)
                if j is None or j in used:
                    break
                chain.append((j, recs[j]["nodes"][-1] == n))
                used.add(j)
            off, parts = 0.0, []
            for i, rv in chain:
                r = recs[i]
                L = r["s"][-1]
                parts.append((i, (off + (L - r["s"])) if rv else (off + r["s"])))
                off += L
            out.append((recs[chain[0][0]]["kind"], chain, parts, off,
                        near(*chain[0]), far(*chain[-1])))
        return out

    def _bridge_runs(self, recs, use, node_z, xing):
        """Bridge/tunnel chains, never draped on the (surface) DEM:
        - ends: abutment/portal ground z (DEM, dem_ext.tif beyond it); an end cut by the data (not connected to any
          road) gets DEM + layer x BRIDGE_CLEARANCE_PER_LAYER (estimate, warned);
        - base gradeline = straight line between the two ends;
        - bridges are processed by layer (lowest first); at every crossing with a LOWER-layer road the deck must be
          >= that road + BRIDGE_MIN_CLEARANCE, over area water >= water + BRIDGE_WATER_CLEARANCE. Each requirement is
          a flat-topped tent falling at the ramp cap (RAMP_CAP, slow classes RAMP_CAP_SLOW) towards both ends; a side
          that cannot reach a connected end at the cap uses the grade needed (reported in report.info.layers)."""
        runs = self._chains(recs)
        area_water = [w for w in self._water_polys if w.get("area")]
        order = sorted(range(len(runs)), key=lambda k: (runs[k][0] != "tunnel",
                                                         min(recs[i]["layer"] for i, _ in runs[k][1])))
        static, water_reqs = {}, {}
        ord_nodes = {n for r in recs if not r["kind"] for n in r["nodes"]}
        for k in order:  # ends (ground z at abutments / portals), ramp cap - once per run
            kind, chain, parts, Ltot, n_start, n_end = runs[k]
            first, frev = chain[0]
            last, lrev = chain[-1]
            pa = recs[first]["dens"][-1 if frev else 0]
            pb = recs[last]["dens"][0 if lrev else -1]
            ids = ", ".join(osm_id(recs[i]["el"]) for i, _ in chain)
            ends = []
            for node, p, i_end in ((n_start, pa, first), (n_end, pb, last)):
                z, src = self._end_z(float(p[0]), float(p[1]))
                lay = max(recs[i_end]["layer"], 1) if kind == "bridge" else 0
                # ground: an ordinary road meets the end (abutment / portal) -> pinned to the DEM;
                # chain: only other bridge/tunnel ways meet here (elevated branch) -> z of the run that set it first;
                # cut: nothing meets it (data cut at the fetch area) -> estimated ground + layer clearance, free
                conn = "ground" if node in ord_nodes else "chain" if use[node] >= 2 else "cut"
                if src == "clamped":
                    self.counts[f"{kind}_ends_clamped"] += 1
                    self.warnings.append(f"{kind} run [{ids}]: an end lies outside the DEM (and dem_ext); its z is "
                                         "clamped to the DEM edge")
                if conn == "cut":
                    self.counts[f"{kind}_ends_cut"] += 1
                    self.warnings.append(f"{kind} run [{ids}]: an end is not connected to any fetched road (data cut "
                                         f"at the fetch area); its z is estimated as ground + {lay * BRIDGE_CLEARANCE_PER_LAYER:g} m")
                ends.append((node, z, conn, z + (lay * BRIDGE_CLEARANCE_PER_LAYER if kind == "bridge" else 0.0)))
            cls = recs[first]["cls"]
            cap = 10.0 if cls == "steps" else RAMP_CAP_SLOW if cls in RAMP_SLOW_CLASSES else RAMP_CAP
            static[k] = (ends, cap, ids)
            wreq = []   # deck >= area water + clearance wherever a bridge vertex lies in the water (static: no z involved)
            if kind == "bridge":
                for i, sc in parts:
                    xy = recs[i]["dens"]
                    for w in area_water:
                        wreq.extend((float(sc[q]), w["z_abs"] + BRIDGE_WATER_CLEARANCE, ROAD_MAX_SEG / 2)
                                    for q in np.flatnonzero(shapely.contains_xy(w["poly"], xy[:, 0], xy[:, 1])))
            water_reqs[k] = wreq
        # requirements depend on the z of lower-layer ways, which may themselves be raised later in the same pass
        # (chains mixing layers) -> relax until stable (z only ever rises)
        chain_nodes = {n for _, chain, *_ in runs for i, _ in chain for n in recs[i]["nodes"]}
        for _ in range(BRIDGE_PASSES):
            for n in chain_nodes:
                node_z.pop(n, None)
            e0_viol, n_raised, changed = [], 0, 0.0
            for k in order:
                kind, chain, parts, Ltot, n_start, n_end = runs[k]
                ends, cap, ids = static[k]
                ez = []
                for node, zg, conn, zest in ends:
                    if conn == "ground":
                        ez.append((zg, True))
                    elif conn == "chain" and node in node_z:
                        ez.append((node_z[node], True))
                    else:
                        ez.append((zest, False, conn))
                if len(ez[0]) == 3 and ez[0][2] == "chain" and len(ez[1]) == 2:
                    ez[0] = (max(ez[0][0], ez[1][0]), False)  # free branch end: level with the pinned end
                if len(ez[1]) == 3 and ez[1][2] == "chain" and len(ez[0]) == 2:
                    ez[1] = (max(ez[1][0], ez[0][0]), False)
                (za, ca), (zb, cb) = [e[:2] for e in ez]
                reqs = []  # (sc, z_req, plateau half-width) in chain arc length
                if kind == "bridge":
                    for i, sc in parts:
                        r = recs[i]
                        lay = r["layer"]
                        reqs.extend((float(np.interp(si, r["s"], sc)), zj + BRIDGE_MIN_CLEARANCE, ROAD_MAX_SEG / 2 + hw)
                                    for si, zj, hw in xing.of(i, lambda j, lay=lay: recs[j]["layer"] < lay))
                    reqs += water_reqs[k]
                viol = []

                def profile(sc):
                    z = za + (zb - za) * (sc / Ltot if Ltot > 0 else 0.0)
                    for s0, zr, pl in reqs:
                        gl, gr = cap, cap
                        if ca and s0 - pl > 1e-3:
                            gl = max(cap, (zr - za) / (s0 - pl))
                        if cb and Ltot - s0 - pl > 1e-3:
                            gr = max(cap, (zr - zb) / (Ltot - s0 - pl))
                        d = sc - s0
                        t = np.where(d < 0, zr - gl * np.maximum(0, -d - pl), zr - gr * np.maximum(0, d - pl))
                        z = np.maximum(z, t)
                    return z
                for s0, zr, pl in reqs:
                    if ca and s0 - pl > 1e-3 and (zr - za) / (s0 - pl) > cap + 1e-6:
                        viol.append((zr - za) / (s0 - pl))
                    if cb and Ltot - s0 - pl > 1e-3 and (zr - zb) / (Ltot - s0 - pl) > cap + 1e-6:
                        viol.append((zr - zb) / (Ltot - s0 - pl))
                for i, sc in parts:
                    r = recs[i]
                    zl = profile(sc)
                    if r.get("z") is not None and len(r["z"]) == len(zl):
                        changed = max(changed, float(np.abs(zl - r["z"]).max()))
                    else:
                        changed = max(changed, 1.0)
                    r["z"] = zl
                    for n, kk in zip(r["nodes"], r["oidx"]):
                        node_z.setdefault(n, float(zl[kk]))
                if viol:
                    e0_viol.append((ids, max(viol), cap))
                n_raised += bool(reqs)
            if changed < 0.01:
                break
        for k in order:
            self.counts[f"{runs[k][0]}_runs"] += 1
        if n_raised:
            self.counts["bridge_runs_raised"] = n_raised
        if e0_viol:
            self.counts["ramp_cap_exceeded_runs"] = len(e0_viol)
        self._ramp_viol = e0_viol
        for ids, g, cap in e0_viol[:10]:
            self.warnings.append(f"bridge run [{ids}]: clearance over lower layers / water needs a {g:.1%} ramp from an "
                                 f"abutment (cap {cap:.0%}); clearance kept, ramp steeper than the cap")
        if len(e0_viol) > 10:
            self.warnings.append(f"... {len(e0_viol) - 10} more bridge runs exceed the ramp cap")

    def _check_crossings(self, recs, xing):
        """Every crossing of two highway ways with different layer inside the boundary: the higher layer must be
        >= BRIDGE_MIN_CLEARANCE above the lower. Result in report.info.layers + warnings."""
        lines = xing.lines
        n = bad = inv = 0
        worst = []
        for i, r in enumerate(recs):
            for si, zj, _hw in xing.of(i, lambda j: recs[j]["layer"] < r["layer"]):
                p = lines[i].interpolate(si)
                if not self.bpoly.covers(p):
                    continue
                zi = float(np.interp(si, r["s"], r["z"]))
                dz = zi - zj
                n += 1
                if dz < 0:
                    inv += 1
                if dz < BRIDGE_MIN_CLEARANCE - 0.05:
                    bad += 1
                    worst.append((round(dz, 2), osm_id(r["el"]), r["layer"]))
        worst.sort()
        self.layer_info = {"crossings": n, "below_clearance": bad, "inverted": inv,
                           "min_clearance_m": BRIDGE_MIN_CLEARANCE, "ramp_cap": RAMP_CAP,
                           "ramp_cap_exceeded_runs": len(self._ramp_viol),
                           "worst": [{"dz": d, "upper": o, "layer": l} for d, o, l in worst[:5]]}
        if inv:
            self.warnings.append(f"{inv} of {n} crossings of different-layer roads have the HIGHER layer below the "
                                 "lower one (check layer tags / non-bridge roads with layer)")
        elif bad:
            self.warnings.append(f"{bad} of {n} different-layer crossings have < {BRIDGE_MIN_CLEARANCE:g} m clearance "
                                 "(upper road is not a bridge or ramp limits)")

    # ---- trees ---------------------------------------------------------------------
    def _ndom_cells(self):
        """(cx, cy, values) of nDOM cell centres in local site metres (cells inside building footprints +0.5 m
        dropped: a tree next to a wall must not measure the wall). None when the cache has no ndom.tif."""
        p = self.cache / "ndom.tif"
        if not p.exists():
            self.ndom_info = {"available": False, "reason": (self.meta.get("ndom") or {}).get(
                "reason", "no ndom.tif in cache (no fine <=2 m surface model for this source/region; GLO-30 at 30 m is too coarse for single trees)")}
            return None
        import rasterio
        with rasterio.open(p) as ds:
            a = ds.read(1).astype("float64")
            t, crs, nd = ds.transform, ds.crs, ds.nodata
        if t.b != 0 or t.d != 0:
            self.ndom_info = {"available": False, "reason": "rotated ndom.tif not supported"}
            return None
        a = np.where(np.isfinite(a) & ((a != nd) if nd is not None else True), a, np.nan)
        cols, rows = np.meshgrid(np.arange(a.shape[1]), np.arange(a.shape[0]))
        X = (t.c + (cols.ravel() + 0.5) * t.a)
        Y = (t.f + (rows.ravel() + 0.5) * t.e)
        if crs is not None and crs.to_epsg() != self.f.epsg:
            X, Y = Transformer.from_crs(crs, f"EPSG:{self.f.epsg}", always_xy=True).transform(X, Y)
        cx, cy, v = np.asarray(X) - self.f.E0, np.asarray(Y) - self.f.N0, a.ravel()
        bld = self._ground_mask_features()[0]
        if bld:
            foot = unary_union([g.buffer(-(GROUND_BUILDING_BUFFER - 0.5)) for g in bld])
            if not foot.is_empty:
                v = np.where(shapely.contains_xy(foot, cx, cy), np.nan, v)
        self.ndom_info = {"available": True, "source": (self.meta.get("ndom") or {}).get("source", "nDOM"),
                          "cells": int(np.isfinite(v).sum()), "radius_m": treelib.MEASURE_RADIUS_M,
                          "accepted_m": list(treelib.MEASURE_ACCEPT_M)}
        return cx, cy, v

    def _cap_crowns(self, trees: list[dict]) -> None:
        """Ratio-derived crowns overlap in avenues (e.g. 35 m planes -> 23 m crowns at 10 m spacing).
        Cap them at 1.15 x the distance to the nearest other tree (min 2 m). Tagged crowns are kept."""
        ratio = [i for i, t in enumerate(trees) if t.get("crown_source") == "ratio"]
        if not ratio or len(trees) < 2:
            return
        pts = shapely.points(np.array([(t["x"], t["y"]) for t in trees], float))
        tree = shapely.STRtree(pts)
        # nearest OTHER tree: a coincident duplicate counts (distance 0); otherwise exclusive=True skips the tree itself
        (inp, _), dist = tree.query_nearest(pts[ratio], return_distance=True, all_matches=True)
        dup = np.bincount(inp[dist == 0], minlength=len(ratio)) >= 2
        nn = np.zeros(len(ratio))
        rest = np.flatnonzero(~dup)
        if len(rest):
            (inp, _), dist = tree.query_nearest(pts[np.asarray(ratio)[rest]], return_distance=True, exclusive=True)
            nn[rest[inp]] = dist
        for i, nni in zip(ratio, nn):
            t = trees[i]
            cap = max(2.0, 1.15 * float(nni))
            if t["crown_diameter"] > cap:
                t["crown_diameter"] = round(cap, 2)
                t["crown_source"] = "ratio_capped"
                self.tree_crowns["ratio"] -= 1
                self.tree_crowns["ratio_capped"] += 1

    def trees(self) -> list[dict]:
        out, node_xy = [], []
        tree_nodes = [el for el in self.els if el["type"] == "node" and el.get("tags", {}).get("natural") == "tree"]
        if tree_nodes:   # one batched projection + covers() instead of per-node pyproj / shapely calls
            nx, ny = self.f.to_local([el["lon"] for el in tree_nodes], [el["lat"] for el in tree_nodes])
            nx, ny = np.asarray(nx, float), np.asarray(ny, float)
            node_xy = list(zip(nx.tolist(), ny.tolist()))
            inside = shapely.covers(self.bpoly, shapely.points(nx, ny))
            out = [(osm_id(el), x, y, el.get("tags", {}), "node")
                   for el, (x, y), ok in zip(tree_nodes, node_xy, inside) if ok]
        nxy = np.asarray(node_xy, float).reshape(-1, 2)
        for el in self.els:
            t = el.get("tags", {})
            if el["type"] == "way" and t.get("natural") == "tree_row":
                ll = way_lonlat(el.get("geometry"))
                if len(ll) < 2:
                    continue
                ln = self.f.geom_to_local(LineString(ll))
                L = ln.length
                ds = np.arange(0.0, L + 1e-9, TREE_ROW_SPACING)
                if L - ds[-1] >= TREE_ROW_SPACING / 2:
                    ds = np.append(ds, L)
                pts = shapely.line_interpolate_point(ln, ds)
                pts = pts[shapely.covers(self.bpoly, pts)]
                px, py = shapely.get_x(pts), shapely.get_y(pts)
                for x, y, dup in zip(px.tolist(), py.tolist(), _near_any(px, py, nxy, 2.0).tolist()):
                    if dup:
                        self.counts["tree_row_points_dup_skipped"] += 1
                        continue
                    out.append((osm_id(el), x, y, t, "row"))
            if (t.get("natural") == "wood" or t.get("landuse") == "forest") and not self.s.get("forest_scatter"):
                b = el.get("bounds")
                if b:
                    x0, y0 = self.f.to_local(b["minlon"], b["minlat"])
                    x1, y1 = self.f.to_local(b["maxlon"], b["maxlat"])
                    if shapely.box(float(x0), float(y0), float(x1), float(y1)).intersects(self.bpoly):
                        self.counts["wood_forest_areas_not_scattered"] += 1
        if self.s.get("forest_scatter"):
            out += veg.forest_points(self, node_xy)
        out += veg.area_tree_points(self, node_xy + [(o[1], o[2]) for o in out if o[4] == "forest"])
        self.tree_xy = [(o[1], o[2]) for o in out]
        res = []
        area_settings: dict = {}   # one settings copy per distinct area default height
        if out:
            zs = self.z_rel([o[1] for o in out], [o[2] for o in out])
            ndom = veg.ndom_index(self)
            for o, z in zip(out, zs):
                oid, x, y, t, src = o[:5]
                meas = ndom.measure(x, y, treelib.MEASURE_RADIUS_M, *treelib.MEASURE_ACCEPT_M) if ndom is not None else None
                th = parse_length(t.get("height"), max_value=150.0)
                tc = parse_length(t.get("diameter_crown"), max_value=60.0)
                area_h = (o[6] or {}).get("height") if len(o) > 6 else None
                if area_h:   # estimated area tree: the area type's default height replaces the global defaults
                    if area_h not in area_settings:
                        area_settings[area_h] = {**self.s, "tree_height": area_h, "broadleaf_height": area_h,
                                                 "conifer_height": area_h}
                    sz = treelib.size_tree(t, th, tc, meas, area_settings[area_h])
                    if sz["height_source"] in ("leaf_type", "default"):
                        sz["height_source"] = "area_default"
                else:
                    sz = treelib.size_tree(t, th, tc, meas, self.s)
                self.tree_heights[sz["height_source"]] += 1
                self.tree_crowns[sz["crown_source"]] += 1
                self.counts[{"node": "trees_from_nodes", "row": "trees_from_rows"}.get(src, f"trees_from_{src}")] += 1
                tree = {"osm_id": oid, "x": round(x, 3), "y": round(y, 3), "z": round(float(z), 3),
                        "tags": {k: str(t[k]) for k in ("leaf_type", "leaf_cycle", "genus", "species", "denotation")
                                 if t.get(k)},
                        **sz}
                if len(o) > 5:
                    tree["source_area"] = o[5]
                    tree["source_area_type"] = src
                    tree["estimated"] = True
                circ = parse_length(t.get("circumference"), max_value=15.0)
                if circ is not None:
                    tree["trunk_diameter"] = treelib.trunk_diameter(circ)
                res.append(tree)
        self.counts["trees"] = len(res)
        self.counts["trees_scattered_estimated"] = sum(1 for t in res if t.get("estimated"))
        if self.counts["wood_forest_areas_not_scattered"]:
            self.warnings.append(f"{self.counts['wood_forest_areas_not_scattered']} wood/forest area(s) touch the site; "
                                 "trees are NOT scattered in v1")
        return res

    # ---- water ---------------------------------------------------------------------
    def _water_z_abs(self, p: Polygon) -> tuple[float, str]:
        """Water surface (abs) of a polygon. On a DSM (GLO-30) the water body is flattened to one value: when >= 3 DEM
        cell centres lie inside and one value covers >= 50 % of them, that value is used ('dem_flat_water');
        otherwise the 5th percentile of DEM samples inside + on the ring ('dem_p5')."""
        shapely.prepare(p)
        if self.dem_info["dem_accuracy"].get("kind") == "DSM":
            cx, cy = self._cell_xy()
            x0, y0, x1, y1 = p.bounds   # only cells inside the polygon's bbox go through the point-in-polygon test
            near = np.flatnonzero((cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1))
            sel = near[shapely.contains_xy(p, cx[near], cy[near])]
            if len(sel) >= 3:
                v = np.round(self.dem.a.ravel()[sel], 2)
                vals, cnt = np.unique(v, return_counts=True)
                if cnt.max() >= 0.5 * len(sel):
                    return float(vals[cnt.argmax()]), "dem_flat_water"
        ixy = self.ixy
        xs, ys = [], []
        if len(ixy):
            m = shapely.contains_xy(p, ixy[:, 0], ixy[:, 1])
            xs.extend(ixy[m, 0])
            ys.extend(ixy[m, 1])
        ring = np.asarray(shapely.segmentize(p.exterior, max(self.spacing, 1.0)).coords)
        xs.extend(ring[:, 0])
        ys.extend(ring[:, 1])
        rp = p.representative_point()
        xs.append(rp.x)
        ys.append(rp.y)
        return float(np.percentile(self.site.z_abs(np.asarray(xs), np.asarray(ys)), 5)), "dem_p5"

    def _cell_xy(self):
        """DEM cell centres in local metres. Depend on the DEM grid only (never on the z values), so despike /
        bare_earth, which replace dem.a, cannot make this stale."""
        if self._cxy is None:
            lon, lat, _ = self.dem.cell_centres()
            cx, cy = self.f.to_local(lon, lat)
            self._cxy = (np.asarray(cx, float), np.asarray(cy, float))
        return self._cxy

    def _cells_in_boundary(self) -> np.ndarray:
        """Mask of DEM cell centres inside the boundary (grid geometry only, memoised)."""
        if self._cell_in_b is None:
            self._cell_in_b = shapely.contains_xy(self.bpoly, *self._cell_xy())
        return self._cell_in_b

    def _dem_extent_local(self) -> Polygon:
        cx, cy = self._cell_xy()
        cw, ch = self.dem.cell_size_m(self.f.lat0)
        box = shapely.box(cx.min() - cw, cy.min() - ch, cx.max() + cw, cy.max() + ch)
        return box.union(self.bpoly.buffer(1.0)).envelope

    def _water_geoms(self) -> dict:
        """All water areas as full local geometries (computed once): natural=water / riverbank areas, buffered
        waterways (minus areas and sea) and the sea from natural=coastline over the DEM extent.
        Returns {"items": [(osm_id, kind, geom, sea_linked)], "sea_full": sea union over the DEM extent or None,
        "has_coast": bool}."""
        if self._wg is not None:
            return self._wg
        areas = []  # (el, kind, full local geom)
        lines = []
        coast = []
        for el in self.els:
            t = el.get("tags", {})
            nat, ww = t.get("natural"), t.get("waterway")
            if nat == "water" or ww == "riverbank":
                kind = "river" if (ww == "riverbank" or t.get("water") in RIVER_WATER_VALUES) else "water"
                g, bad, why = self._area_local(el, t)
                if bad:
                    self.warnings.append(f"{osm_id(el)}: {bad} unclosed water ring piece(s) dropped")
                if why is None:
                    areas.append((el, kind, g))
                elif why == "unclosed":
                    self.counts["water_ways_unclosed_skipped"] += 1
                elif why == "other":
                    self.counts["water_other_skipped"] += 1
            elif nat == "coastline" and el["type"] == "way":
                ll = way_lonlat(el.get("geometry"))
                if len(ll) >= 2:
                    coast.append(self.f.geom_to_local(LineString(ll)))
            elif ww and el["type"] == "way":
                if ww not in WATERWAY_WIDTHS:
                    self.counts[f"waterway_{ww}_skipped"] += 1
                elif t.get("tunnel", "no") != "no":
                    self.counts["waterway_culverts_skipped"] += 1
                else:
                    ll = way_lonlat(el.get("geometry"))
                    if len(ll) >= 2:
                        lines.append((el, self.f.geom_to_local(LineString(ll))))
            elif ww and el["type"] == "relation":
                self.counts["waterway_relations_skipped"] += 1
        sea_full = None
        sea_items = []
        if coast:
            sea_items = self._sea_faces(coast, self._dem_extent_local())
            if sea_items:
                sea_full = make_valid(unary_union([g for _, _, g in sea_items]))
        area_union = unary_union([make_valid(g) for _, _, g in areas]) if areas else None
        area_buf = area_union.buffer(1.0) if area_union is not None else None   # once, not per waterway
        geoms = [(osm_id(el), kind, g) for el, kind, g in areas]
        self._area_water_ids = {oid for oid, _, _ in geoms}
        for el, ln in lines:
            t = el.get("tags", {})
            if area_buf is not None and ln.length > 0:
                cov = ln.intersection(area_buf).length / ln.length
                if cov >= 0.5:
                    self.counts["waterways_covered_by_area_skipped"] += 1
                    continue
            w = parse_length(t.get("width"), max_value=5000.0)
            wsrc = "tag" if w else "default"
            w = w or WATERWAY_WIDTHS[t["waterway"]]
            g = ln.buffer(w / 2, cap_style="flat", join_style="round")
            if area_union is not None:
                g = g.difference(area_union)
            if g.intersects(self.bpoly):
                self.counts["waterways_buffered"] += 1
                self.waterway_widths[wsrc] += 1
            else:
                self.counts["waterways_outside"] += 1
            geoms.append((osm_id(el), "river", g))
        items = []
        n_clip = 0
        for oid, kind, g in geoms:
            g = make_valid(g)
            linked = False
            if sea_full is not None:
                if g.intersects(sea_full):
                    n_clip += 1
                    g = make_valid(g.difference(sea_full))  # estuary / riverbank drawn into the sea: sea wins
                linked = (not g.is_empty) and g.distance(sea_full) <= SEA_LINK_DIST
            items.append((oid, kind, g, linked))
        items += [(oid, kind, g, True) for oid, kind, g in sea_items]
        if n_clip:
            self.counts["water_clipped_by_sea"] = n_clip
        self._wg = {"items": items, "sea_full": sea_full, "has_coast": bool(coast)}
        return self._wg

    def _sea_faces(self, coast: list[LineString], area: Polygon) -> list:
        """Sea faces of `area` split by natural=coastline (OSM: land on the LEFT, water on the RIGHT of the way
        direction). A face that shares coastline with the split is classified by a point just right of the shared
        piece (majority vote over the pieces); a face without shared coastline (coast does not cross `area`) by the
        side of the nearest coastline point."""
        ab = area.buffer(1.0)
        clip = [c for c in coast if c.intersects(ab)]
        faces = list(polygonize(unary_union([c.intersection(ab) for c in clip] + [area.exterior]))) if clip else []
        faces = [f for f in faces if area.covers(f.representative_point())] or [area]
        res = []
        for f in faces:
            if f.area <= MIN_AREA:
                continue
            votes = 0
            fb = f.buffer(1e-6)
            for c in clip:
                shared = lines_of(f.exterior.intersection(c))
                for piece in shared:
                    if piece.length < 0.2:
                        continue
                    mid = piece.interpolate(0.5, normalized=True)
                    d = c.project(mid)
                    a = c.interpolate(max(d - 0.25, 0))
                    b = c.interpolate(min(d + 0.25, c.length))
                    dx, dy = b.x - a.x, b.y - a.y
                    n = math.hypot(dx, dy) or 1.0
                    q = Point(mid.x + 0.2 * dy / n, mid.y - 0.2 * dx / n)  # right-hand side of the way
                    votes += 1 if fb.contains(q) else -1
            if votes == 0:  # coast does not touch this face: side of the nearest coastline point
                p = f.representative_point()
                ln = min(coast, key=lambda c: c.distance(p))
                d = ln.project(p)
                a = ln.interpolate(max(d - 0.5, 0))
                b = ln.interpolate(min(d + 0.5, ln.length))
                votes = -1 if (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x) > 0 else 1
            if votes > 0:
                res.append(("natural=coastline", "coastline", f))
        return res

    def _prepare_water(self):
        """Water polygons clipped to the boundary with absolute surface z (needs self.ixy / self.spacing).
        Sea = SEA_LEVEL; sea-linked water >= SEA_LEVEL."""
        wg = self._water_geoms()
        out = []
        for oid, kind, g, linked in wg["items"]:
            polys, _ = clean_polygons(make_valid(g).intersection(self.bpoly))
            for p in polys:
                if kind == "coastline":
                    z, src = SEA_LEVEL, "sea_level"
                else:
                    z, src = self._water_z_abs(p)
                    if linked and z < SEA_LEVEL:
                        z, src = SEA_LEVEL, src + "+sea_level_min"
                out.append({"osm_id": oid, "kind": kind, "poly": p, "z_abs": z, "z_source": src, "sea_linked": linked,
                            "area": kind == "coastline" or oid in self._area_water_ids})
        self._water_polys = out
        self._water_union_b = unary_union([w["poly"] for w in out]) if out else None
        self._water_tree = shapely.STRtree([w["poly"] for w in out]) if out else None
        if self._water_union_b is not None:
            shapely.prepare(self._water_union_b)

    def water(self) -> list[dict]:
        e0 = self.site.elev0
        out = []
        datum = self.dem_info.get("dem_vertical_datum")
        for w in self._water_polys:
            p = w["poly"]
            rec = {"osm_id": w["osm_id"], "kind": w["kind"], "polygon": ring_xy(p.exterior),
                   "holes": [ring_xy(r) for r in p.interiors], "z": round(w["z_abs"] - e0, 3),
                   "z_source": w["z_source"]}
            if w["sea_linked"]:
                rec["sea_linked"] = True
            if w["kind"] == "coastline":
                rec["vertical_datum"] = datum
            out.append(rec)
        self.counts["water"] = len(out)
        if any(w["kind"] == "coastline" for w in self._water_polys):
            self.warnings.append(f"sea area derived from natural=coastline (land left / water right of the way), "
                                 f"surface z = {SEA_LEVEL:g} m abs ({datum or 'DEM datum'}); verify the side in the preview")
        return out

    # ---- land cover / points of interest / cliffs (data for the builder, M4) ----------
    def land_cover(self) -> list[dict]:
        """natural=beach|sand|wetland (+ wetland=tidalflat|saltmarsh|mangrove) areas clipped to the boundary.
        They lie ON the terrain (no z of their own); kind = wetland subtype when given, else the natural value."""
        out = []
        for el in self.els:
            t = el.get("tags", {})
            nat, wet = t.get("natural"), t.get("wetland")
            green = next((k for (tk, tv), k in LAND_COVER_GREEN.items() if t.get(tk) == tv), None)
            if nat not in LAND_COVER_NATURAL and wet not in LAND_COVER_WETLAND and green is None:
                continue
            if nat in LAND_COVER_NATURAL or wet in LAND_COVER_WETLAND:
                kind = wet if (wet in LAND_COVER_WETLAND or (nat == "wetland" and wet)) else nat
            else:
                kind = green
            g, _, why = self._area_local(el, t)
            if why:
                if why == "unclosed":
                    self.counts["land_cover_unclosed_skipped"] += 1
                continue
            polys, _ = clean_polygons(make_valid(g).intersection(self.bpoly))
            for p in polys:
                out.append({"osm_id": osm_id(el), "kind": kind, "polygon": ring_xy(p.exterior),
                            "holes": [ring_xy(r) for r in p.interiors], "area_m2": round(p.area, 1),
                            "tags": {k: str(t[k]) for k in ("natural", "wetland", "landuse", "surface", "tidal", "name") if t.get(k)}})
                self.counts[f"land_cover_{kind}"] += 1
        self.counts["land_cover"] = len(out)
        return out

    def points_of_interest(self) -> list[dict]:
        """natural=peak|volcano nodes inside the boundary with OSM `ele` vs the DEM; never reshapes the DEM."""
        out = []
        tol = max(PEAK_TOLERANCE_M, 2.0 * float(self.dem_info["dem_accuracy"].get("vertical_m") or 0.0))
        for el in self.els:
            t = el.get("tags", {})
            if el["type"] != "node" or t.get("natural") not in ("peak", "volcano"):
                continue
            x, y = (float(v) for v in self.f.to_local(el["lon"], el["lat"]))
            if not self.bpoly.covers(Point(x, y)):
                continue
            zd = float(self.site.z_abs(x, y)[0])
            ele = parse_length(t.get("ele"), max_value=9000.0)
            rec = {"osm_id": osm_id(el), "kind": t["natural"], "name": t.get("name:en") or t.get("name"),
                   "x": round(x, 3), "y": round(y, 3), "z": round(zd - self.site.elev0, 3),
                   "dem_z_abs": round(zd, 2), "ele": ele, "ele_source": "osm" if ele is not None else None}
            if ele is not None:
                rec["dem_minus_ele"] = round(zd - ele, 2)
                if abs(zd - ele) > tol:
                    self.counts["poi_dem_mismatch"] += 1
                    self.warnings.append(
                        f"{rec['kind']} {rec['name'] or rec['osm_id']}: DEM summit {zd:.0f} m vs surveyed {ele:.0f} m "
                        f"(OSM ele; {zd - ele:+.0f} m). The terrain is NOT reshaped - the {rec['kind']} is exported "
                        "as data (points_of_interest) for the builder to mark")
            out.append(rec)
        self.counts["points_of_interest"] = len(out)
        return out

    def cliffs(self) -> list[dict]:
        """natural=cliff ways clipped to the boundary, z draped on the DEM (the DEM smooths the drop itself)."""
        out, total = [], 0.0
        for el in self.els:
            t = el.get("tags", {})
            if t.get("natural") != "cliff" or el["type"] != "way":
                continue
            ll = way_lonlat(el.get("geometry"))
            if len(ll) < 2:
                continue
            ln = self.f.geom_to_local(LineString(ll))
            dens, _ = densify(np.asarray(ln.coords), 5.0)
            pts3 = np.column_stack([dens, self.z_rel(dens[:, 0], dens[:, 1])])
            for p in clip_polyline_z(pts3, self.bpoly):
                if arclen(p)[-1] < 1.0:
                    continue
                total += arclen(p)[-1]
                out.append({"osm_id": osm_id(el), "points": [[round(a, 3), round(b, 3), round(c, 3)] for a, b, c in p],
                            "height": parse_length(t.get("height"), max_value=2000.0),
                            "tags": {k: str(t[k]) for k in ("name", "height", "natural") if t.get(k)}})
        self.counts["cliffs"] = len(out)
        if out:
            cw, ch = self.dem.cell_size_m(self.f.lat0)
            self.warnings.append(
                f"{len(out)} natural=cliff line(s) ({total:.0f} m) in the site (exported as `cliffs`): the ~{max(cw, ch):.0f} m "
                "DEM smooths a cliff into a ramp several cells wide - terrain near the edge can be off by tens of "
                "metres (land drawn at sea level, sea rising on the ramp), and buildings/paths near the edge use the "
                "smoothed surface. Use a survey or a fine DTM for anything near the edge")
        return out

    def slope_stats(self) -> dict:
        """DEM slope inside the boundary (+1 cell): max/p95 in %, share of cells steeper than STEEP_SLOPE_DEG and
        the vertical error a horizontal DEM error of one cell implies there."""
        cw, ch = self.dem.cell_size_m(self.f.lat0)
        gy, gx = np.gradient(self.dem.a, ch, cw) if min(self.dem.a.shape) >= 2 else (np.zeros(1), np.zeros(1))
        sl = np.hypot(gx, gy).ravel()
        cx, cy = self._cell_xy()
        m = shapely.intersects_xy(self.bpoly.buffer(max(cw, ch)), cx, cy)
        if sl.size != m.size or not m.any():
            return {}
        s = sl[m]
        p95 = float(np.percentile(s, 95))
        steep = float((s > math.tan(math.radians(STEEP_SLOPE_DEG))).mean())
        return {"max_pct": round(float(s.max()) * 100, 1), "p95_pct": round(p95 * 100, 1),
                "cells_steeper_than_deg": STEEP_SLOPE_DEG, "cells_steep_pct": round(steep * 100, 1),
                "vertical_error_one_cell_m_p95": round(p95 * max(cw, ch), 1)}

    # ---- run -----------------------------------------------------------------------
    def run(self) -> dict:
        inc = self.s["include"]
        if self.dem_info["dem_accuracy"].get("kind") == "DSM":
            self.despike()
        if self.ground_mode == "bare_earth_approx":
            self.bare_earth()
        terrain = self.terrain()  # also prepares water polygons + surfaces (origin uses land cells)
        if not inc.get("terrain", True):
            terrain = {"points": [], "boundary": []}
        water = self.water() if inc.get("water", True) else []
        buildings = self.buildings() if inc.get("buildings", True) else []
        roads = self.roads() if (inc.get("roads", True) or inc.get("railways", True)) else []
        if not inc.get("roads", True):
            roads = []
        rails = list(self.rail_ways)
        trees = self.trees() if inc.get("trees", True) else []
        if self.s.get("crown_neighbour_cap", True):
            self._cap_crowns(trees)
        shrubs, hedges = veg.shrubs_and_hedges(self, self.tree_xy) if inc.get("shrubs", True) else ([], [])
        land_cover = self.land_cover() if inc.get("land_cover", True) else []
        pois = self.points_of_interest()
        cliffs = self.cliffs()
        extra, buildings = features.run(self, buildings, roads, rails)
        origin = {"lat": round(self.f.lat0, 8), "lon": round(self.f.lon0, 8), "epsg": self.f.epsg,
                  "easting": round(self.f.E0, 3), "northing": round(self.f.N0, 3),
                  "elevation": round(self.site.elev0, 3)}
        site = {"version": 1, "origin": origin, "boundary": ring_xy(self.bpoly.exterior), "terrain": terrain,
                "buildings": buildings, "roads": roads, "trees": trees, "water": water,
                "shrubs": shrubs, "hedges": hedges, "land_cover": land_cover, "points_of_interest": pois,
                "cliffs": cliffs, **extra}
        site["report"] = self.report(site)
        return site

    def overture_info(self) -> dict:
        """report.info.overture: what the Overture fill did (counts per dataset, release, credits, why not)."""
        ov = self._overture()
        m = self.meta.get("overture") or {}
        info = {"enabled": bool(self.s["include"].get("buildings", True) and self.s["include"].get("overture_buildings", True)),
                "applied": bool(ov["on"]), "reason": ov.get("reason"), "release": m.get("release"),
                "overlap_max": float(self.s["overture_overlap_max"]), "fetched": m.get("count"),
                "fetch_status": m.get("status"), "download_est_bytes": m.get("download_est_bytes")}
        if ov["on"]:
            used = ov.get("by_dataset") or {}
            info.update(stats=ov["stats"], added=ov.get("inside", 0), by_dataset=used,
                        osm_buildings=self.counts.get("buildings_osm", 0),
                        attribution=ovt.buildings_credit(list(used)), licence=ovt.THEME_LICENCE,
                        source_credits=[ovt.credit_for(k) for k in used])
        return info

    def report(self, site) -> dict:
        lat = self.f.lat0
        cw, ch = self.dem.cell_size_m(lat)
        minx, miny, maxx, maxy = self.bpoly.bounds
        di = self.dem_info
        acc = di["dem_accuracy"]
        w = []
        if di["dem_source_id"] == "copernicus":
            w.append("DEM is Copernicus GLO-30, a digital SURFACE model (DSM) at ~30 m: buildings, bridges and "
                     "vegetation raise it locally and bilinear interpolation smooths everything; expect bumps of "
                     "several metres in urban/wooded areas and no detail finer than ~30 m. Heights are EGM2008 (geoid).")
            if self.ground_mode == "dsm":
                w.append("ground_mode=dsm on a surface model: building bases sit on the DSM and may float "
                         "several metres above the real ground.")
        elif acc.get("kind") == "unknown":
            w.append("DEM source unknown (not recorded in meta.json): used as-is; if it is a surface model (DSM) "
                     "buildings may float above the real ground.")
        else:
            w.append(f"DEM: {di['dem_source']} ({acc.get('kind')}, ~{acc.get('horizontal_m')} m grid, "
                     f"+-{acc.get('vertical_m')} m vertical; heights {di.get('dem_vertical_datum')}).")
        w.extend(self.ground_warnings)
        if max(cw, ch) > 5.0:
            w.append(f"Low-res DEM: {self.dem.nrows}x{self.dem.ncols} cells of ~{cw:.0f}x{ch:.0f} m for a "
                     f"{maxx - minx:.0f}x{maxy - miny:.0f} m site; terrain points at {self.spacing:g} m are "
                     "interpolated, not measured.")
        vals, cnt = np.unique(self.dem.a, return_counts=True)
        if di["dem_source_id"] in ("copernicus", None) and cnt.max() >= 4 and cnt.max() >= 0.1 * self.dem.a.size:
            w.append(f"DEM has {cnt.max()} cells with the identical value {vals[cnt.argmax()]:.2f} m "
                     "(GLO-30 flattened water body); river bed / bathymetry is not represented.")
        if di.get("dem_fallback_reason"):
            w.append(f"DEM fallback: {di['dem_fallback_reason']}")
        if self.dem.n_outside:
            w.append(f"{self.dem.n_outside} DEM samples fell outside the DEM extent (boundary + fetch margin) and "
                     "were clamped to the edge; these are on parts of roads/buildings/water beyond the boundary "
                     "(only affects kept-whole building bases and smoothing near the edge).")
        conv = self.f.crs_info()["meridian_convergence_deg"]
        if abs(conv) > 1.0:
            w.append(f"Grid north differs from true north by {conv:.2f} deg (UTM meridian convergence).")
        w.extend(self.dem.warnings)
        ovi = self.overture_info()
        if ovi.get("applied") and ovi.get("added"):
            n_est = sum(1 for b in site["buildings"] if b.get("estimated"))
            w.append(f"{ovi['added']} building(s) added from Overture Maps (release {ovi.get('release')}; "
                     f"{', '.join(f'{v} {k}' for k, v in ovi['by_dataset'].items())}) where OSM has none; {n_est} are "
                     "ML-derived footprints (estimated: positions/outlines less reliable than mapped OSM, heights from "
                     "Overture height/floors or the default). " + ovi["attribution"])
        elif ovi.get("enabled") and not ovi.get("applied") and ovi.get("reason"):
            w.append(f"Overture buildings not used: {ovi['reason']}")
        w.extend(f"fetch: {m}" for m in self.meta.get("warnings", []))
        skipped = {k: v for k, v in self.counts.items() if "skipped" in k or "outside" in k or "not_" in k}
        if skipped:
            w.append("skipped/not imported: " + ", ".join(f"{k}={v}" for k, v in sorted(skipped.items())))
        w.extend(self.warnings)
        ranges = {}
        for name, arr in (("terrain", site["terrain"]["points"]),
                          ("buildings", [[x, y, b["base_z"]] for b in site["buildings"] for x, y in b["footprint"]]),
                          ("roads", [p for r in site["roads"] for p in r["points"]]),
                          ("trees", [[t["x"], t["y"], t["z"]] for t in site["trees"]]),
                          ("water", [[x, y, wt["z"]] for wt in site["water"] for x, y in wt["polygon"]])):
            if arr:
                a = np.asarray(arr, float)
                ranges[name] = {k: [round(float(a[:, i].min()), 2), round(float(a[:, i].max()), 2)]
                                for i, k in enumerate("xyz")}
        dem_accuracy = dict(acc)
        dem_accuracy.update({"ground_mode": self.ground_mode, "source": di["dem_source"]})
        if self.ground_mode == "bare_earth_approx" and acc.get("kind") == "DSM":
            if self.ground_info.get("applied"):
                dem_accuracy["note"] = (acc.get("note", "") + " Ground approximated from the DSM by masking buildings/"
                                        "vegetation (bare_earth_approx); building bases remain uncertain by several metres.")
            else:
                dem_accuracy["note"] = (acc.get("note", "") + " bare_earth_approx was skipped (too few open land cells): "
                                        "the surface model is used as is.")
        slope = self.slope_stats()
        if slope:
            dem_accuracy["slope"] = slope
            if slope["cells_steep_pct"] > 0 or site.get("cliffs"):
                note = (f" Steep terrain: max slope {slope['max_pct']:.0f} %, {slope['cells_steep_pct']:.0f} % of cells "
                        f"steeper than {STEEP_SLOPE_DEG:g} deg; one cell of horizontal error means ~"
                        f"{slope['vertical_error_one_cell_m_p95']:.0f} m vertical error there (p95)"
                        + (", and mapped cliffs are smoothed into ramps" if site.get("cliffs") else "") + ".")
                dem_accuracy["note"] = dem_accuracy.get("note", "") + note
                w.append("DEM accuracy on steep ground:" + note)
        return {"counts": dict(sorted(self.counts.items())),
                "dem_accuracy": dem_accuracy,
                "defaults_used": {"heights": dict(self.heights), "widths": dict(self.widths),
                                  "tree_heights": dict(self.tree_heights),
                                  "tree_crowns": dict(self.tree_crowns),
                                  "shrub_heights": dict(getattr(self, "shrub_heights", {})),
                                  "hedge_heights": dict(getattr(self, "hedge_heights", {})),
                                  "waterway_widths": dict(self.waterway_widths), "settings": self.s},
                "warnings": w,
                "info": {"crs": self.f.crs_info(), "terrain": self.terrain_info, "tree_measure": self.ndom_info,
                         "tree_scatter": {"enabled": {k: veg.scatter_enabled(self.s, k) for k in ["forest", *veg.TREE_AREA_ORDER]},
                                          "by_area_type": self.scatter_report,
                                          "note": "estimated from mapped areas, not individual surveyed trees"},
                         "dem": {"shape": [self.dem.nrows, self.dem.ncols], "cell_m": [round(cw, 2), round(ch, 2)],
                                 "source": di["dem_source"], "source_id": di["dem_source_id"],
                                 "crs": di["dem_crs"], "attribution": di["dem_attribution"],
                                 "licence": di["dem_licence"], "vertical_datum": di["dem_vertical_datum"],
                                 "requested": di["dem_requested"], "fallback_reason": di["dem_fallback_reason"]},
                         "ground": self.ground_info,
                         "layers": self.layer_info,
                         "sea": {"present": any(x["kind"] == "coastline" for x in site["water"]),
                                 "z_abs": SEA_LEVEL, "vertical_datum": di.get("dem_vertical_datum"),
                                 "coastline_ways": int(self._water_geoms()["has_coast"])},
                         "overture": ovi,
                         "osm": {"overpass_url": self.meta.get("overpass_url"), "osm_base": self.meta.get("osm_base"),
                                 "osm_age_days": self.meta.get("osm_age_days"),
                                 "bridge_chains": self.meta.get("bridge_chains"),
                                 "dem_ext": bool(self.dem_ext is not None)},
                         "ranges": ranges,
                         "notes": ["road width_source 'default' covers both the class table and the fallback; "
                                   "see defaults_used.widths (tag/class/default)",
                                   "bridges/tunnels: never draped on the DEM; straight gradeline between abutment/portal "
                                   "ground z (dem_ext.tif beyond the DEM), raised over lower-layer roads by "
                                   f">= {BRIDGE_MIN_CLEARANCE:g} m and over area water by >= {BRIDGE_WATER_CLEARANCE:g} m "
                                   f"with ramps <= {RAMP_CAP:.0%} ({RAMP_CAP_SLOW:.0%} paths) where the abutments allow "
                                   "(see info.layers)",
                                   "non-bridge roads: z smoothed (triangular kernel, 15 m half-width), "
                                   "pinned to raw terrain at ends/junctions",
                                   "origin.elevation = lowest LAND point / water surface in the boundary (DEM artefacts "
                                   "below the water are ignored); terrain points under water may be < 0"]}}


# --------------------------------------------------------------------------- preview

def _contours(site: Site, poly: Polygon, step_m: float):
    """Whole-metre contour lines of the DEM surface over `poly`'s bbox: marching squares on a step_m grid, vectorised
    per level. Segments come out in row-major cell order, edges k=0..3 (x+, y+, x-, y- sides), paired (0,1),(2,3)."""
    minx, miny, maxx, maxy = poly.bounds
    g = max(1.0, step_m)
    xs = np.arange(minx, maxx + g, g)
    ys = np.arange(miny, maxy + g, g)
    X, Y = np.meshgrid(xs, ys)
    Z = site.z_abs(X.ravel(), Y.ravel()).reshape(X.shape)
    levels = np.arange(math.ceil(Z.min()), math.floor(Z.max()) + 1)
    grid = (len(ys) - 1, len(xs) - 1)
    # the 4 corners of every cell, in the order (i,j) (i,j+1) (i+1,j+1) (i+1,j)
    cx = np.stack([np.broadcast_to(c, grid) for c in (xs[None, :-1], xs[None, 1:], xs[None, 1:], xs[None, :-1])])
    cy = np.stack([np.broadcast_to(c, grid) for c in (ys[:-1, None], ys[:-1, None], ys[1:, None], ys[1:, None])])
    cz = np.stack([Z[:-1, :-1], Z[:-1, 1:], Z[1:, 1:], Z[1:, :-1]])
    nx_, ny_, nz_ = (np.roll(a, -1, axis=0) for a in (cx, cy, cz))   # far end of edge k
    res = []
    for lv in levels:
        below = cz < lv
        cross = below != np.roll(below, -1, axis=0)
        cells = np.nonzero(cross.any(axis=0))
        if not len(cells[0]):
            continue
        cr = cross[:, cells[0], cells[1]]
        az, bz = cz[:, cells[0], cells[1]], nz_[:, cells[0], cells[1]]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(cr, (lv - az) / (bz - az), 0.0)   # t only matters on crossed edges
        ax, ay = cx[:, cells[0], cells[1]], cy[:, cells[0], cells[1]]
        px = ax + t * (nx_[:, cells[0], cells[1]] - ax)
        py = ay + t * (ny_[:, cells[0], cells[1]] - ay)
        pts = np.column_stack([px.T[cr.T], py.T[cr.T]])   # (cell, edge) order
        segs = [LineString(c) for c in pts.reshape(-1, 2, 2)]
        g2 = linemerge(segs).intersection(poly)
        res.append((float(lv), lines_of(g2)))
    return res


def write_preview(site_json: dict, conv: Converter, path: Path, width_px: int = 900):
    b = conv.bpoly
    minx, miny, maxx, maxy = b.bounds
    for x in site_json["buildings"]:   # bounds of boundary + footprints (no polygon union needed)
        if len(x["footprint"]) >= 3:
            fx, fy = zip(*x["footprint"])
            minx, miny, maxx, maxy = min(minx, *fx), min(miny, *fy), max(maxx, *fx), max(maxy, *fy)
    pad = 40
    k = (width_px - 2 * pad) / max(maxx - minx, maxy - miny, 1)
    H = int((maxy - miny) * k + 2 * pad + 30)
    W = int((maxx - minx) * k + 2 * pad)

    def P(x, y):
        return f"{(x - minx) * k + pad:.1f},{(maxy - y) * k + pad:.1f}"

    def path_ring(r):
        return "M" + " L".join(P(x, y) for x, y in r) + " Z"

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="sans-serif">', f'<rect width="{W}" height="{H}" fill="#fbfaf7"/>']
    o.append('<g fill="none" stroke="#b9936b">')
    for lv, lines in _contours(conv.site, b, max(1.0, conv.spacing / 2)):
        major = int(lv) % 5 == 0
        for ln in lines:
            o.append(f'<path d="M{" L".join(P(x, y) for x, y in ln.coords)}" stroke-width="{1.0 if major else 0.4}"/>')
            if major and ln.length > 15:
                m = ln.interpolate(0.5, normalized=True)
                o.append(f'<text x="{P(m.x, m.y).split(",")[0]}" y="{P(m.x, m.y).split(",")[1]}" font-size="9" '
                         f'fill="#8a6a4a" stroke="none">{lv:.0f}</text>')
    o.append("</g>")
    lc_fill = {"beach": "#f3e2a9", "sand": "#eedc9a", "tidalflat": "#d8d0b0", "saltmarsh": "#b9d3a0",
               "mangrove": "#8fbf8a", "wetland": "#b9d3c0"}
    for lc in site_json.get("land_cover", []):
        d = path_ring(lc["polygon"]) + "".join(path_ring(h) for h in lc.get("holes", []))
        o.append(f'<path d="{d}" fill="{lc_fill.get(lc["kind"], "#e0d8c0")}" stroke="#b8a060" stroke-width="0.5" '
                 f'fill-rule="evenodd"><title>{lc["osm_id"]} {lc["kind"]}</title></path>')
    for wt in site_json["water"]:
        d = path_ring(wt["polygon"]) + "".join(path_ring(h) for h in wt.get("holes", []))
        o.append(f'<path d="{d}" fill="#a8d0f0" stroke="#4a8cc8" stroke-width="0.8" fill-rule="evenodd">'
                 f'<title>{wt["osm_id"]} {wt["kind"]} z={wt["z"]} ({wt.get("z_source", "")})</title></path>')
    for cl in site_json.get("cliffs", []):
        d = "M" + " L".join(P(x, y) for x, y, _ in cl["points"])
        o.append(f'<path d="{d}" fill="none" stroke="#6b4a2b" stroke-width="2.2" stroke-dasharray="1,3" '
                 f'stroke-linecap="round"><title>{cl["osm_id"]} natural=cliff</title></path>')
    for r in sorted(site_json["roads"], key=lambda r: (r["layer"], r["bridge"])):
        d = "M" + " L".join(P(x, y) for x, y, _ in r["points"])
        sw = r["width"] * k
        dash = ' stroke-dasharray="4,3" opacity="0.5"' if r["tunnel"] else ""
        case = "#222" if r["bridge"] else "#999"
        o.append(f'<path d="{d}" fill="none" stroke="{case}" stroke-width="{sw + 1.2:.1f}" stroke-linecap="butt" '
                 f'stroke-linejoin="round"{dash}/>')
        o.append(f'<path d="{d}" fill="none" stroke="#f2efe6" stroke-width="{sw:.1f}" stroke-linecap="butt" '
                 f'stroke-linejoin="round"{dash}/>')
    for bl in site_json["buildings"]:
        d = path_ring(bl["footprint"]) + "".join(path_ring(h) for h in bl["holes"])
        o.append(f'<path d="{d}" fill="#9a9a9a" stroke="#444" stroke-width="0.8" fill-rule="evenodd">'
                 f'<title>{bl["osm_id"]} h={bl["height"]} ({bl["height_source"]}) base_z={bl["base_z"]}</title></path>')
    for t in site_json["trees"]:
        x, y = P(t["x"], t["y"]).split(",")
        o.append(f'<circle cx="{x}" cy="{y}" r="{max(2.0, 0.5 * float(t.get("crown_diameter") or 4.0) * k):.1f}" fill="#3f9b4a" fill-opacity="0.75" '
                 f'stroke="#256b2f" stroke-width="0.6"/>')
    for h in site_json.get("hedges", []):
        d = "M" + " L".join(P(x, y) for x, y, _ in h["points"])
        o.append(f'<path d="{d}" fill="none" stroke="#2f7d32" stroke-width="{max(1.5, h["width"] * k):.1f}" stroke-linecap="round"/>')
    for sh in site_json.get("shrubs", []):
        x, y = P(sh["x"], sh["y"]).split(",")
        o.append(f'<circle cx="{x}" cy="{y}" r="{max(1.2, 0.5 * float(sh.get("crown_diameter") or 1.0) * k):.1f}" fill="#7bc47f" '
                 f'fill-opacity="0.8" stroke="#2f7d32" stroke-width="0.4"/>')
    for pi in site_json.get("points_of_interest", []):
        x, y = (float(v) for v in P(pi["x"], pi["y"]).split(","))
        lab = f'{pi.get("name") or pi["kind"]} OSM {pi["ele"]:.0f} / DEM {pi["dem_z_abs"]:.0f}' if pi.get("ele") \
            else f'{pi.get("name") or pi["kind"]} DEM {pi["dem_z_abs"]:.0f}'
        o.append(f'<path d="M{x},{y - 7} l-6,10 h12 Z" fill="#7a3b12"/><text x="{x + 8}" y="{y + 3}" font-size="10" '
                 f'fill="#7a3b12">{html.escape(lab)}</text>')
    o.append(f'<path d="{path_ring(site_json["boundary"])}" fill="none" stroke="#d0342c" stroke-width="1.6" '
             f'stroke-dasharray="6,4"/>')
    ox, oy = P(0, 0).split(",")
    o.append(f'<path d="M{float(ox) - 6},{oy} h12 M{ox},{float(oy) - 6} v12" stroke="#d0342c" stroke-width="1"/>')
    # north arrow (grid north) + scale bar
    ax, ay = W - pad, pad + 10
    o.append(f'<path d="M{ax},{ay - 22} l-7,20 l7,-5 l7,5 Z" fill="#222"/>'
             f'<text x="{ax}" y="{ay + 12}" font-size="12" text-anchor="middle">N</text>')
    sx, sy = pad, H - 18
    o.append(f'<path d="M{sx},{sy} h{10 * k:.1f}" stroke="#222" stroke-width="3"/>'
             f'<text x="{sx + 10 * k + 6:.1f}" y="{sy + 4}" font-size="11">10 m</text>')
    og = site_json["origin"]
    conv_deg = conv.f.crs_info()["meridian_convergence_deg"]
    o.append(f'<text x="{pad}" y="16" font-size="11" fill="#333">origin {og["lat"]:.6f}, {og["lon"]:.6f}  '
             f'EPSG:{og["epsg"]}  E {og["easting"]:.2f} N {og["northing"]:.2f}  elev {og["elevation"]:.2f} m  |  '
             f'contours 1 m (abs.)  |  grid north, convergence {conv_deg:.3f} deg</text>')
    o.append("</svg>")
    path.write_text("\n".join(o), encoding="utf-8")


# --------------------------------------------------------------------------- API / CLI

def convert(cache_dir, settings: dict | None = None, out_dir="out") -> dict:
    conv = Converter(cache_dir, settings)
    site = conv.run()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "site.json").write_text(json.dumps(site, ensure_ascii=False), encoding="utf-8")
    write_preview(site, conv, out / "preview.svg")
    return site


def format_report(site: dict) -> str:
    r = site["report"]
    og = site["origin"]
    lines = [f"origin: lat {og['lat']} lon {og['lon']} EPSG:{og['epsg']} E {og['easting']} N {og['northing']} "
             f"elevation {og['elevation']} m",
             "counts: " + ", ".join(f"{k}={v}" for k, v in r["counts"].items()),
             "heights: " + json.dumps(r["defaults_used"]["heights"]),
             "widths: " + json.dumps(r["defaults_used"]["widths"]),
             "tree heights: " + json.dumps(r["defaults_used"]["tree_heights"]),
             "tree crowns: " + json.dumps(r["defaults_used"].get("tree_crowns", {})),
             "shrub heights: " + json.dumps(r["defaults_used"].get("shrub_heights", {})) +
             " | hedge heights: " + json.dumps(r["defaults_used"].get("hedge_heights", {})),
             "crs: " + json.dumps(r["info"]["crs"]["meridian_convergence_deg"]) + " deg convergence, scale "
             + str(r["info"]["crs"]["point_scale_factor"]),
             "terrain: " + json.dumps({k: v for k, v in r["info"]["terrain"].items() if k != "points_order"})]
    for k, v in r["info"]["ranges"].items():
        lines.append(f"range {k}: x {v['x']} y {v['y']} z {v['z']}")
    lines += [f"WARNING: {w}" for w in r["warnings"]]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert cache/<hash> to out/site.json + preview.svg")
    ap.add_argument("cache_dir")
    ap.add_argument("--out", default="out")
    ap.add_argument("--settings", help="JSON file or inline JSON with settings overrides")
    a = ap.parse_args(argv)
    st = None
    if a.settings:
        p = Path(a.settings)
        st = json.loads(p.read_text(encoding="utf-8") if p.exists() else a.settings)
    site = convert(a.cache_dir, st, a.out)
    print(f"wrote {Path(a.out) / 'site.json'} and {Path(a.out) / 'preview.svg'}")
    print(format_report(site))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
