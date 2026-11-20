"""Vegetation beyond single trees: forest/shrub scatter, shrubs and hedges (M4).

Pure helpers (scatter, NdomIndex) plus functions that take the Converter (`cv`) and read its OSM elements /
frame / terrain sampler. convert.py only calls `forest_points`, `shrubs_and_hedges`, `ndom_index`.

Scatter is deterministic: the random stream is seeded from (area OSM id, `scatter_seed` setting), so the same
area always gives the same points. Points are dart-thrown inside the area (clipped to the site boundary) with a
minimum spacing; the target count is area / area-per-point (capped per area).
"""
from __future__ import annotations

import math
import zlib
from collections import Counter

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

VEG_DEFAULTS = {
    "forest_scatter": False,            # scatter tree points in natural=wood / landuse=forest areas
    "forest_area_per_tree": 40.0,       # m2 per scattered tree (1 per 40 m2)
    "forest_min_spacing": 4.0,          # m between scattered trees
    "forest_max_trees_per_area": 3000,  # safety cap per forest area
    "scatter_seed": 1,
    # area-based tree scatter per area type (estimated trees, not surveyed ones); `forest` is the legacy
    # `forest_scatter` switch (either one turns it on). Per-type overrides: tree_scatter_params[type] =
    # {area_per_tree, min_spacing, height, max_trees}
    "tree_scatter": {"mangrove": True, "orchard": True, "plant_nursery": True, "park": True, "cemetery": False,
                     "forest": False},
    "tree_scatter_params": {},
    "shrubbery_area_per_shrub": 6.0,    # m2 per shrub in natural=shrubbery
    "scrub_area_per_shrub": 20.0,       # m2 per shrub in natural=scrub
    "shrub_min_spacing": 1.5,
    "shrub_height": 1.5,                # default shrub height (m)
    "shrub_crown_ratio": 1.0,           # crown diameter / height
    "shrub_max_per_area": 2000,
    "hedge_height": 1.2,                # default hedge height (m)
    "hedge_width": 0.8,                 # default hedge width (m)
    "hedge_max_segment": 6.0,           # m, hedge lines are densified so no segment is longer
}
VEG_FLOAT_KEYS = ("forest_area_per_tree", "forest_min_spacing", "shrubbery_area_per_shrub", "scrub_area_per_shrub",
                  "shrub_min_spacing", "shrub_height", "shrub_crown_ratio", "hedge_height", "hedge_width",
                  "hedge_max_segment")
VEG_INT_KEYS = ("forest_max_trees_per_area", "scatter_seed", "shrub_max_per_area")
HEDGE_MEASURE_R = 1.0
HEDGE_ACCEPT_M = (0.4, 3.0)       # nDOM also sees tree canopy above a hedge; > 3 m is not trusted
SHRUB_MEASURE_R = 1.0
SHRUB_ACCEPT_M = (0.3, 4.0)       # same: > 4 m is more likely canopy than shrub
SHRUB_CROWN_CAP = 1.3          # x shrub_min_spacing: scattered crowns may not overlap much
MIN_HEDGE_RUN_M = 0.3
HEDGE_TAG_KEYS = ("species", "genus", "leaf_type", "hedge", "height", "width")
SHRUB_TAG_KEYS = ("leaf_type", "leaf_cycle", "genus", "species")


# area type -> defaults. area_per_tree m2 per tree, min_spacing m, height m (default when nothing is mapped or
# measured), grid = regular rows instead of dart throwing, avoid = keep out of buildings/water/roads/pitches.
TREE_AREA_TYPES = {
    "mangrove": {"label": "Mangrove", "area_per_tree": 25.0, "min_spacing": 3.0, "height": 6.0, "leaf_type": "broadleaved",
                 "grid": False, "avoid": False},
    "orchard": {"label": "Orchard", "area_per_tree": 30.0, "min_spacing": 5.0, "height": 5.0, "leaf_type": "broadleaved",
                "grid": True, "avoid": False},
    "plant_nursery": {"label": "Nursery", "area_per_tree": 15.0, "min_spacing": 2.5, "height": 3.0,
                      "leaf_type": "broadleaved", "grid": False, "avoid": False},
    "park": {"label": "Park", "area_per_tree": 120.0, "min_spacing": 6.0, "height": 10.0, "leaf_type": "broadleaved",
             "grid": False, "avoid": True},
    "cemetery": {"label": "Cemetery", "area_per_tree": 120.0, "min_spacing": 6.0, "height": 10.0,
                 "leaf_type": "broadleaved", "grid": False, "avoid": True},
}
TREE_AREA_ORDER = ("mangrove", "orchard", "plant_nursery", "park", "cemetery")
TREE_AREA_MAX_TREES = 3000
AVOID_BUILDING_BUFFER_M = 1.0
AVOID_WATER_BUFFER_M = 1.0
AVOID_ROAD_EXTRA_M = 1.0           # road buffer = width / 2 + this
AVOID_PITCH_BUFFER_M = 1.0
AREA_GROUP_LABEL = {"forest": "Forest", **{k: v["label"] for k, v in TREE_AREA_TYPES.items()}}


def area_tree_type(t: dict) -> str | None:
    """Tree-scatter area type of an OSM tag set (mangrove | orchard | plant_nursery | park | cemetery), else None.
    Woods/forests are handled by forest_points."""
    if t.get("wetland") == "mangrove":
        return "mangrove"
    if t.get("landuse") == "orchard":
        return "orchard"
    if t.get("landuse") == "plant_nursery":
        return "plant_nursery"
    if t.get("leisure") in ("park", "garden"):
        return "park"
    if t.get("landuse") == "cemetery" or t.get("amenity") == "grave_yard":
        return "cemetery"
    return None


def scatter_params(s: dict, typ: str) -> dict:
    """Effective {area_per_tree, min_spacing, height, max_trees} for an area type (defaults + overrides)."""
    d = TREE_AREA_TYPES[typ]
    o = (s.get("tree_scatter_params") or {}).get(typ) or {}
    return {"area_per_tree": float(o.get("area_per_tree") or d["area_per_tree"]),
            "min_spacing": float(o.get("min_spacing") or d["min_spacing"]),
            "height": float(o.get("height") or d["height"]),
            "max_trees": int(o.get("max_trees") or TREE_AREA_MAX_TREES)}


def scatter_enabled(s: dict, typ: str) -> bool:
    ts = s.get("tree_scatter") or {}
    if typ == "forest":
        return bool(s.get("forest_scatter") or ts.get("forest"))
    return bool(ts.get(typ, False))


def normalize_settings(s: dict) -> None:
    """In place: types of the vegetation keys (called by convert._merge_settings)."""
    ts = {k: bool(v) for k, v in (s.get("tree_scatter") or {}).items()}
    for k, v in VEG_DEFAULTS["tree_scatter"].items():
        ts.setdefault(k, v)
    s["tree_scatter"] = ts
    s["tree_scatter_params"] = {k: dict(v) for k, v in (s.get("tree_scatter_params") or {}).items()
                                if isinstance(v, dict)}
    for k in VEG_FLOAT_KEYS:
        s[k] = float(s[k])
    for k in VEG_INT_KEYS:
        s[k] = int(s[k])
    s["forest_scatter"] = bool(s.get("forest_scatter")) or ts.get("forest", False)
    ts["forest"] = s["forest_scatter"]


# --------------------------------------------------------------------------- scatter (pure)

def scatter(geom, area_per_point: float, min_spacing: float, key: str, seed: int = 1, cap: int = 3000,
            avoid=(), avoid_r: float = 0.0) -> tuple[list[tuple[float, float]], dict]:
    """Deterministic dart-throwing inside `geom` (Polygon/MultiPolygon, metres).

    target = floor(area / area_per_point) capped at `cap`; points keep >= min_spacing from each other and
    >= avoid_r from the `avoid` points. -> (points, {"target", "placed", "capped", "area_m2"}). Stops early (placed <
    target) when the area is too tight for the spacing."""
    info = {"target": 0, "placed": 0, "capped": False, "area_m2": 0.0}
    if geom is None or geom.is_empty or area_per_point <= 0:
        return [], info
    area = float(geom.area)
    info["area_m2"] = round(area, 1)
    target = int(area // area_per_point)
    if target > cap:
        info["capped"] = True
        target = int(cap)
    info["target"] = target
    if target <= 0:
        return [], info
    rng = np.random.default_rng([zlib.crc32(str(key).encode("utf-8")) & 0xFFFFFFFF, int(seed) & 0xFFFFFFFF])
    minx, miny, maxx, maxy = geom.bounds
    bbox_area = max((maxx - minx) * (maxy - miny), 1e-9)
    shapely.prepare(geom)
    cell = max(min_spacing, avoid_r, 1e-6)
    grid: dict[tuple[int, int], list[tuple[float, float, float]]] = {}

    def put(x, y, r):
        grid.setdefault((int(x // cell), int(y // cell)), []).append((x, y, r))

    for ax, ay in avoid:
        put(float(ax), float(ay), avoid_r)

    def clear(x, y):
        cx, cy = int(x // cell), int(y // cell)
        for i in (cx - 1, cx, cx + 1):
            for j in (cy - 1, cy, cy + 1):
                for px, py, need in grid.get((i, j), ()):   # need = distance a new point must keep from this one
                    if (x - px) ** 2 + (y - py) ** 2 < need * need:
                        return False
        return True

    pts: list[tuple[float, float]] = []
    max_attempts = int(min(5e6, 60 * target * max(1.0, bbox_area / max(area, 1e-9)) + 1000))
    attempts, batch = 0, int(min(max(512, target * 2), 50000))
    while len(pts) < target and attempts < max_attempts:
        xs = rng.uniform(minx, maxx, batch)
        ys = rng.uniform(miny, maxy, batch)
        attempts += batch
        inside = shapely.contains_xy(geom, xs, ys)
        for x, y in zip(xs[inside].tolist(), ys[inside].tolist()):
            if min_spacing > 0 and not clear(x, y):
                continue
            pts.append((x, y))
            put(x, y, min_spacing)
            if len(pts) >= target:
                break
    info["placed"] = len(pts)
    return pts, info


class NdomIndex:
    """nDOM cells sorted by x so a radius query touches only a thin column slice (no scipy needed)."""

    def __init__(self, cx, cy, vals):
        cx, cy, vals = np.asarray(cx, float), np.asarray(cy, float), np.asarray(vals, float)
        o = np.argsort(cx, kind="stable")
        self.cx, self.cy, self.v = cx[o], cy[o], vals[o]

    def measure(self, x: float, y: float, r: float, lo: float, hi: float):
        """max nDOM value within r of (x, y); None when no cell or outside [lo, hi]."""
        a, b = np.searchsorted(self.cx, [x - r, x + r])
        if b <= a:
            return None
        cx, cy, v = self.cx[a:b], self.cy[a:b], self.v[a:b]
        m = ((cx - x) ** 2 + (cy - y) ** 2 <= r * r) & np.isfinite(v)
        if not m.any():
            return None
        h = float(v[m].max())
        return h if lo <= h <= hi else None

    def measure_many(self, pts, r: float, lo: float, hi: float) -> list:
        return [self.measure(x, y, r, lo, hi) for x, y in pts]


def ndom_index(cv):
    """NdomIndex over the converter's nDOM (None when the cache has no ndom.tif). Memoised on `cv`."""
    if not hasattr(cv, "_ndom_idx"):
        cells = cv._ndom_cells()
        cv._ndom_idx = NdomIndex(*cells) if cells is not None else None
    return cv._ndom_idx


# --------------------------------------------------------------------------- OSM area geometry

def _polys(g) -> list[Polygon]:
    from .convert import polygons_of
    return [p for p in polygons_of(g) if p.area > 0]


def area_geoms(cv, pred, count: bool = True) -> list[tuple[dict, object]]:
    """(element, local geometry) for closed ways and multipolygon relations whose tags satisfy pred(tags).
    Unclosed ways / broken relations are counted in cv.counts[veg_areas_unusable]."""
    from .convert import is_closed_way, relation_area, way_lonlat
    out = []
    for el in cv.els:
        t = el.get("tags") or {}
        if not pred(t):
            continue
        g = None
        if el["type"] == "way":
            ll = way_lonlat(el.get("geometry"))
            if is_closed_way(ll):
                g = cv.f.geom_to_local(Polygon(ll))
        elif el["type"] == "relation" and t.get("type") in (None, "multipolygon"):
            g, _ = relation_area(el, cv.f)
        if g is None or g.is_empty:
            if count:
                cv.counts["veg_areas_unusable"] += 1
            continue
        out.append((el, make_valid(g)))
    return out


def _free_area(cv, g, done, excl=None):
    """`g` clipped to the site boundary, minus the already-scattered area `done` and (park trees) the exclusion
    zone `excl`, as one union of its polygon parts (empty geometry when nothing is left)."""
    g = g.intersection(cv.bpoly)
    if done is not None and not g.is_empty:
        g = g.difference(done)
    if excl is not None and not g.is_empty:
        g = g.difference(excl)
    return unary_union(_polys(g)) if not g.is_empty else g


def _note_scatter(cv, label, oid, info, cap):
    """Warning when the area hit its cap, counter when the spacing left it short of the target."""
    if info["capped"]:
        cv.warnings.append(f"{oid}: {label} scatter capped at {cap} points (area {info['area_m2']:.0f} m2)")
    if info["placed"] < info["target"]:
        cv.counts[f"{label}_scatter_short"] += 1


def _scatter_areas(cv, areas, min_spacing, seed, cap, avoid, avoid_r, label):
    """Scatter inside each (el, geom, area_per_point) in order; an area already covered by an earlier one
    (wood + forest drawn twice) is not scattered twice. Yields (el, points, info)."""
    from .convert import osm_id
    done = None
    for el, g, app in areas:
        g = _free_area(cv, g, done)
        if g.is_empty:
            continue
        done = g if done is None else unary_union([done, g])
        pts, info = scatter(g, app, min_spacing, osm_id(el), seed, cap, avoid, avoid_r)
        _note_scatter(cv, label, osm_id(el), info, cap)
        yield el, pts, info


# --------------------------------------------------------------------------- forests (trees)

def forest_points(cv, node_xy) -> list[tuple]:
    """Scatter tree points in wood/forest areas -> [(osm_id, x, y, tags, "forest", source_area)] shaped like the
    converter's tree candidates. Tags for sizing = the area's leaf_type/genus/species (height/crown tags of the
    area are NOT applied to every scattered tree)."""
    from .convert import osm_id
    s = cv.s

    mangrove_on = scatter_enabled(s, "mangrove")

    def is_forest(t):
        if mangrove_on and t.get("wetland") == "mangrove":    # scattered as mangrove instead
            return False
        return t.get("natural") == "wood" or t.get("landuse") == "forest"

    areas = [(el, g, s["forest_area_per_tree"]) for el, g in area_geoms(cv, is_forest)]
    out: list[tuple] = []
    n_areas = 0
    for el, pts, info in _scatter_areas(cv, areas, s["forest_min_spacing"], s["scatter_seed"],
                                        s["forest_max_trees_per_area"], node_xy, s["forest_min_spacing"] / 2.0,
                                        "forest"):
        t = el.get("tags") or {}
        tags = {k: t[k] for k in ("leaf_type", "leaf_cycle", "genus", "species", "taxon") if t.get(k)}
        oid = osm_id(el)
        out += [(oid, x, y, tags, "forest", oid) for x, y in pts]
        n_areas += 1
        _tally(cv, "forest", oid, len(pts), info["area_m2"])
    cv.counts["forest_areas_scattered"] = n_areas
    return out


def _tally(cv, typ, oid, n, area_m2):
    d = cv.scatter_report.setdefault(typ, {"areas": 0, "trees": 0, "area_m2": 0.0})
    d["areas"] += 1
    d["trees"] += n
    d["area_m2"] = round(d["area_m2"] + float(area_m2), 1)


# --------------------------------------------------------------------------- mangrove / orchard / nursery / park

def grid_scatter(geom, spacing: float, key: str, seed: int = 1, cap: int = 3000, avoid=(), avoid_r: float = 0.0):
    """Regular rows (orchard) inside `geom`: lattice `spacing` apart, rotated to the long side of the area's
    minimum rotated rectangle. Points < avoid_r from `avoid` are dropped. -> (points, info like scatter())."""
    import shapely.affinity as aff
    info = {"target": 0, "placed": 0, "capped": False, "area_m2": 0.0}
    if geom is None or geom.is_empty or spacing <= 0:
        return [], info
    info["area_m2"] = round(float(geom.area), 1)
    rect = geom.minimum_rotated_rectangle
    ang = 0.0
    if isinstance(rect, Polygon):
        c = list(rect.exterior.coords)
        e = max(zip(c, c[1:]), key=lambda ab: math.hypot(ab[1][0] - ab[0][0], ab[1][1] - ab[0][1]))
        ang = math.degrees(math.atan2(e[1][1] - e[0][1], e[1][0] - e[0][0]))
    rg = aff.rotate(geom, -ang, origin=(0, 0))
    minx, miny, maxx, maxy = rg.bounds
    xs = np.arange(minx + spacing / 2.0, maxx, spacing)
    ys = np.arange(miny + spacing / 2.0, maxy, spacing)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    m = shapely.contains_xy(rg, gx, gy)
    gx, gy = gx[m], gy[m]
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    px, py = gx * ca - gy * sa, gx * sa + gy * ca
    pts = [(float(a), float(b)) for a, b in zip(px, py)]
    if avoid_r > 0 and len(avoid) and pts:
        av = np.asarray(avoid, float).reshape(-1, 2)
        pa = np.asarray(pts, float)
        # candidate pairs from a spatial index (dwithin is <=, padded), the exact strict `< avoid_r` test as before
        qi, ai = shapely.STRtree(shapely.points(av[:, 0], av[:, 1])).query(
            shapely.points(pa[:, 0], pa[:, 1]), predicate="dwithin", distance=avoid_r * (1 + 1e-9) + 1e-9)
        hit = np.hypot(av[ai, 0] - pa[qi, 0], av[ai, 1] - pa[qi, 1]) < avoid_r
        keep = np.ones(len(pts), dtype=bool)
        keep[qi[hit]] = False
        pts = [p for p, k in zip(pts, keep.tolist()) if k]
    info["target"] = len(pts)
    if len(pts) > cap:
        info["capped"] = True
        pts = pts[:cap]
    info["placed"] = len(pts)
    return pts, info


def _park_exclusion(cv):
    """Union of what a park/cemetery tree must stay out of: buildings (+1 m), water (+1 m), road lines
    (width/2 + 1 m), sports pitches (+1 m). None when nothing is mapped."""
    from .convert import parse_length, way_lonlat

    def excl_buf(t):
        if "building" in t and t.get("building") != "no":
            return AVOID_BUILDING_BUFFER_M
        if t.get("natural") == "water" or t.get("waterway") in ("riverbank", "dock") or t.get("landuse") == "basin":
            return AVOID_WATER_BUFFER_M
        if t.get("leisure") in ("pitch", "swimming_pool"):
            return AVOID_PITCH_BUFFER_M
        return None

    parts = [g.buffer(excl_buf(el["tags"])) for el, g in area_geoms(cv, lambda t: excl_buf(t) is not None, count=False)]
    widths = cv.s.get("road_widths") or {}
    for el in cv.els:
        t = el.get("tags") or {}
        if el["type"] != "way":
            continue
        if "highway" in t:
            ll = way_lonlat(el.get("geometry"))
            if len(ll) < 2:
                continue
            w = parse_length(t.get("width"), max_value=60.0) or float(widths.get(t["highway"], widths.get("default", 4.0)))
            parts.append(cv.f.geom_to_local(LineString(ll)).buffer(w / 2.0 + AVOID_ROAD_EXTRA_M))
        elif t.get("waterway") in ("river", "canal", "stream", "drain", "ditch"):
            ll = way_lonlat(el.get("geometry"))
            if len(ll) >= 2:
                wd = parse_length(t.get("width"), max_value=200.0) or {"river": 20.0, "canal": 10.0}.get(t["waterway"], 2.0)
                parts.append(cv.f.geom_to_local(LineString(ll)).buffer(wd / 2.0 + AVOID_WATER_BUFFER_M))
    return unary_union(parts) if parts else None


def area_tree_points(cv, avoid_xy=(), taken=None) -> list[tuple]:
    """Scatter estimated trees in mangrove / orchard / plant_nursery / park(+garden) / cemetery areas (only the
    types switched on in settings.tree_scatter). -> [(osm_id, x, y, tags, area_type, osm_id, {"height"})]
    shaped like forest_points (+ the area default). `taken` = geometry already scattered (woods) left out."""
    from .convert import osm_id
    s = cv.s
    on = [t for t in TREE_AREA_ORDER if scatter_enabled(s, t)]
    if not on:
        return []
    cands = {t: [] for t in on}
    for el, g in area_geoms(cv, lambda t: area_tree_type(t) in on, count=False):
        cands[area_tree_type(el["tags"])].append((el, g))
    excl = None
    if any(TREE_AREA_TYPES[t]["avoid"] and cands[t] for t in on):
        excl = _park_exclusion(cv)
    out: list[tuple] = []
    avoid = list(avoid_xy)
    done = taken
    for typ in on:
        prm = scatter_params(s, typ)
        d = TREE_AREA_TYPES[typ]
        for el, g in sorted(cands[typ], key=lambda a: osm_id(a[0])):
            oid = osm_id(el)
            g = _free_area(cv, g, done, excl if d["avoid"] else None)
            if g.is_empty:
                continue
            done = g if done is None else unary_union([done, g])
            if d["grid"]:
                sp = max(math.sqrt(prm["area_per_tree"]), prm["min_spacing"])
                pts, info = grid_scatter(g, sp, oid, s["scatter_seed"], prm["max_trees"], avoid, prm["min_spacing"] / 2.0)
            else:
                pts, info = scatter(g, prm["area_per_tree"], prm["min_spacing"], oid, s["scatter_seed"],
                                    prm["max_trees"], avoid, prm["min_spacing"] / 2.0)
            _note_scatter(cv, typ, oid, info, prm["max_trees"])
            t = el["tags"]
            tags = {k: t[k] for k in ("leaf_type", "leaf_cycle", "genus", "species", "taxon") if t.get(k)}
            tags.setdefault("leaf_type", d["leaf_type"])
            extra = {"height": prm["height"]}
            out += [(oid, x, y, tags, typ, oid, extra) for x, y in pts]
            avoid += pts
            _tally(cv, typ, oid, len(pts), info["area_m2"])
            cv.counts[f"{typ}_areas_scattered"] += 1
    return out


# --------------------------------------------------------------------------- shrubs + hedges

def _shrub_dict(oid, x, y, z, t, nd, s, kind, area_id, crown_cap=None):
    from .convert import parse_length
    th = parse_length(t.get("height"), max_value=15.0)
    tc = parse_length(t.get("diameter_crown"), max_value=30.0)
    meas = nd.measure(x, y, SHRUB_MEASURE_R, *SHRUB_ACCEPT_M) if nd is not None else None
    if th is not None:
        h, hs = th, "tag"
    elif meas is not None:
        h, hs = meas, "measured"
    else:
        h, hs = s["shrub_height"], "default"
    if tc is not None:
        c, cs = tc, "tag"
    else:
        c, cs = max(0.6, h * s["shrub_crown_ratio"]), "ratio"
        if crown_cap is not None and c > crown_cap:
            c, cs = max(0.6, crown_cap), "ratio_capped"
    return {"osm_id": oid, "x": round(x, 3), "y": round(y, 3), "z": round(float(z), 3), "kind": kind,
            "source_area": area_id,
            "tags": {k: str(t[k]) for k in SHRUB_TAG_KEYS if t.get(k)},
            "height": round(h, 3), "height_source": hs, "crown_diameter": round(c, 3), "crown_source": cs}


def shrubs_and_hedges(cv, tree_xy=()) -> tuple[list[dict], list[dict]]:
    """-> (shrubs, hedges) for site.json. shrubs: natural=shrub nodes + scatter in natural=shrubbery /
    natural=scrub areas. hedges: barrier=hedge ways clipped to the boundary, densified, z from the terrain."""
    from .convert import (clip_polyline_z, densify, osm_id, parse_length, way_lonlat)
    s = cv.s
    nd = ndom_index(cv)
    shrub_h, hedge_h = Counter(), Counter()
    # ---- shrubs: nodes
    cand = []   # (oid, x, y, tags, kind, area_id, crown_cap)
    for el in cv.els:
        t = el.get("tags") or {}
        if el["type"] == "node" and t.get("natural") == "shrub":
            x, y = cv.f.to_local(el["lon"], el["lat"])
            if cv.bpoly.covers(Point(float(x), float(y))):
                cand.append((osm_id(el), float(x), float(y), t, "shrub", None, None))
    node_xy = [(c[1], c[2]) for c in cand]
    # ---- shrubs: areas (shrubbery before scrub, deterministic order inside each)
    def per_pt(t):
        return s["shrubbery_area_per_shrub"] if t.get("natural") == "shrubbery" else s["scrub_area_per_shrub"]

    areas = [(el, g, per_pt(el["tags"])) for el, g in
             area_geoms(cv, lambda t: t.get("natural") in ("shrubbery", "scrub"))]
    areas.sort(key=lambda a: 0 if a[0]["tags"].get("natural") == "shrubbery" else 1)
    cap_crown = SHRUB_CROWN_CAP * s["shrub_min_spacing"]
    n_areas = 0
    for el, pts, _info in _scatter_areas(cv, areas, s["shrub_min_spacing"], s["scatter_seed"],
                                        s["shrub_max_per_area"], list(node_xy) + list(tree_xy),
                                        s["shrub_min_spacing"] / 2.0, "shrub"):
        n_areas += 1
        for x, y in pts:
            cand.append((osm_id(el), x, y, el["tags"], el["tags"]["natural"], osm_id(el), cap_crown))
    shrubs: list[dict] = []
    if cand:
        zs = cv.z_rel([c[1] for c in cand], [c[2] for c in cand])
        for (oid, x, y, t, kind, area_id, cap), z in zip(cand, zs):
            d = _shrub_dict(oid, x, y, z, t, nd, s, kind, area_id, cap)
            shrub_h[d["height_source"]] += 1
            shrubs.append(d)
    cv.counts["shrubs"] = len(shrubs)
    cv.counts["shrubs_from_nodes"] = sum(1 for d in shrubs if d["source_area"] is None)
    cv.counts["shrubs_from_areas"] = len(shrubs) - cv.counts["shrubs_from_nodes"]
    cv.counts["shrub_areas_scattered"] = n_areas
    # ---- hedges
    hedges: list[dict] = []
    ways = 0
    max_seg = max(s["hedge_max_segment"], 0.5)
    for el in cv.els:
        t = el.get("tags") or {}
        if el["type"] != "way" or t.get("barrier") != "hedge":
            continue
        ll = way_lonlat(el.get("geometry"))
        if len(ll) < 2:
            continue
        ln = cv.f.geom_to_local(LineString(ll))
        if ln.length < MIN_HEDGE_RUN_M:
            continue
        xy = np.asarray(ln.coords, float)
        dense, _ = densify(xy, max_seg)
        runs = [r for r in clip_polyline_z(np.column_stack([dense, np.zeros(len(dense))]), cv.bpoly)
                if float(np.hypot(*np.diff(r[:, :2], axis=0).T).sum()) >= MIN_HEDGE_RUN_M]
        if not runs:
            continue
        ways += 1
        th = parse_length(t.get("height"), max_value=10.0)
        tw = parse_length(t.get("width"), max_value=10.0)
        for r in runs:
            z = cv.z_rel(r[:, 0], r[:, 1])
            if th is not None:
                h, hs = th, "tag"
            else:
                vals = [m for m in (nd.measure_many(r[:, :2].tolist(), HEDGE_MEASURE_R, *HEDGE_ACCEPT_M)
                                    if nd is not None else []) if m is not None]
                if nd is not None and len(vals) >= max(2, len(r) // 2):
                    h, hs = float(np.median(vals)), "measured"
                else:
                    h, hs = s["hedge_height"], "default"
            w, ws = (tw, "tag") if tw is not None else (s["hedge_width"], "default")
            hedge_h[hs] += 1
            hedges.append({"osm_id": osm_id(el), "points": [[round(float(a), 3), round(float(b), 3), round(float(c), 3)]
                                                            for (a, b), c in zip(r[:, :2], z)],
                           "height": round(h, 3), "height_source": hs, "width": round(w, 3), "width_source": ws,
                           "tags": {k: str(t[k]) for k in HEDGE_TAG_KEYS if t.get(k)}})
    cv.counts["hedge_ways"] = ways
    cv.counts["hedges"] = len(hedges)
    cv.counts["hedge_length_m"] = round(sum(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(h["points"], h["points"][1:]))
                                            for h in hedges), 1)
    cv.shrub_heights, cv.hedge_heights = shrub_h, hedge_h
    return shrubs, hedges
