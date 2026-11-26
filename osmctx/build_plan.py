"""Pure mapping: site.json -> Archicad (Tapir) command payload items. No Archicad access.

All coordinates are project coordinates in metres; z = site z + z_offset
(site z is already relative to origin.elevation = Archicad project zero).
"""
from __future__ import annotations

import bisect
import datetime as _dt
import math
import zlib

import numpy as np
import shapely
from shapely.geometry import Polygon

from . import contours as ct
from . import features as feat
from . import trees as treelib

ROAD_HEIGHT = 0.2
WATER_THICKNESS = 0.1
MIN_BUILDING_HEIGHT = 0.5
TERRAIN_SKIRT = 5.0  # mesh solid body extends 5 m below the lowest terrain point (user, 2026-11-15)
MIN_SEGMENT = 0.01


def _r(v, n=4):
    return round(float(v), n)


def _p2(p):
    return {"x": _r(p[0]), "y": _r(p[1])}


def _p3(p, dz=0.0):
    return {"x": _r(p[0]), "y": _r(p[1]), "z": _r(p[2] + dz)}


def _z_relative_to_level(item: dict) -> dict:
    """Archicad reads mesh vertex z RELATIVE to the mesh `level` (found in M5-T2: the old 6.6 cm / 0.42 m terrain
    zMax error was exactly the level). Convert absolute z (as built by _p3) to level-relative, in place."""
    lvl = item["level"]
    for q in item.get("polygonCoordinates", []):
        q["z"] = _r(q["z"] - lvl)
    for sl in item.get("sublines", []):
        for q in sl["coordinates"]:
            q["z"] = _r(q["z"] - lvl)
    for h in item.get("holes", []) or []:
        for q in h.get("polygonCoordinates", []):
            if "z" in q:
                q["z"] = _r(q["z"] - lvl)
    return item


def _open_ring(ring):
    ring = [list(p) for p in ring]
    if len(ring) > 1 and abs(ring[0][0] - ring[-1][0]) < 1e-9 and abs(ring[0][1] - ring[-1][1]) < 1e-9:
        ring = ring[:-1]
    return ring


def building_source(b: dict, default: str) -> str:
    """Source property of a building element: OSM ones use the build source string, Overture ones state the
    dataset ("Overture Maps (Google Open Buildings)")."""
    s = str(b.get("source") or "")
    if s.startswith("overture:"):
        return f"Overture Maps ({s.split(':', 1)[1]})"
    if s.startswith("lod2:"):       # M10: "LoD2 (<dataset>)", dataset label from the lod2 block, else the region id
        return f"LoD2 ({(b.get('lod2') or {}).get('dataset') or s.split(':', 1)[1]})"
    return default


def is_overture(osm_id: str) -> bool:
    return str(osm_id).startswith("overture/")


def source_string(site: dict, build_date: str | None = None) -> str:
    info = (site.get("report") or {}).get("info") or {}
    dem = (info.get("dem") or {}).get("source", "DEM")
    dem = str(dem).split(" (")[0]
    build_date = build_date or _dt.date.today().isoformat()
    return f"OSM / {dem} / build {build_date}"


# ---------------------------------------------------------------- terrain
def terrain_rows(site: dict, spacing: float | None = None):
    """Interior terrain points grouped into rows (by y) sorted by x. Returns list[list[[x,y,z]]]."""
    t = site["terrain"]
    nb = len(t["boundary"])
    interior = list(t["points"][nb:])
    if not interior:
        return []
    if spacing is None:
        spacing = (((site.get("report") or {}).get("info") or {}).get("terrain") or {}).get("spacing_m") or 5.0
    interior.sort(key=lambda p: (p[1], p[0]))
    rows, cur = [], [interior[0]]
    for p in interior[1:]:
        if abs(p[1] - cur[0][1]) > spacing * 0.5:
            rows.append(cur)
            cur = [p]
        else:
            cur.append(p)
    rows.append(cur)
    return [sorted(r, key=lambda p: p[0]) for r in rows]


# Terrain plan readability (T4): AllSharp draws every triangulation line on the plan. "clean" = ridges UserDefined
# (only along the leveling sublines), secondary lines off, contour/level-line pen light grey (pen 18 "Fills - color 1");
# 3D geometry is unchanged. "legacy" = the old AllSharp look.
TERRAIN_STYLES = ("clean", "legacy")
TERRAIN_CONTOUR_PEN = 18
TERRAIN_PLAN_TINT = {"red": 0.66, "green": 0.89, "blue": 0.65}  # GO rule: empty fill + this background = light green tint
PLAN_FILL_RGB = {
    "water": (0.68, 0.85, 0.98),       # light blue
    "ground": (0.60, 0.88, 0.60),      # light green
    "buildings": (0.80, 0.80, 0.80),   # light gray
    "roads": (0.38, 0.38, 0.38),       # dark gray
}
# 2D style presets (build_ac.apply_plan_styles / pick_contour_pens / apply_terrain_go):
#   colour = default (PLAN_FILL_RGB, TERRAIN_PLAN_TINT), mono = grey palette below, none = no restyle.
PLAN_STYLES = ("colour", "mono", "none")
PLAN_FILL_RGB_MONO = {
    "water": (0.88, 0.88, 0.88),       # light grey
    "ground": (0.97, 0.97, 0.97),      # white / very light grey
    "buildings": (0.55, 0.55, 0.55),   # mid grey
    "roads": (0.28, 0.28, 0.28),       # dark grey
}
MONO_TERRAIN_PLAN_TINT = {"red": 0.97, "green": 0.97, "blue": 0.97}   # GO rule background for the ground Mesh
CONTOUR_RGB_MONO = {"major": (0.0, 0.0, 0.0), "minor": (0.5, 0.5, 0.5)}   # black major, mid-grey minor


def terrain_style_params(style: str = "clean") -> dict:
    """Mesh fields for the terrain plan look (merged into the CreateMeshes item)."""
    if style == "legacy":
        return {"ridges": "AllSharp"}
    return {"ridges": "UserDefined", "showLines": False, "contourPen": TERRAIN_CONTOUR_PEN}


TERRAIN_POINT_MODES = ("points", "rows")
CONTOUR_TYPES = ("mesh", "stepped", "morph", "gdl")   # stepped = Slab terrain; morph/gdl = M11 3D contours


def _check_point_mode(mode: str) -> str:
    if mode not in TERRAIN_POINT_MODES:
        raise ValueError(f"terrain_points must be one of {TERRAIN_POINT_MODES}, got {mode!r}")
    return mode


def terrain_stepped_slabs(site: dict, dz: float = 0.0, skirt: float = TERRAIN_SKIRT,
                          interval: float | None = None, scale: float = 1.0) -> tuple[list[dict], dict]:
    """Stepped contour model: stacked horizontal Slabs extruded per contour interval (zero slanted surfaces)."""
    data = ct.stepped_terraces(site, interval=interval, scale=scale)
    items = []
    for step in data["steps"]:
        bot_rel = (step["z_bot"] - skirt) if step["is_base"] else step["z_bot"]
        thick = max(step["z_top"] - bot_rel, 0.1 if step["is_base"] else 0.05)
        bot = bot_rel + dz
        items += [_slab(p["polygon"], p.get("holes"), bot, thick, "Bottom", fill=True) for p in step["polygons"]]
    return items, data


def terrain_mesh(site: dict, dz: float = 0.0, skirt: float = TERRAIN_SKIRT, style: str = "clean",
                 terrain_points: str = "points") -> dict:
    """One CreateMeshes item. Outline = terrain.boundary with z (taken from terrain.points ring).
    Interior grid points: terrain_points="points" (default) = one single-point subline per point (Archicad level
    points: they carry the height, have no ridge to draw, verified live 2026-11-17); "rows" = the old one polyline
    per grid row (draws a leveling line on the plan), kept for comparison."""
    _check_point_mode(terrain_points)
    t = site["terrain"]
    bnd = _open_ring(t["boundary"])
    pts = t["points"]
    ring = []
    for i, b in enumerate(bnd):
        p = pts[i] if i < len(pts) and abs(pts[i][0] - b[0]) < 1e-6 and abs(pts[i][1] - b[1]) < 1e-6 else None
        if p is None:  # fall back to nearest terrain point
            p = min(pts, key=lambda q: (q[0] - b[0]) ** 2 + (q[1] - b[1]) ** 2)
        ring.append([b[0], b[1], p[2]])
    poly = Polygon([(p[0], p[1]) for p in ring]).buffer(-0.05)
    rows = terrain_rows(site)
    z_min = min(p[2] for p in ring)
    sub = []
    if rows:
        flat = [p for r in rows for p in r]
        arr = np.asarray(flat, dtype=float)
        z_min = min(z_min, float(arr[:, 2].min()))
        shapely.prepare(poly)
        mask = shapely.covers(poly, shapely.points(arr[:, 0], arr[:, 1]))
        if terrain_points == "points":
            sub = [[flat[i]] for i in np.flatnonzero(mask)]
        else:
            start = 0
            for row in rows:
                keeps = mask[start:start + len(row)]
                start += len(row)
                run = []
                for p, keep in zip(row, keeps):
                    if keep:
                        run.append(p)
                    else:
                        if len(run) > 1:
                            sub.append(run)
                        run = []
                if len(run) > 1:
                    sub.append(run)
                elif len(run) == 1:  # lone point: 2-vertex stub so it still becomes a level point
                    q = run[0]
                    sub.append([q, [q[0] + 0.05, q[1], q[2]]])
    return _z_relative_to_level({
        "floorIndex": 0,
        "level": _r(z_min + dz),
        "skirtType": "SolidBodyWithSkirt",
        "skirtLevel": skirt,
        **terrain_style_params(style),
        "polygonCoordinates": [_p3(p, dz) for p in ring],
        "sublines": [{"coordinates": [_p3(p, dz) for p in r]} for r in sub],
    })


# ------------------------------------------------------------- land cover
# natural=beach/sand, wetland=tidalflat/saltmarsh/mangrove areas (site.json land_cover[]) -> one draped Mesh per area
# (terrain tool, no solid operator): outline z + interior level points sampled from the site terrain, lifted by
# LAND_COVER_LIFT so it does not z-fight with the terrain mesh. Mesh surface cannot be set through the API (gap 10);
# the matching BUILDING MATERIAL is set instead (Sand / Soil).
LAND_COVER_LIFT = 0.03
AREA_GROUP_LABEL = {"forest": "Forest", "mangrove": "Mangrove", "orchard": "Orchard", "plant_nursery": "Nursery",
                    "park": "Park", "cemetery": "Cemetery"}   # == vegetation.AREA_GROUP_LABEL
LAND_COVER_MATERIAL = {"beach": "Sand", "sand": "Sand", "tidalflat": "Sand", "wetland": "Soil", "saltmarsh": "Soil",
                       "mangrove": "Soil", "grass": "Grass", "meadow": "Grass", "heath": "Grass"}
# names that may not exist in the project: resolved at build time (exact name, then a building material whose name
# contains the word, case-insensitive), else this fallback
LAND_COVER_FALLBACK = {"Grass": "Soil", "Asphalt": "Concrete", "Paving": "Stone - Finish"}
LAND_COVER_MIN_AREA = 1.0  # m2, smaller pieces are skipped
LAND_COVER_SKIRT = "SurfaceOnly"


class TerrainSampler:
    """z(x, y) on the site terrain: bilinear on the regular interior grid, local plane fit elsewhere."""

    def __init__(self, site: dict):
        t = site["terrain"]
        nb = len(t["boundary"])
        self.pts_arr = np.asarray(t["points"], dtype=float) if len(t["points"]) else np.empty((0, 3), dtype=float)
        self.grid = {(round(p[0], 3), round(p[1], 3)): p[2] for p in t["points"][nb:]}
        self.xs = sorted({k[0] for k in self.grid})
        self.ys = sorted({k[1] for k in self.grid})

    def _local(self, x, y):
        """Outside the interior grid (outer half cell): planar least-squares through the 8 nearest terrain points
        (exact on planes); inverse-distance fallback when the points are degenerate."""
        arr = self.pts_arr
        d2 = (arr[:, 0] - x) ** 2 + (arr[:, 1] - y) ** 2
        k = min(8, len(d2))
        idx = np.argpartition(d2, k - 1)[:k] if len(d2) > k else np.arange(len(d2))
        near = arr[idx]
        d2_near = d2[idx]
        if d2_near.min() < 1e-12:
            return float(near[int(np.argmin(d2_near)), 2])
        A = np.column_stack([near[:, 0] - x, near[:, 1] - y, np.ones(len(near), dtype=float)])
        zz = near[:, 2]
        if np.linalg.matrix_rank(A) == 3:
            return float(np.linalg.lstsq(A, zz, rcond=None)[0][2])
        w = 1.0 / np.maximum(A[:, 0] ** 2 + A[:, 1] ** 2, 1e-12)
        return float((w * zz).sum() / w.sum())

    def z(self, x, y):
        xs, ys = self.xs, self.ys
        if len(xs) > 1 and len(ys) > 1 and xs[0] <= x <= xs[-1] and ys[0] <= y <= ys[-1]:
            i = min(max(bisect.bisect_right(xs, x) - 1, 0), len(xs) - 2)
            j = min(max(bisect.bisect_right(ys, y) - 1, 0), len(ys) - 2)
            c = [self.grid.get((xs[i + a], ys[j + b])) for b in (0, 1) for a in (0, 1)]
            if None not in c:
                u = (x - xs[i]) / (xs[i + 1] - xs[i])
                v = (y - ys[j]) / (ys[j + 1] - ys[j])
                return (c[0] * (1 - u) + c[1] * u) * (1 - v) + (c[2] * (1 - u) + c[3] * u) * v
        return self._local(x, y)


def _cover_parts(site: dict, areas) -> list[tuple[dict, Polygon]]:
    """[(area, polygon part)] of the land-cover-like areas ({polygon, holes, ...}) clipped to the terrain outline;
    parts smaller than LAND_COVER_MIN_AREA or with < 3 outline vertices are dropped."""
    if not areas:
        return []
    tb = Polygon([(p[0], p[1]) for p in _open_ring(site["terrain"]["boundary"])])
    shapely.prepare(tb)
    out = []
    for a in areas:
        try:
            poly = Polygon([(p[0], p[1]) for p in _open_ring(a["polygon"])],
                           [[(p[0], p[1]) for p in _open_ring(h)] for h in (a.get("holes") or []) if len(h) >= 3])
            if not poly.is_valid:
                poly = poly.buffer(0)
            poly = poly.intersection(tb)
        except Exception:  # noqa: BLE001
            continue
        # get_parts: Multi* and GeometryCollection (polygon + stray lines from an edge touch) alike
        out += [(a, g) for g in shapely.get_parts(poly)
                if isinstance(g, Polygon) and g.area >= LAND_COVER_MIN_AREA and len(g.exterior.coords) - 1 >= 3]
    return out


def _draped_meshes(site: dict, areas, dz, lift, sampler, terrain_points) -> list[tuple[dict, dict]]:
    """[(area, CreateMeshes item)] for the areas, draped on terrain z + lift (see land_cover_meshes)."""
    parts = _cover_parts(site, areas)
    if not parts:
        return []
    smp = sampler or TerrainSampler(site)
    nb = len(site["terrain"]["boundary"])
    interior = site["terrain"]["points"][nb:]
    int_arr = np.asarray(interior, dtype=float) if interior else np.empty((0, 3), dtype=float)
    int_tree = shapely.STRtree(shapely.points(int_arr[:, 0], int_arr[:, 1])) if len(int_arr) else None
    clean_style = terrain_style_params("clean")

    def drape(p):
        return [p[0], p[1], smp.z(p[0], p[1]) + lift]

    out = []
    for a, g in parts:
        outline = [drape(p) for p in list(g.exterior.coords)[:-1]]
        holes = [{"polygonCoordinates": [_p3(drape(p), dz) for p in list(h.coords)[:-1]]} for h in g.interiors
                 if len(h.coords) > 3]
        inner = g.buffer(-0.3)
        sub = []
        if not inner.is_empty and int_tree is not None:
            idx = np.sort(int_tree.query(inner, predicate="covers"))   # index order = the old boolean-mask order
            rows = {}
            for px, py, pz in int_arr[idx].tolist():
                rows.setdefault(round(py, 3), []).append([px, py, pz + lift])
            for y in sorted(rows):
                r = sorted(rows[y], key=lambda q: q[0])
                if terrain_points == "points":
                    sub += [{"coordinates": [_p3(q, dz)]} for q in r]
                elif len(r) > 1:
                    sub.append({"coordinates": [_p3(q, dz) for q in r]})
                else:
                    q = r[0]
                    sub.append({"coordinates": [_p3(q, dz), _p3([q[0] + 0.05, q[1], q[2]], dz)]})
        item = {"floorIndex": 0, "level": _r(min(p[2] for p in outline) + dz), "skirtType": LAND_COVER_SKIRT,
                **clean_style, "polygonCoordinates": [_p3(p, dz) for p in outline]}
        if sub:   # CreateMeshes: sublines minItems 1 (small areas hold no interior terrain point: omit the key)
            item["sublines"] = sub
        if holes:
            item["holes"] = holes
        out.append((a, _z_relative_to_level(item)))
    return out


def land_cover_meshes(site: dict, dz: float = 0.0, lift: float = LAND_COVER_LIFT, sampler=None,
                      terrain_points: str = "points") -> list[dict]:
    """-> [{"osm_id", "kind", "material", "item"}]: one CreateMeshes item per land_cover polygon part, clipped to the
    terrain outline. Outline vertices and interior terrain grid points (as sublines, like the terrain mesh) are draped
    on terrain z + lift. Holes become mesh holes. terrain_points as in terrain_mesh (single level points | rows)."""
    _check_point_mode(terrain_points)
    return [{"osm_id": a.get("osm_id"), "kind": a.get("kind"),
             "material": LAND_COVER_MATERIAL.get(a.get("kind"), "Soil"), "item": item}
            for a, item in _draped_meshes(site, site.get("land_cover") or [], dz, lift, sampler, terrain_points)]


def land_cover_expected_bbox(m: dict) -> dict:
    pts = m["item"]["polygonCoordinates"]
    lvl = m["item"]["level"]  # vertex z are level-relative
    zs = [p["z"] + lvl for p in pts] + [p["z"] + lvl for s in m["item"].get("sublines", []) for p in s["coordinates"]]
    return {"xMin": min(p["x"] for p in pts), "xMax": max(p["x"] for p in pts), "yMin": min(p["y"] for p in pts),
            "yMax": max(p["y"] for p in pts), "zMin": min(zs), "zMax": max(zs)}


# --------------------------------------------------------------- buildings
def block_height(b: dict) -> float:
    """Height of the extruded block: the building height minus the roof height when a Roof element is built on top."""
    h = max(float(b.get("height") or 0), MIN_BUILDING_HEIGHT)
    if b.get("roof"):
        h = max(h - float(b["roof"]["height"]), MIN_BUILDING_HEIGHT)
    return h


def building_slab(b: dict, dz: float = 0.0) -> dict:
    return _slab(b["footprint"], b.get("holes"), b["base_z"] + dz, block_height(b), "Bottom", fill=True)


def building_expected_bbox_z(b: dict, dz: float = 0.0):
    h = block_height(b)
    return (b["base_z"] + dz, b["base_z"] + dz + h)


# ------------------------------------------------------------------- roads
ROAD_BEAM_DISPLAY = {
    "storyVisibility": {"showOnHome": True, "showAllAbove": False, "showAllBelow": False, "showRelAbove": 0, "showRelBelow": 0},
    "displayOption": "OutLinesOnly",
}


def road_beams(r: dict, dz: float = 0.0, height: float = ROAD_HEIGHT) -> list[dict]:
    """One beam per consecutive point pair. TopCenter anchor, zCoordinate = top z at the begin point,
    slantAngle = atan(dz/len) so the top surface follows the road z at both ends."""
    pts = r["points"]
    width = float(r.get("width") or 3.0)
    out = []
    for a, b in zip(pts, pts[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L < MIN_SEGMENT:
            continue
        rise = b[2] - a[2]
        item = {
            "floorIndex": 0,
            "begCoordinate": _p2(a),
            "endCoordinate": _p2(b),
            "zCoordinate": _r(a[2] + dz),
            "width": _r(width),
            "height": height,
            "isWidthAndHeightLinked": False,
            "anchorPoint": "TopCenter",
            "isSlanted": abs(rise) > 1e-4,
        }
        if item["isSlanted"]:
            item["slantAngle"] = _r(math.atan2(rise, L), 6)
        out.append(item)
    return out


# ------------------------------------------------------------------- trees
# Tree representation (UI "Trees" select -> builder tree_mode). Names are AC30 library parts (checked with
# Tapir FindLibraryParts). 2D-only part is the plan symbol; fallback = light.
TREE_MODES = ("light", "2d", "full3d")
TREE_PART_CANDIDATES = {
    "light": {"broad": ["Deciduous Trees"], "needle": ["Pine Trees"]},
    "full3d": {"broad": ["Tree Model Detailed"], "needle": ["Pine Tree Model"]},
    "2d": {"broad": ["Tree Plan"], "needle": ["Tree Plan"]},
}


def parse_tree_mode(mode):
    """'light'|'2d'|'full3d'|'lib:<name>' -> (kind, lib_name|None). Unknown/empty -> ('light', None)."""
    m = (mode or "full3d").strip()
    if m.lower().startswith("lib:"):
        name = m[4:].strip()
        return ("lib", name) if name else ("full3d", None)
    m = m.lower()
    return (m, None) if m in TREE_MODES else ("full3d", None)


def tree_kind(t: dict) -> str:
    """'needle' when tags.leaf_type=needleleaved (or, without leaf_type, a conifer genus), else 'broad'."""
    tags = dict(t.get("tags") or {})
    if t.get("leaf_type") and not tags.get("leaf_type"):
        tags["leaf_type"] = t["leaf_type"]
    return "needle" if treelib.is_conifer(tags) else "broad"


def resolve_tree_parts(mode, exists):
    """-> ({'broad': name, 'needle': name}, warnings). exists(name)->bool tests loaded library parts.
    A mode without a loaded fitting part falls back to light (and records a warning); a needle part
    missing falls back to the broad part of the same mode."""
    kind, lib = parse_tree_mode(mode)
    warnings: list[str] = []

    def pick(k, which):
        for n in TREE_PART_CANDIDATES[k][which]:
            if exists(n):
                return n
        return None

    if kind == "lib":
        if exists(lib):
            return {"broad": lib, "needle": lib}, warnings
        warnings.append(f"tree_mode lib:{lib}: library part not loaded; fell back to light")
        kind = "light"
    parts = {w: pick(kind, w) for w in ("broad", "needle")}
    if parts["broad"] is None:
        if kind != "light":
            warnings.append(f"tree_mode {kind}: no fitting library part loaded; fell back to light")
        parts = {w: pick("light", w) for w in ("broad", "needle")}
    if parts["broad"] is None:
        for n in ("Tree Model Detailed", "Tree Model Simple 1", "Conceptual Tree Model"):
            if exists(n):
                parts["broad"] = n
                break
    if parts["needle"] is None:
        parts["needle"] = parts["broad"]
    return parts, warnings


def tree_lib_for(t: dict, parts: dict) -> str:
    return parts[tree_kind(t)]


def _tree_angle(t: dict) -> float:
    """Deterministic pseudo-random rotation in [0, 2*pi) around the tree vertical axis (radians)."""
    key = f"{t.get('osm_id', '')}:{float(t.get('x') or 0.0):.3f}:{float(t.get('y') or 0.0):.3f}"
    return round((zlib.crc32(key.encode("utf-8")) / 2**32) * (2.0 * math.pi), 6)


def tree_object(t: dict, lib_name: str, dz: float = 0.0) -> dict:
    h = float(t.get("height") or 8.0)
    cd = t.get("crown_diameter")
    crown = max(1.0, float(cd)) if cd else max(2.0, 0.63 * h)
    return {
        "floorIndex": 0,
        "libraryPartName": lib_name,
        "coordinates": {"x": _r(t["x"]), "y": _r(t["y"]), "z": _r(t["z"] + dz)},
        "dimensions": {"x": _r(crown), "y": _r(crown), "z": _r(h)},
        "angle": _tree_angle(t),
    }


# ------------------------------------------------------------------- shrubs + hedges
# Shrub family (AC30 Plants library, checked with Tapir GetLibraryPartScripts / GetGDLParametersOfLibraryParts):
#   "Shrubs"        3D (picture-based, bShowasBitmap) + top-view 2D; A,B = crown, ZZYZX = height; min width 1.5 m
#   "Hedge Model"   linear hedge block: A = length (x), B = width (y), zzyzx = height, origin at the box corner
#                   (x 0..A, y 0..B); `nul` = number of leaves; iDetLev3D 0 = by MVO, 1 = schematic block
#   "Tree Plan"     2D-only (3D script absent): plan symbol for tree_mode 2d, iTopViewSymb 100 = filled circle
SHRUB_PART_3D, SHRUB_PART_2D, HEDGE_PART = "Shrubs", "Tree Plan", "Hedge Model"
SHRUB_2D_PARAMS = [{"name": "iTopViewSymb", "value": 100}]
SHRUB_MIN_CROWN = 1.5          # "Shrubs" renders at least 1.5 m wide (DefSizeValues)
HEDGE_MAX_PIECE = 8.0          # m, longer hedge segments are split (leaf count is per object)
HEDGE_LEAVES_PER_M2 = 40       # leaves per m2 of hedge surface (library default: 300 per 5 m2 = 60)
HEDGE_MIN_SEGMENT = 0.05
HEDGE_SCHEMATIC = 1            # iDetLev3D: schematic block, used with tree_mode 2d (light)


def resolve_shrub_parts(mode, exists):
    """-> ({'shrub': name|None, 'hedge': name|None}, warnings). 2d mode: 2D-only 'Tree Plan' circle for shrubs
    and a schematic 'Hedge Model' (there is no 2D-only hedge part). Missing parts -> None + warning."""
    kind, _ = parse_tree_mode(mode)
    w: list[str] = []
    shrub = SHRUB_PART_2D if kind == "2d" else SHRUB_PART_3D
    if not exists(shrub):
        w.append(f"shrub part '{shrub}' not loaded: shrubs skipped")
        shrub = None
    hedge = HEDGE_PART if exists(HEDGE_PART) else None
    if hedge is None:
        w.append(f"hedge part '{HEDGE_PART}' not loaded: hedges skipped")
    return {"shrub": shrub, "hedge": hedge}, w


def shrub_object(s: dict, lib_name: str, dz: float = 0.0) -> dict:
    h = max(0.1, float(s.get("height") or 1.5))
    crown = max(SHRUB_MIN_CROWN if lib_name == SHRUB_PART_3D else 0.5, float(s.get("crown_diameter") or h))
    item = {"floorIndex": 0,
            "libraryPartName": lib_name,
            "coordinates": {"x": _r(s["x"]), "y": _r(s["y"]), "z": _r(s["z"] + dz)},
            "dimensions": {"x": _r(crown), "y": _r(crown), "z": _r(h)}}
    if lib_name == SHRUB_PART_2D:
        item["parameters"] = [dict(p) for p in SHRUB_2D_PARAMS]
    return item


def hedge_pieces(points, max_piece: float = HEDGE_MAX_PIECE):
    """Hedge polyline [[x,y,z],...] -> straight pieces [((x0,y0,z0),(x1,y1,z1)), ...], none longer than max_piece,
    none shorter than HEDGE_MIN_SEGMENT."""
    out = []
    for a, b in zip(points, points[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L < HEDGE_MIN_SEGMENT:
            continue
        n = max(1, int(math.ceil(L / max_piece - 1e-9)))
        for i in range(n):
            t0, t1 = i / n, (i + 1) / n
            out.append((tuple(a[k] + (b[k] - a[k]) * t0 for k in range(3)), tuple(a[k] + (b[k] - a[k]) * t1 for k in range(3))))
    return out


def hedge_segment_object(p0, p1, width: float, height: float, lib_name: str = HEDGE_PART, dz: float = 0.0,
                         schematic: bool = False) -> dict:
    """One Hedge Model object along p0 -> p1. Library box is x 0..A, y 0..B from the origin, so the origin goes
    half a width to the right of the start (local y = left normal). Tapir `angle` is in RADIANS (checked live)."""
    L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    th = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    w = max(0.1, float(width))
    ox, oy = p0[0] + (w / 2.0) * math.sin(th), p0[1] - (w / 2.0) * math.cos(th)
    h = max(0.1, float(height))
    params = [{"name": "nul", "value": int(min(2000, max(60, round(HEDGE_LEAVES_PER_M2 * (2 * L * h + L * w)))))}]
    if schematic:
        params.append({"name": "iDetLev3D", "value": HEDGE_SCHEMATIC})
    return {"floorIndex": 0,
            "libraryPartName": lib_name,
            "coordinates": {"x": _r(ox), "y": _r(oy), "z": _r(min(p0[2], p1[2]) + dz)},
            "dimensions": {"x": _r(L), "y": _r(w), "z": _r(h)},
            "angle": round(th, 6), "parameters": params}


def hedge_objects(h: dict, lib_name: str = HEDGE_PART, dz: float = 0.0, schematic: bool = False,
                  max_piece: float = HEDGE_MAX_PIECE) -> list[dict]:
    return [hedge_segment_object(a, b, h.get("width") or 0.8, h.get("height") or 1.2, lib_name, dz, schematic)
            for a, b in hedge_pieces(h.get("points") or [], max_piece)]


def hedge_expected_bbox(obj: dict) -> dict:
    """xy extent of the hedge rectangle + z range from the object payload (what Get3DBoundingBoxes should give)."""
    o, d, th = obj["coordinates"], obj["dimensions"], obj["angle"]
    c, s_ = math.cos(th), math.sin(th)
    pts = [(o["x"] + a * c - b * s_, o["y"] + a * s_ + b * c) for a in (0.0, d["x"]) for b in (0.0, d["y"])]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return {"xMin": min(xs), "xMax": max(xs), "yMin": min(ys), "yMax": max(ys), "zMin": o["z"], "zMax": o["z"] + d["z"]}


# ------------------------------------------------------------------- water
def water_slab(w: dict, dz: float = 0.0, thickness: float = WATER_THICKNESS) -> dict:
    return _slab(w["polygon"], w.get("holes"), w["z"] + dz, thickness, "Top", fill=True)


# ------------------------------------------------------------ more features (M6)
# railways -> Beams (road_beams, width per track); platforms / ground / piers -> thin Slabs; walls -> Wall tool;
# fences + guard rails -> Railing tool (thin Wall fallback); building parts -> Slabs; roofs -> Roof tool.
ROOF_THICKNESS = 0.2
GROUND_LIFT = 0.02          # m, ground slab top above the highest terrain point under it
BARRIER_MIN_SEGMENT = 0.05
POWER_COLUMN = {"tower": 1.2, "pole": 0.3}   # m, column cross-section by kind


def _slab(polygon, holes, level, thickness, ref, fill: bool = False) -> dict:
    """CreateSlabs item (Basic structure, floor 0). fill=True adds floorFill {"use": True}."""
    item = {"floorIndex": 0, "level": _r(level), "thickness": _r(thickness), "structureType": "Basic",
            "referencePlaneLocation": ref}
    if fill:
        item["floorFill"] = {"use": True}
    item["polygonCoordinates"] = [_p2(p) for p in _open_ring(polygon)]
    hs = [{"polygonCoordinates": [_p2(p) for p in _open_ring(h_)]} for h_ in (holes or []) if len(h_) >= 3]
    if hs:
        item["holes"] = hs
    return item


def _part_bottom_thickness(p: dict, dz: float):
    mn = float(p.get("min_height") or 0)
    h = max(float(p["height"]) - mn - (float(p["roof"]["height"]) if p.get("roof") else 0.0), MIN_BUILDING_HEIGHT)
    return p["base_z"] + mn + dz, h


def part_slab(p: dict, dz: float = 0.0) -> dict:
    """Building part: bottom = base + min_height, thickness up to the top (minus the roof when one is built)."""
    bottom, h = _part_bottom_thickness(p, dz)
    return _slab(p["footprint"], p.get("holes"), bottom, h, "Bottom", fill=True)


def part_expected_bbox_z(p: dict, dz: float = 0.0):
    bottom, h = _part_bottom_thickness(p, dz)
    return (_r(bottom), _r(bottom) + _r(h))


def platform_slab(p: dict, dz: float = 0.0) -> dict:
    return _slab(p["polygon"], p.get("holes"), p["base_z"] + dz, p["thickness"], "Bottom")


def pier_slab(p: dict, dz: float = 0.0) -> dict:
    return _slab(p["polygon"], p.get("holes"), p["z"] + dz, p["thickness"], "Top")


def ground_is_slab(g: dict) -> bool:
    return not g.get("sloped")


def ground_slab(g: dict, dz: float = 0.0, thickness: float = 0.05) -> dict:
    """Flat ground: thin slab, top = highest terrain under it + GROUND_LIFT."""
    return _slab(g["polygon"], g.get("holes"), g["z_max"] + GROUND_LIFT + dz, thickness, "Top", fill=True)


def _sloped_ground(site: dict) -> list[dict]:
    return [g for g in site.get("ground_surfaces", []) if not ground_is_slab(g)]


def ground_meshes(site: dict, dz: float = 0.0, terrain_points: str = "points", sampler=None) -> list[dict]:
    """Sloped ground surfaces -> draped meshes (land-cover mechanism): [{osm_id, kind, material, item}]."""
    sloped = _sloped_ground(site)
    if not sloped:
        return []
    _check_point_mode(terrain_points)
    return [{"osm_id": g.get("osm_id"), "kind": g.get("kind"), "material": g.get("material") or "Soil", "item": item}
            for g, item in _draped_meshes(site, sloped, dz, LAND_COVER_LIFT, sampler or TerrainSampler(site),
                                          terrain_points)]


def barrier_segments(bar: dict, dz: float = 0.0) -> list[dict]:
    """One entry per point pair: {a, b, base, height, spread} with base = lower end z (+dz) and height = tag/default
    height + the z difference of the ends, so the top is >= `height` above the highest end."""
    pts, out = bar["points"], []
    for a, b in zip(pts, pts[1:]):
        if math.hypot(b[0] - a[0], b[1] - a[1]) < BARRIER_MIN_SEGMENT:
            continue
        spread = abs(b[2] - a[2])
        out.append({"a": a, "b": b, "base": min(a[2], b[2]) + dz, "spread": spread,
                    "height": float(bar["height"]) + spread})
    return out


def wall_items(bar: dict, dz: float = 0.0) -> list[dict]:
    """Wall tool (CreateWalls): Basic wall, reference line centred, base on the terrain."""
    return [{"begCoordinate": _p2(s["a"]), "endCoordinate": _p2(s["b"]), "zCoordinate": _r(s["base"]),
             "height": _r(s["height"]), "thickness": _r(bar["thickness"]), "structureType": "Basic",
             "referenceLineLocation": "Center", "floorIndex": 0} for s in barrier_segments(bar, dz)]


def railing_items(bar: dict, dz: float = 0.0) -> list[dict]:
    """Railing tool (CreateRailings): one railing per <= barrier_max_segment piece; the look (posts/rails/panels) is the
    Railing tool default of the open project, only height + base are set."""
    return [{"coordinates": [_p2(s["a"]), _p2(s["b"])], "floorIndex": 0, "bottomOffset": _r(s["base"]),
             "height": _r(s["height"])} for s in barrier_segments(bar, dz)]


def barrier_expected_z(item: dict, kind: str):
    base = item["zCoordinate"] if kind == "wall" else item["bottomOffset"]
    return base, base + item["height"]


def roof_items(owner: dict, dz: float = 0.0) -> list[dict]:
    """Roof tool items for a building / building part that carries `roof` ({shape, height, ...}):
    [{"item": CreateRoofs RoofData, "expected": bbox dict, "check_z": bool}]. skillion = one single-plane roof (pivot on
    the eave, rises over the footprint depth), gabled = two single-plane roofs (pivot on the ridge, falling away),
    hipped / pyramidal = one multi-plane roof (one slope angle, eaves overhang 0). Roof level = underside at the
    pivot / eave (verified live, AC30)."""
    r = owner.get("roof")
    if not r:
        return []
    if r.get("planes") is not None:
        return _lod2_roof_items(owner, r, dz)
    poly = Polygon(owner["footprint"])
    top = owner["base_z"] + float(owner["height"]) - float(r["height"]) + dz   # eave z = top of the block
    out = []
    for pl in feat.roof_planes(poly, r):
        ang = pl["angle"]
        item = {"polygonCoordinates": [_p2(q) for q in pl["polygon"]], "thickness": ROOF_THICKNESS,
                "structureType": "Basic", "floorIndex": 0}
        if pl["kind"] == "single":
            item.update(level=_r(top + pl["pivot_rise"]), angle=_r(ang, 6),
                        pivotLine={"begCoordinate": _p2(pl["pivot"][0]), "endCoordinate": _p2(pl["pivot"][1])})
        else:
            item.update(level=_r(top), eavesOverhang=0.0,   # eavesOverhang is multi-plane only (single-plane rejects it)
                        levels=[{"levelHeight": _r(r["height"]), "levelAngle": _r(ang, 6)}])
        xs = [q[0] for q in pl["polygon"]]
        ys = [q[1] for q in pl["polygon"]]
        zmax = top + float(r["height"]) + ROOF_THICKNESS / math.cos(ang)
        rect = False
        if pl["kind"] == "multi":
            pg = Polygon(pl["polygon"])
            rect = pg.area > 0 and abs(pg.area - pg.minimum_rotated_rectangle.area) < 0.01 * pg.area
        out.append({"item": item, "check_z": pl["kind"] == "single" or rect,
                    "expected": {"xMin": min(xs), "xMax": max(xs), "yMin": min(ys), "yMax": max(ys), "zMax": zmax}})
    return out


# ------------------------------------------------------------ LoD2 buildings (M10 step 7)
# Input contract: osmctx/features.py module doc. Mapping (user decision 2026-11-24): Slab block(s) from the ground up to
# the eave + one single-plane Roof (Roof tool) per planar roof surface; Morph only for surfaces the Roof tool cannot
# take (near-vertical / non-planar). lod2_expand() turns LoD2 building records into the M6 shapes the builder already
# handles (buildings + building_parts with skip_block, roof dicts), so build_ac / verify / group_plan stay generic.
LOD2_MAX_ROOFS = 400        # default cap on LoD2 Roof + Morph elements per build (recon: ~550 on a dense 300 m site;
                            # M6: Archicad hung at 753 beams + roofs). Setting `lod2_max_roofs`, < 0 = no cap
LOD2_MODES = ("roofs", "blocks")   # blocks = one flat volume-equivalent Slab per LoD2 building (super-light)
LOD2_BLOCK_TOL = ROOF_THICKNESS    # m: roof surfaces whose block heights differ by <= this share one block at the
                                   # lowest height (the Roof's T/cos underside closes the gap)
LOD2_MIN_REGION = 1.0       # m2: block pieces smaller than this are merged into the main block / dropped (slivers)
LOD2_MORPH_EDGES = "HardVisible"


def _polygonal(g, min_area: float = 1e-4):
    """Polygon / MultiPolygon part of any geometry (GeometryCollection from make_valid / overlays), or None."""
    if g is None or g.is_empty:
        return None
    if not g.is_valid:
        g = shapely.make_valid(g)
    parts = [p for p in shapely.get_parts(g) if p.geom_type in ("Polygon", "MultiPolygon")]
    parts = [q for p in parts for q in shapely.get_parts(p) if q.area >= min_area]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else shapely.MultiPolygon(parts)


def _poly(ring, holes=()):
    try:
        return _polygonal(Polygon([tuple(p[:2]) for p in ring],
                                  [[tuple(q[:2]) for q in h] for h in (holes or []) if len(h) >= 3]))
    except (ValueError, TypeError, shapely.errors.GEOSException):
        return None


def _lod2_planes_of(src) -> list:
    """Contract PLANEs of a building / part / lod2 block: roof_planes | planes | roof.planes, plus flat_roofs (the
    osmctx/lod2.py record keeps flat surfaces apart)."""
    if not isinstance(src, dict):
        return []
    out = []
    for key in ("roof_planes", "planes"):
        if isinstance(src.get(key), list):
            out = list(src[key])
            break
    else:
        r = src.get("roof")
        if isinstance(r, dict) and isinstance(r.get("planes"), list):
            out = list(r["planes"])
    if isinstance(src.get("flat_roofs"), list):
        out += list(src["flat_roofs"])
    return out


def _lod2_units(b: dict) -> tuple[list[dict], bool]:
    """(units, has_parts). Unit = {id, footprint, holes, min_h, planes, fallback_h}: the building itself, or one per
    LoD2 BuildingPart. Building-level planes are used only when no part carries planes of its own (else they are a
    duplicate summary); with several parts they go to the part they overlap most."""
    m = b.get("lod2") or {}
    parts = [p for p in (m.get("parts") or []) if isinstance(p, dict) and len(p.get("footprint") or []) >= 3]
    part_planes = any(_lod2_planes_of(p) for p in parts)
    bplanes = [] if part_planes else (_lod2_planes_of(b) or _lod2_planes_of(m))
    fallback = float(b.get("height") or 0.0)
    if not parts or (len(parts) == 1 and float(parts[0].get("min_h") or 0.0) < 0.05):
        planes = bplanes + (_lod2_planes_of(parts[0]) if parts else [])
        return [{"id": str(b["osm_id"]), "footprint": b["footprint"], "holes": b.get("holes") or [], "min_h": 0.0,
                 "planes": planes, "fallback_h": fallback}], False
    units = [{"id": str(p.get("id") or f"{b['osm_id']}#part{k}"), "footprint": p["footprint"],
              "holes": p.get("holes") or [], "min_h": float(p.get("min_h") or 0.0), "planes": _lod2_planes_of(p),
              "fallback_h": float(p.get("top_h") or p.get("eave_h") or fallback)} for k, p in enumerate(parts)]
    if bplanes:
        polys = [_poly(u["footprint"], u["holes"]) for u in units]
        for pl in bplanes:
            pts = feat.plane_points(pl)
            q = _poly(pts) if pts and len(pts) >= 3 else None
            best, j = -1.0, 0
            for i, up in enumerate(polys):
                if up is None or q is None:
                    continue
                ov = up.intersection(q).area
                if ov > best:
                    best, j = ov, i
            units[j]["planes"].append(pl)
    return units, True


def _lod2_unit_plan(u: dict) -> dict:
    """Classify the unit's planes and split its footprint into Slab blocks: each plane claims its plan polygon at its
    block height (single / morph: lowest edge = eave; flat: the surface itself = block top); heights within
    LOD2_BLOCK_TOL share one block (lowest height); higher blocks win overlaps (dormers); the largest block also takes
    the footprint no plane covers. -> {cls [(record, plane)], singles, morphs, blocks [(Polygon|None, h)], top}."""
    cls = []
    for p in u["planes"]:
        cls.append((feat.classify_plane(p, ROOF_THICKNESS), p))
    singles = [p for r, p in cls if r["kind"] == "single"]
    morphs = [p for r, p in cls if r["kind"] == "morph"]
    live = [r for r, _p in cls if r["kind"] != "skip"]
    top = max((r["z_max"] for r in live), default=u["fallback_h"])
    floor_h = u["min_h"] + MIN_BUILDING_HEIGHT
    fp = _poly(u["footprint"], u["holes"])
    hs = []
    for r in live:
        h = r["z_max"] if r["kind"] == "flat" else r["z_min"]
        g = _poly(r["polygon"]) if len(r.get("polygon") or []) >= 3 else None
        hs.append((h, g))
    regional = sorted([(h, g) for h, g in hs if g is not None], key=lambda t: t[0])
    clusters: list = []
    for h, g in regional:
        if clusters and h - clusters[-1][0] <= LOD2_BLOCK_TOL:
            clusters[-1][1].append(g)
        else:
            clusters.append([h, [g]])
    if not hs:
        blocks = [(fp, max(u["fallback_h"], floor_h))]
    elif fp is None or len(clusters) <= 1:
        blocks = [(fp, max(min(h for h, _g in hs), floor_h))]
    else:
        regs, taken = [], None
        for h, gs in reversed(clusters):                     # highest first: it takes the overlaps
            reg = shapely.union_all(gs).intersection(fp)
            if taken is not None:
                reg = reg.difference(taken)
            reg = _polygonal(reg)
            if reg is None:
                continue
            taken = reg if taken is None else taken.union(reg)
            regs.append([h, reg])
        if not regs:
            return {"cls": cls, "singles": singles, "morphs": morphs, "top": max(top, floor_h),
                    "blocks": [(fp, max(min(h for h, _g in hs), floor_h))]}
        main = max(range(len(regs)), key=lambda i: regs[i][1].area)
        small = [i for i, (_h, g) in enumerate(regs) if g.area < LOD2_MIN_REGION and i != main]
        rest = fp.difference(taken)
        regs[main][1] = _polygonal(shapely.union_all([regs[main][1], rest] + [regs[i][1] for i in small])) or regs[main][1]
        regs = [r_ for i, r_ in enumerate(regs) if i not in small]
        blocks = [(g, max(h, floor_h)) for h, g in regs]
        if len(blocks) == 1:
            blocks = [(fp, blocks[0][1])]
    return {"cls": cls, "singles": singles, "morphs": morphs, "blocks": blocks, "top": max(top, floor_h)}


def _lod2_roof(top, h, singles, morphs):
    th = _r(max(top, h), 3)
    return th, {"shape": "lod2", "height": _r(th - _r(h, 3), 3), "planes": singles, "morphs": morphs}


def _lod2_emit(b: dict, units_plans, has_parts: bool) -> tuple[dict, list[dict]]:
    """Full LoD2 building -> (building record, extra part records) in M6 shapes."""
    if not has_parts and len(units_plans) == 1 and len(units_plans[0][1]["blocks"]) == 1:
        _u, pl = units_plans[0]
        h = pl["blocks"][0][1]
        nb = dict(b)
        if pl["singles"] or pl["morphs"]:
            nb["height"], nb["roof"] = _lod2_roof(pl["top"], h, pl["singles"], pl["morphs"])
        else:
            nb.pop("roof", None)
            nb["height"] = _r(h, 3)
        return nb, []
    m = b.get("lod2") or {}
    nb = dict(b, skip_block=True)
    nb.pop("roof", None)
    parts = []
    for u, pl in units_plans:
        pieces = []
        for g, h in pl["blocks"]:
            if g is None:
                g = _poly(u["footprint"], u["holes"]) or Polygon([tuple(p[:2]) for p in u["footprint"]])
            got = [p for p in shapely.get_parts(g) if p.area >= LOD2_MIN_REGION] or \
                  ([max(shapely.get_parts(g), key=lambda p: p.area)] if len(pl["blocks"]) == 1 else [])
            pieces += [(p, h) for p in got]
        pieces.sort(key=lambda t: -t[0].area)
        for k, (p, h) in enumerate(pieces):
            ext = [[_r(x, 3), _r(y, 3)] for x, y in list(p.exterior.coords)[:-1]]
            holes = [[[_r(x, 3), _r(y, 3)] for x, y in list(r.coords)[:-1]] for r in p.interiors]
            rec = {"osm_id": u["id"] if k == 0 else f"{u['id']}#b{k}", "parent": b["osm_id"], "footprint": ext,
                   "holes": holes, "base_z": b["base_z"], "min_height": _r(u["min_h"], 3), "height": _r(h, 3),
                   "height_source": "lod2_measured", "source": b.get("source"),
                   "lod2": {"dataset": m.get("dataset"), "region": m.get("region"), "unit": u["id"]}, "tags": {}}
            if k == 0 and (pl["singles"] or pl["morphs"]):      # the largest block carries the unit's roof elements
                rec["height"], rec["roof"] = _lod2_roof(pl["top"], h, pl["singles"], pl["morphs"])
            parts.append(rec)
    nb["parts"] = [p["osm_id"] for p in parts]
    return nb, parts


def _lod2_block_height(b: dict, units_plans) -> float:
    """Volume-equivalent flat block height: area-weighted plane height at each surface's plan centroid (exact volume
    under planar roofs that tile the footprint), else the record height."""
    num = den = 0.0
    for _u, pl in units_plans:
        for r, _p in pl["cls"]:
            a = float(r.get("area_xy") or 0.0)
            if r["kind"] in ("single", "flat"):
                num, den = num + a * r["z_centroid"], den + a
            elif r["kind"] == "morph" and a > 0:
                num, den = num + a * float(np.mean(np.asarray(r["polygon3d"])[:, 2])), den + a
    h = num / den if den > 0.5 else float(b.get("height") or 0.0)
    return _r(max(h, MIN_BUILDING_HEIGHT), 3)


def lod2_settings(site: dict | None = None, max_roofs=None, mode: str | None = None) -> tuple[int | None, str]:
    """(cap, mode): arguments > converter settings in site.report.defaults_used.settings (`lod2_max_roofs`,
    `lod2_mode`, `include.lod2_roofs` False = blocks) > defaults (LOD2_MAX_ROOFS, "roofs"). cap None = no cap."""
    s = ((((site or {}).get("report") or {}).get("defaults_used") or {}).get("settings")) or {}
    cap = max_roofs if max_roofs is not None else s.get("lod2_max_roofs")
    cap = LOD2_MAX_ROOFS if cap is None else int(cap)
    if mode is None:
        mode = s.get("lod2_mode") or ("blocks" if (s.get("include") or {}).get("lod2_roofs") is False else "roofs")
    if mode not in LOD2_MODES:
        raise ValueError(f"unknown lod2 mode {mode!r} {LOD2_MODES}")
    return (cap if cap >= 0 else None), mode


def lod2_expand(site: dict, max_roofs=None, mode: str | None = None) -> tuple[dict, dict]:
    """LoD2 building records -> the M6 shapes the builder handles. Returns (site', info); site' is a shallow copy
    (input untouched; the same object when no LoD2 building is present) with:
      * one-block buildings: `height` = top, `roof` = {"shape": "lod2", "height": top - eave, "planes": [single-plane
        PLANEs], "morphs": [Morph PLANEs]} so block_height() = eave and roof_items() builds one Roof per plane;
      * buildings with LoD2 BuildingParts or several eave heights: `skip_block` + `parts`, one building_parts record per
        block (bottom = base + min_h, top = block height; the largest block of each unit carries its roof);
      * degraded buildings (mode "blocks", or the cap: projected Roof + Morph count > cap, most expensive buildings
        first): one flat block at the volume-equivalent height, no roof (`lod2_degraded`).
    info (also site'["lod2_plan"], so a second call is a no-op) = {buildings, mode, max_roofs, roofs, morphs, blocks,
    parts, degraded, roofs_uncapped, over_cap, planes {single, flat, morph, skip}, morph_reasons, datasets, warnings}."""
    if site.get("lod2_plan") is not None:
        return site, site["lod2_plan"]
    blds = site.get("buildings") or []
    idx = [i for i, b in enumerate(blds) if feat.is_lod2(b)]
    cap, mode = lod2_settings(site, max_roofs, mode)
    info = {"buildings": len(idx), "mode": mode, "max_roofs": cap, "roofs": 0, "morphs": 0, "blocks": 0, "parts": 0,
            "degraded": 0, "roofs_uncapped": 0, "over_cap": False,
            "planes": {"single": 0, "flat": 0, "morph": 0, "skip": 0}, "morph_reasons": {}, "datasets": [],
            "warnings": []}
    if not idx:
        return site, info
    an = {}
    for i in idx:
        units, has_parts = _lod2_units(blds[i])
        ups = [(u, _lod2_unit_plan(u)) for u in units]
        an[i] = (ups, has_parts, sum(len(pl["singles"]) + len(pl["morphs"]) for _u, pl in ups))
        for _u, pl in ups:
            for r, _p in pl["cls"]:
                info["planes"][r["kind"]] += 1
                if r["kind"] == "morph":
                    why = str(r.get("reason", "")).split(" ")[0]
                    info["morph_reasons"][why] = info["morph_reasons"].get(why, 0) + 1
    total = sum(c for _ups, _hp, c in an.values())
    info["roofs_uncapped"] = total
    degrade: set = set()
    if mode == "blocks":
        degrade = set(idx)
    elif cap is not None and total > cap:
        info["over_cap"] = True
        left = total
        for i in sorted(idx, key=lambda i: (-an[i][2], str(blds[i].get("osm_id")))):
            if left <= cap or an[i][2] == 0:
                break
            degrade.add(i)
            left -= an[i][2]
        info["warnings"].append(f"LoD2: {total} roof elements (Roof + Morph) exceed lod2_max_roofs={cap}; "
                                f"{len(degrade)} building(s) with the most roof surfaces built as flat blocks at the "
                                f"volume-equivalent height instead")
    if info["planes"]["skip"]:
        info["warnings"].append(f"LoD2: {info['planes']['skip']} roof surface(s) skipped (degenerate, tiny or no z)")
    new_b, new_p = [], []
    for i, b in enumerate(blds):
        if i not in an:
            new_b.append(b)
            continue
        ups, has_parts, cost = an[i]
        if i in degrade:
            nb = dict(b, height=_lod2_block_height(b, ups), lod2_degraded="blocks mode" if mode == "blocks" else "cap")
            for k in ("roof", "skip_block", "parts"):
                nb.pop(k, None)
            new_b.append(nb)
            info["blocks"] += 1
            continue
        nb, parts = _lod2_emit(b, ups, has_parts)
        new_b.append(nb)
        new_p += parts
        info["roofs"] += sum(len(pl["singles"]) for _u, pl in ups)
        info["morphs"] += sum(len(pl["morphs"]) for _u, pl in ups)
        info["blocks"] += len(parts) if nb.get("skip_block") else 1
    info["degraded"] = len(degrade) if mode == "roofs" else 0
    info["parts"] = len(new_p)
    info["datasets"] = sorted({str((blds[i].get("lod2") or {}).get("dataset") or blds[i].get("source")) for i in idx})
    out = dict(site, buildings=new_b, lod2_plan=info)
    out["building_parts"] = list(site.get("building_parts") or []) + new_p
    return out, info


def lod2_counts(site_or_buildings, max_roofs=None, mode: str | None = None) -> dict:
    """Projected LoD2 element counts for limits.estimate (pure, no Archicad): {buildings, blocks (Slabs), roofs (Roof
    tool), morphs, elements = blocks + roofs + morphs, degraded, roofs_uncapped, over_cap, max_roofs, mode}. Takes a
    site dict or a list of building records (contract in features.py)."""
    site = site_or_buildings if isinstance(site_or_buildings, dict) else {"buildings": list(site_or_buildings or [])}
    _s, info = lod2_expand(site, max_roofs=max_roofs, mode=mode)
    keys = ("buildings", "blocks", "roofs", "morphs", "degraded", "roofs_uncapped", "over_cap", "max_roofs", "mode")
    out = {k: info[k] for k in keys}
    out["elements"] = info["blocks"] + info["roofs"] + info["morphs"]
    return out


def _lod2_roof_items(owner: dict, r: dict, dz: float) -> list[dict]:
    """roof_items for a LoD2 owner: one single-plane Roof per "single" plane (a 0-degree Roof for a "flat" plane that
    reaches here un-expanded); "morph" / "skip" planes are not Roofs (see lod2_morph_items). z: base_z + plane z."""
    ground = float(owner["base_z"]) + dz
    out = []
    for pl in feat.roof_planes(None, r, ROOF_THICKNESS):
        if pl["kind"] not in ("single", "flat"):
            continue
        item = {"polygonCoordinates": [_p2(q) for q in pl["polygon"]], "thickness": ROOF_THICKNESS,
                "structureType": "Basic", "floorIndex": 0, "level": _r(ground + pl["level"]),
                "angle": _r(pl["angle"], 6),
                "pivotLine": {"begCoordinate": _p2(pl["pivot"][0]), "endCoordinate": _p2(pl["pivot"][1])}}
        xs = [q[0] for q in pl["polygon"]]
        ys = [q[1] for q in pl["polygon"]]
        out.append({"item": item, "check_z": True,
                    "expected": {"xMin": min(xs), "xMax": max(xs), "yMin": min(ys), "yMax": max(ys),
                                 "zMax": ground + pl["z_max"]}})
    return out


def lod2_morph_items(owner: dict, dz: float = 0.0) -> list[dict]:
    """Morph fallback (Tapir CreateMorphs, same body format as terrain_morph) for the LoD2 roof surfaces the Roof tool
    cannot take: [{"item", "expected" bbox, "check_z"}]. One Surface Morph per surface: a single face when planar
    (near-vertical case), triangulated in its best-fit plane when not. basePoint = (min x, min y, base_z + dz),
    vertices local (z = height above the LoD2 ground). Uses roof["morphs"] (lod2_expand) or classifies roof["planes"]."""
    r = owner.get("roof") or {}
    if "morphs" in r:
        polys = [feat.plane_points(p) for p in r.get("morphs") or []]
    elif r.get("planes") is not None:
        polys = [feat.plane_points(p) for p in r["planes"] if feat.classify_plane(p, ROOF_THICKNESS)["kind"] == "morph"]
    else:
        return []
    ground = float(owner["base_z"]) + dz
    out = []
    for pts in polys:
        if not pts or len(pts) < 3:
            continue
        P = feat._dedupe_ring(np.asarray(pts, float)[:, :3])
        if len(P) < 3:
            continue
        rec = feat.roof_plane_from_polygon(P, ROOF_THICKNESS)
        if rec["kind"] == "skip":
            continue
        if rec["deviation"] <= feat.LOD2_PLANAR_TOL:
            faces = [{"vertexIds": list(range(len(P)))}]
        else:
            faces = [{"vertexIds": list(t)} for t in feat.triangulate_ring_3d(P)]
        if not faces:
            continue
        bx, by = float(P[:, 0].min()), float(P[:, 1].min())
        item = {"basePoint": {"x": _r(bx), "y": _r(by), "z": _r(ground)},
                "body": {"bodyType": "Surface", "edgeDefault": LOD2_MORPH_EDGES,
                         "vertices": [{"x": _r(x - bx), "y": _r(y - by), "z": _r(z)} for x, y, z in P],
                         "polygons": faces, "wireEdges": []}}
        out.append({"item": item, "check_z": True, "reason": rec.get("reason"),
                    "expected": {"xMin": bx, "xMax": float(P[:, 0].max()), "yMin": by, "yMax": float(P[:, 1].max()),
                                 "zMin": ground + float(P[:, 2].min()), "zMax": ground + float(P[:, 2].max())}})
    return out


def power_column(t: dict, dz: float = 0.0) -> dict:
    w = POWER_COLUMN.get(t.get("kind"), 0.3)
    return {"coordinates": {"x": _r(t["x"]), "y": _r(t["y"]), "z": _r(t["z"] + dz)}, "height": _r(t["height"]),
            "width": w, "depth": w, "circleBased": t.get("kind") == "pole", "isWidthAndHeightLinked": True,
            "floorIndex": 0}


def active_buildings(site: dict) -> list[dict]:
    """Buildings whose block is built (parts-covered outlines carry skip_block)."""
    return [b for b in site.get("buildings", []) if not b.get("skip_block")]


def feature_expected(site: dict, fence_tool: str = "railing") -> dict:
    """Elements the more-features plan creates, per kind (+ "building_morphs" when LoD2 buildings are present; the
    LoD2 records are expanded with lod2_expand first, as the builder does)."""
    has_lod2 = any(feat.is_lod2(b) for b in site.get("buildings", []))
    if has_lod2:
        site, _info = lod2_expand(site)
    bars = site.get("barriers", [])
    walls = sum(len(barrier_segments(b)) for b in bars if b["tool"] == "wall" or fence_tool == "wall")
    rails = sum(len(barrier_segments(b)) for b in bars if b["tool"] == "railing" and fence_tool != "wall")
    g = site.get("ground_surfaces", [])
    out = {"building_parts": len(site.get("building_parts", [])),
            "roofs": sum(len(roof_items(o)) for o in active_buildings(site) + site.get("building_parts", [])),
            "railways": sum(len(road_beams(r)) for r in site.get("railways", [])),
            "rail_platforms": len(site.get("railway_platforms", [])),
            "walls": walls, "railings": rails,
            "ground_slabs": sum(1 for x in g if ground_is_slab(x)),
            "ground_meshes": len(_cover_parts(site, _sloped_ground(site))),
            "piers": len(site.get("piers", [])), "power": len(site.get("power_towers", []))}
    if has_lod2:
        out["building_morphs"] = sum(len(lod2_morph_items(o))
                                     for o in active_buildings(site) + site.get("building_parts", []))
    return out


# ------------------------------------------------------------------- plan
def expected_counts(site: dict) -> dict:
    """Elements the plan will create, per type."""
    return {
        "Mesh": 1 + len(_cover_parts(site, site.get("land_cover") or [])),
        "Slab:buildings": len(active_buildings(site)),
        "Beam": sum(len(road_beams(r)) for r in site.get("roads", [])),
        "Object": len(site.get("trees", [])) + len(site.get("shrubs", [])) +
                  sum(len(hedge_objects(h)) for h in site.get("hedges", [])),
        "Slab:water": len(site.get("water", [])),
    }


def geo_location_payload(site: dict, crs_name: str | None = None) -> dict:
    o = site["origin"]
    epsg = o.get("epsg")
    return {
        "projectLocation": {"longitude": o["lon"], "latitude": o["lat"], "altitude": o.get("elevation", 0.0)},
        "surveyPoint": {
            "position": {"eastings": o["easting"], "northings": o["northing"], "elevation": o.get("elevation", 0.0)},
            "positionInProject": {"x": 0.0, "y": 0.0, "z": 0.0},
            "geoReferencingParameters": {"crsName": crs_name or (f"EPSG:{epsg}" if epsg else "")},
        },
    }


# ------------------------------------------------------- solid element ops
SEO_OPERATIONS = ("Subtraction", "SubtractionUpwards", "SubtractionDownwards", "Intersection", "Addition")


def seo_links(target_guid: str, operator_guids: list[str], operation: str = "SubtractionUpwards") -> list[dict]:
    """CreateSolidElementLinks items: terrain mesh = target, water slabs / road beams = operators."""
    if operation not in SEO_OPERATIONS:
        raise ValueError(f"unknown solid operation {operation!r}")
    return [{"targetId": {"guid": target_guid}, "operatorId": {"guid": g}, "operation": operation,
             "linkFlags": {"inheritOperatorAttributes": False, "skipPolygonHoles": False}}
            for g in operator_guids]


# ------------------------------------------------------------------ groups
TOP_GROUP = "OSM Site Context"


def group_plan(created: dict, top: str = TOP_GROUP):
    """Nested group tree from {kind: [(guid, osm_id), ...]} (kinds: terrain, buildings, roads, trees, water).

    Node = {"name": str, "children": [guid_str | node, ...]}. Groups need >= 2 items, so a node with
    one child is dissolved into its parent; the top group stays (None if fewer than 2 top-level items).
      top
        terrain mesh (element), water slab (element) or "Water" group if >= 2
        "Buildings" group
        "Roads" group -> "Road <osm_id>" subgroup per road with >= 2 beams, single beams directly
        "Trees" group -> "<Forest|Mangrove|Orchard|Nursery|Park|Cemetery> <osm_id>" subgroup per scattered area
                         (ids in created["tree_area_ids"], types in created["tree_area_types"])
        "Shrubs" group -> "Shrub area <osm_id>" subgroup per shrubbery/scrub area with >= 2 shrubs
        "Hedges" group -> "Hedge <osm_id>" subgroup per hedge way with >= 2 segments
        "Railways" group -> "Railway <osm_id>" subgroup per way with >= 2 beams, platform slabs directly
        "Walls & fences" group -> "Barrier <osm_id>" subgroup per way with >= 2 segments
        "Ground surfaces" / "Piers" / "Power" groups (flat; ground = slabs + draped meshes)
        "Buildings" group holds a "Buildings (Overture)" subgroup for the Overture Maps gap-fill buildings (ids
                         "overture/<uuid>") when there are >= 2
        "Buildings" group also holds a "Building <osm_id>" subgroup per building with parts and/or a roof
                         (created kinds building_parts / building_roofs / building_morphs (LoD2 Morph fallback),
                         side map building_parent: part id -> parent id)
        "Contours" group -> "Contours minor" / "Contours major" / "Contour labels" subgroups (2D; created kinds
                         contours_minor|contours_major|contour_labels; subgroups with 1 element dissolve)
      (Shrubs and Hedges are siblings of Trees, not inside it.)
    """
    def collapse(node):
        kids = []
        for k in node["children"]:
            if isinstance(k, dict):
                k = collapse(k)
                if len(k["children"]) == 1:
                    k = k["children"][0]
            kids.append(k)
        return {"name": node["name"], "children": kids}

    def guids(key):
        return [g for g, _ in created.get(key, [])]

    def by_osm(key):
        d: dict = {}
        for g, osm in created.get(key, []):
            d.setdefault(osm, []).append(g)
        return d

    road_kids = [{"name": f"Road {osm}", "children": gs} for osm, gs in by_osm("roads").items()]
    top_kids = guids("terrain")
    top_kids.append({"name": "Land cover", "children": guids("land_cover")})
    top_kids.append({"name": "Water", "children": guids("water")})
    # building parts + roofs: one subgroup per building ("Building <osm_id>", the block slab joins it)
    parent = created.get("building_parent") or {}
    sub: dict = {}
    for g, osm in list(created.get("building_parts", [])) + list(created.get("building_roofs", [])) + \
            list(created.get("building_morphs", [])):
        sub.setdefault(parent.get(osm, osm), []).append(g)
    bld_kids, ov_kids = [], []
    for g, osm in created.get("buildings", []):
        if osm in sub:
            sub[osm].insert(0, g)
        elif is_overture(osm):
            ov_kids.append(g)       # Overture gap-fill buildings: own subgroup (dissolved when < 2)
        else:
            bld_kids.append(g)
    bld_kids += [{"name": f"Building {k}", "children": gs} for k, gs in sub.items()]
    if ov_kids:
        bld_kids.append({"name": "Buildings (Overture)", "children": ov_kids})
    top_kids.append({"name": "Buildings", "children": bld_kids})
    top_kids.append({"name": "Roads", "children": road_kids})

    def per_way(kind_keys, label):
        ways: dict = {}
        for key in kind_keys:
            for osm, gs in by_osm(key).items():
                ways.setdefault(osm, []).extend(gs)
        return [{"name": f"{label} {osm}", "children": gs} for osm, gs in ways.items()]

    top_kids.append({"name": "Railways", "children": per_way(("railways",), "Railway") + guids("rail_platforms")})
    top_kids.append({"name": "Walls & fences", "children": per_way(("walls", "railings"), "Barrier")})
    top_kids.append({"name": "Ground surfaces", "children": guids("ground")})
    top_kids.append({"name": "Piers", "children": guids("piers")})
    top_kids.append({"name": "Power", "children": guids("power")})
    forest_ids = set(created.get("tree_area_ids") or ())
    area_types = created.get("tree_area_types") or {}   # osm id -> area type (forest|mangrove|park|...), default forest
    tree_kids, forests = [], {}
    for g, osm in created.get("trees", []):
        if osm in forest_ids:
            forests.setdefault(osm, []).append(g)
        else:
            tree_kids.append(g)
    tree_kids += [{"name": f"{AREA_GROUP_LABEL.get(area_types.get(osm, 'forest'), 'Forest')} {osm}", "children": gs}
                  for osm, gs in forests.items()]
    top_kids.append({"name": "Trees", "children": tree_kids})
    area_ids = set(created.get("shrub_area_ids") or ())
    shrub_kids, areas = [], {}
    for g, osm in created.get("shrubs", []):
        if osm in area_ids:
            areas.setdefault(osm, []).append(g)
        else:
            shrub_kids.append(g)
    shrub_kids += [{"name": f"Shrub area {osm}", "children": gs} for osm, gs in areas.items()]
    top_kids.append({"name": "Shrubs", "children": shrub_kids})
    top_kids.append({"name": "Hedges", "children": [{"name": f"Hedge {osm}", "children": gs}
                                                     for osm, gs in by_osm("hedges").items()]})
    con_kids = [{"name": name, "children": guids(key)}
                for key, name in (("contours_minor", "Contours minor"), ("contours_major", "Contours major"),
                                  ("contour_labels", "Contour labels"))]
    top_kids.append({"name": "Contours", "children": [k for k in con_kids if k["children"]]})
    top_kids = [k for k in top_kids if not (isinstance(k, dict) and not k["children"])]
    node = collapse({"name": top, "children": top_kids})
    return node if len(node["children"]) >= 2 else None


def group_nodes_postorder(node):
    """Yield nodes children-first (inner groups must exist before the group that contains them)."""
    for k in node["children"]:
        if isinstance(k, dict):
            yield from group_nodes_postorder(k)
    yield node


def group_depth(node) -> int:
    return 1 + max([group_depth(k) for k in node["children"] if isinstance(k, dict)] or [0])
