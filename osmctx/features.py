"""More OSM feature types (M6): railways + platforms, walls/fences/guard rails, building parts + roofs, ground
surfaces, piers/bridge decks, power towers.

Pure helpers (roof geometry, tag parsing) plus functions that take the Converter (`cv`) and read its OSM elements /
frame / terrain sampler. convert.py only calls `rail_records` / `rail_out` (inside roads(), so railways share the
road drape + bridge/tunnel logic) and `run` (everything else). The builder side lives in build_plan.py
(payload mapping, pure) and build_ac.py (Archicad calls).

Native tool mapping (a gap is logged wherever the tool is a workaround, see out/gaps.md):
  railway ways        -> Beam per segment (like roads)         GAP: no rail/track element
  platforms           -> thin Slab
  wall/retaining/city -> Wall tool (CreateWalls), <= 5 m segments base on terrain
  fence/guard_rail    -> Railing tool (CreateRailings), thin Wall fallback
  building:part       -> Slab per part (bottom = base + min_height); the parent outline only where no part covers it
  roof:shape + height -> Roof tool (CreateRoofs): skillion/gabled = single-plane roofs, hipped/pyramidal = multi-plane
  parking/pitch/...   -> Slab 0.05 m on flat ground, draped Mesh on sloped ground
  pier / bridge area  -> Slab at deck height
  power tower/pole    -> Column (default off)
"""
from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

# ---------------------------------------------------------------- settings
INCLUDE_DEFAULTS = {"railways": True, "walls": True, "building_parts": True, "ground": True, "piers": True,
                    "power": False}
FEATURE_DEFAULTS = {
    "rail_track_width": 3.0,        # m per track (ballast bed); gauge tag: gauge_m + 1.5; tracks tag multiplies
    "barrier_max_segment": 5.0,     # m, walls/fences are split so no element is longer (<= 10 m by the brief)
    "barrier_defaults": {           # kind -> default height / thickness (m) when the tags say nothing
        "wall": {"height": 2.0, "thickness": 0.25},
        "retaining_wall": {"height": 1.5, "thickness": 0.4},
        "city_wall": {"height": 4.0, "thickness": 1.0},
        "fence": {"height": 1.5, "thickness": 0.05},
        "guard_rail": {"height": 0.8, "thickness": 0.1},
    },
    "platform_height": 0.35,        # m, platform slab top above the highest ground under it
    "ground_thickness": 0.05,       # m, thin ground-surface slab
    "pier_deck_above_water": 1.0,   # m, pier deck top above the water surface
    "pier_thickness": 0.5,
    "pier_width": 3.0,              # m, line piers (man_made=pier ways) are buffered to this width
    "bridge_deck_thickness": 0.5,
    "tower_height": 30.0,           # power=tower default (m)
    "pole_height": 9.0,             # power=pole default (m)
    "building_part_min_remainder": 4.0,  # m2, parent outline pieces smaller than this are not built
}
FEATURE_FLOAT_KEYS = ("rail_track_width", "barrier_max_segment", "platform_height", "ground_thickness",
                      "pier_deck_above_water", "pier_thickness", "pier_width", "bridge_deck_thickness",
                      "tower_height", "pole_height", "building_part_min_remainder")

RAIL_CLASSES = ("rail", "light_rail", "tram", "subway", "narrow_gauge")
BARRIER_KINDS = ("wall", "retaining_wall", "city_wall", "fence", "guard_rail")
BARRIER_TOOL = {"wall": "wall", "retaining_wall": "wall", "city_wall": "wall", "fence": "railing",
                "guard_rail": "railing"}
BARRIER_MATERIAL = {"brick": "Brick", "stone": "Stone - Structural", "concrete": "Concrete", "masonry": "Brick",
                    "wood": "Timber - Structural", "metal": "Steel", "steel": "Steel", "cement_block": "Concrete"}
BARRIER_MATERIAL_DEFAULT = {"wall": "Concrete", "retaining_wall": "Concrete", "city_wall": "Stone - Structural",
                            "fence": "Steel", "guard_rail": "Steel"}
ROOF_SHAPES = {"gabled": "gabled", "hipped": "hipped", "half-hipped": "hipped", "pyramidal": "pyramidal",
               "skillion": "skillion", "lean_to": "skillion"}
MIN_BLOCK_HEIGHT = 0.5          # a roofed block below the roof keeps at least this height
ROOF_MIN_HEIGHT = 0.3
GROUND_FLAT_SPREAD = 0.3        # m of terrain spread under the polygon: <= flat Slab, > draped Mesh
GROUND_KINDS = ("parking", "pitch", "playground", "pedestrian", "road_area", "square")
# surface tag -> canonical building material name the builder resolves in the open project
SURFACE_MATERIAL = {"asphalt": "Asphalt", "paved": "Asphalt", "concrete": "Concrete", "paving_stones": "Paving",
                    "sett": "Paving", "cobblestone": "Paving", "unhewn_cobblestone": "Paving", "bricks": "Paving",
                    "grass": "Grass", "artificial_turf": "Grass", "sand": "Sand", "gravel": "Gravel",
                    "fine_gravel": "Gravel", "compacted": "Gravel", "pebblestone": "Gravel", "dirt": "Soil",
                    "ground": "Soil", "earth": "Soil", "clay": "Soil", "tartan": "Asphalt", "rubber": "Asphalt",
                    "metal": "Steel", "wood": "Timber - Floor"}
KIND_MATERIAL = {"parking": "Asphalt", "pitch": "Asphalt", "playground": "Sand", "pedestrian": "Paving",
                 "road_area": "Asphalt", "square": "Paving"}


def normalize_settings(s: dict) -> None:
    """In place: defaults + types of the feature keys (called by convert._merge_settings)."""
    for k, v in FEATURE_DEFAULTS.items():
        if k == "barrier_defaults":
            merged = {kd: dict(d) for kd, d in v.items()}
            for kd, d in (s.get(k) or {}).items():
                merged.setdefault(kd, {}).update({a: float(b) for a, b in (d or {}).items() if b is not None})
            s[k] = merged
        elif s.get(k) is None:
            s[k] = v
    for k in FEATURE_FLOAT_KEYS:
        s[k] = float(s[k])
    inc = dict(INCLUDE_DEFAULTS)
    inc.update(s.get("include") or {})
    s["include"] = inc


def _C():
    from . import convert
    return convert


# ---------------------------------------------------------------- shared helpers
def area_geom(cv, el):
    """Closed way / multipolygon relation -> local Polygon/MultiPolygon, else None."""
    C = _C()
    t = el.get("tags", {})
    if el["type"] == "way":
        ll = C.way_lonlat(el.get("geometry"))
        if not C.is_closed_way(ll):
            return None
        return cv.f.geom_to_local(Polygon(ll))
    if el["type"] == "relation" and t.get("type") == "multipolygon":
        g, _bad = C.relation_area(el, cv.f)
        return g
    return None


def clipped_polys(cv, g, min_area: float = 1.0) -> list[Polygon]:
    C = _C()
    polys, _ = C.clean_polygons(make_valid(g).intersection(cv.bpoly), min_area=min_area)
    return polys


def z_stats(cv, poly: Polygon) -> tuple[float, float]:
    """(min, max) site-relative terrain z under a polygon: outline every 3 m, a grid inside, the centroid."""
    pts = [tuple(p) for p in shapely.segmentize(poly.exterior, 3.0).coords]
    minx, miny, maxx, maxy = poly.bounds
    step = max(3.0, math.sqrt(max(poly.area, 1.0) / 150.0))
    gx, gy = np.meshgrid(np.arange(minx + step / 2, maxx, step), np.arange(miny + step / 2, maxy, step))
    if gx.size:
        shapely.prepare(poly)
        m = shapely.contains_xy(poly, gx.ravel(), gy.ravel())
        pts += list(zip(gx.ravel()[m], gy.ravel()[m]))
    pts.append((poly.centroid.x, poly.centroid.y))
    a = np.asarray(pts, float)
    z = cv.z_rel(a[:, 0], a[:, 1])
    return float(z.min()), float(z.max())


def _tags(t: dict, keys) -> dict:
    """The present (non-empty) tags among `keys`, as strings."""
    return {k: str(t[k]) for k in keys if t.get(k)}


def _ring_fields(C, p: Polygon) -> tuple[list, list]:
    """(exterior ring, hole rings) of a polygon as site.json xy lists."""
    return C.ring_xy(p.exterior), [C.ring_xy(r) for r in p.interiors]


def _tag_material(t: dict, table: dict, default: str | None = None) -> str | None:
    m = str(t.get("material") or t.get("barrier:material") or "").split(";")[0].strip().lower()
    return table.get(m, default)


# ---------------------------------------------------------------- railways
def rail_width(t: dict, s: dict) -> tuple[float, str, int]:
    """(width m, source 'tag'|'gauge'|'tracks'|'default', tracks)."""
    C = _C()
    tracks = C.parse_number(t.get("tracks"))
    tracks = int(tracks) if tracks and 1 <= tracks <= 8 else 1
    w = C.parse_length(t.get("width"), max_value=60.0)
    if w:
        return w, "tag", tracks
    per, src = s["rail_track_width"], "default"
    g = C.parse_number(t.get("gauge"))
    if g and 500 <= g <= 3000:
        per, src = round(g / 1000.0 + 1.5, 3), "gauge"
    if tracks > 1:
        src = "tracks" if src == "default" else src + "+tracks"
    return per * tracks, src, tracks


def rail_records(cv) -> list[dict]:
    """Railway ways as road-like records (same keys as convert.Converter.roads recs) so they share drape + bridge/
    tunnel logic. railway=disused|abandoned (and disused=yes / abandoned=yes) are skipped; a subway way without
    bridge/tunnel tags is treated as a tunnel (assumed underground, counted)."""
    C = _C()
    out = []
    for el in cv.els:
        t = el.get("tags", {})
        cls = t.get("railway")
        if el["type"] != "way" or cls not in RAIL_CLASSES:
            continue
        if t.get("disused") == "yes" or t.get("abandoned") == "yes" or t.get("construction"):
            cv.counts["railways_skipped_disused"] += 1
            continue
        nodes, ll = C.way_nodes_lonlat(el)
        if len(ll) < 2:
            continue
        x, y = cv.f.to_local([p[0] for p in ll], [p[1] for p in ll])
        xy = np.column_stack([x, y])
        keys = [n if n is not None else f"{round(a, 2)},{round(b, 2)}" for n, (a, b) in zip(nodes, xy)]
        bridge = t.get("bridge", "no") not in ("no", "low_water_crossing")
        tunnel = t.get("tunnel", "no") != "no" and not bridge
        layer = C.parse_layer(t.get("layer"))
        if cls == "subway" and not bridge and not tunnel:
            tunnel = True
            cv.counts["railways_subway_assumed_tunnel"] += 1
        if tunnel and layer >= 0:
            layer = -1
        dens, oidx = C.densify(xy, C.ROAD_MAX_SEG)
        out.append({"el": el, "t": t, "cls": cls, "nodes": keys, "dens": dens, "oidx": oidx, "s": C.arclen(dens),
                    "bridge": bridge, "tunnel": tunnel, "layer": layer,
                    "kind": "bridge" if bridge else "tunnel" if tunnel else None, "rail": True})
    return out


def rail_out(cv, r: dict, pieces: list, out: list) -> None:
    """Final railway records (road pieces with site-relative z) -> out."""
    C = _C()
    w, src, tracks = rail_width(r["t"], cv.s)
    cv.counts["railway_ways"] += 1
    cv.counts[f"railway_{r['cls']}"] += 1
    if r["bridge"]:
        cv.counts["railway_bridge_ways"] += 1
    if r["tunnel"]:
        cv.counts["railway_tunnel_ways"] += 1
    for p in pieces:
        out.append({"osm_id": C.osm_id(r["el"]), "class": r["cls"], "width": round(w, 3), "width_source": src,
                    "tracks": tracks, "bridge": r["bridge"], "tunnel": r["tunnel"], "layer": r["layer"],
                    "points": [[round(a, 3), round(b, 3), round(c, 3)] for a, b, c in p]})


def platforms(cv) -> list[dict]:
    """railway=platform / public_transport=platform areas (closed ways, multipolygons); open platform ways are
    buffered to `width` (default 3 m). One thin Slab each: base = lowest ground, top = highest ground + platform_height."""
    C = _C()
    out = []
    for el in cv.els:
        t = el.get("tags", {})
        if t.get("railway") != "platform" and t.get("public_transport") != "platform":
            continue
        if el["type"] not in ("way", "relation"):
            continue
        g = area_geom(cv, el)
        how = "area"
        if g is None and el["type"] == "way":
            ll = C.way_lonlat(el.get("geometry"))
            if len(ll) < 2:
                continue
            w = C.parse_length(t.get("width"), max_value=30.0) or 3.0
            g = cv.f.geom_to_local(LineString(ll)).buffer(w / 2.0, cap_style="flat")
            how = "line"
        if g is None:
            cv.counts["platforms_unclosed_skipped"] += 1
            continue
        for p in clipped_polys(cv, g, 1.0):
            z0, z1 = z_stats(cv, p)
            ext, holes = _ring_fields(C, p)
            out.append({"osm_id": C.osm_id(el), "kind": "platform", "shape": how, "polygon": ext,
                        "holes": holes, "base_z": round(z0, 3),
                        "thickness": round(z1 - z0 + cv.s["platform_height"], 3), "area_m2": round(p.area, 1),
                        "tags": _tags(t, ("railway", "public_transport", "surface", "name"))})
    cv.counts["railway_platforms"] = len(out)
    return out


# ---------------------------------------------------------------- walls / fences / guard rails
class BridgeZ:
    """z of the nearest bridge deck (road or railway with bridge=true) within reach of a point, so guard rails and
    parapets on a bridge stand on the deck, not on the ground far below."""

    def __init__(self, ways: list[dict]):
        self.items = []          # (line, cumulative length, z, reach) per bridge way
        for r in ways:
            if not r.get("bridge") or len(r["points"]) < 2:
                continue
            a = np.asarray(r["points"], float)
            self.items.append((LineString(a[:, :2]), np.r_[0.0, np.cumsum(np.hypot(*np.diff(a[:, :2], axis=0).T))],
                               a[:, 2], float(r.get("width") or 3.0) / 2.0 + 1.5))
        if self.items:
            self._lines = np.array([it[0] for it in self.items], dtype=object)
            self._reach = np.array([it[3] for it in self.items], dtype=float)
            # bounding box grown by `reach`: only the ways whose box holds the point are measured
            self._tree = shapely.STRtree(shapely.box(*(np.array([it[0].bounds for it in self.items]).T
                                                       + np.array([-1, -1, 1, 1])[:, None] * self._reach)))

    def z_many(self, xy) -> list:
        """z of the nearest deck within reach for each (x, y) row (None where there is none)."""
        n = len(xy)
        out: list = [None] * n
        if not self.items or not n:
            return out
        pts = shapely.points(np.asarray(xy, float)[:, 0], np.asarray(xy, float)[:, 1])
        qi, ti = self._tree.query(pts)
        if not len(qi):
            return out
        d = shapely.distance(self._lines[ti], pts[qi])
        ok = d <= self._reach[ti]
        qi, ti, d = qi[ok], ti[ok], d[ok]
        order = np.lexsort((ti, d, qi))               # per point: nearest way, ties -> the first listed
        qi, ti = qi[order], ti[order]
        first = np.r_[True, qi[1:] != qi[:-1]] if len(qi) else qi.astype(bool)
        for i, j in zip(qi[first].tolist(), ti[first].tolist()):
            line, s, z, _ = self.items[j]
            out[i] = float(np.interp(line.project(pts[i]), s, z))
        return out

    def z_at(self, x: float, y: float):
        return self.z_many([(x, y)])[0]


def barrier_spec(t: dict, kind: str, s: dict) -> dict:
    C = _C()
    d = s["barrier_defaults"][kind]
    h = C.parse_length(t.get("height"), max_value=60.0)
    th = C.parse_length(t.get("width"), max_value=6.0) if kind in ("wall", "retaining_wall", "city_wall") else None
    return {"height": h or d["height"], "height_source": "tag" if h else "default",
            "thickness": th or d["thickness"], "thickness_source": "tag" if th else "default"}


def barriers(cv, ways: list[dict]) -> list[dict]:
    """barrier=wall|retaining_wall|city_wall|fence|guard_rail lines clipped to the boundary, densified to
    <= barrier_max_segment, z on the terrain (bridge deck z where the line runs along a bridge). The builder makes one
    Wall (or Railing) per point pair with its base at the lower end's z."""
    C = _C()
    bz = BridgeZ(ways)
    out = []
    for el in cv.els:
        t = el.get("tags", {})
        kind = t.get("barrier")
        if el["type"] != "way" or kind not in BARRIER_KINDS:
            continue
        if t.get("tunnel", "no") != "no":
            cv.counts["barriers_skipped_tunnel"] += 1
            continue
        ll = C.way_lonlat(el.get("geometry"))
        if len(ll) < 2:
            continue
        x, y = cv.f.to_local([p[0] for p in ll], [p[1] for p in ll])
        pts = np.column_stack([x, y, np.zeros(len(x))])
        spec = barrier_spec(t, kind, cv.s)
        for piece in C.clip_polyline_z(pts, cv.bpoly):
            if C.arclen(piece)[-1] < 0.3:
                continue
            dens, _ = C.densify(piece[:, :2], cv.s["barrier_max_segment"])
            z = cv.z_rel(dens[:, 0], dens[:, 1])
            on_bridge = False
            for i, zb in enumerate(bz.z_many(dens)):
                if zb is not None:
                    z[i] = zb
                    on_bridge = True
            mat = _tag_material(t, BARRIER_MATERIAL, BARRIER_MATERIAL_DEFAULT[kind])
            out.append({"osm_id": C.osm_id(el), "kind": kind, "tool": BARRIER_TOOL[kind], **spec,
                        "material": mat, "on_bridge": on_bridge,
                        "points": [[round(a, 3), round(b, 3), round(c, 3)] for (a, b), c in zip(dens, z)],
                        "tags": _tags(t, ("barrier", "height", "material", "name"))})
            cv.counts[f"barriers_{kind}"] += 1
    cv.counts["barriers"] = len(out)
    return out


# ---------------------------------------------------------------- roofs (pure geometry)
def roof_spec(t: dict, total_h: float, min_h: float, s: dict, footprint: Polygon | None = None):
    """roof dict from the tags, or None (flat block). Needs roof:shape in ROOF_SHAPES and roof:height (or
    roof:angle, resolved with the footprint span). The block under the roof keeps >= MIN_BLOCK_HEIGHT."""
    C = _C()
    shape = ROOF_SHAPES.get(str(t.get("roof:shape", "")).strip().lower())
    if shape is None:
        return None
    rh = C.parse_length(t.get("roof:height"), max_value=100.0)
    src = "height"
    if rh is None and footprint is not None:
        ang = C.parse_number(t.get("roof:angle"))
        if ang and 5.0 <= ang <= 75.0:
            span = roof_basis(footprint, shape, t.get("roof:orientation"), None)["span"]
            rh, src = math.tan(math.radians(ang)) * span, "angle"
    if rh is None or rh < ROOF_MIN_HEIGHT:
        return None
    if total_h - min_h - rh < MIN_BLOCK_HEIGHT:
        return None   # ambiguous (roof taller than the building): keep the flat block
    d = C.parse_number(t.get("roof:direction"))
    return {"shape": shape, "height": round(rh, 3), "height_source": src,
            "orientation": str(t.get("roof:orientation") or "").lower() or None,
            "direction": float(d) if d is not None and 0 <= d <= 360 else None}


def _mrr_axes(poly: Polygon):
    """Centre, unit long axis u, unit short axis v, long length L, short length W of the min rotated rectangle."""
    c = list(poly.minimum_rotated_rectangle.exterior.coords)[:4]
    e1 = np.array(c[1]) - np.array(c[0])
    e2 = np.array(c[2]) - np.array(c[1])
    l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    if l1 >= l2:
        u, v, L, W = e1 / l1, e2 / l2, l1, l2
    else:
        u, v, L, W = e2 / l2, e1 / l1, l2, l1
    ctr = np.mean(np.array(c), axis=0)
    return ctr, u, v, L, W


def roof_basis(poly: Polygon, shape: str, orientation=None, direction=None) -> dict:
    """Axes for the roof planes: ridge axis u (gabled: long axis, 'across' = short axis), perpendicular v, half-span
    `span` the slope runs over (gabled/hipped: half the width across the ridge, skillion: the depth, pyramidal:
    inscribed radius)."""
    ctr, u, v, L, W = _mrr_axes(poly)
    if shape in ("gabled", "hipped") and str(orientation or "").lower() == "across":
        u, v, L, W = v, u, W, L
    if shape == "skillion":
        if direction is not None:
            th = math.radians(direction)       # compass degrees clockwise from north = direction the roof falls to
            w = np.array([math.sin(th), math.cos(th)])
        else:
            w = v if v[1] <= 0 else -v         # falls towards the south side (arbitrary but deterministic)
        proj = np.asarray(poly.exterior.coords)[:, :2] @ w
        depth = float(proj.max() - proj.min())
        return {"center": ctr, "u": u, "v": v, "L": L, "W": W, "w": w, "depth": depth,
                "span": max(depth, 0.1), "proj_max": float(proj.max())}
    if shape == "pyramidal":
        try:
            r = float(shapely.maximum_inscribed_circle(poly, 0.1).length)
        except Exception:  # noqa: BLE001
            r = W / 2.0
        return {"center": ctr, "u": u, "v": v, "L": L, "W": W, "span": max(r, 0.1)}
    return {"center": ctr, "u": u, "v": v, "L": L, "W": W, "span": max(W / 2.0, 0.1)}


def roof_planes(poly: Polygon, roof: dict) -> list[dict]:
    """Roof as Archicad-agnostic planes: [{"kind": "single", "polygon": [[x,y]..], "pivot": [[x,y],[x,y]], "angle": rad,
    "pivot_rise": m}] (pivot_rise = height of the pivot line above the roof base = eave top) or [{"kind": "multi",
    "polygon", "angle", "rise"}]. skillion/gabled -> single-plane roofs, hipped/pyramidal -> one multi-plane roof."""
    shape, rh = roof["shape"], float(roof["height"])
    b = roof_basis(poly, shape, roof.get("orientation"), roof.get("direction"))
    ang = math.atan2(rh, b["span"])
    ring = [[float(x), float(y)] for x, y in list(poly.exterior.coords)[:-1]]
    if shape in ("hipped", "pyramidal"):
        return [{"kind": "multi", "polygon": ring, "angle": ang, "rise": rh}]
    if shape == "skillion":
        w = b["w"]
        d = np.array([-w[1], w[0]])           # polygon (towards -w) lies on the LEFT of d, the plane rises there
        proj_d = np.asarray(poly.exterior.coords)[:, :2] @ d
        q = w * b["proj_max"] + d * float((proj_d.min() + proj_d.max()) / 2.0)
        half = float(proj_d.max() - proj_d.min()) / 2.0 + 0.5
        return [{"kind": "single", "polygon": ring, "pivot": [(q - d * half).tolist(), (q + d * half).tolist()],
                 "angle": ang, "pivot_rise": 0.0}]
    # gabled: ridge along u through the MRR centre, two single-plane halves falling away from the ridge
    c, u, v = b["center"], b["u"], b["v"]
    R = (b["L"] + b["W"]) * 2.0 + 10.0
    planes = []
    for sgn in (1.0, -1.0):
        side = Polygon([c - u * R, c + u * R, c + u * R + sgn * v * R, c - u * R + sgn * v * R])
        half_poly = poly.intersection(side)
        for g in shapely.get_parts(half_poly):      # Polygon, MultiPolygon or GeometryCollection (edge touch)
            if not isinstance(g, Polygon) or g.area < 0.5:
                continue
            d = np.array([-v[1], v[0]]) * sgn   # half on +v side needs the left normal to be -v
            hl = b["L"] / 2.0 + 0.5
            planes.append({"kind": "single", "polygon": [[float(x), float(y)] for x, y in list(g.exterior.coords)[:-1]],
                           "pivot": [(c - d * hl).tolist(), (c + d * hl).tolist()], "angle": ang, "pivot_rise": rh})
    return planes


# ---------------------------------------------------------------- building parts
def _poly_of(b: dict) -> Polygon:
    return Polygon(b["footprint"], [h for h in (b.get("holes") or []) if len(h) >= 3])


def building_parts(cv, buildings: list[dict]) -> tuple[list[dict], list[dict]]:
    """building:part ways/relations -> parts [{osm_id, parent, footprint, holes, base_z, min_height, height, ...}] and
    the (possibly reduced) building list. Per OSM convention the parent outline is not a full block where parts cover
    it: it is replaced by the uncovered remainder (tag osmctx:remainder=yes) or dropped (`skip_block`) when < 4 m2 is left.
    Parts: bottom = base + min_height (min_height tag, else building:min_level x floor height), top = base + height
    (height tag, else levels, else the parent's height). A part with min_height >= height is dropped (counted)."""
    C = _C()
    s = cv.s
    cands = []
    for el in cv.els:
        t = el.get("tags", {})
        if "building:part" not in t or t.get("building:part") == "no" or el["type"] == "node":
            continue
        g = area_geom(cv, el)
        if g is None:
            cv.counts["building_parts_unclosed_skipped"] += 1
            continue
        for p in clipped_polys(cv, g, C.MIN_AREA):
            if cv.bpoly.covers(p.centroid):
                cands.append((el, t, p))
            else:
                cv.counts["building_parts_outside"] += 1
    if not cands:
        return [], buildings
    polys = [_poly_of(b) for b in buildings]
    tree = shapely.STRtree(polys)
    parent_parts: dict[int, list[Polygon]] = {}
    parent_part_ids: dict[int, set[str]] = {}
    parts = []
    for el, t, p in cands:
        pi, best = None, 0.5
        for j in tree.query(p):
            ar = p.intersection(polys[j]).area / p.area if p.area else 0.0
            if ar >= best:
                pi, best = int(j), ar
        parent = buildings[pi] if pi is not None else None
        h, src = C.building_height(t, s)
        if src == "default" and parent is not None:
            h, src = float(parent["height"]), "parent_" + str(parent.get("height_source"))
        mn = C.parse_length(t.get("min_height"), max_value=1000.0)
        if mn is None:
            lv = C.parse_number(t.get("building:min_level"))
            mn = lv * s["floor_height"] if lv and lv > 0 else 0.0
        if mn >= h - 0.3:
            cv.counts["building_parts_dropped_bad_height"] += 1
            continue
        base = float(parent["base_z"]) if parent is not None else z_stats(cv, p)[0]
        ext, holes = _ring_fields(C, p)
        entry = {"osm_id": C.osm_id(el), "parent": parent["osm_id"] if parent else None,
                 "footprint": ext, "holes": holes,
                 "base_z": round(base, 3), "min_height": round(mn, 3), "height": round(h, 3), "height_source": src,
                 "tags": dict(t)}
        roof = roof_spec(t, h, mn, s, p) if not p.interiors else None
        if roof:
            entry["roof"] = roof
            cv.counts[f"roofs_{roof['shape']}"] += 1
        elif t.get("roof:shape") in ROOF_SHAPES:
            cv.counts["roofs_ambiguous_flat"] += 1
        parts.append(entry)
        if pi is not None:
            parent_parts.setdefault(pi, []).append(p)
            parent_part_ids.setdefault(pi, set()).add(entry["osm_id"])
        cv.counts["building_parts"] += 1
    keep, extra = [], []
    for i, b in enumerate(buildings):
        pp = parent_parts.get(i)
        if not pp:
            keep.append(b)
            continue
        poly = polys[i]
        try:
            rest = poly.difference(unary_union(pp).buffer(0.3))
            rest_polys, _ = C.clean_polygons(rest, min_area=max(s["building_part_min_remainder"], 0.0)) if not rest.is_empty else ([], 0)
        except Exception:  # noqa: BLE001
            rest_polys = [poly]       # ambiguous geometry: fall back to the full block
        b = dict(b, parts=sorted(parent_part_ids.get(i, ())))
        if not rest_polys:
            b["skip_block"] = True
            cv.counts["buildings_block_suppressed_by_parts"] += 1
            keep.append(b)
            continue
        cv.counts["buildings_block_reduced_by_parts"] += 1
        for k, rp in enumerate(rest_polys):
            ext, holes = _ring_fields(C, rp)
            nb = dict(b, footprint=ext, holes=holes, tags=dict(b["tags"], **{"osmctx:remainder": "yes"}))
            nb.pop("roof", None)
            (keep if k == 0 else extra).append(nb)
    return parts, keep + extra


def building_roofs(cv, buildings: list[dict]) -> None:
    """Plain buildings (no parts) with roof:shape + roof:height get a `roof` dict (in place)."""
    for b in buildings:
        if b.get("parts") or b.get("skip_block") or b["tags"].get("osmctx:flag"):
            continue
        t = b["tags"]
        if t.get("roof:shape") not in ROOF_SHAPES or b.get("holes"):
            continue
        roof = roof_spec(t, float(b["height"]), 0.0, cv.s, _poly_of(b))
        if roof:
            b["roof"] = roof
            cv.counts[f"roofs_{roof['shape']}"] += 1
        else:
            cv.counts["roofs_ambiguous_flat"] += 1


# ---------------------------------------------------------------- ground surfaces
def ground_kind(t: dict) -> str | None:
    if t.get("amenity") == "parking":
        if t.get("parking") in ("multi-storey", "underground", "rooftop") or "building" in t:
            return None
        return "parking"
    lei = t.get("leisure")
    if lei in ("pitch", "playground"):
        return lei
    if t.get("highway") == "pedestrian" and t.get("area") == "yes":
        return "pedestrian"
    ah = t.get("area:highway")
    if ah:
        if ah in ("pedestrian", "footway", "path", "cycleway", "steps"):
            return "pedestrian"
        if ah in ("crossing", "traffic_island", "kerb"):
            return None
        return "road_area"
    if t.get("place") == "square":
        return "square"
    return None


def ground_material(t: dict, kind: str) -> str:
    sf = str(t.get("surface") or "").split(";")[0].strip().lower()
    return SURFACE_MATERIAL.get(sf) or KIND_MATERIAL[kind]


def ground_surfaces(cv) -> list[dict]:
    """parking / pitch / playground / pedestrian areas / area:highway / place=square areas on the terrain. Each carries the
    terrain z range under it; the builder makes a thin Slab when the ground is flat (<= GROUND_FLAT_SPREAD) and a draped
    Mesh otherwise (a slab would float or sink on a slope)."""
    C = _C()
    out = []
    for el in cv.els:
        t = el.get("tags", {})
        kind = ground_kind(t)
        if kind is None or el["type"] == "node":
            continue
        g = area_geom(cv, el)
        if g is None:
            cv.counts["ground_unclosed_skipped"] += 1
            continue
        for p in clipped_polys(cv, g, 1.0):
            z0, z1 = z_stats(cv, p)
            ext, holes = _ring_fields(C, p)
            out.append({"osm_id": C.osm_id(el), "kind": kind, "material": ground_material(t, kind),
                        "polygon": ext, "holes": holes,
                        "area_m2": round(p.area, 1), "z_min": round(z0, 3), "z_max": round(z1, 3),
                        "sloped": (z1 - z0) > GROUND_FLAT_SPREAD,
                        "tags": _tags(t, ("amenity", "leisure", "highway", "area:highway", "place", "surface",
                                          "sport", "name"))})
            cv.counts[f"ground_{kind}"] += 1
    cv.counts["ground_surfaces"] = len(out)
    cv.counts["ground_sloped"] = sum(1 for g in out if g["sloped"])
    return out


# ---------------------------------------------------------------- piers + bridge decks
def piers(cv, ways: list[dict]) -> list[dict]:
    """man_made=pier|bridge areas (open pier ways are buffered to pier_width) -> deck Slab.
    pier: top = water surface under/next to it + pier_deck_above_water (no water: highest terrain + 0.5 m).
    bridge: top = highest z of a bridge road/railway inside the area - 0.2 m (flush under the road beams); a bridge
    area without a bridge way inside is skipped (counted)."""
    C = _C()
    s = cv.s
    out = []
    bridge_ways = [(LineString(np.asarray(r["points"], float)[:, :2]), max(p[2] for p in r["points"]))
                   for r in ways if r.get("bridge") and len(r["points"]) >= 2]
    bridge_tree = shapely.STRtree([ln for ln, _ in bridge_ways]) if bridge_ways else None
    for el in cv.els:
        t = el.get("tags", {})
        kind = t.get("man_made")
        if kind not in ("pier", "bridge") or el["type"] == "node":
            continue
        g = area_geom(cv, el)
        if g is None and el["type"] == "way" and kind == "pier":
            ll = C.way_lonlat(el.get("geometry"))
            if len(ll) >= 2:
                w = C.parse_length(t.get("width"), max_value=60.0) or s["pier_width"]
                g = cv.f.geom_to_local(LineString(ll)).buffer(w / 2.0, cap_style="flat")
        if g is None:
            cv.counts["piers_unclosed_skipped"] += 1
            continue
        for p in clipped_polys(cv, g, 1.0):
            if kind == "bridge":
                zs = [bridge_ways[j][1] for j in bridge_tree.query(p, predicate="intersects")] if bridge_tree else []
                if not zs:
                    cv.counts["piers_bridge_no_deck_skipped"] += 1
                    continue
                top, src, th = max(zs) - 0.2, "bridge_road_max", s["bridge_deck_thickness"]
            else:
                wz = cv._in_water_z(p) if getattr(cv, "_water_union_b", None) is not None else None
                if wz is None and getattr(cv, "_water_polys", None):
                    near = [w for w in cv._water_polys if w["poly"].distance(p) < 5.0]
                    wz = (near[0]["z_abs"] - cv.site.elev0) if near else None
                if wz is not None:
                    top, src = wz + s["pier_deck_above_water"], "water_plus"
                else:
                    top, src = z_stats(cv, p)[1] + 0.5, "terrain_plus"
                th = s["pier_thickness"]
            ext, holes = _ring_fields(C, p)
            out.append({"osm_id": C.osm_id(el), "kind": kind, "polygon": ext,
                        "holes": holes, "z": round(top, 3), "thickness": th,
                        "z_source": src, "area_m2": round(p.area, 1),
                        "tags": _tags(t, ("man_made", "bridge", "width", "name"))})
            cv.counts[f"piers_{kind}"] += 1
    cv.counts["piers"] = len(out)
    return out


# ---------------------------------------------------------------- power
def power(cv) -> list[dict]:
    """power=tower|pole nodes inside the boundary (include.power). power=line|minor_line are only counted."""
    C = _C()
    on = cv.s["include"].get("power", False)
    out = []
    for el in cv.els:
        t = el.get("tags", {})
        pw = t.get("power")
        if pw in ("line", "minor_line", "cable") and el["type"] == "way":
            cv.counts["power_lines"] += 1
            continue
        if pw not in ("tower", "pole") or el["type"] != "node":
            continue
        x, y = cv.f.to_local(el["lon"], el["lat"])
        x, y = float(x), float(y)
        if not cv.bpoly.covers(Point(x, y)):
            continue
        cv.counts[f"power_{pw}s_found"] += 1
        if not on:
            continue
        h = C.parse_length(t.get("height"), max_value=400.0)
        out.append({"osm_id": C.osm_id(el), "kind": pw, "x": round(x, 3), "y": round(y, 3),
                    "z": round(float(cv.z_rel(x, y)[0]), 3),
                    "height": h or (cv.s["tower_height"] if pw == "tower" else cv.s["pole_height"]),
                    "height_source": "tag" if h else "default",
                    "tags": _tags(t, ("power", "height", "structure", "name"))})
    cv.counts["power_towers"] = len(out)
    return out


# ---------------------------------------------------------------- orchestration
def run(cv, buildings: list[dict], roads: list[dict], rails: list[dict]) -> tuple[dict, list[dict]]:
    """Everything except railway ways (built inside roads()). -> (site keys, buildings)."""
    inc = cv.s["include"]
    out = {"railways": rails if inc.get("railways", True) else [], "railway_platforms": [], "barriers": [],
           "building_parts": [], "ground_surfaces": [], "piers": [], "power_towers": []}
    if inc.get("railways", True):
        out["railway_platforms"] = platforms(cv)
    if inc.get("building_parts", True) and inc.get("buildings", True):
        parts, buildings = building_parts(cv, buildings)
        out["building_parts"] = parts
        building_roofs(cv, buildings)
    if inc.get("walls", True):
        out["barriers"] = barriers(cv, roads + rails)
    if inc.get("ground", True):
        out["ground_surfaces"] = ground_surfaces(cv)
    if inc.get("piers", True):
        out["piers"] = piers(cv, roads + rails)
    out["power_towers"] = power(cv)
    return out, buildings
