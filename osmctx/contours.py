"""Site-plan contour lines from site.json terrain (pure geometry, no Archicad access).

Why: Archicad's Mesh has no native "contour every X m" display (gap, out/gaps.md), and its plan view shows either the
triangulation or only user-defined ridges. Proper site plans carry minor/major contour lines with elevation labels, so
the importer computes them and draws them as 2D Polylines + Texts (regenerated with every import).

Surface = the SAME surface the terrain Mesh uses: the TIN through terrain.points (densified outline ring + interior
grid points, z relative to origin.elevation), i.e. a Delaunay triangulation, contoured exactly per triangle
("marching triangles": the triangle version of marching squares; exact on the linear facets, no resampling and no
quad ambiguity at saddles). Levels are ABSOLUTE elevations (origin.elevation + z), multiples of the interval.

Pipeline: levels -> segments per triangle -> chains joined on shared edge crossings -> clipped to the site boundary
-> Douglas-Peucker 0.1 m -> tiny closed loops dropped -> major (every 5th interval) / minor -> labels on majors.
"""
from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, MultiPoint, Polygon
from shapely.ops import unary_union

INTERVALS = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)   # auto choice / UI choices (m)
TARGET_MAX_LINES = 25                            # most minor lines across the relief the auto interval allows (aims 15-25)
MAJOR_EVERY = 5                                  # major = every 5th contour (absolute z multiple of 5 x interval)
SIMPLIFY_M = 0.1                                 # Douglas-Peucker tolerance
MIN_LOOP_M = 2.0                                 # closed loops shorter than this are dropped
MIN_LINE_M = 0.5                                 # open fragments shorter than this are dropped (clip slivers)
LABEL_MIN_LINE_M = 15.0                          # no label on lines shorter than this
LABEL_SPACING_M = 40.0                           # labels at least this far apart (any two labels)
LABEL_MAX_PER_LINE = 4
MAX_LEVELS = 300                                 # safety: interval is raised when relief / interval exceeds this
EPS = 1e-9


# ------------------------------------------------------------------ interval
def terrain_relief(site: dict) -> tuple[float, float]:
    """(z_min, z_max) of the terrain points, ABSOLUTE elevation (origin.elevation + z)."""
    e0 = float((site.get("origin") or {}).get("elevation") or 0.0)
    zs = [p[2] for p in site["terrain"]["points"]]
    return e0 + min(zs), e0 + max(zs)


def auto_interval(relief: float) -> float:
    """Smallest of INTERVALS that gives <= 25 minor lines over `relief` metres (so 15-25 on most sites; a flat site
    gets 0.25, a mountain gets 10)."""
    for iv in INTERVALS:
        if relief / iv <= TARGET_MAX_LINES:
            return iv
    return INTERVALS[-1]


def resolve_interval(site: dict, interval: float | None = None, scale: float = 1.0) -> dict:
    """-> {interval, auto_interval, relief_m, z_min, z_max, est_lines, source, note}. `interval` None = auto;
    `scale` multiplies the result (super-light preset: 2). Raises the interval when it would give > MAX_LEVELS lines."""
    zmin, zmax = terrain_relief(site)
    relief = zmax - zmin
    auto = auto_interval(relief)
    sc = float(scale or 1.0)
    iv = (float(interval) if interval else auto) * sc
    note = None
    if iv <= 0:
        raise ValueError("contour interval must be > 0")
    if relief / iv > MAX_LEVELS:
        was = iv
        iv = next((c * sc for c in INTERVALS if relief / (c * sc) <= MAX_LEVELS), INTERVALS[-1] * sc)
        note = f"interval {was:g} m would give {relief / was:.0f} lines: raised to {iv:g} m"
    return {"interval": iv, "auto_interval": auto, "relief_m": round(relief, 3), "z_min": round(zmin, 3),
            "z_max": round(zmax, 3), "est_lines": int(relief // iv) + 1,
            "source": "user" if interval else "auto", "note": note}


def preview(site: dict, interval: float | None = None) -> dict:
    """Small dict for the UI after Convert (auto value, relief)."""
    r = resolve_interval(site, interval)
    return {"auto_interval": r["auto_interval"], "relief_m": r["relief_m"], "z_min": r["z_min"], "z_max": r["z_max"],
            "est_lines": r["est_lines"]}


# ------------------------------------------------------------------ TIN
def _tin(site: dict):
    """-> (pts Nx3 float array [x,y,z rel], tris Mx3 int array) of the terrain TIN inside the boundary."""
    raw = site["terrain"]["points"]
    seen, pts = {}, []
    for p in raw:
        k = (round(p[0], 4), round(p[1], 4))
        if k not in seen:
            seen[k] = len(pts)
            pts.append((float(p[0]), float(p[1]), float(p[2])))
    P = np.array(pts, dtype=float)
    if len(P) < 3:
        return P, np.zeros((0, 3), dtype=int)
    try:
        gc = shapely.delaunay_triangles(MultiPoint([(x, y) for x, y, _ in pts]))
    except Exception:  # noqa: BLE001  (collinear / degenerate)
        return P, np.zeros((0, 3), dtype=int)
    parts = shapely.get_parts(gc)
    if len(parts) and (shapely.get_num_coordinates(parts) == 4).all():      # closed triangle rings: one bulk read
        c = shapely.get_coordinates(parts).reshape(-1, 4, 2)[:, :3].reshape(-1, 2).tolist()
    else:
        c = [xy for g in parts for xy in list(g.exterior.coords)[:3]]
    idx = np.array([seen.get((round(x, 4), round(y, 4)), -1) for x, y in c], dtype=int).reshape(-1, 3)
    T = idx[(idx >= 0).all(axis=1)]                                          # a vertex off the point set: drop the triangle
    bnd = site["terrain"].get("boundary") or site.get("boundary")
    if bnd and len(T):
        poly = Polygon([(p[0], p[1]) for p in bnd]).buffer(0.01)
        cen = P[T][:, :, :2].mean(axis=1)
        T = T[shapely.contains_xy(poly, cen[:, 0], cen[:, 1])]
    return P, T


def tin_z(site: dict, x: float, y: float) -> float | None:
    """z (relative to origin.elevation) on the contour surface at (x, y); None outside the TIN. Used by tests/checks."""
    P, T = _tin(site)
    if not len(T):
        return None
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    ok = np.abs(den) >= 1e-12
    den = np.where(ok, den, 1.0)
    l1 = ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / den
    l2 = ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / den
    l3 = 1 - l1 - l2
    hit = np.flatnonzero(ok & (np.minimum(np.minimum(l1, l2), l3) >= -1e-9))
    if not len(hit):
        return None
    i = hit[0]                                                               # first triangle containing the point
    return float(l1[i] * a[i, 2] + l2[i] * b[i, 2] + l3[i] * c[i, 2])


# ------------------------------------------------------------------ marching triangles
def _level_chains(P, T, level_rel: float):
    """Contour chains of one level on the TIN -> list of (coords [(x, y), ...], closed)."""
    z = P[:, 2] - level_rel
    z = np.where(np.abs(z) < EPS, EPS, z)          # nudge vertices exactly on the level: no degenerate segments
    s = z[T] > 0                                    # M x 3 booleans
    mixed = s.any(axis=1) & ~s.all(axis=1)
    if not mixed.any():
        return []
    Tm, cross = T[mixed], s[mixed] != np.roll(s[mixed], -1, axis=1)   # edge e = (v[e], v[e+1]); a mixed tri crosses on exactly 2
    e = np.argsort(~cross, axis=1, kind="stable")[:, :2]               # its 2 crossing edges, in edge order
    ea = np.take_along_axis(Tm, e, axis=1)
    eb = np.take_along_axis(np.roll(Tm, -1, axis=1), e, axis=1)
    n = len(P)
    code = np.minimum(ea, eb).astype(np.int64) * n + np.maximum(ea, eb)   # edge key (low, high vertex) as one int
    uniq = np.unique(code)
    lo, hi = uniq // n, uniq % n
    t = z[lo] / (z[lo] - z[hi])
    pos = dict(zip(uniq.tolist(), zip((P[lo, 0] + t * (P[hi, 0] - P[lo, 0])).tolist(),
                                      (P[lo, 1] + t * (P[hi, 1] - P[lo, 1])).tolist())))
    adj: dict = {}
    for k0, k1 in code.tolist():
        adj.setdefault(k0, []).append(k1)
        adj.setdefault(k1, []).append(k0)
    chains, used = [], set()

    def walk(start):
        path, prev, cur = [start], None, start
        used.add(start)
        while True:
            nxt = next((n for n in adj[cur] if n != prev and n not in used), None)
            if nxt is None:
                nxt = next((n for n in adj[cur] if n == start and prev is not None and n != prev), None)
                if nxt is not None:
                    path.append(nxt)
                    return path, True
                return path, False
            path.append(nxt)
            used.add(nxt)
            prev, cur = cur, nxt

    for k in adj:                                   # open chains first (start at degree-1 nodes)
        if k not in used and len(adj[k]) == 1:
            path, closed = walk(k)
            chains.append(([pos[q] for q in path], closed))
    for k in adj:                                   # then closed loops
        if k not in used:
            path, closed = walk(k)
            if closed and path[0] != path[-1]:
                path.append(path[0])
            chains.append(([pos[q] for q in path], closed or path[0] == path[-1]))
    return chains


def _clean(coords, closed, clip_poly, simplify=SIMPLIFY_M):
    """Clip to the boundary, simplify, drop tiny loops/slivers -> list of (coords, closed)."""
    if len(coords) < 2:
        return []
    ls = LineString(coords)
    out = []
    if clip_poly is not None:
        g = ls.intersection(clip_poly)
        if g.is_empty:
            return []
        parts = [g] if isinstance(g, LineString) else list(getattr(g, "geoms", []))
        pieces = []
        for p in parts:
            if isinstance(p, LineString):
                pieces.append(p)
            elif isinstance(p, MultiLineString):
                pieces += list(p.geoms)
    else:
        pieces = [ls]
    for p in pieces:
        if p.length <= 0:
            continue
        was_closed = p.is_closed
        q = p.simplify(simplify, preserve_topology=True) if simplify else p
        if q.is_empty or len(q.coords) < 2:
            continue
        if was_closed and p.length < MIN_LOOP_M:
            continue
        if not was_closed and p.length < MIN_LINE_M:
            continue
        out.append(([(round(x, 3), round(y, 3)) for x, y in q.coords], bool(was_closed and q.is_closed)))
    return out


# ------------------------------------------------------------------ labels
def _fmt_z(z_abs: float, interval: float) -> str:
    dec = 2 if interval < 0.5 - 1e-9 else 1
    return f"{z_abs:.{dec}f}"


def _label_angle(line, dist: float) -> float:
    """Rotation (radians) along the line at arc length `dist`, kept readable (-90, 90]."""
    a = line.interpolate(max(0.0, dist - 1.5))
    b = line.interpolate(min(line.length, dist + 1.5))
    ang = math.atan2(b.y - a.y, b.x - a.x)
    if ang > math.pi / 2 + 1e-9:
        ang -= math.pi
    elif ang <= -math.pi / 2 + 1e-9:
        ang += math.pi
    return ang


def place_labels(major_lines: list[dict], interval: float, inside_poly=None, spacing: float = LABEL_SPACING_M,
                 min_line: float = LABEL_MIN_LINE_M, max_per_line: int = LABEL_MAX_PER_LINE) -> list[dict]:
    """Labels on major lines: at the polyline midpoint (or the nearest spot along the line that keeps `spacing` m from
    every other label), rotated along the line, none on lines shorter than `min_line`. Longest lines first.
    -> [{x, y, angle, text, z_abs}]"""
    placed: list[dict] = []
    px = np.empty(0)      # placed label x / y (rounded, as in `placed`) for the spacing test
    py = np.empty(0)
    sp2 = spacing ** 2
    with_ls = [(LineString(m["points"]), m) for m in major_lines if len(m["points"]) >= 2]
    with_ls.sort(key=lambda item: -item[0].length)
    for ls, m in with_ls:
        if ls.length < min_line:
            continue
        mid = ls.length / 2
        step = max(2.0, min(10.0, ls.length / 40))
        offs = [0.0]
        k = 1
        while k * step <= ls.length / 2 + 1e-9:
            offs += [k * step, -k * step]
            k += 1
        n_line = 0
        for o in offs:
            d = mid + o
            pt = ls.interpolate(d)
            if inside_poly is not None and not inside_poly.contains(pt):
                continue
            if len(px) and ((pt.x - px) ** 2 + (pt.y - py) ** 2 < sp2).any():
                continue
            placed.append({"x": round(pt.x, 3), "y": round(pt.y, 3), "angle": round(_label_angle(ls, d), 4),
                           "text": _fmt_z(m["z_abs"], interval), "z_abs": m["z_abs"]})
            px, py = np.append(px, placed[-1]["x"]), np.append(py, placed[-1]["y"])
            n_line += 1
            if n_line >= max_per_line:
                break
    return placed


# ------------------------------------------------------------------ public
def _clip_polygon(site: dict):
    """Prepared site-boundary polygon (made valid), or None when the site has no usable boundary."""
    bnd = site["terrain"].get("boundary") or site.get("boundary")
    if not (bnd and len(bnd) >= 3):
        return None
    clip = Polygon([(p[0], p[1]) for p in bnd])
    if not clip.is_valid:
        clip = clip.buffer(0)
    shapely.prepare(clip)
    return clip


def _level_ks(info: dict, iv: float) -> range:
    """Integer multiples k of the interval strictly between the lowest and highest terrain point."""
    return range(math.floor((info["z_min"] - 1e-9) / iv) + 1, math.ceil((info["z_max"] + 1e-9) / iv))


def build_contours(site: dict, interval: float | None = None, scale: float = 1.0, labels: bool = True,
                   simplify: float = SIMPLIFY_M, label_spacing: float = LABEL_SPACING_M) -> dict:
    """Contours of site.json's terrain.
    -> {interval, auto_interval, relief_m, major_every, minor: [{z_abs, points, closed}], major: [...],
        labels: [{x, y, angle, text, z_abs}], counts: {minor, major, labels, vertices}, warnings}
    Coordinates are site metres (= Archicad project metres, x east / y north)."""
    info = resolve_interval(site, interval, scale)
    iv = info["interval"]
    e0 = float((site.get("origin") or {}).get("elevation") or 0.0)
    P, T = _tin(site)
    clip = _clip_polygon(site)
    out = {"interval": iv, "auto_interval": info["auto_interval"], "interval_source": info["source"],
           "relief_m": info["relief_m"], "z_min": info["z_min"], "z_max": info["z_max"], "major_every": MAJOR_EVERY,
           "minor": [], "major": [], "labels": [], "warnings": []}
    if info["note"]:
        out["warnings"].append(info["note"])
    if len(T):
        for k in _level_ks(info, iv):
            z_abs = round(k * iv, 6)
            kind = "major" if k % MAJOR_EVERY == 0 else "minor"
            for coords, closed in _level_chains(P, T, z_abs - e0):
                for pts, cl in _clean(coords, closed, clip, simplify):
                    out[kind].append({"z_abs": z_abs, "points": pts, "closed": cl})
    else:
        out["warnings"].append("terrain has too few points for contours")
    if labels:
        inside = clip.buffer(-1.0) if clip is not None else None
        if inside is not None:
            shapely.prepare(inside)
        out["labels"] = place_labels(out["major"], iv, inside, spacing=label_spacing)
    out["counts"] = {"minor": len(out["minor"]), "major": len(out["major"]), "labels": len(out["labels"]),
                     "vertices": sum(len(m["points"]) for m in out["minor"] + out["major"])}
    return out


# ------------------------------------------------------------------ stepped terraces
def _poly_parts(geom) -> list:
    out = []
    stack = [geom]
    while stack:
        g = stack.pop()
        if g is None or g.is_empty:
            continue
        if isinstance(g, Polygon):
            out.append(g)
        elif hasattr(g, "geoms"):
            stack.extend(reversed(list(g.geoms)))
    return out


def _poly_dict(g) -> dict | None:
    if not isinstance(g, Polygon) or g.is_empty or g.area < 1.0:
        return None
    ext = [(round(float(x), 3), round(float(y), 3)) for x, y in list(g.exterior.coords)[:-1]]
    if len(ext) < 3:
        return None
    holes = [[(round(float(x), 3), round(float(y), 3)) for x, y in list(r.coords)[:-1]]
             for r in g.interiors if len(r.coords) >= 4 and Polygon(r).area >= 0.5]
    return {"polygon": ext, "holes": holes}


def _superlevel_polys(P, T, level_rel: float, clip_poly, simplify: float = SIMPLIFY_M) -> list[dict]:
    """Polygons of the TIN surface strictly above `level_rel`, clipped to `clip_poly`."""
    if not len(T):
        return []
    z = P[:, 2] - level_rel
    z = np.where(np.abs(z) < EPS, EPS, z)
    s = z[T] > 0
    full = s.all(axis=1)
    mixed = s.any(axis=1) & ~full
    pieces = []
    if full.any():
        pieces.extend(shapely.polygons(P[:, :2][T[full]]).tolist())
    if mixed.any():
        pieces.extend(_clipped_triangles(P, z, T[mixed], s[mixed]))
    if not pieces:
        return []
    pieces = np.array(pieces, dtype=object)
    pieces = pieces[~shapely.is_empty(pieces)]
    bad = ~shapely.is_valid(pieces)
    if bad.any():
        pieces[bad] = shapely.make_valid(pieces[bad])
    try:    # triangles of a TIN share exact edge points, so the (much faster) coverage union applies
        u = shapely.coverage_union_all(pieces)
    except shapely.errors.GEOSException:        # "incorrectly noded" input (overlaps / T-junctions): general union
        u = unary_union(pieces)
    if clip_poly is not None:
        u = u.intersection(clip_poly)
    if u.is_empty:
        return []
    if simplify:
        u = u.simplify(simplify, preserve_topology=True)
    return [d for d in map(_poly_dict, _poly_parts(u)) if d is not None]


def _clipped_triangles(P, z, Tm, sm) -> list:
    """Part of each sign-mixed triangle (vertices Tm, signs sm) above the level, as polygons in triangle order.
    Ring = the positive vertices plus the edge crossings, walking the edges (v0,v1), (v1,v2), (v2,v0); each crossing
    is computed from the lower-index vertex so neighbouring triangles get bit-identical points (clean union)."""
    na = np.roll(Tm, -1, axis=1)
    cross = sm != np.roll(sm, -1, axis=1)
    lo, hi = np.minimum(Tm, na), np.maximum(Tm, na)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(cross, z[lo] / (z[lo] - z[hi]), 0.0)               # crossing parameter (unused off the crossings)
    xy = np.stack([P[lo, 0] + t * (P[hi, 0] - P[lo, 0]), P[lo, 1] + t * (P[hi, 1] - P[lo, 1])], axis=-1)
    own = np.stack([P[Tm, 0], P[Tm, 1]], axis=-1)
    ring = np.stack([own, xy], axis=2).reshape(len(Tm), 6, 2)           # per edge: vertex a, then the crossing
    ok = np.stack([sm, cross], axis=2).reshape(len(Tm), 6)
    n = ok.sum(axis=1)
    out = np.empty(len(Tm), dtype=object)
    for k in (3, 4):
        sel = n == k
        if sel.any():
            out[sel] = shapely.polygons(ring[sel][ok[sel]].reshape(-1, k, 2))
    return [g for g in out if g is not None]


def stepped_terraces(site: dict, interval: float | None = None, scale: float = 1.0,
                     simplify: float = SIMPLIFY_M) -> dict:
    """Horizontal contour-stepped terraces of site.json's terrain.
    -> {interval, auto_interval, relief_m, z_min_rel, z_max_rel, steps: [{level_abs, level_rel, z_bot, z_top, is_base, polygons}]}.
    Step 0 covers the site boundary (`is_base=True`); each subsequent step holds all areas where z >= level, extruded vertically from the step below."""
    info = resolve_interval(site, interval, scale)
    iv = info["interval"]
    e0 = float((site.get("origin") or {}).get("elevation") or 0.0)
    P, T = _tin(site)
    clip = _clip_polygon(site)
    zs = [float(p[2]) for p in site["terrain"]["points"]]
    z_min_rel, z_max_rel = min(zs), max(zs)
    levels = [round(k * iv, 6) for k in _level_ks(info, iv)]
    if levels and (levels[0] - e0) - z_min_rel < 0.05 * iv:
        levels = levels[1:]
    base_polys = [d for p in _poly_parts(clip) if (d := _poly_dict(p)) is not None] if clip is not None else []
    base_top = z_max_rel if not levels else z_min_rel
    steps = []
    if base_polys:
        steps.append({"level_abs": round(e0 + base_top, 6), "level_rel": round(base_top, 4),
                      "z_bot": round(z_min_rel, 4), "z_top": round(base_top, 4),
                      "is_base": True, "polygons": base_polys})
    prev_top = base_top
    for z_abs in levels:
        lrel = z_abs - e0
        polys = _superlevel_polys(P, T, lrel, clip, simplify)
        if not polys:
            continue
        steps.append({"level_abs": z_abs, "level_rel": round(lrel, 4), "z_bot": round(prev_top, 4),
                      "z_top": round(lrel, 4), "is_base": False, "polygons": polys})
        prev_top = lrel
    if len(T) and z_max_rel - prev_top >= 0.25 * iv:
        peak_mid = 0.5 * (prev_top + z_max_rel)
        polys = _superlevel_polys(P, T, peak_mid, clip, simplify)
        if polys:
            steps.append({"level_abs": round(e0 + z_max_rel, 6), "level_rel": round(z_max_rel, 4),
                          "z_bot": round(prev_top, 4), "z_top": round(z_max_rel, 4),
                          "is_base": False, "polygons": polys})
            prev_top = z_max_rel
    return {"interval": iv, "auto_interval": info["auto_interval"], "relief_m": info["relief_m"],
            "z_min_rel": round(z_min_rel, 4), "z_max_rel": round(prev_top, 4), "steps": steps}
