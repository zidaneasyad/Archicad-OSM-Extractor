"""Terrain as Tapir `CreateMorphs` payload items (terrain type "morph"). Pure payload code, no Archicad calls.

items[0]  terrain body: one closed Solid Morph = TIN top faces + vertical side faces along the TIN boundary
          down to (z_min - skirt) + one bottom face (boundary loops; interior holes become polygon holes).
          edgeDefault "HardHidden": hides the triangulation lines in 3D (only honoured by Archicad 29+).
          Closed + watertight on purpose: the terrain is a solid-element-operation target (roads/water subtract).
items[1:] contour wire Morphs ("minor", then "major"; empty ones skipped): bodyType "Surface", wireEdges only,
          edgeDefault "HardVisible" (two Morphs so pens/layers can differ). Lifted CONTOUR_LIFT m above the
          surface to avoid z-fighting with the terrain faces.

Orientation (vertexIds CCW seen from outside): top faces CCW in plan (normal up); sides outward; bottom face
reversed (CW in plan, normal down), its holes CCW in plan (opposite the outer ring; hole winding is UNVERIFIED live).

Z / basePoint convention: basePoint = (bbox min x, bbox min y, dz); every vertex is LOCAL to it, so
world = basePoint + vertex, and local z = site-relative z (relative to origin.elevation, like contours._tin)
 -> world z = z_rel + dz. Same dz convention as build_plan.terrain_stepped_slabs / terrain_mesh.
TODO-LIVE: UNVERIFIED in Archicad whether Morph z is project-zero-relative (assumed, floorIndex omitted) or
level-relative like the Mesh gotcha (build_plan._z_relative_to_level, M5). If level-relative, subtract the
level elevation from dz at the caller. Also unverified: that `bodyType: "Surface"` is kept (may come back Solid).
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from . import contours as ct
from .build_plan import TERRAIN_SKIRT

CONTOUR_LIFT = 0.01        # m, contour wires sit this far above the terrain surface (z-fighting)
MAX_FACES_WARN = 50_000    # faces in one Morph above which import becomes risky/slow
MAX_VERTS_WARN = 50_000


def _r(v: float) -> float:
    return round(float(v), 4)


def _boundary_loops(T: np.ndarray) -> list[list[int]]:
    """Directed boundary loops of a CCW triangle set (interior on the left)."""
    edges = set()
    for a, b, c in T:
        edges.update(((int(a), int(b)), (int(b), int(c)), (int(c), int(a))))
    bnd = [(u, v) for (u, v) in edges if (v, u) not in edges]
    out: dict[int, list[int]] = {}
    for u, v in bnd:
        out.setdefault(u, []).append(v)
    loops, used = [], set()
    for u0, v0 in bnd:
        if (u0, v0) in used:
            continue
        loop, u, v = [u0], u0, v0
        while (u, v) not in used:
            used.add((u, v))
            if v == u0:
                break
            loop.append(v)
            nxt = [w for w in out[v] if (v, w) not in used]
            if not nxt:
                break
            u, v = v, nxt[0]
        loops.append(loop)
    return loops


def _terrain_body(P: np.ndarray, T: np.ndarray, skirt: float):
    """-> (vertices [(x, y, z_rel)], polygons). Only TIN-used points are emitted."""
    used = np.unique(T)
    remap = {int(o): i for i, o in enumerate(used)}
    V = [tuple(P[o]) for o in used]
    z_bot = float(P[used, 2].min()) - skirt
    polys = [{"vertexIds": [remap[int(a)], remap[int(b)], remap[int(c)]]} for a, b, c in T]
    loops = [[remap[v] for v in lp] for lp in _boundary_loops(T)]
    bot_of = {}
    for lp in loops:                                  # bottom copies of every boundary vertex
        for v in lp:
            if v not in bot_of:
                bot_of[v] = len(V)
                V.append((V[v][0], V[v][1], z_bot))
    for lp in loops:                                  # side faces: a->b has the interior on its left, outside right
        for i, a in enumerate(lp):
            b = lp[(i + 1) % len(lp)]
            polys.append({"vertexIds": [bot_of[a], bot_of[b], b, a]})
    rings = []
    for lp in loops:
        poly = Polygon([(V[v][0], V[v][1]) for v in lp])
        rings.append((lp, poly, poly.exterior.is_ccw))
    holes = [(lp, poly) for lp, poly, ccw in rings if not ccw]
    for lp, poly, ccw in rings:
        if not ccw:
            continue
        mine = [h for h, hp in holes if poly.buffer(1e-9).contains(hp.representative_point())]
        face = {"vertexIds": [bot_of[v] for v in reversed(lp)]}          # CW in plan = outward (down)
        if mine:
            face["holes"] = [{"vertexIds": [bot_of[v] for v in reversed(h)]} for h in mine]  # CCW in plan
        polys.append(face)
    return V, polys


def _contour_morph(lines: list[dict], e0: float, base: tuple, lift: float, layer_index):
    verts, wires = [], []
    for ln in lines:
        pts = ln["points"]
        z = ln["z_abs"] - e0 + lift
        k0 = len(verts)
        verts += [{"x": _r(p[0] - base[0]), "y": _r(p[1] - base[1]), "z": _r(z)} for p in pts]
        ids = list(range(k0, k0 + len(pts)))
        wires += [{"vertexIds": [a, b]} for a, b in zip(ids, ids[1:])]
        if ln.get("closed") and len(pts) > 2:
            wires.append({"vertexIds": [ids[-1], ids[0]]})
    if not wires:
        return None
    it = {"basePoint": {"x": _r(base[0]), "y": _r(base[1]), "z": _r(base[2])},
          "body": {"bodyType": "Surface", "edgeDefault": "HardVisible", "vertices": verts, "polygons": [],
                   "wireEdges": wires}}
    if layer_index is not None:
        it["layerIndex"] = layer_index
    return it


def terrain_morph_items(site: dict, dz: float = 0.0, skirt: float = TERRAIN_SKIRT, interval: float | None = None,
                        scale: float = 1.0, layer_index: int | None = None) -> tuple[list[dict], dict]:
    """-> (items, info). items[0] = closed Solid terrain Morph, then minor / major contour wire Morphs (see module doc).
    info = {interval, triangles, vertices, minor_lines, major_lines, warnings}. Empty/degenerate terrain -> ([], info)
    with a warning. `layer_index` goes on every item when given (layer must be visible + unlocked)."""
    warnings = ["Morph edge hiding (edgeDefault HardHidden) needs Archicad 29+; older versions show the TIN lines"]
    info = {"interval": None, "triangles": 0, "vertices": 0, "minor_lines": 0, "major_lines": 0, "warnings": warnings}
    P, T = ct._tin(site)
    if len(T):                                        # drop degenerate triangles, force CCW in plan
        a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        keep = np.abs(area) > 1e-9
        T, area = T[keep], area[keep]
        T = np.where((area < 0)[:, None], T[:, ::-1], T)
    if not len(T):
        warnings.append("terrain has too few points for a Morph terrain")
        return [], info
    V, polys = _terrain_body(P, T, skirt)
    Va = np.array(V, dtype=float)
    base = (float(Va[:, 0].min()), float(Va[:, 1].min()), float(dz))
    body = {"bodyType": "Solid", "edgeDefault": "HardHidden",
            "vertices": [{"x": _r(v[0] - base[0]), "y": _r(v[1] - base[1]), "z": _r(v[2])} for v in V],
            "polygons": polys, "wireEdges": []}
    terrain = {"basePoint": {"x": _r(base[0]), "y": _r(base[1]), "z": _r(base[2])}, "body": body}
    if layer_index is not None:
        terrain["layerIndex"] = layer_index
    items = [terrain]
    info.update(triangles=len(T), vertices=len(V))
    if len(polys) > MAX_FACES_WARN or len(V) > MAX_VERTS_WARN:
        warnings.append(f"terrain Morph is large ({len(polys)} faces, {len(V)} vertices): one Morph this size may "
                        f"import slowly or time out; use a coarser DEM grid or the Mesh/Slab terrain")
    cd = ct.build_contours(site, interval=interval, scale=scale, labels=False)
    info["interval"] = cd["interval"]
    warnings += list(cd.get("warnings") or [])
    e0 = float((site.get("origin") or {}).get("elevation") or 0.0)
    for kind in ("minor", "major"):
        info[f"{kind}_lines"] = len(cd[kind])
        it = _contour_morph(cd[kind], e0, base, CONTOUR_LIFT, layer_index)
        if it:
            items.append(it)
    return items, info
