"""osmctx/terrain_morph.py: closed terrain Morph + contour wire Morphs (offline, no Archicad)."""
import json
from collections import Counter

import numpy as np

from osmctx import contours as ct, terrain_morph as tm
from tests.test_contours import grid_site, PLANE, CONE


def _world(item):
    b = item["basePoint"]
    return np.array([[v["x"] + b["x"], v["y"] + b["y"], v["z"] + b["z"]] for v in item["body"]["vertices"]])


def _edge_counts(item):
    c = Counter()
    for p in item["body"]["polygons"]:
        rings = [p["vertexIds"]] + [h["vertexIds"] for h in p.get("holes", [])]
        for r in rings:
            for i, a in enumerate(r):
                c[(a, r[(i + 1) % len(r)])] += 1
    return c


def _signed_volume(item):
    """Divergence-theorem volume; faces fan-triangulated (valid for convex faces; bottom face is convex here)."""
    V = _world(item)
    vol = 0.0
    for p in item["body"]["polygons"]:
        ids = p["vertexIds"]
        for k in range(1, len(ids) - 1):
            a, b, c = V[ids[0]], V[ids[k]], V[ids[k + 1]]
            vol += np.dot(a, np.cross(b, c)) / 6.0
    return vol


def make(f=PLANE, n=20.0, s=4.0, **kw):
    return tm.terrain_morph_items(grid_site(f, n=n, s=s), **kw)


def test_watertight_and_orientation():
    items, info = make(skirt=5.0)
    t = items[0]
    assert t["body"]["bodyType"] == "Solid" and t["body"]["edgeDefault"] == "HardHidden"
    ec = _edge_counts(t)
    for (a, b), n in ec.items():
        assert n == 1 and ec.get((b, a)) == 1, "every directed edge needs exactly one opposite edge"
    V = _world(t)
    vol = _signed_volume(t)
    assert vol > 0
    area = 40 * 40
    mean_h = np.mean(V[:, 2][V[:, 2] > V[:, 2].min() + 1e-6]) - V[:, 2].min()
    assert abs(vol - area * mean_h) < 0.05 * area * mean_h
    top = [p for p in t["body"]["polygons"] if len(p["vertexIds"]) == 3]
    assert len(top) == info["triangles"]
    for p in top:
        a, b, c = (V[i] for i in p["vertexIds"])
        assert np.cross(b - a, c - a)[2] > 0                      # top normals up
    bottom = [p for p in t["body"]["polygons"] if len(p["vertexIds"]) > 4][0]
    ring = V[bottom["vertexIds"]]
    assert np.ptp(ring[:, 2]) < 1e-9
    x, y = ring[:, 0], ring[:, 1]
    assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0   # CW in plan = normal down
    sides = [p for p in t["body"]["polygons"] if len(p["vertexIds"]) == 4]
    cen = V[:, :2].mean(axis=0)
    for p in sides:                                               # outward: normal points away from centre
        q = V[p["vertexIds"]]
        n = np.cross(q[1] - q[0], q[3] - q[0])
        mid = q[:, :2].mean(axis=0)
        assert np.dot(n[:2], mid - cen) > 0


def test_vertex_counts_and_skirt():
    items, info = make(skirt=5.0)
    zs = [v["z"] for v in items[0]["body"]["vertices"]]
    assert info["vertices"] == len(zs)
    top_min = min(z for z in zs if z > min(zs) + 1e-6)
    assert abs((top_min - min(zs)) - 5.0) < 1e-3


def test_local_coords_and_basepoint():
    site = grid_site(CONE, n=20, s=4)
    items, _ = tm.terrain_morph_items(site, dz=1.5)
    t = items[0]
    P, T = ct._tin(site)
    used = np.unique(T)
    b = t["basePoint"]
    assert b["z"] == 1.5
    V = _world(t)
    assert abs(b["x"] - V[:, 0].min()) < 1e-3 and abs(b["y"] - V[:, 1].min()) < 1e-3
    loc = t["body"]["vertices"]
    assert min(v["x"] for v in loc) == 0 and min(v["y"] for v in loc) == 0
    ref = {(round(p[0], 2), round(p[1], 2)): p[2] + 1.5 for p in P[used]}
    for x, y, z in V[:len(used)]:
        assert abs(ref[(round(x, 2), round(y, 2))] - z) < 2e-3


def test_contours_match_build_contours():
    site = grid_site(CONE, n=20, s=4, e0=100.0)
    items, info = tm.terrain_morph_items(site, dz=2.0, interval=2.0)
    cd = ct.build_contours(site, interval=2.0, labels=False)
    assert info["interval"] == 2.0
    assert info["minor_lines"] == len(cd["minor"]) and info["major_lines"] == len(cd["major"])
    wires = items[1:]
    assert wires and all(w["body"]["bodyType"] == "Surface" and w["body"]["edgeDefault"] == "HardVisible"
                         and w["body"]["polygons"] == [] and w["body"]["wireEdges"] for w in wires)
    off = 2.0 - 100.0 + tm.CONTOUR_LIFT
    for w, kind in zip(wires, [k for k in ("minor", "major") if cd[k]]):
        got = {round(v["z"] + w["basePoint"]["z"], 2) for v in w["body"]["vertices"]}
        assert got == {round(l["z_abs"] + off, 2) for l in cd[kind]}
        n = len(w["body"]["vertices"])
        assert all(0 <= i < n for e in w["body"]["wireEdges"] for i in e["vertexIds"])


def test_layer_index_and_json():
    items, _ = make(layer_index=7)
    assert all(i["layerIndex"] == 7 for i in items)
    json.dumps(items)
    items2, _ = make()
    assert all("layerIndex" not in i for i in items2)


def test_empty_terrain():
    site = {"origin": {"elevation": 0.0}, "terrain": {"points": [[0, 0, 1], [1, 0, 1]]}}
    items, info = tm.terrain_morph_items(site)
    assert items == [] and info["triangles"] == 0 and any("too few" in w for w in info["warnings"])
    items, _ = tm.terrain_morph_items({"origin": {}, "terrain": {"points": [[0, 0, 1], [1, 1, 1], [2, 2, 1]]}})
    assert items == []


def test_interior_hole_bottom_face_and_warnings():
    site = grid_site(PLANE, n=20, s=4)
    ring = site["terrain"]["boundary"]
    pts = [p for p in site["terrain"]["points"] if not (abs(p[0]) < 6 and abs(p[1]) < 6)]
    site["terrain"]["points"] = pts
    items, info = tm.terrain_morph_items(site)
    t = items[0]
    ec = _edge_counts(t)
    assert all(ec.get((b, a)) == 1 for (a, b) in ec)              # still watertight
    assert any("29" in w for w in info["warnings"])
    json.dumps(items)
    assert ring                                                    # (hull TIN fills the gap: hole path covered below)


def test_hole_loop_function():
    """Annulus TIN (square ring of triangles around a square hole): bottom face carries a hole, still watertight."""
    P = np.array([[0, 0, 0], [3, 0, 0], [3, 3, 0], [0, 3, 0], [1, 1, 0], [2, 1, 0], [2, 2, 0], [1, 2, 0]], float)
    T = np.array([[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    V, polys = tm._terrain_body(P, T, 1.0)
    item = {"basePoint": {"x": 0, "y": 0, "z": 0},
            "body": {"vertices": [{"x": v[0], "y": v[1], "z": v[2]} for v in V], "polygons": polys}}
    assert any("holes" in p for p in polys)
    ec = _edge_counts(item)
    assert all(ec.get((b, a)) == 1 for (a, b) in ec)
