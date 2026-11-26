"""Offline tests for M4 shrubs/hedges/forest scatter: scatter function, converter keys, builder plan, groups,
super-light flags, fetch hooks."""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box

import server
from osmctx import build_plan as bp
from osmctx import fetch as F
from osmctx import limits
from osmctx import vegetation as veg
from osmctx.boundary import bbox, square_boundary, to_polygon

from test_convert import flat, node, run, way


@pytest.fixture(autouse=True)
def _no_overture_network(monkeypatch):
    """Keep these tests offline: F.fetch would otherwise query the live Overture S3 bucket."""
    monkeypatch.setattr(F.OVT, "try_fetch", lambda *a, **k: {"status": "failed", "error": "offline test stub"})
    monkeypatch.setattr(F.L2, "detect_region", lambda *a, **k: None)   # no LoD2 job (it would query the live LGL server)


# ------------------------------------------------------------------ scatter (pure)
def _min_dist(pts):
    a = np.asarray(pts)
    d = np.hypot(a[:, None, 0] - a[None, :, 0], a[:, None, 1] - a[None, :, 1])
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def test_scatter_deterministic_density_spacing_inside():
    sq = box(0, 0, 100, 100)
    p1, i1 = veg.scatter(sq, 40.0, 4.0, "way/1", seed=1)
    p2, _ = veg.scatter(sq, 40.0, 4.0, "way/1", seed=1)
    assert p1 == p2                                              # deterministic
    assert p1 != veg.scatter(sq, 40.0, 4.0, "way/2", seed=1)[0]  # another area -> another pattern
    assert p1 != veg.scatter(sq, 40.0, 4.0, "way/1", seed=2)[0]  # another seed -> another pattern
    assert i1["target"] == 250 and len(p1) == 250 and not i1["capped"]   # 10000 m2 / 40 m2
    assert _min_dist(p1) >= 4.0 - 1e-9
    assert all(0 <= x <= 100 and 0 <= y <= 100 for x, y in p1)


def test_scatter_respects_holes_and_multipolygon():
    g = MultiPolygon([Polygon([(0, 0), (60, 0), (60, 60), (0, 60)], [[(20, 20), (40, 20), (40, 40), (20, 40)]]),
                      box(100, 0, 130, 30)])
    pts, info = veg.scatter(g, 40.0, 4.0, "r/7", seed=3)
    assert len(pts) == info["target"] == int(g.area // 40)
    assert all(g.contains(Point(x, y)) for x, y in pts)
    assert not any(20 < x < 40 and 20 < y < 40 for x, y in pts)


def test_scatter_cap_avoid_and_tight_area():
    sq = box(0, 0, 100, 100)
    pts, info = veg.scatter(sq, 1.0, 1.0, "c", cap=50)
    assert len(pts) == 50 and info["capped"]
    av = [(50.0, 50.0)]
    pts, _ = veg.scatter(sq, 40.0, 4.0, "a", avoid=av, avoid_r=2.0)
    assert min(math.hypot(x - 50, y - 50) for x, y in pts) >= 2.0
    pts, info = veg.scatter(box(0, 0, 10, 10), 5.0, 6.0, "t")    # 20 wanted, spacing 6 m fits ~4
    assert 1 <= len(pts) < info["target"]
    assert veg.scatter(None, 40, 4, "x")[0] == [] and veg.scatter(sq, 40, 4, "x", cap=0)[0] == []


def test_ndom_index_matches_brute_force():
    rng = np.random.default_rng(5)
    cx, cy = rng.uniform(0, 50, 4000), rng.uniform(0, 50, 4000)
    v = rng.uniform(0, 10, 4000)
    v[::7] = np.nan
    idx = veg.NdomIndex(cx, cy, v)
    for x, y in [(10, 10), (25.3, 40.1), (49, 1)]:
        m = ((cx - x) ** 2 + (cy - y) ** 2 <= 4.0) & np.isfinite(v)
        want = float(v[m].max()) if m.any() else None
        assert idx.measure(x, y, 2.0, 0.0, 99.0) == want
    assert idx.measure(10, 10, 2.0, 50.0, 99.0) is None      # outside accept window


# ------------------------------------------------------------------ converter
WOOD = [(-40, -40), (40, -40), (40, 40), (-40, 40), (-40, -40)]     # 80 x 80 m = 6400 m2


def _wood_els(tags=None):
    return [way(1, WOOD, {"natural": "wood", **(tags or {})}),
            node(2, 0.0, 0.0, {"natural": "tree"})]


def test_forest_scatter_off_by_default_and_on(tmp_path):
    off = run(tmp_path / "a", _wood_els(), flat())
    assert [t["osm_id"] for t in off["trees"]] == ["node/2"]
    on = run(tmp_path / "b", _wood_els({"leaf_type": "needleleaved"}), flat(), settings={"forest_scatter": True})
    sc = [t for t in on["trees"] if t.get("source_area")]
    assert abs(len(sc) - 160) <= 2   # 1 per 40 m2 of the 6400 m2 wood (inside the 100 m boundary)
    assert all(t["source_area"] == "way/1" and t["osm_id"] == "way/1" for t in sc)
    assert all(t["height_source"] == "leaf_type" and t["height"] == 15.0 and t["tags"]["leaf_type"] == "needleleaved" for t in sc)
    single = next(t for t in on["trees"] if t["osm_id"] == "node/2")
    assert "source_area" not in single
    assert on["report"]["counts"]["trees_from_forest"] == len(sc)
    # min spacing 4 m between scattered trees, and from the mapped tree node (>= 2 m)
    assert _min_dist([(t["x"], t["y"]) for t in sc]) >= 4.0 - 0.01
    assert min(math.hypot(t["x"], t["y"]) for t in sc) >= 2.0 - 0.01
    again = run(tmp_path / "c", _wood_els({"leaf_type": "needleleaved"}), flat(), settings={"forest_scatter": True})
    assert [(t["x"], t["y"]) for t in again["trees"]] == [(t["x"], t["y"]) for t in on["trees"]]   # deterministic


def test_forest_scatter_density_and_wood_forest_overlap(tmp_path):
    els = [way(1, WOOD, {"natural": "wood"}), way(2, WOOD, {"landuse": "forest"})]   # same area drawn twice
    s = run(tmp_path, els, flat(), settings={"forest_scatter": True, "forest_area_per_tree": 100.0, "forest_min_spacing": 3.0})
    assert 62 <= len(s["trees"]) <= 64 and {t["source_area"] for t in s["trees"]} == {"way/1"}


def test_shrubs_and_hedges_converter(tmp_path):
    els = [node(1, 5.0, 5.0, {"natural": "shrub", "height": "2.2"}),
           node(2, -5.0, 5.0, {"natural": "shrub"}),
           way(3, [(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)], {"natural": "shrubbery"}),   # 400 m2 -> 66 shrubs
           way(4, [(-30, -30), (-10, -30), (-10, -10), (-30, -10), (-30, -30)], {"natural": "scrub"}),  # 400 m2 -> 20
           way(5, [(-45, 20), (-5, 20), (-5, 22)], {"barrier": "hedge", "height": "1.8", "width": "1.1"}),
           way(6, [(0, -45), (0, -20)], {"barrier": "hedge"}),
           way(7, [(0, 45), (0, 80)], {"barrier": "hedge"})]    # leaves the 100 m boundary at y = 50
    s = run(tmp_path, els, flat(), settings={"shrubbery_area_per_shrub": 6, "scrub_area_per_shrub": 20})
    sh = s["shrubs"]
    by = {}
    for x in sh:
        by.setdefault(x["osm_id"], []).append(x)
    assert len(by["node/1"]) == 1 and by["node/1"][0]["height"] == 2.2 and by["node/1"][0]["height_source"] == "tag"
    assert by["node/2"][0]["height"] == 1.5 and by["node/2"][0]["height_source"] == "default" and by["node/2"][0]["source_area"] is None
    assert 60 <= len(by["way/3"]) <= 66 and 18 <= len(by["way/4"]) <= 20
    assert all(x["source_area"] == "way/3" and x["kind"] == "shrubbery" for x in by["way/3"])
    assert all(10 <= x["x"] <= 30 and 10 <= x["y"] <= 30 for x in by["way/3"])
    assert _min_dist([(x["x"], x["y"]) for x in by["way/3"]]) >= 1.5 - 0.01
    assert all(x["crown_diameter"] <= 1.3 * 1.5 + 1e-6 for x in by["way/3"])
    assert s["report"]["counts"]["shrubs"] == len(sh)
    hg = {h["osm_id"]: h for h in s["hedges"]}
    assert (hg["way/5"]["height"], hg["way/5"]["height_source"], hg["way/5"]["width"], hg["way/5"]["width_source"]) == (1.8, "tag", 1.1, "tag")
    assert (hg["way/6"]["height"], hg["way/6"]["width"], hg["way/6"]["height_source"]) == (1.2, 0.8, "default")
    assert all(len(h["points"][0]) == 3 for h in s["hedges"])
    ys7 = [p[1] for p in hg["way/7"]["points"]]
    assert max(ys7) == pytest.approx(50.0, abs=0.05)                    # clipped at the boundary
    seg_max = max(math.hypot(b[0] - a[0], b[1] - a[1]) for h in s["hedges"] for a, b in zip(h["points"], h["points"][1:]))
    assert seg_max <= 6.0 + 1e-6                                          # densified
    assert s["report"]["counts"]["hedge_ways"] == 3
    off = run(tmp_path / "x", els, flat(), settings={"include": {"shrubs": False}})
    assert off["shrubs"] == [] and off["hedges"] == []


def test_shrub_hedge_measured_height_from_ndom(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from osmctx.convert import Frame, convert
    from test_convert import LAT0, LON0, make_cache
    els = [way(1, [(-20, 0), (20, 0)], {"barrier": "hedge"}), node(2, 0.0, 30.0, {"natural": "shrub"})]
    d = make_cache(tmp_path / "cache", els, flat())
    f = Frame(to_polygon(square_boundary(LAT0, LON0, 100.0)))
    n = 120
    arr = np.zeros((n, n), dtype="float32")
    x0, y0 = f.E0 - 60.0, f.N0 + 60.0

    def put(x0_, x1_, y0_, y1_, val):
        c0, c1 = int(np.floor(f.E0 + x0_ - x0)), int(np.ceil(f.E0 + x1_ - x0))
        r0, r1 = int(np.floor(y0 - (f.N0 + y1_))), int(np.ceil(y0 - (f.N0 + y0_)))
        arr[r0:r1, c0:c1] = val
    put(-22, 22, -1, 1, 1.9)         # the hedge, 1.9 m
    put(-1, 1, 29, 31, 2.6)          # the shrub, 2.6 m
    with rasterio.open(d / "ndom.tif", "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       crs=f"EPSG:{f.epsg}", transform=from_origin(x0, y0, 1.0, 1.0), nodata=-9999.0) as ds:
        ds.write(arr, 1)
    s = convert(d, None, tmp_path / "out")
    h = s["hedges"][0]
    assert h["height_source"] == "measured" and h["height"] == pytest.approx(1.9, abs=0.01)
    sh = s["shrubs"][0]
    assert sh["height_source"] == "measured" and sh["height"] == pytest.approx(2.6, abs=0.01)
    assert s["report"]["defaults_used"]["hedge_heights"] == {"measured": 1}


# ------------------------------------------------------------------ build plan: hedges + shrubs
def test_hedge_pieces_split_and_skip_tiny():
    pts = [[0, 0, 5], [20, 0, 7], [20.01, 0, 7], [20, 10, 7]]
    pcs = bp.hedge_pieces(pts, 8.0)
    assert [round(math.hypot(b[0] - a[0], b[1] - a[1]), 3) for a, b in pcs] == [6.667] * 3 + [5.0, 5.0]
    assert pcs[0][0] == (0, 0, 5) and pcs[2][1] == pytest.approx((20, 0, 7))
    assert bp.hedge_pieces([[0, 0, 0]], 8.0) == []


@pytest.mark.parametrize("p0,p1", [((0, 0, 3), (5, 0, 4)), ((0, 0, 3), (0, 5, 3)), ((10, 10, 0), (7, 6, 1)), ((0, 0, 2), (-4, 3, 2))])
def test_hedge_segment_geometry_centred_on_line(p0, p1):
    w, h = 0.8, 1.2
    o = bp.hedge_segment_object(p0, p1, w, h, dz=0.5)
    L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    d = o["dimensions"]
    assert (d["x"], d["y"], d["z"]) == (pytest.approx(L, abs=1e-3), w, h)
    assert o["coordinates"]["z"] == min(p0[2], p1[2]) + 0.5
    th = o["angle"]
    assert th == pytest.approx(math.atan2(p1[1] - p0[1], p1[0] - p0[0]), abs=1e-5)       # radians
    # box centre (local L/2, w/2 rotated) is the segment midpoint; bbox extent = rotated rectangle
    cx = o["coordinates"]["x"] + (L / 2) * math.cos(th) - (w / 2) * math.sin(th)
    cy = o["coordinates"]["y"] + (L / 2) * math.sin(th) + (w / 2) * math.cos(th)
    assert (cx, cy) == (pytest.approx((p0[0] + p1[0]) / 2, abs=1e-3), pytest.approx((p0[1] + p1[1]) / 2, abs=1e-3))
    bb = bp.hedge_expected_bbox(o)
    assert bb["zMax"] - bb["zMin"] == pytest.approx(h)
    assert bb["xMax"] - bb["xMin"] == pytest.approx(abs(p1[0] - p0[0]) + w * abs(math.sin(th)), abs=1e-3)


def test_hedge_leaves_and_schematic_param():
    o = bp.hedge_segment_object((0, 0, 0), (6, 0, 0), 0.8, 1.2)
    nul = next(p["value"] for p in o["parameters"] if p["name"] == "nul")
    assert nul == round(40 * (2 * 6 * 1.2 + 6 * 0.8)) and not any(p["name"] == "iDetLev3D" for p in o["parameters"])
    s = bp.hedge_segment_object((0, 0, 0), (6, 0, 0), 0.8, 1.2, schematic=True)
    assert {"name": "iDetLev3D", "value": 1} in s["parameters"]
    assert len(bp.hedge_objects({"points": [[0, 0, 0], [20, 0, 0]], "width": 1, "height": 1}, max_piece=8.0)) == 3


def test_shrub_object_and_parts():
    s = {"x": 1, "y": 2, "z": 3, "height": 1.2, "crown_diameter": 0.9}
    o = bp.shrub_object(s, "Shrubs", dz=0.1)
    assert o["dimensions"] == {"x": 1.5, "y": 1.5, "z": 1.2} and o["coordinates"]["z"] == 3.1   # min width 1.5
    o2 = bp.shrub_object(s, "Tree Plan")
    assert o2["dimensions"]["x"] == 0.9 and o2["parameters"] == [{"name": "iTopViewSymb", "value": 100}]
    ex = lambda n: n in {"Shrubs", "Hedge Model", "Tree Plan"}
    assert bp.resolve_shrub_parts("full3d", ex)[0] == {"shrub": "Shrubs", "hedge": "Hedge Model"}
    assert bp.resolve_shrub_parts("2d", ex)[0] == {"shrub": "Tree Plan", "hedge": "Hedge Model"}
    p, w = bp.resolve_shrub_parts("light", lambda n: False)
    assert p == {"shrub": None, "hedge": None} and len(w) == 2


def test_expected_counts_include_shrubs_and_hedge_segments():
    site = {"trees": [{}], "shrubs": [{}, {}], "hedges": [{"points": [[0, 0, 0], [20, 0, 0]]}]}
    assert bp.expected_counts(site)["Object"] == 1 + 2 + 3


# ------------------------------------------------------------------ groups
def test_group_plan_shrubs_hedges_forests_siblings_of_trees():
    created = {"terrain": [("T", "t")],
               "trees": [("t1", "node/1"), ("f1", "way/9"), ("f2", "way/9"), ("f3", "way/9"), ("g1", "way/10")],
               "tree_area_ids": ["way/9", "way/10"],
               "shrubs": [("s1", "way/20"), ("s2", "way/20"), ("s3", "node/5"), ("s4", "node/6"), ("s5", "way/21")],
               "shrub_area_ids": ["way/20", "way/21"],
               "hedges": [("h1", "way/30"), ("h2", "way/30"), ("h3", "way/31")]}
    p = bp.group_plan(created)
    kids = {k["name"]: k for k in p["children"] if isinstance(k, dict)}
    assert set(kids) == {"Trees", "Shrubs", "Hedges"}
    assert p["children"].count("T") == 1
    trees = kids["Trees"]["children"]
    assert "t1" in trees and "g1" in trees          # single tree / single-tree forest area: direct, no subgroup
    assert {"name": "Forest way/9", "children": ["f1", "f2", "f3"]} in trees
    shrubs = kids["Shrubs"]["children"]
    assert {"name": "Shrub area way/20", "children": ["s1", "s2"]} in shrubs
    assert "s3" in shrubs and "s4" in shrubs and "s5" in shrubs   # singles direct
    hedges = kids["Hedges"]["children"]
    assert {"name": "Hedge way/30", "children": ["h1", "h2"]} in hedges and "h3" in hedges
    order = [n["name"] for n in bp.group_nodes_postorder(p)]
    assert order.index("Forest way/9") < order.index("Trees") < order.index("OSM Site Context")
    assert bp.group_depth(p) == 3


def test_group_plan_without_vegetation_unchanged():
    p = bp.group_plan({"terrain": [("T", "t")], "buildings": [("B1", "b1"), ("B2", "b2")]})
    assert [k["name"] for k in p["children"] if isinstance(k, dict)] == ["Buildings"]


# ------------------------------------------------------------------ super-light + estimate
def test_super_light_vegetation_flags():
    assert limits.SUPER_LIGHT["include"]["shrubs"] is False
    assert limits.SUPER_LIGHT["forest_scatter"] is False and limits.SUPER_LIGHT["skip_ndom"] is True
    r = limits.resolve(6.0, {"include": {"trees": True, "shrubs": True}, "forest_scatter": True})
    assert r["settings"]["include"] == {"trees": True, **limits.SUPER_LIGHT["include"]}      # merged, user's other flags kept
    assert r["settings"]["forest_scatter"] is False and r["settings"]["skip_ndom"] is True
    o = limits.resolve(6.0, {"include": {"shrubs": True}, "forest_scatter": True}, override=True)
    assert o["settings"] == {"include": {"shrubs": True}, "forest_scatter": True}
    n = limits.resolve(0.1, {"forest_scatter": True})
    assert n["settings"] == {"forest_scatter": True}
    m = server.map_settings(r["settings"])
    assert "skip_ndom" not in m and m["forest_scatter"] is False and m["include"]["shrubs"] is False


def test_estimate_counts_vegetation():
    counts = {"buildings": 10, "roads": 0, "trees": 5, "water": 0, "hedges": 4, "shrubs": 3,
              "shrub_m2": 400, "forest_m2": 8000}
    tiles = {"source": "bw_dgm1", "dgm1_total": 0, "dgm1_missing": 0, "ndom_total": 0, "ndom_missing": 0, "copernicus_tiles": 0}
    base = limits.estimate(0.1, counts, "cache", tiles)
    assert base["elements"]["shrubs"] == int(3 + 400 * 0.5 / 10) and base["elements"]["hedge_segments"] == 24
    assert base["elements"]["scatter_trees"] == 0                 # scatter off by default
    on = limits.estimate(0.1, counts, "cache", tiles, settings={"forest_scatter": True})
    assert on["elements"]["scatter_trees"] == 100 and on["elements"]["trees"] == 105
    assert on["elements"]["total"] == base["elements"]["total"] + 100
    off = limits.estimate(0.1, counts, "cache", tiles, settings={"include": {"shrubs": False}})
    assert off["elements"]["shrubs"] == 0 and off["elements"]["hedge_segments"] == 0
    sl = limits.estimate(0.1, counts, "cache", tiles, preset=True, settings={"forest_scatter": True})
    assert sl["elements"]["shrubs"] == 0 and sl["elements"]["scatter_trees"] == 0


def test_count_from_osm_vegetation_extras():
    b = {"minlat": 48.5, "minlon": 9.0, "maxlat": 48.501, "maxlon": 9.001}
    osm = {"elements": [{"type": "way", "tags": {"barrier": "hedge"}}, {"type": "node", "tags": {"natural": "shrub"}, "lat": 48.5, "lon": 9.0},
                        {"type": "way", "tags": {"natural": "wood"}, "bounds": b},
                        {"type": "relation", "tags": {"natural": "scrub"}, "bounds": b}]}
    c = server.count_from_osm(osm)
    assert c["hedges"] == 1 and c["shrubs"] == 1 and 6000 < c["forest_m2"] < 9000 and c["shrub_m2"] == c["forest_m2"]


# ------------------------------------------------------------------ fetch hooks
def test_query_has_vegetation():
    q = F.build_query(bbox(to_polygon(square_boundary(48.5193, 9.0565, 100))))
    for k in ('"barrier"="hedge"', '"natural"="shrubbery"', '"natural"="scrub"', '"natural"="shrub"'):
        assert k in q
    c = F.count_categories({"elements": [{"tags": {"barrier": "hedge"}}, {"tags": {"natural": "shrub"}},
                                         {"tags": {"natural": "scrub"}}]})
    assert (c["hedge"], c["shrub"], c["shrubbery/scrub"]) == (1, 1, 1)


def _fake_cache(root, rev=None, dem_id="bw_dgm1", ndom=False):
    d = Path(root) / "k"
    d.mkdir(parents=True)
    for f in ("osm.json", "dem.tif"):
        (d / f).write_text('{"elements": []}' if f == "osm.json" else "x")
    meta = {"boundary": square_boundary(48.5193, 9.0565, 100), "bbox": {"south": 1, "west": 2, "north": 3, "east": 4},
            "osm_bbox": {"south": 1, "west": 2, "north": 3, "east": 4}, "dem_source_id": dem_id}
    if rev:
        meta["osm_query_rev"] = rev
    if ndom:
        meta["ndom"] = {"available": False}
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def test_fetch_cached_refreshes_osm_once_and_ndom_flag(tmp_path, monkeypatch):
    b = square_boundary(48.5193, 9.0565, 100)
    d = _fake_cache(tmp_path)
    monkeypatch.setattr(F, "cache_key", lambda boundary, src: "k")
    calls = {"q": [], "ndom": 0}
    monkeypatch.setattr(F, "overpass", lambda q: calls["q"].append(q) or ({"elements": [{"type": "node", "id": 1, "tags": {"natural": "shrub"}}]}, "mock"))
    monkeypatch.setattr(F, "ensure_ndom", lambda cache_d, tile_root=None: calls.__setitem__("ndom", calls["ndom"] + 1) or {})
    F.fetch(b, str(tmp_path), "bw_dgm1", ndom=False)
    assert len(calls["q"]) == 1 and '"barrier"="hedge"' in calls["q"][0] and calls["ndom"] == 0   # no nDOM when ndom=False
    meta = json.loads((d / "meta.json").read_text())
    assert meta["osm_query_rev"] == F.OSM_QUERY_REV and "shrub" in (d / "osm.json").read_text()
    F.fetch(b, str(tmp_path), "bw_dgm1", ndom=True)                  # second call: no re-query, nDOM now allowed
    assert len(calls["q"]) == 1 and calls["ndom"] == 1


def test_refresh_osm_failure_keeps_old(tmp_path, monkeypatch):
    d = _fake_cache(tmp_path)
    monkeypatch.setattr(F, "overpass", lambda q: (_ for _ in ()).throw(RuntimeError("down")))
    assert F.refresh_osm(d) is False
    assert "RuntimeError" in json.loads((d / "meta.json").read_text())["osm_refresh_error"]
    assert (d / "osm.json").read_text() == '{"elements": []}'


def test_server_fetch_skips_ndom_only_for_preset(monkeypatch):
    from fastapi.testclient import TestClient
    seen = []
    monkeypatch.setattr(F, "fetch", lambda b, cache_dir="cache", dem_source="auto", **kw: seen.append(kw) or "cache/x")
    c = TestClient(server.app)

    def run_job(km2, override=False):
        r = c.post("/api/fetch", json={"boundary": square_boundary(48.5193, 9.0565, (km2 * 1e6) ** 0.5),
                                       "settings": {"dem_source": "copernicus"}, "override": override})
        jid = r.json()["job"]
        for _ in range(100):
            if c.get(f"/api/job/{jid}").json()["state"] != "running":
                break
            time.sleep(0.05)
        return r.json()
    assert run_job(2)["skip_ndom"] is False and seen[-1] == {}
    assert run_job(7)["skip_ndom"] is True and seen[-1] == {"ndom": False}
    assert run_job(7, override=True)["skip_ndom"] is False and seen[-1] == {}
