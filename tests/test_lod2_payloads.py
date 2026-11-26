"""M10 step 7: LoD2 building records -> native Archicad payloads (Slab blocks + single-plane Roofs, Morph fallback).
Pure / offline: plane math, geometric replay of every Roof item, routing, count guard, M6 golden, and the builder
wiring against a fake client (no Archicad, no Tapir). Fixture: tests/fixtures/lod2_site_min.json (contract in the
osmctx/features.py module doc)."""
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest
from shapely.geometry import Polygon

from osmctx import build_ac as bac
from osmctx import build_plan as bp
from osmctx import features as F

FIX = Path(__file__).resolve().parent / "fixtures" / "lod2_site_min.json"
T = bp.ROOF_THICKNESS


def site_min():
    return json.loads(FIX.read_text(encoding="utf-8"))


def by_id(recs):
    return {r["osm_id"]: r for r in recs}


def left(pivot, xy):
    """Signed distance of xy to the LEFT of the directed pivot line."""
    (x0, y0), (x1, y1) = pivot
    L = math.hypot(x1 - x0, y1 - y0)
    return ((x1 - x0) * (xy[1] - y0) - (y1 - y0) * (xy[0] - x0)) / L


def replay_top_z(item, xy):
    """Top-face z of a single-plane Roof item at plan point xy, rebuilt from the payload alone: underside rises
    tan(angle) per metre to the LEFT of the pivot line from `level`; the top face is thickness / cos(angle) above."""
    pv = item["pivotLine"]
    piv = [(pv["begCoordinate"]["x"], pv["begCoordinate"]["y"]), (pv["endCoordinate"]["x"], pv["endCoordinate"]["y"])]
    a = item["angle"]
    return item["level"] + math.tan(a) * left(piv, xy) + item["thickness"] / math.cos(a), piv


# ------------------------------------------------------------------ plane math
GABLE_S = [[0, 0, 6], [12, 0, 6], [12, 4, 9], [0, 4, 9]]          # rises to +y, tilt atan(3/4)
GABLE_N = [[12, 8, 6], [0, 8, 6], [0, 4, 9], [12, 4, 9]]          # rises to -y
HIP_W = [[0, 10, 5], [0, 0, 5], [5, 5, 8]]                         # hip end triangle, rises to +x
SKILLION = [[0, 0, 4], [8, 0, 4], [8, 6, 6], [0, 6, 6]]


@pytest.mark.parametrize("ring,tilt,uphill,z0,az", [
    (GABLE_S, math.atan2(3, 4), (0, 1), 6.0, 180.0),
    (GABLE_N, math.atan2(3, 4), (0, -1), 6.0, 0.0),
    (HIP_W, math.atan2(3, 5), (1, 0), 5.0, 270.0),
    (SKILLION, math.atan2(2, 6), (0, 1), 4.0, 180.0),
])
def test_plane_tilt_pivot_side_level(ring, tilt, uphill, z0, az):
    for r_ in (ring, ring[::-1]):                                  # winding does not matter
        r = F.roof_plane_from_polygon(r_, T)
        assert r["kind"] == "single"
        assert r["angle"] == pytest.approx(tilt, abs=1e-9) and r["tilt"] == pytest.approx(tilt, abs=1e-9)
        assert r["azimuth_deg"] == pytest.approx(az, abs=1e-6)     # compass direction the surface falls to
        assert r["level"] == pytest.approx(z0 - T / math.cos(tilt), abs=1e-9)   # underside at the pivot
        assert r["z_min"] == pytest.approx(z0) and r["z_max"] == pytest.approx(max(p[2] for p in ring))
        (x0, y0), (x1, y1) = r["pivot"]
        d = np.array([x1 - x0, y1 - y0]) / math.hypot(x1 - x0, y1 - y0)
        assert np.array([-d[1], d[0]]) @ np.array(uphill) == pytest.approx(1.0)  # uphill = left normal of pivot
        for p in ring:                                             # polygon on the left; pivot through the low edge
            assert left(r["pivot"], p) >= -1e-9
            assert left(r["pivot"], p) * math.tan(tilt) == pytest.approx(p[2] - z0, abs=1e-9)
        proj = [(np.array(p[:2]) - np.array([x0, y0])) @ d for p in ring]
        assert min(proj) >= F.LOD2_PIVOT_PAD - 1e-9 and max(proj) <= math.hypot(x1 - x0, y1 - y0) - F.LOD2_PIVOT_PAD + 1e-9


def test_skillion_convention_matches_m6():
    """Same pivot convention as the M6 skillion (roof_planes): plane rises to the LEFT of the pivot direction."""
    m6 = F.roof_planes(Polygon([(0, 0), (8, 0), (8, 6), (0, 6)]), {"shape": "skillion", "height": 2.0})[0]
    l2 = F.roof_plane_from_polygon(SKILLION, T)
    c = (4.0, 3.0)
    assert left(m6["pivot"], c) > 0 and left(l2["pivot"], c) > 0
    assert m6["angle"] == pytest.approx(l2["angle"])


def test_routing_flat_vertical_nonplanar_skip():
    flat = F.roof_plane_from_polygon([[0, 0, 3.5], [6, 0, 3.5], [6, 6, 3.5], [0, 6, 3.5]], T)
    assert flat["kind"] == "flat" and flat["angle"] == 0.0 and flat["level"] == pytest.approx(3.5 - T)
    almost = F.roof_plane_from_polygon([[0, 0, 3.5], [6, 0, 3.5], [6, 6, 3.5 + 6 * math.tan(math.radians(0.9))],
                                        [0, 6, 3.5 + 6 * math.tan(math.radians(0.9))]], T)
    assert almost["kind"] == "flat"
    low = F.roof_plane_from_polygon([[0, 0, 3.5], [6, 0, 3.5], [6, 6, 3.5 + 6 * math.tan(math.radians(1.2))],
                                     [0, 6, 3.5 + 6 * math.tan(math.radians(1.2))]], T)
    assert low["kind"] == "single"
    steep = F.roof_plane_from_polygon([[8, 0, 4], [8, 6, 4], [8.4, 6, 6.4], [8.4, 0, 6.4]], T)     # 80.5 deg
    assert steep["kind"] == "morph" and steep["reason"].startswith("near-vertical")
    ok74 = F.roof_plane_from_polygon([[0, 0, 0], [0, 6, 0], [1, 6, math.tan(math.radians(74))], [1, 0, math.tan(math.radians(74))]], T)
    assert ok74["kind"] == "single"
    wall = F.roof_plane_from_polygon([[0, 0, 0], [6, 0, 0], [6, 0, 3], [0, 0, 3]], T)              # exactly vertical
    assert wall["kind"] == "morph"
    twisted = F.roof_plane_from_polygon([[0, -14, 6], [8, -14, 6], [8, -10, 7.2], [0, -10, 6.6]], T)
    assert twisted["kind"] == "morph" and twisted["reason"].startswith("non-planar") and twisted["deviation"] > 0.05
    nearly = F.roof_plane_from_polygon([[0, 0, 6], [12, 0, 6], [12, 4, 9.03], [0, 4, 9]], T)        # 7.5 mm off
    assert nearly["kind"] == "single" and nearly["deviation"] < F.LOD2_PLANAR_TOL
    tiny = F.roof_plane_from_polygon([[0, 0, 4], [0.3, 0, 4], [0.3, 0.3, 4.1]], T)
    assert tiny["kind"] == "skip"
    assert F.roof_plane_from_polygon([[0, 0, 1], [1, 1, 1]], T)["kind"] == "skip"
    assert F.roof_plane_from_polygon([[0, 0, 1], [1, 0, 1], [2, 0, 1]], T)["kind"] == "skip"           # collinear


def test_plane_points_contract_tolerance():
    assert F.plane_points({"polygon": [[0, 0, 1], [1, 0, 1], [1, 1, 2], [0, 0, 1]]}) == [[0, 0, 1], [1, 0, 1], [1, 1, 2]]
    assert F.plane_points([[0, 0, 1], [1, 0, 1], [1, 1, 2]]) == [[0, 0, 1], [1, 0, 1], [1, 1, 2]]
    assert F.plane_points({"polygon": [[0, 0], [1, 0], [1, 1]], "z": [1, 1, 2]}) == [[0, 0, 1], [1, 0, 1], [1, 1, 2]]
    assert F.plane_points({"polygon": [[0, 0], [1, 0], [1, 1]]}) is None                      # 2D without z
    assert F.plane_points({"polygon": [["a", 0, 0]] * 3}) is None
    r = F.roof_planes(None, {"planes": [{"polygon": [[0, 0], [1, 0], [1, 1]]}, GABLE_S]})
    assert [p["kind"] for p in r] == ["skip", "single"]


def test_lod2_py_style_record_is_consumed():
    """osmctx/lod2.py plane records: 2D polygon + plane (normal, p0, z0_h), flat surfaces in flat_roofs, producer
    planarity flag; building-level summary planes ignored when the parts carry planes."""
    s_pl = {"polygon": [[0, 0], [12, 0], [12, 4], [0, 4]], "normal": [0.0, -0.6, 0.8], "p0": [0, 0], "z0_h": 6.0,
            "tilt": math.atan2(3, 4), "dev": 0.0, "planar": True, "fits": True}
    n_pl = {"polygon": [[12, 8], [0, 8], [0, 4], [12, 4]], "normal": [0.0, 0.6, 0.8], "p0": [12, 8], "z0_h": 6.0,
            "dev": 0.0, "planar": True}
    bent = dict(n_pl, planar=False, dev=0.12)
    flat = {"polygon": [[12, 0], [16, 0], [16, 8], [12, 8]], "normal": [0, 0, 1], "p0": [14, 4], "z0_h": 3.0,
            "zmax_h": 3.0, "dev": 0.0}
    assert F.plane_points(s_pl)[2] == pytest.approx([12, 4, 9.0])
    assert F.classify_plane(s_pl, T)["kind"] == "single" and F.classify_plane(bent, T)["kind"] == "morph"
    rec = {"osm_id": "lod2/X", "source": "lod2:lgl-bw", "footprint": [[0, 0], [16, 0], [16, 8], [0, 8]], "base_z": 0.0,
           "height": 9.0, "tags": {},
           "lod2": {"id": "X", "parts": [{"id": "X_1", "footprint": [[0, 0], [16, 0], [16, 8], [0, 8]], "holes": [],
                                          "min_h": 0.0, "eave_h": 3.0, "top_h": 9.0, "roof_planes": [s_pl, n_pl],
                                          "flat_roofs": [flat]}]},
           "roof": {"shape": "lod2", "height": 3.0, "planes": [s_pl, n_pl]}}          # duplicate summary: ignored
    s2, info = bp.lod2_expand({"buildings": [rec]})
    assert info["planes"] == {"single": 2, "flat": 1, "morph": 0, "skip": 0} and info["roofs"] == 2
    P = by_id(s2["building_parts"])
    assert bp.part_expected_bbox_z(P["lod2/X"]) == pytest.approx((0.0, 6.0))       # gable block to the eave
    assert bp.part_expected_bbox_z(P["lod2/X#b1"]) == pytest.approx((0.0, 3.0))    # flat annex block
    assert bp.building_source(P["lod2/X"], "OSM") == "LoD2 (lgl-bw)"                # no dataset label: region id
    items = bp.roof_items(P["lod2/X"])
    for it, pl in zip(items, [s_pl, n_pl]):
        for x, y, z in F.plane_points(pl):
            assert abs(replay_top_z(it["item"], (x, y))[0] - z) < 0.01
    rec["lod2"]["parts"][0]["roof_planes"] = [s_pl, bent]
    _s, info2 = bp.lod2_expand({"buildings": [rec]})
    assert info2["roofs"] == 1 and info2["morphs"] == 1


# ------------------------------------------------------------------ expansion of the fixture
def test_fixture_expansion_blocks_parts_roofs():
    site = site_min()
    raw = copy.deepcopy(site)
    s2, info = bp.lod2_expand(site)
    assert site == raw                                                   # input untouched
    assert info["buildings"] == 4 and info["mode"] == "roofs" and info["max_roofs"] == bp.LOD2_MAX_ROOFS
    assert info["planes"] == {"single": 9, "flat": 2, "morph": 2, "skip": 1}
    assert info["roofs"] == 9 and info["morphs"] == 2 and info["blocks"] == 7 and info["parts"] == 5
    assert info["morph_reasons"] == {"non-planar": 1, "near-vertical": 1} and not info["over_cap"]
    assert any("skipped" in w for w in info["warnings"])
    B = by_id(s2["buildings"])
    g = B["lod2/DEBW_GABLE"]
    assert not g.get("skip_block") and g["height"] == 9.0 and g["roof"]["shape"] == "lod2"
    assert bp.block_height(g) == pytest.approx(6.0)                       # block: ground -> eave
    assert bp.building_expected_bbox_z(g, 0.5) == pytest.approx((1.5, 7.5))
    assert bp.block_height(B["way/4242"]) == pytest.approx(5.0) and len(B["way/4242"]["roof"]["planes"]) == 4
    m, o = B["lod2/DEBW_MIXED"], B["lod2/DEBW_ODD"]
    assert m["skip_block"] and "roof" not in m and m["parts"] == ["DEBW_MIXED_1", "DEBW_MIXED_1#b1", "DEBW_MIXED_2"]
    assert o["skip_block"] and o["parts"] == ["lod2/DEBW_ODD", "lod2/DEBW_ODD#b1"]
    assert [b["osm_id"] for b in bp.active_buildings(s2)] == ["lod2/DEBW_GABLE", "way/4242"]
    P = by_id(s2["building_parts"])
    main, dormer, annex = P["DEBW_MIXED_1"], P["DEBW_MIXED_1#b1"], P["DEBW_MIXED_2"]
    assert all(p["parent"] == "lod2/DEBW_MIXED" and p["base_z"] == 2.0 for p in (main, dormer, annex))
    assert bp.part_expected_bbox_z(main) == pytest.approx((2.0, 9.0))     # gable part: eave 7 m
    assert len(main["roof"]["planes"]) == 2 and main["roof"]["morphs"] == []
    assert len(main["holes"]) == 1 and Polygon(main["holes"][0]).area == pytest.approx(6.0)   # dormer block cut out
    assert bp.part_expected_bbox_z(dormer) == pytest.approx((2.0, 11.0)) and "roof" not in dormer   # flat 9 m dormer
    assert Polygon(dormer["footprint"]).area == pytest.approx(6.0)
    assert bp.part_expected_bbox_z(annex) == pytest.approx((2.0, 5.5)) and "roof" not in annex       # flat annex 3.5
    o0, o1 = P["lod2/DEBW_ODD"], P["lod2/DEBW_ODD#b1"]
    assert bp.part_expected_bbox_z(o0) == pytest.approx((0.0, 4.0)) and len(o0["roof"]["planes"]) == 1
    assert len(o0["roof"]["morphs"]) == 2
    assert bp.part_expected_bbox_z(o1) == pytest.approx((0.0, 6.0))       # mixed eave without parts: own block
    assert Polygon(o0["footprint"]).area == pytest.approx(48.0) and Polygon(o1["footprint"]).area == pytest.approx(32.0)
    for p in s2["building_parts"]:
        assert bp.building_source(p, "OSM") == "LoD2 (LGL Baden-Wuerttemberg)"
    s3, info3 = bp.lod2_expand(s2)                                       # idempotent
    assert s3 is s2 and info3 is info
    json.dumps(s2)


def test_replay_every_vertex_on_roof_plane_within_1cm():
    """Rebuild each Roof's plane from the payload only and check every source vertex of its LoD2 surface."""
    site = site_min()
    s2, _ = bp.lod2_expand(site)
    dz = 0.75
    n_items = 0
    for o in bp.active_buildings(s2) + s2["building_parts"]:
        planes = (o.get("roof") or {}).get("planes") or []
        items = bp.roof_items(o, dz)
        assert len(items) == len(planes)
        for ri, pl in zip(items, planes):
            it = ri["item"]
            assert it["structureType"] == "Basic" and it["thickness"] == T and it["floorIndex"] == 0
            assert "levels" not in it and "eavesOverhang" not in it                     # single-plane Roof
            pts = F.plane_points(pl)
            ground = o["base_z"] + dz
            for x, y, z in pts:
                top, piv = replay_top_z(it, (x, y))
                assert abs(top - (ground + z)) < 0.01, (o["osm_id"], (x, y, z), top)
                assert left(piv, (x, y)) >= -1e-3                                        # polygon left of pivot
            assert ri["check_z"] and ri["expected"]["zMax"] == pytest.approx(ground + max(p[2] for p in pts))
            ring = [(q["x"], q["y"]) for q in it["polygonCoordinates"]]
            assert Polygon(ring).area == pytest.approx(Polygon([p[:2] for p in pts]).area, abs=1e-3)
            n_items += 1
    assert n_items == 9


def test_roof_items_unexpanded_owner_and_flat_zero_degree_roof():
    """A LoD2 owner reaching roof_items without lod2_expand: single + flat planes become Roofs (flat = 0 deg,
    underside = top - T), morph / skip planes do not."""
    o = {"osm_id": "x", "footprint": [[0, 0], [8, 0], [8, 6], [0, 6]], "base_z": 1.0, "height": 6.0,
         "roof": {"shape": "lod2", "height": 2.0, "planes": [
             {"polygon": SKILLION}, {"polygon": [[0, 0, 3.5], [6, 0, 3.5], [6, 6, 3.5]]},
             {"polygon": [[8, 0, 4], [8, 6, 4], [8.4, 6, 6.4], [8.4, 0, 6.4]]}, {"polygon": [[0, 0], [1, 0], [1, 1]]}]}}
    items = bp.roof_items(o, dz=0.5)
    assert len(items) == 2
    flat = items[1]["item"]
    assert flat["angle"] == 0.0 and flat["level"] == pytest.approx(1.5 + 3.5 - T)
    assert items[1]["expected"]["zMax"] == pytest.approx(1.5 + 3.5)
    assert len(bp.lod2_morph_items(o, 0.5)) == 1


def test_morph_items_body_format():
    s2, _ = bp.lod2_expand(site_min())
    o0 = by_id(s2["building_parts"])["lod2/DEBW_ODD"]
    ms = bp.lod2_morph_items(o0, dz=1.0)
    assert len(ms) == 2
    twisted, steep = ms
    for m in ms:
        it, ex = m["item"], m["expected"]
        body = it["body"]
        assert body["bodyType"] == "Surface" and body["wireEdges"] == [] and body["edgeDefault"] == bp.LOD2_MORPH_EDGES
        bpnt = it["basePoint"]
        assert bpnt["z"] == pytest.approx(1.0)                           # base_z 0 + dz
        xs = [bpnt["x"] + v["x"] for v in body["vertices"]]
        zs = [bpnt["z"] + v["z"] for v in body["vertices"]]
        assert min(v["x"] for v in body["vertices"]) == 0.0 and min(v["y"] for v in body["vertices"]) == 0.0
        assert (min(xs), max(xs), min(zs), max(zs)) == pytest.approx((ex["xMin"], ex["xMax"], ex["zMin"], ex["zMax"]))
        ids = {i for f in body["polygons"] for i in f["vertexIds"]}
        assert ids == set(range(len(body["vertices"])))
    assert len(twisted["item"]["body"]["polygons"]) == 2 and all(len(f["vertexIds"]) == 3 for f in twisted["item"]["body"]["polygons"])
    assert len(steep["item"]["body"]["polygons"]) == 1 and len(steep["item"]["body"]["polygons"][0]["vertexIds"]) == 4
    assert twisted["expected"]["zMax"] == pytest.approx(1.0 + 7.2) and steep["expected"]["zMin"] == pytest.approx(1.0 + 4.0)
    # triangles of a non-planar ring reproduce its corners (no new vertices) and cover its plan area
    P = np.array([[0, -14, 6], [8, -14, 6], [8, -10, 7.2], [0, -10, 6.6]], float)
    tris = F.triangulate_ring_3d(P)
    assert sum(Polygon(P[list(t), :2]).area for t in tris) == pytest.approx(32.0)


# ------------------------------------------------------------------ count guard / degrade
def test_count_guard_degrades_most_expensive_first():
    site = site_min()
    c = bp.lod2_counts(site)
    assert c == {"buildings": 4, "blocks": 7, "roofs": 9, "morphs": 2, "degraded": 0, "roofs_uncapped": 11,
                 "over_cap": False, "max_roofs": 400, "mode": "roofs", "elements": 18}
    s2, info = bp.lod2_expand(site, max_roofs=8)
    assert info["over_cap"] and info["degraded"] == 1 and info["roofs"] + info["morphs"] == 7 <= 8
    hip = by_id(s2["buildings"])["way/4242"]
    assert hip["lod2_degraded"] == "cap" and "roof" not in hip
    # volume-equivalent block: hip roof 14 x 10, rise 3 -> V = h W (3L - W) / 6 = 160 m3 over 140 m2
    assert hip["height"] == pytest.approx(5.0 + 160.0 / 140.0, abs=1e-3)
    assert bp.roof_items(hip) == [] and bp.lod2_morph_items(hip) == []
    assert any("lod2_max_roofs=8" in w for w in info["warnings"])
    _s, i0 = bp.lod2_expand(site, max_roofs=0)
    assert i0["roofs"] == i0["morphs"] == 0 and i0["degraded"] == 4 and i0["blocks"] == 4
    _s, iu = bp.lod2_expand(site, max_roofs=-1)
    assert iu["max_roofs"] is None and iu["roofs"] == 9 and not iu["over_cap"]
    sb, ib = bp.lod2_expand(site, mode="blocks")                          # super-light: blocks only
    assert ib["mode"] == "blocks" and ib["roofs"] == ib["morphs"] == 0 and ib["blocks"] == 4 and ib["degraded"] == 0
    assert all(not b.get("skip_block") and "roof" not in b for b in sb["buildings"])
    assert by_id(sb["buildings"])["lod2/DEBW_GABLE"]["height"] == pytest.approx(7.5)   # gable 6/9: mean 7.5
    assert sb["building_parts"] == [] and bp.feature_expected(sb)["roofs"] == 0
    with pytest.raises(ValueError):
        bp.lod2_expand(site, mode="fancy")


def test_settings_from_site_report_and_counts_from_list():
    site = site_min()
    site["report"] = {"defaults_used": {"settings": {"include": {"lod2_roofs": False}}}}
    assert bp.lod2_settings(site) == (bp.LOD2_MAX_ROOFS, "blocks")
    site["report"]["defaults_used"]["settings"] = {"lod2_max_roofs": 5, "include": {}}
    assert bp.lod2_settings(site) == (5, "roofs")
    assert bp.lod2_settings(site, max_roofs=50, mode="blocks") == (50, "blocks")
    assert bp.lod2_expand(site)[1]["over_cap"] is True
    c = bp.lod2_counts(site_min()["buildings"], max_roofs=8)
    assert c["degraded"] == 1 and c["elements"] == 7 + 5 + 2
    assert bp.lod2_counts([]) ["buildings"] == 0


def test_feature_expected_with_lod2():
    fe = bp.feature_expected(site_min())
    assert fe["roofs"] == 9 and fe["building_parts"] == 5 and fe["building_morphs"] == 2


# ------------------------------------------------------------------ existing payloads unchanged (golden)
def _m6_dump():
    """M6 (OSM roof:shape / building:part) payloads on synthetic owners; digest pinned from the pre-M10 code."""
    fps = [[[0, 0], [20, 0], [20, 10], [0, 10], [0, 0]],
           [[3, 1], [17.5, 4.2], [15.1, 14.9], [0.6, 11.7]],
           [[0, 0], [12, 0], [12, 5], [6, 5], [6, 12], [0, 12]],
           [[100.25, -40.5], [109.75, -40.5], [109.75, -31.0], [100.25, -31.0]]]
    roofs = [{"shape": "gabled", "height": 3.0}, {"shape": "gabled", "height": 2.2, "orientation": "across"},
             {"shape": "hipped", "height": 2.5}, {"shape": "pyramidal", "height": 4.0},
             {"shape": "skillion", "height": 1.8}, {"shape": "skillion", "height": 2.0, "direction": 135.0}]
    out = []
    for i, fp in enumerate(fps):
        for r in roofs:
            for dz in (0.0, 1.25):
                o = {"osm_id": f"way/{i}", "footprint": fp, "base_z": 3.5 + i, "height": 11.0, "min_height": 1.0,
                     "roof": dict(r)}
                out.append({"roof_items": bp.roof_items(o, dz), "planes": F.roof_planes(Polygon(fp), dict(r)),
                            "building_slab": bp.building_slab(o, dz), "part_slab": bp.part_slab(o, dz),
                            "bbox": [bp.building_expected_bbox_z(o, dz), bp.part_expected_bbox_z(o, dz)]})
    site = {"buildings": [{"osm_id": "way/1", "footprint": fps[0], "base_z": 1.0, "height": 9.0, "roof": roofs[0]},
                          {"osm_id": "way/2", "footprint": fps[1], "base_z": 1.0, "height": 9.0, "skip_block": True}],
            "building_parts": [{"osm_id": "way/3", "parent": "way/2", "footprint": fps[2], "base_z": 1.0, "height": 8.0,
                                "min_height": 0.0, "roof": roofs[2]}]}
    out.append({"feature_expected": bp.feature_expected(site), "expected_counts": bp.expected_counts(site),
                "source": [bp.building_source(b, "OSM") for b in site["buildings"]]})
    created = {"buildings": [("B1", "way/1")], "building_parts": [("P1", "way/3")], "building_roofs": [("R1", "way/3")],
               "building_parent": {"way/3": "way/2"}, "terrain": [("T", "t")]}
    out.append({"group_plan": bp.group_plan(created)})
    return json.dumps(out, sort_keys=True, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))


M6_GOLDEN = "7998ede8b27cbe61671a523a1f1a57652b362d43f315d7591b57265924f6fd94"   # git HEAD eecdd7d, 2026-11-24


def test_m6_roof_and_part_payloads_byte_identical():
    assert hashlib.sha256(_m6_dump().encode()).hexdigest() == M6_GOLDEN


def test_non_lod2_site_is_returned_unchanged():
    site = {"buildings": [{"osm_id": "way/1", "footprint": [[0, 0], [5, 0], [5, 5]], "base_z": 0, "height": 6,
                           "source": "osm"}], "building_parts": []}
    s2, info = bp.lod2_expand(site)
    assert s2 is site and info["buildings"] == 0 and "lod2_plan" not in site
    assert "building_morphs" not in bp.feature_expected(site)


def test_sources_and_group_plan_morphs():
    b = {"source": "lod2:3dbag"}
    assert bp.building_source(b, "OSM / x") == "LoD2 (3dbag)"
    assert bp.building_source(dict(b, lod2={"dataset": "3DBAG"}), "OSM / x") == "LoD2 (3DBAG)"
    assert bp.building_source({"source": "overture:Google Open Buildings"}, "d") == "Overture Maps (Google Open Buildings)"
    assert bp.building_source({}, "OSM / x") == "OSM / x"
    plan = bp.group_plan({"terrain": [("T", "t")], "buildings": [("B1", "a"), ("B2", "b")],
                          "building_parts": [("P1", "b#1"), ("P2", "b#2")], "building_roofs": [("R1", "b#1")],
                          "building_morphs": [("M1", "b#1"), ("M2", "a")], "building_parent": {"b#1": "b", "b#2": "b"}})
    bld = next(k for k in plan["children"] if isinstance(k, dict) and k["name"] == "Buildings")
    sub = {k["name"]: k["children"] for k in bld["children"] if isinstance(k, dict)}
    assert set(sub["Building b"]) == {"B2", "P1", "P2", "R1", "M1"} and set(sub["Building a"]) == {"B1", "M2"}


def test_osm_roof_and_part_tags_do_not_override_lod2(monkeypatch):
    cv = SimpleNamespace(counts=Counter(), s={"floor_height": 3.2, "building_part_min_remainder": 4.0})
    lod = {"osm_id": "way/9", "source": "lod2:lgl-bw", "footprint": [[0, 0], [10, 0], [10, 10], [0, 10]], "height": 9.0,
           "tags": {"roof:shape": "gabled", "roof:height": "3"}, "lod2": {"dataset": "x"},
           "roof": {"shape": "lod2", "height": 3.0, "planes": [GABLE_S]}}
    F.building_roofs(cv, [lod])
    assert lod["roof"]["shape"] == "lod2" and not cv.counts
    part_poly = Polygon([(0, 0), (5, 0), (5, 10), (0, 10)])
    monkeypatch.setattr(F, "area_geom", lambda cv_, el: part_poly)
    monkeypatch.setattr(F, "clipped_polys", lambda cv_, g, min_area=1.0: [part_poly])
    cv.els = [{"type": "way", "id": 5, "tags": {"building:part": "yes", "height": "4"}}]
    cv.bpoly = Polygon([(-50, -50), (50, -50), (50, 50), (-50, 50)])
    parts, keep = F.building_parts(cv, [lod])
    assert parts == [] and keep == [lod] and cv.counts["building_parts_skipped_lod2"] == 1


# ------------------------------------------------------------------ builder wiring (fake client, no Archicad)
class FakeAC:
    """Answers every call the builder makes with Tapir/native-shaped data; records creates, classes, properties."""
    LAYERS = ["Site & Landscape - Terrain", "Site-Context Buildings", "Site-Roads", "Site-Trees", "Site-Water",
              "Site & Landscape - General"]
    CLASSES = ["Terrain", "Massing", "Civil Element", "Flora & Fauna", "Geographic Element", "Site", "Roof"]

    def __init__(self):
        self.n = 0
        self.calls = []
        self.elements = {}          # guid -> {"type", "layer", "item"}
        self.classes = {}           # guid -> classification item id
        self.props = {}             # guid -> {property guid: value}
        self.prop_ids = []

    def g(self, p="g"):
        self.n += 1
        return f"{p}-{self.n:05d}-0000-0000-0000-000000000000"

    def call(self, cmd, p=None):
        p = p or {}
        self.calls.append(cmd)
        if cmd == "API.GetAttributesByType":
            if p["attributeType"] == "Layer":
                return {"attributeIds": [{"attributeId": {"guid": f"layer-{n}"}} for n in self.LAYERS]}
            return {"attributeIds": [{"attributeId": {"guid": "EMPTY"}}]}
        if cmd == "API.GetLayerAttributes":
            return {"attributes": [{"layerAttribute": {"attributeId": x["attributeId"],
                                                       "name": x["attributeId"]["guid"][6:], "isHidden": False,
                                                       "isLocked": False}} for x in p["attributeIds"]]}
        if cmd == "API.GetActivePenTables":
            return {"modelViewPenTableId": {"attributeId": {"guid": "pt"}}}
        if cmd == "API.GetPenTableAttributes":
            pens = [(1, 0.13, (0, 0, 0)), (12, 0.18, (.5, .25, .25)), (19, 0, (1, 1, 1)), (32, 0.35, (.5, .25, .25)),
                    (61, 0.18, (.42, .42, .42)), (99, 0.13, (.67, .82, 1)), (102, 0.13, (.8, .8, .8)),
                    (104, 0.13, (.59, .88, .59))]
            return {"attributes": [{"penTableAttribute": {"name": "fake", "pens": [
                {"pen": {"index": i, "weight": w, "color": dict(zip(("red", "green", "blue"), c)), "description": ""}}
                for i, w, c in pens]}}]}
        if cmd == "API.GetAllPropertyIds":
            return {"propertyIds": [{"propertyId": {"guid": "struct-fn"}}]}
        if cmd == "API.GetDetailsOfProperties":
            return {"propertyDefinitions": [{"propertyDefinition": {"propertyId": x["propertyId"],
                                                                    "name": "Structural Function",
                                                                    "group": {"name": "Structural"}}}
                                            for x in p["properties"]]}
        if cmd == "API.GetBuildingMaterialAttributes":
            return {"attributes": [{"buildingMaterialAttribute": {"name": nm, "attributeId": {"guid": f"bm-{nm}"}}}
                                   for nm in ("Sand", "Soil", "Concrete", "GENERIC - ENVIRONMENT")]}
        if cmd == "API.GetSurfaceAttributes":
            return {"attributes": [{"surfaceAttribute": {"name": nm, "attributeId": {"guid": f"sf-{nm}"}}}
                                   for nm in ("Earth - Brown", "Concrete - 02")]}
        if cmd == "API.GetFillAttributes":
            return {"attributes": [{"fillAttribute": {"attributeId": {"guid": "EMPTY"}, "subType": "Empty"}}]}
        if cmd == "API.GetAllClassificationsInSystem":
            return {"classificationItems": [{"classificationItem": {"id": i, "classificationItemId": {"guid": f"cls-{i}"},
                                                                    "children": []}} for i in self.CLASSES]}
        if cmd == "API.GetPropertyValuesOfElements":
            return {"propertyValuesForElements": [{"propertyValues": [{"propertyValue": {"value": "x"}}]}
                                                  for _ in p["elements"]]}
        if cmd == "API.SetPropertyValuesOfElements":
            for v in p["elementPropertyValues"]:
                self.props.setdefault(v["elementId"]["guid"], {})[v["propertyId"]["guid"]] = v["propertyValue"]["value"]
            return {"executionResults": [{"success": True} for _ in p["elementPropertyValues"]]}
        raise AssertionError(f"unexpected native call {cmd}")

    def _create(self, p, key, ty):
        out = []
        for it in p[key]:
            x = self.g(ty)
            self.elements[x] = {"type": ty, "layer": it.get("layerIndex", 1), "item": it}
            out.append({"elementId": {"guid": x}})
        return {"elements": out}

    def tapir(self, c, p=None):
        p = p or {}
        self.calls.append(c)
        ok = {"success": True}
        creates = {"CreateMeshes": ("meshesData", "Mesh"), "CreateSlabs": ("slabsData", "Slab"),
                   "CreateBeams": ("beamsData", "Beam"), "CreateObjects": ("objectsData", "Object"),
                   "CreateRoofs": ("roofsData", "Roof"), "CreateMorphs": ("morphsData", "Morph"),
                   "CreatePolylines": ("polylinesData", "PolyLine"), "CreateTexts": ("textsData", "Text")}
        if c in creates:
            return self._create(p, *creates[c])
        if c == "GetProjectInfo":
            return {"isUntitled": True, "projectPath": None, "projectName": "fake"}
        if c == "GetStories":
            return {"stories": [{"index": 0, "level": 0.0, "name": "Ground"}]}
        if c == "GetLayers":
            return {"layers": [{"attributeId": x["attributeId"], "name": x["attributeId"]["guid"][6:],
                                "index": self.LAYERS.index(x["attributeId"]["guid"][6:]) + 1,
                                "isHidden": False, "isLocked": False} for x in p["attributeIds"]]}
        if c == "GetAllClassificationSystems":
            return {"classificationSystems": [{"classificationSystemId": {"guid": "sys"}, "name": "Archicad Classification"}]}
        if c == "FindLibraryParts":
            return {"libraryParts": [{"name": p["nameContains"] + ".gsm"}]}
        if c == "CreatePropertyGroups":
            return {"propertyGroupIds": [{"propertyGroupId": {"guid": "pg"}} for _ in p["propertyGroups"]]}
        if c == "CreatePropertyDefinitions":
            self.prop_ids = [self.g("prop") for _ in p["propertyDefinitions"]]
            return {"propertyIds": [{"propertyId": {"guid": x}} for x in self.prop_ids]}
        if c == "UpdatePropertyDefinitions":
            return {"executionResults": [ok for _ in p["propertyDefinitions"]]}
        if c == "SetGeoLocation":
            return ok
        if c == "GetGeoLocation":
            return {"surveyPoint": {"position": {}, "geoReferencingParameters": {"crsName": ""}}}
        if c == "GetSuspendGroupsMode":
            return {"suspendGroups": False}
        if c == "SetSuspendGroupsMode":
            return {"executionResult": ok}
        if c == "GetElementsByType":
            return {"elements": [{"elementId": {"guid": x}} for x, e in self.elements.items() if e["type"] == p["elementType"]]}
        if c == "DeleteElements":
            return {"executionResults": [ok for _ in p["elements"]]}
        if c == "SetDetailsOfElements":
            for it in p["elementsWithDetails"]:
                if "layerIndex" in it["details"]:
                    self.elements[it["elementId"]["guid"]]["layer"] = it["details"]["layerIndex"]
            return {"executionResults": [ok for _ in p["elementsWithDetails"]]}
        if c == "SetClassificationsOfElements":
            for it in p["elementClassifications"]:
                self.classes[it["elementId"]["guid"]] = it["classificationId"]["classificationItemId"]["guid"]
            return {"executionResults": [ok for _ in p["elementClassifications"]]}
        if c in ("ModifyBeams", "ModifySlabs", "ModifyMeshes"):
            key = next(k for k in p if k.endswith("WithDetails"))
            return {"executionResults": [ok for _ in p[key]]}
        if c == "Get3DBoundingBoxes":
            return {"boundingBoxes3D": [{"boundingBox3D": dict.fromkeys(("xMin", "yMin", "zMin", "xMax", "yMax", "zMax"), 0.0)}
                                        for _ in p["elements"]]}
        if c == "GetDetailsOfElements":
            return {"detailsOfElements": [{"elementId": e["elementId"], "type": self.elements[e["elementId"]["guid"]]["type"],
                                           "floorIndex": 0, "layerIndex": self.elements[e["elementId"]["guid"]]["layer"]}
                                          for e in p["elements"]]}
        if c == "GetGraphicOverrideRules":
            return {"graphicOverrideRules": [{"graphicOverrideRuleId": {"guid": "r1"}, "name": bac.TERRAIN_GO_RULE,
                                              "style": {}}]}
        if c in ("CreateGraphicOverrideRules", "ModifyGraphicOverrideRules", "CreateGraphicOverrideCombinations",
                 "ModifyGraphicOverrideCombinations"):
            key = next(iter(p))
            return {"executionResults": [ok for _ in p[key]]}
        if c == "GetGraphicOverrideCombinations":
            return {"graphicOverrideCombinations": [{"graphicOverrideCombinationId": {"guid": "c1"}, "name": "X",
                                                     "graphicOverrideRuleIds": []}]}
        if c == "GetActiveGraphicalOverrideCombination":
            return {"combinationId": {"guid": "c1"}, "name": "X"}
        if c == "SetActiveGraphicalOverrideCombination":
            return ok
        if c == "CreateSolidElementLinks":
            return {"executionResults": [ok for _ in p["solidLinks"]]}
        if c == "CreateGroups":
            return {"groupGuids": [{"groupId": {"guid": self.g("grp")}} for _ in p["elementGroups"]]}
        if c == "SetUserDataOfGroups":
            return {"executionResults": [ok for _ in p["groups"]]}
        if c in ("GetElementUserData",):
            return {"elementUserData": [{"userData": []} for _ in p["elements"]]}
        if c == "SetElementUserData":
            return {"executionResults": [ok for _ in p["elementUserData"]]}
        if c == "ChangeWindow":
            return ok
        raise AssertionError(f"fake has no answer for {c}")


@pytest.fixture(scope="module")
def fake_build(tmp_path_factory):
    from osmctx.convert import convert
    from test_convert import make_cache, way
    tmp = tmp_path_factory.mktemp("lod2_build")
    d = make_cache(tmp / "cache", [way(1, [(20, 20), (30, 20), (30, 28), (20, 28), (20, 20)],
                                       {"building": "house", "height": "7"})], lambda lon, lat: 100.0, size=100.0)
    site = convert(d, None, tmp / "out")
    site["buildings"] += site_min()["buildings"]
    setup = tmp / "setup.json"
    setup.write_text(json.dumps({"recommended_attributes": {
        "terrain": {"surface": "Earth - Brown", "building_material": "Soil"},
        "building": {"surface": "Concrete - 02", "building_material": "GENERIC - ENVIRONMENT"},
        "road": {"building_material": "Concrete"}, "water": {"building_material": "Sand"}}}), encoding="utf-8")
    fake = FakeAC()
    orig = bac.Client
    bac.Client = lambda *a, **k: fake
    try:
        rep = bac.build(str(tmp / "site.json"), str(setup), log_path=str(tmp / "log.txt"),
                        report_path=str(tmp / "rep.json"), save=False, group=True, label=False, tree_mode="light",
                        contours=False, site_data=site)
    finally:
        bac.Client = orig
    return fake, rep, site


def test_build_ac_kinds_counts_and_payloads(fake_build):
    fake, rep, site = fake_build
    assert rep["lod2"]["buildings"] == 4 and rep["lod2"]["roofs"] == 9 and rep["lod2"]["morphs"] == 2
    cr = rep["created"]
    assert cr["buildings"] == 3                    # OSM house + LoD2 gable + LoD2 hip; mixed / odd -> parts
    assert cr["building_parts"] == 5 and cr["building_roofs"] == 9 and cr["building_morphs"] == 2
    assert rep["expected"]["building_morphs"] == 2 and not rep["errors"].get("building_morphs")
    assert rep["roofs"]["by_shape"] == {"lod2": 9}
    roofs = [e for e in fake.elements.values() if e["type"] == "Roof"]
    morphs = [e for e in fake.elements.values() if e["type"] == "Morph"]
    assert len(roofs) == 9 and len(morphs) == 2
    lidx = rep["layers"]["buildings"]
    assert all(e["layer"] == lidx for e in roofs + morphs)
    assert all(e["item"]["layerIndex"] == lidx and e["item"]["body"]["bodyType"] == "Surface" for e in morphs)
    assert all("pivotLine" in e["item"] and "levels" not in e["item"] for e in roofs)
    assert rep["checks"]["per_type_layer_mismatch"] == {} and rep["checks"]["counts_match"]
    assert any(g.startswith("LoD2 buildings:") for g in rep["gaps"])


def test_build_ac_classification_source_osm_id(fake_build):
    fake, rep, _site = fake_build
    pid, sid = fake.prop_ids
    by_type = {}
    for gd, e in fake.elements.items():
        by_type.setdefault(e["type"], []).append(gd)
    for gd in by_type["Roof"] + by_type["Morph"]:
        assert fake.classes[gd] == "cls-Roof"
        assert fake.props[gd][sid] == "LoD2 (LGL Baden-Wuerttemberg)"
    slabs = {fake.props[gd][pid]: gd for gd in by_type["Slab"] if gd in fake.props}
    for osm in ("lod2/DEBW_GABLE", "way/4242", "DEBW_MIXED_1", "DEBW_MIXED_1#b1", "DEBW_MIXED_2", "lod2/DEBW_ODD#b1"):
        assert fake.classes[slabs[osm]] == "cls-Massing"
        assert fake.props[slabs[osm]][sid] == "LoD2 (LGL Baden-Wuerttemberg)"
    osm_house = slabs["way/1"]
    assert fake.props[osm_house][sid].startswith("OSM /")          # OSM building keeps the build source string
    roof_ids = Counter(fake.props[gd][pid] for gd in by_type["Roof"])
    assert roof_ids == {"lod2/DEBW_GABLE": 2, "way/4242": 4, "DEBW_MIXED_1": 2, "lod2/DEBW_ODD": 1}
    assert Counter(fake.props[gd][pid] for gd in by_type["Morph"]) == {"lod2/DEBW_ODD": 2}


def test_build_ac_groups_per_building(fake_build):
    _fake, rep, _site = fake_build
    groups = rep["groups"]["direct_children"]
    assert groups["Building lod2/DEBW_GABLE"] == 3            # block + 2 roofs
    assert groups["Building way/4242"] == 5                   # block + 4 roofs
    assert groups["Building lod2/DEBW_MIXED"] == 5            # 3 part blocks + 2 roofs
    assert groups["Building lod2/DEBW_ODD"] == 5              # 2 blocks + 1 roof + 2 morphs


def test_build_ac_registry_and_cli():
    assert bac.KIND_TYPE_ROLE["building_morphs"] == ("Morph", "buildings")
    assert bac.CLASS_ITEMS["building_morphs"] == "Roof" and "Morph" in bac.OSM_ELEMENT_TYPES
    src = Path(bac.__file__).read_text(encoding="utf-8")
    assert "--lod2-mode" in src and "--lod2-max-roofs" in src
    import inspect
    sig = inspect.signature(bac.build).parameters
    assert "lod2_mode" in sig and "lod2_max_roofs" in sig
    with pytest.raises(bac.ArchicadError):
        bac._build_impl(FakeAC(), lambda *_: None, 0, "x", "x", "x", True, False, 0.0, "none", False, "light",
                        False, "clean", True, "single", None, "points", False, None, 1.0, None, "railing", {},
                        lod2_mode="fancy")
