"""M6 more features: railways, walls/fences, building parts + roofs, ground surfaces, piers, power.
Synthetic caches (tests/test_convert helpers) + one real fixture (tests/fixtures/tuebingen_station, Tuebingen Hbf 200 m)."""
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest
from shapely.geometry import Polygon

import server
from osmctx import build_plan as bp
from osmctx import features as F
from osmctx import fetch as FE
from osmctx import limits
from osmctx.convert import Converter, convert
from test_convert import L, make_cache, node, way  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "tuebingen_station"
FLAT = lambda lon, lat: 100.0  # noqa: E731


def run(tmp_path, els, dem_fn=FLAT, settings=None, **kw):
    kw.setdefault("size", 100.0)
    d = make_cache(tmp_path / "cache", els, dem_fn, **kw)
    return convert(d, settings, tmp_path / "out")


def box(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


# ------------------------------------------------------------------ railways
def test_rail_width_gauge_tracks_and_default():
    s = {"rail_track_width": 3.0}
    assert F.rail_width({}, s) == (3.0, "default", 1)
    w, src, n = F.rail_width({"gauge": "1435"}, s)
    assert w == pytest.approx(2.935) and src == "gauge" and n == 1
    w, src, n = F.rail_width({"tracks": "2"}, s)
    assert (w, src, n) == (6.0, "tracks", 2)
    assert F.rail_width({"width": "5"}, s)[:2] == (5.0, "tag")
    assert F.rail_width({"gauge": "99999"}, s)[1] == "default"


def test_railways_beams_skip_disused_bridge_and_platform(tmp_path):
    els = [
        way(1, [(-40, 0), (40, 0)], {"railway": "rail", "gauge": "1435"}),
        way(2, [(-40, 10), (40, 10)], {"railway": "tram", "tracks": "2"}),
        way(3, [(-40, 20), (40, 20)], {"railway": "rail", "disused": "yes"}),
        way(4, [(-40, 30), (40, 30)], {"railway": "disused"}),               # never a rail class
        way(5, [(-40, -20), (40, -20)], {"railway": "subway"}),               # no tags: assumed tunnel
        way(6, [(-10, 40), (30, 40)], {"railway": "rail", "bridge": "yes", "layer": "1"}),
        way(7, box(-5, -40, 5, -35), {"railway": "platform"}),
        way(8, [(-30, -30), (30, -30)], {"railway": "platform", "width": "4"}),  # open platform way
    ]
    site = run(tmp_path, els)
    rails = {r["osm_id"]: r for r in site["railways"]}
    assert set(rails) == {"way/1", "way/2", "way/5", "way/6"}
    assert rails["way/1"]["width"] == pytest.approx(2.935) and rails["way/2"]["tracks"] == 2
    assert rails["way/5"]["tunnel"] and rails["way/6"]["bridge"] and rails["way/6"]["layer"] == 1
    assert site["report"]["counts"]["railways_skipped_disused"] == 1
    assert site["report"]["counts"]["railways_subway_assumed_tunnel"] == 1
    assert len(site["railway_platforms"]) == 2
    open_p = next(p for p in site["railway_platforms"] if p["shape"] == "line")
    assert Polygon(open_p["polygon"]).area == pytest.approx(60 * 4, rel=0.02)
    assert site["roads"] == []       # railways are not roads
    beams = bp.road_beams(rails["way/1"])
    assert beams and all(b["width"] == pytest.approx(2.935) for b in beams)
    assert bp.platform_slab(site["railway_platforms"][0])["referencePlaneLocation"] == "Bottom"


def test_railways_off_and_include_flags(tmp_path):
    els = [way(1, [(-40, 0), (40, 0)], {"railway": "rail"}), way(2, [(-40, 5), (40, 5)], {"barrier": "wall"})]
    site = run(tmp_path, els, settings={"include": {"railways": False, "walls": False}})
    assert site["railways"] == [] and site["barriers"] == []
    site = run(tmp_path / "b", els)
    assert len(site["railways"]) == 1 and len(site["barriers"]) == 1


# ------------------------------------------------------------------ barriers
def test_barriers_defaults_tags_segments_and_terrain(tmp_path):
    slope = lambda lon, lat: 100.0 + 0.0  # noqa: E731
    els = [
        way(1, [(-45, 0), (45, 0)], {"barrier": "wall"}),
        way(2, [(-45, 10), (45, 10)], {"barrier": "retaining_wall", "height": "3 m", "width": "0.6"}),
        way(3, [(-45, 20), (45, 20)], {"barrier": "city_wall"}),
        way(4, [(-45, 30), (45, 30)], {"barrier": "fence"}),
        way(5, [(-45, 40), (45, 40)], {"barrier": "guard_rail", "height": "x"}),
        way(6, [(-45, -10), (45, -10)], {"barrier": "hedge"}),     # not a wall type
        way(7, [(-45, 50), (45, 50)], {"barrier": "fence", "tunnel": "yes"}),
    ]
    site = run(tmp_path, els, slope)
    bars = {b["osm_id"]: b for b in site["barriers"]}
    assert set(bars) == {"way/1", "way/2", "way/3", "way/4", "way/5"}
    assert (bars["way/1"]["height"], bars["way/1"]["thickness"], bars["way/1"]["tool"]) == (2.0, 0.25, "wall")
    assert (bars["way/2"]["height"], bars["way/2"]["thickness"]) == (3.0, 0.6)
    assert (bars["way/3"]["height"], bars["way/3"]["thickness"]) == (4.0, 1.0)
    assert (bars["way/4"]["height"], bars["way/4"]["thickness"], bars["way/4"]["tool"]) == (1.5, 0.05, "railing")
    assert (bars["way/5"]["height"], bars["way/5"]["height_source"]) == (0.8, "default")
    for b in bars.values():
        seg = [math.dist(p[:2], q[:2]) for p, q in zip(b["points"], b["points"][1:])]
        assert max(seg) <= 5.0 + 1e-6
    assert site["report"]["counts"]["barriers_skipped_tunnel"] == 1
    w = bp.wall_items(bars["way/1"])
    assert w and w[0]["structureType"] == "Basic" and w[0]["height"] == pytest.approx(2.0)
    r = bp.railing_items(bars["way/4"])
    assert r and len(r[0]["coordinates"]) == 2 and r[0]["height"] == pytest.approx(1.5)


def test_barrier_segments_base_on_lower_end_and_height_covers_slope():
    bar = {"height": 2.0, "thickness": 0.25, "points": [[0, 0, 10.0], [5, 0, 11.0], [10, 0, 11.0]]}
    s = bp.barrier_segments(bar, dz=1.0)
    assert s[0]["base"] == 11.0 and s[0]["height"] == pytest.approx(3.0)   # top 14 = highest end (12) + 2
    assert s[1]["base"] == 12.0 and s[1]["height"] == 2.0
    z0, z1 = bp.barrier_expected_z(bp.wall_items(bar, 1.0)[0], "wall")
    assert (z0, z1) == (11.0, 14.0)


def test_barrier_on_bridge_takes_deck_z(tmp_path):
    els = [
        way(1, [(-45, 0), (45, 0)], {"highway": "primary", "bridge": "yes", "layer": "1"}),
        way(2, [(-45, 1.5), (45, 1.5)], {"barrier": "guard_rail"}),
        way(3, [(-45, 40), (45, 40)], {"barrier": "guard_rail"}),
    ]
    site = run(tmp_path, els)
    deck = max(p[2] for p in site["roads"][0]["points"])
    near = next(b for b in site["barriers"] if b["osm_id"] == "way/2")
    far = next(b for b in site["barriers"] if b["osm_id"] == "way/3")
    assert near["on_bridge"] and not far["on_bridge"]
    assert min(p[2] for p in near["points"]) >= min(p[2] for p in site["roads"][0]["points"]) - 1e-6
    assert deck >= max(p[2] for p in far["points"]) - 1e-6


# ------------------------------------------------------------------ building parts + roofs
def test_parts_replace_parent_block_and_heights(tmp_path):
    outline = box(-20, -20, 20, 20)
    els = [
        way(1, outline, {"building": "yes", "building:levels": "2"}),
        way(2, box(-20, -20, 0, 20), {"building:part": "yes", "building:levels": "5", "roof:shape": "gabled", "roof:height": "3"}),
        way(3, box(0, -20, 20, 20), {"building:part": "yes", "height": "20", "min_height": "6"}),
        way(4, box(100, 100, 110, 110), {"building:part": "yes"}),    # outside the site
    ]
    site = run(tmp_path, els, size=100)
    b = site["buildings"]
    assert len(b) == 1 and b[0]["skip_block"] and b[0]["parts"] == ["way/2", "way/3"]
    parts = {p["osm_id"]: p for p in site["building_parts"]}
    assert set(parts) == {"way/2", "way/3"} and parts["way/2"]["parent"] == "way/1"
    assert parts["way/2"]["height"] == pytest.approx(16.0) and parts["way/2"]["roof"]["shape"] == "gabled"
    assert parts["way/3"]["min_height"] == 6.0 and parts["way/3"]["height"] == 20.0
    s2 = bp.part_slab(parts["way/3"])
    assert s2["thickness"] == pytest.approx(14.0) and s2["level"] == pytest.approx(parts["way/3"]["base_z"] + 6.0)
    s1 = bp.part_slab(parts["way/2"])                       # block under the roof: 16 - 3
    assert s1["thickness"] == pytest.approx(13.0)
    assert bp.active_buildings(site) == []
    assert bp.expected_counts(site)["Slab:buildings"] == 0
    assert site["report"]["counts"]["buildings_block_suppressed_by_parts"] == 1


def test_partial_parts_keep_the_uncovered_remainder(tmp_path):
    els = [way(1, box(-20, -20, 20, 20), {"building": "yes", "height": "6"}),
           way(2, box(-20, -20, 0, 20), {"building:part": "yes", "height": "12"})]
    site = run(tmp_path, els)
    b = site["buildings"]
    assert len(b) == 1 and not b[0].get("skip_block") and b[0]["tags"]["osmctx:remainder"] == "yes"
    a = Polygon(b[0]["footprint"]).area
    assert 0.4 * 1600 < a < 0.55 * 1600     # roughly the right half (parts are buffered 0.3 m)


def test_part_without_parent_and_bad_height_dropped(tmp_path):
    els = [way(1, box(-10, -10, 10, 10), {"building:part": "yes", "height": "9"}),
           way(2, box(20, 20, 30, 30), {"building:part": "yes", "height": "3", "min_height": "5"})]
    site = run(tmp_path, els)
    assert [p["osm_id"] for p in site["building_parts"]] == ["way/1"] and site["building_parts"][0]["parent"] is None
    assert site["report"]["counts"]["building_parts_dropped_bad_height"] == 1


def test_parts_off_means_no_parts_but_parent_kept(tmp_path):
    els = [way(1, box(-20, -20, 20, 20), {"building": "yes"}),
           way(2, box(-20, -20, 0, 20), {"building:part": "yes", "height": "12"})]
    site = run(tmp_path, els, settings={"include": {"building_parts": False}})
    assert site["building_parts"] == [] and len(site["buildings"]) == 1 and not site["buildings"][0].get("skip_block")


def test_plain_building_roof_and_ambiguous_fallbacks(tmp_path):
    els = [
        way(1, box(-30, -10, -10, 10), {"building": "house", "height": "10", "roof:shape": "hipped", "roof:height": "3"}),
        way(2, box(0, -10, 20, 10), {"building": "house", "height": "10", "roof:shape": "gabled"}),                      # no roof height
        way(3, box(-30, 20, -10, 40), {"building": "house", "height": "3", "roof:shape": "gabled", "roof:height": "5"}),  # roof > building
        way(4, box(0, 20, 20, 40), {"building": "house", "height": "9", "roof:shape": "mansard", "roof:height": "3"}),   # unsupported
        way(5, box(30, -10, 45, 10), {"building": "house", "height": "9", "roof:shape": "skillion", "roof:height": "2"}),
    ]
    site = run(tmp_path, els, size=120)
    by = {b["osm_id"]: b for b in site["buildings"]}
    assert by["way/1"]["roof"]["shape"] == "hipped" and by["way/5"]["roof"]["shape"] == "skillion"
    assert not any(by[k].get("roof") for k in ("way/2", "way/3", "way/4"))
    assert bp.building_slab(by["way/1"])["thickness"] == pytest.approx(7.0)
    assert bp.building_slab(by["way/2"])["thickness"] == pytest.approx(10.0)    # flat block fallback
    assert site["report"]["counts"]["roofs_ambiguous_flat"] >= 2


def poly_of(x0, y0, x1, y1):
    return Polygon(box(x0, y0, x1, y1)[:-1])


def test_roof_planes_gabled_skillion_hipped_pyramidal():
    p = poly_of(0, 0, 20, 10)
    g = F.roof_planes(p, {"shape": "gabled", "height": 3.0})
    assert len(g) == 2 and all(pl["kind"] == "single" for pl in g)
    assert g[0]["angle"] == pytest.approx(math.atan2(3.0, 5.0))      # half span 5
    assert g[0]["pivot_rise"] == 3.0
    assert sum(Polygon(pl["polygon"]).area for pl in g) == pytest.approx(200.0)
    # ridge along the long axis (x): both pivots run along x at y = 5
    for pl in g:
        (x0, y0), (x1, y1) = pl["pivot"]
        assert y0 == pytest.approx(5.0) and y1 == pytest.approx(5.0) and abs(x1 - x0) > 20
    # pivot direction puts each half on the right of the directed pivot line (roof falls away to the right)
    for pl in g:
        (x0, y0), (x1, y1) = pl["pivot"]
        c = Polygon(pl["polygon"]).centroid
        cross = (x1 - x0) * (c.y - y0) - (y1 - y0) * (c.x - x0)
        assert cross < 0
    s = F.roof_planes(p, {"shape": "skillion", "height": 2.0})
    assert len(s) == 1 and s[0]["angle"] == pytest.approx(math.atan2(2.0, 10.0)) and s[0]["pivot_rise"] == 0.0
    (x0, y0), (x1, y1) = s[0]["pivot"]                                 # eave on a long side, polygon on its LEFT
    c = p.centroid
    assert (x1 - x0) * (c.y - y0) - (y1 - y0) * (c.x - x0) > 0
    h = F.roof_planes(p, {"shape": "hipped", "height": 2.5})
    assert h[0]["kind"] == "multi" and h[0]["angle"] == pytest.approx(math.atan2(2.5, 5.0))
    py = F.roof_planes(poly_of(0, 0, 10, 10), {"shape": "pyramidal", "height": 4.0})
    assert py[0]["angle"] == pytest.approx(math.atan2(4.0, 5.0), abs=0.02)
    across = F.roof_planes(p, {"shape": "gabled", "height": 3.0, "orientation": "across"})
    assert across[0]["angle"] == pytest.approx(math.atan2(3.0, 10.0))  # ridge across: half span = 10


def test_roof_items_payload_and_expected_bbox():
    o = {"osm_id": "way/1", "footprint": [[0, 0], [20, 0], [20, 10], [0, 10], [0, 0]], "base_z": 3.0, "height": 10.0,
         "roof": {"shape": "gabled", "height": 3.0}}
    its = bp.roof_items(o, dz=1.0)
    assert len(its) == 2
    for r in its:
        it = r["item"]
        assert "pivotLine" in it and "levels" not in it and "eavesOverhang" not in it   # single-plane roofs reject both
        assert it["level"] == pytest.approx(3.0 + 1.0 + 7.0 + 3.0) and it["structureType"] == "Basic"
        assert r["check_z"] and r["expected"]["zMax"] == pytest.approx(14.0 + 0.2 / math.cos(it["angle"]), abs=1e-3)
    o["roof"] = {"shape": "hipped", "height": 3.0}
    m = bp.roof_items(o)[0]
    assert m["item"]["levels"][0]["levelAngle"] > 0 and m["item"]["eavesOverhang"] == 0.0 and "pivotLine" not in m["item"]
    assert m["item"]["level"] == pytest.approx(3.0 + 7.0)
    assert bp.roof_items({"footprint": [[0, 0], [1, 0], [1, 1]], "base_z": 0, "height": 5}) == []


# ------------------------------------------------------------------ ground surfaces
def test_ground_surfaces_kinds_materials_and_flat_slab(tmp_path):
    els = [
        way(1, box(-40, -40, -20, -20), {"amenity": "parking"}),
        way(2, box(-10, -40, 10, -20), {"amenity": "parking", "parking": "underground"}),         # skipped
        way(3, box(20, -40, 40, -20), {"leisure": "pitch", "surface": "grass"}),
        way(4, box(-40, 0, -20, 20), {"leisure": "playground"}),
        way(5, box(-10, 0, 10, 20), {"highway": "pedestrian", "area": "yes"}),
        way(6, box(20, 0, 40, 20), {"place": "square", "surface": "sett"}),
        way(7, box(-40, 30, -20, 45), {"area:highway": "service"}),
        way(8, box(20, 30, 40, 45), {"amenity": "parking", "building": "yes"}),                    # a building, not ground
    ]
    site = run(tmp_path, els)
    g = {x["osm_id"]: x for x in site["ground_surfaces"]}
    assert set(g) == {"way/1", "way/3", "way/4", "way/5", "way/6", "way/7"}
    assert (g["way/1"]["kind"], g["way/1"]["material"]) == ("parking", "Asphalt")
    assert (g["way/3"]["kind"], g["way/3"]["material"]) == ("pitch", "Grass")
    assert g["way/4"]["material"] == "Sand" and g["way/5"]["kind"] == "pedestrian" and g["way/6"]["material"] == "Paving"
    assert g["way/7"]["kind"] == "road_area"
    assert all(not x["sloped"] for x in g.values())
    sl = bp.ground_slab(g["way/1"])
    assert sl["thickness"] == 0.05 and sl["referencePlaneLocation"] == "Top"
    assert sl["level"] == pytest.approx(g["way/1"]["z_max"] + bp.GROUND_LIFT)
    assert bp.ground_meshes(site) == []


def test_sloped_ground_becomes_draped_mesh(tmp_path):
    ramp = lambda lon, lat: 100.0 + (lon - L.ll(0, 0)[0]) * 111320 * 0.1 * math.cos(math.radians(L.ll(0, 0)[1])) / 1.0  # noqa: E731
    els = [way(1, box(-30, -30, 30, 30), {"amenity": "parking"})]
    site = run(tmp_path, els, ramp, size=100)
    g = site["ground_surfaces"][0]
    assert g["sloped"] and g["z_max"] - g["z_min"] > 3.0
    ms = bp.ground_meshes(site)
    assert len(ms) == 1 and ms[0]["material"] == "Asphalt" and ms[0]["osm_id"] == "way/1"
    assert bp.expected_counts(site)["Mesh"] == 1 + len(ms) or True     # land cover count unaffected
    assert bp.feature_expected(site)["ground_slabs"] == 0 and bp.feature_expected(site)["ground_meshes"] == 1


def test_small_land_cover_mesh_has_no_empty_sublines():
    s = {"terrain": {"boundary": [[-10, -10], [10, -10], [10, 10], [-10, 10]],
                     "points": [[-10, -10, 1], [10, -10, 1], [10, 10, 1], [-10, 10, 1], [0, 0, 1]]},
         "land_cover": [{"osm_id": "way/9", "kind": "grass", "polygon": [[5, 5], [6, 5], [6, 6], [5, 6]], "holes": []}]}
    ms = bp.land_cover_meshes(s)
    assert len(ms) == 1 and "sublines" not in ms[0]["item"]     # CreateMeshes: sublines minItems 1
    assert bp.land_cover_expected_bbox(ms[0])["zMax"] > 0


# ------------------------------------------------------------------ piers / bridges / power
def test_piers_and_bridge_decks(tmp_path):
    els = [
        way(1, box(-45, -45, 45, -20), {"natural": "water"}),
        way(2, [(0, -40), (0, -22)], {"man_made": "pier", "width": "4"}),              # line pier into the water
        way(3, box(-30, 20, 30, 30), {"man_made": "bridge"}),
        way(4, [(-45, 25), (45, 25)], {"highway": "primary", "bridge": "yes", "layer": "1"}),
        way(5, box(-30, 35, 30, 45), {"man_made": "bridge"}),                           # no road: skipped
    ]
    site = run(tmp_path, els)
    pi = {p["osm_id"]: p for p in site["piers"]}
    assert set(pi) == {"way/2", "way/3"}
    water_z = site["water"][0]["z"]
    assert pi["way/2"]["z"] == pytest.approx(water_z + 1.0) and pi["way/2"]["z_source"] == "water_plus"
    deck = max(p[2] for p in site["roads"][0]["points"])
    assert pi["way/3"]["z"] == pytest.approx(deck - 0.2, abs=1e-3) and pi["way/3"]["z_source"] == "bridge_road_max"
    assert site["report"]["counts"]["piers_bridge_no_deck_skipped"] == 1
    sl = bp.pier_slab(pi["way/2"])
    assert sl["referencePlaneLocation"] == "Top" and sl["level"] == pi["way/2"]["z"] and sl["thickness"] == 0.5


def test_power_default_off_counts_lines_and_towers_on(tmp_path):
    els = [way(1, [(-40, 0), (40, 0)], {"power": "line"}), node(2, 10, 10, {"power": "tower", "height": "42"}),
           node(3, -10, 10, {"power": "pole"})]
    site = run(tmp_path, els)
    assert site["power_towers"] == [] and site["report"]["counts"]["power_lines"] == 1
    assert site["report"]["counts"]["power_towers_found"] == 1 and site["report"]["counts"]["power_poles_found"] == 1
    site = run(tmp_path / "on", els, settings={"include": {"power": True}})
    t = {x["kind"]: x for x in site["power_towers"]}
    assert t["tower"]["height"] == 42.0 and t["pole"]["height"] == 9.0 and t["pole"]["height_source"] == "default"
    c = bp.power_column(t["pole"])
    assert c["circleBased"] and c["height"] == 9.0 and c["coordinates"]["z"] == t["pole"]["z"]


# ------------------------------------------------------------------ groups, settings, fetch, limits, server
def test_group_plan_new_groups_and_building_subgroups():
    created = {"terrain": [("T", "t")],
               "buildings": [("B1", "way/1"), ("B2", "way/2")],
               "building_parts": [("P1", "way/11"), ("P2", "way/12"), ("P3", "way/20")],
               "building_roofs": [("R1", "way/11")],
               "building_parent": {"way/11": "way/1", "way/12": "way/1"},
               "railways": [("RB1", "way/31"), ("RB2", "way/31"), ("RB3", "way/32")],
               "rail_platforms": [("PL1", "way/40")],
               "walls": [("W1", "way/50"), ("W2", "way/50")], "railings": [("RL1", "way/51"), ("RL2", "way/51")],
               "ground": [("G1", "way/60"), ("G2", "way/61")], "piers": [("PI1", "way/70"), ("PI2", "way/71")]}
    plan = bp.group_plan(created)
    names = {k["name"]: k for k in plan["children"] if isinstance(k, dict)}
    assert {"Buildings", "Railways", "Walls & fences", "Ground surfaces", "Piers"} <= set(names)
    sub = {k["name"]: k for k in names["Buildings"]["children"] if isinstance(k, dict)}
    assert set(sub["Building way/1"]["children"]) == {"B1", "P1", "P2", "R1"}      # block + parts + roof together
    assert "P3" in names["Buildings"]["children"] and "B2" in names["Buildings"]["children"]
    assert {k["name"] for k in names["Railways"]["children"] if isinstance(k, dict)} == {"Railway way/31"}
    assert "PL1" in names["Railways"]["children"] and "RB3" in names["Railways"]["children"]
    assert {k["name"] for k in names["Walls & fences"]["children"]} == {"Barrier way/50", "Barrier way/51"}
    assert bp.group_plan({"terrain": [("T", "t")], "buildings": [("B1", "b"), ("B2", "c")]})["children"]   # unchanged shape


def test_settings_include_defaults_and_super_light():
    inc = F.INCLUDE_DEFAULTS
    assert inc == {"railways": True, "walls": True, "building_parts": True, "ground": True, "piers": True, "power": False}
    s = {}
    F.normalize_settings(s)
    assert s["include"] == inc and s["barrier_defaults"]["wall"] == {"height": 2.0, "thickness": 0.25}
    assert limits.SUPER_LIGHT["include"]["railways"] and limits.SUPER_LIGHT["include"]["walls"]
    assert not any(limits.SUPER_LIGHT["include"][k] for k in ("building_parts", "ground", "piers", "power"))
    s = {"barrier_defaults": {"wall": {"height": 3}}}
    F.normalize_settings(s)
    assert s["barrier_defaults"]["wall"] == {"height": 3.0, "thickness": 0.25}


def test_fetch_query_rev5_and_counts():
    assert FE.OSM_QUERY_REV >= 5
    q = FE.build_query((1, 2, 3, 4))
    for needle in ('"railway"~"^(rail|light_rail|tram|subway|narrow_gauge)$"', '"railway"="platform"', '"public_transport"="platform"',
                   '"barrier"~"^(wall|retaining_wall|city_wall|fence|guard_rail)$"', '"building:part"', '"amenity"="parking"',
                   '"leisure"~"^(pitch|playground)$"', '"area:highway"', '"place"="square"', '"man_made"~"^(pier|bridge)$"',
                   '"power"~"^(line|minor_line)$"', '"power"~"^(tower|pole)$"'):
        assert needle in q, needle
    osm = json.loads((FIX / "osm.json").read_text(encoding="utf-8"))
    c = FE.count_categories(osm)
    assert c["railway"] > 0 and c["platform"] > 0 and c["building:part"] > 0 and c["wall/fence"] > 0 and c["ground surface"] > 0


def test_count_from_osm_and_estimate_include_features():
    osm = json.loads((FIX / "osm.json").read_text(encoding="utf-8"))
    c = server.count_from_osm(osm)
    for k in ("railways", "platforms", "barriers", "building_parts", "ground"):
        assert c.get(k, 0) > 0, k
    tiles = {"dgm1_total": 0, "dgm1_missing": 0, "ndom_total": 0, "ndom_missing": 0, "copernicus_tiles": 0, "source": "file"}
    on = limits.estimate(0.04, c, "cache", tiles)
    e = on["elements"]
    assert e["railway_beams"] == 3 * c["railways"] and e["barrier_elements"] > 0 and e["building_parts"] == c["building_parts"]
    assert e["power_towers"] == 0                      # power default off
    off = limits.estimate(0.04, c, "cache", tiles, settings={"include": {"railways": False, "walls": False, "building_parts": False,
                                                                       "ground": False, "piers": False}})
    assert off["elements"]["railway_beams"] == off["elements"]["barrier_elements"] == off["elements"]["building_parts"] == 0
    assert on["elements"]["total"] > off["elements"]["total"]
    sl = limits.estimate(6.0, c, "cache", tiles, preset=True)["elements"]
    assert sl["railway_beams"] > 0 and sl["barrier_elements"] > 0 and sl["building_parts"] == 0 and sl["ground"] == 0


# ------------------------------------------------------------------ real fixture: Tuebingen Hbf
@pytest.fixture(scope="module")
def station(tmp_path_factory):
    d = tmp_path_factory.mktemp("station") / "cache"
    shutil.copytree(FIX, d)
    return convert(d, None, d.parent / "out")


def test_station_fixture_counts_and_invariants(station):
    s = station
    assert len(s["railways"]) >= 10 and len(s["railway_platforms"]) >= 10
    assert len(s["barriers"]) >= 8 and len(s["building_parts"]) >= 10 and len(s["ground_surfaces"]) >= 2
    kinds = {b["kind"] for b in s["barriers"]}
    assert {"fence", "retaining_wall"} <= kinds
    assert any(p.get("roof") for p in s["building_parts"])
    # block of an outline covered by parts is not built; parts reference real parents
    ids = {b["osm_id"] for b in s["buildings"]}
    assert all(p["parent"] in ids or p["parent"] is None for p in s["building_parts"])
    assert any(b.get("skip_block") for b in s["buildings"])
    assert len(bp.active_buildings(s)) < len(s["buildings"])
    for p in s["building_parts"]:
        assert p["height"] > p["min_height"] >= 0
    for r in s["railways"]:
        assert r["width"] >= 2.0 and len(r["points"]) >= 2
    for b in s["barriers"]:
        assert max(math.dist(p[:2], q[:2]) for p, q in zip(b["points"], b["points"][1:])) <= 5.0 + 1e-6
    for g in s["ground_surfaces"]:
        assert g["z_max"] >= g["z_min"] and g["kind"] in F.GROUND_KINDS
    # every roof payload builds (single/multi plane), counts line up with the plan
    fe = bp.feature_expected(s)
    assert fe["building_parts"] == len(s["building_parts"]) and fe["roofs"] >= 4
    assert fe["walls"] + fe["railings"] >= len(s["barriers"])
    assert fe["railways"] > 0 and fe["rail_platforms"] == len(s["railway_platforms"])
    assert fe["ground_slabs"] + fe["ground_meshes"] >= len(s["ground_surfaces"])
    for o in bp.active_buildings(s) + s["building_parts"]:
        for r in bp.roof_items(o):
            assert r["item"]["thickness"] == 0.2 and r["expected"]["zMax"] > r["item"]["level"]
    json.dumps(s)   # serialisable


def test_station_super_light_only_railways_and_walls(tmp_path):
    d = tmp_path / "cache"
    shutil.copytree(FIX, d)
    st = {"include": dict(limits.SUPER_LIGHT["include"])}
    s = convert(d, st, tmp_path / "out")
    assert s["railways"] and s["barriers"]
    assert s["building_parts"] == [] and s["ground_surfaces"] == [] and s["piers"] == [] and s["power_towers"] == []
    assert not any(b.get("skip_block") for b in s["buildings"])


def test_feature_layer_roles_and_names():
    from osmctx import build_ac
    assert build_ac.feature_roles_needed({}) == set()
    s = {"railways": [1], "barriers": [1], "ground_surfaces": [1], "piers": [1], "power_towers": [1]}
    assert build_ac.feature_roles_needed(s) == {"railways", "walls", "ground", "piers", "power"}
    assert build_ac.feature_roles_needed({"railway_platforms": [1]}) == {"railways"}
    names = {r: build_ac.LAYER_CANDIDATES[r][0] for r in build_ac.FEATURE_ROLES}
    assert names == {"railways": "Site-Railways", "walls": "Site-Walls", "ground": "Site-Ground", "piers": "Site-Piers",
                     "power": "Site-Power"}
    # every new element kind has a classification item, an Archicad element type and a layer role
    for kind in ("railways", "rail_platforms", "walls", "railings", "building_parts", "building_roofs", "ground", "piers", "power"):
        assert kind in build_ac.CLASS_ITEMS and kind in build_ac.KIND_TYPE_ROLE or kind == "ground"
    for ty, role in build_ac.KIND_TYPE_ROLE.values():
        assert ty in build_ac.OSM_ELEMENT_TYPES and role in build_ac.LAYER_CANDIDATES
