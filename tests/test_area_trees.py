"""Offline tests: area-based tree scatter (mangrove / orchard / plant nursery / park / cemetery), grass land cover,
group names, settings, super-light, estimate, fetch query. Synthetic polygons + a real Jakarta fixture."""
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from shapely.geometry import Point, box

import server
from osmctx import build_ac
from osmctx import build_plan as bp
from osmctx import fetch as F
from osmctx import limits
from osmctx import vegetation as veg
from osmctx.convert import convert

from test_convert import flat, node, run, way

FIX = Path(__file__).resolve().parent / "fixtures" / "jakarta"


def sq(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def scattered(site, typ=None):
    return [t for t in site["trees"] if t.get("source_area") and (typ is None or t["source_area_type"] == typ)]


def mind(pts):
    a = np.asarray(pts)
    d = np.hypot(a[:, None, 0] - a[None, :, 0], a[:, None, 1] - a[None, :, 1])
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def test_mangrove_dense_estimated_and_land_cover_kept(tmp_path):
    s = run(tmp_path, [way(1, sq(-20, -20, 20, 20), {"natural": "wetland", "wetland": "mangrove"})], flat())
    tr = scattered(s, "mangrove")
    assert 62 <= len(tr) <= 64                                        # 1600 m2 / 25 m2
    assert mind([(t["x"], t["y"]) for t in tr]) >= 3.0 - 0.01
    assert all(t["estimated"] and t["height"] == 6.0 and t["height_source"] == "area_default"
               and t["source_area"] == "way/1" and t["osm_id"] == "way/1" and t["tags"]["leaf_type"] == "broadleaved"
               for t in tr)
    assert [lc["kind"] for lc in s["land_cover"]] == ["mangrove"]    # flat mesh stays under the trees
    c = s["report"]["counts"]
    assert c["trees_from_mangrove"] == len(tr) == c["trees_scattered_estimated"]
    info = s["report"]["info"]["tree_scatter"]
    assert info["by_area_type"]["mangrove"]["trees"] == len(tr) and info["enabled"]["mangrove"]


def test_wood_tagged_mangrove_is_scattered_as_mangrove(tmp_path):
    els = [way(1, sq(-20, -20, 20, 20), {"natural": "wood", "wetland": "mangrove"})]
    s = run(tmp_path, els, flat(), settings={"forest_scatter": True})
    assert {t["source_area_type"] for t in scattered(s)} == {"mangrove"}
    off = run(tmp_path / "o", els, flat(), settings={"forest_scatter": True, "tree_scatter": {"mangrove": False}})
    assert {t["source_area_type"] for t in scattered(off)} == {"forest"}


def test_orchard_rows_and_nursery(tmp_path):
    els = [way(1, sq(-30, -20, 30, 10), {"landuse": "orchard"}),            # 60 x 30 = 1800 m2
           way(2, sq(-30, 20, -10, 40), {"landuse": "plant_nursery"})]      # 400 m2
    s = run(tmp_path, els, flat())
    orch, nur = scattered(s, "orchard"), scattered(s, "plant_nursery")
    sp = math.sqrt(30.0)
    assert abs(len(orch) - 1800 / 30) <= 8
    xy = np.array([(t["x"], t["y"]) for t in orch])
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    np.fill_diagonal(d, np.inf)
    assert np.allclose(d.min(axis=1), sp, atol=0.05)                         # regular rows: every tree sp from a neighbour
    assert all(t["height"] == 5.0 for t in orch)
    assert 24 <= len(nur) <= 27 and all(t["height"] == 3.0 for t in nur)     # 400 / 15
    assert mind([(t["x"], t["y"]) for t in nur]) >= 2.5 - 0.01
    assert s["land_cover"] == []                                              # orchard/nursery are not land cover


def test_park_keeps_out_of_buildings_water_roads_pitches(tmp_path):
    els = [way(1, sq(-45, -45, 45, 45), {"leisure": "park"}),                          # 8100 m2
           way(2, sq(-10, -10, 10, 10), {"building": "yes"}),
           way(3, sq(20, 20, 40, 40), {"natural": "water"}),
           way(4, sq(-40, 20, -20, 40), {"leisure": "pitch"}),
           way(5, [(-45, -30), (45, -30)], {"highway": "residential"})]               # 5 m wide -> 3.5 m half-buffer
    s = run(tmp_path, els, flat())
    tr = scattered(s, "park")
    assert tr and all(t["height"] == 10.0 and t["estimated"] for t in tr)
    assert mind([(t["x"], t["y"]) for t in tr]) >= 6.0 - 0.01
    for t in tr:
        p = Point(t["x"], t["y"])
        assert not box(-10, -10, 10, 10).buffer(0.99).contains(p)
        assert not box(20, 20, 40, 40).buffer(0.99).contains(p)
        assert not box(-40, 20, -20, 40).buffer(0.99).contains(p)
        assert abs(t["y"] + 30) >= 3.4
    free = 8100 - 400 - 400 - 400 - 7 * 90
    assert 0.4 * free / 120 <= len(tr) <= free / 120 + 1      # dart throwing at 6 m spacing keeps a bit under target


def test_garden_and_cemetery_optional_grass_land_cover(tmp_path):
    els = [way(1, sq(-40, -40, -10, -10), {"leisure": "garden"}),
           way(2, sq(10, 10, 40, 40), {"landuse": "cemetery"}),
           way(3, sq(-40, 10, -10, 40), {"landuse": "grass"}),
           way(4, sq(10, -40, 40, -10), {"landuse": "meadow"}),
           way(5, sq(-5, -5, 5, 5), {"natural": "heath"})]
    s = run(tmp_path, els, flat())
    assert scattered(s, "park") and not scattered(s, "cemetery")                # cemetery trees off by default
    kinds = {lc["osm_id"]: lc["kind"] for lc in s["land_cover"]}
    assert kinds == {"way/2": "grass", "way/3": "grass", "way/4": "meadow", "way/5": "heath"}   # garden/park: no mesh
    on = run(tmp_path / "c", els, flat(), settings={"tree_scatter": {"cemetery": True}})
    assert scattered(on, "cemetery")
    assert {t["source_area"] for t in scattered(on)} == {"way/1", "way/2"}       # grass/meadow/heath never get trees


def test_switches_density_override_and_deterministic(tmp_path):
    els = [way(1, sq(-20, -20, 20, 20), {"wetland": "mangrove", "natural": "wetland"}),
           way(2, sq(-40, 25, -20, 45), {"landuse": "plant_nursery"}),
           way(3, sq(25, 25, 45, 45), {"leisure": "park"})]
    allon = run(tmp_path / "a", els, flat())
    assert {t["source_area_type"] for t in scattered(allon)} == {"mangrove", "plant_nursery", "park"}
    off = run(tmp_path / "b", els, flat(), settings={"tree_scatter": {"mangrove": False, "plant_nursery": False, "park": False}})
    assert scattered(off) == []
    only = run(tmp_path / "c", els, flat(), settings={"tree_scatter": {"mangrove": False, "plant_nursery": False}})
    assert {t["source_area_type"] for t in scattered(only)} == {"park"}
    dens = run(tmp_path / "d", els[:1], flat(), settings={"tree_scatter_params": {"mangrove": {"area_per_tree": 100, "height": 4,
                                                                                               "min_spacing": 2}}})
    tr = scattered(dens, "mangrove")
    assert 14 <= len(tr) <= 16 and all(t["height"] == 4.0 for t in tr)
    again = run(tmp_path / "e", els, flat())
    assert [(t["x"], t["y"]) for t in again["trees"]] == [(t["x"], t["y"]) for t in allon["trees"]]
    nt = run(tmp_path / "f", els, flat(), settings={"include": {"trees": False}})
    assert nt["trees"] == []


def test_measured_height_beats_area_default(tmp_path):
    els = [way(1, sq(-20, -20, 20, 20), {"landuse": "plant_nursery"})]
    s = run(tmp_path, els, flat(), settings={"tree_scatter_params": {"plant_nursery": {"height": 2.0}}})
    assert all(t["height"] == 2.0 and t["height_source"] == "area_default" for t in scattered(s))
    # area tags: an explicit genus on the area is used by the chain (genus table) and not flagged area_default
    els = [way(1, sq(-20, -20, 20, 20), {"landuse": "orchard", "genus": "Mangifera"})]
    t0 = scattered(run(tmp_path / "g", els, flat()))
    assert t0 and all(t["estimated"] for t in t0)


# --------------------------------------------------------------------------- builder / groups
def test_land_cover_grass_material_and_resolution():
    assert bp.LAND_COVER_MATERIAL["grass"] == bp.LAND_COVER_MATERIAL["meadow"] == bp.LAND_COVER_MATERIAL["heath"] == "Grass"
    assert bp.LAND_COVER_MATERIAL["mangrove"] == "Soil"
    r = build_ac.resolve_material_names({"Grass", "Sand"}, {"Soil", "Sand", "Grass - Lawn", "Lawn Grass Long"})
    assert r == {"Grass": "Grass - Lawn", "Sand": "Sand"}               # contains-match, shortest first
    assert build_ac.resolve_material_names({"Grass"}, {"Soil", "Sand"}) == {"Grass": "Soil"}
    assert build_ac.resolve_material_names({"Grass"}, {"Grass", "Grass2"}) == {"Grass": "Grass"}
    assert build_ac.resolve_material_names({"Grass"}, {"Sand"}) == {"Grass": None}      # nothing -> no override


def test_land_cover_meshes_grass(tmp_path):
    s = run(tmp_path, [way(1, sq(-20, -20, 20, 20), {"landuse": "grass"})], flat())
    m = bp.land_cover_meshes(s)
    assert [(x["kind"], x["material"]) for x in m] == [("grass", "Grass")]


def test_group_plan_area_names():
    created = {"terrain": [("tm", "terrain")], "trees": [("a1", "way/9"), ("a2", "way/9"), ("b1", "way/10"), ("b2", "way/10"),
                                                         ("c1", "way/11"), ("c2", "way/11"), ("n1", "node/1")],
               "tree_area_ids": ["way/9", "way/10", "way/11"],
               "tree_area_types": {"way/9": "mangrove", "way/10": "park", "way/11": "forest"}}
    node_ = bp.group_plan(created)
    trees = next(k for k in node_["children"] if isinstance(k, dict) and k["name"] == "Trees")["children"]
    names = {k["name"] for k in trees if isinstance(k, dict)}
    assert names == {"Mangrove way/9", "Park way/10", "Forest way/11"} and "n1" in trees


# --------------------------------------------------------------------------- settings / limits / fetch
def test_super_light_turns_all_area_scatter_off():
    r = limits.resolve(6.0, {"tree_scatter": {"park": True, "mangrove": True}, "forest_scatter": True})
    assert not any(r["settings"]["tree_scatter"].values()) and r["settings"]["forest_scatter"] is False
    assert set(limits.SUPER_LIGHT["tree_scatter"]) == {"mangrove", "orchard", "plant_nursery", "park", "cemetery", "forest"}
    o = limits.resolve(6.0, {"tree_scatter": {"park": True}}, override=True)
    assert o["settings"]["tree_scatter"] == {"park": True}


def test_estimate_counts_area_scatter_and_server_counts():
    counts = {"buildings": 0, "roads": 0, "trees": 5, "water": 0, "mangrove_m2": 10000, "park_m2": 24000, "orchard_m2": 6000}
    tiles = {"source": "copernicus", "dgm1_total": 0, "dgm1_missing": 0, "ndom_total": 0, "ndom_missing": 0, "copernicus_tiles": 1}
    e = limits.estimate(0.1, counts, "cache", tiles)["elements"]
    want = {"mangrove": int(10000 * 0.5 / 25), "park": int(24000 * 0.5 / 120), "orchard": int(6000 * 0.5 / 30)}
    assert e["scatter_by_type"] == want and e["scatter_trees"] == sum(want.values()) and e["trees"] == 5 + sum(want.values())
    off = limits.estimate(0.1, counts, "cache", tiles, settings={"tree_scatter": {"mangrove": False, "park": False,
                                                                                  "orchard": False}})["elements"]
    assert off["scatter_trees"] == 0
    sl = limits.estimate(0.1, counts, "cache", tiles, preset=True)["elements"]
    assert sl["scatter_trees"] == 0
    osm = {"elements": [{"type": "way", "tags": {"leisure": "park"}, "bounds": {"minlat": 0, "maxlat": 0.0009, "minlon": 0, "maxlon": 0.0009}},
                        {"type": "way", "tags": {"landuse": "orchard"}, "bounds": {"minlat": 0, "maxlat": 0.0009, "minlon": 0, "maxlon": 0.0009}}]}
    c = server.count_from_osm(osm)
    assert 9000 < c["park_m2"] < 11000 and 9000 < c["orchard_m2"] < 11000


def test_fetch_query_rev4_green_tags():
    q = F.build_query((0, 0, 1, 1))
    for t in ('"leisure"~"^(park|garden)$"', "plant_nursery", "orchard", "village_green", "recreation_ground",
              '"natural"~"^(grassland|heath)$"', '"wetland"~"^(tidalflat|saltmarsh|mangrove)$"'):
        assert t in q
    assert F.OSM_QUERY_REV >= 4
    c = F.count_categories({"elements": [{"tags": {"leisure": "park"}}, {"tags": {"landuse": "plant_nursery"}},
                                         {"tags": {"landuse": "grass"}}, {"tags": {"wetland": "mangrove"}}]})
    assert c["park/garden"] == 1 and c["plant_nursery"] == 1 and c["grass/meadow/heath"] == 1 and c["mangrove"] == 1


# --------------------------------------------------------------------------- real fixture (Jakarta, Kamal Muara)
def test_jakarta_fixture_trees_from_areas(tmp_path):
    meta = json.loads((FIX / "meta.json").read_text(encoding="utf-8"))
    assert meta["osm_query_rev"] >= 4
    d = tmp_path / "cache"
    shutil.copytree(FIX, d)
    s = convert(d, None, tmp_path / "out")
    by = s["report"]["info"]["tree_scatter"]["by_area_type"]
    assert not [t for t in s["trees"] if not t.get("source_area")]           # the OSM has 0 natural=tree here
    assert by.get("mangrove", {}).get("trees", 0) > 0                          # 3 mangrove areas -> trees
    assert all(t["estimated"] and t["source_area_type"] in ("mangrove", "orchard", "plant_nursery", "park") for t in s["trees"])
    assert s["report"]["counts"]["trees"] == len(s["trees"]) > 0
    again = convert(d, None, tmp_path / "out2")
    assert [(t["x"], t["y"]) for t in again["trees"]] == [(t["x"], t["y"]) for t in s["trees"]]
    assert any(lc["kind"] == "mangrove" for lc in s["land_cover"])
