import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from osmctx import build_plan as bp

SITE = Path(__file__).resolve().parent.parent / "out" / "site.json"


def mini_site():
    ring = [[-10, -10], [10, -10], [10, 10], [-10, 10]]
    interior = [[x, y, 5.0 + x * 0.1] for y in (-5, 0, 5) for x in (-5, 0, 5)]
    return {
        "origin": {"lat": 48.5, "lon": 9.0, "epsg": 32632, "easting": 500000.5, "northing": 5400000.25, "elevation": 300.0},
        "terrain": {"boundary": ring, "points": [[x, y, 1.0 + x * 0.1] for x, y in ring] + interior},
        "buildings": [
            {"osm_id": "way/1", "footprint": [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
             "holes": [[[1, 1], [2, 1], [2, 2], [1, 2]]], "base_z": 3.0, "height": 9.0},
            {"osm_id": "way/2", "footprint": [[5, 5], [6, 5], [6, 6]], "holes": [], "base_z": 2.0, "height": 0.0},
        ],
        "roads": [{"osm_id": "way/3", "width": 4.0, "points": [[0, 0, 1.0], [10, 0, 3.0], [10, 0, 3.0], [10, 10, 3.0]]}],
        "trees": [{"osm_id": "node/4", "x": 1, "y": 2, "z": 3.5, "height": 10.0}],
        "water": [{"osm_id": "way/5", "kind": "water", "polygon": [[0, 0], [5, 0], [5, 5], [0, 5]], "z": 0.0,
                   "holes": [[[1, 1], [2, 1], [2, 2]]]}],
        "report": {"info": {"dem": {"source": "Copernicus GLO-30 (x)"}, "terrain": {"spacing_m": 5.0}}},
    }


def test_terrain_mesh_outline_and_sublines():
    m = bp.terrain_mesh(mini_site(), terrain_points="rows")
    assert len(m["polygonCoordinates"]) == 4
    assert m["polygonCoordinates"][1] == {"x": 10.0, "y": -10.0, "z": 2.0}
    assert len(m["sublines"]) == 3  # one per grid row
    assert sum(len(s["coordinates"]) for s in m["sublines"]) == 9
    assert m["level"] == pytest.approx(min(p["z"] for p in m["polygonCoordinates"]))
    assert m["skirtType"] == "SolidBodyWithSkirt"


def test_terrain_z_offset():
    # dz moves the mesh level; vertex z are relative to the level (Archicad semantics), so they don't change.
    base = bp.terrain_mesh(mini_site())
    m = bp.terrain_mesh(mini_site(), dz=100.0)
    assert m["level"] == pytest.approx(base["level"] + 100.0)
    assert m["polygonCoordinates"][0]["z"] == pytest.approx(base["polygonCoordinates"][0]["z"])


def test_terrain_vertex_z_relative_to_level():
    s = mini_site()
    for p in s["terrain"]["points"]:
        p[2] += 7.5  # lift the whole terrain: level must follow, relative z must not
    m = bp.terrain_mesh(s)
    assert m["level"] == pytest.approx(7.5)
    assert min(q["z"] for q in m["polygonCoordinates"]) == pytest.approx(0.0)
    assert all(q["z"] >= -1e-9 for sl in m["sublines"] for q in sl["coordinates"])


def test_building_slab_bottom_reference_and_thickness():
    s = bp.building_slab(mini_site()["buildings"][0])
    assert s["level"] == 3.0 and s["thickness"] == 9.0 and s["referencePlaneLocation"] == "Bottom"
    assert len(s["polygonCoordinates"]) == 4  # closing point removed
    assert len(s["holes"]) == 1 and len(s["holes"][0]["polygonCoordinates"]) == 4
    assert bp.building_expected_bbox_z(mini_site()["buildings"][0]) == (3.0, 12.0)


def test_building_min_height():
    s = bp.building_slab(mini_site()["buildings"][1])
    assert s["thickness"] == bp.MIN_BUILDING_HEIGHT and "holes" not in s


def test_road_beams_one_per_pair_slope_and_zero_length_dropped():
    beams = bp.road_beams(mini_site()["roads"][0])
    assert len(beams) == 2  # duplicate point dropped
    b0, b1 = beams
    assert b0["zCoordinate"] == 1.0 and b0["isSlanted"]
    assert b0["slantAngle"] == pytest.approx(math.atan2(2.0, 10.0), abs=1e-5)
    assert not b1["isSlanted"] and "slantAngle" not in b1
    assert b0["width"] == 4.0 and b0["height"] == bp.ROAD_HEIGHT and b0["anchorPoint"] == "TopCenter"


def test_road_beam_descending_negative_angle():
    b = bp.road_beams({"width": 2, "points": [[0, 0, 5], [10, 0, 3]]})[0]
    assert b["slantAngle"] < 0


def test_tree_object():
    t = bp.tree_object(mini_site()["trees"][0], "Deciduous Trees")
    assert t["coordinates"] == {"x": 1.0, "y": 2.0, "z": 3.5}
    assert t["dimensions"]["z"] == 10.0 and t["dimensions"]["x"] == pytest.approx(6.3)
    assert t["libraryPartName"] == "Deciduous Trees"


def test_water_slab_top_at_water_level():
    w = bp.water_slab(mini_site()["water"][0])
    assert w["referencePlaneLocation"] == "Top" and w["level"] == 0.0 and w["thickness"] == 0.1
    assert len(w["holes"]) == 1


def test_expected_counts_and_geo():
    s = mini_site()
    ec = bp.expected_counts(s)
    assert ec == {"Mesh": 1, "Slab:buildings": 2, "Beam": 2, "Object": 1, "Slab:water": 1}
    g = bp.geo_location_payload(s)
    assert g["surveyPoint"]["position"] == {"eastings": 500000.5, "northings": 5400000.25, "elevation": 300.0}
    assert g["surveyPoint"]["geoReferencingParameters"]["crsName"] == "EPSG:32632"
    assert "Copernicus GLO-30" in bp.source_string(s, "2026-01-01") and "2026-01-01" in bp.source_string(s, "2026-01-01")


@pytest.mark.skipif(not SITE.exists(), reason="out/site.json missing")
def test_real_site_json_maps_cleanly():
    site = json.loads(SITE.read_text(encoding="utf-8"))
    m = bp.terrain_mesh(site)
    assert len(m["polygonCoordinates"]) == len(site["terrain"]["boundary"])
    n_int = len(site["terrain"]["points"]) - len(site["terrain"]["boundary"])
    assert sum(len(s["coordinates"]) for s in m["sublines"]) >= n_int
    for b in site["buildings"]:
        s = bp.building_slab(b)
        assert s["thickness"] > 0 and len(s["polygonCoordinates"]) >= 3
    for r in site["roads"]:
        for b in bp.road_beams(r):
            assert b["width"] > 0
    assert bp.expected_counts(site)["Slab:buildings"] == len(site["buildings"])


def test_seo_links_payload():
    links = bp.seo_links("T", ["W", "R1", "R2"])
    assert len(links) == 3
    assert links[0] == {"targetId": {"guid": "T"}, "operatorId": {"guid": "W"}, "operation": "SubtractionUpwards",
                        "linkFlags": {"inheritOperatorAttributes": False, "skipPolygonHoles": False}}
    assert {l["targetId"]["guid"] for l in links} == {"T"}
    assert bp.seo_links("T", ["W"], "Subtraction")[0]["operation"] == "Subtraction"
    with pytest.raises(ValueError):
        bp.seo_links("T", ["W"], "Bogus")


def _created():
    return {"terrain": [("T", "terrain")],
            "buildings": [(f"B{i}", f"way/b{i}") for i in range(3)],
            "roads": [("R1a", "way/1"), ("R1b", "way/1"), ("R2", "way/2")],
            "trees": [("T1", "node/1"), ("T2", "node/2")],
            "water": [("W", "way/w")]}


def test_group_plan_three_levels_and_collapse():
    p = bp.group_plan(_created())
    assert p["name"] == "OSM Site Context"
    names = [k["name"] for k in p["children"] if isinstance(k, dict)]
    assert names == ["Buildings", "Roads", "Trees"]  # single water slab sits directly in top
    assert "T" in p["children"] and "W" in p["children"]
    roads = next(k for k in p["children"] if isinstance(k, dict) and k["name"] == "Roads")
    assert roads["children"][0] == {"name": "Road way/1", "children": ["R1a", "R1b"]}
    assert roads["children"][1] == "R2"  # single-beam road directly in Roads
    assert bp.group_depth(p) == 3
    order = [n["name"] for n in bp.group_nodes_postorder(p)]
    assert order.index("Road way/1") < order.index("Roads") < order.index("OSM Site Context")


def test_group_plan_water_group_when_many_and_none_when_single():
    c = _created()
    c["water"] = [("W1", "a"), ("W2", "b")]
    p = bp.group_plan(c)
    assert any(isinstance(k, dict) and k["name"] == "Water" for k in p["children"])
    assert bp.group_plan({"terrain": [("T", "t")]}) is None


# ---------------------------------------------------------------- tree modes
ALL_PARTS = {"Deciduous Trees", "Pine Trees", "Tree Model Detailed", "Pine Tree Model", "Tree Plan"}


def test_parse_tree_mode():
    assert bp.parse_tree_mode(None) == ("full3d", None)
    assert bp.parse_tree_mode("2d") == ("2d", None)
    assert bp.parse_tree_mode("FULL3D") == ("full3d", None)
    assert bp.parse_tree_mode("lib:My Tree") == ("lib", "My Tree")
    assert bp.parse_tree_mode("lib:") == ("full3d", None)
    assert bp.parse_tree_mode("nonsense") == ("full3d", None)


def test_tree_kind():
    assert bp.tree_kind({}) == "broad"
    assert bp.tree_kind({"tags": {"leaf_type": "needleleaved"}}) == "needle"
    assert bp.tree_kind({"tags": {"leaf_type": "broadleaved"}}) == "broad"


def test_resolve_tree_parts():
    ex = lambda n: n in ALL_PARTS
    p, w = bp.resolve_tree_parts("light", ex)
    assert p == {"broad": "Deciduous Trees", "needle": "Pine Trees"} and not w
    p, w = bp.resolve_tree_parts("full3d", ex)
    assert p == {"broad": "Tree Model Detailed", "needle": "Pine Tree Model"} and not w
    p, w = bp.resolve_tree_parts("2d", ex)
    assert p["broad"] == "Tree Plan" and not w
    p, w = bp.resolve_tree_parts("lib:Deciduous Trees", ex)
    assert p == {"broad": "Deciduous Trees", "needle": "Deciduous Trees"} and not w


def test_resolve_tree_parts_fallback_to_light():
    ex = lambda n: n in {"Deciduous Trees", "Pine Trees"}
    for mode in ("2d", "full3d", "lib:Nope"):
        p, w = bp.resolve_tree_parts(mode, ex)
        assert p["broad"] == "Deciduous Trees" and p["needle"] == "Pine Trees"
        assert len(w) == 1 and "light" in w[0]


def test_tree_lib_for_picks_conifer():
    parts = {"broad": "B", "needle": "N"}
    assert bp.tree_lib_for({"tags": {"leaf_type": "needleleaved"}}, parts) == "N"
    assert bp.tree_lib_for({"x": 0}, parts) == "B"


def test_terrain_style_params():
    from osmctx import build_plan as bp
    assert bp.terrain_style_params("legacy") == {"ridges": "AllSharp"}
    clean = bp.terrain_style_params("clean")
    assert clean["ridges"] == "UserDefined" and clean["showLines"] is False and clean["contourPen"] == bp.TERRAIN_CONTOUR_PEN


def test_2d_plan_styling_and_tree_rotation():
    from osmctx import build_ac as bac
    w = bp.water_slab(mini_site()["water"][0])
    assert w["floorFill"] == {"use": True}
    gs = bp.ground_slab({"polygon": [[0, 0], [5, 0], [5, 5], [0, 5]], "z_max": 1.0})
    assert gs["floorFill"] == {"use": True}
    t1 = bp.tree_object({"osm_id": "way/1", "x": 1.0, "y": 2.0, "z": 0.0, "height": 8.0}, "Deciduous Trees")
    t2 = bp.tree_object({"osm_id": "way/1", "x": 4.0, "y": 2.0, "z": 0.0, "height": 8.0}, "Deciduous Trees")
    assert 0.0 <= t1["angle"] < 2.0 * math.pi and t1["angle"] != t2["angle"]
    pens = [
        {"index": 1, "weight": 0.13, "color": {"red": 0.0, "green": 0.0, "blue": 0.0}},
        {"index": 19, "weight": 0.0, "color": {"red": 1.0, "green": 1.0, "blue": 1.0}},
        {"index": 61, "weight": 0.2, "color": {"red": 0.42, "green": 0.42, "blue": 0.42}},
        {"index": 99, "weight": 0.13, "color": {"red": 0.67, "green": 0.82, "blue": 1.0}},
        {"index": 102, "weight": 0.13, "color": {"red": 0.8, "green": 0.8, "blue": 0.8}},
        {"index": 104, "weight": 0.13, "color": {"red": 0.59, "green": 0.88, "blue": 0.59}},
    ]
    picked = bac.pick_plan_pens(pens)
    assert picked == {"water": 99, "ground": 104, "buildings": 102, "roads": 61}
