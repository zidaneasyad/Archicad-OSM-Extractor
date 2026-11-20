"""M4-C: land-cover meshes (build_plan) + accuracy-panel notes (server)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import server
from osmctx import build_plan as bp
from tests.test_build_plan import mini_site

ROOT = Path(__file__).resolve().parent.parent


def site_with_lc():
    s = mini_site()
    s["terrain"]["points"] = [[x, y, 5.0 + x * 0.1] for x, y, *_ in s["terrain"]["points"]]   # one consistent plane z = 5 + 0.1 x
    s["land_cover"] = [
        {"osm_id": "way/10", "kind": "beach", "polygon": [[-6, -6], [6, -6], [6, 2], [-6, 2], [-6, -6]], "holes": [], "area_m2": 96},
        {"osm_id": "way/11", "kind": "mangrove", "polygon": [[4, 4], [30, 4], [30, 9], [4, 9]], "holes": [], "area_m2": 130},
        {"osm_id": "way/12", "kind": "wetland", "polygon": [[50, 50], [60, 50], [60, 60]], "holes": [], "area_m2": 50},
        {"osm_id": "way/13", "kind": "sand", "polygon": [[-9, -9], [9, -9], [9, 9], [-9, 9]],
         "holes": [[[-2, -2], [2, -2], [2, 2], [-2, 2]]], "area_m2": 300},
    ]
    return s


def test_sampler_bilinear_matches_plane():
    s = site_with_lc()
    smp = bp.TerrainSampler(s)
    for x, y in [(-5, -5), (0, 0), (2.5, 3.0), (-3.3, 4.1)]:
        assert smp.z(x, y) == pytest.approx(5.0 + x * 0.1, abs=1e-6)
    assert abs(smp.z(9, 9) - 5.9) < 0.6   # outside the interior grid: IDW over the nearest points, finite


def test_land_cover_meshes_drape_clip_and_skip():
    ms = bp.land_cover_meshes(site_with_lc(), dz=0.0, terrain_points="rows")
    ids = [m["osm_id"] for m in ms]
    assert ids == ["way/10", "way/11", "way/13"]   # way/12 lies fully outside the terrain outline
    m = ms[0]
    assert m["material"] == "Sand" and ms[1]["material"] == "Soil"
    it = m["item"]
    assert it["skirtType"] == "SurfaceOnly" and it["ridges"] == "UserDefined"
    for p in it["polygonCoordinates"]:   # outline z = terrain + lift (vertex z are relative to the mesh level)
        assert p["z"] + it["level"] == pytest.approx(5.0 + p["x"] * 0.1 + bp.LAND_COVER_LIFT, abs=1e-3)
    assert min(p["z"] for p in it["polygonCoordinates"]) == pytest.approx(0.0, abs=1e-3)
    assert it["sublines"] and all(len(s["coordinates"]) >= 2 for s in it["sublines"])
    # mangrove polygon sticks out of the outline (x to 30): clipped to x <= 10
    assert max(p["x"] for p in ms[1]["item"]["polygonCoordinates"]) <= 10 + 1e-6
    assert len(ms[2]["item"]["holes"]) == 1


def test_land_cover_expected_bbox_and_counts_and_dz():
    s = site_with_lc()
    ms = bp.land_cover_meshes(s, dz=1.5)
    bb = bp.land_cover_expected_bbox(ms[0])
    assert bb["xMin"] == pytest.approx(-6) and bb["xMax"] == pytest.approx(6)
    assert bb["zMin"] == pytest.approx(5.0 - 0.6 + 1.5 + bp.LAND_COVER_LIFT, abs=1e-3)
    assert bp.expected_counts(s)["Mesh"] == 1 + 3
    assert bp.expected_counts(mini_site())["Mesh"] == 1


def test_land_cover_none_is_empty():
    assert bp.land_cover_meshes(mini_site()) == []


def test_group_plan_has_land_cover_group():
    created = {"terrain": [("T", "terrain")], "land_cover": [("L1", "way/1"), ("L2", "way/2")],
               "buildings": [("B1", "b1"), ("B2", "b2")]}
    plan = bp.group_plan(created)
    names = [k["name"] for k in plan["children"] if isinstance(k, dict)]
    assert "Land cover" in names and "Buildings" in names


def test_real_depok_land_cover_maps():
    p = ROOT / "out" / "m4" / "parangtritis_depok" / "site.json"
    if not p.exists():
        pytest.skip("no m4 site")
    site = json.loads(p.read_text(encoding="utf-8"))
    ms = bp.land_cover_meshes(site)
    assert len(ms) >= 1 and ms[0]["kind"] == "beach" and ms[0]["material"] == "Sand"
    for m in ms:
        assert len(m["item"]["polygonCoordinates"]) >= 3


# ------------------------------------------------------------- accuracy notes
def note_site(**kw):
    s = {"report": {"counts": {}, "dem_accuracy": {}, "info": {"osm": {"osm_age_days": 0.5}}}, "points_of_interest": [], "cliffs": []}
    for k, v in kw.items():
        if k in ("counts", "dem_accuracy"):
            s["report"][k] = v
        elif k == "osm":
            s["report"]["info"]["osm"] = v
        else:
            s[k] = v
    return s


def test_notes_empty_for_flat_fresh_site():
    s = note_site(dem_accuracy={"slope": {"max_pct": 5.0, "p95_pct": 3.0, "cells_steep_pct": 0.0, "vertical_error_one_cell_m_p95": 1.0}})
    assert server.accuracy_notes(s) == []


def test_notes_slope_poi_stale_buildings():
    slope = {"max_pct": 174.6, "p95_pct": 152.4, "cells_steeper_than_deg": 45.0, "cells_steep_pct": 35.2,
             "vertical_error_one_cell_m_p95": 46.8}
    s = note_site(dem_accuracy={"slope": slope}, osm={"osm_age_days": 21.0},
                  counts={"buildings_steep_footprint": 3, "buildings_in_water": 2},
                  points_of_interest=[{"osm_id": "node/1", "kind": "peak", "name": "Mount Everest", "dem_z_abs": 8713.33,
                                       "ele": 8848.86, "dem_minus_ele": -135.53}])
    n = {x["kind"]: x for x in server.accuracy_notes(s)}
    assert set(n) == {"slope", "poi", "stale_osm", "buildings"}
    assert "175 %" in n["slope"]["text"] and n["slope"]["level"] == "warn"
    assert "Mount Everest" in n["poi"]["text"] and "-136" in n["poi"]["text"]
    assert "21 days" in n["stale_osm"]["text"]
    assert "3 on steep ground" in n["buildings"]["text"] and "2 mostly in water" in n["buildings"]["text"]


def test_notes_poi_small_diff_ignored_and_moderate_slope_is_info():
    s = note_site(dem_accuracy={"slope": {"max_pct": 40.1, "p95_pct": 29.7, "cells_steep_pct": 0.0, "vertical_error_one_cell_m_p95": 9.1}},
                  points_of_interest=[{"osm_id": "n", "kind": "peak", "dem_minus_ele": 2.0}])
    n = server.accuracy_notes(s)
    assert [x["kind"] for x in n] == ["slope"] and n[0]["level"] == "info"


def test_notes_real_everest():
    p = ROOT / "out" / "m4" / "everest" / "site.json"
    if not p.exists():
        pytest.skip("no m4 site")
    kinds = {x["kind"] for x in server.accuracy_notes(json.loads(p.read_text(encoding="utf-8")))}
    assert {"slope", "poi"} <= kinds


def test_build_has_land_cover_flag():
    import inspect

    from osmctx import build_ac
    assert inspect.signature(build_ac.build).parameters["land_cover"].default is True
    assert build_ac.CLASS_ITEMS["land_cover"] == "Terrain" and "land_cover" in build_ac.LAYERS
