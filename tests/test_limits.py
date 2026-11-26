"""Offline tests: area tiers, super-light preset, estimate math, server guards, road_classes_exclude, terrain_min_spacing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

import server
from osmctx import limits
from osmctx.boundary import area_m2, square_boundary
from osmctx.convert import _merge_settings

from test_convert import run, way


def sq_km2(km2, lat=48.5193, lon=9.0565):
    return square_boundary(lat, lon, (km2 * 1e6) ** 0.5)


# ---------------------------------------------------------------- tiers
@pytest.mark.parametrize("a,t", [(0.01, "normal"), (0.25, "normal"), (0.2501, "large"), (5.0, "large"),
                                 (5.0001, "super_light"), (25.0, "super_light"), (25.01, "blocked")])
def test_tier_of(a, t):
    assert limits.tier_of(a) == t


def test_area_m2_matches_square():
    assert area_m2(sq_km2(1.0)) / 1e6 == pytest.approx(1.0, rel=0.01)
    assert area_m2({"type": "circle", "center": [9.0, 48.5], "radius_m": 1000}) / 1e6 == pytest.approx(3.14, rel=0.01)


def test_resolve_override_and_preset():
    r = limits.resolve(6.0, {"tree_mode": "full3d", "include": {"trees": True}})
    assert r["preset"] == "super_light" and r["settings"]["tree_mode"] == "2d"
    assert r["settings"]["road_classes_exclude"] == limits.SUPER_LIGHT["road_classes_exclude"]
    assert r["settings"]["include"] == {"trees": True, **limits.SUPER_LIGHT["include"]} and not r["blocked"]
    o = limits.resolve(6.0, {"tree_mode": "full3d"}, override=True)
    assert o["preset"] is None and o["override"] and o["settings"] == {"tree_mode": "full3d"}
    assert limits.resolve(2.0, {"tree_mode": "full3d"})["preset"] is None
    assert limits.resolve(30.0, {})["blocked"]


def test_super_light_mapping_to_converter():
    eff = limits.resolve(6.0, {"tree_mode": "full3d", "floor_height_m": 3.0})["settings"]
    s = _merge_settings(server.map_settings(eff))
    assert s["terrain_min_spacing"] == 20.0 and s["floor_height"] == 3.0
    assert s["road_classes_exclude"] == ["cycleway", "footway", "path", "service", "steps"]
    assert "tree_mode" not in server.map_settings(eff)  # builder-only
    assert server.map_settings({"override": True}) == {}


# ---------------------------------------------------------------- estimate
NO_TILES = {"source": "bw_dgm1", "dgm1_total": 0, "dgm1_missing": 0, "ndom_total": 0, "ndom_missing": 0,
            "copernicus_tiles": 0}


def test_estimate_math_counts_and_tiles():
    tiles = {**NO_TILES, "dgm1_total": 4, "dgm1_missing": 2, "ndom_total": 4, "ndom_missing": 2}
    e = limits.estimate(1.0, {"buildings": 100, "roads": 20, "trees": 30, "water": 2}, "cache", tiles)
    assert e["elements"]["road_beams"] == 60 and e["elements"]["total"] == 100 + 60 + 30 + 2 + 1
    b = e["breakdown_s"]
    assert b["dem_s"] == 2 * 70 + 2 * 70 and b["build_s"] == round(193 * 0.3)
    assert e["seconds"]["mid"] == pytest.approx(sum(b.values()), abs=3)
    assert e["seconds"]["low"] == pytest.approx(0.6 * e["seconds"]["mid"], abs=2)
    assert e["seconds"]["high"] == pytest.approx(1.7 * e["seconds"]["mid"], abs=2)
    assert e["counts_source"] == "cache" and e["text"].startswith("about ") and "min" in e["text"]


def test_estimate_area_only_and_preset():
    e = limits.estimate(2.0, None, "x", NO_TILES)
    assert e["counts_source"] == "area" and e["elements"]["buildings"] == 1200
    assert any("area only" in n for n in e["notes"])
    p = limits.estimate(2.0, None, "x", NO_TILES, preset=True)
    assert p["elements"]["roads_ways"] == e["elements"]["roads_ways"] // 2
    assert p["seconds"]["mid"] < e["seconds"]["mid"]


def test_format_range():
    assert limits.format_range(20, 50) == "about 20-50 s"
    assert limits.format_range(240, 480) == "about 4-8 min"
    assert "h" in limits.format_range(3600, 9000)


def test_dem_tiles_counts_cache(tmp_path):
    b = square_boundary(48.5193, 9.0565, 100)
    t = limits.dem_tiles(b, "auto", tmp_path)
    assert t["source"] == "bw_dgm1" and t["dgm1_total"] >= 1 and t["dgm1_missing"] == t["dgm1_total"] == t["ndom_missing"]
    d = tmp_path / "_tiles" / "bw_dgm1"
    d.mkdir(parents=True)
    from osmctx import fetch as F
    from osmctx.boundary import bbox, to_polygon
    subs = F.bw_subtiles(F.bbox_25832(bbox(to_polygon(b), F.DEM_MARGIN_M)))
    for e, n in subs:
        (d / f"dgm1_32_{e}_{n}.tif").write_bytes(b"")
        (d / f"dgm1_32_{e}_{n}.json").write_text("{}")
    t2 = limits.dem_tiles(b, "bw_dgm1", tmp_path)
    assert t2["dgm1_missing"] == 0 and t2["ndom_missing"] == len(subs)
    c = limits.dem_tiles(square_boundary(-8.8, 115.1, 100), "auto", tmp_path)
    assert c["source"] == "copernicus" and c["copernicus_tiles"] >= 1
    assert limits.dem_tiles(b, "file:x.tif", tmp_path)["source"] == "file"


# ---------------------------------------------------------------- server
@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    return TestClient(server.app)


def test_api_limits(client):
    d = client.get("/api/limits").json()
    assert (d["normal_max_km2"], d["super_light_above_km2"], d["block_above_km2"]) == (0.25, 5.0, 25.0)
    assert d["super_light"]["tree_mode"] == "2d"
    assert d["super_light"]["include"]["shrubs"] is False and d["super_light"]["include"]["railways"] is True and d["super_light"]["include"]["walls"] is True and not d["super_light"]["include"]["ground"] and d["super_light"]["forest_scatter"] is False and d["super_light"]["skip_ndom"]


def test_fetch_blocked_over_25_km2_and_old_1km2_block_gone(client, monkeypatch):
    import osmctx.fetch as f
    monkeypatch.setattr(f, "fetch", lambda b, cache_dir="cache", dem_source="auto", **kw: "cache/xyz")
    r = client.post("/api/fetch", json={"boundary": sq_km2(30), "settings": {}})
    assert r.status_code == 400 and r.json()["tier"] == "blocked" and "25" in r.json()["error"]
    r = client.post("/api/fetch", json={"boundary": sq_km2(2), "settings": {"dem_source": "copernicus"}})
    assert r.status_code == 200 and r.json()["tier"] == "large"
    r = client.post("/api/fetch", json={"boundary": sq_km2(7), "settings": {"dem_source": "copernicus"}})
    assert r.status_code == 200 and r.json()["preset"] == "super_light"
    r = client.post("/api/fetch", json={"boundary": sq_km2(7), "settings": {"dem_source": "copernicus"}, "override": True})
    assert r.status_code == 200 and r.json()["preset"] is None


def test_import_guard(client, tmp_path, monkeypatch):
    import osmctx.build_ac as b
    seen = {}
    monkeypatch.setattr(b, "build", lambda path, tree_mode="full3d", **k: seen.update(tm=tree_mode) or {"ok": 1})
    monkeypatch.setattr(server, "OUT_DIR", tmp_path)
    (tmp_path / "site.json").write_text('{"boundary": [[0,0],[1000,0],[1000,1000],[0,1000]]}')
    big = client.post("/api/import", json={"boundary": sq_km2(30), "tree_mode": "full3d"})
    assert big.status_code == 400 and not seen
    r = client.post("/api/import", json={"boundary": sq_km2(7), "tree_mode": "full3d"})
    assert r.status_code == 200 and seen["tm"] == "2d" and r.json()["preset"] == "super_light"
    r = client.post("/api/import", json={"boundary": sq_km2(7), "tree_mode": "full3d", "override": True})
    assert seen["tm"] == "full3d" and r.json()["preset"] is None
    # no boundary in body: area from site.json (1 km2 -> large, tree mode untouched)
    r = client.post("/api/import", json={"tree_mode": "full3d"})
    assert seen["tm"] == "full3d" and r.json()["tier"] == "large"
    (tmp_path / "site.json").write_text('{"boundary": [[0,0],[2450,0],[2450,2450],[0,2450]]}')
    client.post("/api/import", json={"tree_mode": "full3d"})
    assert seen["tm"] == "2d"


def test_estimate_endpoint(client, monkeypatch):
    monkeypatch.setattr(server, "get_counts", lambda b, s=None: ({"buildings": 50, "roads": 10, "trees": 5, "water": 1},
                                                                 {"source": "cache", "seconds": 0.01}))
    # sq_km2 lies in Baden-Wuerttemberg: LoD2 off here (tests/test_lod2_convert.py covers the LoD2 estimate)
    d = client.post("/api/estimate", json={"boundary": sq_km2(1), "settings": {"dem_source": "copernicus", "include": {"lod2": False}}}).json()
    assert d["tier"] == "large" and d["counts_source"] == "cache" and d["elements"]["total"] == 50 + 30 + 5 + 1 + 1
    monkeypatch.setattr(server, "get_counts", lambda b, s=None: (None, {"error": "down"}))
    d = client.post("/api/estimate", json={"boundary": sq_km2(1), "settings": {"dem_source": "copernicus"}}).json()
    assert d["counts_source"] == "area" and any("area only" in n for n in d["notes"])
    d = client.post("/api/estimate", json={"boundary": sq_km2(30)}).json()
    assert d["blocked"] and d["counts"] is None


# ---------------------------------------------------------------- converter
def _roads():
    return [way(1, [(-30, 0), (30, 0)], {"highway": "residential"}),
            way(2, [(-30, 10), (30, 10)], {"highway": "footway"}),
            way(3, [(-30, 20), (30, 20)], {"highway": "service"}),
            way(4, [(-30, 30), (30, 30)], {"highway": "cycleway"}),
            way(5, [(900, 900), (950, 900)], {"highway": "path"})]  # outside the boundary: not counted


def test_road_classes_exclude(tmp_path):
    site = run(tmp_path, _roads(), settings={"road_classes_exclude": ["Footway", "path", "service", "cycleway"]})
    assert [r["class"] for r in site["roads"]] == ["residential"]
    c = site["report"]["counts"]
    assert c["roads_skipped_excluded"] == 3 and c["roads_skipped_excluded_footway"] == 1
    assert c["roads_skipped_excluded_service"] == 1 and c["roads_skipped_excluded_cycleway"] == 1
    assert "roads_skipped_excluded_path" not in c
    assert any("roads_skipped_excluded=3" in w for w in site["report"]["warnings"])


def test_road_classes_exclude_default_keeps_all(tmp_path):
    site = run(tmp_path, _roads())
    assert len(site["roads"]) == 4 and "roads_skipped_excluded" not in site["report"]["counts"]


def test_terrain_min_spacing(tmp_path):
    a = run(tmp_path / "a", [], size=400.0)
    b = run(tmp_path / "b", [], size=400.0, settings={"terrain_min_spacing": 40.0})
    assert b["report"]["info"]["terrain"]["spacing_m"] >= 40.0
    assert b["report"]["counts"]["terrain_points"] < a["report"]["counts"]["terrain_points"]
