import time

import pytest

import server


def test_map_settings_renames_and_passes_rest():
    ui = {"floor_height_m": 3.4, "tree_height_m": 9, "default_levels": 4,
          "include": {"trees": False}, "road_widths": {"default": 4}}
    m = server.map_settings(ui)
    assert m == {"floor_height": 3.4, "tree_height": 9, "default_levels": 4,
                 "include": {"trees": False}, "road_widths": {"default": 4}}
    assert "floor_height_m" not in m and "tree_height_m" not in m
    assert server.map_settings(None) == {}


def test_map_settings_accepted_by_converter():
    from osmctx.convert import _merge_settings
    s = _merge_settings(server.map_settings({"floor_height_m": 3.5, "tree_height_m": 10, "default_levels": 5}))
    assert (s["floor_height"], s["tree_height"], s["default_levels"]) == (3.5, 10.0, 5.0)


def test_disclaimer_defaults_glo30():
    d = server.disclaimer()
    assert "\u00b15 m" in d["text"] and "\u00b110 m" in d["text"]
    assert "OpenStreetMap contributors" in " ".join(d["credits"])
    assert any("Copernicus" in c for c in d["credits"])


def test_disclaimer_dtm_numbers_from_report():
    acc = {"kind": "DTM", "vertical_m": 0.5, "source": "LGL Baden-Wuerttemberg DGM1"}
    d = server.disclaimer(acc, "LGL-BW (2026) dl-de/by-2-0", "dl-de/by-2-0")
    assert "\u00b10.5 m" in d["text"] and "DGM1" in d["text"] and "surface model" not in d["text"]
    assert any("LGL-BW" in c and "dl-de/by-2-0" in c for c in d["credits"])


def test_count_from_osm():
    osm = {"elements": [
        {"type": "way", "tags": {"building": "yes"}},
        {"type": "relation", "tags": {"building": "yes"}},
        {"type": "way", "tags": {"highway": "residential"}},
        {"type": "node", "tags": {"highway": "bus_stop"}},
        {"type": "node", "tags": {"natural": "tree"}},
        {"type": "way", "tags": {"natural": "tree_row"}},
        {"type": "way", "tags": {"waterway": "river"}},
        {"type": "relation", "tags": {"natural": "water"}},
        {"type": "way", "tags": {"natural": "wood"}},
    ]}
    assert server.count_from_osm(osm) == {"buildings": 2, "roads": 1, "trees": 2, "water": 2}


def _ok():
    return {"elements": [{"type": "count", "tags": {"total": str(n)}} for n in (1, 2, 3, 4)]}


def test_race_first_valid_wins_and_does_not_wait():
    def post(url, q):
        if url == "slow":
            time.sleep(3)
            return _ok()
        if url == "bad":
            raise RuntimeError("boom")
        if url == "empty":
            return {"elements": []}
        time.sleep(0.1)
        return _ok()
    t0 = time.monotonic()
    out, url, errs = server.race_overpass("q", ["slow", "bad", "empty", "fast"], timeout=5, post=post)
    assert time.monotonic() - t0 < 1.5
    assert out == {"buildings": 1, "roads": 2, "trees": 3, "water": 4} and url == "fast"


def test_race_all_fail_and_timeout():
    def bad(url, q):
        raise RuntimeError("x")
    out, url, errs = server.race_overpass("q", ["a", "b"], timeout=2, post=bad)
    assert out is None and len(errs) == 2

    def slow(url, q):
        time.sleep(2)
        return _ok()
    out, url, errs = server.race_overpass("q", ["a"], timeout=0.3, post=slow)
    assert out is None and "no valid answer" in errs[-1]


def test_count_query_timeout_25_and_mirrors():
    assert "[timeout:25]" in server.count_query((0, 0, 1, 1))
    assert len(server.OVERPASS) == 4 and any("private.coffee" in u for u in server.OVERPASS)


def test_site_to_geojson_roundtrip():
    site = {"origin": {"lat": 48.5193, "lon": 9.0565, "epsg": 32632, "easting": 504172.175, "northing": 5374022.424},
            "boundary": [[-50, -50], [50, -50], [50, 50], [-50, 50]],
            "buildings": [{"osm_id": "way/1", "footprint": [[0, 0], [10, 0], [10, 10], [0, 10]], "holes": [], "height": 9}],
            "roads": [{"osm_id": "way/2", "class": "residential", "width": 5, "points": [[-50, 0, 1], [50, 0, 1]]}],
            "trees": [{"osm_id": "node/3", "x": 0, "y": 0, "z": 0, "height": 8}],
            "water": [{"osm_id": "way/4", "kind": "river", "polygon": [[0, 0], [5, 0], [5, 5]], "holes": []}]}
    g = server.site_to_geojson(site)
    kinds = [f["properties"]["kind"] for f in g["features"]]
    assert sorted(kinds) == ["boundary", "building", "road", "tree", "water"]
    tree = next(f for f in g["features"] if f["properties"]["kind"] == "tree")
    lon, lat = tree["geometry"]["coordinates"]
    assert abs(lon - 9.0565) < 2e-5 and abs(lat - 48.5193) < 2e-5   # origin (0,0) -> origin lon/lat
    b = next(f for f in g["features"] if f["properties"]["kind"] == "building")
    ring = b["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    return TestClient(server.app)


def test_endpoint_sun_preview_and_disclaimer(client):
    r = client.get("/api/sun_preview", params={"lat": 48.5193, "lon": 9.0565, "date": "2026-06-21", "time": "12:00"})
    d = r.json()
    assert r.status_code == 200 and d["timeZoneInMinutes"] == 60 and d["sumTime"] is True
    assert abs(d["compass_azimuth"] - 138.4) < 0.3
    assert client.get("/api/sun_preview", params={"lat": 1, "lon": 1, "date": "nope"}).status_code == 400
    assert "OpenStreetMap" in client.get("/api/disclaimer").json()["text"]


def test_endpoint_convert_rejects_bad_cache(client):
    r = client.post("/api/convert", json={"cache": "C:/Windows"})
    assert r.status_code in (400, 501)
    r = client.post("/api/convert", json={})
    assert r.status_code in (400, 501)


def test_count_from_osm_bbox_filter():
    osm = {"elements": [
        {"type": "way", "tags": {"building": "yes"}, "bounds": {"minlat": 1, "minlon": 1, "maxlat": 2, "maxlon": 2}},
        {"type": "way", "tags": {"building": "yes"}, "bounds": {"minlat": 5, "minlon": 5, "maxlat": 6, "maxlon": 6}},
        {"type": "node", "tags": {"natural": "tree"}, "lat": 1.5, "lon": 1.5},
        {"type": "node", "tags": {"natural": "tree"}, "lat": 9, "lon": 9},
    ]}
    assert server.count_from_osm(osm, (0, 0, 3, 3)) == {"buildings": 1, "roads": 0, "trees": 1, "water": 0}
    assert server.count_from_osm(osm)["buildings"] == 2


def test_dem_source_and_ground_mapping():
    assert server.dem_source_of(None) == "auto"
    assert server.dem_source_of({"dem_source": "bw_dgm1"}) == "bw_dgm1"
    assert server.dem_source_of({"dem_source": "file", "dem_file": " C:/x.tif "}) == "file:C:/x.tif"
    with pytest.raises(ValueError):
        server.dem_source_of({"dem_source": "file"})
    m = server.map_settings({"dem_source": "bw_dgm1", "dem_file": "", "ground_mode": "auto"})
    assert m == {"ground_mode": None}
    assert server.map_settings({"ground_mode": "dsm"}) == {"ground_mode": "dsm"}


def test_fetch_job_flow(client, monkeypatch):
    import osmctx.fetch as f
    monkeypatch.setattr(f, "fetch", lambda b, cache_dir="cache", dem_source="auto", **kw: "cache/xyz")
    r = client.post("/api/fetch", json={"boundary": {"type": "circle"}, "settings": {"dem_source": "copernicus"}})
    jid = r.json()["job"]
    for _ in range(50):
        j = client.get("/api/job/" + jid).json()
        if j["state"] != "running":
            break
        time.sleep(0.1)
    assert j["state"] == "done" and j["cache"] == "cache/xyz"
    assert client.get("/api/job/nope").status_code == 404
    assert client.post("/api/fetch", json={"boundary": {}, "settings": {"dem_source": "file"}}).status_code == 400


def test_map_settings_drops_builder_keys():
    m = server.map_settings({"tree_mode": "2d", "tree_lib": "x", "tree_height_m": 9})
    assert m == {"tree_height": 9}
