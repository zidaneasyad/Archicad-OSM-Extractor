"""Offline tests for osmctx.convert (synthetic caches: osm.json + dem.tif + meta.json)."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import rasterio
from pyproj import CRS, Geod, Proj, Transformer
from rasterio.transform import from_origin
from shapely.geometry import Point, Polygon

from osmctx.boundary import bbox, square_boundary, to_polygon
from osmctx.convert import (Dem, Frame, building_height, convert, densify, parse_length, road_width,
                            utm_epsg, DEFAULT_SETTINGS)

G = Geod(ellps="WGS84")
LAT0, LON0 = 48.5, 9.0  # on the UTM 32 central meridian
S = json.loads(json.dumps(DEFAULT_SETTINGS))


# --------------------------------------------------------------------------- helpers

class Local:
    """Independent (AEQD) local metric frame used to author synthetic geometry."""

    def __init__(self, lat=LAT0, lon=LON0):
        crs = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"
        self.fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self.inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def ll(self, x, y):
        lon, lat = self.inv.transform(x, y)
        return float(lon), float(lat)

    def xy(self, lon, lat):
        return self.fwd.transform(lon, lat)


L = Local()
_nid = [1000]


def geom(pts):
    return [{"lat": L.ll(x, y)[1], "lon": L.ll(x, y)[0]} for x, y in pts]


def way(wid, pts, tags, nodes=None):
    if nodes is None:
        nodes = []
        for _ in pts:
            _nid[0] += 1
            nodes.append(_nid[0])
    return {"type": "way", "id": wid, "nodes": nodes, "geometry": geom(pts), "tags": tags}


def node(nid, x, y, tags):
    lon, lat = L.ll(x, y)
    return {"type": "node", "id": nid, "lat": lat, "lon": lon, "tags": tags}


def make_cache(tmp, elements, dem_fn, boundary=None, size=100.0, res=1 / 3600.0, margin=60.0):
    boundary = boundary or square_boundary(LAT0, LON0, size)
    poly = to_polygon(boundary)
    s, w, n, e = bbox(poly, margin)
    ncols = int(math.ceil((e - w) / res))
    nrows = int(math.ceil((n - s) / res))
    lon = w + (np.arange(ncols) + 0.5) * res
    lat = n - (np.arange(nrows) + 0.5) * res
    LON, LAT = np.meshgrid(lon, lat)
    arr = np.asarray(dem_fn(LON, LAT), dtype="float32") * np.ones_like(LON, dtype="float32")
    d = Path(tmp)
    d.mkdir(parents=True, exist_ok=True)
    with rasterio.open(d / "dem.tif", "w", driver="GTiff", height=nrows, width=ncols, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(w, n, res, res)) as ds:
        ds.write(arr, 1)
    (d / "osm.json").write_text(json.dumps({"elements": elements}), encoding="utf-8")
    (d / "meta.json").write_text(json.dumps({"boundary": boundary, "warnings": []}), encoding="utf-8")
    return d


def flat(v=200.0):
    return lambda lon, lat: np.full_like(lon, v)


def plane(lon, lat):
    return 300.0 + 2000.0 * (lon - LON0) + 5000.0 * (lat - LAT0)


def demz(tmp_path, x, y):
    """Absolute DEM height (bilinear, as the converter samples it) at local AEQD x/y."""
    return float(Dem(tmp_path / "cache" / "dem.tif").sample(*L.ll(x, y))[0])


def run(tmp_path, elements, dem_fn=flat(), settings=None, **kw):
    d = make_cache(tmp_path / "cache", elements, dem_fn, **kw)
    return convert(d, settings, tmp_path / "out")


# --------------------------------------------------------------------------- coordinates

def test_utm_epsg():
    assert utm_epsg(9.0565, 48.5193) == 32632
    assert utm_epsg(-43.2, -22.9) == 32723
    assert utm_epsg(115.085, -8.829) == 32750
    assert utm_epsg(5.0, 60.0) == 32632      # Norway exception
    assert utm_epsg(15.0, 78.0) == 32633     # Svalbard
    assert utm_epsg(179.99, 10) == 32660
    assert utm_epsg(-180.0, 10) == 32601


@pytest.mark.parametrize("lat,lon", [(48.5193, 9.0565), (48.5, 11.9), (-8.829, 115.085)])
def test_projection_distance_and_north(lat, lon):
    f = Frame(to_polygon(square_boundary(lat, lon, 100)))
    assert f.epsg == utm_epsg(lon, lat)
    assert f.lat0 == pytest.approx(lat, abs=1e-6) and f.lon0 == pytest.approx(lon, abs=1e-6)
    x0, y0 = f.to_local(lon, lat)
    assert abs(x0) < 0.01 and abs(y0) < 0.01
    # distance between two projected points matches the geodesic within 0.1 %
    lon2, lat2, _ = G.fwd(lon, lat, 37.0, 800.0)
    x2, y2 = f.to_local(lon2, lat2)
    assert abs(math.hypot(x2 - x0, y2 - y0) - 800.0) / 800.0 < 1e-3
    # due north: y > 0, x ~ 0 (exactly -y*tan(convergence) away from the UTM central meridian)
    lonn, latn, _ = G.fwd(lon, lat, 0.0, 50.0)
    xn, yn = f.to_local(lonn, latn)
    assert yn > 49.9
    conv = Proj(CRS.from_epsg(f.epsg)).get_factors(lon, lat).meridian_convergence
    assert abs(xn) < 50 * math.tan(math.radians(abs(conv))) + 0.01
    assert abs(abs(math.degrees(math.atan2(xn, yn))) - abs(conv)) < 0.01
    # due east: x > 0
    lone, late, _ = G.fwd(lon, lat, 90.0, 50.0)
    xe, ye = f.to_local(lone, late)
    assert xe > 49.9 and abs(ye) < 2.0
    # round trip
    lo, la = f.to_lonlat(xe, ye)
    assert float(lo) == pytest.approx(lone, abs=1e-9) and float(la) == pytest.approx(late, abs=1e-9)


def test_dem_bilinear_pixel_centre_convention(tmp_path):
    d = make_cache(tmp_path, [], plane)
    dem = Dem(d / "dem.tif")
    rng = np.random.default_rng(1)
    lon = LON0 + rng.uniform(-5e-4, 5e-4, 200)
    lat = LAT0 + rng.uniform(-4e-4, 4e-4, 200)
    # bilinear reproduces a plane exactly only if values sit at pixel centres (half-pixel shift = 0.3-0.7 m)
    assert np.abs(dem.sample(lon, lat) - plane(lon, lat)).max() < 1e-3


def test_origin_terrain(tmp_path):
    site = run(tmp_path, [], plane)
    o = site["origin"]
    assert o["epsg"] == 32632
    assert o["lat"] == pytest.approx(LAT0, abs=1e-6) and o["lon"] == pytest.approx(LON0, abs=1e-6)
    pts = np.asarray(site["terrain"]["points"])
    # plane rises N and E -> minimum at SW corner of the boundary
    sw = to_polygon(square_boundary(LAT0, LON0, 100)).exterior.coords[0]
    assert o["elevation"] == pytest.approx(plane(*sw), abs=0.02)
    assert pts[:, 2].min() == pytest.approx(0.0, abs=1e-3)
    assert site["report"]["info"]["terrain"]["spacing_m"] == 5.0
    assert 400 <= len(pts) <= 2500
    assert np.abs(pts[:, :2]).max() <= 50.1
    # terrain z equals plane - elevation at the point's lon/lat
    f = Frame(to_polygon(square_boundary(LAT0, LON0, 100)))
    lon, lat = f.to_lonlat(pts[:, 0], pts[:, 1])
    assert np.abs(plane(lon, lat) - o["elevation"] - pts[:, 2]).max() < 2e-3
    # boundary ring is first in terrain.points and CCW
    nb = len(site["terrain"]["boundary"])
    assert np.allclose(pts[:nb, :2], np.asarray(site["terrain"]["boundary"]), atol=1e-3)
    assert Polygon(site["boundary"]).exterior.is_ccw


def test_auto_spacing_large(tmp_path):
    site = run(tmp_path, [], flat(), size=1500.0, res=3 / 3600.0)
    n = len(site["terrain"]["points"])
    assert site["report"]["info"]["terrain"]["spacing_m"] > 5.0
    assert 1500 < n <= 2500


# --------------------------------------------------------------------------- parsing

@pytest.mark.parametrize("s,v", [("12", 12), ("12 m", 12), ("12.5m", 12.5), ("12,5", 12.5), ("40'", 12.192),
                                 ("40 ft", 12.192), ("40'6\"", 12.3444), ("3 km", 3000), ("~7", 7), ("10;12", 10),
                                 (" 8 metres", 8), ("350 cm", 3.5)])
def test_parse_length(s, v):
    assert parse_length(s, max_value=1e5) == pytest.approx(v, abs=1e-3)


@pytest.mark.parametrize("s", [None, "", "yes", "unknown", "-3", "0", "12 storeys", "5000"])
def test_parse_length_garbage(s):
    assert parse_length(s) is None


def test_building_height_precedence():
    assert building_height({"height": "15", "building:levels": "4"}, S) == (15.0, "height")
    assert building_height({"height": "40'"}, S)[0] == pytest.approx(12.192)
    assert building_height({"building:levels": "4"}, S) == (pytest.approx(12.8), "levels")
    assert building_height({"building:levels": "4", "roof:levels": "2"}, S) == (pytest.approx(16.0), "levels")
    assert building_height({"height": "bogus", "building:levels": "2"}, S) == (pytest.approx(6.4), "levels")
    assert building_height({}, S) == (pytest.approx(9.6), "default")
    assert building_height({}, {**S, "default_levels": 2.0, "floor_height": 3.0}) == (6.0, "default")


def test_road_width():
    assert road_width({"highway": "residential", "width": "6.5 m"}, S)[:2] == (6.5, "tag")
    assert road_width({"highway": "residential", "width": "20'"}, S)[0] == pytest.approx(6.096)
    assert road_width({"highway": "primary"}, S) == (8.0, "default", "class")
    assert road_width({"highway": "primary_link"}, S) == (8.0, "default", "class")
    assert road_width({"highway": "weird"}, S) == (4.0, "default", "default")
    assert road_width({"highway": "service", "width": "x"}, S)[0] == 3.5


def test_densify():
    p, idx = densify(np.array([[0, 0], [25, 0], [25, 3]]), 10)
    assert list(idx) == [0, 3, 4] and len(p) == 5 and np.diff(p[:, 0]).max() <= 10


# --------------------------------------------------------------------------- buildings

def test_building_rings_and_multipolygon(tmp_path):
    cw_with_junk = [(0, 0), (0, 10), (0, 10), (10, 10), (10, 5), (10, 0), (0, 0)]  # CW, dup, collinear
    mp = {"type": "relation", "id": 7, "tags": {"type": "multipolygon", "building": "yes", "height": "12 m"},
          "members": [
              {"type": "way", "ref": 1, "role": "outer", "geometry": geom([(-40, -40), (-20, -40), (-20, -20)])},
              {"type": "way", "ref": 2, "role": "outer", "geometry": geom([(-20, -20), (-40, -20), (-40, -40)])},
              {"type": "way", "ref": 3, "role": "inner", "geometry": geom([(-35, -35), (-35, -25), (-25, -25),
                                                                          (-25, -35), (-35, -35)])}]}
    sliver = way(3, [(30, 30), (31, 30), (31, 31.5), (30, 31.5), (30, 30)], {"building": "shed"})
    part = way(4, [(20, 20), (25, 20), (25, 25), (20, 20)], {"building:part": "yes"})
    site = run(tmp_path, [way(1, cw_with_junk, {"building": "house", "building:levels": "2"}), mp, sliver, part],
               settings={"include": {"building_parts": False}})   # parts handled in tests/test_features.py
    b = {x["osm_id"]: x for x in site["buildings"]}
    assert set(b) == {"way/1", "relation/7"}
    fp = b["way/1"]["footprint"]
    assert len(fp) == 4 and fp[0] != fp[-1] and Polygon(fp).exterior.is_ccw
    assert len({tuple(p) for p in fp}) == 4
    assert b["way/1"]["height_source"] == "levels" and b["way/1"]["height"] == pytest.approx(6.4)
    r = b["relation/7"]
    assert len(r["holes"]) == 1 and not Polygon(r["holes"][0]).exterior.is_ccw
    assert Polygon(r["footprint"], r["holes"]).area == pytest.approx(300.0, rel=0.01)
    assert r["height"] == 12.0 and r["height_source"] == "height"
    c = site["report"]["counts"]
    assert c["building_slivers_dropped"] == 1 and c["building_parts_skipped"] == 1
    assert site["report"]["defaults_used"]["heights"] == {"levels": 1, "height": 1}


def test_building_base_z_is_min_terrain(tmp_path):
    site = run(tmp_path, [way(1, [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)], {"building": "yes"})], plane)
    b = site["buildings"][0]
    # plane rises to N/E: min at the building's SW corner
    lon, lat = L.ll(0, 0)
    assert b["base_z"] == pytest.approx(plane(lon, lat) - site["origin"]["elevation"], abs=0.05)
    assert b["height_source"] == "default" and b["height"] == pytest.approx(9.6)


# --------------------------------------------------------------------------- clipping

def test_clipping(tmp_path):
    els = [
        way(1, [(-80, 10), (0, 10), (30, 10)], {"highway": "residential"}),               # crosses W edge
        way(2, [(40, -5), (60, -5), (60, 5), (40, 5), (40, -5)], {"building": "yes"}),     # centroid inside (x=50 edge)
        way(3, [(45, 20), (70, 20), (70, 30), (45, 30), (45, 20)], {"building": "yes"}),   # centroid outside
        node(10, 10, 10, {"natural": "tree"}), node(11, 70, 0, {"natural": "tree"}),
        way(4, [(-200, -60), (200, -60), (200, -20), (-200, -20), (-200, -60)], {"natural": "water"}),
        way(5, [(0, -10), (0, -200)], {"highway": "footway", "area": "yes"}),
    ]
    site = run(tmp_path, els, plane, margin=200.0)
    r = [x for x in site["roads"] if x["osm_id"] == "way/1"]
    assert len(r) == 1
    p = np.asarray(r[0]["points"])
    assert np.abs(p[:, :2]).max() <= 50.01 and p[0, 0] == pytest.approx(-50.0, abs=0.05)
    assert r[0]["width"] == 5.0 and r[0]["width_source"] == "default"
    assert "way/5" not in {x["osm_id"] for x in site["roads"]}
    b = {x["osm_id"]: x for x in site["buildings"]}
    assert set(b) == {"way/2"} and max(x for x, _ in b["way/2"]["footprint"]) > 59  # kept whole
    assert [t["osm_id"] for t in site["trees"]] == ["node/10"]
    w = site["water"][0]
    wp = Polygon(w["polygon"])
    assert wp.area == pytest.approx(100 * 30, rel=0.02) and wp.bounds[3] <= -19.9
    assert w["kind"] == "water"
    # water z = 5th percentile of terrain inside (plane: near the low/south part of the water polygon)
    zs = np.asarray([pt[2] for pt in site["terrain"]["points"] if wp.covers(Point(pt[:2]))])
    assert np.percentile(zs, 0) - 0.5 <= w["z"] <= np.percentile(zs, 30)


def test_circle_and_corridor_boundaries(tmp_path):
    road = way(1, [(-100, 0), (100, 0)], {"highway": "service"})
    circ = {"type": "circle", "center": [LON0, LAT0], "radius_m": 40}
    s1 = run(tmp_path / "c", [road], flat(), boundary=circ)
    p = np.asarray(s1["roads"][0]["points"])
    assert np.hypot(p[:, 0], p[:, 1]).max() <= 40.05
    assert len(s1["boundary"]) == 64
    corr = {"type": "polyline", "coordinates": [list(L.ll(-60, -30)), list(L.ll(60, 30))], "buffer_m": 10}
    s2 = run(tmp_path / "p", [road], flat(), boundary=corr)
    poly = Polygon(s2["boundary"]).buffer(0.05)
    assert all(poly.covers(Point(x, y)) for r in s2["roads"] for x, y, _ in r["points"])
    assert len(s2["terrain"]["points"]) > 50


# --------------------------------------------------------------------------- bridges

def valley(lon, lat):
    x, _ = L.xy(lon, lat)
    return 100.0 - 30.0 * np.exp(-(np.asarray(x) / 25.0) ** 2)


def test_bridge_never_draped(tmp_path):
    # approach A -> bridge B1 -> bridge B2 (drawn reversed) -> approach C, all sharing nodes
    els = [way(1, [(-90, 0), (-45, 0)], {"highway": "secondary"}, nodes=[1, 2]),
           way(2, [(-45, 0), (-20, 0), (0, 0)], {"highway": "secondary", "bridge": "yes"}, nodes=[2, 3, 4]),
           way(3, [(45, 0), (20, 0), (0, 0)], {"highway": "secondary", "bridge": "viaduct"}, nodes=[5, 6, 4]),
           way(4, [(45, 0), (90, 0)], {"highway": "secondary"}, nodes=[5, 7])]
    site = run(tmp_path, els, valley, size=200, res=0.3 / 3600)
    e0 = site["origin"]["elevation"]
    za = demz(tmp_path, -45, 0) - e0
    zc = demz(tmp_path, 45, 0) - e0
    bridge_pts = [p for r in site["roads"] if r["bridge"] for p in r["points"]]
    assert len(bridge_pts) >= 10
    for x, y, z in bridge_pts:
        expect = za + (zc - za) * (x + 45) / 90
        assert z == pytest.approx(expect, abs=0.05)
    mid = min(bridge_pts, key=lambda p: abs(p[0]))
    assert mid[2] - (valley(*L.ll(mid[0], 0)) - e0) > 25  # 30 m valley under the deck
    # approaches are draped and meet the bridge ends
    for r in site["roads"]:
        if not r["bridge"]:
            ends = [r["points"][0], r["points"][-1]]
            inner = min(ends, key=lambda p: abs(p[0]))
            assert inner[2] == pytest.approx(za if inner[0] < 0 else zc, abs=0.05)
    assert site["report"]["counts"]["bridge_runs"] == 1


def test_bridge_layer_clearance(tmp_path):
    els = [way(1, [(-90, 0), (-50, 0)], {"highway": "primary"}, nodes=[1, 2]),
           way(2, [(-50, 0), (50, 0)], {"highway": "primary", "bridge": "yes", "layer": "1"}, nodes=[2, 3]),
           way(3, [(50, 0), (90, 0)], {"highway": "primary"}, nodes=[3, 4]),
           way(4, [(0, -90), (0, 90)], {"highway": "primary"})]
    site = run(tmp_path, els, flat(50.0), size=200)
    br = [r for r in site["roads"] if r["bridge"]][0]
    p = np.asarray(br["points"])
    assert br["layer"] == 1
    # M4-A: deck >= lower road + 5.5 m at the crossing (flat over the crossed road), abutments stay on the ground;
    # 50 m abutment-to-crossing cannot reach 5.5 m at the 8 % cap -> steeper ramp, reported
    assert p[np.abs(p[:, 0]).argmin(), 2] == pytest.approx(5.5, abs=1e-3)
    assert p[0, 2] == pytest.approx(0.0, abs=1e-3) and p[-1, 2] == pytest.approx(0.0, abs=1e-3)
    assert (p[:, 2] >= -1e-6).all() and (np.abs(np.diff(p[:, 2])) <= 5.5 / (50 - 9) * 10 + 2e-3).all()
    lay = site["report"]["info"]["layers"]
    assert lay["inverted"] == 0 and lay["below_clearance"] == 0 and lay["ramp_cap_exceeded_runs"] == 1
    assert any("ramp" in w for w in site["report"]["warnings"])


def test_bridge_clipped_keeps_full_span_z(tmp_path):
    # bridge abutments are outside a 60 m site: z must still come from the real abutments
    els = [way(1, [(-90, 0), (90, 0)], {"highway": "secondary", "bridge": "yes"}, nodes=[1, 2]),
           way(2, [(-120, 0), (-90, 0)], {"highway": "secondary"}, nodes=[3, 1]),
           way(3, [(90, 0), (120, 0)], {"highway": "secondary"}, nodes=[2, 4])]
    site = run(tmp_path, els, valley, size=60, res=0.3 / 3600, margin=100)
    e0 = site["origin"]["elevation"]
    za = demz(tmp_path, -90, 0) - e0
    br = [r for r in site["roads"] if r["bridge"]][0]
    assert all(z == pytest.approx(za, abs=0.05) for _, _, z in br["points"])
    assert za > 25


def test_tunnel_flag_and_warning(tmp_path):
    els = [way(1, [(-40, 0), (40, 0)], {"highway": "tertiary", "tunnel": "yes", "layer": "-1"}),
           way(2, [(-40, 20), (40, 20)], {"highway": "tertiary"})]
    site = run(tmp_path, els, valley)
    t = [r for r in site["roads"] if r["tunnel"]][0]
    assert t["layer"] == -1 and not t["bridge"]
    assert any("tunnel" in w for w in site["report"]["warnings"])
    e0 = site["origin"]["elevation"]
    za, zb = demz(tmp_path, -40, 0) - e0, demz(tmp_path, 40, 0) - e0
    for x, _, z in t["points"]:  # linear between portals, not draped into the valley
        assert z == pytest.approx(za + (zb - za) * (x + 40) / 80, abs=0.05)


def test_road_smoothing_pinned_at_junctions(tmp_path):
    a = way(1, [(-40, 0), (0, 0), (40, 0)], {"highway": "residential"}, nodes=[1, 2, 3])
    b = way(2, [(0, 0), (0, 40)], {"highway": "residential"}, nodes=[2, 4])
    site = run(tmp_path, [a, b], plane)
    ra = [r for r in site["roads"] if r["osm_id"] == "way/1"][0]
    rb = [r for r in site["roads"] if r["osm_id"] == "way/2"][0]
    za = [p[2] for p in ra["points"] if abs(p[0]) < 1e-3 and abs(p[1]) < 1e-3][0]
    assert za == pytest.approx(rb["points"][0][2], abs=1e-3)


# --------------------------------------------------------------------------- trees / water

def test_trees(tmp_path):
    els = [way(1, [(-20, 0), (20, 0)], {"natural": "tree_row"}),
           node(5, -20, 0.5, {"natural": "tree", "height": "15 m"}),
           {"type": "way", "id": 9, "bounds": {"minlat": LAT0 - 1e-3, "minlon": LON0 - 1e-3, "maxlat": LAT0 + 1e-3,
                                                "maxlon": LON0 + 1e-3}, "tags": {"natural": "wood"},
            "geometry": geom([(-10, -10), (10, -10), (10, 10), (-10, -10)]), "nodes": [1, 2, 3, 1]}]
    site = run(tmp_path, els, flat())
    t = site["trees"]
    assert len(t) == 6  # row: 0,8,...,40 m (6 pts) minus the one duplicating node 5, plus node 5
    assert site["report"]["counts"]["trees_from_rows"] == 5
    n5 = [x for x in t if x["osm_id"] == "node/5"][0]
    assert n5["height"] == 15.0 and n5["z"] == pytest.approx(0.0, abs=1e-3)
    assert all(x["height"] == 8.0 for x in t if x["osm_id"] == "way/1")
    assert site["report"]["counts"]["wood_forest_areas_not_scattered"] == 1


def test_waterways(tmp_path):
    els = [way(1, [(-100, -30), (100, -30)], {"waterway": "stream"}),
           way(2, [(-100, 30), (100, 30)], {"waterway": "river"}),
           way(3, [(-200, 10), (200, 10), (200, 50), (-200, 50), (-200, 10)], {"natural": "water", "water": "river"}),
           way(4, [(-100, 0), (100, 0)], {"waterway": "ditch", "tunnel": "culvert"})]
    site = run(tmp_path, els, flat(), margin=200)
    kinds = sorted((w["osm_id"], w["kind"]) for w in site["water"])
    assert kinds == [("way/1", "river"), ("way/3", "river")]
    st = Polygon([w for w in site["water"] if w["osm_id"] == "way/1"][0]["polygon"])
    assert st.area == pytest.approx(100 * 3, rel=0.02)
    c = site["report"]["counts"]
    assert c["waterways_covered_by_area_skipped"] == 1 and c["waterway_culverts_skipped"] == 1


# --------------------------------------------------------------------------- schema

def test_schema_and_files(tmp_path):
    els = [way(1, [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)], {"building": "yes"}),
           way(2, [(-30, -30), (30, -30)], {"highway": "path"}),
           node(3, 5, -20, {"natural": "tree"}),
           way(4, [(-60, 20), (60, 20), (60, 40), (-60, 40), (-60, 20)], {"natural": "water"})]
    site = run(tmp_path, els, plane)
    out = tmp_path / "out"
    assert json.loads((out / "site.json").read_text(encoding="utf-8")) == site
    svg = (out / "preview.svg").read_text(encoding="utf-8")
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>") and "10 m" in svg
    assert set(site) >= {"version", "origin", "boundary", "terrain", "buildings", "roads", "trees", "water", "report"}
    assert site["version"] == 1
    assert set(site["origin"]) == {"lat", "lon", "epsg", "easting", "northing", "elevation"}
    assert set(site["terrain"]) == {"points", "boundary"}
    assert set(site["buildings"][0]) >= {"osm_id", "footprint", "holes", "base_z", "height", "height_source", "tags"}
    assert set(site["roads"][0]) >= {"osm_id", "class", "width", "width_source", "bridge", "tunnel", "layer", "points"}
    assert set(site["trees"][0]) >= {"osm_id", "x", "y", "z", "height"}
    assert set(site["water"][0]) >= {"osm_id", "kind", "polygon", "z"}
    assert set(site["report"]) >= {"counts", "defaults_used", "warnings"}
    assert any("DSM" in w for w in site["report"]["warnings"])
    assert all(len(p) == 3 for p in site["terrain"]["points"]) and all(len(p) == 2 for p in site["boundary"])


def test_include_flags_and_settings(tmp_path):
    els = [way(1, [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)], {"building": "yes"}),
           node(3, 5, -20, {"natural": "tree"})]
    site = run(tmp_path, els, flat(), settings={"include": {"trees": False}, "default_levels": 5,
                                                 "floor_height": 3.0, "grid_spacing": 10})
    assert site["trees"] == [] and site["buildings"][0]["height"] == 15.0
    assert site["report"]["info"]["terrain"]["spacing_m"] == 10.0
