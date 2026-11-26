"""Offline tests for osmctx.lod2 (fixtures: tests/fixtures/lod2/, HTTP mocked)."""
from __future__ import annotations

import copy
import io
import json
import math
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests
from pyproj import Transformer

from osmctx import lod2

FIX = Path(__file__).parent / "fixtures" / "lod2"
BW = FIX / "bw_sample.gml"
UTM32 = 32632


def _by_id(recs):
    return {r["id"]: r for r in recs}


@pytest.fixture(scope="module")
def bw():
    return _by_id(lod2.parse_citygml(BW, "bw", out_epsg=UTM32))


# ------------------------------------------------------------------------------------------ BW (CityGML 1.0)
def test_bw_counts_and_ids(bw):
    assert set(bw) == {"DEBW_001000szzox", "DEBW_001000szznD", "DEBW_001000szNaS", "DEBW_001000szAAf"}
    assert all(b["lod"] == 2 and b["source"] == "lod2:lgl-bw" and b["crs_src"] == "EPSG:25832" for b in bw.values())
    assert len(bw["DEBW_001000szAAf"]["parts"]) == 2          # Building with two BuildingParts
    assert [len(bw[k]["parts"]) for k in ("DEBW_001000szzox", "DEBW_001000szznD", "DEBW_001000szNaS")] == [1, 1, 1]
    gable = bw["DEBW_001000szzox"]["parts"][0]
    assert len(gable["roof_planes"]) == 2 and gable["roof_raw"] == 2 and gable["flat_roofs"] == []
    assert bw["DEBW_001000szNaS"]["parts"][0]["roof_raw"] == 4
    assert bw["DEBW_001000szzox"]["credit"] == "Datengrundlage: LGL, www.lgl-bw.de"
    assert bw["DEBW_001000szzox"]["function"] == "31001_1010" or bw["DEBW_001000szzox"]["function"]


def test_json_serialisable_and_plain(bw):
    s = json.dumps(list(bw.values()))
    assert "numpy" not in s and len(s) > 1000


def test_roof_planes_planar_and_pivot_convention(bw):
    for b in bw.values():
        for part in b["parts"]:
            for pl in part["roof_planes"]:
                n = np.array(pl["normal"])
                assert n[2] > 0 and abs(np.linalg.norm(n) - 1) < 1e-4
                assert pl["planar"] and pl["dev"] <= lod2.PLANAR_TOL_M
                assert math.degrees(pl["tilt"]) >= lod2.FLAT_TILT_DEG
                poly = np.array(pl["polygon"])
                z = pl["z0_h"] - (n[0] * (poly[:, 0] - pl["p0"][0]) + n[1] * (poly[:, 1] - pl["p0"][1])) / n[2]
                assert abs(z.min() - pl["z0_h"]) < 2e-3
                assert abs(z.max() - pl["zmax_h"]) < 2e-3
                # existing Roof convention: plane rises toward the LEFT of the pivot direction
                p0, p1 = (np.array(p) for p in pl["pivot"])
                d = (p1 - p0) / np.linalg.norm(p1 - p0)
                u = -n[:2] / np.hypot(n[0], n[1])
                assert abs(d @ u) < 1e-3                                  # pivot is a contour line
                cen = poly.mean(axis=0) - (p0 + p1) / 2
                assert d[0] * cen[1] - d[1] * cen[0] > -1e-6              # polygon lies left of the pivot
                # pivot covers the polygon along d (+0.5 m pad)
                along = (poly - p0) @ d
                assert along.min() >= -1e-3 and along.max() <= np.linalg.norm(p1 - p0) + 1e-3


def test_gable_planes_face_opposite_ways(bw):
    a, b = bw["DEBW_001000szzox"]["parts"][0]["roof_planes"]
    na, nb = np.array(a["normal"][:2]), np.array(b["normal"][:2])
    assert na @ nb < 0                                                  # opposite slopes
    assert 25 < math.degrees(a["tilt"]) < 35 and 25 < math.degrees(b["tilt"]) < 35


def test_heights_are_above_each_buildings_ground(bw):
    g = bw["DEBW_001000szzox"]
    assert g["ground_z_src"] == pytest.approx(342.554, abs=0.01)         # source datum, informational only
    p = g["parts"][0]
    assert p["min_h"] == 0.0 and 0.0 < p["eave_h"] < p["top_h"]
    assert g["ridge_h"] == p["top_h"] == pytest.approx(4.411, abs=0.01)  # == measuredHeight in the file
    assert g["height"] == pytest.approx(4.411, abs=0.01)
    assert g["eave_h"] == p["eave_h"] == pytest.approx(3.173, abs=0.01)
    for b in bw.values():                                               # no absolute z leaks into the heights
        assert 0 <= b["eave_h"] <= b["ridge_h"] < 40
        for part in b["parts"]:
            for pl in part["roof_planes"]:
                assert 0 <= pl["z0_h"] <= pl["zmax_h"] < 40


def test_building_with_parts_own_eaves(bw):
    b = bw["DEBW_001000szAAf"]
    e = [p["eave_h"] for p in b["parts"]]
    assert len(set(e)) == 2                                             # parts have different eaves
    largest = max(b["parts"], key=lambda p: abs(_area(p["footprint"])))
    assert b["eave_h"] == largest["eave_h"]
    assert b["ridge_h"] == max(p["top_h"] for p in b["parts"])
    assert all(p["min_h"] >= 0 for p in b["parts"])


def _area(ring):
    a = np.array(ring)
    return 0.5 * float(np.sum(a[:, 0] * np.roll(a[:, 1], -1) - np.roll(a[:, 0], -1) * a[:, 1]))


def test_flat_roof_is_block_top_not_a_roof_plane(bw):
    p = bw["DEBW_001000szznD"]["parts"][0]
    assert p["roof_planes"] == [] and len(p["flat_roofs"]) == 1
    assert p["flat_top_h"] == p["top_h"] == p["eave_h"] == pytest.approx(3.0, abs=0.01)


def test_footprints_are_ccw_open_rings(bw):
    for b in bw.values():
        ring = b["footprint"]
        assert ring[0] != ring[-1] and _area(ring) > 0
        x0, y0, x1, y1 = b["bbox"]
        assert x0 <= min(p[0] for p in ring) + 1e-3 and y1 >= max(p[1] for p in ring) - 1e-3


def test_reprojection_sanity():
    raw = _by_id(lod2.parse_citygml(BW, "bw"))["DEBW_001000szzox"]
    utm = _by_id(lod2.parse_citygml(BW, "bw", out_epsg=UTM32))["DEBW_001000szzox"]
    assert raw["footprint"][0][0] == pytest.approx(504438.33, abs=1.0)    # untouched EPSG:25832 metres
    assert abs(raw["footprint"][0][0] - utm["footprint"][0][0]) < 1.5       # ETRS89/UTM32 vs WGS84/UTM32
    origin = (504400.0, 5374000.0)
    loc = _by_id(lod2.parse_citygml(BW, "bw", out_epsg=UTM32, origin=origin))["DEBW_001000szzox"]
    x, y = loc["footprint"][0]
    assert abs(x - (utm["footprint"][0][0] - origin[0])) < 1e-2 and 0 < x < 200 and 0 < y < 200
    lon, lat = Transformer.from_crs(f"EPSG:{UTM32}", "EPSG:4326", always_xy=True).transform(x + origin[0], y + origin[1])
    assert 9.04 < lon < 9.08 and 48.50 < lat < 48.54                        # Tuebingen


def test_src_rect_prefilter_skips_buildings():
    none = lod2.parse_citygml(BW, "bw", src_rect=(0, 0, 10, 10))
    assert none == []
    near = lod2.parse_citygml(BW, "bw", src_rect=(504430, 5374040, 504450, 5374060))
    assert [b["id"] for b in near] == ["DEBW_001000szzox"]


# ------------------------------------------------------------------------------------------ coplanar merge
def _slab(x0, x1, y0, y1, slope, z_off=0.0, flip=False):
    """3D ring of a rectangle on a plane rising with +y at `slope` (tan), optional vertical offset."""
    ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    pts = np.array([(x, y, y * slope + z_off) for x, y in ring], float)
    return pts[::-1] if flip else pts


def _recs(slabs, ground=0.0):
    return lod2._roof_records([(s, []) for s in slabs], ground)


def test_merge_coplanar_joins_adjacent_same_plane():
    a, b = _slab(0, 5, 0, 4, 0.5), _slab(5, 10, 0, 4, 0.5)
    recs = _recs([a, b])
    assert len(recs) == 2
    merged = lod2.merge_coplanar(recs)
    assert len(merged) == 1
    m = merged[0]
    assert m["area"] == pytest.approx(recs[0]["area"] + recs[1]["area"], rel=1e-3)
    assert m["tilt"] == pytest.approx(math.atan(0.5), abs=1e-3)
    assert m["z0_h"] == pytest.approx(0.0, abs=1e-3)
    assert Polygon_area(m["polygon"]) == pytest.approx(40.0, abs=0.05)


def Polygon_area(ring):
    return abs(_area(ring))


def test_merge_keeps_different_slopes_and_offsets():
    a = _slab(0, 5, 0, 4, 0.5)
    other_slope = _slab(5, 10, 0, 4, 0.6)                       # ~3 deg steeper: normals differ > 2 deg
    offset = _slab(5, 10, 0, 4, 0.5, z_off=0.3)                 # same slope, 30 cm higher
    opposite = _slab(0, 5, 4, 8, -0.5, z_off=2.0)                # gable's other side
    assert len(lod2.merge_coplanar(_recs([a, other_slope]))) == 2
    assert len(lod2.merge_coplanar(_recs([a, offset]))) == 2
    assert len(lod2.merge_coplanar(_recs([a, opposite]))) == 2
    far = _slab(20, 25, 0, 4, 0.5)                              # coplanar but not adjacent
    assert len(lod2.merge_coplanar(_recs([a, far]))) == 2


def test_merge_chain_and_tolerances():
    s = [_slab(i * 3, i * 3 + 3, 0, 4, 0.5) for i in range(4)]
    assert len(lod2.merge_coplanar(_recs(s))) == 1
    near = _slab(3, 6, 0, 4, 0.5, z_off=0.04)                   # 4 cm: inside the 5 cm tolerance
    assert len(lod2.merge_coplanar(_recs([s[0], near]))) == 1
    assert len(lod2.merge_coplanar(_recs([s[0], near]), offset_tol=0.01)) == 2


def test_merge_reduces_count_on_real_data():
    raw = lod2.parse_citygml(BW, "bw", out_epsg=UTM32, merge=False)
    mer = lod2.parse_citygml(BW, "bw", out_epsg=UTM32, merge=True)
    n = lambda rs: sum(len(p["roof_planes"]) + len(p["flat_roofs"]) for r in rs for p in r["parts"])  # noqa: E731
    assert n(mer) <= n(raw)
    assert all(p["roof_raw"] >= len(p["roof_planes"]) + len(p["flat_roofs"]) for r in mer for p in r["parts"])


# ------------------------------------------------------------------------------------------ 3DBAG (CityJSON)
@pytest.fixture(scope="module")
def page1():
    return json.loads((FIX / "3dbag_page1.json").read_text(encoding="utf-8"))


def test_cityjson_parse(page1):
    recs = _by_id(lod2.parse_cityjson(page1["features"], page1["metadata"], out_epsg=28992, origin=(121000.0, 487000.0)))
    assert set(recs) == {"NL.IMBAG.Pand.0000000000000001", "NL.IMBAG.Pand.0000000000000002"}
    g = recs["NL.IMBAG.Pand.0000000000000001"]
    assert g["source"] == "lod2:3dbag" and g["lod"] == 2 and g["crs_src"] == "EPSG:7415"
    assert g["bbox"] == pytest.approx([0, 0, 10, 8], abs=0.01)           # v * scale + translate, then origin
    assert g["ground_z_src"] == pytest.approx(-0.5, abs=1e-3)           # NAP ground below zero
    p = g["parts"][0]
    # LoD2.2 geometry picked (the part also carries a LoD1.2 box with a 7.5 m flat roof)
    assert p["eave_h"] == pytest.approx(6.5, abs=0.01) and p["top_h"] == pytest.approx(9.5, abs=0.01)
    assert len(p["roof_planes"]) == 2 and p["flat_roofs"] == []
    for pl in p["roof_planes"]:
        assert math.degrees(pl["tilt"]) == pytest.approx(math.degrees(math.atan(3 / 4)), abs=0.05)
        assert pl["area"] == pytest.approx(10 * 5, rel=1e-3) and pl["z0_h"] == pytest.approx(6.5, abs=0.01)
    assert g["storeys"] == 2 and g["year"] == 1927 and g["attrs"]["b3_dak_type"] == "slanted"
    assert g["height"] == g["ridge_h"] == pytest.approx(9.5, abs=0.01)


def test_cityjson_merges_split_flat_roof(page1):
    recs = _by_id(lod2.parse_cityjson(page1["features"], page1["metadata"]))
    flat = recs["NL.IMBAG.Pand.0000000000000002"]["parts"][0]
    assert flat["roof_raw"] == 2 and len(flat["flat_roofs"]) == 1 and flat["roof_planes"] == []
    assert flat["flat_roofs"][0]["area"] == pytest.approx(36.0, abs=0.05)
    assert flat["top_h"] == pytest.approx(3.0, abs=0.01)
    unmerged = lod2.parse_cityjson(page1["features"], page1["metadata"], merge=False)
    assert len(_by_id(unmerged)["NL.IMBAG.Pand.0000000000000002"]["parts"][0]["flat_roofs"]) == 2


# ------------------------------------------------------------------------------------------ PLATEAU (lat lon h, LoD1 fallback)
def test_plateau_axis_order_lod1_fallback():
    recs = _by_id(lod2.parse_citygml(FIX / "plateau_sample.gml", "jp", out_epsg=32654))
    assert set(recs) == {"13104-bldg-7584", "13104-bldg-9999"}            # uro:buildingID, not gml:id
    a, b = recs["13104-bldg-7584"], recs["13104-bldg-9999"]
    # 10 x 8 m box: only right when the posList is read as lat, lon (axis swap) and reprojected
    w, h = a["bbox"][2] - a["bbox"][0], a["bbox"][3] - a["bbox"][1]
    assert w == pytest.approx(10.0, abs=0.05) and h == pytest.approx(8.0, abs=0.05)
    assert a["lod"] == 2 and a["height"] == 9.0 and a["storeys"] == 2 and a["year"] == 1985
    assert a["parts"][0]["eave_h"] == pytest.approx(6.0, abs=0.01) and len(a["parts"][0]["roof_planes"]) == 2
    assert a["ground_z_src"] == pytest.approx(36.2, abs=0.01)
    assert b["lod"] == 1 and b["parts"][0]["roof_planes"] == []           # LoD1-only: block from the solid
    assert b["height"] == 12.5 and b["parts"][0]["top_h"] == pytest.approx(12.5, abs=0.01)
    assert b["ground_z_src"] == pytest.approx(37.2, abs=0.01)
    assert b["bbox"][2] - b["bbox"][0] == pytest.approx(8.0, abs=0.05)
    lon, lat = Transformer.from_crs("EPSG:32654", "EPSG:4326", always_xy=True).transform(*a["footprint"][0])
    assert lon == pytest.approx(139.7005, abs=1e-4) and lat == pytest.approx(35.6895, abs=1e-4)


def test_mesh3_bounds_and_file_selection():
    s, w, n, e = lod2.mesh3_bounds("53394526")
    assert s <= 35.6895 <= n and w <= 139.7005 <= e
    assert (n - s) == pytest.approx(30 / 3600) and (e - w) == pytest.approx(45 / 3600)
    cat = {"cities": [{"cityCode": "13104", "cityName": "x", "files": {"bldg": [
        {"code": "53394526", "url": "u1", "fileSize": 5}, {"code": "53394536", "url": "u2", "fileSize": 7}]}},
        {"cityCode": "13113", "cityName": "y", "files": {"bldg": [{"code": "53394526", "url": "u1", "fileSize": 5}]}}]}
    got = lod2.plateau_files(cat, (35.6895, 139.7005, 35.6900, 139.7010))
    assert [f["code"] for f in got] == ["53394526"]                       # other mesh dropped, same url deduped


# ------------------------------------------------------------------------------------------ Swiss (experimental)
def test_swiss_sample_parses():
    recs = lod2.parse_citygml(FIX / "swiss_sample.gml", "ch", out_epsg=UTM32)
    assert len(recs) == 2 and all(r["source"] == "lod2:swisstopo" and r["crs_src"] == "EPSG:2056" for r in recs)
    for r in recs:
        assert r["parts"][0]["roof_planes"] or r["parts"][0]["flat_roofs"]
        assert 0 <= r["eave_h"] <= r["ridge_h"] < 60
        lon, lat = Transformer.from_crs(f"EPSG:{UTM32}", "EPSG:4326", always_xy=True).transform(*r["footprint"][0])
        assert 7.3 < lon < 7.6 and 46.8 < lat < 47.1                      # Bern


def test_swiss_items_pick_only_citygml_sheets():
    stac = {"features": [
        {"id": "a_2018", "properties": {"datetime": "2018-01-01T00:00:00Z"}, "assets": {"a.dwg.zip": {"href": "h1"}}},
        {"id": "a_2021", "properties": {"datetime": "2021-01-01T00:00:00Z"},
         "assets": {"a_2056_5728.citygml.zip": {"href": "h2"}, "a.gdb.zip": {"href": "h3"}}},
        {"id": "a_2023", "properties": {"datetime": "2023-11-29T00:00:00Z"}, "assets": {"x.gdb.zip": {"href": "h4"}}}]}
    assert lod2.swiss_items(stac) == [{"id": "a_2021", "href": "h2", "datetime": "2021-01-01T00:00:00Z"}]
    assert lod2.REGIONS["ch"]["status"] == "experimental" and "2021" in lod2.REGIONS["ch"]["note"]


# ------------------------------------------------------------------------------------------ region detection
TUE = (48.5194, 9.0592, 48.5200, 9.0600)


def test_detect_region():
    ok, no = (lambda u: 200), (lambda u: 404)
    assert lod2.detect_region(TUE, head=ok) == "bw"
    assert lod2.detect_region(TUE, head=no) is None                        # Bavaria / RP / Hessen style: no tile
    assert lod2.detect_region((47.376, 8.540, 47.377, 8.541), head=no) == "ch"    # Zurich
    assert lod2.detect_region((52.372, 4.893, 52.373, 4.894), head=no) == "nl"    # Amsterdam
    assert lod2.detect_region((35.689, 139.700, 35.690, 139.701), head=no) == "jp"  # Tokyo
    assert lod2.detect_region((48.857, 2.352, 48.858, 2.353), head=no) is None      # Paris
    assert lod2.detect_region((-6.2, 106.8, -6.19, 106.81), head=no) is None        # Jakarta

    def boom(u):
        raise requests.ConnectionError("offline")
    assert lod2.detect_region(TUE, head=boom) is None                       # HEAD failure never raises


def test_lgl_tiles_rule():
    assert lod2.lgl_tiles(TUE) == [(503, 5374)]                            # recon: 504.43 / 5374.10 -> 503_5374
    t = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
    w, s = t.transform(504900, 5373900)
    e, n = t.transform(505100, 5374100)                                    # crosses E = 505 km and N = 5374 km
    assert lod2.lgl_tiles((s, w, n, e)) == [(503, 5372), (503, 5374), (505, 5372), (505, 5374)]


# ------------------------------------------------------------------------------------------ HTTP mocks
class Resp:
    def __init__(self, status=200, payload=None, content=b"", headers=None):
        self.status_code, self._payload, self.content = status, payload, content
        self.headers = headers or {}

    def json(self):
        return copy.deepcopy(self._payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)

    def iter_content(self, n):
        for i in range(0, len(self.content), n):
            yield self.content[i:i + n]

    def close(self):
        pass


def _rd_bb(x0=120990.0, y0=486990.0, x1=121040.0, y1=487020.0):
    t = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    w, s = t.transform(x0, y0)
    e, n = t.transform(x1, y1)
    return (s, w, n, e)


def _page2(page1):
    p = copy.deepcopy(page1)
    p["features"] = [p["features"][0]]
    old = "NL.IMBAG.Pand.0000000000000001"
    p["features"][0] = json.loads(json.dumps(p["features"][0]).replace(old, "NL.IMBAG.Pand.0000000000000003"))
    p["links"] = [lk for lk in p["links"] if lk["rel"] != "next"]
    return p


def test_3dbag_pagination_and_cache(monkeypatch, tmp_path, page1):
    p2 = _page2(page1)
    urls = []

    def fake(url, **kw):
        urls.append(url)
        return Resp(200, p2 if "offset=2" in url else page1)

    monkeypatch.setattr(lod2, "_get", fake)
    recs, info = lod2.fetch_3dbag(_rd_bb(), tmp_path, out_epsg=28992, origin=(121000.0, 487000.0))
    assert len(urls) == 2 and "offset=2" in urls[1] and info["pages"] == 2
    assert "bbox=" in urls[0] and f"limit={lod2.PAGE_LIMIT}" in urls[0]
    assert [r["id"] for r in recs] == ["NL.IMBAG.Pand.0000000000000001", "NL.IMBAG.Pand.0000000000000002",
                                       "NL.IMBAG.Pand.0000000000000003"]
    # second run: served from cache/_lod2/3dbag, no HTTP
    urls.clear()
    recs2, info2 = lod2.fetch_3dbag(_rd_bb(), tmp_path, out_epsg=28992, origin=(121000.0, 487000.0))
    assert urls == [] and info2["pages_cached"] == 2 and len(recs2) == 3
    assert any((tmp_path / "_lod2" / "3dbag").glob("*.json"))


def test_3dbag_page_cap_warns(monkeypatch, tmp_path, page1):
    monkeypatch.setattr(lod2, "_get", lambda url, **kw: Resp(200, page1))       # endless `next`
    recs, info = lod2.fetch_3dbag(_rd_bb(), tmp_path, max_pages=3)
    assert info["pages"] == 3 and info["warnings"] and "page cap" in info["warnings"][0]


def _bw_cache(root: Path):
    d = root / "_lod2" / "bw"
    d.mkdir(parents=True)
    with zipfile.ZipFile(d / "LoD2_32_503_5374_2_bw.zip", "w") as zf:
        zf.writestr("LoD2_32_504_5374_1_BW.gml", BW.read_bytes())
        zf.writestr("LoD2_32_503_5374_1_BW.gml", b"<not-xml-never-read/>")      # other cell: must not be parsed
        zf.writestr("INFO.txt", "x")


def test_fetch_lod2_bw_cache_reuse_and_frame(monkeypatch, tmp_path):
    _bw_cache(tmp_path)
    monkeypatch.setattr(lod2, "_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used")))
    t = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
    w, s = t.transform(504400, 5374000)
    e, n = t.transform(504700, 5374200)
    out = tmp_path / "lod2.json"
    frame = SimpleNamespace(epsg=UTM32, E0=504500.0, N0=5374100.0)
    meta = lod2.fetch_lod2((s, w, n, e), tmp_path, out, region="bw", frame=frame)
    assert meta["status"] == "ok" and meta["region"] == "bw" and meta["count"] == 4
    assert meta["roof_planes"] == sum(len(p["roof_planes"]) for r in lod2.load(out)["buildings"] for p in r["parts"])
    assert meta["tiles"] == ["LoD2_32_503_5374_2_bw.zip"] and meta["warnings"] == []
    assert "LGL" in meta["credit"] and "DL-DE BY 2.0" in meta["attribution"] and meta["region_status"] == "working"
    d = lod2.load(out)
    assert d["crs"] == f"EPSG:{UTM32}" and d["origin"] == [504500.0, 5374100.0] and d["version"] == 1
    xs = [p[0] for b in d["buildings"] for p in b["footprint"]]
    assert -200 < min(xs) and max(xs) < 200                                   # site frame, not absolute UTM
    # bbox filter: a site elsewhere in the same tile keeps nothing but still succeeds with a warning
    w2, s2 = t.transform(503100, 5374600)
    e2, n2 = t.transform(503200, 5374700)
    m2 = lod2.fetch_lod2((s2, w2, n2, e2), tmp_path, out, region="bw")
    assert m2["count"] == 0 and any("no LoD2 buildings" in x for x in m2["warnings"])


def test_download_atomic_and_cached(monkeypatch, tmp_path):
    dest = tmp_path / "sub" / "f.bin"
    calls = []

    def ok(url, **kw):
        calls.append(url)
        return Resp(200, content=b"x" * (3 << 20), headers={"Content-Length": str(3 << 20)})

    monkeypatch.setattr(lod2, "_get", ok)
    assert lod2._download("u", dest, None, "t") == 3 << 20 and dest.exists()
    assert lod2._download("u", dest, None, "t") == 3 << 20 and len(calls) == 1      # cached

    dest2 = tmp_path / "g.bin"
    with pytest.raises(TimeoutError):
        lod2._download("u", dest2, deadline=0.0, label="t")                         # time box hit mid-stream
    assert not dest2.exists() and not [p for p in tmp_path.iterdir() if p.name.startswith(".g.bin")]


def test_http_404_tile_is_skipped_then_error(monkeypatch, tmp_path):
    monkeypatch.setattr(lod2, "_get", lambda url, **kw: Resp(404))
    with pytest.raises(lod2.LoD2Error):
        lod2.fetch_lgl(TUE, tmp_path)


# ------------------------------------------------------------------------------------------ never blocks OSM
def test_try_fetch_timebox_is_failure_not_exception(monkeypatch, tmp_path, page1):
    out = tmp_path / FILE_NAME
    meta = lod2.try_fetch(_rd_bb(), tmp_path, out, timeout=0.0, region="nl")
    assert meta["status"] == "failed" and "TimeoutError" in meta["error"]
    assert not out.exists()

    def slow(url, **kw):
        raise requests.ConnectTimeout("slow")

    monkeypatch.setattr(lod2, "_get", slow)
    meta2 = lod2.try_fetch(_rd_bb(), tmp_path, out, region="nl")
    assert meta2["status"] == "failed" and "ConnectTimeout" in meta2["error"]
    m = {"warnings": ["Overture x", "LoD2 old"]}
    lod2.apply(m, meta2)
    assert m["lod2"] is meta2 and "Overture x" in m["warnings"] and "LoD2 old" not in m["warnings"]
    assert any(w.startswith("LoD2 buildings not fetched") and "OSM" in w for w in m["warnings"])


FILE_NAME = lod2.FILE


def test_unsupported_location_is_a_warning(monkeypatch, tmp_path):
    monkeypatch.setattr(lod2, "detect_region", lambda bb: None)
    meta = lod2.try_fetch((48.85, 2.35, 48.86, 2.36), tmp_path, tmp_path / "x.json")
    assert meta["status"] == "failed" and "no LoD2 source" in meta["error"]


def test_large_download_needs_opt_in(monkeypatch, tmp_path):
    cat = {"cities": [{"cityCode": "13104", "cityName": "x", "files": {"bldg": [
        {"code": "53394526", "url": "https://example.invalid/a.gml", "fileSize": 94_000_000}]}}]}
    monkeypatch.setattr(lod2, "_get", lambda url, **kw: Resp(200, cat))
    bb = (35.6895, 139.7005, 35.6900, 139.7010)
    meta = lod2.try_fetch(bb, tmp_path, tmp_path / "x.json", region="jp")
    assert meta["status"] == "failed" and "LargeDownload" in meta["error"] and "94 MB" in meta["error"]
    # opt-in downloads, parses (fixture served as the "file") and finds the two buildings
    gml = (FIX / "plateau_sample.gml").read_bytes()
    monkeypatch.setattr(lod2, "_get", lambda url, **kw: Resp(200, cat) if "datacatalog" in url else
                        Resp(200, content=gml, headers={"Content-Length": str(len(gml))}))
    meta2 = lod2.try_fetch((35.6895, 139.7005, 35.6897, 139.7008), tmp_path, tmp_path / "x.json", region="jp",
                           allow_large=True, frame=SimpleNamespace(epsg=32654, E0=382418.0, N0=3950288.0))
    assert meta2["status"] == "ok" and meta2["count"] >= 1 and meta2["lod1_only"] <= 1
    # raw file cached: a second call needs no opt-in and no download
    monkeypatch.setattr(lod2, "_get", lambda url, **kw: Resp(200, cat) if "datacatalog" in url else
                        (_ for _ in ()).throw(AssertionError("re-download")))
    meta3 = lod2.try_fetch((35.6895, 139.7005, 35.6897, 139.7008), tmp_path, tmp_path / "x.json", region="jp")
    assert meta3["status"] == "ok"


def test_ensure_updates_meta_json(monkeypatch, tmp_path):
    d = tmp_path / "abc"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"bbox": {"south": 48.85, "west": 2.35, "north": 48.86, "east": 2.36},
                                             "warnings": []}), encoding="utf-8")
    monkeypatch.setattr(lod2, "detect_region", lambda bb: None)
    l2 = lod2.ensure(d, tmp_path)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert l2["status"] == "failed" and meta["lod2"]["status"] == "failed"
    assert any(w.startswith("LoD2 buildings not fetched") for w in meta["warnings"])
    assert lod2.needs_fetch(d, meta) is True
    (d / lod2.FILE).write_text("{}", encoding="utf-8")
    assert lod2.needs_fetch(d, {"lod2": {"status": "ok"}}) is False
    assert lod2.needs_fetch(d, {"lod2": {"status": "ok"}}, force=True) is True


# ------------------------------------------------------------------------------------------ credit / estimate / table
def test_credit_and_attribution():
    assert lod2.credit_for("bw") == "Datengrundlage: LGL, www.lgl-bw.de (DL-DE BY 2.0)"
    assert "3DBAG" in lod2.credit_for("nl") and "CC BY 4.0" in lod2.credit_for("nl")
    assert "swisstopo" in lod2.credit_for("ch") and "PLATEAU" in lod2.credit_for("jp")
    blk = lod2.attribution_block("bw")
    assert blk["credit_confirmed"] is True and lod2.attribution_block("nl")["credit_confirmed"] is False
    assert set(lod2.REGIONS) == {"bw", "nl", "jp", "ch"}
    assert {r: v["status"] for r, v in lod2.REGIONS.items()} == {"bw": "working", "nl": "working", "jp": "working",
                                                                  "ch": "experimental"}


def test_estimate(bw):
    ex = lod2.estimate(TUE, records=list(bw.values()))
    assert ex["basis"] == "records" and ex["buildings"] == 4
    assert ex["roof_planes"] == sum(len(p["roof_planes"]) for b in bw.values() for p in b["parts"])
    pr = lod2.estimate((48.5190, 9.0580, 48.5280, 9.0715), region="bw")      # ~1 km2
    assert pr["basis"] == "prior" and 500 < pr["buildings"] < 1500 and pr["roof_planes"] > pr["buildings"]
    assert lod2.estimate(TUE, region="xx")["buildings"] == 0
