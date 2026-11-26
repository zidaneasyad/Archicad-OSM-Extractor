"""Offline tests: fetch_layered_dem source selection, F.fetch layered branch, _predict_margin, polar utm_epsg."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from osmctx import dem_sources as DS
from osmctx import fetch as F
from osmctx.boundary import bbox, square_boundary, to_polygon
from osmctx.convert import Frame, utm_epsg

AS = 1.0 / 3600.0


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(F.OVT, "try_fetch", lambda *a, **k: {"status": "failed", "error": "offline test stub"})
    monkeypatch.setattr(F.L2, "detect_region", lambda *a, **k: None)   # no LoD2 job (it would query the live LGL server)
    monkeypatch.setattr(F, "overpass", lambda q: ({"elements": []}, "stub"))


def _tif(path, value=1.0, holes=False, n=8, crs="EPSG:4326"):
    a = np.full((n, n), value, np.float32)
    if holes:
        a[2:4, 2:4] = np.nan
    DS.write_float32_tif(path, a, from_origin(0, n * 0.001, 0.001, 0.001), crs)


class Stack:
    """Fake _run_source: records call order; behaviour per id is 'ok', 'ok_holes' or an exception instance."""

    def __init__(self, monkeypatch, behaviour=None):
        self.calls = []
        self.behaviour = behaviour or {}
        monkeypatch.setattr(DS, "_run_source", self)

    def __call__(self, src_id, bb, out_path, cache_dir, bw_fn=None):
        self.calls.append(src_id)
        b = self.behaviour.get(src_id, RuntimeError(f"{src_id} not available"))
        if isinstance(b, Exception):
            raise b
        _tif(out_path, value=float(len(self.calls)), holes=(b == "ok_holes"))
        return {"dem_crs": "EPSG:4326", "marker": src_id}, ["w:" + src_id]


def _bb(lat, lon, d=0.001):
    return (lat, lon, lat + d, lon + d)


NL = _bb(52.37, 4.90)             # nl_ahn envelope only
ID = _bb(-6.20, 106.80)           # id_demnas envelope
NORWAY_ARCTIC = _bb(65.0, 15.0)   # no_kartverket envelope AND > 60 N
GREENLAND = _bb(72.0, -40.0)      # no Tier-1 envelope, > 60 N
ANTARCTIC = _bb(-80.0, 100.0)     # < -60 S
EQUATOR = _bb(0.0, -30.0)         # nothing regional (mid Atlantic)


def run(bb, tmp_path, **kw):
    return DS.fetch_layered_dem(bb, tmp_path / "dem.tif", tmp_path, **kw)


def test_regional_hit_is_used_and_stops(tmp_path, monkeypatch):
    st = Stack(monkeypatch, {"nl_ahn": "ok"})
    res = run(NL, tmp_path)
    assert res["used"] == "nl_ahn" and res["dem_crs"] == "EPSG:4326"
    assert st.calls == ["bw_dgm1", "nl_ahn"]               # bw_dgm1 is always tried first; nothing after the hit
    assert res["extra"] == {"marker": "nl_ahn"} and "w:nl_ahn" in res["warnings"]


def test_bw_dgm1_hit_first(tmp_path, monkeypatch):
    st = Stack(monkeypatch, {"bw_dgm1": "ok", "nl_ahn": "ok"})
    res = run(NL, tmp_path)
    assert res["used"] == "bw_dgm1" and st.calls == ["bw_dgm1"] and res["fallback_reason"] is None


def test_regional_error_then_mapterhorn(tmp_path, monkeypatch):
    st = Stack(monkeypatch, {"nl_ahn": ConnectionError("down"), "mapterhorn": "ok"})
    res = run(NL, tmp_path)
    assert res["used"] == "mapterhorn" and st.calls == ["bw_dgm1", "nl_ahn", "mapterhorn"]
    assert "nl_ahn: ConnectionError: down" in res["fallback_reason"]


def test_regional_miss_envelope_skipped(tmp_path, monkeypatch):
    st = Stack(monkeypatch, {"mapterhorn": "ok"})
    run(EQUATOR, tmp_path)
    assert st.calls == ["bw_dgm1", "mapterhorn"]            # no Tier-1 envelope hit -> straight to mapterhorn
    st.calls.clear()
    run(ID, tmp_path)
    assert st.calls == ["bw_dgm1", "mapterhorn"]            # M9b: id_demnas is flagged unavailable (token) -> skipped


def test_polar_north_after_regional_failure(tmp_path, monkeypatch):
    st = Stack(monkeypatch, {"no_kartverket": RuntimeError("x"), "arcticdem": "ok"})
    res = run(NORWAY_ARCTIC, tmp_path)
    assert res["used"] == "arcticdem" and st.calls == ["bw_dgm1", "no_kartverket", "arcticdem"]


def test_polar_before_mapterhorn_greenland(tmp_path, monkeypatch):
    st = Stack(monkeypatch, {"arcticdem": RuntimeError("x"), "mapterhorn": "ok"})
    res = run(GREENLAND, tmp_path)
    assert res["used"] == "mapterhorn" and st.calls == ["bw_dgm1", "arcticdem", "mapterhorn"]


def test_polar_south_uses_rema_not_arcticdem(tmp_path, monkeypatch):
    st = Stack(monkeypatch, {"rema": "ok"})
    res = run(ANTARCTIC, tmp_path)
    assert res["used"] == "rema" and st.calls == ["bw_dgm1", "rema"]


@pytest.mark.parametrize("lat,polar", [(59.0, False), (59.999, True), (60.5, True), (-59.5, False), (-60.2, True)])
def test_polar_threshold(lat, polar, tmp_path, monkeypatch):
    st = Stack(monkeypatch, {"mapterhorn": "ok"})
    run((lat, 100.0, lat + 0.01, 100.01), tmp_path)
    assert any(c in ("arcticdem", "rema") for c in st.calls) is polar


def test_full_order_when_everything_fails_but_gedtm30(tmp_path, monkeypatch):
    st = Stack(monkeypatch, {"gedtm30": "ok"})
    res = run(NORWAY_ARCTIC, tmp_path)
    assert res["used"] == "gedtm30"
    assert st.calls == ["bw_dgm1", "no_kartverket", "arcticdem", "mapterhorn", "gedtm30"]
    assert res["fallback_reason"].count(";") == 3


def test_all_fail_without_copernicus_raises(tmp_path, monkeypatch):
    Stack(monkeypatch)
    with pytest.raises(RuntimeError, match="all layered DEM tiers failed"):
        run(NL, tmp_path)


def test_all_fail_uses_copernicus_fn(tmp_path, monkeypatch):
    Stack(monkeypatch)
    seen = []

    def cop(bb, path):
        seen.append(bb)
        _tif(path, 5.0)
        return {"warnings": ["cop-w"]}
    res = run(NL, tmp_path, copernicus_fn=cop)
    assert res["used"] == "copernicus" and res["dem_crs"] == "EPSG:4326" and res["warnings"] == ["cop-w"]
    assert "bw_dgm1" in res["fallback_reason"] and "nl_ahn" in res["fallback_reason"] and seen == [NL]
    assert rasterio.open(tmp_path / "dem.tif").read(1)[0, 0] == 5.0


def test_specific_source_no_fallback(tmp_path, monkeypatch):
    st = Stack(monkeypatch, {"mapterhorn": RuntimeError("tile gone"), "gedtm30": "ok"})
    with pytest.raises(RuntimeError, match="tile gone"):
        run(NL, tmp_path, requested="mapterhorn")
    assert st.calls == ["mapterhorn"]
    res = run(NL, tmp_path, requested="gedtm30")
    assert res["used"] == "gedtm30" and res["fallback_reason"] is None


def test_requested_non_stack_id_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown layered DEM source"):
        run(NL, tmp_path, requested="copernicus")


def test_regional_result_is_feathered_with_copernicus(tmp_path, monkeypatch):
    """Primary with nodata holes: interior kept, hole filled from the copernicus fallback, warning added."""
    n = 60
    d = 0.0001

    def fake_run(src_id, bb, out_path, cache_dir, bw_fn=None):
        if src_id != "nl_ahn":
            raise RuntimeError("n/a")
        a = np.full((n, n), 100.0, np.float32)
        a[27:33, 27:33] = np.nan
        DS.write_float32_tif(out_path, a, from_origin(5.0, 52.0, d, d), "EPSG:4326")
        return {"dem_crs": "EPSG:4326"}, []
    monkeypatch.setattr(DS, "_run_source", fake_run)

    def cop(bb, path):
        s, w, nn, e = bb
        DS.write_float32_tif(path, np.full((10, 10), 50.0, np.float32),
                             from_origin(w - 0.001, nn + 0.001, (e - w + 0.002) / 10, (nn - s + 0.002) / 10),
                             "EPSG:4326")
        return {"warnings": []}
    bb = (52.0 - n * d, 5.0, 52.0, 5.0 + n * d)
    res = run(bb, tmp_path, copernicus_fn=cop)
    assert res["used"] == "nl_ahn" and any("Feathered 36" in w for w in res["warnings"])
    with rasterio.open(tmp_path / "dem.tif") as ds:
        back = ds.read(1)
    np.testing.assert_allclose(back[27:33, 27:33], 50.0, atol=1e-3)
    np.testing.assert_allclose(back[:15, :15], 100.0, atol=1e-4)
    assert np.isfinite(back).all() and not (back == -9999.0).any()


def test_complete_primary_not_touched_by_copernicus(tmp_path, monkeypatch):
    Stack(monkeypatch, {"nl_ahn": "ok"})

    def cop(bb, path):
        raise AssertionError("copernicus must not run when the primary has no holes")
    res = run(NL, tmp_path, copernicus_fn=cop)
    assert res["used"] == "nl_ahn"


def test_feather_failure_keeps_primary(tmp_path, monkeypatch):
    Stack(monkeypatch, {"nl_ahn": "ok_holes", "mapterhorn": "ok_holes"})

    def cop(bb, path):
        raise RuntimeError("copernicus down")
    res = run(NL, tmp_path, copernicus_fn=cop)
    assert res["used"] == "nl_ahn"


# --------------------------------------------------------------------------- fetch.py layered branch

def test_predict_margin():
    poly = to_polygon(square_boundary(52.37, 4.9, 100))
    assert F.DEM_MARGIN_M == 30.0 and F.DSM_MARGIN_M == 90.0
    for sid in ("layered", "gedtm30", "fabdem"):
        assert F._predict_margin(sid, poly) == F.DSM_MARGIN_M
    for sid in ("id_demnas", "us_3dep", "nl_ahn", "fr_ign", "uk_ea", "no_kartverket", "arcticdem", "rema", "mapterhorn"):
        assert F._predict_margin(sid, poly) == F.DEM_MARGIN_M
    assert F._predict_margin("bw_dgm1", poly) == F.DEM_MARGIN_M
    assert F._predict_margin("file:/x.tif", poly) == F.DEM_MARGIN_M
    assert F._predict_margin("copernicus", poly) == F.DSM_MARGIN_M
    assert F._predict_margin("auto", to_polygon(square_boundary(10.0, 10.0, 100))) == F.DSM_MARGIN_M


def test_predict_margin_matches_fetch_bbox_for_every_layered_id(tmp_path, monkeypatch):
    """The Overpass bbox predicted up-front must equal the DEM bbox _fetch_fresh ends up with (no re-query)."""
    Stack(monkeypatch, {"layered": "ok"})
    for sid in sorted(DS.LAYERED_DEM_INFO):
        poly = to_polygon(square_boundary(52.37, 4.9, 100))
        pred = bbox(poly, F._predict_margin(sid, poly))
        margin = F.DSM_MARGIN_M if sid in ("layered", "gedtm30", "fabdem") else F.DEM_MARGIN_M
        assert pred == bbox(poly, margin), sid


def _fetch(tmp_path, source, lat=52.37, lon=4.9):
    return F.fetch(square_boundary(lat, lon, 100), tmp_path / "cache", dem_source=source, ndom=False)


def _meta(d):
    return json.loads((d / "meta.json").read_text(encoding="utf-8"))


def test_fetch_layered_regional(tmp_path, monkeypatch):
    st = Stack(monkeypatch, {"nl_ahn": "ok"})
    osm_bboxes = []
    monkeypatch.setattr(F, "build_query", lambda bb: osm_bboxes.append(bb) or "q")
    d = _fetch(tmp_path, "layered")
    m = _meta(d)
    assert m["dem_requested"] == "layered" and m["dem_source_id"] == "nl_ahn" and m["dem_crs"] == "EPSG:4326"
    assert m["dem_source"] == DS.LAYERED_DEM_INFO["nl_ahn"]["name"] and m["dem_licence"] == "CC0 1.0"
    assert m["dem_accuracy"]["vertical_m"] == 0.15 and m["dem_vertical_datum"] == "NAP"
    assert m["dem_source_detail"] == {"marker": "nl_ahn"} and "w:nl_ahn" in m["warnings"]
    poly = to_polygon(square_boundary(52.37, 4.9, 100))
    b90 = bbox(poly, 90.0)
    assert m["dem_bbox"] == dict(zip(("south", "west", "north", "east"), b90))
    assert m["osm_bbox"] == m["dem_bbox"] and osm_bboxes == [b90]       # predicted == real: single Overpass query
    assert (d / "dem.tif").exists() and st.calls[-1] == "nl_ahn"


def test_fetch_layered_fallback_reason_in_meta(tmp_path, monkeypatch):
    Stack(monkeypatch, {"nl_ahn": TimeoutError("slow"), "mapterhorn": "ok"})
    m = _meta(_fetch(tmp_path, "layered"))
    assert m["dem_source_id"] == "mapterhorn" and "nl_ahn: TimeoutError: slow" in m["dem_fallback_reason"]
    assert m["dem_vertical_datum"] == "MSL / EGM2008"


def test_fetch_specific_layered_id_uses_30m_margin(tmp_path, monkeypatch):
    Stack(monkeypatch, {"mapterhorn": "ok"})
    m = _meta(_fetch(tmp_path, "mapterhorn"))
    poly = to_polygon(square_boundary(52.37, 4.9, 100))
    assert m["dem_requested"] == "mapterhorn" and m["dem_source_id"] == "mapterhorn"
    assert m["dem_bbox"]["north"] == pytest.approx(bbox(poly, 30.0)[2])
    assert "dem_fallback_reason" not in m


def test_fetch_layered_final_fallback_copernicus(tmp_path, monkeypatch):
    Stack(monkeypatch)
    lon0, lat_top = 4.8, 52.5
    p = tmp_path / "glo.tif"
    t = from_origin(lon0 - AS / 2, lat_top + AS / 2, AS * 1.5, AS)
    with rasterio.open(p, "w", driver="GTiff", height=1000, width=1000, count=1, dtype="float32", crs="EPSG:4326",
                       transform=t) as ds:
        ds.write(np.full((1000, 1000), 3.0, np.float32), 1)
    monkeypatch.setattr(F, "copernicus_tile_urls", lambda bb: [str(p)])
    m = _meta(_fetch(tmp_path, "layered"))
    assert m["dem_source_id"] == "copernicus" and "nl_ahn" in m["dem_fallback_reason"]


def test_fetch_layered_ids_accepted_and_cached(tmp_path, monkeypatch):
    st = Stack(monkeypatch, {"mapterhorn": "ok", "nl_ahn": "ok"})
    d1 = _fetch(tmp_path, "layered")
    n = len(st.calls)
    d2 = _fetch(tmp_path, "layered")
    assert d1 == d2 and len(st.calls) == n                               # complete cache -> no refetch
    assert _fetch(tmp_path, "mapterhorn") != d1                          # dem_source is part of the cache key


def test_fetch_unknown_still_rejected(tmp_path):
    with pytest.raises(ValueError):
        F.fetch(square_boundary(52.37, 4.9, 100), tmp_path, dem_source="srtm")


# --------------------------------------------------------------------------- polar utm_epsg

@pytest.mark.parametrize("lon,lat,epsg", [
    (0.0, 90.0, 3413), (-45.0, 84.5, 3413), (120.0, 84.0001, 3413),
    (0.0, -90.0, 3031), (100.0, -80.5, 3031), (-60.0, -85.0, 3031),
    (106.8, -6.2, 32748), (5.0, 52.0, 32631), (9.0, 48.5, 32632), (-74.0, 40.7, 32618),
    (180.0, 0.0, 32601), (-180.0, 0.0, 32601), (179.9, 0.0, 32660),
    (5.0, 60.0, 32632),                 # Norway exception (56-64 N, 3-12 E)
    (15.0, 78.0, 32633), (3.0, 78.0, 32631), (25.0, 78.0, 32635), (35.0, 78.0, 32637),   # Svalbard exceptions
])
def test_utm_epsg(lon, lat, epsg):
    assert utm_epsg(lon, lat) == epsg


def test_utm_epsg_polar_boundaries():
    assert utm_epsg(0.0, 84.0) == 32631          # strictly > 84 is polar
    assert utm_epsg(10.0, 83.99) == 32633
    assert utm_epsg(0.0, -80.0) == 32731         # strictly < -80 is polar; UTM south zones are valid to -80
    assert utm_epsg(10.0, -79.99) == 32732


@pytest.mark.parametrize("lat,epsg", [(86.0, 3413), (-86.0, 3031)])
def test_frame_at_pole_roundtrips(lat, epsg):
    poly = to_polygon(square_boundary(lat, 10.0, 200))
    fr = Frame(poly)
    assert fr.epsg == epsg
    x, y = fr.fwd.transform(10.0, lat)
    lon, la = fr.inv.transform(x, y)
    assert lon == pytest.approx(10.0, abs=1e-6) and la == pytest.approx(lat, abs=1e-6)
