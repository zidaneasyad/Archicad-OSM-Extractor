"""Offline tests (mocked HTTP) for the M9b regional DEM sources and the repaired M9a endpoints."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from osmctx import dem_sources as DS

NEW_IDS = ["vt_vcgi", "il_isgs", "ky_kyfromabove", "cz_cuzk", "hk_cedd", "au_qld", "tz_msimbazi", "ee_maaamet",
           "nz_linz", "ht_cnigs", "jp_gsj", "mx_inegi"]
UNAVAILABLE = {"id_demnas", "fabdem", "nz_linz", "tz_msimbazi"}


class Resp:
    def __init__(self, content=b"", status=200, ctype="image/tiff"):
        self.content = content
        self.status_code = status
        self.headers = {"content-type": ctype}
        self.text = content[:200].decode("latin1") if isinstance(content, bytes) else ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def tif_bytes(arr, nodata=None, epsg=3857):
    arr = np.asarray(arr, dtype=np.float32)
    kw = {"nodata": nodata} if nodata is not None else {}
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32",
                      crs=f"EPSG:{epsg}", transform=from_origin(0, arr.shape[0], 1, 1), **kw) as ds:
            ds.write(arr, 1)
        return mem.read()


def read_tif(path):
    with rasterio.open(path) as ds:
        return ds.read(1), ds.nodata, ds.crs


def rgba_png(h, base, interval, alpha=None):
    """Encode heights (m) as a Terrain-RGB RGBA PNG: v = round((h - base) / interval)."""
    v = np.round((np.asarray(h, dtype=np.float64) - base) / interval).astype(np.int64)
    rgba = np.zeros(v.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = (v >> 16) & 255
    rgba[..., 1] = (v >> 8) & 255
    rgba[..., 2] = v & 255
    rgba[..., 3] = 255 if alpha is None else alpha
    return DS.encode_rgba_png(rgba)


# --------------------------------------------------------------------------- registry

@pytest.mark.parametrize("sid", NEW_IDS)
def test_new_registry_entry_complete(sid):
    inf = DS.LAYERED_DEM_INFO[sid]
    for k in ("name", "attribution", "licence", "vertical_datum"):
        assert isinstance(inf[k], str) and inf[k]
    acc = inf["accuracy"]
    assert {"horizontal_m", "vertical_m", "kind", "note"} <= set(acc)
    assert acc["kind"] in ("DTM", "DSM") and acc["horizontal_m"] > 0 and acc["vertical_m"] > 0
    assert sid in DS.ENVELOPES and sid in DS.TIER1_ORDER
    s, w, n, e = DS.ENVELOPES[sid]
    assert -90 <= s < n <= 90 and -180 <= w < e <= 180


def test_unavailable_flags_carry_a_reason():
    for sid in DS.LAYERED_DEM_INFO:
        inf = DS.LAYERED_DEM_INFO[sid]
        if sid in UNAVAILABLE:
            assert inf["available"] is False and len(inf["why"]) > 10
        else:
            assert inf.get("available", True) is True


def test_repaired_endpoints():
    assert "latest" in DS.ENDPOINTS["arcticdem"][0] and "latest" in DS.ENDPOINTS["rema"][0]
    assert "/wms-r/wms" in DS.ENDPOINTS["fr_ign"][0]                       # IGN WCS route is gone
    assert DS.ENDPOINTS["uk_ea"][1] != "1" and "Elevation_DTM_1m" in DS.ENDPOINTS["uk_ea"][1]
    assert "gedtm" in DS.ENDPOINTS["gedtm30"][0] and "/v1.2/" in DS.ENDPOINTS["gedtm30"][0]
    for sid in ("jp_gsj", "mx_inegi"):
        tmpl = DS.ENDPOINTS[sid][0]
        assert all(k in tmpl for k in ("{z}", "{x}", "{y}")) and tmpl.index("{y}") < tmpl.index("{x}")
    assert "no key" not in str(DS.ENDPOINTS).lower() and "api=" not in str(DS.ENDPOINTS).lower()


# --------------------------------------------------------------------------- envelope selection

class Recorder:
    def __init__(self, monkeypatch, ok=()):
        self.calls, self.ok = [], set(ok)
        monkeypatch.setattr(DS, "_run_source", self)

    def __call__(self, src_id, bb, out_path, cache_dir, bw_fn=None):
        self.calls.append(src_id)
        if src_id not in self.ok:
            raise RuntimeError("down")
        DS.write_float32_tif(out_path, np.ones((4, 4), np.float32), from_origin(0, 4, 1, 1), "EPSG:4326")
        return {"dem_crs": "EPSG:4326"}, []


def _bb(lat, lon, d=0.001):
    return (lat, lon, lat + d, lon + d)


REGIONS = [
    ("vt_vcgi", 44.4759, -73.2121), ("il_isgs", 39.7817, -89.6501), ("ky_kyfromabove", 38.2009, -84.8733),
    ("cz_cuzk", 50.0755, 14.4378), ("hk_cedd", 22.30, 114.17), ("au_qld", -27.4698, 153.0251),
    ("ee_maaamet", 59.4370, 24.7536), ("ht_cnigs", 18.54, -72.335), ("jp_gsj", 35.6895, 139.6917),
    ("mx_inegi", 19.4326, -99.1332),
]


@pytest.mark.parametrize("sid,lat,lon", REGIONS)
def test_envelope_selection_picks_regional_source(sid, lat, lon, tmp_path, monkeypatch):
    rec = Recorder(monkeypatch, ok={sid})
    res = DS.fetch_layered_dem(_bb(lat, lon), tmp_path / "d.tif", tmp_path)
    assert res["used"] == sid and rec.calls[-1] == sid


@pytest.mark.parametrize("sid,lat,lon", REGIONS)
def test_other_regions_not_tried_elsewhere(sid, lat, lon, tmp_path, monkeypatch):
    rec = Recorder(monkeypatch, ok={"mapterhorn"})
    DS.fetch_layered_dem(_bb(52.37, 4.9), tmp_path / "d.tif", tmp_path)       # Amsterdam: only nl_ahn
    assert sid not in rec.calls


def test_state_sources_before_us_3dep_and_mexico_after(tmp_path, monkeypatch):
    rec = Recorder(monkeypatch, ok={"mapterhorn"})
    DS.fetch_layered_dem(_bb(39.78, -89.65), tmp_path / "d.tif", tmp_path)          # Springfield IL
    assert rec.calls.index("il_isgs") < rec.calls.index("us_3dep")
    rec.calls.clear()
    DS.fetch_layered_dem(_bb(31.76, -106.48), tmp_path / "e.tif", tmp_path)         # El Paso / Juarez border
    assert rec.calls.index("us_3dep") < rec.calls.index("mx_inegi")


def test_unavailable_sources_skipped_with_warning(tmp_path, monkeypatch):
    rec = Recorder(monkeypatch, ok={"mapterhorn"})
    res = DS.fetch_layered_dem(_bb(-6.2, 106.8), tmp_path / "d.tif", tmp_path)       # Jakarta: id_demnas (token)
    assert res["used"] == "mapterhorn" and "id_demnas" not in rec.calls
    assert any("id_demnas skipped (unavailable)" in w and "Token" in w for w in res["warnings"])
    rec.calls.clear()
    res = DS.fetch_layered_dem(_bb(-36.85, 174.76), tmp_path / "n.tif", tmp_path)    # Auckland: nz_linz (key)
    assert "nz_linz" not in rec.calls and any("nz_linz skipped" in w for w in res["warnings"])


@pytest.mark.parametrize("sid", sorted(UNAVAILABLE))
def test_explicit_unavailable_request_raises_clearly(sid, tmp_path, monkeypatch):
    Recorder(monkeypatch, ok={sid})
    with pytest.raises(RuntimeError, match=f"{sid} unavailable"):
        DS.fetch_layered_dem(_bb(0, 0), tmp_path / "d.tif", tmp_path, requested=sid)


@pytest.mark.parametrize("sid", ["nz_linz", "tz_msimbazi"])
def test_dispatcher_refuses_unavailable_new_ids(sid, tmp_path):
    with pytest.raises(RuntimeError, match="unavailable"):
        DS._run_source(sid, (1, 2, 1.1, 2.1), tmp_path / "o.tif", tmp_path)


# --------------------------------------------------------------------------- unit conversion / nodata (ArcGIS)

def _arcgis_get(monkeypatch, arr, nodata=None, store=None):
    def get(url, params=None, headers=None, timeout=None):
        if store is not None:
            store.append((url, params))
        return Resp(tif_bytes(arr, nodata))
    monkeypatch.setattr(DS.requests, "get", get)


def test_illinois_feet_to_metres(tmp_path, monkeypatch):
    feet = np.array([[100.0, 600.0], [-3.4e38, 1000.0]], np.float32)       # one ArcGIS-style nodata cell
    store = []
    _arcgis_get(monkeypatch, feet, store=store)
    extra, _ = DS._run_source("il_isgs", (39.78, -89.65, 39.781, -89.649), tmp_path / "il.tif", tmp_path)
    back, nd, _ = read_tif(tmp_path / "il.tif")
    assert back[0, 0] == pytest.approx(30.48, abs=1e-3) and back[0, 1] == pytest.approx(182.88, abs=1e-3)
    assert back[1, 1] == pytest.approx(304.8, abs=1e-3)
    assert back[1, 0] == -9999.0 and nd == -9999.0                           # nodata is not scaled into data
    assert store[0][0].startswith("https://data.isgs.illinois.edu/") and store[0][0].endswith("/exportImage")


def test_ky_and_other_metre_sources_are_not_scaled(tmp_path, monkeypatch):
    _arcgis_get(monkeypatch, np.array([[152.0, 153.5]], np.float32))
    for sid in ("ky_kyfromabove", "vt_vcgi", "cz_cuzk", "hk_cedd", "au_qld"):
        DS._run_source(sid, (38.2, -84.87, 38.201, -84.869), tmp_path / f"{sid}.tif", tmp_path)
        back, _, _ = read_tif(tmp_path / f"{sid}.tif")
        assert back[0, 0] == pytest.approx(152.0) and back[0, 1] == pytest.approx(153.5), sid


def test_arcgis_nodata_minus9999_masked(tmp_path, monkeypatch):
    _arcgis_get(monkeypatch, np.array([[240.0, -9999.0]], np.float32), nodata=-9999.0)
    DS._run_source("cz_cuzk", (50.07, 14.43, 50.071, 14.431), tmp_path / "c.tif", tmp_path)
    back, nd, _ = read_tif(tmp_path / "c.tif")
    assert back[0, 0] == pytest.approx(240.0) and back[0, 1] == nd == -9999.0


# --------------------------------------------------------------------------- Terrain-RGB

def test_decode_terrain_rgb_mapbox_known_values():
    h = np.array([[0.0, 21.2], [2240.0, -50.0]])
    rgb = np.zeros((2, 2, 3))
    for (i, j), v in np.ndenumerate(np.round((h + 10000.0) / 0.1)):
        rgb[i, j] = [(int(v) >> 16) & 255, (int(v) >> 8) & 255, int(v) & 255]
    out = DS.decode_terrain_rgb(rgb)
    assert out.dtype == np.float32 and np.allclose(out, h, atol=0.051)
    # a documented Mapbox sample: R=1,G=134,B=160 -> -10000 + 100000*... = 6553.6+34304+160 = ...
    assert DS.decode_terrain_rgb(np.array([[[1, 134, 160]]]))[0, 0] == pytest.approx(
        -10000.0 + (65536 + 134 * 256 + 160) * 0.1, abs=1e-3)


def test_decode_terrain_rgb_inegi_base_shift_known_points():
    # values measured from the live INEGI tiles: Cancun (0,39,116) -> 10.0 m, Monterrey (0,60,58) -> ~541.8 m
    out = DS.decode_terrain_rgb(np.array([[[0, 39, 116], [0, 60, 58], [0, 126, 35]]]), base_m=-1000.0)
    assert out[0, 0] == pytest.approx(10.0, abs=1e-3)
    assert out[0, 1] == pytest.approx(541.8, abs=1e-3)
    assert out[0, 2] == pytest.approx(2229.1, abs=1e-3)


def test_decode_terrain_rgb_channel_first_layout():
    rgb = np.zeros((3, 2, 2))
    rgb[1] = 126
    rgb[2] = 35
    out = DS.decode_terrain_rgb(rgb, base_m=-1000.0)
    assert out.shape == (2, 2) and out[0, 0] == pytest.approx(2229.1, abs=1e-3)


class TileSession:
    """Stands in for requests.Session; handler(url) -> Resp."""
    handler = None
    urls: list = []

    def __init__(self):
        self.headers = {}

    def get(self, url, timeout=None):
        TileSession.urls.append(url)
        return TileSession.handler(url)

    def close(self):
        pass


def _tiles(monkeypatch, handler):
    TileSession.handler = staticmethod(handler)
    TileSession.urls = []
    monkeypatch.setattr(DS.requests, "Session", TileSession)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    return TileSession


BB_TOKYO = (35.6895, 139.6917, 35.6900, 139.6922)


def test_jp_gsj_tiles_decode_and_url_order(tmp_path, monkeypatch):
    heights = np.full((8, 8), 30.0)
    heights[0, 0] = 45.1
    heights[7, 7] = -9999.0 + 0.0                         # GSJ no-data style value (decodes below -900 m)
    sess = _tiles(monkeypatch, lambda url: Resp(rgba_png(np.where(heights < -900, -9999.9, heights), -10000.0, 0.1),
                                                ctype="image/png"))
    extra, _ = DS._run_source("jp_gsj", BB_TOKYO, tmp_path / "j.tif", tmp_path)
    back, nd, crs = read_tif(tmp_path / "j.tif")
    assert crs.to_epsg() == 3857 and extra["dem_crs"] == "EPSG:3857" and extra["zoom"] == 14
    assert back[0, 0] == pytest.approx(45.1, abs=0.06) and back[3, 3] == pytest.approx(30.0, abs=0.06)
    assert back[7, 7] == -9999.0 and nd == -9999.0         # below -900 m -> nodata
    z, y, x = sess.urls[0].split("/")[-3:]
    assert "seamless/elev/terrainRGB" in sess.urls[0] and int(z) == 14           # {z}/{y}/{x} order in the template
    tx, ty = DS._latlon_to_tile(BB_TOKYO[2], BB_TOKYO[1], 14)
    assert sess.urls[0].endswith(f"/14/{ty}/{tx}.png")


def test_mx_inegi_alpha_masks_nodata_and_base_shift(tmp_path, monkeypatch):
    h = np.full((8, 8), 2240.0)
    alpha = np.full((8, 8), 255, np.uint8)
    alpha[:, 4:] = 0                                      # transparent half (outside Mexico / ocean)
    _tiles(monkeypatch, lambda url: Resp(rgba_png(h, -1000.0, 0.1, alpha), ctype="image/png"))
    DS._run_source("mx_inegi", (19.4326, -99.1332, 19.4330, -99.1328), tmp_path / "m.tif", tmp_path)
    back, nd, _ = read_tif(tmp_path / "m.tif")
    assert back[2, 1] == pytest.approx(2240.0, abs=0.06)
    assert back[2, 6] == -9999.0 and nd == -9999.0
    assert (tmp_path / "_tiles" / "mx_inegi").exists()


def test_terrain_rgb_404_everywhere_raises_and_5xx_retries(tmp_path, monkeypatch):
    _tiles(monkeypatch, lambda url: Resp(b"", status=404))
    with pytest.raises(RuntimeError, match="no Terrain-RGB tile"):
        DS._run_source("jp_gsj", BB_TOKYO, tmp_path / "a.tif", tmp_path)
    n = {"c": 0}

    def flaky(url):
        n["c"] += 1
        return Resp(b"", status=504) if n["c"] == 1 else Resp(rgba_png(np.full((8, 8), 10.0), -10000.0, 0.1),
                                                                 ctype="image/png")
    _tiles(monkeypatch, flaky)
    DS._run_source("jp_gsj", BB_TOKYO, tmp_path / "b.tif", tmp_path / "c2")
    back, _, _ = read_tif(tmp_path / "b.tif")
    assert n["c"] >= 2 and back[0, 0] == pytest.approx(10.0, abs=0.06)       # one 504 retried, then good


def test_terrain_rgb_tile_cache_reused(tmp_path, monkeypatch):
    sess = _tiles(monkeypatch, lambda url: Resp(rgba_png(np.full((8, 8), 5.0), -10000.0, 0.1), ctype="image/png"))
    DS._run_source("jp_gsj", BB_TOKYO, tmp_path / "a.tif", tmp_path)
    n = len(sess.urls)
    DS._run_source("jp_gsj", BB_TOKYO, tmp_path / "b.tif", tmp_path)
    assert len(sess.urls) == n


# --------------------------------------------------------------------------- WMS (IGN) + WCS (EE, UK, NL)

def test_fr_ign_wms_geotiff_nodata_and_crs(tmp_path, monkeypatch):
    store = []

    def get(url, params=None, headers=None, timeout=None):
        store.append((url, params))
        return Resp(tif_bytes(np.array([[35.0, -99999.0], [36.5, 34.0]], np.float32), nodata=-99999.0, epsg=2154),
                    ctype="image/geotiff")
    monkeypatch.setattr(DS.requests, "get", get)
    extra, _ = DS._run_source("fr_ign", (48.8565, 2.3521, 48.8575, 2.3535), tmp_path / "f.tif", tmp_path)
    back, nd, crs = read_tif(tmp_path / "f.tif")
    assert crs.to_epsg() == 2154 and extra["dem_crs"] == "EPSG:2154"
    assert back[0, 0] == pytest.approx(35.0) and back[0, 1] == nd == -9999.0
    url, p = store[0]
    assert url == "https://data.geopf.fr/wms-r/wms" and p["LAYERS"] == "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES"
    assert p["FORMAT"] == "image/geotiff" and p["CRS"] == "EPSG:2154" and p["VERSION"] == "1.3.0"


def test_wms_xml_error_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(DS.requests, "get", lambda *a, **k: Resp(b"<ServiceException>x</ServiceException>",
                                                                  ctype="text/xml"))
    with pytest.raises(RuntimeError, match="XML/HTML error"):
        DS._run_source("fr_ign", (48.85, 2.35, 48.86, 2.36), tmp_path / "f.tif", tmp_path)


def test_ee_wcs_uses_dtm1_and_3301(tmp_path, monkeypatch):
    store = []

    def get(url, params=None, headers=None, timeout=None):
        store.append((url, dict(params) if isinstance(params, dict) else list(params)))
        return Resp(tif_bytes(np.array([[2.6, 8.1]], np.float32), epsg=3301))
    monkeypatch.setattr(DS.requests, "get", get)
    extra, _ = DS._run_source("ee_maaamet", (59.4369, 24.7535, 59.4372, 24.7540), tmp_path / "e.tif", tmp_path)
    assert extra["dem_crs"] == "EPSG:3301" and store[0][0] == "https://teenus.maaamet.ee/ows/wcs-dtm"
    assert ("COVERAGEID", "dtm-1") in store[0][1] and ("VERSION", "2.0.1") in store[0][1]


def test_uk_wcs_long_coverage_id(tmp_path, monkeypatch):
    store = []

    def get(url, params=None, headers=None, timeout=None):
        store.append(list(params))
        return Resp(tif_bytes(np.array([[5.0]], np.float32), epsg=27700))
    monkeypatch.setattr(DS.requests, "get", get)
    DS._run_source("uk_ea", (51.5, -0.13, 51.501, -0.129), tmp_path / "u.tif", tmp_path)
    assert ("COVERAGEID", DS.ENDPOINTS["uk_ea"][1]) in store[0]


def test_ht_cnigs_dispatches_to_vrt(tmp_path, monkeypatch):
    calls = []

    def cog(urls, bb, out_path, **kw):
        calls.append(urls)
        DS.write_float32_tif(out_path, np.full((2, 2), 35.0, np.float32), from_origin(0, 2, 1, 1), "EPSG:32618")
        return {"crs": "EPSG:32618"}
    monkeypatch.setattr(DS, "fetch_cog_vrt", cog)
    extra, _ = DS._run_source("ht_cnigs", (18.54, -72.34, 18.541, -72.339), tmp_path / "h.tif", tmp_path)
    assert calls == [DS.ENDPOINTS["ht_cnigs"]] and extra["dem_crs"] == "EPSG:32618"


# --------------------------------------------------------------------------- mapterhorn sparse zoom

def test_mapterhorn_steps_down_zoom_when_fine_tiles_missing(tmp_path, monkeypatch):
    def handler(url):
        z = int(url.split("/")[-3])
        if "mapterhorn" in url and z > 12:
            return Resp(b"Tile not found", status=404)          # Jakarta-style: nothing finer than z12
        px = np.full((8, 8), 12.0)
        r = np.floor((px + 32768.0) / 256.0)
        rgba = np.zeros((8, 8, 4), np.uint8)
        rgba[..., 0] = r
        rgba[..., 1] = np.floor(px + 32768.0 - r * 256.0)
        rgba[..., 3] = 255
        return Resp(DS.encode_rgba_png(rgba), ctype="image/png")
    _tiles(monkeypatch, handler)
    info = DS.fetch_mapterhorn((-6.2090, 106.8445, -6.2080, 106.8460), tmp_path / "m.tif", tmp_path)
    assert info["zoom"] == 12 and info["sources"] == ["mapterhorn"]
    back, _, _ = read_tif(tmp_path / "m.tif")
    assert back[0, 0] == pytest.approx(12.0, abs=0.01)
