"""Offline tests for osmctx/dem_sources.py (M9a layered terrain stack). No network, no Archicad."""
import math
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import rasterio
import requests
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from osmctx import dem_sources as DS
from osmctx import fetch as F


# --------------------------------------------------------------------------- helpers

def tif_bytes(arr, transform, crs, nodata=None):
    arr = np.asarray(arr, dtype=np.float32)
    kw = {"nodata": nodata} if nodata is not None else {}
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32",
                     crs=crs, transform=transform, **kw) as ds:
            ds.write(arr, 1)
        return mf.read()


def write_tif(path, arr, transform, crs="EPSG:4326", nodata=None):
    arr = np.asarray(arr, dtype=np.float32)
    kw = {"nodata": nodata} if nodata is not None else {}
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs=crs, transform=transform, **kw) as ds:
        ds.write(arr, 1)


def read_tif(path):
    with rasterio.open(path) as ds:
        return ds.read(1), ds.nodata, ds.transform, ds.crs


class Resp:
    def __init__(self, content=b"", ctype="image/tiff", status=200):
        self.content = content
        self.headers = {"content-type": ctype}
        self.status_code = status

    @property
    def text(self):
        return self.content.decode("latin1")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def terrarium_rgb(h):
    """Height (m, multiples of 1/256) -> (R, G, B) uint8 as stored in Terrarium tiles."""
    v = np.asarray(h, dtype=np.float64) + 32768.0
    r = np.floor(v / 256.0)
    g = np.floor(v) % 256.0
    b = np.floor((v - np.floor(v)) * 256.0)
    return r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)


def terrarium_png(h):
    r, g, b = terrarium_rgb(h)
    rgba = np.dstack((r, g, b, np.full(r.shape, 255, np.uint8)))
    return DS.encode_rgba_png(rgba)


# --------------------------------------------------------------------------- atomic_path / parallel_fetch

def test_atomic_path_success_replaces(tmp_path):
    p = tmp_path / "sub" / "out.bin"
    with DS.atomic_path(p) as tmp:
        assert tmp.parent == p.parent and tmp != p
        tmp.write_bytes(b"data")
        assert not p.exists()          # not visible before exit
    assert p.read_bytes() == b"data"
    assert [x.name for x in p.parent.iterdir()] == ["out.bin"]   # temp file gone


def test_atomic_path_exception_leaves_nothing(tmp_path):
    p = tmp_path / "out.bin"
    with pytest.raises(RuntimeError):
        with DS.atomic_path(p) as tmp:
            tmp.write_bytes(b"partial")
            raise RuntimeError("boom")
    assert not p.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_path_exception_keeps_existing_target(tmp_path):
    p = tmp_path / "out.bin"
    p.write_bytes(b"old")
    with pytest.raises(ValueError):
        with DS.atomic_path(p) as tmp:
            tmp.write_bytes(b"new-partial")
            raise ValueError("x")
    assert p.read_bytes() == b"old"
    assert [x.name for x in tmp_path.iterdir()] == ["out.bin"]


def test_parallel_fetch_order_empty_and_single():
    assert list(DS.parallel_fetch(lambda x: x, [])) == []
    assert list(DS.parallel_fetch(lambda x: x * 2, range(50), workers=4)) == [x * 2 for x in range(50)]
    assert list(DS.parallel_fetch(lambda x: x + 1, [1, 2, 3], workers=1)) == [2, 3, 4]
    assert list(DS.parallel_fetch(lambda x: x + 1, [7], workers=8)) == [8]
    assert list(DS.parallel_fetch(lambda x: x, [1, 2], workers=0)) == [1, 2]   # workers clamped to >= 1


def test_parallel_fetch_propagates_exception():
    def fn(x):
        if x == 3:
            raise KeyError("bad")
        return x
    with pytest.raises(KeyError):
        list(DS.parallel_fetch(fn, range(10), workers=2))


def test_parallel_fetch_bounds_queue():
    started = []
    lock = threading.Lock()

    def fn(x):
        with lock:
            started.append(x)
        return x

    workers = 2
    gen = DS.parallel_fetch(fn, range(200), workers=workers)
    assert next(gen) == 0
    # at most 2*workers + 1 futures submitted before the first result is handed out
    assert len(started) <= 2 * workers + 1
    gen.close()


# --------------------------------------------------------------------------- Terrarium

@pytest.mark.parametrize("rgb,expected", [
    ((128, 0, 0), 0.0),
    ((128, 1, 0), 1.0),
    ((128, 10, 128), 10.5),
    ((127, 255, 255), -1.0 / 256.0),
    ((127, 255, 0), -1.0),
    ((0, 0, 0), -32768.0),
    ((130, 0, 0), 512.0),
    ((255, 255, 255), 32767.0 + 255.0 / 256.0),
    ((129, 44, 64), 256.0 + 44.25),   # 129*256 + 44.25 - 32768 = 300.25
])
def test_decode_terrarium_known_values(rgb, expected):
    arr = np.array([[rgb]], dtype=np.uint8)           # (1, 1, 3)
    out = DS.decode_terrarium(arr)
    assert out.dtype == np.float32 and out.shape == (1, 1)
    assert out[0, 0] == pytest.approx(expected, abs=1e-3)


def test_decode_terrarium_roundtrip_and_layouts():
    h = np.array([[-430.0, 0.0, 8.5], [1234.25, 3.75, -1.0 / 256]], dtype=np.float64)
    r, g, b = terrarium_rgb(h)
    hwc = np.dstack((r, g, b))
    np.testing.assert_allclose(DS.decode_terrarium(hwc), h, atol=1e-3)
    h2 = np.array([[-430.0, 0.0, 8.5, 1.0, 2.0], [1234.25, 3.75, -1.0 / 256, 5.0, 6.0]])   # W=5: unambiguous CHW
    r2, g2, b2 = terrarium_rgb(h2)
    chw = np.stack((r2, g2, b2))                    # (3, H, W) band-first layout (what rasterio.read gives)
    np.testing.assert_allclose(DS.decode_terrarium(chw), h2, atol=1e-3)
    rgba = np.dstack((r, g, b, np.full(r.shape, 255, np.uint8)))   # alpha ignored
    np.testing.assert_allclose(DS.decode_terrarium(rgba), h, atol=1e-3)


def test_decode_terrarium_rejects_bad_shape():
    with pytest.raises(ValueError):
        DS.decode_terrarium(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        DS.decode_terrarium(np.zeros((6, 6, 2)))


# --------------------------------------------------------------------------- PNG / hillshade

def test_encode_rgba_png_roundtrip():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (7, 9, 4), dtype=np.uint8)
    png = DS.encode_rgba_png(a)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    with MemoryFile(png) as mf, mf.open() as ds:
        assert (ds.width, ds.height, ds.count) == (9, 7, 4)
        back = ds.read()
    np.testing.assert_array_equal(np.moveaxis(back, 0, -1), a)


def test_encode_rgba_png_rejects_bad_shape():
    with pytest.raises(ValueError):
        DS.encode_rgba_png(np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ValueError):
        DS.encode_rgba_png(np.zeros((4, 4), np.uint8))


def _read_png(png):
    with MemoryFile(png) as mf, mf.open() as ds:
        return np.moveaxis(ds.read(), 0, -1)


def test_hillshade_direction_and_flat(tmp_path):
    n = 30
    t = from_origin(500000, 5000000, 1.0, 1.0)
    cols, rows = np.meshgrid(np.arange(n), np.arange(n))
    se = (0.5 * cols + 0.5 * rows).astype(np.float32)      # rises to the SE: faces NW -> lit
    nw = (-se).astype(np.float32)                           # rises to the NW: faces SE -> shadow
    flat = np.full((n, n), 10.0, np.float32)
    for name, a in (("se", se), ("nw", nw), ("flat", flat)):
        write_tif(tmp_path / f"{name}.tif", a, t, "EPSG:32632")
    se_px = _read_png(DS.hillshade_png(tmp_path / "se.tif"))
    nw_px = _read_png(DS.hillshade_png(tmp_path / "nw.tif"))
    flat_px = _read_png(DS.hillshade_png(tmp_path / "flat.tif"))
    c = n // 2
    assert se_px.shape == (n, n, 4)
    assert se_px[c, c, 0] == 255 and se_px[c, c, 3] > 0        # light tone
    assert nw_px[c, c, 0] == 0 and nw_px[c, c, 3] > 0          # dark tone
    assert flat_px[c, c, 3] == 0                                # flat terrain is transparent


def test_hillshade_nodata_transparent_and_stride(tmp_path):
    t = from_origin(500000, 5000000, 1.0, 1.0)
    a = np.random.default_rng(0).normal(100, 3, (80, 100)).astype(np.float32)
    a[10:20, 10:20] = -9999.0
    write_tif(tmp_path / "d.tif", a, t, "EPSG:32632", nodata=-9999.0)
    px = _read_png(DS.hillshade_png(tmp_path / "d.tif", max_dim=25))   # stride = ceil(100/25) = 4
    assert px.shape == (20, 25, 4)
    assert (px[3:4, 3:4, 3] == 0).all()                                 # nodata block (rows 10-20 -> 2..4)
    full = _read_png(DS.hillshade_png(tmp_path / "d.tif"))
    assert full.shape == (80, 100, 4) and (full[10:20, 10:20, 3] == 0).all()


def test_hillshade_geographic_crs_ok(tmp_path):
    n = 20
    t = from_origin(9.0, 48.0, 1 / 3600, 1 / 3600)
    a = np.add.outer(np.arange(n), np.arange(n)).astype(np.float32)
    write_tif(tmp_path / "g.tif", a, t, "EPSG:4326")
    assert _read_png(DS.hillshade_png(tmp_path / "g.tif")).shape == (n, n, 4)


# --------------------------------------------------------------------------- bbox_intersects

def test_bbox_intersects():
    env = (10.0, 20.0, 30.0, 40.0)
    assert DS.bbox_intersects((15, 25, 16, 26), env)
    assert DS.bbox_intersects((0, 0, 50, 50), env)             # contains
    assert DS.bbox_intersects((30.0, 40.0, 31.0, 41.0), env)   # touching corner
    assert not DS.bbox_intersects((31, 25, 32, 26), env)       # north of it
    assert not DS.bbox_intersects((0, 25, 9.9, 26), env)       # south of it
    assert not DS.bbox_intersects((15, 41, 16, 42), env)       # east
    assert not DS.bbox_intersects((15, 0, 16, 19.9), env)      # west


# --------------------------------------------------------------------------- write_float32_tif

def test_write_float32_tif_nodata_tagged(tmp_path):
    a = np.array([[1.0, np.nan], [np.inf, 4.0]], dtype=np.float32)
    p = tmp_path / "o.tif"
    DS.write_float32_tif(p, a, from_origin(0, 2, 1, 1), "EPSG:4326")
    back, nd, _, crs = read_tif(p)
    assert nd == DS.NODATA == -9999.0
    assert back[0, 1] == -9999.0 and back[1, 0] == -9999.0 and back[0, 0] == 1.0 and back[1, 1] == 4.0
    assert back.dtype == np.float32 and crs.to_epsg() == 4326
    with rasterio.open(p) as ds:
        assert ds.tags().get("AREA_OR_POINT") == "Area"
    assert [x.name for x in tmp_path.iterdir()] == ["o.tif"]   # no stray temp file


def test_write_float32_tif_fill_has_no_nodata_tag(tmp_path):
    a = np.array([[1.0, np.nan]], dtype=np.float32)
    p = tmp_path / "o.tif"
    DS.write_float32_tif(p, a, from_origin(0, 1, 1, 1), "EPSG:4326", nodata_fill=0.0, predictor=3, tag_area=False)
    back, nd, _, _ = read_tif(p)
    assert nd is None and back[0, 1] == 0.0


# --------------------------------------------------------------------------- geoid offset

class _FakeTr:
    def __init__(self, h_ortho):
        self.h = h_ortho
        self.calls = []

    def transform(self, lon, lat, h):
        self.calls.append((lon, lat, h))
        return lon, lat, self.h


def _patch_from_crs(monkeypatch, fn):
    import pyproj
    monkeypatch.setattr(pyproj.Transformer, "from_crs", classmethod(lambda cls, *a, **k: fn(*a, **k)))


def test_geoid_offset_sign_adds_orthometric_of_zero_ellipsoid(monkeypatch):
    """h_ortho = transform(ellipsoidal h = 0) is added to the array: Netherlands N=+45 -> offset -45 m."""
    fake = _FakeTr(-45.0)
    _patch_from_crs(monkeypatch, lambda *a, **k: fake)
    arr = np.array([[100.0, np.nan], [0.0, 50.5]], dtype=np.float64)
    out, off, warn = DS.apply_geoid_offset(arr, 5.0, 52.0)
    assert warn is None and off == -45.0 and out.dtype == np.float32
    assert fake.calls == [(5.0, 52.0, 0.0)]                      # lon, lat order, zero ellipsoidal height
    assert out[0, 0] == pytest.approx(55.0) and out[1, 1] == pytest.approx(5.5)
    assert np.isnan(out[0, 1])                                  # NaN stays NaN


@pytest.mark.parametrize("failure", ["raise", "nan"])
def test_geoid_offset_unavailable_is_identity_with_warning(monkeypatch, failure):
    if failure == "raise":
        def boom(*a, **k):
            raise RuntimeError("no grid")
        _patch_from_crs(monkeypatch, boom)
    else:
        _patch_from_crs(monkeypatch, lambda *a, **k: _FakeTr(float("nan")))
    arr = np.array([[10.0, 20.0]], dtype=np.float64)
    out, off, warn = DS.apply_geoid_offset(arr, 0.0, 0.0)
    assert off == 0.0 and out.dtype == np.float32 and np.array_equal(out, [[10.0, 20.0]])
    assert warn and "EGM2008" in warn


def _egm2008_grid_missing():
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4979", "EPSG:4326+3855", always_xy=True)
    return "ballpark" in tr.description


def test_geoid_offset_real_grid_or_warns():
    import pyproj
    if pyproj.network.is_network_enabled():
        pytest.skip("pyproj network enabled: keep offline")
    out, off, warn = DS.apply_geoid_offset(np.zeros((1, 1)), 5.0, 52.0)
    if warn:                                  # grid unavailable AND reported: acceptable
        assert off == 0.0
        return
    assert -50.0 < off < -40.0                # geoid height N ~ +45 m over NL -> orthometric = ellipsoidal - N


# --------------------------------------------------------------------------- ArcGIS ImageServer

BB_ID = (-6.2000, 106.8000, -6.1990, 106.8010)


def _wire(monkeypatch, handler):
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        calls.append({"url": url, "params": params, "headers": headers})
        return handler(len(calls), url, params)

    monkeypatch.setattr(DS.requests, "get", get)
    return calls


def test_imageserver_nodata_masking(tmp_path, monkeypatch):
    a = np.array([[10.0, -9999.0, 3.4e38], [-3.4e38, 20.0, np.nan], [30.0, 40.0, 50.0]], dtype=np.float32)
    body = tif_bytes(a, from_origin(106.8, -6.199, 0.0004, 0.0004), "EPSG:4326")
    calls = _wire(monkeypatch, lambda i, u, p: Resp(body))
    out = tmp_path / "dem.tif"
    info = DS.fetch_arcgis_imageserver("https://x.test/arcgis/ImageServer/", BB_ID, out, 8.0)
    back, nd, _, crs = read_tif(out)
    bad = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 0]], bool)
    assert nd == -9999.0
    assert (back[bad] == -9999.0).all()
    np.testing.assert_allclose(back[~bad], [10, 20, 30, 40, 50])
    assert info["nan_count"] == 4 and info["shape"] == [3, 3] and info["crs"] == "EPSG:4326"
    assert calls[0]["url"] == "https://x.test/arcgis/ImageServer/exportImage"
    p = calls[0]["params"]
    assert p["bbox"] == "106.8000000,-6.2000000,106.8010000,-6.1990000" and p["pixelType"] == "F32"
    assert p["noData"] == "-9999.0" and p["f"] == "image" and p["bboxSR"] == "4326"
    assert calls[0]["headers"]["User-Agent"] == DS.UA


def test_imageserver_custom_nodata_and_unit_scale(tmp_path, monkeypatch):
    a = np.array([[-32768.0, 1000.0], [2000.0, 3000.0]], dtype=np.float32)
    body = tif_bytes(a, from_origin(0, 1, 0.5, 0.5), "EPSG:4326", nodata=-32768.0)    # dataset tag nodata
    _wire(monkeypatch, lambda i, u, p: Resp(body))
    out = tmp_path / "d.tif"
    DS.fetch_arcgis_imageserver("https://x.test/IS", BB_ID, out, 8.0, unit_scale=0.01, nodata_val=-1.0)
    back, _, _, _ = read_tif(out)
    assert back[0, 0] == -9999.0
    np.testing.assert_allclose([back[0, 1], back[1, 0], back[1, 1]], [10.0, 20.0, 30.0], rtol=1e-6)


def test_imageserver_size_clamped(monkeypatch, tmp_path):
    body = tif_bytes(np.ones((2, 2)), from_origin(0, 1, 1, 1), "EPSG:4326")
    calls = _wire(monkeypatch, lambda i, u, p: Resp(body))
    DS.fetch_arcgis_imageserver("https://x/IS", (0.0, 0.0, 0.0001, 0.0001), tmp_path / "a.tif", 8.0)
    assert calls[-1]["params"]["size"] == "16,16"                     # floor 16 px
    DS.fetch_arcgis_imageserver("https://x/IS", (0.0, 0.0, 5.0, 5.0), tmp_path / "b.tif", 1.0)
    assert calls[-1]["params"]["size"] == "4096,4096"                 # ceiling 4096 px


def test_imageserver_projected_bbox(monkeypatch, tmp_path):
    body = tif_bytes(np.ones((2, 2)), from_origin(0, 1, 1, 1), "EPSG:25833")
    calls = _wire(monkeypatch, lambda i, u, p: Resp(body))
    out = tmp_path / "n.tif"
    DS.fetch_arcgis_imageserver("https://x/IS", (60.0, 10.0, 60.01, 10.02), out, 1.0, bbox_sr=25833, image_sr=25833)
    x0, y0, x1, y1 = (float(v) for v in calls[0]["params"]["bbox"].split(","))
    assert x1 > x0 and y1 > y0 and 100000 < x0 < 1000000 and 6.5e6 < y0 < 7.0e6
    assert calls[0]["params"]["bboxSR"] == "25833" and calls[0]["params"]["imageSR"] == "25833"


def test_imageserver_error_json_then_failover(monkeypatch, tmp_path):
    body = tif_bytes(np.full((2, 2), 5.0), from_origin(0, 1, 1, 1), "EPSG:4326")

    def handler(i, url, params):
        if i == 1:
            return Resp(b'{"error":{"code":400}}', ctype="application/json")
        return Resp(body)
    calls = _wire(monkeypatch, handler)
    info = DS.fetch_arcgis_imageserver(["https://a/IS", "https://b/IS"], BB_ID, tmp_path / "o.tif", 8.0)
    assert info["url"] == "https://b/IS" and len(calls) == 2


def test_imageserver_json_body_without_ctype_detected(monkeypatch, tmp_path):
    _wire(monkeypatch, lambda i, u, p: Resp(b'  {"error":1}', ctype="text/plain"))
    with pytest.raises(RuntimeError, match="ImageServer failed"):
        DS.fetch_arcgis_imageserver("https://a/IS", BB_ID, tmp_path / "o.tif", 8.0)


def test_imageserver_all_nodata_and_http_error(monkeypatch, tmp_path):
    body = tif_bytes(np.full((2, 2), -9999.0), from_origin(0, 1, 1, 1), "EPSG:4326")
    _wire(monkeypatch, lambda i, u, p: Resp(body))
    with pytest.raises(RuntimeError, match="only nodata"):
        DS.fetch_arcgis_imageserver("https://a/IS", BB_ID, tmp_path / "o.tif", 8.0)
    assert not (tmp_path / "o.tif").exists()
    _wire(monkeypatch, lambda i, u, p: Resp(b"", status=503))
    with pytest.raises(RuntimeError, match="ImageServer failed"):
        DS.fetch_arcgis_imageserver("https://a/IS", BB_ID, tmp_path / "o.tif", 8.0)


# --------------------------------------------------------------------------- WCS

BB_NL = (52.3700, 4.9000, 52.3710, 4.9015)


def test_wcs_nodata_masking_and_grid_snap(tmp_path, monkeypatch):
    a = np.array([[1.5, 3.4e38, -3.4e38], [-9999.0, 2.5, -9000.5], [4.5, np.nan, 7.5]], dtype=np.float32)
    body = tif_bytes(a, from_origin(120000, 487000, 0.5, 0.5), "EPSG:28992")
    calls = _wire(monkeypatch, lambda i, u, p: Resp(body))
    out = tmp_path / "w.tif"
    info = DS.fetch_wcs_tiff("https://wcs.test/x", "dtm_05m", BB_NL, out, 28992, 0.5)
    back, nd, _, crs = read_tif(out)
    bad = np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]], bool)
    assert (back[bad] == -9999.0).all() and nd == -9999.0
    np.testing.assert_allclose(back[~bad], [1.5, 2.5, 4.5, 7.5])
    assert info["nan_count"] == 5 and crs.to_epsg() == 28992
    subs = [v for k, v in calls[0]["params"] if k == "SUBSET"]
    assert len(subs) == 2 and subs[0].startswith("x(") and subs[1].startswith("y(")
    x0, x1 = (float(v) for v in subs[0][2:-1].split(","))
    assert x0 % 0.5 == 0 and x1 % 0.5 == 0 and x1 > x0                  # snapped outward to the grid
    assert ("COVERAGEID", "dtm_05m") in calls[0]["params"]


def test_wcs_xml_error_falls_through_to_wcs100(tmp_path, monkeypatch):
    body = tif_bytes(np.full((2, 2), 3.0), from_origin(0, 2, 1, 1), "EPSG:28992")

    def handler(i, url, params):
        if i <= 2:
            return Resp(b"<ServiceExceptionReport/>", ctype="text/xml")
        return Resp(body)
    calls = _wire(monkeypatch, handler)
    out = tmp_path / "w.tif"
    info = DS.fetch_wcs_tiff("https://wcs.test/x", "cov", BB_NL, out, 28992, 0.5)
    assert len(calls) == 3
    assert isinstance(calls[0]["params"], list) and isinstance(calls[2]["params"], dict)
    p = calls[2]["params"]
    assert p["VERSION"] == "1.0.0" and p["COVERAGE"] == "cov" and p["CRS"] == "EPSG:28992"
    assert int(p["WIDTH"]) >= 16 and int(p["HEIGHT"]) >= 16
    assert info["shape"] == [2, 2]


def test_wcs_all_attempts_fail(tmp_path, monkeypatch):
    calls = _wire(monkeypatch, lambda i, u, p: Resp(b"<html>no</html>", ctype="text/html"))
    with pytest.raises(RuntimeError, match="WCS fetch failed"):
        DS.fetch_wcs_tiff("https://wcs.test/x", "cov", BB_NL, tmp_path / "w.tif", 28992, 0.5)
    assert len(calls) == 3 and not (tmp_path / "w.tif").exists()


def test_wcs_all_nodata_rejected(tmp_path, monkeypatch):
    body = tif_bytes(np.full((2, 2), 3.4e38), from_origin(0, 2, 1, 1), "EPSG:28992")
    _wire(monkeypatch, lambda i, u, p: Resp(body))
    with pytest.raises(RuntimeError, match="WCS fetch failed"):
        DS.fetch_wcs_tiff("https://wcs.test/x", "cov", BB_NL, tmp_path / "w.tif", 28992, 0.5)


# --------------------------------------------------------------------------- COG / VRT (local file standing in)

def test_cog_vrt_clip_nodata_scale(tmp_path):
    n = 100
    a = np.arange(n * n, dtype=np.float32).reshape(n, n)
    a[40:44, 40:44] = -32768.0
    src = tmp_path / "src.tif"
    t = from_origin(10.0, 50.0, 0.001, 0.001)
    with rasterio.open(src, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32", crs="EPSG:4326",
                       transform=t, nodata=-32768.0) as ds:
        ds.write(a, 1)
        ds.scales = (0.5,)
        ds.offsets = (10.0,)
    bb = (49.95, 10.03, 49.97, 10.06)       # rows 30..50, cols 30..60
    out = tmp_path / "o.tif"
    info = DS.fetch_cog_vrt(str(src), bb, out)
    back, nd, bt, crs = read_tif(out)
    assert info["url"] == str(src) and crs.to_epsg() == 4326
    assert back.shape == (20, 30) and bt.a == pytest.approx(0.001)
    assert back[0, 0] == pytest.approx(a[30, 30] * 0.5 + 10.0)
    assert back[10, 10] == -9999.0                     # nodata block (row 40, col 40) masked
    assert back[11, 12] == pytest.approx(a[41 + 0, 42] * 0.5 + 10.0) or back[11, 12] == -9999.0


def test_cog_vrt_no_overlap_and_failover(tmp_path):
    src = tmp_path / "src.tif"
    write_tif(src, np.ones((10, 10)), from_origin(10.0, 50.0, 0.001, 0.001))
    with pytest.raises(RuntimeError, match="COG/VRT fetch failed"):
        DS.fetch_cog_vrt(str(src), (0.0, 0.0, 0.01, 0.01), tmp_path / "o.tif")
    info = DS.fetch_cog_vrt([str(tmp_path / "missing.tif"), str(src)], (49.992, 10.002, 49.998, 10.008),
                            tmp_path / "o2.tif")
    assert info["url"] == str(src)


def test_cog_vrt_all_nodata_rejected(tmp_path):
    src = tmp_path / "src.tif"
    write_tif(src, np.full((10, 10), -9999.0), from_origin(10.0, 50.0, 0.001, 0.001), nodata=-9999.0)
    with pytest.raises(RuntimeError, match="COG/VRT fetch failed"):
        DS.fetch_cog_vrt(str(src), (49.992, 10.002, 49.998, 10.008), tmp_path / "o.tif")


# --------------------------------------------------------------------------- zoom picker

def _tile_count(bb, z):
    s, w, n, e = bb
    x0, y0 = DS._latlon_to_tile(n, w, z)
    x1, y1 = DS._latlon_to_tile(s, e, z)
    return (x1 - x0 + 1) * (y1 - y0 + 1)


def test_pick_zoom_small_site_gets_max_zoom():
    assert DS._pick_terrarium_zoom((52.370, 4.900, 52.3709, 4.9013)) == 15


@pytest.mark.parametrize("span", [0.002, 0.01, 0.03, 0.08, 0.2])
def test_pick_zoom_caps_tile_count(span):
    bb = (52.3, 4.9, 52.3 + span, 4.9 + span)
    z = DS._pick_terrarium_zoom(bb)
    assert 11 <= z <= 15
    assert _tile_count(bb, z) <= 16
    if z < 15:                                          # highest zoom within the cap is chosen
        assert _tile_count(bb, z + 1) > 16


def test_pick_zoom_decreases_monotonically_with_size():
    zs = [DS._pick_terrarium_zoom((52.3, 4.9, 52.3 + s, 4.9 + s)) for s in (0.002, 0.02, 0.2, 2.0)]
    assert zs == sorted(zs, reverse=True) and zs[0] == 15


def test_pick_zoom_huge_bbox_floors_at_11():
    # documented behaviour: floor z=11 even though the tile count then exceeds the cap
    bb = (50.0, 0.0, 55.0, 10.0)
    assert DS._pick_terrarium_zoom(bb) == 11 and _tile_count(bb, 11) > 16


def test_pick_zoom_custom_cap():
    bb = (52.3, 4.9, 52.4, 5.0)
    assert _tile_count(bb, DS._pick_terrarium_zoom(bb, max_tiles=4)) <= 4 or DS._pick_terrarium_zoom(bb, 4) == 11
    assert DS._pick_terrarium_zoom(bb, max_tiles=10_000) == 15


def test_latlon_to_tile_known():
    assert DS._latlon_to_tile(0.0, 0.0, 1) == (1, 1)
    assert DS._latlon_to_tile(85.0, -180.0, 2) == (0, 0)
    assert DS._latlon_to_tile(-89.9, 179.9, 3) == (7, 7)     # clamped to the Web-Mercator limit


# --------------------------------------------------------------------------- Mapterhorn fetcher

class FakeSession:
    """Stands in for requests.Session; `handler(url) -> Resp`."""
    handler = None
    urls: list = []

    def __init__(self):
        self.headers = {}

    def get(self, url, timeout=None):
        FakeSession.urls.append(url)
        return FakeSession.handler(url)

    def close(self):
        pass


def _install_session(monkeypatch, handler):
    FakeSession.handler = staticmethod(handler)
    FakeSession.urls = []
    monkeypatch.setattr(DS.requests, "Session", FakeSession)
    return FakeSession


def _tile_heights(tx, ty, px=16):
    r, c = np.mgrid[0:px, 0:px]
    return (1000.0 + 50 * (tx % 7) + 5 * (ty % 5) + r * 0.25 + c * 0.0625).astype(np.float64)


def _tile_url_xyz(url):
    parts = url.replace(".png", "").replace(".webp", "").split("/")
    return int(parts[-3]), int(parts[-2]), int(parts[-1])


def test_mapterhorn_mosaic_values_and_georef(tmp_path, monkeypatch):
    def handler(url):
        z, x, y = _tile_url_xyz(url)
        return Resp(terrarium_png(_tile_heights(x, y)), ctype="image/png")
    sess = _install_session(monkeypatch, handler)
    bb = (52.3700, 4.9000, 52.3720, 4.9040)
    z = 14
    out = tmp_path / "mh.tif"
    info = DS.fetch_mapterhorn(bb, out, tmp_path / "cache", zoom=z, workers=2)
    x0, y0 = DS._latlon_to_tile(bb[2], bb[1], z)
    x1, y1 = DS._latlon_to_tile(bb[0], bb[3], z)
    ntiles = (x1 - x0 + 1) * (y1 - y0 + 1)
    assert info["zoom"] == z and info["tiles"] == ntiles and info["tile_px"] == 16 and info["crs"] == "EPSG:3857"
    assert info["sources"] == ["mapterhorn"] and all(u.startswith("https://tiles.mapterhorn.com/") for u in sess.urls)
    back, nd, t, crs = read_tif(out)
    assert crs.to_epsg() == 3857 and nd == -9999.0
    assert back.shape == ((y1 - y0 + 1) * 16, (x1 - x0 + 1) * 16)
    tile_m = 2 * 20037508.342789244 / (1 << z)
    assert t.a == pytest.approx(tile_m / 16) and t.e == pytest.approx(-tile_m / 16)
    assert t.c == pytest.approx(-20037508.342789244 + x0 * tile_m)
    assert t.f == pytest.approx(20037508.342789244 - y0 * tile_m)
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            blk = back[(ty - y0) * 16:(ty - y0 + 1) * 16, (tx - x0) * 16:(tx - x0 + 1) * 16]
            np.testing.assert_allclose(blk, _tile_heights(tx, ty), atol=1e-3)


def test_mapterhorn_default_zoom_and_cache_reuse(tmp_path, monkeypatch):
    sess = _install_session(monkeypatch, lambda url: Resp(terrarium_png(_tile_heights(1, 1)), ctype="image/png"))
    bb = (52.3700, 4.9000, 52.3709, 4.9013)
    info = DS.fetch_mapterhorn(bb, tmp_path / "a.tif", tmp_path / "cache")
    assert info["zoom"] == DS._pick_terrarium_zoom(bb) == 15
    n1 = len(sess.urls)
    assert n1 == info["tiles"]
    DS.fetch_mapterhorn(bb, tmp_path / "b.tif", tmp_path / "cache")
    assert len(sess.urls) == n1                       # second run served from the tile cache
    assert any((tmp_path / "cache" / "_tiles" / "mapterhorn").glob("15_*_s0.webp"))


def test_mapterhorn_falls_back_to_aws_endpoint(tmp_path, monkeypatch):
    def handler(url):
        if "mapterhorn" in url:
            return Resp(b"", status=404)
        z, x, y = _tile_url_xyz(url)
        return Resp(terrarium_png(_tile_heights(x, y)), ctype="image/png")
    _install_session(monkeypatch, handler)
    info = DS.fetch_mapterhorn((52.3700, 4.9000, 52.3709, 4.9013), tmp_path / "m.tif", tmp_path / "c")
    assert info["sources"] == ["aws_terrarium"]


def test_mapterhorn_rejects_tiny_or_undecodable_then_all_fail(tmp_path, monkeypatch):
    def handler(url):
        return Resp(b"x" * 10) if "mapterhorn" in url else Resp(b"y" * 200)   # tiny / undecodable
    _install_session(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="no Terrarium tile available"):
        DS.fetch_mapterhorn((52.3700, 4.9000, 52.3709, 4.9013), tmp_path / "m.tif", tmp_path / "c")
    assert not (tmp_path / "m.tif").exists()
    assert not list((tmp_path / "c" / "_tiles" / "mapterhorn").glob("*_s*"))   # bad tiles are not cached


# --------------------------------------------------------------------------- _distance_to_invalid / feathering

def test_distance_to_invalid():
    v = np.ones((11, 11), bool)
    assert (DS._distance_to_invalid(v, 5) == 5).all()                 # no invalid cell: capped
    v[5, 5] = False
    d = DS._distance_to_invalid(v, 5)
    assert d[5, 5] == 0
    assert d[5, 6] == pytest.approx(1.0) and d[5, 8] == pytest.approx(3.0)
    assert d[6, 6] == pytest.approx(1.4142, abs=1e-3)
    assert d.max() == 5.0 and d[0, 0] == 5.0


def _primary(tmp_path, holes=((27, 33, 27, 33),), n=60, val=100.0):
    arr = np.full((n, n), val, np.float32)
    for r0, r1, c0, c1 in holes:
        arr[r0:r1, c0:c1] = np.nan
    d = 0.0001
    t = from_origin(5.0, 52.0, d, d)
    p = tmp_path / "primary.tif"
    DS.write_float32_tif(p, arr, t, "EPSG:4326")
    return p, (52.0 - n * d, 5.0, 52.0, 5.0 + n * d), t, arr


def _fallback_const(value, log=None):
    def fb(bb, path):
        if log is not None:
            log.append((bb, Path(path)))
        s, w, n, e = bb
        m = 20                                          # coarser grid than the primary
        t = from_origin(w - 0.0005, n + 0.0005, (e - w + 0.001) / m, (n - s + 0.001) / m)
        DS.write_float32_tif(path, np.full((m, m), value, np.float32), t, "EPSG:4326")
        return {"warnings": ["fb-note"]}
    return fb


def test_feather_keeps_interior_and_fills_holes(tmp_path):
    p, bb, t, arr = _primary(tmp_path)
    log = []
    warns = DS.feather_with_fallback(p, bb, _fallback_const(50.0, log), feather_px=6)
    back, nd, _, _ = read_tif(p)
    assert len(log) == 1 and log[0][0] == bb
    assert "fb-note" in warns and any("Feathered 36" in w for w in warns)
    assert not (back == -9999.0).any() and np.isfinite(back).all()
    hole = back[27:33, 27:33]
    np.testing.assert_allclose(hole, 50.0, atol=1e-3)                  # fallback inside the hole
    far = np.ones(back.shape, bool)
    far[27 - 7:33 + 7, 27 - 7:33 + 7] = False                          # > feather_px from the hole
    np.testing.assert_allclose(back[far], 100.0, atol=1e-4)            # primary untouched in the interior
    ramp = back[30, 26::-1][:7]                                         # moving away from the hole
    assert 50.0 < ramp[0] < 100.0 and (np.diff(ramp) >= -1e-4).all() and ramp[-1] == pytest.approx(100.0, abs=1e-3)


def test_feather_no_holes_does_not_call_fallback(tmp_path):
    p, bb, t, arr = _primary(tmp_path, holes=())

    def boom(bb, path):
        raise AssertionError("fallback must not be called")
    assert DS.feather_with_fallback(p, bb, boom) == []
    back, _, _, _ = read_tif(p)
    np.testing.assert_array_equal(back, arr)


def test_feather_zero_px_plain_fill(tmp_path):
    p, bb, t, arr = _primary(tmp_path)
    DS.feather_with_fallback(p, bb, _fallback_const(50.0), feather_px=0)
    back, _, _, _ = read_tif(p)
    np.testing.assert_allclose(back[27:33, 27:33], 50.0, atol=1e-3)
    np.testing.assert_allclose(back[26, 26], 100.0)                    # no blending near the seam


def test_feather_fallback_in_other_crs(tmp_path):
    p, bb, t, arr = _primary(tmp_path)
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)

    def fb(b, path):
        s, w, n, e = b
        x0, y0 = tr.transform(w - 0.001, s - 0.001)
        x1, y1 = tr.transform(e + 0.001, n + 0.001)
        px = 3.0
        wd, ht = int((x1 - x0) / px) + 2, int((y1 - y0) / px) + 2
        DS.write_float32_tif(path, np.full((ht, wd), 70.0, np.float32), from_origin(x0, y1 + px, px, px), "EPSG:32631")
        return {}
    DS.feather_with_fallback(p, bb, fb, feather_px=4)
    back, _, _, _ = read_tif(p)
    np.testing.assert_allclose(back[28:32, 28:32], 70.0, atol=1e-3)
    np.testing.assert_allclose(back[0, 0], 100.0, atol=1e-4)


def test_feather_fallback_without_overlap_leaves_nodata(tmp_path):
    p, bb, t, arr = _primary(tmp_path)

    def fb(b, path):
        DS.write_float32_tif(path, np.full((4, 4), 9.0, np.float32), from_origin(100.0, 10.0, 0.01, 0.01), "EPSG:4326")
        return {}
    DS.feather_with_fallback(p, bb, fb)
    back, nd, _, _ = read_tif(p)
    assert (back[27:33, 27:33] == -9999.0).all() and back[0, 0] == pytest.approx(100.0)


# --------------------------------------------------------------------------- registry consistency

REQ_KEYS = {"name", "attribution", "licence", "vertical_datum", "accuracy"}
ACC_KEYS = {"horizontal_m", "vertical_m", "kind", "note"}
SOURCE_IDS = sorted(set(DS.LAYERED_DEM_INFO) - {"layered"})


@pytest.mark.parametrize("sid", sorted(DS.LAYERED_DEM_INFO))
def test_registry_entry_complete(sid):
    inf = DS.LAYERED_DEM_INFO[sid]
    assert REQ_KEYS <= set(inf)
    for k in ("name", "attribution", "licence", "vertical_datum"):
        assert isinstance(inf[k], str) and inf[k].strip()
    acc = inf["accuracy"]
    assert ACC_KEYS <= set(acc)
    assert acc["kind"] in ("DTM", "DSM") and isinstance(acc["note"], str) and acc["note"]
    if sid != "layered":
        assert acc["horizontal_m"] > 0 and acc["vertical_m"] > 0
    # must survive fetch._dem_meta (attribution .format(year=...)) via the merged DEM_INFO
    meta = F._dem_meta(sid, 2026)
    assert meta["dem_source_id"] == sid and meta["dem_accuracy"]["kind"] == acc["kind"]


def test_registry_merged_into_fetch_dem_info():
    for sid in DS.LAYERED_DEM_INFO:
        assert F.DEM_INFO[sid] is DS.LAYERED_DEM_INFO[sid]
    assert "bw_dgm1" in F.DEM_INFO and "copernicus" in F.DEM_INFO
    assert not {"bw_dgm1", "copernicus"} & set(DS.LAYERED_DEM_INFO)   # these two are handled outside the stack


def test_registry_expected_ids():
    assert set(DS.LAYERED_DEM_INFO) == {"layered", "id_demnas", "us_3dep", "nl_ahn", "fr_ign", "uk_ea",
                                        "no_kartverket", "arcticdem", "rema", "mapterhorn", "gedtm30", "fabdem",
                                        # M9b
                                        "vt_vcgi", "il_isgs", "ky_kyfromabove", "cz_cuzk", "hk_cedd", "au_qld",
                                        "tz_msimbazi", "ee_maaamet", "nz_linz", "ht_cnigs", "jp_gsj", "mx_inegi"}


def test_tier1_order_and_envelopes_consistent():
    assert DS.TIER1_ORDER[0] == "bw_dgm1"
    for sid in DS.TIER1_ORDER:
        assert sid == "bw_dgm1" or (sid in DS.ENVELOPES and sid in DS.LAYERED_DEM_INFO)
    assert set(DS.ENVELOPES) <= set(DS.TIER1_ORDER)
    for sid, (s, w, n, e) in DS.ENVELOPES.items():
        assert -90 <= s < n <= 90 and -180 <= w < e <= 180, sid


@pytest.mark.parametrize("sid,lat,lon", [
    ("id_demnas", -6.2, 106.8), ("us_3dep", 40.7, -74.0), ("us_3dep", 21.3, -157.8), ("us_3dep", 61.2, -149.9),
    ("nl_ahn", 52.37, 4.9), ("fr_ign", 48.85, 2.35), ("uk_ea", 51.5, -0.12), ("no_kartverket", 59.9, 10.75),
    ("no_kartverket", 69.65, 18.96),
])
def test_envelopes_contain_known_cities(sid, lat, lon):
    assert DS.bbox_intersects((lat, lon, lat + 0.001, lon + 0.001), DS.ENVELOPES[sid])


@pytest.mark.parametrize("sid,lat,lon", [
    ("nl_ahn", -6.2, 106.8), ("id_demnas", 52.37, 4.9), ("uk_ea", 48.85, 2.35), ("us_3dep", 52.37, 4.9),
    ("no_kartverket", 40.7, -74.0),
])
def test_envelopes_exclude_far_places(sid, lat, lon):
    assert not DS.bbox_intersects((lat, lon, lat + 0.001, lon + 0.001), DS.ENVELOPES[sid])


def test_endpoints_shape():
    for sid in SOURCE_IDS:
        if sid in ("mapterhorn",):
            continue
        assert sid in DS.ENDPOINTS or sid == "mapterhorn", sid
    for sid in ("nl_ahn", "fr_ign", "uk_ea"):
        url, cov, epsg, res = DS.ENDPOINTS[sid]
        assert url.startswith("https://") and cov and isinstance(epsg, int) and res > 0
    for sid in ("id_demnas", "us_3dep", "no_kartverket", "arcticdem", "rema", "gedtm30", "fabdem", "mapterhorn_xyz"):
        urls = DS.ENDPOINTS[sid]
        assert isinstance(urls, list) and urls and all(u.startswith("https://") for u in urls)
    for tmpl in DS.ENDPOINTS["mapterhorn_xyz"]:
        assert all(k in tmpl for k in ("{z}", "{x}", "{y}"))


# --------------------------------------------------------------------------- _run_source dispatcher

@pytest.fixture
def stub_fetchers(monkeypatch):
    """Replace every protocol fetcher with a recorder that writes a small valid GeoTIFF."""
    calls = []

    def writer(out_path, crs="EPSG:4326"):
        DS.write_float32_tif(out_path, np.array([[10.0, 20.0], [np.nan, 40.0]], np.float32),
                             from_origin(0, 2, 1, 1), crs)

    def arcgis(url, bb, out_path, target_res_m, **kw):
        calls.append(("arcgis", url, target_res_m, kw))
        crs = f"EPSG:{kw['image_sr']}" if "image_sr" in kw else "EPSG:4326"
        writer(out_path, crs)
        return {"crs": crs, "shape": [2, 2]}

    def wcs(url, cov, bb, out_path, epsg, res_m, **kw):
        calls.append(("wcs", url, cov, epsg, res_m))
        writer(out_path, f"EPSG:{epsg}")
        return {"crs": f"EPSG:{epsg}"}

    def mapterhorn(bb, out_path, cache_dir, **kw):
        calls.append(("mapterhorn", Path(cache_dir)))
        writer(out_path, "EPSG:3857")
        return {"crs": "EPSG:3857"}

    def cog(urls, bb, out_path, **kw):
        calls.append(("cog", urls))
        writer(out_path)
        return {"crs": "EPSG:4326"}

    def wms(url, layer, bb, out_path, epsg, res_m, **kw):      # M9b: fr_ign moved from WCS to WMS-R GetMap
        calls.append(("wms", url, layer, epsg, res_m))
        writer(out_path, f"EPSG:{epsg}")
        return {"crs": f"EPSG:{epsg}"}

    def xyz_rgb(url_tmpl, bb, out_path, cache_dir, cache_key, **kw):
        calls.append(("xyz_rgb", url_tmpl, cache_key))
        writer(out_path, "EPSG:3857")
        return {"crs": "EPSG:3857"}

    monkeypatch.setattr(DS, "fetch_arcgis_imageserver", arcgis)
    monkeypatch.setattr(DS, "fetch_wms_geotiff", wms)
    monkeypatch.setattr(DS, "fetch_xyz_terrain_rgb", xyz_rgb)
    monkeypatch.setattr(DS, "fetch_wcs_tiff", wcs)
    monkeypatch.setattr(DS, "fetch_mapterhorn", mapterhorn)
    monkeypatch.setattr(DS, "fetch_cog_vrt", cog)
    monkeypatch.setattr(DS, "apply_geoid_offset", lambda arr, lon, lat: ((arr + 7.0).astype(np.float32), 7.0, None))
    return calls


@pytest.mark.parametrize("sid", [i for i in SOURCE_IDS if i not in ("nz_linz", "tz_msimbazi")])   # those two raise
def test_every_registry_id_reachable_by_dispatcher(sid, tmp_path, stub_fetchers):
    bw = (lambda bb, p: (DS.write_float32_tif(p, np.ones((2, 2), np.float32), from_origin(0, 2, 1, 1), "EPSG:25832"),
                         {"warnings": ["w"]})[1])
    extra, warns = DS._run_source(sid, (1.0, 2.0, 1.1, 2.1), tmp_path / "o.tif", tmp_path, bw_fn=bw)
    assert extra["dem_crs"] and (tmp_path / "o.tif").exists() and isinstance(warns, list)


def test_dispatcher_bw_dgm1(tmp_path):
    def bw(bb, p):
        DS.write_float32_tif(p, np.ones((2, 2), np.float32), from_origin(0, 2, 1, 1), "EPSG:25832")
        return {"warnings": ["partial"], "tiles": 2}
    extra, warns = DS._run_source("bw_dgm1", (1, 2, 3, 4), tmp_path / "o.tif", tmp_path, bw_fn=bw)
    assert extra["dem_crs"] == "EPSG:25832" and extra["tiles"] == 2 and warns == ["partial"]
    with pytest.raises(RuntimeError, match="bw_dgm1"):
        DS._run_source("bw_dgm1", (1, 2, 3, 4), tmp_path / "o2.tif", tmp_path)


def test_dispatcher_unknown_and_non_stack_ids(tmp_path):
    for bad in ("nope", "copernicus", "layered", "file", ""):
        with pytest.raises(ValueError, match="unknown layered DEM source"):
            DS._run_source(bad, (1, 2, 3, 4), tmp_path / "o.tif", tmp_path)


def test_dispatcher_arguments(tmp_path, stub_fetchers):
    calls = stub_fetchers
    bb = (1.0, 2.0, 1.1, 2.1)
    run = lambda sid: DS._run_source(sid, bb, tmp_path / f"{sid}.tif", tmp_path)
    extra, _ = run("id_demnas")
    assert calls[-1][1] == DS.ENDPOINTS["id_demnas"] and calls[-1][2] == 8.0 and extra["dem_crs"] == "EPSG:4326"
    run("us_3dep")
    assert calls[-1][1] == DS.ENDPOINTS["us_3dep"] and calls[-1][2] == 1.0
    extra, _ = run("no_kartverket")
    assert calls[-1][3]["image_sr"] == 25833 and extra["dem_crs"] == "EPSG:25833"
    extra, _ = run("nl_ahn")
    assert calls[-1] == ("wcs", DS.ENDPOINTS["nl_ahn"][0], "dtm_05m", 28992, 0.5) and extra["dem_crs"] == "EPSG:28992"
    run("fr_ign")
    assert calls[-1] == ("wms", DS.ENDPOINTS["fr_ign"][0], "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES", 2154, 1.0)
    run("uk_ea")
    assert calls[-1][3] == 27700
    run("gedtm30")
    assert calls[-1] == ("cog", DS.ENDPOINTS["gedtm30"])
    run("fabdem")
    assert calls[-1] == ("cog", DS.ENDPOINTS["fabdem"])
    extra, _ = run("mapterhorn")
    assert calls[-1][1] == tmp_path and extra["dem_crs"] == "EPSG:3857"


@pytest.mark.parametrize("sid", ["arcticdem", "rema"])
def test_dispatcher_polar_applies_geoid_and_keeps_nodata(sid, tmp_path, stub_fetchers):
    extra, warns = DS._run_source(sid, (70.0, -40.0, 70.01, -39.99), tmp_path / "p.tif", tmp_path)
    back, nd, _, _ = read_tif(tmp_path / "p.tif")
    assert stub_fetchers[-1][2] == 2.0 and extra["geoid_offset_m"] == 7.0 and warns == []
    assert back[0, 0] == pytest.approx(17.0) and back[0, 1] == pytest.approx(27.0) and back[1, 1] == pytest.approx(47.0)
    assert back[1, 0] == -9999.0 and nd == -9999.0                      # nodata cell not shifted into valid data


def test_dispatcher_polar_geoid_warning_propagates(tmp_path, stub_fetchers, monkeypatch):
    monkeypatch.setattr(DS, "apply_geoid_offset", lambda arr, lon, lat: (arr.astype(np.float32), 0.0, "grid missing"))
    _, warns = DS._run_source("arcticdem", (70.0, -40.0, 70.01, -39.99), tmp_path / "p.tif", tmp_path)
    assert warns == ["grid missing"]
