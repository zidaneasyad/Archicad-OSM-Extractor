"""Offline tests: DEM written on the SOURCE grid (no stretching/shift), DEM source plumbing."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from osmctx import fetch as F
from osmctx.boundary import square_boundary
from osmctx.convert import Dem


@pytest.fixture(autouse=True)
def _no_overture_network(monkeypatch):
    """Keep these tests offline: F.fetch would otherwise query the live Overture S3 bucket."""
    monkeypatch.setattr(F.OVT, "try_fetch", lambda *a, **k: {"status": "failed", "error": "offline test stub"})

AS = 1.0 / 3600.0  # one arc-second


def fake_glo30_tiles(tmp: Path, lon0: float, lat_top: float, dx: float, ncols: int, nrows: int, ntiles: int = 2):
    """GLO-30-like tiles side by side in longitude: pixel centres on whole multiples of dx/1" from
    (lon0, lat_top) (GDAL transform origin = half a pixel west/north, as in the real COGs).
    Value of global pixel (R, C) = 1000*R + C (exact in float32)."""
    paths = []
    for k in range(ntiles):
        x0 = lon0 + k * ncols * dx - dx / 2
        y0 = lat_top + AS / 2
        C, R = np.meshgrid(np.arange(ncols) + k * ncols, np.arange(nrows))
        a = (1000.0 * R + C).astype("float32")
        p = tmp / f"tile{k}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=nrows, width=ncols, count=1, dtype="float32",
                           crs="EPSG:4326", transform=from_origin(x0, y0, dx, AS)) as ds:
            ds.write(a, 1)
            ds.update_tags(AREA_OR_POINT="Point")
        paths.append(p)
    return paths


def check_on_grid(out_path, lon0, lat_top, dx):
    """Every output pixel centre must coincide with a source pixel centre and carry its value."""
    with rasterio.open(out_path) as ds:
        t = ds.transform
        a = ds.read(1)
    assert t.a == pytest.approx(dx, rel=1e-12) and -t.e == pytest.approx(AS, rel=1e-12)  # not stretched
    rows, cols = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    lon = t.c + (cols + 0.5) * t.a
    lat = t.f + (rows + 0.5) * t.e
    C = (lon - lon0) / dx
    R = (lat_top - lat) / AS
    assert np.abs(C - np.round(C)).max() < 1e-6      # centres on the source lattice (lon)
    assert np.abs(R - np.round(R)).max() < 1e-6      # centres on the source lattice (lat)
    assert np.array_equal(a, (1000.0 * np.round(R) + np.round(C)).astype("float32"))  # values not shifted
    return t, a


@pytest.mark.parametrize("lat_top,dx", [(48.55, AS), (55.05, 1.5 * AS)])  # 1.5" lon spacing above 50 deg
def test_dem_on_source_grid(tmp_path, lat_top, dx):
    lon0 = 9.0
    nc, nr = 180, 180
    tiles = fake_glo30_tiles(tmp_path, lon0, lat_top, dx, nc, nr)
    seam = lon0 + nc * dx - dx / 2
    # unsnapped bbox straddling the tile seam (s, w, n, e)
    bb = (lat_top - 0.031234567, seam - 0.0123456789, lat_top - 0.0187654321, seam + 0.0098765432)
    out = tmp_path / "dem.tif"
    info = F.dem_from_sources(tiles, bb, out)
    assert not info["warnings"]
    t, a = check_on_grid(out, lon0, lat_top, dx)
    s, w, n, e = bb
    assert t.c <= w and t.c + a.shape[1] * t.a >= e and t.f >= n and t.f + a.shape[0] * t.e <= s
    assert t.c > w - t.a and t.f < n + AS                  # snapped outward by less than one pixel
    # a known source pixel, read back at its exact lon/lat
    Rk, Ck = 100, nc + 7                                   # in the second tile
    lon_k, lat_k = lon0 + Ck * dx, lat_top - Rk * AS
    with rasterio.open(out) as ds:
        r, c = ds.index(lon_k, lat_k)
        assert ds.read(1)[r, c] == 1000.0 * Rk + Ck
    assert Dem(out).sample(lon_k, lat_k)[0] == pytest.approx(1000.0 * Rk + Ck, abs=1e-6)  # bilinear at centre


def test_snap_bounds_outward():
    t = from_origin(9.0 - AS / 2, 49.0 + AS / 2, AS, AS)
    W, S, E, N = F.snap_bounds((9.00012, 48.99, 9.0003, 48.995), t)
    for v, ref in ((W, t.c), (E, t.c)):
        assert abs((v - ref) / AS - round((v - ref) / AS)) < 1e-9
    assert W <= 9.00012 < W + AS and E - AS < 9.0003 <= E
    assert S <= 48.99 < S + AS and N - AS < 48.995 <= N


def test_bw_tile_naming():
    assert F.bw_zip_tile(504, 5374) == (503, 5374)   # verified live 2026-11-15: dgm1_32_503_5374_2_bw.zip
    assert F.bw_zip_tile(503, 5375) == (503, 5374)
    assert F.bw_zip_tile(513, 5405) == (513, 5404)
    assert F.bw_subtiles((504092.0, 5373942.0, 504252.0, 5374102.0)) == [(504, 5373), (504, 5374)]
    assert F.BW_DGM1_ZIP.format(e=503, n=5374).endswith("/dgm/dgm1_32_503_5374_2_bw.zip")


def test_parse_dgm1_xyz():
    lines = []
    for r in range(1000):
        for c in range(1000):
            if (r, c) == (3, 4):
                continue  # one missing cell
            lines.append(f"{504000 + c + 0.5:.2f} {5374999.5 - r:.2f} {300 + r * 0.01 + c * 0.001:.2f}")
    a = F.parse_dgm1_xyz("\r\n".join(lines).encode(), 504, 5374)
    assert a.shape == (1000, 1000) and np.isnan(a[3, 4]) and np.isnan(a).sum() == 1
    assert a[0, 0] == pytest.approx(300.0) and a[10, 20] == pytest.approx(300.12, abs=1e-4)


def test_file_source_projected(tmp_path):
    """file:<path> in EPSG:25832, kept on its own 1 m grid and CRS; converter samples it CRS-aware."""
    src = tmp_path / "dtm.tif"
    E0, N0 = 503800.0, 5374400.0  # pixel corners on whole metres, 1 m
    C, R = np.meshgrid(np.arange(800), np.arange(800))
    a = (300.0 + 0.01 * C + 0.02 * R).astype("float32")
    with rasterio.open(src, "w", driver="GTiff", height=800, width=800, count=1, dtype="float32",
                       crs="EPSG:25832", transform=from_origin(E0, N0, 1.0, 1.0)) as ds:
        ds.write(a, 1)
    poly_bb = (48.5185, 9.0554, 48.5201, 9.0576)
    out = tmp_path / "dem.tif"
    info = F.fetch_file_dem(str(src), poly_bb, out)
    assert info["pixel_m"] == 1.0 and not info["warnings"]
    with rasterio.open(out) as ds:
        t, b = ds.transform, ds.read(1)
        assert ds.crs.to_epsg() == 25832
    assert t.a == 1.0 and t.c == round(t.c) and t.f == round(t.f)
    r0, c0 = int(N0 - t.f), int(t.c - E0)
    assert np.array_equal(b, a[r0:r0 + b.shape[0], c0:c0 + b.shape[1]])
    # Dem samples by lon/lat through the file CRS: value at a pixel centre is exact
    from pyproj import Transformer
    lon, lat = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True).transform(t.c + 10.5, t.f - 20.5)
    assert Dem(out).sample(lon, lat)[0] == pytest.approx(b[20, 10], abs=1e-3)


def test_fetch_copernicus_meta(tmp_path, monkeypatch):
    """dem_source='copernicus' end to end offline: meta carries source, licence, accuracy (DSM), crs."""
    tiles = fake_glo30_tiles(tmp_path, 9.0, 48.55, AS, 360, 180, ntiles=1)
    monkeypatch.setattr(F, "copernicus_tile_urls", lambda bb: tiles)
    monkeypatch.setattr(F, "overpass", lambda q: ({"elements": []}, "stub"))
    d = F.fetch(square_boundary(48.52, 9.06, 100), tmp_path / "cache", dem_source="copernicus")
    m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert m["dem_source_id"] == "copernicus" and m["dem_crs"] == "EPSG:4326"
    assert m["dem_accuracy"]["kind"] == "DSM" and m["dem_accuracy"]["horizontal_m"] == 30.0
    assert "Copernicus" in m["dem_attribution"]
    check_on_grid(d / "dem.tif", 9.0, 48.55, AS)
    # cache key depends on dem_source
    assert F.cache_key(square_boundary(48.52, 9.06, 100), "copernicus") != \
        F.cache_key(square_boundary(48.52, 9.06, 100), "auto")


def test_fetch_auto_falls_back_outside_bw(tmp_path, monkeypatch):
    tiles = fake_glo30_tiles(tmp_path, 106.0, -6.10, AS, 360, 180, ntiles=1)
    monkeypatch.setattr(F, "copernicus_tile_urls", lambda bb: tiles)
    monkeypatch.setattr(F, "overpass", lambda q: ({"elements": []}, "stub"))

    def boom(*a, **k):
        raise AssertionError("bw_dgm1 must not be tried outside BW")
    monkeypatch.setattr(F, "fetch_bw_dgm1", boom)
    d = F.fetch(square_boundary(-6.15, 106.05, 100), tmp_path / "cache", dem_source="auto")
    m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert m["dem_source_id"] == "copernicus" and "outside Baden" in m["dem_fallback_reason"]


def test_fetch_auto_falls_back_when_bw_fails(tmp_path, monkeypatch):
    tiles = fake_glo30_tiles(tmp_path, 9.0, 48.55, AS, 360, 180, ntiles=1)
    monkeypatch.setattr(F, "copernicus_tile_urls", lambda bb: tiles)
    monkeypatch.setattr(F, "overpass", lambda q: ({"elements": []}, "stub"))

    def fail(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(F, "fetch_bw_dgm1", fail)
    d = F.fetch(square_boundary(48.52, 9.06, 100), tmp_path / "cache", dem_source="auto")
    m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert m["dem_source_id"] == "copernicus" and "network down" in m["dem_fallback_reason"]
    with pytest.raises(RuntimeError):
        F.fetch(square_boundary(48.52, 9.06, 100), tmp_path / "cache2", dem_source="bw_dgm1")


def test_unknown_source_rejected(tmp_path):
    with pytest.raises(ValueError):
        F.fetch(square_boundary(48.52, 9.06, 100), tmp_path, dem_source="srtm")


def test_file_source_lgl_xyz(tmp_path):
    """Manual fallback: an LGL DGM1 .xyz (no CRS) works directly via file:<path>, on its 1 m grid."""
    p = tmp_path / "dgm1_32_504_5373_1_bw_2020.xyz"
    rows = []
    for r in range(400):
        y = 5374199.5 - r
        rows.append("\n".join(f"{504000.5 + c:.2f} {y:.2f} {300 + 0.01 * c + 0.02 * r:.2f}" for c in range(400)))
    p.write_text("\n".join(rows) + "\n")
    out = tmp_path / "dem.tif"
    info = F.fetch_file_dem(str(p), (48.5185, 9.0554, 48.5201, 9.0576), out)
    assert any("assumed EPSG:25832" in w for w in info["warnings"])
    with rasterio.open(out) as ds:
        t, b = ds.transform, ds.read(1)
        assert ds.crs.to_epsg() == 25832 and t.a == 1.0 and t.c == round(t.c) and t.f == round(t.f)
    r, c = int(5374200 - t.f), int(t.c - 504000)
    assert b[0, 0] == pytest.approx(300 + 0.01 * c + 0.02 * r, abs=1e-3)
