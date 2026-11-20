"""Offline tests: ground_mode (bare_earth_approx on a DSM) and dem_accuracy reporting."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

from osmctx.convert import Converter, convert
from test_convert import L, make_cache, plane, way, node  # noqa: E402

BLD = [(-30, -30), (30, -30), (30, 30), (-30, 30), (-30, -30)]
DSM_META = {"dem_source": "Copernicus GLO-30 (copernicus-dem-30m.s3.amazonaws.com)"}  # legacy M1 meta
DTM_META = {"dem_source": "LGL Baden-Wuerttemberg DGM1 (opengeodata.lgl-bw.de)", "dem_source_id": "bw_dgm1",
            "dem_accuracy": {"horizontal_m": 1.0, "vertical_m": 0.5, "kind": "DTM", "note": "test"},
            "dem_attribution": "LGL-BW (2026) Datenlizenz Deutschland - Namensnennung - Version 2.0, www.lgl-bw.de",
            "dem_crs": "EPSG:4326"}


def bumped(ground, h=15.0, half=30.0):
    """ground(lon, lat) + h inside the building footprint square (a DSM 'sees' the roof)."""
    def f(lon, lat):
        x, y = L.xy(lon, lat)
        return ground(lon, lat) + np.where((np.abs(x) < half) & (np.abs(y) < half), h, 0.0)
    return f


def flat200(lon, lat):
    return np.full_like(lon, 200.0)


def cache(tmp_path, els, dem_fn, meta_extra):
    d = make_cache(tmp_path / "cache", els, dem_fn, margin=90.0)
    m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    m.update(meta_extra)
    (d / "meta.json").write_text(json.dumps(m), encoding="utf-8")
    return d


def test_bare_earth_removes_building_bump_flat(tmp_path):
    els = [way(1, BLD, {"building": "yes"})]
    d = cache(tmp_path, els, bumped(flat200), DSM_META)
    site = convert(d, None, tmp_path / "out")
    g = site["report"]["info"]["ground"]
    assert g["mode"] == "bare_earth_approx" and g["requested"] == "auto" and g["applied"]
    assert g["cells_masked"] > 0 and g["cells_masked"] < g["cells_total"]
    assert g["lowering_m"]["max"] == pytest.approx(15.0, abs=0.01)
    assert site["origin"]["elevation"] == pytest.approx(200.0, abs=0.01)
    assert site["buildings"][0]["base_z"] == pytest.approx(0.0, abs=0.01)
    assert max(p[2] for p in site["terrain"]["points"]) == pytest.approx(0.0, abs=0.01)
    acc = site["report"]["dem_accuracy"]
    assert acc["kind"] == "DSM" and acc["ground_mode"] == "bare_earth_approx" and acc["horizontal_m"] == 30.0
    assert any("APPROXIMATION" in w for w in site["report"]["warnings"])


def test_bare_earth_plane_fit_on_slope(tmp_path):
    els = [way(1, BLD, {"building": "yes"})]
    d = cache(tmp_path, els, bumped(plane), DSM_META)
    site = convert(d, None, tmp_path / "out")
    e0 = site["origin"]["elevation"]
    corners = [plane(*L.ll(x, y)) for x, y in BLD[:-1]]
    assert site["buildings"][0]["base_z"] + e0 == pytest.approx(min(corners), abs=0.05)


def test_dsm_mode_keeps_bump(tmp_path):
    els = [way(1, BLD, {"building": "yes"})]
    d = cache(tmp_path, els, bumped(flat200), DSM_META)
    site = convert(d, {"ground_mode": "dsm"}, tmp_path / "out")
    assert site["report"]["info"]["ground"]["applied"] is False
    assert max(p[2] for p in site["terrain"]["points"]) > 10.0  # roof bump stays in the terrain
    assert any("may float" in w for w in site["report"]["warnings"])


def test_dtm_default_is_as_is(tmp_path):
    els = [way(1, BLD, {"building": "yes"})]
    d = cache(tmp_path, els, bumped(flat200), DTM_META)
    site = convert(d, None, tmp_path / "out")
    r = site["report"]
    assert r["info"]["ground"]["mode"] == "dsm" and not r["info"]["ground"]["applied"]
    assert r["dem_accuracy"]["kind"] == "DTM" and r["dem_accuracy"]["vertical_m"] == 0.5
    assert r["info"]["dem"]["attribution"].startswith("LGL-BW (2026)")


def test_bare_earth_never_raises_and_masks_trees_and_wood(tmp_path):
    rng = np.random.default_rng(1)

    def noisy(lon, lat):
        return 250.0 + rng.normal(0, 3, size=np.shape(lon))
    els = [way(1, BLD, {"building": "yes"}), node(2, 70, 70, {"natural": "tree"}),
           way(3, [(-100, 60), (-60, 60), (-60, 100), (-100, 100), (-100, 60)], {"natural": "wood"})]
    d = cache(tmp_path, els, noisy, DSM_META)
    conv = Converter(d, None)
    raw = conv.dem.a.copy()
    conv.run()
    assert (conv.dem.a <= raw + 1e-9).all()
    mf = conv.ground_info["mask_features"]
    assert mf == {"buildings": 1, "wood_forest": 1, "trees": 1, "water": 0}


def test_bad_ground_mode(tmp_path):
    d = cache(tmp_path, [], flat200, DSM_META)
    with pytest.raises(ValueError):
        Converter(d, {"ground_mode": "dtm_please"})
