"""M4-A extreme terrain / infrastructure: synthetic caches + real fixtures (tests/fixtures/m4, fetched 2026-11-15)."""
import gzip
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from osmctx.convert import Converter, convert
from test_convert import L, make_cache, node, way  # noqa: E402

DSM_META = {"dem_source": "Copernicus GLO-30 (copernicus-dem-30m.s3.amazonaws.com)"}  # legacy meta -> DSM kind
FIX = Path(__file__).resolve().parent / "fixtures" / "m4"


def cache(tmp_path, els, dem_fn, meta_extra=DSM_META, **kw):
    kw.setdefault("margin", 90.0)
    d = make_cache(tmp_path / "cache", els, dem_fn, **kw)
    m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    m.update(meta_extra)
    (d / "meta.json").write_text(json.dumps(m), encoding="utf-8")
    return d


def run(tmp_path, els, dem_fn, settings=None, meta_extra=DSM_META, **kw):
    return convert(cache(tmp_path, els, dem_fn, meta_extra, **kw), settings, tmp_path / "out")


def xy_fn(f):
    def g(lon, lat):
        x, y = L.xy(lon, lat)
        return f(np.asarray(x), np.asarray(y))
    return g


def sea_polys(site):
    return [Polygon(w["polygon"], w.get("holes") or []) for w in site["water"] if w["kind"] == "coastline"]


# --------------------------------------------------------------------------- coastline / sea

def coast_dem(x, y):
    """land 20 m north of y=0, sea 0 m south (+ a -4 m void cell artefact in the sea)."""
    z = np.where(y > 0, 20.0, 0.0)
    return np.where((np.abs(x - 30) < 15) & (np.abs(y + 30) < 15), -4.0, z)


def test_coastline_sea_on_the_right_z0_and_origin_on_land(tmp_path):
    els = [way(1, [(-200, 0), (200, 0)], {"natural": "coastline"})]  # W->E: land left (north), sea right (south)
    site = run(tmp_path, els, xy_fn(coast_dem), size=100)
    seas = sea_polys(site)
    assert seas and all(p.bounds[3] <= 0.01 for p in seas)
    w = [w for w in site["water"] if w["kind"] == "coastline"][0]
    e0 = site["origin"]["elevation"]
    assert w["z"] + e0 == pytest.approx(0.0, abs=1e-6) and w["z_source"] == "sea_level"
    assert w["vertical_datum"] == "EGM2008"
    assert e0 >= -0.01  # the -4 m void cell in the sea does not drag the origin down
    assert site["report"]["info"]["sea"]["present"]


def test_coastline_reversed_puts_sea_north(tmp_path):
    els = [way(1, [(200, 0), (-200, 0)], {"natural": "coastline"})]
    site = run(tmp_path, els, xy_fn(lambda x, y: np.where(y < 0, 20.0, 0.0)), size=100)
    seas = sea_polys(site)
    assert seas and all(p.bounds[1] >= -0.01 for p in seas)


@pytest.mark.parametrize("direction,expect_sea", [(1, False), (-1, True)])
def test_coastline_outside_the_site(tmp_path, direction, expect_sea):
    # coast 80 m south of a 100 m site: W->E -> site is land; E->W -> site is all sea
    pts = [(-300, -80), (300, -80)][::direction]
    site = run(tmp_path, [way(1, pts, {"natural": "coastline"})], xy_fn(lambda x, y: 5.0 + 0 * x), size=100)
    seas = sea_polys(site)
    assert bool(seas) == expect_sea
    if expect_sea:
        assert sum(p.area for p in seas) == pytest.approx(100 * 100, rel=0.02)


def test_river_mouth_clipped_by_sea_and_sea_linked(tmp_path):
    els = [way(1, [(-200, 0), (200, 0)], {"natural": "coastline"}),
           way(2, [(0, 80), (0, -60)], {"waterway": "river"})]  # river drawn into the sea
    dem = xy_fn(lambda x, y: np.where(y > 0, np.where(np.abs(x) < 10, -2.0, 6.0), 0.0))
    site = run(tmp_path, els, dem, size=120)
    sea = sea_polys(site)[0]
    riv = [w for w in site["water"] if w["kind"] == "river"]
    assert riv
    for r in riv:
        assert Polygon(r["polygon"]).intersection(sea).area < 0.5
        assert r.get("sea_linked") is True
        assert r["z"] + site["origin"]["elevation"] >= -1e-6  # sea-linked water never below sea level
    assert site["report"]["counts"]["water_clipped_by_sea"] == 1


# --------------------------------------------------------------------------- bare earth

def test_bare_earth_skips_when_only_sea_cells_remain(tmp_path):
    """Uluwatu: forest covers the whole headland, the rest is sea -> no land reference -> keep the clifftop."""
    els = [way(1, [(-300, -40), (300, -40)], {"natural": "coastline"}),
           way(2, [(-300, -40), (300, -40), (300, 300), (-300, 300), (-300, -40)], {"natural": "wood"})]
    dem = xy_fn(lambda x, y: np.where(y > -40, 70.0, 0.0))
    site = run(tmp_path, els, dem, size=160)
    g = site["report"]["info"]["ground"]
    assert g["method"] == "skipped_too_few_land_refs" and not g["applied"]
    zmax = max(p[2] for p in site["terrain"]["points"]) + site["origin"]["elevation"]
    assert zmax > 65.0
    assert any("SKIPPED" in w for w in site["report"]["warnings"])


def test_bare_earth_lowering_cap_for_buildings(tmp_path):
    bld = [(-40, -40), (40, -40), (40, 40), (-40, 40), (-40, -40)]
    els = [way(1, bld, {"building": "yes", "height": "5"})]

    def dem(lon, lat):
        x, y = L.xy(lon, lat)
        return 100.0 + np.where((np.abs(x) < 40) & (np.abs(y) < 40), 40.0, 0.0)
    site = run(tmp_path, els, dem, size=200)
    g = site["report"]["info"]["ground"]
    assert g["applied"] and g["lowering_m"]["max"] == pytest.approx(10.0, abs=0.01)  # 2 x 5 m
    assert g["cells_lowering_capped"] > 0


# --------------------------------------------------------------------------- buildings

def test_building_in_water_flagged_and_not_below_water(tmp_path):
    lake = [(-60, -60), (60, -60), (60, 60), (-60, 60), (-60, -60)]
    els = [way(1, lake, {"natural": "water"}), way(2, [(-5, -5), (5, -5), (5, 5), (-5, 5), (-5, -5)], {"building": "hut"})]
    dem = xy_fn(lambda x, y: np.where((np.abs(x) < 60) & (np.abs(y) < 60), 3.0, 8.0))
    site = run(tmp_path, els, dem, size=200)
    b = site["buildings"][0]
    wz = [w for w in site["water"] if w["kind"] == "water"][0]["z"]
    assert b["tags"]["osmctx:in_water"] == "yes" and b["base_z"] >= wz - 1e-6
    assert site["report"]["counts"]["buildings_in_water"] == 1


def test_steep_footprint_flagged(tmp_path):
    els = [way(1, [(-20, -20), (20, -20), (20, 20), (-20, 20), (-20, -20)], {"building": "yes"})]
    site = run(tmp_path, els, xy_fn(lambda x, y: 100.0 + 0.5 * y), size=120, meta_extra={})
    b = site["buildings"][0]
    assert b["base_spread"] == pytest.approx(20.0, abs=0.5)
    assert float(b["tags"]["osmctx:steep_footprint"]) == pytest.approx(20.0, abs=0.5)
    assert any("osmctx:steep_footprint" in w for w in site["report"]["warnings"])


# --------------------------------------------------------------------------- land cover / peaks / cliffs

def test_land_cover_beach_and_tidalflat(tmp_path):
    els = [way(1, [(-50, -50), (50, -50), (50, -20), (-50, -20), (-50, -50)], {"natural": "beach", "surface": "sand"}),
           way(2, [(-50, 0), (50, 0), (50, 30), (-50, 30), (-50, 0)], {"natural": "wetland", "wetland": "tidalflat"})]
    site = run(tmp_path, els, xy_fn(lambda x, y: 2.0 + 0 * x), size=120)
    kinds = sorted(lc["kind"] for lc in site["land_cover"])
    assert kinds == ["beach", "tidalflat"]
    assert site["report"]["counts"]["land_cover"] == 2
    assert "beach_not_imported" not in site["report"]["counts"]


def test_peak_is_data_not_terrain(tmp_path):
    els = [node(1, 0, 0, {"natural": "peak", "ele": "8848.86", "name": "Top"})]
    site = run(tmp_path, els, xy_fn(lambda x, y: 8711.0 - 0.01 * (x * x + y * y) ** 0.5), size=100)
    poi = site["points_of_interest"][0]
    assert poi["ele"] == pytest.approx(8848.86) and poi["dem_minus_ele"] == pytest.approx(-137.86, abs=0.5)
    assert max(p[2] for p in site["terrain"]["points"]) + site["origin"]["elevation"] < 8712  # DEM not reshaped
    assert any("surveyed" in w and "NOT reshaped" in w for w in site["report"]["warnings"])


def test_cliff_export_and_steep_slope_note(tmp_path):
    els = [way(1, [(-100, 0), (100, 0)], {"natural": "cliff"})]
    site = run(tmp_path, els, xy_fn(lambda x, y: np.where(y > 0, 70.0, 0.0)), size=150, meta_extra={})
    assert len(site["cliffs"]) == 1 and len(site["cliffs"][0]["points"]) >= 2
    sl = site["report"]["dem_accuracy"]["slope"]
    assert sl["cells_steep_pct"] > 0 and sl["max_pct"] > 100
    assert "Steep terrain" in site["report"]["dem_accuracy"]["note"]
    assert any("natural=cliff" in w for w in site["report"]["warnings"])


# --------------------------------------------------------------------------- bridges / interchange

def z_at(road, x, y):
    ln = LineString([p[:2] for p in road["points"]])
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(np.asarray(road["points"])[:, :2], axis=0).T))])
    return float(np.interp(ln.project(__import__("shapely").geometry.Point(x, y)), s, [p[2] for p in road["points"]]))


def test_three_level_stack_ordered_with_clearance(tmp_path):
    els = [way(1, [(0, -150), (0, 150)], {"highway": "primary"}, nodes=[1, 2]),
           way(2, [(-150, 0), (-120, 0)], {"highway": "primary"}, nodes=[10, 11]),
           way(3, [(-120, 0), (120, 0)], {"highway": "primary", "bridge": "yes", "layer": "1"}, nodes=[11, 12]),
           way(4, [(120, 0), (150, 0)], {"highway": "primary"}, nodes=[12, 13]),
           way(5, [(-150, -150), (-130, -130)], {"highway": "primary"}, nodes=[20, 21]),
           way(6, [(-130, -130), (130, 130)], {"highway": "primary", "bridge": "yes", "layer": "2"}, nodes=[21, 22]),
           way(7, [(130, 130), (150, 150)], {"highway": "primary"}, nodes=[22, 23])]
    site = run(tmp_path, els, xy_fn(lambda x, y: 50.0 + 0 * x), size=300)
    r = {rr["osm_id"]: rr for rr in site["roads"]}
    z0, z1, z2 = z_at(r["way/1"], 0, 0), z_at(r["way/3"], 0, 0), z_at(r["way/6"], 0, 0)
    assert z1 >= z0 + 5.5 - 0.02 and z2 >= z1 + 5.5 - 0.02
    lay = site["report"]["info"]["layers"]
    assert lay["inverted"] == 0 and lay["below_clearance"] == 0 and lay["crossings"] == 3
    for rr in (r["way/3"], r["way/6"]):  # long approaches: the 8 % cap is met
        p = np.asarray(rr["points"])
        g = np.abs(np.diff(p[:, 2])) / np.hypot(*np.diff(p[:, :2], axis=0).T)
        assert g.max() <= 0.08 + 1e-3


def test_mixed_layer_chain_never_inverted(tmp_path):
    # chain A: L1 way then L2 way; chain B (L1) passes under A's L2 part. Both chains have min layer 1.
    els = [way(1, [(-160, 20), (-140, 20)], {"highway": "secondary"}, nodes=[1, 2]),
           way(2, [(-140, 20), (0, 20)], {"highway": "secondary", "bridge": "yes", "layer": "1"}, nodes=[2, 3]),
           way(3, [(0, 20), (140, 20)], {"highway": "secondary", "bridge": "yes", "layer": "2"}, nodes=[3, 4]),
           way(4, [(140, 20), (160, 20)], {"highway": "secondary"}, nodes=[4, 5]),
           way(5, [(60, -160), (60, -140)], {"highway": "secondary"}, nodes=[6, 7]),
           way(6, [(60, -140), (60, 140)], {"highway": "secondary", "bridge": "yes", "layer": "1"}, nodes=[7, 8]),
           way(7, [(60, 140), (60, 160)], {"highway": "secondary"}, nodes=[8, 9]),
           way(8, [(-160, -60), (160, -60)], {"highway": "secondary"}, nodes=[10, 11])]  # ground road under B
    site = run(tmp_path, els, xy_fn(lambda x, y: 10.0 + 0 * x), size=300)
    r = {rr["osm_id"]: rr for rr in site["roads"]}
    assert z_at(r["way/3"], 60, 20) >= z_at(r["way/6"], 60, 20) + 5.5 - 0.02
    assert z_at(r["way/6"], 60, -60) >= z_at(r["way/8"], 60, -60) + 5.5 - 0.02
    assert site["report"]["info"]["layers"]["inverted"] == 0


def test_bridge_ignores_dsm_deck_bump(tmp_path):
    # a DSM 'sees' the deck as a 12 m bump; the bridge must stay on its abutment gradeline (never draped)
    els = [way(1, [(-150, 0), (-80, 0)], {"highway": "tertiary"}, nodes=[1, 2]),
           way(2, [(-80, 0), (80, 0)], {"highway": "tertiary", "bridge": "yes"}, nodes=[2, 3]),
           way(3, [(80, 0), (150, 0)], {"highway": "tertiary"}, nodes=[3, 4])]
    site = run(tmp_path, els, xy_fn(lambda x, y: 30.0 + np.where((np.abs(x) < 60) & (np.abs(y) < 8), 12.0, 0.0)),
               size=300, settings={"ground_mode": "dsm"})
    br = [rr for rr in site["roads"] if rr["bridge"]][0]
    zs = [p[2] for p in br["points"]]
    assert max(zs) - min(zs) < 0.05


def test_branching_bridge_junction_is_not_a_ground_abutment(tmp_path):
    # Y of three bridge ways meeting mid-air over a 30 m valley: the junction must not be pinned to the valley floor
    els = [way(1, [(-160, 0), (-120, 0)], {"highway": "primary"}, nodes=[1, 2]),
           way(2, [(-120, 0), (0, 0)], {"highway": "primary", "bridge": "yes"}, nodes=[2, 3]),
           way(3, [(0, 0), (120, 0)], {"highway": "primary", "bridge": "yes"}, nodes=[3, 4]),
           way(4, [(120, 0), (160, 0)], {"highway": "primary"}, nodes=[4, 5]),
           way(5, [(0, 0), (0, 120)], {"highway": "primary_link", "bridge": "yes"}, nodes=[3, 6]),
           way(6, [(0, 120), (0, 160)], {"highway": "primary_link"}, nodes=[6, 7])]
    valley = xy_fn(lambda x, y: 100.0 - 30.0 * np.exp(-((x / 40.0) ** 2 + (y / 40.0) ** 2)))
    site = run(tmp_path, els, valley, size=300, settings={"ground_mode": "dsm"})
    r = {rr["osm_id"]: rr for rr in site["roads"]}
    e0 = site["origin"]["elevation"]
    assert z_at(r["way/2"], 0, 0) + e0 > 95.0
    assert abs(z_at(r["way/2"], 0, 0) - z_at(r["way/5"], 0, 0)) < 0.05


# --------------------------------------------------------------------------- real fixtures (2026-11-15)

def real(tmp_path, name, settings=None):
    src = FIX / name
    d = tmp_path / name
    d.mkdir()
    (d / "osm.json").write_bytes(gzip.decompress((src / "osm.json.gz").read_bytes()))
    for fn in ("dem.tif", "dem_ext.tif", "meta.json"):
        if (src / fn).exists():
            shutil.copy2(src / fn, d / fn)
    return Converter(d, settings).run()


def test_real_uluwatu_keeps_clifftop(tmp_path):
    s = real(tmp_path, "uluwatu")
    r = s["report"]
    assert r["info"]["ground"]["method"] == "skipped_too_few_land_refs"
    assert r["info"]["terrain"]["z_abs_max"] > 60.0  # was 0.0 before M4-A (cliff planed to sea level)
    assert any(w["kind"] == "coastline" and w["z_source"] == "sea_level" for w in s["water"])
    assert s["cliffs"] and r["dem_accuracy"]["slope"]["cells_steep_pct"] > 5
    assert r["counts"]["buildings_steep_footprint"] >= 1


def test_real_everest_peak(tmp_path):
    s = real(tmp_path, "everest")
    p = s["points_of_interest"][0]
    assert p["ele"] == pytest.approx(8848.86) and p["dem_minus_ele"] < -100
    assert s["origin"]["elevation"] > 8500


def test_real_huangjuewan_layers(tmp_path):
    s = real(tmp_path, "huangjuewan")
    lay = s["report"]["info"]["layers"]
    assert lay["crossings"] >= 50 and lay["inverted"] == 0 and lay["below_clearance"] == 0  # was 5 inverted
    assert s["report"]["counts"].get("bridge_ends_cut", 0) == 0  # chains completed beyond the bbox


def test_real_ampera_water_and_origin(tmp_path):
    s = real(tmp_path, "ampera")
    e0 = s["origin"]["elevation"]
    assert e0 >= -0.01  # was -5.37 (GLO-30 artefact)
    river = [w for w in s["water"] if w["osm_id"] == "relation/3995667"][0]
    assert river["z"] + e0 == pytest.approx(0.0, abs=0.01) and river["z_source"] == "dem_flat_water"  # was -2.08
    c = s["report"]["counts"]
    assert c.get("bridge_ends_cut", 0) == 0 and c.get("bridge_ends_clamped", 0) == 0
    assert c["buildings_in_water"] == 1
    for rr in s["roads"]:
        if rr["bridge"]:
            assert min(p[2] for p in rr["points"]) + e0 > 2.0  # deck above the 0 m river


def test_real_parangtritis_mouth_and_depok_beach(tmp_path):
    s = real(tmp_path, "parangtritis")
    sea = [Polygon(w["polygon"]) for w in s["water"] if w["kind"] == "coastline"]
    assert sea
    for w in s["water"]:
        if w["kind"] != "coastline":
            assert all(Polygon(w["polygon"]).intersection(q).area < 1.0 for q in sea)
    d = real(tmp_path, "parangtritis_depok")
    assert [lc["kind"] for lc in d["land_cover"]] == ["beach"]
