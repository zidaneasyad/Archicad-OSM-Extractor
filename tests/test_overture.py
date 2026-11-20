"""Offline tests for osmctx.overture (conflation, OSM-source skip, height chain, credits) and the converter / builder
plumbing. Fixtures are real caches saved from live queries (Overture release 2026-10-21.0):
  overture_kulonprogo  -7.85475, 110.17125, 150 m (rural Java: 0 OSM buildings inside, 129 Overture gap buildings)
  overture_kamal_muara -6.095053, 106.719792, 150 m (Jakarta; Overture mostly duplicates OSM)"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from shapely.geometry import box

from osmctx import build_plan as bp
from osmctx import limits
from osmctx import overture as ovt
from osmctx.convert import convert

FIX = Path(__file__).parent / "fixtures"
KUL, KAM = FIX / "overture_kulonprogo", FIX / "overture_kamal_muara"


def props(dataset, **kw):
    return {"id": "x", "sources": [{"dataset": dataset, "license": None}], **kw}


# --------------------------------------------------------------------------- pure conflation
def test_overlap_rule_threshold_is_configurable():
    osm = [box(0, 0, 10, 10)]
    ml = props("Microsoft ML Buildings")
    cand19 = box(-8.1, 0, 1.9, 10)    # 10 x 10, 1.9 m x 10 m (19 %) on the OSM building -> kept
    cand21 = box(-7.9, 0, 2.1, 10)    # 21 % on the OSM building -> dropped
    kept, st = ovt.conflate([(cand19, ml), (cand21, ml)], osm, 0.2)
    assert [k[0] for k in kept] == [cand19] and st["skipped_overlap"] == 1 and st["kept"] == 1
    kept2, _ = ovt.conflate([(cand21, ml)], osm, 0.25)
    assert len(kept2) == 1


def test_no_overlap_kept_and_no_osm_means_all_kept():
    kept, st = ovt.conflate([(box(20, 20, 30, 30), props("Google Open Buildings"))], [box(0, 0, 10, 10)])
    assert len(kept) == 1
    kept, st = ovt.conflate([(box(0, 0, 5, 5), props("Google Open Buildings"))], [])
    assert len(kept) == 1


def test_osm_sourced_record_is_skipped_even_without_overlap():
    kept, st = ovt.conflate([(box(50, 50, 60, 60), props("OpenStreetMap"))], [box(0, 0, 10, 10)])
    assert kept == [] and st["skipped_osm_source"] == 1
    mixed = {"id": "m", "sources": [{"dataset": "Microsoft ML Buildings"}, {"dataset": "OpenStreetMap"}]}
    assert ovt.is_osm_sourced(mixed)


def test_height_chain():
    assert ovt.height_for({"height": 12.5, "num_floors": 3}, 3.2, 3) == (12.5, "overture_height")
    assert ovt.height_for({"height": None, "num_floors": 4}, 3.0, 3) == (12.0, "overture_floors")
    assert ovt.height_for({"height": 0.2, "num_floors": None}, 3.2, 3) == (pytest.approx(9.6), "default")
    assert ovt.height_for({}, 3.2, 3)[1] == "default"


def test_estimated_flag_and_source_tag():
    assert ovt.is_estimated(props("Microsoft ML Buildings")) and ovt.is_estimated(props("Google Open Buildings"))
    assert not ovt.is_estimated(props("Esri Community Maps"))
    assert ovt.source_tag(props("Google Open Buildings")) == "overture:Google Open Buildings"


def test_credits_use_verified_licences():
    c = ovt.buildings_credit(["OpenStreetMap", "Google Open Buildings", "Microsoft ML Buildings"])
    assert "OpenStreetMap contributors (ODbL)" in c and "Google Open Buildings (CC BY 4.0)" in c
    assert "Microsoft" in c and "ODbL" in c and "CDLA" not in c
    assert ovt.THEME_LICENCE == "ODbL-1.0"


def test_select_files_by_stac_bbox():
    files = [f"s3://b/x/part-{i:05d}-abc-c000.zstd.parquet" for i in range(3)]
    boxes = [[-180, -90, -100, 0], [100, -10, 120, 10], [100, 20, 120, 30]]
    assert ovt.select_files(files, boxes, (-6.1, 106.7, -6.09, 106.72)) == [files[1]]
    with pytest.raises(ovt.OvertureError):
        ovt.select_files(files, boxes[:2], (0, 0, 1, 1))


# --------------------------------------------------------------------------- real fixtures
def test_fixture_osm_source_skip_matches_data():
    feats = ovt.load(KUL / ovt.FILE)
    osm_src = [f for f in feats if ovt.is_osm_sourced(f["properties"])]
    assert len(feats) == 162 and len(osm_src) == 33


def _convert(cache, tmp_path, **inc):
    return convert(cache, {"include": {"shrubs": False, **inc}}, out_dir=str(tmp_path))


def test_rural_site_gets_overture_buildings(tmp_path):
    site = _convert(KUL, tmp_path)
    c = site["report"]["counts"]
    assert c["buildings_osm"] == 0 and c["buildings_overture"] == 65 and c["buildings"] == 65
    ov = site["report"]["info"]["overture"]
    assert ov["applied"] and ov["by_dataset"] == {"Google Open Buildings": 49, "Microsoft ML Buildings": 16}
    b = site["buildings"][0]
    assert b["osm_id"].startswith("overture/") and b["source"].startswith("overture:") and b["estimated"] is True
    assert b["height_source"] == "default" and b["height"] == pytest.approx(9.6)
    assert all(x["footprint"] and "source" in x for x in site["buildings"])
    assert "Google Open Buildings (CC BY 4.0)" in ov["attribution"]


def test_overture_off_gives_osm_only(tmp_path):
    site = _convert(KUL, tmp_path, overture_buildings=False)
    assert site["report"]["counts"]["buildings"] == 0
    assert site["report"]["info"]["overture"]["applied"] is False
    assert all("source" not in b for b in site["buildings"])


def test_dense_osm_site_skips_duplicates(tmp_path):
    site = _convert(KAM, tmp_path)
    c = site["report"]["counts"]
    assert c["buildings_osm"] == 4 and c["buildings_overture"] == 2
    st = site["report"]["info"]["overture"]["stats"]
    assert st["skipped_osm_source"] == 11 and st["candidates"] == 27


def test_missing_overture_file_never_blocks(tmp_path):
    d = tmp_path / "cache"
    shutil.copytree(KUL, d)
    (d / ovt.FILE).unlink()
    site = convert(d, {"include": {"shrubs": False}}, out_dir=str(tmp_path / "out"))
    assert site["report"]["info"]["overture"]["applied"] is False
    assert any("Overture buildings not used" in w for w in site["report"]["warnings"])


# --------------------------------------------------------------------------- builder plan
def test_group_plan_overture_subgroup():
    created = {"buildings": [("g1", "way/1"), ("g2", "overture/a"), ("g3", "overture/b"), ("g4", "overture/c")],
               "terrain": [("t", "terrain")]}
    plan = bp.group_plan(created)
    blds = next(k for k in plan["children"] if isinstance(k, dict) and k["name"] == "Buildings")
    ov = [k for k in blds["children"] if isinstance(k, dict)]
    assert [o["name"] for o in ov] == ["Buildings (Overture)"] and ov[0]["children"] == ["g2", "g3", "g4"]


def test_group_plan_single_overture_building_dissolves():
    plan = bp.group_plan({"buildings": [("g1", "way/1"), ("g2", "overture/a")], "terrain": [("t", "x")]})
    blds = next(k for k in plan["children"] if isinstance(k, dict) and k["name"] == "Buildings")
    assert blds["children"] == ["g1", "g2"]


def test_building_source_property():
    assert bp.building_source({"source": "overture:Google Open Buildings"}, "OSM / x") == "Overture Maps (Google Open Buildings)"
    assert bp.building_source({}, "OSM / x") == "OSM / x"


# --------------------------------------------------------------------------- estimate / server
def test_estimate_counts_overture_buildings():
    tiles = {"source": "copernicus", "dgm1_total": 0, "dgm1_missing": 0, "ndom_total": 0, "ndom_missing": 0,
             "copernicus_tiles": 1}
    cnt = {"buildings": 4, "roads": 2, "trees": 0, "water": 0, "overture_buildings": 65}
    on = limits.estimate(0.02, cnt, "cache", tiles, settings={"include": {"overture_buildings": True}})
    off = limits.estimate(0.02, cnt, "cache", tiles, settings={"include": {"overture_buildings": False}})
    assert on["elements"]["buildings"] == 69 and on["elements"]["buildings_overture"] == 65
    assert off["elements"]["buildings"] == 4 and on["elements"]["total"] == off["elements"]["total"] + 65


def test_server_count_and_disclaimer():
    import server
    assert server.count_overture(KUL, None) == 129      # non-OSM-sourced records, before the overlap rule
    d = server.disclaimer()
    assert any("Overture" in c for c in d["credits"])
    off = server.disclaimer(None, None, None, {"applied": False})
    assert not any("Overture" in c for c in off["credits"])
    on = server.disclaimer(None, None, None, {"applied": True, "attribution": "Buildings: X"})
    assert "Buildings: X" in on["credits"]
