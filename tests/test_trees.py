"""Offline tests for tree sizing: osmctx/trees.py, converter tree fields, build_plan tree objects."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from osmctx import build_plan as bp
from osmctx import trees as tl
from osmctx.boundary import square_boundary, to_polygon
from osmctx.convert import DEFAULT_SETTINGS, Frame, _merge_settings, convert
from test_convert import LAT0, LON0, flat, make_cache, node, run, way

S = _merge_settings(None)


# ------------------------------------------------------------------ data module
def test_genus_table_covers_requested_genera_and_is_sane():
    want = ("Platanus Tilia Acer Quercus Fagus Aesculus Betula Fraxinus Carpinus Prunus Malus Robinia Populus "
            "Salix Pinus Picea Abies Larix Taxus Thuja Ficus Mangifera Swietenia Tectona Cocos Samanea "
            "Pterocarpus Delonix").split()
    for g in want:
        h, r, lt = tl.GENUS_TABLE[g]
        assert 3 <= h <= 40 and 0.2 <= r <= 1.5 and lt in ("broadleaved", "needleleaved")
    assert {"Pinus", "Picea", "Abies", "Larix", "Taxus", "Thuja"} <= tl.CONIFER_GENERA
    assert "Quercus" not in tl.CONIFER_GENERA


def test_taxon_parsing():
    assert tl.taxon({"genus": "tilia"}) == ("Tilia", None)
    assert tl.taxon({"species": "Tilia cordata"}) == ("Tilia", "Tilia cordata")
    assert tl.taxon({"species": "Platanus x hispanica"}) == ("Platanus", None)
    assert tl.taxon({"species": "Acer platanoides;Acer campestre"}) == ("Acer", "Acer platanoides")
    assert tl.taxon({"species": "???"}) == (None, None)
    assert tl.taxon({}) == (None, None)


def test_lookup_species_over_genus():
    assert tl.lookup({"species": "Acer pseudoplatanus"}) == (25.0, 0.6)
    assert tl.lookup({"genus": "Acer"}) == (15.0, 0.6)
    assert tl.lookup({"species": "Acer unknownus"}) == (15.0, 0.6)  # falls to genus
    assert tl.lookup({"genus": "Nothofagus"}) is None


# ------------------------------------------------------------------ fallback chain
def test_chain_tag_wins_and_crown_tag():
    r = tl.size_tree({"genus": "Tilia"}, 17.0, 9.0, 30.0, S)
    assert (r["height"], r["height_source"], r["crown_diameter"], r["crown_source"]) == (17.0, "tag", 9.0, "tag")


def test_chain_measured_beats_genus():
    r = tl.size_tree({"genus": "Tilia"}, None, None, 14.0, S)
    assert r["height"] == 14.0 and r["height_source"] == "measured"
    assert r["crown_diameter"] == pytest.approx(14.0 * 0.6) and r["crown_source"] == "ratio"  # genus ratio


def test_chain_genus_then_leaf_type_then_default():
    r = tl.size_tree({"genus": "Platanus", "leaf_type": "broadleaved"}, None, None, None, S)
    assert (r["height"], r["height_source"]) == (25.0, "genus") and r["crown_diameter"] == pytest.approx(15.0)
    r = tl.size_tree({"leaf_type": "broadleaved"}, None, None, None, S)
    assert (r["height"], r["height_source"], r["crown_diameter"]) == (12.0, "leaf_type", pytest.approx(7.8))
    r = tl.size_tree({"leaf_type": "needleleaved"}, None, None, None, S)
    assert (r["height"], r["height_source"], r["crown_diameter"]) == (15.0, "leaf_type", pytest.approx(5.25))
    r = tl.size_tree({}, None, None, None, S)
    assert (r["height"], r["height_source"], r["crown_diameter"]) == (8.0, "default", pytest.approx(4.8))  # 0.6 unknown


def test_settings_override_and_genus_switch():
    s = _merge_settings({"broadleaf_height": 10, "broadleaf_crown_ratio": 0.5, "tree_height": 6})
    r = tl.size_tree({"leaf_type": "broadleaved"}, None, None, None, s)
    assert (r["height"], r["crown_diameter"]) == (10.0, 5.0)
    off = _merge_settings({"use_genus_table": False})
    r = tl.size_tree({"genus": "Platanus", "leaf_type": "broadleaved"}, None, None, None, off)
    assert (r["height"], r["height_source"], r["crown_diameter"]) == (12.0, "leaf_type", pytest.approx(7.8))


def test_conifer_genus_without_leaf_type_uses_conifer_ratio_when_table_off():
    off = _merge_settings({"use_genus_table": False})
    r = tl.size_tree({"genus": "Pinus"}, None, None, None, off)
    assert r["height_source"] == "default" and r["crown_diameter"] == pytest.approx(8.0 * 0.35)


def test_default_settings_values():
    for k, v in {"tree_height": 8.0, "broadleaf_height": 12.0, "conifer_height": 15.0,
                 "broadleaf_crown_ratio": 0.65, "conifer_crown_ratio": 0.35, "use_genus_table": True}.items():
        assert DEFAULT_SETTINGS[k] == v


# ------------------------------------------------------------------ measuring
def test_measure_height_radius_and_accept_window():
    g = np.arange(0, 10, 1.0)
    cx, cy = np.meshgrid(g + 0.5, g + 0.5)
    cx, cy = cx.ravel(), cy.ravel()
    v = np.zeros_like(cx)
    v[(cx == 5.5) & (cy == 5.5)] = 20.0
    v[(cx == 8.5) & (cy == 8.5)] = 40.0
    assert tl.measure_height(cx, cy, v, 5.0, 6.0) == 20.0           # 0.7 m away
    assert tl.measure_height(cx, cy, v, 7.0, 7.0) is None           # max within 2 m of (7,7) is 0 -> < 2 m
    assert tl.measure_height(cx, cy, v, 8.0, 8.0) == 40.0
    v[(cx == 5.5) & (cy == 5.5)] = 60.0                              # implausible -> rejected
    assert tl.measure_height(cx, cy, v, 5.0, 5.0) is None
    v[:] = np.nan
    assert tl.measure_height(cx, cy, v, 5.0, 5.0) is None
    assert tl.measure_height([], [], [], 0, 0) is None


def test_trunk_diameter():
    assert tl.trunk_diameter(1.2) == pytest.approx(0.382, abs=1e-3)
    assert tl.trunk_diameter(None) is None


# ------------------------------------------------------------------ converter integration
def _tree_els():
    return [node(1, 10, 10, {"natural": "tree", "height": "18", "diameter_crown": "9", "circumference": "120 cm",
                             "leaf_type": "broadleaved", "genus": "Tilia", "species": "Tilia cordata",
                             "leaf_cycle": "deciduous", "denotation": "avenue", "source": "x"}),
            node(2, -10, 10, {"natural": "tree", "genus": "Quercus"}),
            node(3, 10, -10, {"natural": "tree", "leaf_type": "needleleaved"}),
            node(4, -10, -10, {"natural": "tree", "circumference": "1,5"}),
            node(5, 30, 30, {"natural": "tree", "circumference": "junk", "height": "tall"})]


def test_converter_tree_fields_without_ndom(tmp_path):
    site = run(tmp_path, _tree_els(), flat())
    t = {x["osm_id"]: x for x in site["trees"]}
    a = t["node/1"]
    assert a["tags"] == {"leaf_type": "broadleaved", "leaf_cycle": "deciduous", "genus": "Tilia",
                         "species": "Tilia cordata", "denotation": "avenue"}
    assert (a["height"], a["height_source"], a["crown_diameter"], a["crown_source"]) == (18.0, "tag", 9.0, "tag")
    assert a["trunk_diameter"] == pytest.approx(1.2 / np.pi, abs=1e-3)
    assert (t["node/2"]["height"], t["node/2"]["height_source"]) == (22.0, "genus")
    assert t["node/2"]["crown_diameter"] == pytest.approx(22.0 * 0.7)
    assert (t["node/3"]["height"], t["node/3"]["height_source"]) == (15.0, "leaf_type")
    assert t["node/4"]["tags"] == {} and t["node/4"]["height_source"] == "default"
    assert t["node/4"]["trunk_diameter"] == pytest.approx(1.5 / np.pi, abs=1e-3)
    assert "trunk_diameter" not in t["node/5"] and t["node/5"]["height"] == 8.0
    d = site["report"]["defaults_used"]
    assert d["tree_heights"] == {"tag": 1, "genus": 1, "leaf_type": 1, "default": 2}
    assert d["tree_crowns"] == {"tag": 1, "ratio": 4}
    assert site["report"]["info"]["tree_measure"]["available"] is False
    assert "30 m" in site["report"]["info"]["tree_measure"]["reason"]


def test_converter_measured_height_from_ndom(tmp_path):
    els = [node(1, 5.0, 5.0, {"natural": "tree", "genus": "Quercus"}),         # measured 20 m
           node(2, -20.0, -20.0, {"natural": "tree", "height": "11"}),          # tag beats measurement
           node(3, 20.0, -20.0, {"natural": "tree"}),                           # nDOM 0 -> not accepted -> default
           node(4, -5.0, 25.0, {"natural": "tree", "leaf_type": "needleleaved"})]  # next to a building
    els.append(way(7, [(-12, 20), (-8, 20), (-8, 30), (-12, 30), (-12, 20)], {"building": "yes"}))
    d = make_cache(tmp_path / "cache", els, flat())
    f = Frame(to_polygon(square_boundary(LAT0, LON0, 100.0)))
    n = 120
    arr = np.zeros((n, n), dtype="float32")
    x0, y0 = f.E0 - 60.0, f.N0 + 60.0
    t = from_origin(x0, y0, 1.0, 1.0)

    def put(x, y, val, r=1):
        c, rr = int(np.floor(f.E0 + x - x0)), int(np.floor(y0 - (f.N0 + y)))
        arr[rr - r:rr + r + 1, c - r:c + r + 1] = val
    put(5.0, 5.0, 20.0)
    put(-20.0, -20.0, 30.0)
    put(-9.2, 25.0, 31.0)          # building cells right next to node 4 (inside footprint + 0.5 m): ignored
    with rasterio.open(d / "ndom.tif", "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       crs=f"EPSG:{f.epsg}", transform=t, nodata=-9999.0) as ds:
        ds.write(arr, 1)
    site = convert(d, None, tmp_path / "out")
    tr = {x["osm_id"]: x for x in site["trees"]}
    assert (tr["node/1"]["height"], tr["node/1"]["height_source"]) == (20.0, "measured")
    assert tr["node/1"]["crown_diameter"] == pytest.approx(20.0 * 0.7)  # Quercus ratio
    assert (tr["node/2"]["height"], tr["node/2"]["height_source"]) == (11.0, "tag")
    assert tr["node/3"]["height_source"] == "default"
    assert tr["node/4"]["height_source"] == "leaf_type"
    assert site["report"]["info"]["tree_measure"]["available"] is True


# ------------------------------------------------------------------ build_plan
def test_tree_object_uses_crown_and_height():
    t = {"x": 1, "y": 2, "z": 3.0, "height": 20.0, "crown_diameter": 12.5}
    o = bp.tree_object(t, "Deciduous Trees")
    assert o["dimensions"] == {"x": 12.5, "y": 12.5, "z": 20.0}


def test_tree_object_without_crown_keeps_legacy_ratio():
    o = bp.tree_object({"x": 0, "y": 0, "z": 0, "height": 10.0}, "X")
    assert o["dimensions"]["x"] == pytest.approx(6.3)


def test_tree_kind_reads_new_tags():
    assert bp.tree_kind({"tags": {"leaf_type": "needleleaved"}}) == "needle"
    assert bp.tree_kind({"tags": {"leaf_type": "broadleaved", "genus": "Pinus"}}) == "broad"  # leaf_type wins
    assert bp.tree_kind({"tags": {"genus": "Picea"}}) == "needle"
    assert bp.tree_kind({"tags": {"genus": "Tilia"}}) == "broad"
    assert bp.tree_kind({"tags": {}}) == "broad"
    parts = {"broad": "Deciduous Trees", "needle": "Pine Trees"}
    assert bp.tree_lib_for({"tags": {"leaf_type": "needleleaved"}}, parts) == "Pine Trees"


def test_map_settings_passes_tree_keys():
    import server
    m = server.map_settings({"broadleaf_height": 11, "conifer_height": 14, "broadleaf_crown_ratio": 0.5,
                             "conifer_crown_ratio": 0.3, "use_genus_table": False, "tree_height_m": 7})
    s = _merge_settings(m)
    assert (s["broadleaf_height"], s["conifer_height"], s["tree_height"], s["use_genus_table"]) == (11.0, 14.0, 7.0, False)


def test_crown_neighbour_cap():
    from collections import Counter
    from types import SimpleNamespace
    from osmctx.convert import Converter
    trees = [{"x": 0.0, "y": 0.0, "crown_diameter": 23.0, "crown_source": "ratio"},
             {"x": 10.0, "y": 0.0, "crown_diameter": 23.0, "crown_source": "ratio"},
             {"x": 100.0, "y": 0.0, "crown_diameter": 6.0, "crown_source": "tag"}]
    fake = SimpleNamespace(tree_crowns=Counter({"ratio": 2, "tag": 1}))
    Converter._cap_crowns(fake, trees)
    assert trees[0]["crown_diameter"] == 11.5 and trees[0]["crown_source"] == "ratio_capped"
    assert trees[1]["crown_diameter"] == 11.5
    assert trees[2]["crown_diameter"] == 6.0 and trees[2]["crown_source"] == "tag"
    assert fake.tree_crowns["ratio_capped"] == 2 and fake.tree_crowns["ratio"] == 0
