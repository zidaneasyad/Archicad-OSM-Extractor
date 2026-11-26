"""M11 wiring: the four terrain types (Slab/stepped, Mesh, Morph, GDL) and the 2D style presets are consistent
across build_plan, build_ac, server and the UI. Offline only (no Archicad calls)."""
from pathlib import Path

from osmctx import build_ac as bac
from osmctx import build_plan as bp

ROOT = Path(__file__).resolve().parents[1]


def test_contour_types():
    assert bp.CONTOUR_TYPES == ("mesh", "stepped", "morph", "gdl")


def test_terrain_element_type_per_contour_type():
    assert bac.TERRAIN_ELEMENT_TYPE == {"stepped": "Slab", "morph": "Morph", "gdl": "Object"}
    assert bac.KIND_TYPE_ROLE["terrain"][0] == "Mesh"            # default (mesh)
    for ty in bac.TERRAIN_ELEMENT_TYPE.values():                 # re-import must find + delete every terrain type
        assert ty in bac.OSM_ELEMENT_TYPES
    assert bac.KIND_TYPE_ROLE["contours_3d"] == ("Morph", "contours")
    assert bac.CLASS_ITEMS["contours_3d"] == "Terrain"


def test_plan_styles():
    assert bp.PLAN_STYLES == ("colour", "mono", "none")


def test_ui_offers_every_type_and_style():
    html = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    for v in ("stepped", "mesh", "morph", "gdl", "colour", "mono", "none"):
        assert f'value="{v}"' in html, v
    js = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    assert "plan_style: settings.plan_style" in js


def test_server_validates_against_build_plan_constants():
    src = (ROOT / "server.py").read_text(encoding="utf-8")
    assert "_bp.CONTOUR_TYPES" in src and "_bp.PLAN_STYLES" in src
    assert "plan_style=pstyle" in src
