"""plan_style presets (colour | mono | none): pen picking, apply_plan_styles, apply_terrain_go, CLI. Fake client only."""
import copy
import json

import pytest

from osmctx import build_ac as bac
from osmctx import build_plan as bp

PENS = [
    {"index": 1, "weight": 0.13, "color": {"red": 0.0, "green": 0.0, "blue": 0.0}, "description": "black thin"},
    {"index": 7, "weight": 0.35, "color": {"red": 0.0, "green": 0.0, "blue": 0.0}, "description": "black thick"},
    {"index": 19, "weight": 0.0, "color": {"red": 1.0, "green": 1.0, "blue": 1.0}},
    {"index": 20, "weight": 0.13, "color": {"red": 0.5, "green": 0.5, "blue": 0.5}, "description": "grey thin"},
    {"index": 21, "weight": 0.35, "color": {"red": 0.5, "green": 0.5, "blue": 0.5}, "description": "grey thick"},
    {"index": 30, "weight": 0.13, "color": {"red": 0.6, "green": 0.4, "blue": 0.2}, "description": "brown thin"},
    {"index": 31, "weight": 0.35, "color": {"red": 0.6, "green": 0.4, "blue": 0.2}, "description": "brown thick"},
    {"index": 61, "weight": 0.2, "color": {"red": 0.28, "green": 0.28, "blue": 0.28}},
    {"index": 62, "weight": 0.2, "color": {"red": 0.38, "green": 0.38, "blue": 0.38}},
    {"index": 90, "weight": 0.13, "color": {"red": 0.55, "green": 0.55, "blue": 0.55}},
    {"index": 91, "weight": 0.13, "color": {"red": 0.88, "green": 0.88, "blue": 0.88}},
    {"index": 92, "weight": 0.13, "color": {"red": 0.97, "green": 0.97, "blue": 0.97}},
    {"index": 99, "weight": 0.13, "color": {"red": 0.67, "green": 0.82, "blue": 1.0}},
    {"index": 102, "weight": 0.13, "color": {"red": 0.8, "green": 0.8, "blue": 0.8}},
    {"index": 104, "weight": 0.13, "color": {"red": 0.59, "green": 0.88, "blue": 0.59}},
]
EMPTY = {"guid": "EMPTY-FILL"}


class FakeClient:
    """Records every call; answers the few reads apply_plan_styles / apply_terrain_go need."""

    def __init__(self):
        self.calls = []

    def call(self, name, params=None):
        self.calls.append((name, copy.deepcopy(params)))
        if name == "API.GetActivePenTables":
            return {"modelViewPenTableId": {"attributeId": {"guid": "PT"}}}
        if name == "API.GetPenTableAttributes":
            return {"attributes": [{"penTableAttribute": {"name": "T", "pens": [{"pen": p} for p in PENS]}}]}
        if name == "API.GetAttributesByType":
            return {"attributeIds": [{"attributeId": EMPTY}]}
        if name == "API.GetFillAttributes":
            return {"attributes": [{"fillAttribute": {"attributeId": EMPTY, "subType": "Empty"}}]}
        raise AssertionError(name)

    def tapir(self, name, params=None):
        self.calls.append((name, copy.deepcopy(params)))
        if name in ("ModifySlabs", "ModifyBeams"):
            key = "slabsWithDetails" if name == "ModifySlabs" else "beamsWithDetails"
            return {"executionResults": [{"success": True}] * len(params[key])}
        if name == "GetGraphicOverrideRules":
            return {"graphicOverrideRules": [
                {"name": bac.TERRAIN_GO_RULE, "graphicOverrideRuleId": {"guid": "R1"}, "style": {}}]}
        if name == "ModifyGraphicOverrideRules":
            return {"executionResults": [{"success": True}]}
        if name == "CreateGraphicOverrideRules":
            return {}
        if name == "GetGraphicOverrideCombinations":
            return {"graphicOverrideCombinations": [
                {"name": "X", "graphicOverrideCombinationId": {"guid": "C1"}, "graphicOverrideRuleIds": []}]}
        if name == "GetActiveGraphicalOverrideCombination":
            return {"name": "X", "combinationId": {"guid": "C1"}}
        if name == "ModifyGraphicOverrideCombinations":
            return {}
        raise AssertionError(name)


def created():
    def g(x):
        return [({"guid": f"{x}{i}"}["guid"], "o") for i in range(2)]
    return {"water": g("w"), "buildings": g("b"), "building_parts": g("p"), "roads": g("r"), "terrain": g("t")}


def run_styles(**kw):
    c, rep = FakeClient(), {}
    bac.apply_plan_styles(c, lambda *_: None, created(), rep, ground_slab_guids=["gs0"], **kw)
    return c, rep


def slab_pens(c):
    out = {}
    for name, p in c.calls:
        if name == "ModifySlabs":
            for it in p["slabsWithDetails"]:
                out[it["elementId"]["guid"]] = it["floorFill"]["foregroundPen"]
        if name == "ModifyBeams":
            for it in p["beamsWithDetails"]:
                out[it["elementId"]["guid"]] = it["coverFill"]["foregroundPen"]
    return out


def test_constants():
    assert bp.PLAN_STYLES == ("colour", "mono", "none")
    assert set(bp.PLAN_FILL_RGB_MONO) == set(bp.PLAN_FILL_RGB)


def test_colour_is_default_and_identical():
    c0, r0 = run_styles()
    c1, r1 = run_styles(plan_style="colour")
    assert c0.calls == c1.calls and r0 == r1
    assert r1["plan_styles"]["pens"] == {"water": 99, "ground": 104, "buildings": 102, "roads": 62}
    assert bac.pick_plan_pens(PENS) == bac.pick_plan_pens(PENS, "colour")
    assert bac.pick_contour_pens(PENS) == bac.pick_contour_pens(PENS, style="colour")


def test_mono_plan_pens_are_grey_and_ordered():
    pens = bac.pick_plan_pens(PENS, "mono")
    assert pens == {"water": 91, "ground": 92, "buildings": 90, "roads": 61}
    c, rep = run_styles(plan_style="mono")
    used = slab_pens(c)
    assert used["b0"] == used["p1"] == 90 and used["w0"] == 91 and used["gs0"] == 92 and used["r1"] == 61
    assert "t0" not in used            # mesh-contour mode: terrain not a slab
    assert rep["plan_styles"]["slabs"]["updated"] == rep["plan_styles"]["slabs"]["total"]
    # no coloured pen: every chosen pen has no chroma
    by = {p["index"]: p["color"] for p in PENS}
    for i in pens.values():
        col = by[i]
        assert max(col.values()) - min(col.values()) < 0.06


def test_mono_same_mechanism_as_colour():
    cc, _ = run_styles(plan_style="colour")
    cm, _ = run_styles(plan_style="mono")
    assert [n for n, _ in cc.calls] == [n for n, _ in cm.calls]
    s = [p for n, p in cm.calls if n == "ModifySlabs"][0]["slabsWithDetails"][0]
    assert s["floorFill"]["fillId"] == EMPTY and s["cutFillPen"]["overridden"] is True


def test_none_skips_everything():
    c, rep = run_styles(plan_style="none")
    assert c.calls == []
    assert rep["plan_styles"] == {"skipped": "plan_style none"}


def test_contour_pens_mono():
    p = bac.pick_contour_pens(PENS, style="mono")
    assert (p["major"], p["minor"], p["label"]) == (7, 20, 7) and p["fill"] == 19
    # weight logic kept: nearest to 0.35 / 0.13
    assert bac.pick_contour_pens(PENS)["major"] == 31


def test_contour_pens_mono_falls_back_without_grey_thick():
    pens = [x for x in PENS if x["index"] not in (7, 21)]
    assert bac.pick_contour_pens(pens, style="mono") == bac.pick_contour_pens(pens)


def terrain_go(plan_style=None):
    c, rep = FakeClient(), {}
    args = () if plan_style is None else (plan_style,)
    bac.apply_terrain_go(c, lambda *_: None, "clean", rep, *args)
    return c, rep


def tints(c):
    out = []
    for n, p in c.calls:
        if n in ("ModifyGraphicOverrideRules", "CreateGraphicOverrideRules"):
            for r in p["graphicOverrideRules"]:
                out.append(r["style"]["fillBackgroundColor"])
    return out


def test_terrain_go_colour_default_unchanged():
    c0, r0 = terrain_go()
    c1, r1 = terrain_go("colour")
    assert c0.calls == c1.calls and r0 == r1
    assert tints(c1) and all(t == bp.TERRAIN_PLAN_TINT for t in tints(c1))


def test_terrain_go_mono_light_grey():
    c, rep = terrain_go("mono")
    t = tints(c)
    assert t and all(x == bp.MONO_TERRAIN_PLAN_TINT for x in t)
    assert rep["terrain_go"]["ok"] is True


def test_terrain_go_none_no_calls():
    c, rep = terrain_go("none")
    assert c.calls == []
    assert rep["terrain_go"] == "skipped: plan_style none"


def test_build_rejects_unknown_plan_style(tmp_path):
    with pytest.raises(bac.ArchicadError):
        bac._build_impl(FakeClient(), lambda *_: None, 0, "x", "x", "x", True, False, 0.0, "none", False, "light",
                        False, "clean", True, "single", None, "points", False, None, 1.0, None, "railing", {},
                        plan_style="neon")


def test_cli_flag_present():
    import inspect
    src = inspect.getsource(bac.main) if hasattr(bac, "main") else open(bac.__file__, encoding="utf-8").read()
    assert "--plan-style" in src and "plan_style=a.plan_style" in open(bac.__file__, encoding="utf-8").read()
    assert "plan_style" in inspect.signature(bac.build).parameters
