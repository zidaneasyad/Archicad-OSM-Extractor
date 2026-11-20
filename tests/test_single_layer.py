"""Default import mode: ALL OSM elements on ONE existing layer (templates may not allow new layers)."""
import pytest
from fastapi.testclient import TestClient

import server
from osmctx import build_ac
from osmctx.archicad_http import ArchicadError
from tests.test_layers_any_project import FakeAC


class FlagAC(FakeAC):
    def __init__(self, names, hidden=(), locked=()):
        super().__init__(names)
        self.hidden, self.locked, self.calls = set(hidden), set(locked), []

    def tapir(self, cmd, p=None):
        self.calls.append(cmd)
        if cmd == "GetLayers":
            return {"layers": [{"attributeId": x["attributeId"], "name": "\x14" if self.layers[x["attributeId"]["guid"]][0] == 1 else self.layers[x["attributeId"]["guid"]][1],
                                "index": self.layers[x["attributeId"]["guid"]][0],
                                "isHidden": self.layers[x["attributeId"]["guid"]][1] in self.hidden,
                                "isLocked": self.layers[x["attributeId"]["guid"]][1] in self.locked}
                               for x in p["attributeIds"]]}
        return super().tapir(cmd, p)


NAMES = ["Archicad Layer", "Walls", "Site & Landscape - General", "Zeta"]


def test_list_layers_sorted_with_flags():
    ac = FlagAC(NAMES, hidden=["Zeta"], locked=["Walls"])
    ls = build_ac.list_layers(ac)
    assert [d["name"] for d in ls] == ["Archicad Layer", "Site & Landscape - General", "Walls", "Zeta"]
    assert next(d for d in ls if d["name"] == "Walls")["locked"] and next(d for d in ls if d["name"] == "Zeta")["hidden"]
    assert next(d for d in ls if d["name"] == "Walls")["index"] == 2


def test_default_guess():
    assert build_ac.default_single_layer(build_ac.list_layers(FlagAC(NAMES))) == "Site & Landscape - General"
    ls = build_ac.list_layers(FlagAC(["Archicad Layer", "Walls", "Doors"], hidden=["Doors"]))
    assert build_ac.default_single_layer(ls) == "Walls"
    assert build_ac.default_single_layer(build_ac.list_layers(FlagAC(["Archicad Layer"]))) is None


def test_single_maps_all_roles_no_creation():
    ac = FlagAC(NAMES)
    rep = {"gaps": []}
    lidx = build_ac.single_layer(ac, lambda *_: None, rep, "Zeta")
    assert set(lidx) == set(build_ac.LAYER_CANDIDATES) and set(lidx.values()) == {4}
    assert "CreateLayers" not in ac.calls and "CreateLayerCombinations" not in ac.calls
    assert ac.created_layers == [] and ac.combo_writes == []
    assert "groups" in rep["layer_note"] and rep["layer_mode"] == "single"


def test_single_default_target():
    rep = {"gaps": []}
    lidx = build_ac.single_layer(FlagAC(NAMES), lambda *_: None, rep, None)
    assert set(lidx.values()) == {3} and rep["layer_names"]["trees"] == "Site & Landscape - General"


def test_locked_fails_fast_hidden_warns_missing_errors():
    with pytest.raises(ArchicadError, match="locked"):
        build_ac.single_layer(FlagAC(NAMES, locked=["Walls"]), lambda *_: None, {}, "Walls")
    rep = {}
    build_ac.single_layer(FlagAC(NAMES, hidden=["Zeta"]), lambda *_: None, rep, "Zeta")
    assert "hidden" in rep["warnings"][0]
    with pytest.raises(ArchicadError, match="does not exist"):
        build_ac.single_layer(FlagAC(NAMES), lambda *_: None, {}, "Nope")


def test_per_type_unchanged():
    ac = FlagAC(["Archicad Layer", "Site & Landscape - Terrain"])
    lidx = build_ac.ensure_layers(ac, lambda *_: None, {"gaps": []})
    assert "Site-Roads" in ac.created_layers and len(set(lidx.values())) > 1


def test_cli_flags(monkeypatch):
    seen = {}
    monkeypatch.setattr(build_ac, "build", lambda *a, **k: seen.update(k) or {"created": {}, "expected": {}})
    build_ac.main(["x.json"])
    assert seen["layer_mode"] == "single" and seen["target_layer"] is None
    build_ac.main(["x.json", "--layer", "Zeta"])
    assert seen["target_layer"] == "Zeta"
    build_ac.main(["x.json", "--per-type-layers"])
    assert seen["layer_mode"] == "per_type"


def test_api_layers(monkeypatch):
    real = build_ac.list_layers
    monkeypatch.setattr(build_ac, "list_layers", lambda c: real(FlagAC(NAMES)))
    d = TestClient(server.app).get("/api/layers").json()
    assert d["default"] == "Site & Landscape - General" and len(d["layers"]) == 4


def test_api_layers_unreachable(monkeypatch):
    def boom(c):
        raise ArchicadError("Archicad not reachable at x")
    monkeypatch.setattr(build_ac, "list_layers", boom)
    r = TestClient(server.app).get("/api/layers")
    assert r.status_code == 503 and "not reachable" in r.json()["error"]
