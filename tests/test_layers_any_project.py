"""Imports must work in any project / template: missing Site-* layers are created and made visible in every
existing layer combination; template layers are reused; attribute guids are re-resolved by name."""
import itertools

from osmctx import build_ac

_n = itertools.count()


def g():
    return f"00000000-0000-0000-0000-{next(_n):012d}"


class FakeAC:
    def __init__(self, layer_names):
        self.layers = {g(): (i + 1, n) for i, n in enumerate(layer_names)}
        self.combos = {g(): "01 Site", g(): "02 Drafting"}
        self.created_layers, self.combo_writes = [], []

    def call(self, cmd, p=None):
        p = p or {}
        if cmd == "API.GetAttributesByType":
            src = {"Layer": self.layers, "LayerCombination": self.combos}.get(p["attributeType"], {"dummy": 1})
            return {"attributeIds": [{"attributeId": {"guid": x}} for x in src]}
        if cmd == "API.GetLayerAttributes":
            return {"attributes": [{"layerAttribute": {"attributeId": x["attributeId"],
                                                       "name": self.layers[x["attributeId"]["guid"]][1]}}
                                   for x in p["attributeIds"]]}
        if cmd == "API.GetSurfaceAttributes":
            return {"attributes": [{"surfaceAttribute": {"name": "Water - Wavy", "attributeId": {"guid": "S-WATER"}}}]}
        if cmd == "API.GetBuildingMaterialAttributes":
            return {"attributes": [{"buildingMaterialAttribute": {"name": "Soil", "attributeId": {"guid": "B-SOIL"}}}]}
        raise AssertionError(cmd)

    def tapir(self, cmd, p=None):
        p = p or {}
        if cmd == "GetLayers":
            return {"layers": [{"name": self.layers[x["attributeId"]["guid"]][1],
                                "index": self.layers[x["attributeId"]["guid"]][0]} for x in p["attributeIds"]]}
        if cmd == "CreateLayers":
            for d in p["layerDataArray"]:
                assert d["isHidden"] is False
                self.layers[g()] = (len(self.layers) + 1, d["name"])
                self.created_layers.append(d["name"])
            return {"attributeIds": []}
        if cmd == "GetLayerCombinations":
            return {"layerCombinations": [{"layerCombination": {"attributeId": x["attributeId"],
                                                                "name": self.combos[x["attributeId"]["guid"]]}}
                                          for x in p["attributes"]]}
        if cmd == "CreateLayerCombinations":
            assert p["overwriteExisting"] is True
            self.combo_writes = p["layerCombinationDataArray"]
            return {"attributeIds": []}
        raise AssertionError(cmd)


def test_missing_layers_created_and_shown_in_all_combinations():
    ac = FakeAC(["Archicad Layer", "Site & Landscape - Terrain"])   # bare template: only the terrain layer fits
    report = {"gaps": []}
    lidx = build_ac.ensure_layers(ac, lambda *_: None, report)
    assert lidx["terrain"] == 2 and lidx["land_cover"] == 2         # template layer reused
    assert set(ac.created_layers) == {"Site-Context Buildings", "Site-Roads", "Site-Trees", "Site-Water", "Site-Shrubs", "Site-Contours"}
    assert all(v is not None for v in lidx.values())
    assert len(ac.combo_writes) == 2                                 # every existing combination updated
    for lc in ac.combo_writes:
        assert lc["name"] in ("01 Site", "02 Drafting")             # names kept
        assert len(lc["layers"]) == 6 and all(s["isHidden"] is False for s in lc["layers"])  # ONLY new layers


def test_existing_layers_untouched():
    ac = FakeAC(["Site & Landscape - Terrain", "Site-Context Buildings", "Site-Roads", "Site-Trees", "Site-Water",
                 "Site & Landscape - General", "Site-Contours"])
    build_ac.ensure_layers(ac, lambda *_: None, {"gaps": []})
    assert ac.created_layers == [] and ac.combo_writes == []


def test_attribute_guids_resolved_by_name_in_open_project():
    ac = FakeAC([])
    rec = {"water": {"surface": "Water - Wavy", "surface_guid": "FOREIGN", "building_material": "Water"},
           "terrain": {"surface": "Earth - Brown", "surface_guid": "FOREIGN2", "building_material": "Soil"}}
    out = build_ac.resolve_attribute_guids(ac, rec, lambda *_: None)
    assert out["water"]["surface_guid"] == "S-WATER"
    assert out["water"]["building_material_guid"] is None            # not in this project -> no override
    assert out["terrain"]["surface_guid"] is None and out["terrain"]["building_material_guid"] == "B-SOIL"
