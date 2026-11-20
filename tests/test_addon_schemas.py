"""Offline check of the OSMContext add-on command schemas (addon/OSMContext/Schemas/*.json, M5-T2).

Runs the real builder (osmctx.build_ac.build) on out/site.json against a fake Archicad client:
  * every add-on command the builder sends must be either an OSMContext command (payload validated against
    Schemas/<Command>.json) or a native API.* equivalent listed in NATIVE (addon/COMMANDS.md) -> proves the
    builder can run without Tapir;
  * the fake's answers are validated against the OSMContext OUTPUT schemas -> the builder only reads fields
    OSMContext returns.
Also validates the sun payload (osmctx.sun.place_settings) against SetPlaceSettings.
No Archicad needed. Schema validation uses a small built-in Draft-4-style validator (the venv has no
jsonschema); when jsonschema is importable the same checks are repeated with it.
"""
from __future__ import annotations

import itertools
import json
import re
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "addon" / "OSMContext" / "Schemas"
if not SCHEMAS.is_dir():  # the add-on is not part of the public release
    pytest.skip("addon/ not present (public release without the add-on)", allow_module_level=True)
SITE = ROOT / "out" / "m4" / "tuebingen" / "site.json"  # pinned site with trees + hedges (out/site.json = the user's last conversion)
SETUP = ROOT / "out" / "archicad_setup.json"

# Tapir commands the builder calls that Archicad 30 answers natively (probed 2026-11-15: API.<name> exists,
# same request/response shape as Tapir). The builder must send these as API.<name>, not to OSMContext.
NATIVE = {"GetElementsByType", "GetAllClassificationSystems", "SetClassificationsOfElements", "Get3DBoundingBoxes"}
# Site-plan contours (2D polylines + texts) still go through Tapir: no OSMContext command yet (addon/COMMANDS.md, same
# status as the --label helper). The fake answers them so the contour path is exercised offline.
TAPIR_ONLY = {"ChangeWindow", "GetElementUserData", "SetElementUserData", "CreatePolylines", "CreateTexts",
              "ModifyBeams", "ModifySlabs", "CreateGraphicOverrideRules", "GetGraphicOverrideCombinations",
              "CreateGraphicOverrideCombinations", "ModifyGraphicOverrideCombinations",
              "GetActiveGraphicalOverrideCombination", "SetActiveGraphicalOverrideCombination"}


# ------------------------------------------------------------------ schemas + mini validator
def load_schemas():
    defs = json.loads((SCHEMAS / "_definitions.json").read_text(encoding="utf-8"))["definitions"]
    cmds = {}
    for f in SCHEMAS.glob("*.json"):
        if f.name.startswith("_"):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        cmds[d["name"]] = d
    return defs, cmds


DEFS, CMDS = load_schemas()


def _type_ok(v, t):
    if isinstance(t, list):
        return any(_type_ok(v, x) for x in t)
    return {
        "object": lambda: isinstance(v, dict),
        "array": lambda: isinstance(v, list),
        "string": lambda: isinstance(v, str),
        "integer": lambda: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda: isinstance(v, bool),
        "null": lambda: v is None,
    }[t]()


def errors(v, s, path="$"):
    """List of error strings for value v against schema s (refs '#/Name' resolve in DEFS)."""
    out = []
    if "$ref" in s:
        name = s["$ref"][2:]
        assert s["$ref"].startswith("#/") and name in DEFS, f"unresolved ref {s['$ref']}"
        out += errors(v, DEFS[name], path)
    if "type" in s and not _type_ok(v, s["type"]):
        return out + [f"{path}: expected {s['type']}, got {type(v).__name__}"]
    if "enum" in s and v not in s["enum"]:
        out.append(f"{path}: {v!r} not in {s['enum']}")
    if isinstance(v, dict):
        props = s.get("properties", {})
        for r in s.get("required", []):
            if r not in v:
                out.append(f"{path}: missing required '{r}'")
        for k, x in v.items():
            if k in props:
                out += errors(x, props[k], f"{path}.{k}")
            elif s.get("additionalProperties") is False:
                out.append(f"{path}: additional property '{k}' not allowed")
            elif isinstance(s.get("additionalProperties"), dict):
                out += errors(x, s["additionalProperties"], f"{path}.{k}")
    if isinstance(v, list):
        if "minItems" in s and len(v) < s["minItems"]:
            out.append(f"{path}: fewer than {s['minItems']} items")
        if "maxItems" in s and len(v) > s["maxItems"]:
            out.append(f"{path}: more than {s['maxItems']} items")
        if "items" in s:
            for i, x in enumerate(v):
                out += errors(x, s["items"], f"{path}[{i}]")
    if isinstance(v, str):
        if "minLength" in s and len(v) < s["minLength"]:
            out.append(f"{path}: shorter than {s['minLength']}")
        if "pattern" in s and not re.search(s["pattern"], v):
            out.append(f"{path}: {v!r} does not match {s['pattern']}")
    if _type_ok(v, "number"):
        if "minimum" in s and v < s["minimum"]:
            out.append(f"{path}: {v} < minimum {s['minimum']}")
        if "maximum" in s and v > s["maximum"]:
            out.append(f"{path}: {v} > maximum {s['maximum']}")
        if "exclusiveMinimum" in s and not isinstance(s["exclusiveMinimum"], bool) and v <= s["exclusiveMinimum"]:
            out.append(f"{path}: {v} <= exclusiveMinimum {s['exclusiveMinimum']}")
    if "oneOf" in s:
        n = sum(1 for sub in s["oneOf"] if not errors(v, sub, path))
        if n != 1:
            out.append(f"{path}: matches {n} of oneOf (need exactly 1)")
    if "anyOf" in s and not any(not errors(v, sub, path) for sub in s["anyOf"]):
        out.append(f"{path}: matches none of anyOf")
    for sub in s.get("allOf", []):
        out += errors(v, sub, path)
    return out


def jsonschema_errors(v, s):
    js = pytest.importorskip("jsonschema")
    root = dict(DEFS)
    root.update(s)
    return [e.message for e in js.Draft4Validator(root).iter_errors(v)]


def check_input(cmd, params):
    sch = CMDS[cmd]["inputSchema"]
    if sch is None:
        return [] if not params else [f"{cmd} takes no parameters, got {params}"]
    return errors(params, sch)


def check_output(cmd, resp):
    return errors(resp, CMDS[cmd]["outputSchema"])


# ------------------------------------------------------------------ fake Archicad
def g():
    return str(uuid.uuid4())


class FakeArchicad:
    """Answers the builder's calls with Tapir/native-shaped data; records every add-on call."""

    LAYER_NAMES = ["Site & Landscape - Terrain", "Site-Context Buildings", "Site-Roads", "Site-Trees", "Site-Water",
                   "Site & Landscape - General", "Annotation - Text"]
    CLASS_IDS = ["Terrain", "Massing", "Civil Element", "Flora & Fauna", "Geographic Element", "Site"]

    def __init__(self):
        self.calls = []              # (command, params, response, kind) kind = osm | native | missing
        self.elements = {}           # guid -> {"type", "layer"}
        self.layers = {n: i + 1 for i, n in enumerate(self.LAYER_NAMES)}
        self.layer_guids = {g(): n for n in self.LAYER_NAMES}
        self.group_guid = None
        self.go_rules = [{"graphicOverrideRuleId": {"guid": g()}, "name": "Site-Terrain earth", "criterionXML": "",
                          "style": {"surfaceColor": {"red": 0.7, "green": 0.7, "blue": 0.6},
                                    "surfaceAppliesTo": {"cutSurface": True, "uncutSurface": True}}}]
        self.go_combos = [{"graphicOverrideCombinationId": {"guid": g()}, "name": "No Overrides",
                           "graphicOverrideRuleIds": []}]
        self.active_go = self.go_combos[0]

    # native API.* (the builder's c.call)
    def call(self, command, parameters=None):
        p = parameters or {}
        if command == "API.GetAttributesByType":
            n = {"Layer": len(self.layer_guids), "BuildingMaterial": 3, "Fill": 2}.get(p["attributeType"], 1)
            ids = list(self.layer_guids) if p["attributeType"] == "Layer" else [g() for _ in range(n)]
            return {"attributeIds": [{"attributeId": {"guid": x}} for x in ids]}
        if command == "API.GetLayerAttributes":
            return {"attributes": [{"layerAttribute": {"attributeId": x["attributeId"], "name": self.layer_guids[x["attributeId"]["guid"]],
                                                       "isHidden": False, "isLocked": False}} for x in p["attributeIds"]]}
        if command == "API.GetActivePenTables":
            return {"modelViewPenTableId": {"attributeId": {"guid": "pt"}}}
        if command == "API.GetPenTableAttributes":
            return {"attributes": [{"penTableAttribute": {"name": "fake", "pens": [
                {"pen": {"index": i, "weight": w, "color": col, "description": d}} for i, w, col, d in (
                    (1, 0.13, {"red": 0, "green": 0, "blue": 0}, "a"),
                    (12, 0.18, {"red": 0.5, "green": 0.25, "blue": 0.25}, "b"),
                    (19, 0, {"red": 1, "green": 1, "blue": 1}, "w"),
                    (32, 0.35, {"red": 0.5, "green": 0.25, "blue": 0.25}, "c"),
                    (61, 0.18, {"red": 0.42, "green": 0.42, "blue": 0.42}, "road"),
                    (99, 0.13, {"red": 0.67, "green": 0.82, "blue": 1.0}, "water"),
                    (102, 0.13, {"red": 0.8, "green": 0.8, "blue": 0.8}, "bldg"),
                    (104, 0.13, {"red": 0.59, "green": 0.88, "blue": 0.59}, "ground"))]}}]}
        if command == "API.GetAllPropertyIds":
            return {"propertyIds": [{"propertyId": {"guid": g()}}]}
        if command == "API.GetDetailsOfProperties":
            return {"propertyDefinitions": [{"propertyDefinition": {"propertyId": x["propertyId"], "name": "Structural Function",
                                                                    "group": {"name": "Structural"}}} for x in p["properties"]]}
        if command == "API.GetBuildingMaterialAttributes":
            return {"attributes": [{"buildingMaterialAttribute": {"name": nm, "attributeId": {"guid": g()}}}
                                   for nm in ("Sand", "Soil", "Concrete")]}
        if command == "API.GetAllClassificationsInSystem":
            return {"classificationItems": [{"classificationItem": {"id": i, "classificationItemId": {"guid": g()}, "children": []}}
                                            for i in self.CLASS_IDS]}
        if command == "API.GetPropertyValuesOfElements":
            return {"propertyValuesForElements": [{"propertyValues": [{"propertyValue": {"value": "osm"}}]} for _ in p["elements"]]}
        if command == "API.SetPropertyValuesOfElements":
            return {"executionResults": [{"success": True} for _ in p["elementPropertyValues"]]}
        if command == "API.GetSurfaceAttributes":
            return {"attributes": [{"surfaceAttribute": {"name": nm, "attributeId": {"guid": g()}}}
                                   for nm in ("Earth - Brown", "Pavement - Asphalt Dark", "Water - Wavy")]}
        if command == "API.GetFillAttributes":
            return {"attributes": [{"fillAttribute": {"attributeId": {"guid": g()}, "subType": "Empty"}}]}
        raise AssertionError(f"unexpected native call {command}")

    # add-on commands (the builder's c.tapir)
    def tapir(self, command, parameters=None):
        p = parameters or {}
        if command in CMDS:
            kind = "osm"
            errs = check_input(command, p)
            assert not errs, f"{command} payload rejected by OSMContext schema: {errs[:5]}"
        elif command in NATIVE:
            kind = "native"
        elif command in TAPIR_ONLY:
            kind = "tapir-only"
        else:
            self.calls.append((command, p, None, "missing"))
            raise AssertionError(f"builder calls {command}: neither an OSMContext command nor a native API.* equivalent")
        resp = self.answer(command, p)
        if kind == "osm":
            errs = check_output(command, resp)
            assert not errs, f"fake answer for {command} does not match the OSMContext output schema: {errs[:5]}"
        self.calls.append((command, p, resp, kind))
        return resp

    def _create(self, p, key, ty):
        out = []
        for it in p[key]:
            x = g()
            self.elements[x] = {"type": ty, "layer": it.get("layerIndex", 1)}
            out.append({"elementId": {"guid": x}})
        return {"elements": out}

    def answer(self, c, p):
        ok = {"success": True}
        if c == "GetProjectInfo":
            return {"isUntitled": False, "isTeamwork": False, "projectPath": "C:/x/OSM TEST.pln", "projectName": "OSM TEST"}
        if c == "GetStories":
            return {"firstStory": 0, "lastStory": 0, "actStory": 0, "skipNullFloor": False,
                    "stories": [{"index": 0, "floorId": 1, "dispOnSections": True, "level": 0.0, "name": "Ground"}]}
        if c == "GetLayers":
            return {"layers": [{"attributeId": x["attributeId"], "index": self.layers[self.layer_guids[x["attributeId"]["guid"]]],
                                "name": self.layer_guids[x["attributeId"]["guid"]], "isHidden": False, "isLocked": False}
                               for x in p["attributeIds"]]}
        if c == "GetAllClassificationSystems":
            return {"classificationSystems": [{"classificationSystemId": {"guid": g()}, "name": "Archicad Classification"}]}
        if c == "FindLibraryParts":
            return {"libraryParts": [{"name": p["nameContains"] + ".gsm", "path": "lib", "index": 7}]}
        if c == "CreatePropertyGroups":
            return {"propertyGroupIds": [{"propertyGroupId": {"guid": g()}} for _ in p["propertyGroups"]]}
        if c == "CreatePropertyDefinitions":
            return {"propertyIds": [{"propertyId": {"guid": g()}} for _ in p["propertyDefinitions"]]}
        if c in ("UpdatePropertyDefinitions",):
            return {"executionResults": [ok for _ in p["propertyDefinitions"]]}
        if c == "SetGeoLocation":
            return ok
        if c == "GetGeoLocation":
            return {"projectLocation": {"longitude": 9.0, "latitude": 48.5, "altitude": 317.0, "north": 1.57},
                    "surveyPoint": {"position": {"eastings": 1.0, "northings": 2.0, "elevation": 3.0},
                                    "positionInProject": {"x": 0.0, "y": 0.0, "z": 0.0},
                                    "geoReferencingParameters": {k: "" for k in ("crsName", "description", "geodeticDatum", "verticalDatum",
                                                                                 "mapProjection", "mapZone")}
                                    | {"xAxisAbscissa": 1.0, "xAxisOrdinate": 0.0, "scale": 1.0}}}
        if c == "GetSuspendGroupsMode":
            return {"suspendGroups": False}
        if c == "SetSuspendGroupsMode":
            return {"executionResult": ok}
        if c == "GetElementsByType":
            return {"elements": [{"elementId": {"guid": x}} for x, e in self.elements.items() if e["type"] == p["elementType"]]}
        if c == "DeleteElements":
            return {"executionResults": [ok for _ in p["elements"]]}
        if c == "CreateMeshes":
            return self._create(p, "meshesData", "Mesh")
        if c == "CreateSlabs":
            return self._create(p, "slabsData", "Slab")
        if c == "CreateBeams":
            return self._create(p, "beamsData", "Beam")
        if c == "ModifyBeams":
            return {"executionResults": [ok for _ in p["beamsWithDetails"]]}
        if c == "ModifySlabs":
            return {"executionResults": [ok for _ in p["slabsWithDetails"]]}
        if c == "CreateObjects":
            return self._create(p, "objectsData", "Object")
        if c == "ChangeWindow":
            return ok
        if c == "GetElementUserData":
            return {"elementUserData": [{"userData": []} for _ in p["elements"]]}
        if c == "SetElementUserData":
            return {"executionResults": [ok for _ in p["elementUserData"]]}
        if c == "CreatePolylines":
            return self._create(p, "polylinesData", "PolyLine")
        if c == "CreateTexts":
            return self._create(p, "textsData", "Text")
        if c == "SetDetailsOfElements":
            for it in p["elementsWithDetails"]:
                if "layerIndex" in it["details"]:
                    self.elements[it["elementId"]["guid"]]["layer"] = it["details"]["layerIndex"]
            return {"executionResults": [ok for _ in p["elementsWithDetails"]]}
        if c == "SetClassificationsOfElements":
            return {"executionResults": [ok for _ in p["elementClassifications"]]}
        if c == "Get3DBoundingBoxes":
            return {"boundingBoxes3D": [{"boundingBox3D": dict.fromkeys(("xMin", "yMin", "zMin", "xMax", "yMax", "zMax"), 0.0)}
                                        for _ in p["elements"]]}
        if c == "GetDetailsOfElements":
            return {"detailsOfElements": [{"elementId": e["elementId"], "type": self.elements[e["elementId"]["guid"]]["type"], "id": "",
                                           "floorIndex": 0, "layerIndex": self.elements[e["elementId"]["guid"]]["layer"], "drawIndex": 1}
                                          for e in p["elements"]]}
        if c == "GetGraphicOverrideRules":
            return {"graphicOverrideRules": list(self.go_rules)}
        if c == "CreateGraphicOverrideRules":
            for r in p["graphicOverrideRules"]:
                self.go_rules.append({"graphicOverrideRuleId": {"guid": g()}, "name": r["name"],
                                      "criterionXML": "", "style": r.get("style") or {}})
            return {"executionResults": [ok for _ in p["graphicOverrideRules"]]}
        if c == "ModifyGraphicOverrideRules":
            return {"executionResults": [ok for _ in p["graphicOverrideRules"]]}
        if c == "GetGraphicOverrideCombinations":
            return {"graphicOverrideCombinations": list(self.go_combos)}
        if c == "CreateGraphicOverrideCombinations":
            for cb in p["graphicOverrideCombinations"]:
                self.go_combos.append({"graphicOverrideCombinationId": {"guid": g()}, "name": cb["name"],
                                       "graphicOverrideRuleIds": cb["graphicOverrideRuleIds"]})
            return {"executionResults": [ok for _ in p["graphicOverrideCombinations"]]}
        if c == "ModifyGraphicOverrideCombinations":
            return {"executionResults": [ok for _ in p["graphicOverrideCombinations"]]}
        if c == "GetActiveGraphicalOverrideCombination":
            return {"combinationId": self.active_go["graphicOverrideCombinationId"], "name": self.active_go["name"]}
        if c == "SetActiveGraphicalOverrideCombination":
            cb = next((x for x in self.go_combos if x["name"] == p.get("name")), self.active_go)
            self.active_go = cb
            return ok
        if c == "CreateSolidElementLinks":
            return {"executionResults": [ok for _ in p["solidLinks"]]}
        if c == "GetSolidElementLinks":
            return {"solidLinks": [{"solidLinksWithTheGivenTarget": [], "solidLinksWithTheGivenOperator": []} for _ in p["elements"]]}
        if c == "CreateGroups":
            self.group_guid = g()
            return {"groupGuids": [{"groupId": {"guid": self.group_guid}} for _ in p["elementGroups"]]}
        if c == "SetUserDataOfGroups":
            return {"executionResults": [ok for _ in p["groups"]]}
        if c == "GetMainGroups":
            return {"mainGroups": [{"groupId": {"guid": self.group_guid}} for _ in p["groups"]]}
        if c in ("GetElementsOfGroups", "GetAllElementsOfGroups"):
            return {"elementsOfGroups": [{"elements": []} for _ in p["groups"]]}
        if c == "SaveProject":
            return ok
        raise AssertionError(f"fake has no answer for {c}")


# ------------------------------------------------------------------ tests
def test_schema_files_consistent():
    assert CMDS, "no schema files"
    for name, d in CMDS.items():
        assert d["namespace"] == "OSMContext"
        for io in ("inputSchema", "outputSchema"):
            s = d[io]
            if s is None:
                continue
            for r in re.findall(r'"\$ref":\s*"#/([^"]+)"', json.dumps(s)):
                assert r in DEFS, f"{name}.{io}: #/{r} not in _definitions.json"
    hpp = (ROOT / "addon/OSMContext/Src/Commands/OsmSchemas.generated.hpp").read_text(encoding="utf-8")
    for name in CMDS:
        assert f'{{ "{name}",' in hpp, f"{name} missing from OsmSchemas.generated.hpp (re-run make_schemas.py)"


@pytest.fixture(scope="module")
def fake_build(tmp_path_factory):
    if not SITE.exists() or not SETUP.exists():
        pytest.skip("out/site.json or out/archicad_setup.json missing")
    import osmctx.build_ac as bac
    fake = FakeArchicad()
    orig = bac.Client
    bac.Client = lambda *a, **k: fake
    try:
        tmp = tmp_path_factory.mktemp("addon_schemas")
        rep = bac.build(str(SITE), str(SETUP), url=None, log_path=str(tmp / "log.txt"), report_path=str(tmp / "rep.json"),
                        save=True, group=True, label=False)
    finally:
        bac.Client = orig
    return fake, rep


def test_builder_payloads_match_osmcontext_schemas(fake_build):
    fake, rep = fake_build
    missing = [c for c, _, _, k in fake.calls if k == "missing"]
    assert not missing, f"builder needs commands OSMContext lacks: {sorted(set(missing))}"
    site = json.loads(SITE.read_text(encoding="utf-8"))
    assert rep["created"]["buildings"] == len(site["buildings"])
    assert rep["created"]["trees"] == len(site["trees"])
    used = {c for c, _, _, k in fake.calls if k == "osm"}
    # the core build path exercises these (label/test helpers excluded)
    for c in ("CreateMeshes", "CreateSlabs", "CreateBeams", "CreateObjects", "SetDetailsOfElements", "CreateSolidElementLinks",
              "CreateGroups", "SetUserDataOfGroups", "SetGeoLocation", "GetSuspendGroupsMode", "SetSuspendGroupsMode",
              "ModifyGraphicOverrideRules", "SaveProject", "CreatePropertyDefinitions", "UpdatePropertyDefinitions"):
        assert c in used, f"{c} not exercised by the fake build"


def test_builder_payloads_jsonschema_crosscheck(fake_build):
    pytest.importorskip("jsonschema")
    fake, _ = fake_build
    for c, p, _, k in fake.calls:
        if k == "osm" and CMDS[c]["inputSchema"] is not None:
            assert not jsonschema_errors(p, CMDS[c]["inputSchema"]), c


def test_sun_payload_matches_setplacesettings():
    from osmctx import sun
    p = sun.place_settings(48.52, 9.05, "2026-06-21", "12:00", origin_elevation=317.39, north_correction_deg=0.0423)
    assert not check_input("SetPlaceSettings", p)


@pytest.mark.parametrize("cmd,bad", [
    ("CreateSlabs", {"slabsData": [{"level": 0, "polygonCoordinates": [{"x": 0, "y": 0}] * 3, "cutFillPen": {}}]}),
    ("CreateObjects", {"objectsData": [{"libraryPartName": "x", "coordinates": {"x": 0, "y": 0, "z": 0}, "favoriteName": "f"}]}),
    ("SetDetailsOfElements", {"elementsWithDetails": [{"elementId": {"guid": str(uuid.uuid4())},
                                                        "details": {"typeSpecificDetails": {}}}]}),
    ("CreateMeshes", {"meshesData": [{"polygonCoordinates": [{"x": 0, "y": 0}] * 3}]}),     # z missing
    ("CreateBeams", {"beamsData": [{"begCoordinate": {"x": 0, "y": 0}, "endCoordinate": {"x": 1, "y": 0}}]}),  # zCoordinate missing
    ("CreatePropertyDefinitions", {"propertyDefinitions": [{"propertyDefinition": {
        "name": "a", "description": "", "type": "singleEnum", "isEditable": True, "availability": [], "group": {"name": "g"}}}]}),
])
def test_pruned_or_invalid_fields_are_rejected(cmd, bad):
    assert check_input(cmd, bad), f"{cmd} accepted an unsupported payload"


def test_validator_sanity():
    good = {"elements": [{"elementId": {"guid": "12345678-1234-1234-1234-123456789abc"}}]}
    assert not check_input("DeleteElements", good)
    assert check_input("DeleteElements", {"elements": [{"elementId": {"guid": "nope"}}]})
    assert check_input("DeleteElements", {})
    assert not check_output("CreateSlabs", {"elements": [{"error": {"code": -1, "message": "x"}}, good["elements"][0]]})
    assert check_output("CreateSlabs", {"elements": [{"error": {"code": -1}}]})
