"""S5: build site.json into the live Archicad project (native tools, Tapir 2.0.0 over HTTP).

    python -m osmctx.build_ac out/site.json [--setup out/archicad_setup.json] [--keep] [--url ...]
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import sys
import time
from collections import Counter
from pathlib import Path

from . import build_plan as bp
from . import features as feat
from . import contours as ct
from . import paths
from .archicad_http import ArchicadError, Client, chunks, eid
from .build_groups import make_groups

BATCH = 50            # element creation payloads per call (geometry-heavy: keep small)
WRITE_CHUNK = 200     # Tapir per-element write commands (SetDetails/Classifications/UserData/Delete/SEO links)
PROP_CHUNK = 400      # native API.SetPropertyValuesOfElements
READ_CHUNK = 400      # read-only element queries (property values, details, bounding boxes, user data)
GROUP = "OSM Context"
PROP_ID, PROP_SRC = "OSM ID", "Source"
CLASSIFICATION_SYSTEM = "Archicad Classification"
# kind -> classification item id (as named in the open file's Archicad Classification)
CLASS_ITEMS = {"terrain": "Terrain", "buildings": "Massing", "roads": "Civil Element", "trees": "Flora & Fauna", "water": "Geographic Element",
               "shrubs": "Flora & Fauna", "hedges": "Flora & Fauna", "land_cover": "Terrain",
               # more features (M6): closest existing items of 'Archicad Classification'
               "railways": "Civil Element", "rail_platforms": "Civil Element", "walls": "Wall", "railings": "Railing",
               "building_parts": "Massing", "building_roofs": "Roof", "ground": "Site Geometry", "contours_3d": "Terrain",
               "piers": "Civil Element", "power": "Civil Element",
               "building_morphs": "Roof"}   # M10 LoD2: roof surfaces the Roof tool cannot take (Morph fallback)
# shrubs + hedges use the template landscape layer (no extra Site-* layer, no extra GO rule)
LAYERS = {"terrain": "Site & Landscape - Terrain", "buildings": "Site-Context Buildings", "roads": "Site-Roads",
          "trees": "Site-Trees", "water": "Site-Water", "shrubs": "Site & Landscape - General",
          "hedges": "Site & Landscape - General", "land_cover": "Site & Landscape - Terrain"}


class Log:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")  # noqa: SIM115 - lives as long as the build, closed by close()

    def __call__(self, msg):
        line = f"{dt.datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def _memo(c, attr, fn):
    """Per-client cache: fn() once per Client (attribute lookups that do not change during one build)."""
    if hasattr(c, attr):
        return getattr(c, attr)
    val = fn()
    with contextlib.suppress(Exception):
        setattr(c, attr, val)
    return val


def run_batches(c, log, cmd, key, items, label):
    """Send items in batches of BATCH. Returns (guids aligned to items, None where failed; errors)."""
    guids, errors = [], []
    for k, chunk in enumerate(chunks(items, BATCH)):
        missing = {"error": {"message": "no result"}}
        try:
            res = c.tapir(cmd, {key: chunk}).get("elements", [])
        except ArchicadError as e:
            log(f"  {label} batch {k}: FAILED {e}")
            res, missing = [], {"error": {"code": e.code, "message": f"batch failed: {e}"}}
        for j in range(len(chunk)):
            r = res[j] if j < len(res) else missing
            if "elementId" in r:
                guids.append(r["elementId"]["guid"])
            else:
                guids.append(None)
                errors.append({"index": len(guids) - 1, "error": r.get("error")})
        log(f"  {label}: {len([g for g in guids if g])}/{len(guids)} created of {len(items)}")
    return guids, errors


# ------------------------------------------------------------ discovery
# Per role: layers to use if they exist (template first), else the Site-* layer the builder creates itself.
LAYER_CANDIDATES = {
    "terrain": ["Site & Landscape - Terrain", "Site-Terrain"],
    "buildings": ["Site-Context Buildings"],
    "roads": ["Site-Roads"],
    "trees": ["Site-Trees"],
    "water": ["Site-Water"],
    "shrubs": ["Site & Landscape - General", "Site-Shrubs"],
    "hedges": ["Site & Landscape - General", "Site-Shrubs"],
    "land_cover": ["Site & Landscape - Terrain", "Site-Terrain"],
    "contours": ["Site-Contours"],   # per_type mode only (created when contours are on); single mode: the target layer
    # more features (M6), per_type mode only and only when the site has such elements (ensure_layers `roles`)
    "railways": ["Site-Railways"], "walls": ["Site-Walls"], "ground": ["Site-Ground"], "piers": ["Site-Piers"],
    "power": ["Site-Power"],
}
FEATURE_ROLES = ("railways", "walls", "ground", "piers", "power")   # created on demand; building parts/roofs use 'buildings'


PREFERRED_SINGLE_LAYER = "Site & Landscape - General"
DEFAULT_LAYER_NAME = "Archicad Layer"   # Tapir reports it as ""; native API gives the real name


def _attributes(c, kind, getter, key):
    """(attributeIds, [attribute dicts]) of one attribute type: GetAttributesByType + `getter` in chunks of 400."""
    ids = c.call("API.GetAttributesByType", {"attributeType": kind})["attributeIds"]
    out = []
    for part in chunks(ids, 400):
        out += [a.get(key) or {} for a in c.call(getter, {"attributeIds": part})["attributes"]]
    return ids, out


def _guid_by_name(attrs):
    """{name: guid} of attribute dicts with a non-empty name."""
    return {x["name"]: x["attributeId"]["guid"] for x in attrs if x.get("name")}


def list_layers(c):
    """Layers of the open project: [{name, index, hidden, locked}] sorted by name (case-insensitive).
    Index + flags from Tapir GetLayers, names from the native API (Tapir returns '' for 'Archicad Layer')."""
    ats, attrs = _attributes(c, "Layer", "API.GetLayerAttributes", "layerAttribute")
    names = {x["attributeId"]["guid"]: x["name"] for x in attrs if x.get("name") is not None}
    out = []
    for part in chunks(ats, 400):
        for x in c.tapir("GetLayers", {"attributeIds": part, "fields": ["name", "index", "isHidden", "isLocked"]})["layers"]:
            nm = names.get(x.get("attributeId", {}).get("guid"), x.get("name"))
            if nm is None or "index" not in x:
                continue
            out.append({"name": nm, "index": x["index"], "hidden": bool(x.get("isHidden")),
                        "locked": bool(x.get("isLocked"))})
    return sorted(out, key=lambda d: d["name"].lower())


def default_single_layer(layers):
    """Best guess for the one-layer import: 'Site & Landscape - General' if present, else the first visible,
    unlocked layer that is not 'Archicad Layer'. None if nothing fits."""
    for d in layers:
        if d["name"] == PREFERRED_SINGLE_LAYER:
            return d["name"]
    return next((d["name"] for d in layers if not d["hidden"] and not d["locked"] and d["name"] != DEFAULT_LAYER_NAME), None)


def single_layer(c, log, report, target_layer):
    """layer_mode='single': every role -> ONE existing layer. No layer creation, no layer-combination change.
    Locked -> ArchicadError (fail fast, nothing written). Hidden -> build + warn."""
    layers = list_layers(c)
    if not target_layer:
        target_layer = default_single_layer(layers)
        if not target_layer:
            raise ArchicadError("no target layer given and no visible unlocked layer found in the project")
    hit = next((d for d in layers if d["name"] == target_layer), None)
    if hit is None:
        raise ArchicadError(f"target layer '{target_layer}' does not exist in the open project")
    if hit["locked"]:
        raise ArchicadError(f"target layer '{target_layer}' is locked in the current layer combination: unlock it in "
                            "Archicad (Document > Layers) or choose another layer; nothing was imported")
    if hit["hidden"]:
        w = (f"target layer '{target_layer}' is hidden in the current layer combination: elements were built but are "
             "not visible until the layer is shown")
        report.setdefault("warnings", []).append(w)
        log("WARNING " + w)
    report["layer_names"] = dict.fromkeys(LAYER_CANDIDATES, target_layer)
    report["layer_mode"] = "single"
    report["layer_note"] = (f"All OSM elements are on one layer '{target_layer}'. The type distinction lives in groups "
                            "(Buildings/Roads/Trees/..), the OSM ID + Source properties and the classifications, not in layers.")
    return dict.fromkeys(LAYER_CANDIDATES, hit["index"])


def feature_roles_needed(site):
    """per_type mode: the feature layers (Site-Railways ...) are only created for feature types the site contains."""
    r = set()
    if site.get("railways") or site.get("railway_platforms"):
        r.add("railways")
    if site.get("barriers"):
        r.add("walls")
    if site.get("ground_surfaces"):
        r.add("ground")
    if site.get("piers"):
        r.add("piers")
    if site.get("power_towers"):
        r.add("power")
    return r


def ensure_layers(c, log, report, contours=True, roles=()):
    """Resolve a layer per role; create the missing Site-* layers (visible, unlocked) so imports work in any project.
    Archicad adds new layers to existing layer combinations as HIDDEN: reported so the user can add them."""
    have = resolve_layers(c)
    want = {}
    for role, names in LAYER_CANDIDATES.items():
        if role == "contours" and not contours:
            continue
        if role in FEATURE_ROLES and role not in roles:
            continue
        hit = next((n for n in names if n in have), None)
        want[role] = hit or names[-1]
    missing = sorted({n for n in want.values() if n not in have})
    if missing:
        res = c.tapir("CreateLayers", {"layerDataArray": [{"name": n, "isHidden": False, "isLocked": False}
                                                           for n in missing]})
        log(f"created layers: {missing} -> {res}")
        report["layers_created"] = missing
        report["layer_combinations_updated"] = show_layers_in_all_combinations(c, log, missing)
        have = resolve_layers(c)
    lidx = {role: have.get(n) for role, n in want.items()}
    if not contours:
        lidx["contours"] = lidx["terrain"]   # unused, keeps the role map complete
    for role in FEATURE_ROLES:
        lidx.setdefault(role, lidx["terrain"])   # feature types the site does not have: unused, role map stays complete
    if None in lidx.values():
        raise ArchicadError(f"could not resolve/create layers: { {r: want[r] for r, v in lidx.items() if v is None} }")
    report["layer_names"] = want
    return lidx


def show_layers_in_all_combinations(c, log, layer_names):
    """Archicad adds a NEW layer to every existing layer combination as hidden, so an import would be invisible in
    the user's views. Switch ONLY the layers the builder just created to visible in every combination (partial
    overwrite: layers not listed keep their state). Never touches pre-existing layers or combination names."""
    _, attrs = _attributes(c, "Layer", "API.GetLayerAttributes", "layerAttribute")
    guid_of = {x["name"]: x["attributeId"]["guid"] for x in attrs if x.get("name") in layer_names}
    states = [{"attributeId": {"guid": g}, "isHidden": False, "isLocked": False, "isWireframe": False,
               "intersectionGroupNr": 1} for g in guid_of.values()]
    combo_ids = c.call("API.GetAttributesByType", {"attributeType": "LayerCombination"})["attributeIds"]
    if not states or not combo_ids:
        return {"combinations": 0, "layers": list(guid_of)}
    combos = c.tapir("GetLayerCombinations", {"attributes": combo_ids})["layerCombinations"]
    data = [{"attributeId": lc["layerCombination"]["attributeId"], "name": lc["layerCombination"]["name"],
             "layers": states} for lc in combos if "layerCombination" in lc]
    res = c.tapir("CreateLayerCombinations", {"layerCombinationDataArray": data, "overwriteExisting": True})
    log(f"made {list(guid_of)} visible in {len(data)} layer combinations -> {str(res)[:200]}")
    return {"combinations": len(data), "layers": list(guid_of)}


def building_material_map(c):
    """{name: guid} for all BuildingMaterials in the open project (cached on `c`)."""
    return _memo(c, "_bmat_cache", lambda: _guid_by_name(
        _attributes(c, "BuildingMaterial", "API.GetBuildingMaterialAttributes", "buildingMaterialAttribute")[1]))


def resolve_attribute_guids(c, rec, log):
    """setup json stores attribute NAMES (+ guids from the project it was made in). Re-resolve every guid by name in
    the OPEN project; unknown names -> None (no override) instead of a foreign guid."""
    surf = _guid_by_name(_attributes(c, "Surface", "API.GetSurfaceAttributes", "surfaceAttribute")[1])
    bmat = building_material_map(c)
    fixed = {}
    for role, d in rec.items():
        d = dict(d)
        if "surface" in d:
            d["surface_guid"] = surf.get(d["surface"]) or surf.get(d.get("alt_surface", ""))
        if "building_material" in d:
            d["building_material_guid"] = bmat.get(d["building_material"])
        fixed[role] = d
    miss = [f"{r}:{d.get('surface')}" for r, d in fixed.items() if "surface" in d and not d.get("surface_guid")]
    if miss:
        log(f"surfaces not in this project (no override): {miss}")
    return fixed


def resolve_layers(c):
    ats = c.call("API.GetAttributesByType", {"attributeType": "Layer"})["attributeIds"]
    res = c.tapir("GetLayers", {"attributeIds": ats, "fields": ["name", "index"]})["layers"]
    return {x["name"]: x["index"] for x in res if "name" in x}


def find_properties(c, group, names, builtin=False):
    """{name: propertyId} for any subset of `names` found in `group` in one GetAllPropertyIds + GetDetailsOfProperties pass."""
    want = set(names)
    found = {}
    ids = c.call("API.GetAllPropertyIds", {"propertyType": "BuiltIn" if builtin else "UserDefined"})["propertyIds"]
    for part in chunks(ids, 400):
        for x in c.call("API.GetDetailsOfProperties", {"properties": part})["propertyDefinitions"]:
            p = x["propertyDefinition"]
            if p["name"] in want and (group is None or p.get("group", {}).get("name") == group):
                found[p["name"]] = p["propertyId"]
                if len(found) == len(want):
                    return found
    return found


def find_property(c, group, name, builtin=False):
    return find_properties(c, group, (name,), builtin).get(name)


def ensure_properties(c, log, item_ids):
    """Property group 'OSM Context' with 'OSM ID' and 'Source' (string).
    Availability in Archicad is classification-based and NOT inherited by child items:
    list every classification item the importer assigns (empty availability = not available anywhere)."""
    avail = [{"classificationItemId": i} for i in item_ids]
    found = find_properties(c, GROUP, (PROP_ID, PROP_SRC))
    pid, sid = found.get(PROP_ID), found.get(PROP_SRC)
    if pid and sid:
        c.tapir("UpdatePropertyDefinitions", {"propertyDefinitions": [
            {"propertyId": p, "availability": {"set": avail}} for p in (pid, sid)]})
        return pid, sid
    g = c.tapir("CreatePropertyGroups", {"propertyGroups": [{"propertyGroup": {
        "name": GROUP, "description": "OSM Context Archicad importer tags"}}]})
    gid = g["propertyGroupIds"][0]["propertyGroupId"]
    defs = [{"propertyDefinition": {"name": nm, "description": ds, "type": "string", "isEditable": True,
                                    "availability": avail, "group": {"propertyGroupId": gid}}}
            for nm, ds in ((PROP_ID, "OpenStreetMap element id"), (PROP_SRC, "Data source"))]
    r = c.tapir("CreatePropertyDefinitions", {"propertyDefinitions": defs})["propertyIds"]
    c.tapir("UpdatePropertyDefinitions", {"propertyDefinitions": [
        {"propertyId": x["propertyId"], "availability": {"set": avail}} for x in r]})
    log(f"created property group '{GROUP}' with '{PROP_ID}', '{PROP_SRC}'")
    return r[0]["propertyId"], r[1]["propertyId"]


def library_part_exists(c, name):
    """True when a loaded Object library part is named exactly `name` (Tapir FindLibraryParts; cached on `c`)."""
    cache = _memo(c, "_libpart_cache", dict)
    key = name.lower()
    if key not in cache:
        cache[key] = any(p["name"].lower() == key + ".gsm" for p in c.tapir(
            "FindLibraryParts", {"kind": "Object", "nameContains": name}).get("libraryParts", []))
    return cache[key]


def resolve_material_names(wanted, available):
    """{wanted name: available name or None}. Exact name, then (case-insensitive) an available name containing the
    wanted word (shortest first), then bp.LAND_COVER_FALLBACK[name] resolved the same way (exact only)."""
    out = {}
    for w in wanted:
        hit = None
        if w in available:
            hit = w
        else:
            cands = sorted((a for a in available if w.lower() in a.lower()), key=lambda a: (len(a), a))
            hit = cands[0] if cands else None
        if hit is None:
            fb = bp.LAND_COVER_FALLBACK.get(w)
            hit = fb if fb in available else None
        out[w] = hit
    return out


def land_cover_material_guids(c, wanted):
    """{wanted name: guid} for land-cover materials, resolved by name in the open project (Grass -> any '*Grass*'
    building material, else Soil)."""
    by_name = building_material_map(c)
    res = resolve_material_names(set(wanted), set(by_name))
    return {w: by_name[n] for w, n in res.items() if n}


def classification_ids(c):
    systems = c.tapir("GetAllClassificationSystems")["classificationSystems"]
    sid = next(s["classificationSystemId"] for s in systems if s["name"] == CLASSIFICATION_SYSTEM)
    tree = c.call("API.GetAllClassificationsInSystem", {"classificationSystemId": sid})["classificationItems"]
    found = {}

    def walk(items):
        for it in items:
            ci = it["classificationItem"]
            if ci["id"] == "Site" and "_site" not in found:
                found["_site"] = ci["classificationItemId"]
            for k, nm in CLASS_ITEMS.items():
                if ci["id"] == nm and k not in found:
                    found[k] = ci["classificationItemId"]
            walk(ci.get("children", []))
    walk(tree)
    return sid, found


# ------------------------------------------------------------ element ops
def get_osm_ids(c, guids, pid):
    out = {}
    for part in chunks(guids, READ_CHUNK):
        r = c.call("API.GetPropertyValuesOfElements", {
            "elements": [eid(g) for g in part], "properties": [{"propertyId": pid}]})["propertyValuesForElements"]
        for g, x in zip(part, r):
            pv = (x.get("propertyValues") or [{}])[0].get("propertyValue", {})
            out[g] = pv.get("value") or ""
    return out


KIND_TYPE_ROLE = {"terrain": ("Mesh", "terrain"), "land_cover": ("Mesh", "land_cover"), "buildings": ("Slab", "buildings"),
                  "roads": ("Beam", "roads"), "trees": ("Object", "trees"), "shrubs": ("Object", "shrubs"),
                  "hedges": ("Object", "hedges"), "water": ("Slab", "water"), "building_parts": ("Slab", "buildings"),
                  "building_roofs": ("Roof", "buildings"), "railways": ("Beam", "railways"),
                  "rail_platforms": ("Slab", "railways"), "walls": ("Wall", "walls"), "railings": ("Railing", "walls"),
                  "piers": ("Slab", "piers"), "power": ("Column", "power"),
                  "contours_3d": ("Morph", "contours"),   # M11 morph terrain: 3D contour wire Morphs
                  "building_morphs": ("Morph", "buildings")}   # M10 LoD2 roof-surface Morph fallback
TERRAIN_ELEMENT_TYPE = {"stepped": "Slab", "morph": "Morph", "gdl": "Object"}   # contour_type -> terrain element type (default Mesh)
OSM_ELEMENT_TYPES = ("Slab", "Mesh", "Beam", "Object", "Wall", "Railing", "Roof", "Column", "Morph")


def _elements_by_type(c, types=OSM_ELEMENT_TYPES):
    """{element type: [guid]} via one GetElementsByType per type."""
    return {ty: [e["elementId"]["guid"] for e in c.tapir("GetElementsByType", {"elementType": ty}).get("elements", [])]
            for ty in types}


def _osm_tagged_by_type(c, pid):
    """(all elements by type, OSM-tagged elements by type) for OSM_ELEMENT_TYPES: tagged = non-empty 'OSM ID'."""
    by_type = _elements_by_type(c)
    all_els = [g for els in by_type.values() for g in els]
    osm_map = get_osm_ids(c, all_els, pid) if all_els else {}
    return by_type, {ty: [g for g in els if osm_map.get(g)] for ty, els in by_type.items()}


def _delete(c, guids):
    """DeleteElements in chunks -> [(guid, success)] for the elements Archicad answered for."""
    out = []
    for part in chunks(guids, WRITE_CHUNK):
        res = c.tapir("DeleteElements", {"elements": [eid(g) for g in part]}).get("executionResults", [])
        out += [(g, bool(r.get("success"))) for g, r in zip(part, res)]
    return out


def delete_previous(c, log, pid):
    by_type, tagged_by_type = _osm_tagged_by_type(c, pid)
    n = sum(ok for _, ok in _delete(c, [g for els in tagged_by_type.values() for g in els]))
    for ty in OSM_ELEMENT_TYPES:
        log(f"replace: {ty}: {len(by_type[ty])} existing, {len(tagged_by_type[ty])} OSM-tagged deleted")
    return n


def _exec_results(send, items, chunk_size):
    """Run a batched write `send(part) -> response` and return per-item executionResults aligned to `items`
    (missing results count as failures; surplus results are dropped so later slices stay aligned)."""
    out = []
    for part in chunks(items, chunk_size):
        res = send(part).get("executionResults", [])[:len(part)]
        out += res + [{"success": False}] * (len(part) - len(res))
    return out


def _tapir_exec_results(c, cmd, key, items, chunk_size=WRITE_CHUNK):
    """Batched Tapir write command -> per-item executionResults aligned to `items`."""
    return _exec_results(lambda part: c.tapir(cmd, {key: part}), items, chunk_size)


def _prop_exec_results(c, items, chunk_size=PROP_CHUNK):
    """Batched API.SetPropertyValuesOfElements -> per-item executionResults aligned to `items`."""
    return _exec_results(lambda part: c.call("API.SetPropertyValuesOfElements", {"elementPropertyValues": part}),
                         items, chunk_size)


def _failures_per_kind(results, groups, width=1):
    """results aligned to the concatenated items of `groups` [(kind, n_elements)], `width` items per element
    -> [(kind, number of failed items)]."""
    out, pos = [], 0
    for kind, n in groups:
        out.append((kind, sum(1 for r in results[pos:pos + n * width] if not r.get("success"))))
        pos += n * width
    return out


def _tag_prop_values(guids_ids, pid, sid, source):
    """'OSM ID' + 'Source' values, 2 per element. source: one string for all, or a list parallel to guids_ids."""
    vals = []
    for k, (g, osm) in enumerate(guids_ids):
        src = source[k] if isinstance(source, (list, tuple)) else source
        for p, v in ((pid, osm), (sid, src)):
            vals.append({**eid(g), "propertyId": p, "propertyValue": {"type": "string", "status": "normal", "value": v}})
    return vals


def _structural_function_value(g, prop, value="NonLoadBearingElement"):
    return {**eid(g), "propertyId": prop, "propertyValue": {
        "type": "singleEnum", "status": "normal", "value": {"type": "nonLocalizedValue", "nonLocalizedValue": value}}}


def set_road_beam_display(c, log, guids, report):
    """Try setting road beams to Home Story Only + Outlines Only via ModifyBeams.
    Stock Tapir 2.0.0 omits storyVisibility/displayOption from Beam schemas (error 4002);
    catch ArchicadError cleanly and record in report['roads_display']."""
    if not guids:
        return
    bad = 0
    for part in chunks(guids, 100):
        try:
            res = c.tapir("ModifyBeams", {"beamsWithDetails": [
                {**eid(g), "floorIndex": 0, **bp.ROAD_BEAM_DISPLAY} for g in part]}).get("executionResults", [])
            bad += len(part) - sum(1 for r in res[:len(part)] if r.get("success"))   # missing result = failed
        except ArchicadError as e:
            msg = (f"Tapir ModifyBeams does not expose beam storyVisibility/displayOption ({e}); "
                   "floorIndex=0 is set at creation; Home Story Only + Outlines Only are applied afterwards via ModifyBeams (Tapir)")
            report["roads_display"] = {"ok": False, "floorIndex": 0, "note": msg}
            log("  roads display: " + msg)
            return
    report["roads_display"] = {"ok": bad == 0, "floorIndex": 0, "updated": len(guids) - bad, "failed": bad,
                               **bp.ROAD_BEAM_DISPLAY}
    log(f"  roads display set (Home Story Only + Outlines Only) on {len(guids) - bad}/{len(guids)} beams")


def bboxes(c, guids):
    out = {}
    for part in chunks(guids, READ_CHUNK):
        r = c.tapir("Get3DBoundingBoxes", {"elements": [eid(g) for g in part]})["boundingBoxes3D"]
        for g, x in zip(part, r):
            out[g] = x.get("boundingBox3D")
    return out


# ------------------------------------------------------------ site-plan label
LABEL_KEY = "osmctx:label"
LABEL_LAYER = "Annotation - Text"


def label_text(site, today=None):
    """Accuracy disclaimer + data credits from site.json report.dem_accuracy / report.info.dem."""
    rep = site.get("report", {})
    acc = rep.get("dem_accuracy") or {}
    dem = (rep.get("info") or {}).get("dem") or {}
    today = today or dt.date.today().isoformat()
    attribution = (dem.get("attribution") or "").strip().rstrip(".")
    terrain = (f"Terrain: {acc.get('source') or dem.get('source') or 'unknown'} ({acc.get('kind', '?')}, "
               f"~{acc.get('horizontal_m', '?')} m grid, ±{acc.get('vertical_m', '?')} m).")
    parts = ["Site context only — not a survey.", terrain,
             "Buildings/roads/trees: © OpenStreetMap contributors (ODbL); positions ±1–5 m; "
             "heights from floor counts or defaults."]
    ov = (rep.get("info") or {}).get("overture") or {}
    if ov.get("applied") and ov.get("attribution"):
        parts.append(ov["attribution"] + ".")
    if attribution:
        parts.append(attribution + ".")
    parts.append(f"Imported {today}.")
    return " ".join(parts)


def _with_user_data_key(c, guids, key):
    """The guids (in order) whose element user data carries `key` (GetElementUserData in read chunks)."""
    out = []
    for part in chunks(guids, READ_CHUNK):
        r = c.tapir("GetElementUserData", {"elements": [eid(g) for g in part], "keys": [key]})["elementUserData"]
        out += [g for g, x in zip(part, r) if any(u.get("key") == key for u in x.get("userData", []))]
    return out


def delete_old_labels(c, log):
    """Remove previous site-plan labels: Text elements carrying user data key osmctx:label."""
    old = _with_user_data_key(c, _elements_by_type(c, ("Text",))["Text"], LABEL_KEY)
    n = sum(ok for _, ok in _delete(c, old))
    log(f"label: {n} old label(s) removed")
    return n


def place_label(c, log, site, all_layers, report, x=None, y=None, height_mm=2.5, width_mm=190):
    """Text element with the accuracy disclaimer below (outside) the south edge of the boundary, anchored at its south-west corner (block width 190 mm paper = 95 m at 1:500).
    Text elements cannot carry properties or classifications in Archicad (7203 'Element not supported'),
    so the replace marker is element user data osmctx:label (Tapir SetElementUserData)."""
    lidx = all_layers.get(LABEL_LAYER)
    if lidx is None:
        report["label"] = f"skipped: layer '{LABEL_LAYER}' missing"
        log(report["label"])
        return None
    ring = (site.get("boundary") or [[-50, -50], [50, 50]])
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    x = min(xs) if x is None else x      # left-aligned block just below (outside) the south edge, starts at the SW corner
    y = min(ys) - 18.0 if y is None else y   # clears the S-01 section marker (cut line to y = -60)
    c.tapir("ChangeWindow", {"windowType": "FloorPlan", "storyIndex": 0})  # before the delete: GetElementsByType only sees the active window
    delete_old_labels(c, log)
    text = label_text(site)
    r = c.tapir("CreateTexts", {"textsData": [{
        "coordinate": {"x": x, "y": y, "z": 0}, "text": text, "height": height_mm, "width": width_mm,
        "justification": "Left", "layerIndex": lidx, "floorIndex": 0}]})["elements"][0]
    if "elementId" not in r:
        report["label"] = f"failed: {r.get('error')}"
        log(report["label"])
        return None
    g = r["elementId"]["guid"]
    c.tapir("SetElementUserData", {"elementUserData": [{**eid(g), "userData": [{"key": LABEL_KEY, "value": "1"}]}]})
    report["label"] = {"guid": g, "layer": LABEL_LAYER, "position": [x, y], "text": text}
    log(f"label placed at ({x:.1f}, {y:.1f}) on '{LABEL_LAYER}': {len(text)} chars")
    return g


# ------------------------------------------------------------ contours (2D polylines + texts)
CONTOUR_KEY = "osmctx:contour"       # element user data on contour polylines/labels (2D elements hold no properties)
CONTOUR_BROWN = (0.6, 0.4, 0.2)      # preferred contour colour (plan convention: brown)
CONTOUR_LABEL_HEIGHT_MM = 3.0   # paper mm (text height = this x view scale; 1.5 m at 1:500)
CONTOUR_PEN_WEIGHTS = {"minor": 0.13, "major": 0.35}


def pick_contour_pens(pens, want=CONTOUR_PEN_WEIGHTS, style="colour"):
    """Pens of the project's pen table -> {"minor": idx, "major": idx, "label": idx, "fill": idx|None, "note": str}.
    pens = [{"index", "weight", "color": {red, green, blue}, "description"}]. Same colour family: the colour (group of
    identical rgb) closest to a brown that has a thin pen (<= 0.2 mm) and a thick pen (0.3-0.4 mm); within it the
    weights closest to `want` (minor 0.13, major 0.35). Label pen = major pen. Fill = first pure-white pen (text
    background that masks the line under the label).
    style="mono": major = black-ish grey pen with weight 0.3-0.4, minor = mid-grey pen with weight <= 0.2 (closest
    to `want`); falls back to the colour logic when no such grey pens exist. "none" has no project-default pen to
    read (CreatePolylines needs linePenIndex) -> callers use the colour logic."""
    if style == "mono":
        mono = _pick_contour_pens_mono(pens, want)
        if mono is not None:
            return mono
    groups = {}
    for p in pens:
        col = p["color"]
        groups.setdefault((round(col["red"], 3), round(col["green"], 3), round(col["blue"], 3)), []).append(p)
    best = None
    for col, ps in groups.items():
        thin = [p for p in ps if 0 < p["weight"] <= 0.2]
        thick = [p for p in ps if 0.3 <= p["weight"] <= 0.4]
        if not thin or not thick:
            continue
        d = sum((a - b) ** 2 for a, b in zip(col, CONTOUR_BROWN)) ** 0.5
        mn = min(thin, key=lambda p: (abs(p["weight"] - want["minor"]), p["index"]))
        mj = min(thick, key=lambda p: (abs(p["weight"] - want["major"]), p["index"]))
        score = d + abs(mn["weight"] - want["minor"]) + abs(mj["weight"] - want["major"])
        if best is None or score < best[0]:
            best = (score, mn, mj, col)
    white = next((p["index"] for p in sorted(pens, key=lambda p: p["index"]) if p["index"] > 0
                  and all(p["color"][k] > 0.999 for k in ("red", "green", "blue"))), None)
    if best is None:
        return {"minor": 1, "major": 7, "label": 7, "fill": white, "note": "no thin+thick pen of one colour: fell back to pens 1 / 7"}
    _, mn, mj, col = best
    return {"minor": mn["index"], "major": mj["index"], "label": mj["index"], "fill": white,
            "note": f"pen table colour rgb{col}: minor pen {mn['index']} ({mn['weight']} mm, {mn.get('description')}), "
                    f"major pen {mj['index']} ({mj['weight']} mm, {mj.get('description')})"}


def _pick_contour_pens_mono(pens, want):
    def rgb(p):
        c_ = p["color"]
        return (float(c_["red"]), float(c_["green"]), float(c_["blue"]))

    def dist(p, t):
        return sum((a - b) ** 2 for a, b in zip(rgb(p), t)) ** 0.5
    greys = [p for p in pens if p["index"] > 0 and max(rgb(p)) - min(rgb(p)) < 0.06]
    thick = [p for p in greys if 0.3 <= p["weight"] <= 0.4]
    thin = [p for p in greys if 0 < p["weight"] <= 0.2]
    if not thick or not thin:
        return None
    mj = min(thick, key=lambda p: (round(dist(p, bp.CONTOUR_RGB_MONO["major"]), 3),
                                   abs(p["weight"] - want["major"]), p["index"]))
    mn = min(thin, key=lambda p: (round(dist(p, bp.CONTOUR_RGB_MONO["minor"]), 3),
                                  abs(p["weight"] - want["minor"]), p["index"]))
    white = next((p["index"] for p in sorted(pens, key=lambda p: p["index"]) if p["index"] > 0
                  and all(p["color"][k] > 0.999 for k in ("red", "green", "blue"))), None)
    return {"minor": mn["index"], "major": mj["index"], "label": mj["index"], "fill": white,
            "note": f"mono: minor pen {mn['index']} ({mn['weight']} mm), major pen {mj['index']} ({mj['weight']} mm)"}


def load_pens(c):
    """Pens of the model-view pen table currently active in the open project -> (pens, table name)."""
    def load():
        gid = c.call("API.GetActivePenTables")["modelViewPenTableId"]["attributeId"]["guid"]
        t = c.call("API.GetPenTableAttributes", {"attributeIds": [{"attributeId": {"guid": gid}}]})["attributes"][0][
            "penTableAttribute"]
        return [x["pen"] for x in t["pens"]], t.get("name")
    return _memo(c, "_pens_cache", load)


def find_empty_fill(c):
    """AttributeId {'guid': ...} of the project's Empty (0%) Fill attribute, or None."""
    def load():
        ids = c.call("API.GetAttributesByType", {"attributeType": "Fill"})["attributeIds"]
        for part in chunks(ids, 400):   # stop at the first chunk holding it
            fills = c.call("API.GetFillAttributes", {"attributeIds": part})["attributes"]
            empty = next((x["fillAttribute"]["attributeId"] for x in fills
                          if x["fillAttribute"].get("subType") == "Empty"), None)
            if empty is not None:
                return empty
        return None
    return _memo(c, "_empty_fill_cache", load)


def _grey_pen_nearest(pens, target):
    """Index of the pen (index > 0) nearest `target` rgb, restricted to grey pens (chroma < 0.06) when any exist."""
    def rgb(p):
        c_ = p["color"]
        return (float(c_["red"]), float(c_["green"]), float(c_["blue"]))
    greys = [p for p in pens if max(rgb(p)) - min(rgb(p)) < 0.06] or pens
    return min(greys, key=lambda p: (sum((a - b) ** 2 for a, b in zip(rgb(p), target)), p["index"]))["index"]


def pick_plan_pens(pens, style="colour"):
    """Pick 2D fill pens closest to PLAN_FILL_RGB ('water' light blue, 'ground' light green,
    'buildings' light gray, 'roads' dark gray). style="mono": nearest GREY pen to PLAN_FILL_RGB_MONO instead."""
    valid = [p for p in pens if p.get("index", 0) > 0]
    if not valid:
        return dict.fromkeys(bp.PLAN_FILL_RGB)
    if style == "mono":
        return {k: _grey_pen_nearest(valid, t) for k, t in bp.PLAN_FILL_RGB_MONO.items()}

    def rgb(p):
        c_ = p["color"]
        return (float(c_["red"]), float(c_["green"]), float(c_["blue"]))

    def non_wb(p):
        r, g, b = rgb(p)
        return min(r, g, b) < 0.96 and max(r, g, b) > 0.04

    def chroma(p):
        r, g, b = rgb(p)
        return max(r, g, b) - min(r, g, b)

    def avg(p):
        r, g, b = rgb(p)
        return (r + g + b) / 3.0

    def pick(preds, target):
        cands = next((cs for cs in ([p for p in valid if pred(p)] for pred in preds) if cs), valid)
        return min(cands, key=lambda p: (sum((a - b) ** 2 for a, b in zip(rgb(p), target)), p["index"]))["index"]

    rules = {
        "water": [
            lambda p: non_wb(p) and rgb(p)[2] > rgb(p)[0] + 0.08 and rgb(p)[1] >= rgb(p)[0] and avg(p) <= 0.94,
            lambda p: non_wb(p) and rgb(p)[2] > rgb(p)[0],
            non_wb,
        ],
        "ground": [
            lambda p: non_wb(p) and rgb(p)[1] > rgb(p)[0] + 0.05 and rgb(p)[1] > rgb(p)[2] + 0.05 and avg(p) >= 0.45,
            lambda p: non_wb(p) and rgb(p)[1] > max(rgb(p)[0], rgb(p)[2]),
            non_wb,
        ],
        "buildings": [
            lambda p: non_wb(p) and chroma(p) < 0.05 and 0.62 <= avg(p) <= 0.88,
            lambda p: non_wb(p) and chroma(p) < 0.08 and 0.55 <= avg(p) <= 0.94,
            non_wb,
        ],
        "roads": [
            lambda p: non_wb(p) and chroma(p) < 0.05 and 0.15 <= avg(p) <= 0.52,
            lambda p: non_wb(p) and chroma(p) < 0.08 and avg(p) <= 0.58,
            non_wb,
        ],
    }
    return {k: pick(preds, bp.PLAN_FILL_RGB[k]) for k, preds in rules.items()}


PLAN_STYLE_TABLE = """plan_style decision table
preset  | slab/beam plan fills          | contour pens                         | terrain GO tint
colour  | PLAN_FILL_RGB (blue/green/..) | brown family, minor 0.13/major 0.35  | light green (TERRAIN_PLAN_TINT)
mono    | grey: bldg mid, road dark,    | black major (0.35) + grey minor      | very light grey (MONO_TERRAIN_PLAN_TINT)
        | water light, ground ~white   | (0.13); falls back to colour logic   |
none    | skipped (project/favorite     | colour logic (CreatePolylines needs  | skipped (no GO rule/combo read or write)
        | defaults stay)                | a pen; no project default to read)   |"""


def apply_plan_styles(c, log, created, report, contour_type="mesh", ground_slab_guids=None, plan_style="colour"):
    """Apply 2D cover/cut fill colours directly to created Slabs and road Beams using the project's pen table.
    plan_style: "colour" (default) | "mono" (grey palette, same mechanism) | "none" (no-op: elements keep the
    defaults from project/favorites). See PLAN_STYLE_TABLE."""
    if plan_style == "none":
        report["plan_styles"] = {"skipped": "plan_style none"}
        return
    try:
        pens_all, table = load_pens(c)
        empty = find_empty_fill(c)
    except ArchicadError as e:
        report["plan_styles"] = {"skipped": f"attribute lookup failed: {e}"}
        return
    if not empty:
        report["plan_styles"] = {"skipped": "no Empty fill attribute"}
        return
    plan_pens = pick_plan_pens(pens_all, plan_style) if plan_style == "mono" else pick_plan_pens(pens_all)
    slab_groups = [
        ([g for g, _ in created.get("water", [])], plan_pens.get("water")),
        ([g for g, _ in created.get("buildings", [])] + [g for g, _ in created.get("building_parts", [])],
         plan_pens.get("buildings")),
        (([g for g, _ in created.get("terrain", [])] if contour_type == "stepped" else []) +
         list(ground_slab_guids or ()), plan_pens.get("ground")),
    ]
    slab_updates = [
        {
            **eid(g),
            "floorFill": {"use": True, "fillId": empty, "foregroundPen": pen, "backgroundPen": pen,
                          "use3DHatching": False},
            "cutFillPen": {"overridden": True, "penIndex": pen},
            "cutFillBackgroundPen": {"overridden": True, "penIndex": pen},
        }
        for guids, pen in slab_groups if pen is not None for g in guids
    ]
    slabs_ok, slabs_err = 0, None
    if slab_updates:
        try:
            sres = _tapir_exec_results(c, "ModifySlabs", "slabsWithDetails", slab_updates, 100)
            slabs_ok = sum(1 for r in sres if r.get("success"))
        except ArchicadError as e:
            slabs_err = str(e)
    road_guids = [g for g, _ in created.get("roads", [])]
    rpen = plan_pens.get("roads")
    roads_ok, roads_err = 0, None
    if road_guids and rpen is not None:
        road_cover = {
            "use": True, "useFromSurface": False, "orientationComesFrom3D": False,
            "fillId": empty, "foregroundPen": rpen, "backgroundPen": rpen,
            "transformationType": "Global",
            "transformation": {"origin": {"x": 0.0, "y": 0.0}, "xAxis": {"x": 1.0, "y": 0.0},
                               "yAxis": {"x": 0.0, "y": 1.0}},
        }
        beam_updates = [
            {**eid(g), "coverFill": road_cover,
             "cutFillPen": {"overridden": True, "penIndex": rpen},
             "cutFillBackgroundPen": {"overridden": True, "penIndex": rpen}}
            for g in road_guids
        ]
        try:
            bres = _tapir_exec_results(c, "ModifyBeams", "beamsWithDetails", beam_updates, 100)
            roads_ok = sum(1 for r in bres if r.get("success"))
        except ArchicadError as e:
            roads_err = str(e)
    report["plan_styles"] = {
        "pens": plan_pens, "pen_table": table,
        "slabs": {"updated": slabs_ok, "total": len(slab_updates), **({"error": slabs_err} if slabs_err else {})},
        "roads": {"updated": roads_ok, "total": len(road_guids), **({"error": roads_err} if roads_err else {})},
    }
    log(f"plan styles: pens={plan_pens} slabs={slabs_ok}/{len(slab_updates)} roads={roads_ok}/{len(road_guids)}")


def contour_items(con, layer_idx, pens, height_mm=CONTOUR_LABEL_HEIGHT_MM):
    """Contours (osmctx.contours.build_contours result) -> Tapir payloads.
    -> {"minor": [CreatePolylines item], "major": [...], "label": [CreateTexts item]}"""
    def poly(m, pen):
        return {"coordinates": [{"x": x, "y": y} for x, y in m["points"]], "layerIndex": layer_idx, "floorInd": 0,
                "linePenIndex": pen}
    out = {"minor": [poly(m, pens["minor"]) for m in con["minor"]],
           "major": [poly(m, pens["major"]) for m in con["major"]], "label": []}
    for lb in con["labels"]:
        style = {"anchor": "MiddleMiddle", "fixedSize": False}   # fixedSize False: height = paper mm x view scale
        if pens.get("fill") is not None:
            style.update(usedFill=True, fillPenIndex=pens["fill"], contourOffset=0.3)
        out["label"].append({"coordinate": {"x": lb["x"], "y": lb["y"], "z": 0}, "text": lb["text"], "height": height_mm,
                             "angle": lb["angle"], "justification": "Center", "pen": pens["label"], "style": style,
                             "layerIndex": layer_idx, "floorIndex": 0})
    return out


def delete_old_contours(c, log):
    """Remove previous contours: PolyLine + Text elements carrying user data key osmctx:contour. Needs the floor plan
    window active (GetElementsByType only sees the active window) and Suspend Groups on (caller)."""
    c.tapir("ChangeWindow", {"windowType": "FloorPlan", "storyIndex": 0})
    by_type = _elements_by_type(c, ("PolyLine", "Text"))
    ty_of = {g: ty for ty, els in by_type.items() for g in els}
    old = _with_user_data_key(c, [g for els in by_type.values() for g in els], CONTOUR_KEY)
    n = dict.fromkeys(by_type, 0)
    for g, ok in _delete(c, old):
        n[ty_of[g]] += ok
    log(f"replace: contours: {n['PolyLine']} polylines + {n['Text']} labels deleted")
    return n


def build_contour_elements(c, log, site, report, lidx, interval, scale, pen_override, plan_style="colour"):
    """Compute + create contours (polylines minor/major, texts on majors). -> created dict for the groups."""
    con = ct.build_contours(site, interval, scale=scale)
    pens_all, table = load_pens(c)
    pens = pick_contour_pens(pens_all, style=plan_style) if plan_style == "mono" else pick_contour_pens(pens_all)
    for k, v in (pen_override or {}).items():
        if v is not None:
            pens[k] = int(v)
    items = contour_items(con, lidx["contours"], pens)
    # CreatePolylines/Texts go to the active window: always re-activate the plan right before creating (the window
    # switched to by delete_old_contours may have been changed by the user during the 3D part of the import)
    c.tapir("ChangeWindow", {"windowType": "FloorPlan", "storyIndex": 0})
    line_items = items["minor"] + items["major"]
    n_minor = len(items["minor"])
    g_lines, e_lines = run_batches(c, log, "CreatePolylines", "polylinesData", line_items, "contour polylines") if line_items else ([], [])
    g_lbl, e_lbl = run_batches(c, log, "CreateTexts", "textsData", items["label"], "contour label") if items["label"] else ([], [])
    ok_by_kind = {
        "minor": [x for x in g_lines[:n_minor] if x],
        "major": [x for x in g_lines[n_minor:] if x],
        "label": [x for x in g_lbl if x],
    }
    err_by_kind = {
        "minor": [er for er in e_lines if er["index"] < n_minor],
        "major": [{**er, "index": er["index"] - n_minor} for er in e_lines if er["index"] >= n_minor],
        "label": e_lbl,
    }
    tag_items = [{**eid(x), "userData": [{"key": CONTOUR_KEY, "value": kind}]}
                 for kind in ("minor", "major", "label") for x in ok_by_kind[kind]]
    tag_fail = dict(_failures_per_kind(_tapir_exec_results(c, "SetElementUserData", "elementUserData", tag_items),
                                       [(k, len(v)) for k, v in ok_by_kind.items()]))
    created, errs = {}, {}
    for kind in ("minor", "major", "label"):
        ok = ok_by_kind[kind]
        e = err_by_kind[kind]
        errs[kind] = {"errors": len(e), "first": e[0] if e else None, "tag_failures": tag_fail[kind]}
        created["contour_labels" if kind == "label" else f"contours_{kind}"] = [(x, kind) for x in ok]
    cnt = {k: len(v) for k, v in created.items()}
    report["contours"] = {"enabled": True, "interval": con["interval"], "auto_interval": con["auto_interval"],
                          "interval_source": con["interval_source"], "relief_m": con["relief_m"], "z_min": con["z_min"],
                          "z_max": con["z_max"], "major_every": con["major_every"],
                          "counts_planned": con["counts"], "created": cnt, "pens": pens, "pen_table": table,
                          "layer": (report.get("layer_names") or {}).get("contours"), "warnings": con["warnings"],
                          "errors": errs}
    log(f"contours: interval {con['interval']} m (auto {con['auto_interval']}), planned {con['counts']}, created {cnt}, pens {pens['minor']}/{pens['major']}")
    return created


# ------------------------------------------------------------ build
def build(site_json_path, setup_json_path=None, replace=True, url=None,
          log_path=None, report_path=None, save=True, dz=0.0,
          seo_operation="SubtractionUpwards", group=True, tree_mode="full3d", label=False,
          terrain_style="clean", land_cover=True, layer_mode="single", target_layer=None,
          terrain_points="points", contours=True, contour_interval=None, contour_scale=1.0, contour_pens=None,
          fence_tool="railing", site_data=None, contour_type="mesh", plan_style="colour", lod2_max_roofs=None,
          lod2_mode=None):
    t0 = time.time()
    setup_json_path = setup_json_path or paths.setup_json()
    log_path = log_path or paths.OUT_DIR / "build_log.txt"
    report_path = report_path or paths.OUT_DIR / "build_report.json"
    log = Log(log_path)
    c = Client(url) if url else Client()
    try:
        return _build_impl(c, log, t0, site_json_path, setup_json_path, report_path, replace, save, dz,
                           seo_operation, group, tree_mode, label, terrain_style, land_cover, layer_mode,
                           target_layer, terrain_points, contours, contour_interval, contour_scale, contour_pens,
                           fence_tool, site_data, contour_type, plan_style=plan_style,
                           lod2_max_roofs=lod2_max_roofs, lod2_mode=lod2_mode)
    finally:
        getattr(c, "close", lambda: None)()   # test fakes have no close()
        log.close()


def _build_impl(c, log, t0, site_json_path, setup_json_path, report_path, replace, save, dz,
                seo_operation, group, tree_mode, label, terrain_style, land_cover, layer_mode,
                target_layer, terrain_points, contours, contour_interval, contour_scale, contour_pens,
                fence_tool, site_data, contour_type="mesh", plan_style="colour", lod2_max_roofs=None, lod2_mode=None):
    if plan_style not in bp.PLAN_STYLES:
        raise ArchicadError(f"unknown plan_style '{plan_style}' {bp.PLAN_STYLES}")
    if contour_type not in bp.CONTOUR_TYPES:
        raise ArchicadError(f"unknown contour_type '{contour_type}' {bp.CONTOUR_TYPES}")
    if lod2_mode is not None and lod2_mode not in bp.LOD2_MODES:
        raise ArchicadError(f"unknown lod2_mode '{lod2_mode}' {bp.LOD2_MODES}")
    site = site_data if site_data is not None else json.loads(Path(site_json_path).read_text(encoding="utf-8"))
    setup = json.loads(Path(setup_json_path).read_text(encoding="utf-8"))
    rec = setup["recommended_attributes"]

    def guid(d, k="surface_guid"):
        return {"guid": d[k]} if d.get(k) else None

    def surf_override(d):
        g = guid(d)
        if not g:
            return {}
        ov = {"overridden": True, "attributeId": g}
        return {"topMaterial": ov, "sideMaterial": ov, "bottomMaterial": ov}

    report = {"site": str(site_json_path), "started": dt.datetime.now().isoformat(timespec="seconds"),
              "created": {}, "expected": {}, "errors": {}, "gaps": [], "checks": {}, "layers": {}}
    # M10 LoD2: measured buildings -> blocks (+ parts) with roof planes in the M6 shapes (pure; no-op without LoD2)
    site, lod2_info = bp.lod2_expand(site, max_roofs=lod2_max_roofs, mode=lod2_mode)
    if lod2_info["buildings"]:
        report["lod2"] = lod2_info
        log(f"LoD2: {lod2_info['buildings']} buildings, mode {lod2_info['mode']}, {lod2_info['blocks']} blocks, "
            f"{lod2_info['roofs']} roofs, {lod2_info['morphs']} morphs, {lod2_info['degraded']} degraded "
            f"(cap {lod2_info['max_roofs']})")
        for w in lod2_info["warnings"]:
            log("WARNING " + w)

    info = c.tapir("GetProjectInfo")
    log(f"project: {info.get('projectName')} ({info.get('projectPath')}) untitled={info.get('isUntitled')}")
    st = c.tapir("GetStories")["stories"][0]
    report["story0"] = {"index": st["index"], "level": st["level"]}
    if abs(st["level"]) > 1e-6:
        log(f"story 0 level {st['level']} != 0; Tapir z is absolute project z, no offset applied")

    if layer_mode == "single":
        lidx = single_layer(c, log, report, target_layer)
    elif layer_mode == "per_type":
        report["layer_mode"] = "per_type"
        lidx = ensure_layers(c, log, report, contours=bool(contours), roles=feature_roles_needed(site))
    else:
        raise ArchicadError(f"unknown layer_mode '{layer_mode}' (single|per_type)")
    report["layers"] = lidx
    rec = resolve_attribute_guids(c, rec, log)
    cls_sys, cls = classification_ids(c)
    # classification item ids, deduped (Flora & Fauna = trees, shrubs, hedges)
    pid, sid = ensure_properties(c, log, list({v["guid"]: v for k, v in cls.items() if k != "_site"}.values()))
    struct_prop = find_property(c, None, "Structural Function", builtin=True)
    has_part = lambda n: library_part_exists(c, n)  # noqa: E731
    tree_parts, tree_warn = bp.resolve_tree_parts(tree_mode, has_part)
    if tree_parts["broad"] is None:
        raise ArchicadError("no tree library part found")
    tree_kind_requested = bp.parse_tree_mode(tree_mode)[0]
    tree_kind0 = "light" if tree_warn else tree_kind_requested
    report["tree_mode"] = {"requested": tree_mode, "effective": tree_kind0, "parts": tree_parts, "warnings": tree_warn}
    report["gaps"] += tree_warn
    for w in tree_warn:
        log("WARNING " + w)
    shrub_parts, shrub_warn = bp.resolve_shrub_parts(tree_mode, has_part)
    report["shrub_parts"] = {**shrub_parts, "warnings": shrub_warn}
    report["gaps"] += shrub_warn
    for w in shrub_warn:
        log("WARNING " + w)
    source = bp.source_string(site)
    log(f"shrub parts {shrub_parts}; layers {lidx}; tree mode {tree_mode} parts {tree_parts}; source '{source}'")

    try:
        c.tapir("SetGeoLocation", bp.geo_location_payload(site))
        back = c.tapir("GetGeoLocation")
        report["geo_location"] = back
        log(f"survey point: {back['surveyPoint']['position']} crs={back['surveyPoint']['geoReferencingParameters'].get('crsName')}")
    except ArchicadError as e:
        report["gaps"].append(f"SetGeoLocation failed: {e}")
        log(f"geo location FAILED {e}")

    if replace:
        was = c.tapir("GetSuspendGroupsMode")["suspendGroups"]
        c.tapir("SetSuspendGroupsMode", {"suspendGroups": True})  # elements inside groups are otherwise not deletable singly
        try:
            report["deleted_previous"] = delete_previous(c, log, pid)
            report["deleted_contours"] = delete_old_contours(c, log)
        finally:
            c.tapir("SetSuspendGroupsMode", {"suspendGroups": was})

    created, expected = {}, {}
    pending_layers, pending_classes, pending_tags, pending_struct = [], [], [], []

    def finish(kind, guids, ids, errs, move_layer=None, after=None, sources=None):
        ok = [(g, i) for g, i in zip(guids, ids) if g]
        ok_src = [s_ for (g, _i), s_ in zip(zip(guids, ids), sources) if g] if sources else None
        created[kind] = ok
        report["created"][kind] = len(ok)
        report["errors"][kind] = errs[:10]
        if errs:
            log(f"  {kind}: {len(errs)} errors, first: {errs[0]}")
        if not ok:
            return
        gl = [g for g, _ in ok]
        if move_layer is not None:
            pending_layers.append((kind, gl, move_layer))
        # classify first: the OSM Context properties are only available on classified elements
        pending_classes.append((kind, gl, cls.get(kind)))
        pending_tags.append((kind, ok, ok_src or source))
        if after:
            after(gl)

    report["contour_type"] = contour_type
    bm = guid(rec["terrain"], "building_material_guid")
    # .get: the per-kind lookups used to run only when that kind had elements (no KeyError for a missing role)
    bm_building, bm_road, bm_water = (guid(rec.get(r, {}), "building_material_guid") for r in ("building", "road", "water"))
    if contour_type == "stepped":
        log("terrain (stepped terrace slabs)")
        step_items, step_info = bp.terrain_stepped_slabs(site, dz, interval=contour_interval, scale=contour_scale)
        for it in step_items:
            it.update(layerIndex=lidx["terrain"], floorIndex=0, **surf_override(rec["terrain"]))
            if bm:
                it["buildingMaterialId"] = bm
        expected["terrain"] = len(step_items)
        report["_terrain_expected_zmax"] = step_info["z_max_rel"] + dz
        report["terrain_stepped"] = {"interval": step_info["interval"], "steps": len(step_info["steps"]),
                                     "slabs": len(step_items)}
        log(f"  stepped contour: interval {step_info['interval']} m, {len(step_info['steps'])} step(s), {len(step_items)} slab(s)")
        g, e = run_batches(c, log, "CreateSlabs", "slabsData", step_items, "terrain step slab") if step_items else ([], [])
        finish("terrain", g, ["terrain"] * len(step_items), e, move_layer=None)
    elif contour_type == "morph":
        from osmctx import terrain_morph as tm
        log("terrain (Morph body + 3D contour wire Morphs)")
        m_items, m_info = tm.terrain_morph_items(site, dz, interval=contour_interval, scale=contour_scale)
        for w in m_info["warnings"]:
            report["gaps"].append(f"terrain morph: {w}")
        body, wires = m_items[:1], m_items[1:]
        for it in body:
            it.update(layerIndex=lidx["terrain"], floorIndex=0)
            if bm:
                it["buildingMaterialId"] = bm
        for it in wires:
            it.update(layerIndex=lidx["contours"], floorIndex=0)
        expected["terrain"] = len(body)
        report["terrain_morph"] = {k: m_info[k] for k in ("interval", "triangles", "vertices", "minor_lines", "major_lines")}
        log(f"  morph: {m_info['triangles']} triangles, {m_info['vertices']} vertices, "
            f"{len(wires)} contour wire morph(s), interval {m_info['interval']} m")
        g, e = run_batches(c, log, "CreateMorphs", "morphsData", body, "terrain morph") if body else ([], [])
        finish("terrain", g, ["terrain"] * len(body), e, move_layer=None)
        if wires:
            g, e = run_batches(c, log, "CreateMorphs", "morphsData", wires, "contour morph")
            finish("contours_3d", g, ["contours_3d"] * len(wires), e, move_layer=None)
    elif contour_type == "gdl":
        from osmctx import terrain_gdl as tg
        log("terrain (generated GDL object: solid + 3D/2D contours)")
        part, g_info = tg.terrain_gdl_part(site, interval=contour_interval, scale=contour_scale,
                                           source="OSM Context import")
        for w in g_info.get("warnings", []):
            report["gaps"].append(f"terrain gdl: {w}")
        params = {}
        try:   # contour pens from the project's pen table, same choice as the 2D polylines
            pens_all, _t = load_pens(c)
            pk = pick_contour_pens(pens_all, style=plan_style) if plan_style == "mono" else pick_contour_pens(pens_all)
            pk.update({k: int(v) for k, v in (contour_pens or {}).items() if v is not None})
            params = {"penMinor": pk["minor"], "penMajor": pk["major"], "penLabel": pk["label"]}
        except ArchicadError as e:
            report["gaps"].append(f"terrain gdl: pen lookup failed, part defaults used ({e})")
        report["terrain_gdl"] = {"part": part["name"], **{k: v for k, v in g_info.items() if k != "warnings"}}
        expected["terrain"] = 1
        try:
            res = c.tapir("CreateLibraryParts", tg.create_library_parts_params([part]))
            r0 = (res.get("libraryParts") or res.get("executionResults") or [{}])[0]
            lp_err = r0.get("error")
        except ArchicadError as e:
            lp_err = str(e)
        if lp_err:
            report["errors"]["terrain"] = [f"CreateLibraryParts: {lp_err}"]
            log(f"  terrain GDL part FAILED: {lp_err}")
        else:
            obj = tg.terrain_gdl_object(site, part["name"], dz, layer_index=lidx["terrain"], params=params)
            g, e = run_batches(c, log, "CreateObjects", "objectsData", [obj], "terrain gdl object")
            finish("terrain", g, ["terrain"], e, move_layer=None)
    else:
        log("terrain mesh")
        mesh = bp.terrain_mesh(site, dz, style=terrain_style, terrain_points=terrain_points)
        report["terrain_points"] = terrain_points
        mesh["floorIndex"] = 0
        if bm:
            mesh["buildingMaterialId"] = bm
        expected["terrain"] = 1
        log(f"  mesh outline {len(mesh['polygonCoordinates'])} pts, {len(mesh['sublines'])} sublines")
        g, e = run_batches(c, log, "CreateMeshes", "meshesData", [mesh], "mesh")
        finish("terrain", g, ["terrain"], e, move_layer=lidx["terrain"])

    has_sloped_ground = any(not bp.ground_is_slab(g_) for g_ in site.get("ground_surfaces", []))
    smp = bp.TerrainSampler(site) if ((land_cover and site.get("land_cover")) or has_sloped_ground) else None
    LC = bp.land_cover_meshes(site, dz, terrain_points=terrain_points, sampler=smp) if land_cover else []
    report["land_cover"] = {"enabled": bool(land_cover), "areas_in_site": len(site.get("land_cover") or []),
                            "meshes": len(LC), "kinds": sorted({m["kind"] for m in LC})}
    log(f"land cover: {len(LC)} draped mesh(es) from {len(site.get('land_cover') or [])} area(s)")
    expected["land_cover"] = len(LC)
    items = []
    if LC:
        mats = land_cover_material_guids(c, {m["material"] for m in LC})
        report["land_cover"]["building_materials"] = mats
        for m in LC:
            it = dict(m["item"], floorIndex=0)
            if mats.get(m["material"]):
                it["buildingMaterialId"] = {"guid": mats[m["material"]]}
            items.append(it)
    g, e = run_batches(c, log, "CreateMeshes", "meshesData", items, "land-cover mesh") if items else ([], [])
    report["_land_cover_meshes"] = LC
    finish("land_cover", g, [m["osm_id"] for m in LC], e, move_layer=lidx["land_cover"])
    report["land_cover"]["errors"] = e[:5]

    log("buildings")
    B = bp.active_buildings(site)   # outlines fully covered by building:part are not built as a block (OSM convention)
    items = []
    exp_roles = {}   # (Archicad element type, layer role) -> number created; verify() compares it with the model
    exp_feat = {}    # created kind -> expected bbox values per element (aligned with the plan), see verify()
    for b in B:
        it = bp.building_slab(b, dz)
        it.update(layerIndex=lidx["buildings"], floorIndex=0, **surf_override(rec["building"]))
        if bm_building:
            it["buildingMaterialId"] = bm_building
        items.append(it)
    expected["buildings"] = len(B)
    g, e = run_batches(c, log, "CreateSlabs", "slabsData", items, "slab")
    finish("buildings", g, [b["osm_id"] for b in B], e, sources=[bp.building_source(b, source) for b in B])
    report["buildings_block_skipped"] = len(site.get("buildings", [])) - len(B)
    report["buildings_overture"] = sum(1 for b in B if bp.is_overture(b["osm_id"]))
    report["buildings_osm"] = len(B) - report["buildings_overture"]

    log("roads")
    beam_items, beam_ids = [], []
    for r in site.get("roads", []):
        for it in bp.road_beams(r, dz):
            it.update(layerIndex=lidx["roads"], floorIndex=0)
            if bm_road:
                it["buildingMaterialId"] = bm_road
            beam_items.append(it)
            beam_ids.append(r["osm_id"])
    expected["roads"] = len(beam_items)
    g, e = run_batches(c, log, "CreateBeams", "beamsData", beam_items, "beam")

    def roads_after(gl):
        if struct_prop:
            pending_struct.append(("roads", gl))
        set_road_beam_display(c, log, gl, report)
    finish("roads", g, beam_ids, e, after=roads_after)

    log("trees")
    T = site.get("trees", [])
    items = []
    for t in T:
        it = bp.tree_object(t, bp.tree_lib_for(t, tree_parts), dz)
        it.update(layerIndex=lidx["trees"], floorIndex=0)
        items.append(it)
    expected["trees"] = len(T)
    report["tree_mode"]["counts_by_part"] = dict(Counter(it["libraryPartName"] for it in items))
    g, e = run_batches(c, log, "CreateObjects", "objectsData", items, "tree")
    finish("trees", g, [t["osm_id"] for t in T], e)
    created["tree_area_ids"] = sorted({t["osm_id"] for t in T if t.get("source_area")})
    created["tree_area_types"] = {t["osm_id"]: t.get("source_area_type", "forest") for t in T if t.get("source_area")}

    log("shrubs")
    S = site.get("shrubs", []) if shrub_parts["shrub"] else []
    items = [dict(bp.shrub_object(s_, shrub_parts["shrub"], dz), layerIndex=lidx["shrubs"], floorIndex=0) for s_ in S]
    expected["shrubs"] = len(S)
    g, e = run_batches(c, log, "CreateObjects", "objectsData", items, "shrub")
    finish("shrubs", g, [s_["osm_id"] for s_ in S], e)
    created["shrub_area_ids"] = sorted({s_["osm_id"] for s_ in S if s_.get("source_area")})
    report["shrub_parts"]["counts"] = {shrub_parts["shrub"]: len(S)} if S else {}

    log("hedges")
    schematic = tree_kind_requested == "2d"
    items, hedge_ids = [], []
    for h in (site.get("hedges", []) if shrub_parts["hedge"] else []):
        for it in bp.hedge_objects(h, shrub_parts["hedge"], dz, schematic):
            it.update(layerIndex=lidx["hedges"], floorIndex=0)
            items.append(it)
            hedge_ids.append(h["osm_id"])
    expected["hedges"] = len(items)
    report["hedge_items"] = items[:200]   # payloads (first 200), for the geometry check in verify()
    g, e = run_batches(c, log, "CreateObjects", "objectsData", items, "hedge segment")
    finish("hedges", g, hedge_ids, e)
    report["shrub_parts"]["hedge_segments"] = len(items)

    log("water")
    W = site.get("water", [])
    items = []
    for w in W:
        it = bp.water_slab(w, dz)
        it.update(layerIndex=lidx["water"], floorIndex=0, **surf_override(rec["water"]))
        if bm_water:
            it["buildingMaterialId"] = bm_water
        items.append(it)
    expected["water"] = len(W)
    g, e = run_batches(c, log, "CreateSlabs", "slabsData", items, "water slab")
    finish("water", g, [w["osm_id"] for w in W], e)

    # ------------------------------------------------------------ more features (M6)
    feat_names = {b.get("material") for b in site.get("barriers", []) if b.get("material")} | \
                 {g.get("material") for g in site.get("ground_surfaces", []) if g.get("material")}
    feat_mats = land_cover_material_guids(c, {n for n in feat_names if n}) if feat_names else {}
    report["feature_materials"] = {n: bool(feat_mats.get(n)) for n in sorted(n for n in feat_names if n)}

    def mat_for(name, key="buildingMaterialId"):
        return {key: {"guid": feat_mats[name]}} if name and feat_mats.get(name) else {}

    log("building parts + roofs")
    P = site.get("building_parts", [])
    items = []
    for p_ in P:
        it = bp.part_slab(p_, dz)
        it.update(layerIndex=lidx["buildings"], floorIndex=0, **surf_override(rec["building"]))
        if bm_building:
            it["buildingMaterialId"] = bm_building
        items.append(it)
    expected["building_parts"] = len(P)
    g, e = run_batches(c, log, "CreateSlabs", "slabsData", items, "part slab") if items else ([], [])
    finish("building_parts", g, [p_["osm_id"] for p_ in P], e, sources=[bp.building_source(p_, source) for p_ in P])
    created["building_parent"] = {p_["osm_id"]: p_["parent"] for p_ in P if p_.get("parent")}
    exp_feat["building_parts"] = [dict(zip(("zMin", "zMax"), bp.part_expected_bbox_z(p_, dz))) for p_ in P]
    R = [(o, ri) for o in B + P for ri in bp.roof_items(o, dz)]
    expected["building_roofs"] = len(R)
    report["roofs"] = {"count": len(R), "by_shape": dict(Counter(o["roof"]["shape"] for o, _ri in R))}
    g, e = run_batches(c, log, "CreateRoofs", "roofsData", [dict(ri["item"]) for _o, ri in R], "roof") if R else ([], [])
    if R and not any(g):
        report["gaps"].append("CreateRoofs failed (" + str(e[:1]) + "): buildings with roof:shape keep the flat block "
                              "(roof element missing); see report.errors.building_roofs")
    finish("building_roofs", g, [o["osm_id"] for o, _ri in R], e, move_layer=lidx["buildings"],
           sources=[bp.building_source(o, source) for o, _ri in R])
    exp_feat["building_roofs"] = [{k: v for k, v in ri["expected"].items() if k != "zMax" or ri["check_z"]} for _o, ri in R]
    # M10 LoD2: roof surfaces the Roof tool cannot take (near-vertical / non-planar) -> Surface Morphs
    M = [(o, mi) for o in B + P for mi in bp.lod2_morph_items(o, dz)]
    if M:
        items = [dict(mi["item"], layerIndex=lidx["buildings"]) for _o, mi in M]
        expected["building_morphs"] = len(M)
        g, e = run_batches(c, log, "CreateMorphs", "morphsData", items, "LoD2 roof morph")
        finish("building_morphs", g, [o["osm_id"] for o, _mi in M], e, move_layer=lidx["buildings"],
               sources=[bp.building_source(o, source) for o, _mi in M])
        exp_feat["building_morphs"] = [mi["expected"] for _o, mi in M]

    log("railways")
    beam_items, beam_ids, beam_tunnel = [], [], []
    for r in site.get("railways", []):
        for it in bp.road_beams(r, dz):
            it.update(layerIndex=lidx["railways"], floorIndex=0)
            if bm_road:
                it["buildingMaterialId"] = bm_road
            beam_items.append(it)
            beam_ids.append(r["osm_id"])
            beam_tunnel.append(bool(r.get("tunnel")))
    expected["railways"] = len(beam_items)
    g, e = run_batches(c, log, "CreateBeams", "beamsData", beam_items, "rail beam") if beam_items else ([], [])

    def rails_after(gl):
        if struct_prop:
            pending_struct.append(("railways", gl))
    finish("railways", g, beam_ids, e, after=rails_after)
    # solid operators: non-tunnel rail beams cut the terrain like roads (a tunnel beam would carve the terrain above it)
    created["rail_seo"] = [(gg, o) for gg, o, tun in zip(g, beam_ids, beam_tunnel) if gg and not tun]
    exp_feat["railways"] = [{"zMax": max(a[2], b2[2]) + dz} for r in site.get("railways", [])
                            for a, b2 in zip(r["points"], r["points"][1:])
                            if ((a[0] - b2[0]) ** 2 + (a[1] - b2[1]) ** 2) ** 0.5 >= bp.MIN_SEGMENT]
    PL = site.get("railway_platforms", [])
    items = []
    for p_ in PL:
        it = bp.platform_slab(p_, dz)
        it.update(layerIndex=lidx["railways"], floorIndex=0, **surf_override(rec["building"]))
        items.append(it)
    expected["rail_platforms"] = len(PL)
    g, e = run_batches(c, log, "CreateSlabs", "slabsData", items, "platform slab") if items else ([], [])
    finish("rail_platforms", g, [p_["osm_id"] for p_ in PL], e)
    exp_feat["rail_platforms"] = [dict(zip(("zMin", "zMax"), (p_["base_z"] + dz, p_["base_z"] + dz + p_["thickness"]))) for p_ in PL]

    log("walls, fences, guard rails")
    bars = site.get("barriers", [])
    wall_bars = [b for b in bars if b["tool"] == "wall" or fence_tool == "wall"]
    fence_bars = [b for b in bars if b["tool"] == "railing" and fence_tool != "wall"]

    def wall_batch(blist):
        its, ids, exps = [], [], []
        for b in blist:
            for it in bp.wall_items(b, dz):
                it.update(layerIndex=lidx["walls"], **mat_for(b.get("material")))
                its.append(it)
                ids.append(b["osm_id"])
                exps.append(dict(zip(("zMin", "zMax"), bp.barrier_expected_z(it, "wall"))))
        return its, ids, exps

    r_items, r_ids, r_exp = [], [], []
    for b in fence_bars:
        for it in bp.railing_items(b, dz):
            r_items.append(it)
            r_ids.append(b["osm_id"])
            r_exp.append(dict(zip(("zMin", "zMax"), bp.barrier_expected_z(it, "railing"))))
    g_r, e_r = run_batches(c, log, "CreateRailings", "railingsData", r_items, "railing") if r_items else ([], [])
    fell_back = False
    if r_items and not any(g_r):
        fell_back = True   # Railing tool unavailable (or rejects the payload): thin Wall instead, logged as a gap
        report["gaps"].append("Fences / guard rails as Walls: CreateRailings failed (" + str(e_r[:1]) + "); thin Wall used instead")
        log("  railings failed -> thin walls")
        wall_bars, r_items, r_ids, r_exp, e_r, g_r = wall_bars + fence_bars, [], [], [], [], []
    w_items, w_ids, w_exp = wall_batch(wall_bars)
    g_w, e_w = run_batches(c, log, "CreateWalls", "wallsData", w_items, "wall") if w_items else ([], [])
    expected["walls"] = len(w_items)
    expected["railings"] = len(r_items)
    report["barriers"] = {"fence_tool": "wall" if (fence_tool == "wall" or fell_back) else "railing",
                          "wall_ways": len(wall_bars), "railing_ways": len(fence_bars) if not fell_back else 0}
    finish("walls", g_w, w_ids, e_w)
    finish("railings", g_r, r_ids, e_r, move_layer=lidx["walls"])   # CreateRailings has no layerIndex
    exp_feat["walls"], exp_feat["railings"] = w_exp, r_exp

    log("ground surfaces")
    GS = [g_ for g_ in site.get("ground_surfaces", []) if bp.ground_is_slab(g_)]
    GM = bp.ground_meshes(site, dz, terrain_points=terrain_points, sampler=smp)
    items = []
    for g_ in GS:
        it = bp.ground_slab(g_, dz, float(site.get("ground_thickness") or 0.05))
        it.update(layerIndex=lidx["ground"], floorIndex=0, **mat_for(g_.get("material")))
        items.append(it)
    g_s, e_s = run_batches(c, log, "CreateSlabs", "slabsData", items, "ground slab") if items else ([], [])
    ground_slab_guids = [x for x in g_s if x]
    m_items = [dict(m["item"], floorIndex=0, **mat_for(m.get("material"))) for m in GM]
    g_m, e_m = run_batches(c, log, "CreateMeshes", "meshesData", m_items, "ground mesh") if m_items else ([], [])
    expected["ground"] = len(GS) + len(GM)
    finish("ground", list(g_s) + list(g_m), [g_["osm_id"] for g_ in GS] + [m["osm_id"] for m in GM],
           list(e_s) + [{**er, "index": er["index"] + len(items)} for er in e_m],   # mesh indexes follow the slabs
           move_layer=lidx["ground"])
    exp_roles[("Slab", "ground")] = sum(1 for x in g_s if x)
    exp_roles[("Mesh", "ground")] = sum(1 for x in g_m if x)
    exp_feat["ground"] = ([{"zMax": it["level"]} for it in items] +
                          [{k: v for k, v in bp.land_cover_expected_bbox(m).items() if k != "zMin"} for m in GM])
    report["ground"] = {"slabs": len(GS), "meshes": len(GM), "sloped_threshold_m": feat.GROUND_FLAT_SPREAD}

    log("piers + bridge decks")
    PI = site.get("piers", [])
    items = []
    for p_ in PI:
        it = bp.pier_slab(p_, dz)
        it.update(layerIndex=lidx["piers"], floorIndex=0, **surf_override(rec["building"]))
        items.append(it)
    expected["piers"] = len(PI)
    g, e = run_batches(c, log, "CreateSlabs", "slabsData", items, "pier slab") if items else ([], [])
    finish("piers", g, [p_["osm_id"] for p_ in PI], e)
    exp_feat["piers"] = [dict(zip(("zMin", "zMax"), (p_["z"] + dz - p_["thickness"], p_["z"] + dz))) for p_ in PI]

    log("power towers / poles")
    PT = site.get("power_towers", [])
    items = [dict(bp.power_column(t_, dz), layerIndex=lidx["power"]) for t_ in PT]
    expected["power"] = len(PT)
    g, e = run_batches(c, log, "CreateColumns", "columnsData", items, "power column") if items else ([], [])
    finish("power", g, [t_["osm_id"] for t_ in PT], e)
    exp_feat["power"] = [{"zMin": t_["z"] + dz, "zMax": t_["z"] + dz + t_["height"]} for t_ in PT]
    report["_feature_expected"] = exp_feat

    # Flush the deferred per-kind metadata: one combined, chunked pass per command, results sliced back per kind.
    # Order matters: classify before the tags (the OSM Context properties are only available on classified elements).
    def flush(groups, items, results_of, suffix, width=1):
        for kind, bad in _failures_per_kind(results_of(items), groups, width):
            report["errors"][kind + suffix] = bad

    if pending_layers:
        flush([(k, len(gl)) for k, gl, _ in pending_layers],
              [{**eid(g), "details": {"layerIndex": idx}} for _, gl, idx in pending_layers for g in gl],
              lambda it: _tapir_exec_results(c, "SetDetailsOfElements", "elementsWithDetails", it), "_layer_set")

    valid_classes = []
    for kind, gl, item_id in pending_classes:
        if not item_id:
            report["errors"][kind + "_classify"] = -1
        else:
            valid_classes.append((kind, gl, item_id))
    if valid_classes:
        flush([(k, len(gl)) for k, gl, _ in valid_classes],
              [{**eid(g), "classificationId": {"classificationSystemId": cls_sys, "classificationItemId": item_id}}
               for _, gl, item_id in valid_classes for g in gl],
              lambda it: _tapir_exec_results(c, "SetClassificationsOfElements", "elementClassifications", it), "_classify")

    if pending_tags:   # 2 property values per element (OSM ID + Source); failures count property values
        flush([(k, len(ok)) for k, ok, _ in pending_tags],
              [v for _, ok, src in pending_tags for v in _tag_prop_values(ok, pid, sid, src)],
              lambda it: _prop_exec_results(c, it), "_tags", width=2)
        log(f"  tags set on {sum(len(ok) for _, ok, _ in pending_tags)} elements, "
            f"failures {sum(report['errors'][k + '_tags'] for k, _, _ in pending_tags)}")

    if pending_struct and struct_prop:
        flush([(k, len(gl)) for k, gl in pending_struct],
              [_structural_function_value(g, struct_prop) for _, gl in pending_struct for g in gl],
              lambda it: _prop_exec_results(c, it), "_structural_function")

    # (Archicad element type, layer role) -> created count: verify() compares it with the model per type and layer
    for kind, (ty, role) in KIND_TYPE_ROLE.items():
        if kind == "terrain" and contour_type in TERRAIN_ELEMENT_TYPE:
            ty = TERRAIN_ELEMENT_TYPE[contour_type]
        if report["created"].get(kind):
            exp_roles[(ty, role)] = exp_roles.get((ty, role), 0) + report["created"][kind]
    report["expected"] = expected

    if contours and contour_type == "gdl":
        report["contours"] = "drawn by the terrain GDL object (2D + 3D); no separate polylines"
    elif contours:
        try:
            created.update(build_contour_elements(c, log, site, report, lidx, contour_interval, contour_scale,
                                                  contour_pens, plan_style))
            report["gaps"].append(CONTOUR_GAP)
        except ArchicadError as e:
            report["contours"] = f"failed: {e}"
            log(report["contours"])
    else:
        report["contours"] = {"enabled": False}

    report["gaps"] += known_gaps(struct_prop is not None, cls, site)
    verify(c, log, site, created, expected, report, pid, lidx, dz, exp_roles=exp_roles)
    report["plan_style"] = plan_style
    apply_plan_styles(c, log, created, report, contour_type=contour_type, ground_slab_guids=ground_slab_guids,
                      plan_style=plan_style)

    if seo_operation and seo_operation.lower() != "none":
        solid_ops(c, log, created, report, seo_operation)

    if group:
        make_groups(c, log, created, report, source.split("build ")[-1])

    try:
        apply_terrain_go(c, log, terrain_style, report, plan_style)
    except ArchicadError as e:
        report["terrain_go"] = f"failed: {e}"
        log(report["terrain_go"])

    if label:
        try:
            place_label(c, log, site, resolve_layers(c), report)
        except ArchicadError as e:
            report["label"] = f"failed: {e}"
            log(report["label"])

    if save:
        try:
            if info.get("isUntitled"):
                report["save"] = "project is untitled: not saved"
            else:
                c.tapir("SaveProject", {})
                report["save"] = f"saved to {info.get('projectPath')}"
        except ArchicadError as e:
            report["save"] = f"save failed: {e}"
        log(report["save"])

    report.pop("_land_cover_meshes", None)
    report.pop("_feature_expected", None)
    report.pop("_terrain_expected_zmax", None)
    report["seconds"] = round(time.time() - t0, 1)
    Path(report_path).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"done in {report['seconds']} s; report -> {report_path}")
    return report


TERRAIN_GO_RULE = "Site-Terrain earth"
GROUND_GO_RULE = "OSM Context - Ground"
OSM_GO_COMBO = "OSM Site Context"


def _ensure_ground_go_combo(c, log, ground_rule_id):
    """Attach the ground Mesh rule to the active GO combination (or create/activate 'OSM Site Context' when 'No Overrides' is active)."""
    combos = c.tapir("GetGraphicOverrideCombinations").get("graphicOverrideCombinations", [])
    active = c.tapir("GetActiveGraphicalOverrideCombination") or {}
    aname = active.get("name") or ""
    agid = (active.get("combinationId") or {}).get("guid")
    item = {"graphicOverrideRuleId": ground_rule_id}
    gid = ground_rule_id["guid"]

    def prepend_rule(combo):
        cur = combo.get("graphicOverrideRuleIds") or []
        if any((x.get("graphicOverrideRuleId") or {}).get("guid") == gid for x in cur):
            return False
        c.tapir("ModifyGraphicOverrideCombinations", {"graphicOverrideCombinations": [{
            "graphicOverrideCombinationId": combo["graphicOverrideCombinationId"],
            "graphicOverrideRuleIds": [item] + cur,
        }]})
        return True

    if not agid or aname == "No Overrides":
        combo = next((x for x in combos if x.get("name") == OSM_GO_COMBO), None)
        if combo is None:
            c.tapir("CreateGraphicOverrideCombinations", {"graphicOverrideCombinations": [{
                "name": OSM_GO_COMBO, "graphicOverrideRuleIds": [item],
            }]})
        else:
            prepend_rule(combo)
        c.tapir("ChangeWindow", {"windowType": "FloorPlan", "storyIndex": 0})
        c.tapir("SetActiveGraphicalOverrideCombination", {"name": OSM_GO_COMBO})
        log(f"activated GO combination '{OSM_GO_COMBO}' with '{GROUND_GO_RULE}'")
        return {"combination": OSM_GO_COMBO, "activated": True}

    combo = next((x for x in combos if (x.get("graphicOverrideCombinationId") or {}).get("guid") == agid), None)
    updated = prepend_rule(combo) if combo else False
    return {"combination": aname, "prepended": updated}


def apply_terrain_go(c, log, style, report, plan_style="colour"):
    """Keep ground Meshes light green on 2D plan via GO rule(s). 'Site-Terrain earth' (when present) and
    'OSM Context - Ground' (Mesh + Terrain/Site Geometry classification) use Empty fill + light green background.
    plan_style "mono": same rules, background = MONO_TERRAIN_PLAN_TINT (very light grey) instead of green.
    plan_style "none": nothing is read or written (no GO rule/combination change, no combo activation): a GO rule
    is a restyle, the 3D terrain is unaffected, and an untouched project GO setup is the safest 'no restyle'."""
    if plan_style == "none":
        report["terrain_go"] = "skipped: plan_style none"
        log(report["terrain_go"])
        return
    tint = bp.MONO_TERRAIN_PLAN_TINT if plan_style == "mono" else bp.TERRAIN_PLAN_TINT
    rules = c.tapir("GetGraphicOverrideRules", {"includeCriterion": False}).get("graphicOverrideRules", [])
    empty = find_empty_fill(c) if style != "legacy" else None
    every = {"cutFill": True, "coverFill": True, "draftingFill": True}
    if style != "legacy" and empty is None:
        report["terrain_go"] = "skipped: no Empty fill attribute"
        log(report["terrain_go"])
        return

    out = {"style": style}
    legacy_rule = next((r for r in rules if r.get("name") == TERRAIN_GO_RULE), None)
    if legacy_rule is not None:
        old = legacy_rule.get("style") or {}
        style_new = {"surfaceColor": old.get("surfaceColor") or {"red": 0.749019608, "green": 0.701960784, "blue": 0.6},
                     "surfaceAppliesTo": old.get("surfaceAppliesTo") or {"cutSurface": True, "uncutSurface": True}}
        if style != "legacy":
            style_new.update(fillType=empty, fillTypeAppliesTo=every, fillBackgroundColor=tint,
                             fillBackgroundPenAppliesTo=every)
        res = c.tapir("ModifyGraphicOverrideRules", {"graphicOverrideRules": [
            {"graphicOverrideRuleId": legacy_rule["graphicOverrideRuleId"], "style": style_new}]})["executionResults"]
        out["legacy_rule"] = TERRAIN_GO_RULE
        out["ok"] = bool(res and res[0].get("success"))
        log(f"terrain GO rule '{TERRAIN_GO_RULE}' set to {style}: ok={out['ok']}")

    if style != "legacy":
        gstyle = {"fillType": empty, "fillTypeAppliesTo": every,
                  "fillBackgroundColor": tint, "fillBackgroundPenAppliesTo": every}
        gcrit = {"and": [
            {"elementType": "Mesh"},
            {"or": [
                {"classification": "Terrain", "classificationSystem": CLASSIFICATION_SYSTEM},
                {"classification": "Site Geometry", "classificationSystem": CLASSIFICATION_SYSTEM},
            ]},
        ]}
        try:
            grule = next((r for r in rules if r.get("name") == GROUND_GO_RULE), None)
            if grule is None:
                c.tapir("CreateGraphicOverrideRules", {"graphicOverrideRules": [{
                    "name": GROUND_GO_RULE, "ruleGroupName": GROUP, "criterion": gcrit, "style": gstyle,
                }]})
                rules = c.tapir("GetGraphicOverrideRules", {"includeCriterion": False}).get("graphicOverrideRules", [])
                grule = next((r for r in rules if r.get("name") == GROUND_GO_RULE), None)
            else:
                c.tapir("ModifyGraphicOverrideRules", {"graphicOverrideRules": [{
                    "graphicOverrideRuleId": grule["graphicOverrideRuleId"], "style": gstyle,
                }]})
            if grule is not None:
                out["ground_rule"] = GROUND_GO_RULE
                out.update(_ensure_ground_go_combo(c, log, grule["graphicOverrideRuleId"]))
                out["ok"] = True
        except ArchicadError as e:
            out["ground_rule_note"] = str(e)
            log(f"ground GO rule skipped: {e}")

    report["terrain_go"] = out if len(out) > 1 else f"skipped: GO rule '{TERRAIN_GO_RULE}' missing"


def solid_ops(c, log, created, report, operation):
    """Terrain mesh = target, water slabs + road beams = operators."""
    if report.get("contour_type") == "stepped":
        report["solid_ops"] = {"skipped": "stepped contour model uses horizontal terrace slabs"}
        return
    ops = ([g for g, _ in created.get("water", [])] + [g for g, _ in created.get("roads", [])] +
           [g for g, _ in created.get("rail_seo", [])])
    if not created.get("terrain") or not ops:
        report["solid_ops"] = {"skipped": "no terrain or no operators"}
        return
    tgt = created["terrain"][0][0]
    links = bp.seo_links(tgt, ops, operation)
    fail = []
    for part in chunks(links, WRITE_CHUNK):
        res = c.tapir("CreateSolidElementLinks", {"solidLinks": part})["executionResults"]
        fail += [r for r in res if not r.get("success")]
    back = c.tapir("GetSolidElementLinks", {"elements": [eid(tgt)]})["solidLinks"][0]
    n = len(back.get("solidLinksWithTheGivenTarget", []))
    bb = bboxes(c, [tgt])[tgt]
    report["solid_ops"] = {"operation": operation, "requested": len(links), "failed": len(fail),
                           "first_error": fail[0] if fail else None, "links_on_mesh": n,
                           "mesh_bbox_after": bb}
    log(f"solid ops {operation}: {len(links)} requested, {len(fail)} failed, {n} links read back on mesh")


CONTOUR_GAP = ("No native mesh contour interval: Archicad's Mesh cannot show contours every X m with major/minor weights, so "
               "the site-plan contours are 2D Polylines (+ Texts for labels) computed from the terrain and regenerated "
               "with every import (tagged with user data osmctx:contour; replace=True removes the old ones).")


def known_gaps(have_struct, cls, site=None):
    g = [
        "Roads are Beams (no native road element): CreateBeams has no multi-segment option, so one slanted beam per point pair; "
        "slope via slantAngle, TopCenter anchor, 0.2 m high. Beam segment surfaces cannot be written by the API (building material only). "
        "Road beams are placed on story 0 (floorIndex=0) with Home Story Only + Outlines Only (applied via Tapir ModifyBeams; the OSMContext C++ add-on is parked until Graphisoft's developer ID arrives).",
        "Beam 'Structural Analytical Model Generation' is a read-only built-in property (isEditable=false): roads cannot be excluded "
        "from the analytical model through the API. Human step: select Site-Roads beams > Settings > Structural Analytical Model > uncheck.",
        "Mesh (terrain): CreateMeshes has no layerIndex (set afterwards with SetDetailsOfElements) and no surface parameter; "
        "terrain surface 'Earth - Brown' not assignable via API, only building material Soil.",
        "Mesh interior points: only via sublines; since 2026-11-17 each interior point is a single-point subline (a level point: carries the height, draws no ridge on the plan). terrain_points='rows' = old one polyline per grid row.",
    ]
    if not have_struct:
        g.append("Structural Function property not found; roads not set non-load-bearing.")
    missing = [k for k in CLASS_ITEMS if k not in cls]
    if missing:
        g.append(f"No classification item found for {missing}")
    site = site or {}
    if site.get("railways") or site.get("railway_platforms"):
        g.append("Railways are Beams like roads (no native rail/track element; width = ballast bed per track, 0.2 m high); "
                 "platforms are thin Slabs.")
    if site.get("barriers"):
        g.append("Walls use the Wall tool (Basic, centred, <= 5 m pieces, base = lower end on the terrain, no terrain-following "
                 "bottom edge: a Wall has a flat base). Fences / guard rails use the Railing tool: only height + base are "
                 "settable, the look (posts/rails/panels) is the project's Railing tool default; Railing has no layerIndex "
                 "(layer set afterwards) and no per-vertex z (one railing per piece).")
    if site.get("building_parts") or any(b.get("roof") for b in site.get("buildings", [])):
        g.append("Roofs: the Roof tool has gabled only as two single-plane roofs (pivot on the ridge); hipped/pyramidal = "
                 "multi-plane roof with ONE slope angle (exact for rectangles, approximate on other footprints); no per-edge "
                 "angles, no half-hipped/gambrel/mansard (those keep the flat block). Roof/part surfaces: building material only.")
    lp = site.get("lod2_plan") or {}
    if lp.get("buildings"):
        g.append("LoD2 buildings: Slab block(s) from the ground to the eave (one per eave height / BuildingPart, no wall "
                 "elements; gable-end triangles stay open as with OSM roofs) + one single-plane Roof per planar roof "
                 "surface (LoD2 surface = Roof top face, thickness " + str(bp.ROOF_THICKNESS) + " m, no holes: dormer "
                 "cut-outs ignored). Near-vertical (> " + str(feat.LOD2_VERTICAL_DEG) + " deg) or non-planar surfaces "
                 "-> Surface Morph (no Roof-tool equivalent)." +
                 (f" Roof cap lod2_max_roofs={lp.get('max_roofs')}: {lp.get('degraded')} building(s) built as flat "
                  f"volume-equivalent blocks." if lp.get("degraded") else "") +
                 (" Mode 'blocks': LoD2 buildings are flat volume-equivalent blocks only." if lp.get("mode") == "blocks"
                  else ""))
    if site.get("ground_surfaces"):
        g.append("Ground surfaces (parking/pitch/playground/pedestrian/square): flat -> 0.05 m Slab, sloped -> draped Mesh "
                 "(land-cover mechanism); Mesh/Slab surface (asphalt, grass) not settable via the API, only a building "
                 "material (Asphalt -> Concrete, Grass -> Soil fallbacks when the project lacks them).")
    if site.get("piers"):
        g.append("Piers / bridge areas: thin Slabs at the deck height (no native pier/bridge element).")
    if site.get("power_towers"):
        g.append("Power towers/poles: Column (no native pylon); power lines are only counted.")
    g.append("Classification: no road/paving item in 'Archicad Classification'; roads use 'Civil Element', "
             "water 'Geographic Element', buildings 'Massing' (closest items). Property availability is classification-based and not inherited: 'OSM Context' "
             "properties are made available for exactly these items (create with availability alone did not stick; UpdatePropertyDefinitions did).")
    return g


def verify(c, log, site, created, expected, report, pid, lidx, dz, tol=0.05, exp_roles=None):
    chk = report["checks"]
    layer_names = {}
    for k, v in lidx.items():   # shrubs + hedges share one layer: name it "shrubs+hedges"
        layer_names[v] = f"{layer_names[v]}+{k}" if v in layer_names else k
    counts = {}
    per_idx = {}
    _, tagged_by_type = _osm_tagged_by_type(c, pid)
    all_tagged = [g for ty in OSM_ELEMENT_TYPES for g in tagged_by_type[ty]]
    det_map = {}
    for part in chunks(all_tagged, READ_CHUNK):
        for d in c.tapir("GetDetailsOfElements", {"elements": [eid(g) for g in part]})["detailsOfElements"]:
            det_map[d["elementId"]["guid"]] = d
    for ty in OSM_ELEMENT_TYPES:
        tagged = tagged_by_type[ty]
        det = [det_map[g] for g in tagged if g in det_map]
        per = {}
        for d in det:
            nm = layer_names.get(d["layerIndex"], d["layerIndex"])
            per[nm] = per.get(nm, 0) + 1
            per_idx[(ty, d["layerIndex"])] = per_idx.get((ty, d["layerIndex"]), 0) + 1
        counts[ty] = {"tagged": len(tagged), "per_layer": per, "floor": sorted({d["floorIndex"] for d in det})}
    chk["model_counts"] = counts
    ok = all(report["created"].get(k, 0) == n for k, n in expected.items())
    # model totals per (element type, layer) must equal what was created for the roles on that layer (works for the
    # single layer and for per-type layers; roles sharing a layer are summed)
    want = {}
    for (ty, role), n in (exp_roles or {}).items():
        want[(ty, lidx[role])] = want.get((ty, lidx[role]), 0) + n
    mism = {f"{ty}@{li}": {"model": per_idx.get((ty, li), 0), "created": want.get((ty, li), 0)}
            for ty, li in sorted(set(want) | set(per_idx), key=str) if per_idx.get((ty, li), 0) != want.get((ty, li), 0)}
    chk["per_type_layer_mismatch"] = mism
    ok &= not mism
    chk["counts_match"] = bool(ok)
    log(f"verify counts: expected {expected}; model {json.dumps(counts)}; match={ok}")

    dev, n = [], [0]

    def cmp(kind, osm, name, got, want):
        n[0] += 1
        if abs(got - want) > tol:
            dev.append({"kind": kind, "osm_id": osm, "what": name, "got": round(got, 3), "want": round(want, 3)})

    # created[kind] holds only successes: per-element comparisons zip it with the plan, so they only run when nothing
    # of that kind failed (else every later element would be compared with its neighbour's expectation)
    B = bp.active_buildings(site)
    cb = created.get("buildings", [])
    cb = cb if len(cb) == len(B) else []
    ctr = created.get("trees", [])
    ctr = ctr if len(ctr) == len(site.get("trees", [])) else []
    cs = created.get("shrubs", [])
    chd = created.get("hedges", [])
    hitems = report.get("hedge_items") or []
    lcm = (report.get("_land_cover_meshes") or [])
    cl_lc = created.get("land_cover", [])
    feat_exp = report.get("_feature_expected") or {}
    cw = created.get("water", [])
    cw = cw if len(cw) == len(site.get("water", [])) else []
    tops = [max(a[2], b2[2]) + dz for r in site.get("roads", []) for a, b2 in zip(r["points"], r["points"][1:])
            if ((a[0] - b2[0]) ** 2 + (a[1] - b2[1]) ** 2) ** 0.5 >= bp.MIN_SEGMENT]   # one beam per long segment
    cr = created.get("roads", [])
    cr = cr if len(cr) == len(tops) else []

    terrain_guids = ([created["terrain"][0][0], created["terrain"][-1][0]] if len(created.get("terrain", [])) > 1
                     else [created["terrain"][0][0]] if created.get("terrain") else [])
    bb_guids = ([g for g, _ in cb] + [g for g, _ in ctr] +
                ([g for g, _ in cs] if len(cs) == len(site.get("shrubs", [])) else []) +
                ([g for g, _ in chd[:len(hitems)]] if expected.get("hedges") == len(chd) else []) +
                ([g for g, _ in cl_lc] if lcm and len(cl_lc) == len(lcm) else []) +
                [g for kind, expl in feat_exp.items() if created.get(kind) and len(created[kind]) == len(expl)
                 for g, _ in created[kind]] +
                [g for g, _ in cw] + [g for g, _ in cr] + terrain_guids)
    all_bb = bboxes(c, list(dict.fromkeys(bb_guids))) if bb_guids else {}

    for (g, osm), b in zip(cb, B):
        bb = all_bb.get(g)
        if not bb:
            continue
        z0, z1 = bp.building_expected_bbox_z(b, dz)
        cmp("building", osm, "zMin", bb["zMin"], z0)
        cmp("building", osm, "zMax", bb["zMax"], z1)
        cmp("building", osm, "xMin", bb["xMin"], min(p[0] for p in b["footprint"]))
        cmp("building", osm, "yMax", bb["yMax"], max(p[1] for p in b["footprint"]))
    tree_2d = (report.get("tree_mode") or {}).get("effective") == "2d"
    for (g, osm), t in zip(ctr, site.get("trees", [])):
        bb = all_bb.get(g)
        if bb:
            cmp("tree", osm, "zMin", bb["zMin"], t["z"] + dz)
            if not tree_2d:
                cmp("tree", osm, "zMax", bb["zMax"], t["z"] + dz + float(t.get("height") or 8.0))
    shrub_2d = (report.get("shrub_parts") or {}).get("shrub") == bp.SHRUB_PART_2D
    if len(cs) == len(site.get("shrubs", [])):
        for (g, osm), sh in zip(cs, site.get("shrubs", [])):
            bb = all_bb.get(g)
            if bb:
                cmp("shrub", osm, "zMin", bb["zMin"], sh["z"] + dz)
                if not shrub_2d:
                    cmp("shrub", osm, "zMax", bb["zMax"], sh["z"] + dz + max(0.1, float(sh["height"])))
    if expected.get("hedges") == len(chd):   # created list is only aligned with the payloads when no segment failed
        for (g, osm), it in zip(chd, hitems):
            bb, ex = all_bb.get(g), bp.hedge_expected_bbox(it)
            if bb:
                for k in ("xMin", "xMax", "yMin", "yMax", "zMin", "zMax"):
                    cmp("hedge", osm, k, bb[k], ex[k])
    if lcm and len(cl_lc) == len(lcm):
        for (g, osm), m in zip(cl_lc, lcm):
            bb, ex = all_bb.get(g), bp.land_cover_expected_bbox(m)
            if bb:
                for k in ("xMin", "xMax", "yMin", "yMax", "zMax"):
                    cmp("land_cover", osm, k, bb[k], ex[k])
    for kind, expl in feat_exp.items():   # more features: bbox per created element
        cl = created.get(kind, [])
        if not cl or len(cl) != len(expl):   # a failed element shifts the alignment: skip (counts already compared)
            continue
        for (g, osm), ex in zip(cl, expl):
            bb = all_bb.get(g)
            if bb:
                for k, v in ex.items():
                    cmp(kind, osm, k, bb[k], v)
    for (g, osm), w in zip(cw, site.get("water", [])):
        bb = all_bb.get(g)
        if bb:
            cmp("water", osm, "zMax", bb["zMax"], w["z"] + dz)
            cmp("water", osm, "zMin", bb["zMin"], w["z"] + dz - bp.WATER_THICKNESS)
    for (g, osm), zt in zip(cr, tops):
        bb = all_bb.get(g)
        if bb:
            cmp("road", osm, "zMax", bb["zMax"], zt)
    if created.get("terrain"):
        bb0 = all_bb.get(created["terrain"][0][0])
        bb1 = all_bb.get(created["terrain"][-1][0])
        zs = [p[2] for p in site["terrain"]["points"]]
        xs = [p[0] for p in site["terrain"]["boundary"]]
        want_zmax = report.get("_terrain_expected_zmax", max(zs) + dz)
        if bb1:
            cmp("terrain", "terrain", "zMax", bb1["zMax"], want_zmax)
        if bb0:
            cmp("terrain", "terrain", "zMin", bb0["zMin"], min(zs) + dz - bp.TERRAIN_SKIRT)
            cmp("terrain", "terrain", "xMin", bb0["xMin"], min(xs))
            cmp("terrain", "terrain", "xMax", bb0["xMax"], max(xs))
    chk["bbox"] = {"compared": n[0], "tolerance_m": tol, "n_deviations": len(dev), "deviations": dev[:40]}
    log(f"verify bbox: {n[0]} comparisons, {len(dev)} deviations > {tol} m")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("site_json")
    ap.add_argument("--setup", default="out/archicad_setup.json")
    ap.add_argument("--keep", action="store_true", help="do not delete previous OSM elements")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--url", default=None)
    ap.add_argument("--tree-mode", default="full3d",
                    help="tree representation: light (default) | 2d | full3d | lib:<library part name>")
    ap.add_argument("--terrain-style", default="clean", choices=bp.TERRAIN_STYLES,
                    help="terrain plan look: clean (default; user-defined ridges, no triangulation lines, flat earth tint) | legacy (AllSharp)")
    ap.add_argument("--plan-style", default="colour", choices=bp.PLAN_STYLES,
                    help="2D plan look: colour (default; coloured fills + terrain tint) | mono (black-and-white site plan, greys) | none (no restyle: project/favorite defaults, no GO tint)")
    ap.add_argument("--contour-type", default="mesh", choices=bp.CONTOUR_TYPES,
                    help="3D terrain model: mesh (default, real contour map) | stepped (stacked horizontal terrace slabs)")
    ap.add_argument("--no-land-cover", action="store_true", help="do not build draped meshes for beach/sand/wetland/tidalflat/mangrove areas")
    ap.add_argument("--terrain-points", default="points", choices=bp.TERRAIN_POINT_MODES,
                    help="mesh interior: points (default; single level points, no ridge on the plan) | rows (old: one leveling polyline per grid row)")
    ap.add_argument("--no-contours", action="store_true", help="do not draw site-plan contour lines (2D polylines + labels)")
    ap.add_argument("--contour-interval", type=float, default=None, metavar="M",
                    help="contour interval in m (default auto from relief: 0.25|0.5|1|2|5|10)")
    ap.add_argument("--contour-scale", type=float, default=1.0, help="multiplies the interval (super-light preset: 2)")
    ap.add_argument("--contour-pen-minor", type=int, default=None, help="pen index of minor contours (default: auto from the project pen table)")
    ap.add_argument("--contour-pen-major", type=int, default=None, help="pen index of major contours + labels (default: auto)")
    ap.add_argument("--fence-tool", default="railing", choices=("railing", "wall"),
                    help="fences + guard rails: Railing tool (default; falls back to a thin Wall when CreateRailings fails) | thin Wall")
    ap.add_argument("--no-group", action="store_true", help="do not create nested groups")
    ap.add_argument("--label", action="store_true", help="test helper: also place the disclaimer/credits text on the site plan (off by default; tool scope = site elements only)")
    ap.add_argument("--seo-op", default="SubtractionUpwards",
                    help="solid op terrain<-water/roads: Subtraction|SubtractionUpwards|SubtractionDownwards|Intersection|Addition|none")
    ap.add_argument("--layer", default=None, metavar="NAME",
                    help="put ALL OSM elements on this existing layer (default mode; default layer: 'Site & Landscape - General' if present, else first visible unlocked layer). No layers are created.")
    ap.add_argument("--per-type-layers", action="store_true",
                    help="one Site-* layer per type (creates missing layers + shows them in all layer combinations)")
    ap.add_argument("--lod2-mode", default=None, choices=bp.LOD2_MODES,
                    help="LoD2 buildings: roofs (blocks + one Roof per roof surface) | blocks (flat volume-equivalent "
                         "blocks only, super-light); default from the site settings, else roofs")
    ap.add_argument("--lod2-max-roofs", type=int, default=None, metavar="N",
                    help=f"cap on LoD2 Roof + Morph elements (default {bp.LOD2_MAX_ROOFS}; -1 = no cap); above it the "
                         "buildings with the most roof surfaces become flat blocks")
    a = ap.parse_args(argv)
    if a.layer and a.per_type_layers:
        ap.error("--layer and --per-type-layers are mutually exclusive")
    rep = build(a.site_json, a.setup, replace=not a.keep, url=a.url, save=not a.no_save, seo_operation=a.seo_op, group=not a.no_group, tree_mode=a.tree_mode, label=a.label, terrain_style=a.terrain_style, land_cover=not a.no_land_cover,
                layer_mode="per_type" if a.per_type_layers else "single", target_layer=a.layer,
                terrain_points=a.terrain_points, contours=not a.no_contours, contour_interval=a.contour_interval,
                contour_scale=a.contour_scale, fence_tool=a.fence_tool,
                contour_pens={"minor": a.contour_pen_minor, "major": a.contour_pen_major, "label": a.contour_pen_major},
                contour_type=a.contour_type, plan_style=a.plan_style, lod2_mode=a.lod2_mode,
                lod2_max_roofs=a.lod2_max_roofs)
    print(json.dumps({k: rep[k] for k in ("created", "expected")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
