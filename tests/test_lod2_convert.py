"""M10 step 6: LoD2 buildings end to end, offline: fetch wiring (mocked) -> convert (re-frame, conflation against OSM,
DEM base_z, report) -> limits estimate -> build_plan.lod2_expand -> builder against a fake client (no Archicad, no
Tapir, no network).

Real data: the OSM + DEM of tests/fixtures/tuebingen_station and the four BW LoD2 buildings of
tests/fixtures/lod2/bw_sample.gml (parsed by lod2.parse_citygml). The sample buildings stand ~600 m away from the
station fixture, so each one is moved (pure translation, all plane data included) to a chosen spot of the station site:
inside the largest OSM building (replace), in the open (add) and half over an OSM building (below the 50 % rule: both
stay)."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, Polygon

from osmctx import build_ac as bac
from osmctx import build_plan as bp
from osmctx import convert as C
from osmctx import features as F
from osmctx import fetch as FE
from osmctx import limits
from osmctx import lod2 as L2

HERE = Path(__file__).resolve().parent
STATION = HERE / "fixtures" / "tuebingen_station"
BW = HERE / "fixtures" / "lod2" / "bw_sample.gml"
UTM32 = 32632
ID_A, ID_B, ID_C, ID_D = "DEBW_001000szznD", "DEBW_001000szNaS", "DEBW_001000szzox", "DEBW_001000szAAf"

# site.json (without report) + report counts/warnings of the station fixture with default settings, produced by the
# code BEFORE the LoD2 wiring (no lod2.json in the cache -> must stay byte-identical)
STATION_BODY_SHA = "95958c747ac337b9cff286243bbac2187d551993032374995a0753e8bfb50866"
STATION_REPORT_SHA = "cf1d4646b74a63641106d8152620c4e883253ac73949dd0b0ab5d69c1a5964d5"


def _sha(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def sample_records() -> dict[str, dict]:
    return {r["id"]: r for r in L2.parse_citygml(BW, "bw", out_epsg=UTM32, origin=(0.0, 0.0))}


def move(rec: dict, tx: float, ty: float) -> dict:
    """rec translated by (tx, ty) metres (every 2D coordinate: footprints, planes, pivots, p0)."""
    fr = SimpleNamespace(epsg=UTM32, E0=-tx, N0=-ty)
    return C._L2Reframe(UTM32, (0.0, 0.0), fr)(rec)


def centre(rec):
    b = rec["bbox"]
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """cache dir with lod2.json, the Converter used to pick the spots, and the converted site."""
    tmp = tmp_path_factory.mktemp("lod2_conv")
    cache = tmp / "cache"
    shutil.copytree(STATION, cache)
    base = C.Converter(cache, {"include": {"lod2": False}})
    osm = [(el, g) for el, g in base._osm_building_items() if base.bpoly.covers(g.centroid)]
    osm.sort(key=lambda t: -t[1].area)
    E0, N0 = base.f.E0, base.f.N0
    recs = sample_records()
    allg = [g for _, g in base._osm_building_items()]
    tree = shapely.STRtree(allg)

    def spot_poly(rec, x, y):      # footprint (local frame) of rec when its centre sits at local (x, y)
        cx, cy = centre(rec)
        return Polygon([(px - cx + x, py - cy + y) for px, py in rec["footprint"]])

    def place(rec, x, y):
        cx, cy = centre(rec)
        return move(rec, E0 + x - cx, N0 + y - cy)

    # A: inside the largest OSM building, fully covered by it
    a_rec, big = recs[ID_A], None
    for el, g in osm:
        for p in (g.representative_point(), g.centroid):
            if spot_poly(a_rec, p.x, p.y).within(g):
                big, ax, ay = (el, g), p.x, p.y
                break
        if big:
            break
    assert big is not None, "no OSM building of the fixture holds the small LoD2 sample"
    used = [spot_poly(a_rec, ax, ay).buffer(8)]
    inner = base.bpoly.buffer(-20)

    def open_spot(rec):         # centre of the first grid cell where rec (+3 m) touches no OSM building / used spot
        minx, miny, maxx, maxy = inner.bounds
        for y in np.arange(miny, maxy, 4.0):
            for x in np.arange(minx, maxx, 4.0):
                q = spot_poly(rec, x, y).buffer(3)
                if inner.covers(q) and not tree.query(q, predicate="intersects").size and not any(q.intersects(u) for u in used):
                    used.append(q.buffer(3))
                    return x, y
        raise AssertionError("no open spot")

    bx, by = open_spot(recs[ID_B])
    dx, dy = open_spot(recs[ID_D])
    # C: centre on the boundary of another OSM building so that ~30 % of C overlaps it and nothing else
    c_rec, cspot, c_el = recs[ID_C], None, None
    for el, g in osm[1:]:
        if g.area < 150:
            continue
        for s in np.arange(0, g.exterior.length, 1.0):
            pt = g.exterior.interpolate(s)
            q = spot_poly(c_rec, pt.x, pt.y)
            hits = [int(i) for i in tree.query(q, predicate="intersects")]
            if len(hits) == 1 and abs(allg[hits[0]].area - g.area) < 1e-6 and 0.15 < q.intersection(g).area / q.area < 0.35 \
                    and base.bpoly.covers(q) and not any(q.intersects(u) for u in used):
                cspot, c_el = (pt.x, pt.y), el
                break
        if cspot:
            break
    assert cspot is not None, "no half-overlap spot"
    placed = [place(recs[ID_A], ax, ay), place(recs[ID_B], bx, by), place(c_rec, *cspot), place(recs[ID_D], dx, dy)]
    (cache / "lod2.json").write_text(json.dumps({"version": 1, "region": "bw", "crs": f"EPSG:{UTM32}", "origin": [0.0, 0.0],
                                                 "buildings": placed}), encoding="utf-8")
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    meta["lod2"] = {"status": "ok", "region": "bw", "dataset": "LGL-BW LoD2", "count": 4, "roof_planes": 9}
    (cache / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    cv = C.Converter(cache, None)
    site = cv.run()
    n_osm0 = len(C.Converter(cache, {"include": {"lod2": False}}).run()["buildings"])
    return SimpleNamespace(cache=cache, cv=cv, site=site, big=big, c_el=c_el, n_osm0=n_osm0, tmp=tmp, placed=placed,
                           osm_all=base._osm_building_items())


def by_id(site):
    out = {}
    for b in site["buildings"]:
        out.setdefault(b["osm_id"], []).append(b)
    return out


# ------------------------------------------------------------------ conflation
def test_conflation_replaces_keeps_adds(world):
    site, cv = world.site, world.cv
    ids = by_id(site)
    big_id = C.osm_id(world.big[0])
    c_id = C.osm_id(world.c_el)
    assert len(ids[big_id]) == 1 and ids[big_id][0]["source"] == "lod2:lgl-bw"          # A took the OSM id
    assert ids[big_id][0]["lod2"]["id"] == ID_A
    for lid in (ID_B, ID_D):
        assert ids[f"lod2/{lid}"][0]["source"] == "lod2:lgl-bw"                          # no OSM match: added
    assert ids["lod2/" + ID_C][0]["lod2"]["id"] == ID_C                                  # < 50 % of the smaller area
    assert len(ids[c_id]) == 1 and "source" not in ids[c_id][0]                          # ... so the OSM one stays
    k = [b for b in site["buildings"] if str(b.get("source", "")).startswith("lod2:")]
    c = site["report"]["counts"]
    assert len(k) == 4 and c["buildings_lod2"] == 4 and c["buildings_lod2_conflated"] == 1
    assert c["buildings_osm_replaced_by_lod2"] == 1 and c["buildings_osm"] == world.n_osm0 - 1
    assert c["buildings"] == world.n_osm0 + 3
    assert site["report"]["defaults_used"]["heights"]["lod2"] == 4


def test_report_line_and_info(world):
    li = world.site["report"]["info"]["lod2"]
    assert li["summary"] == f"{world.n_osm0 - 1} OSM + 0 Overture + 4 LoD2 (1 replaced)"
    assert li["applied"] and li["region"] == "bw" and li["dataset"] == "LGL-BW LoD2" and li["overlap_min"] == 0.5
    assert li["credit"] == "Datengrundlage: LGL, www.lgl-bw.de" and "DL-DE BY 2.0" in li["attribution"]
    assert any(w.startswith("Buildings: " + li["summary"]) and "Datengrundlage" in w for w in world.site["report"]["warnings"])
    assert "buildings: " + li["summary"] in C.format_report(world.site)


def test_replaced_osm_tags_carried_and_parts_dropped(world):
    el = world.big[0]
    b = by_id(world.site)[C.osm_id(el)][0]
    for k, v in el.get("tags", {}).items():
        if k in C.LOD2_CARRY_TAGS or k.startswith(("addr:", "name:")):
            assert b["tags"][k] == v
    assert b["tags"]["osmctx:conflated"] == "yes" and b["tags"]["osmctx:source"] == "lod2:lgl-bw"
    assert "roof:shape" not in b["tags"] and "height" not in b["tags"]
    assert not any(k.startswith("osmctx:remainder") for k in b["tags"])
    # OSM building:parts inside a replaced building are dropped (features.building_parts: the LoD2 parent wins)
    assert all(not p.get("parent") or p["parent"] != C.osm_id(el) for p in world.site["building_parts"])


def test_overlap_threshold_setting(world):
    """lod2_overlap_min = 0.1: the 30 %-overlap sample now replaces its OSM neighbour too."""
    cv = C.Converter(world.cache, {"lod2_overlap_min": 0.1})
    site = cv.run()
    ids = by_id(site)
    c_id = C.osm_id(world.c_el)
    assert ids[c_id][0]["lod2"]["id"] == ID_C and f"lod2/{ID_C}" not in ids
    assert site["report"]["counts"]["buildings_osm_replaced_by_lod2"] == 2


def test_base_z_from_dem_like_osm_buildings(world):
    cv = world.cv
    for b in world.site["buildings"]:
        if not b.get("lod2"):
            continue
        p = Polygon(b["footprint"], b["holes"])
        ext = np.asarray(shapely.segmentize(p.exterior, 5.0).coords)
        c = p.centroid
        z = cv.z_rel(np.append(ext[:, 0], c.x), np.append(ext[:, 1], c.y))
        assert b["base_z"] == pytest.approx(round(float(z.min()), 3), abs=1e-3)
        assert b["height_source"] == "lod2" and b["height"] > 2.0
        assert b["lod2"]["dataset"] == "LGL-BW LoD2" and b["lod2"]["region"] == "bw"
        assert "footprint" not in b["lod2"] and b["lod2"]["parts"]
    zs = {b["osm_id"]: b["base_z"] for b in world.site["buildings"] if b.get("lod2")}
    assert len(set(zs.values())) > 1            # heights vary with the terrain, not a constant


def test_records_are_in_the_convert_frame(world):
    """footprints of the converted records equal the placed (absolute UTM) footprints minus the site origin."""
    cv = world.cv
    out = {b["lod2"]["id"]: b for b in world.site["buildings"] if b.get("lod2")}
    for rec in world.placed:
        got = Polygon(out[rec["id"]]["footprint"])
        want = Polygon([(x - cv.f.E0, y - cv.f.N0) for x, y in rec["footprint"]])
        assert got.symmetric_difference(want).area < 0.2
        pl = out[rec["id"]]["lod2"]["parts"][0]["roof_planes"]
        if pl:
            assert Polygon(pl[0]["polygon"]).representative_point().within(want.buffer(0.1))


def test_other_utm_zone_reframe_keeps_geometry(world):
    """lod2.json in a different UTM zone: rigid map per record, plane z at every vertex survives (< 1 cm)."""
    rec = sample_records()[ID_C]
    t = L2.Transformer.from_crs(UTM32, 32633, always_xy=True)
    pts = lambda r: [list(t.transform(x, y)) for x, y in r]          # noqa: E731
    cx, cy = centre(rec)
    e0 = np.array(t.transform(cx, cy))
    ex = np.array(t.transform(cx + 10.0, cy)) - e0
    th = math.atan2(ex[1], ex[0])                       # rotation of the zone-32 grid axes in the zone-33 grid
    c, s_ = math.cos(th), math.sin(th)
    rot = lambda n: [c * n[0] - s_ * n[1], s_ * n[0] + c * n[1], n[2]]      # noqa: E731
    rec33 = dict(rec, footprint=pts(rec["footprint"]), holes=[])
    xs, ys = zip(*rec33["footprint"])
    rec33["bbox"] = [min(xs), min(ys), max(xs), max(ys)]
    rec33["parts"] = [dict(p, footprint=pts(p["footprint"]), holes=[], flat_roofs=[],
                           roof_planes=[dict(q, polygon=pts(q["polygon"]), holes=[], p0=list(t.transform(*q["p0"])),
                                             pivot=[list(t.transform(*a)) for a in q["pivot"]], normal=rot(q["normal"]))
                                        for q in p["roof_planes"]]) for p in rec["parts"]]
    out = C._L2Reframe(32633, (0.0, 0.0), SimpleNamespace(epsg=UTM32, E0=0.0, N0=0.0))(rec33)
    assert abs(Polygon(out["footprint"]).area - Polygon(rec["footprint"]).area) < 0.02 * Polygon(rec["footprint"]).area   # grid scale ~0.25 %
    pl_o = rec["parts"][0]["roof_planes"][0]
    pl_n = out["parts"][0]["roof_planes"][0]
    z_old = L2._plane_z(pl_o, np.array(pl_o["polygon"]))
    z_new = L2._plane_z(pl_n, np.array(pl_n["polygon"]))
    assert np.abs(z_old - z_new).max() < 0.02          # plane z at every vertex (grid scale ~0.25 % over 4 m)
    assert math.degrees(pl_n["tilt"]) == pytest.approx(math.degrees(pl_o["tilt"]))


# ------------------------------------------------------------------ ground mask / overture / off
def test_lod2_footprints_feed_the_ground_mask(world):
    bld, _w, _t, _wa = world.cv._ground_mask_features()
    n_osm = sum(1 for _ in world.osm_all)
    assert len(bld) >= n_osm + 4 - 1 or len(bld) > 0
    assert len(world.cv._bld_heights) == len(bld)
    lod = Polygon(by_id(world.site)["lod2/" + ID_B][0]["footprint"])
    assert any(g.contains(lod.centroid) for g in bld)


def test_overture_defers_to_lod2(world, tmp_path):
    """An Overture footprint over an LoD2 building is not added (LoD2 footprints count as existing buildings)."""
    from osmctx import overture as ovt
    d = tmp_path / "c"
    shutil.copytree(world.cache, d)
    b = by_id(world.site)["lod2/" + ID_B][0]
    ring = [[x + world.cv.f.E0, y + world.cv.f.N0] for x, y in b["footprint"]]
    lonlat = [list(world.cv.f.inv.transform(x, y)) for x, y in ring]
    ft = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [lonlat + [lonlat[0]]]},
          "properties": {"id": "ov-1", "sources": [{"dataset": "Microsoft ML Buildings"}], "height": 8.0}}
    (d / ovt.FILE).write_text(json.dumps({"type": "FeatureCollection", "features": [ft]}), encoding="utf-8")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    meta["overture"] = {"status": "ok", "count": 1}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    on = C.Converter(d, None).run()
    off = C.Converter(d, {"include": {"lod2": False}}).run()
    assert not any(str(x["osm_id"]).startswith("overture/") for x in on["buildings"])
    assert any(str(x["osm_id"]).startswith("overture/") for x in off["buildings"])


def test_include_lod2_false_means_no_lod2(world):
    site = C.Converter(world.cache, {"include": {"lod2": False}}).run()
    assert not any(b.get("lod2") or b.get("source") for b in site["buildings"])
    assert site["report"]["info"]["lod2"]["applied"] is False and site["report"]["info"]["lod2"]["reason"] == "off"
    assert "buildings_lod2" not in site["report"]["counts"]
    assert not any("LoD2" in w for w in site["report"]["warnings"])
    assert len(site["buildings"]) == world.n_osm0
    off = C.Converter(world.cache, {"include": {"buildings": False}}).run()
    assert off["buildings"] == []


def test_site_without_lod2_json_is_byte_identical_to_pre_lod2():
    """tuebingen_station (no lod2.json): site body and report counts/warnings equal the pre-M10-step-6 output."""
    cv = C.Converter(STATION, None)
    site = cv.run()
    body = {k: v for k, v in site.items() if k != "report"}
    assert _sha(body) == STATION_BODY_SHA
    assert _sha([site["report"]["counts"], site["report"]["warnings"]]) == STATION_REPORT_SHA
    assert site["report"]["info"]["lod2"]["applied"] is False


def test_settings_defaults_and_normalisation():
    s = C._merge_settings(None)
    assert s["lod2_max_roofs"] == 400 and s["lod2_overlap_min"] == 0.5 and s["lod2_allow_large"] is False
    assert s["include"]["lod2"] is True and s["include"]["lod2_roofs"] is True
    s = C._merge_settings({"lod2_overlap_min": "1.7", "lod2_max_roofs": "12", "lod2_allow_large": 1})
    assert (s["lod2_overlap_min"], s["lod2_max_roofs"], s["lod2_allow_large"]) == (1.0, 12, True)
    import server
    assert server.map_settings({"lod2_allow_large": True, "include": {"lod2": False, "lod2_roofs": False}}) == {
        "lod2_allow_large": True, "include": {"lod2": False, "lod2_roofs": False}}


# ------------------------------------------------------------------ builder: expand + fake client
def test_expand_on_converted_site(world):
    s2, info = bp.lod2_expand(world.site)
    assert info["buildings"] == 4 and info["mode"] == "roofs" and info["roofs"] > 0 and not info["over_cap"]
    assert info["datasets"] and info["morphs"] >= 0
    B = {b["osm_id"]: b for b in bp.active_buildings(s2)}
    big_id = C.osm_id(world.big[0])
    assert big_id in B and B[big_id]["height"] > 0
    assert any(p["parent"] == "lod2/" + ID_D for p in s2["building_parts"])      # the two-part sample


def test_super_light_gives_blocks(world):
    """include.lod2_roofs False (the super-light preset sets it) -> the builder makes flat blocks, no Roof / Morph."""
    assert limits.SUPER_LIGHT["include"]["lod2_roofs"] is False
    eff = limits.resolve(limits.SUPER_LIGHT_ABOVE_KM2 + 1.0, {"include": {"lod2": True}})["settings"]
    assert eff["include"]["lod2_roofs"] is False and eff["include"]["lod2"] is True
    site = C.Converter(world.cache, {"include": eff["include"]}).run()
    assert bp.lod2_settings(site) == (400, "blocks")
    s2, info = bp.lod2_expand(site)
    assert info["mode"] == "blocks" and info["roofs"] == info["morphs"] == 0 and info["blocks"] == 4
    assert info["degraded"] == 0 and bp.feature_expected(s2)["roofs"] == 0
    assert not any("roof" in b for b in s2["buildings"] if F.is_lod2(b))


@pytest.fixture(scope="module")
def fake_build(world):
    from test_lod2_payloads import FakeAC
    tmp = world.tmp
    setup = tmp / "setup.json"
    setup.write_text(json.dumps({"recommended_attributes": {
        "terrain": {"surface": "Earth - Brown", "building_material": "Soil"},
        "building": {"surface": "Concrete - 02", "building_material": "GENERIC - ENVIRONMENT"},
        "road": {"building_material": "Concrete"}, "water": {"building_material": "Sand"}}}), encoding="utf-8")
    fake = FakeAC()
    orig = bac.Client
    bac.Client = lambda *a, **k: fake
    try:
        rep = bac.build(str(tmp / "site.json"), str(setup), log_path=str(tmp / "log.txt"),
                        report_path=str(tmp / "rep.json"), save=False, group=True, label=False, tree_mode="light",
                        contours=False, site_data=dict(world.site, **{k: [] for k in ("barriers", "railways", "railway_platforms", "roads", "trees", "land_cover", "ground_surfaces")}))   # the fake answers only the buildings + terrain calls
    finally:
        bac.Client = orig
    return fake, rep


def test_builder_runs_on_converted_site(world, fake_build):
    fake, rep = fake_build
    assert rep["lod2"]["buildings"] == 4 and rep["lod2"]["roofs"] > 0
    assert not any(v for v in rep["errors"].values()), rep["errors"]
    assert rep["checks"]["counts_match"]
    roofs = [e for e in fake.elements.values() if e["type"] == "Roof"]
    assert rep["roofs"]["by_shape"]["lod2"] == rep["lod2"]["roofs"] and len(roofs) >= rep["lod2"]["roofs"]   # + the OSM roof:shape roofs
    sid = fake.prop_ids[1]
    src = {fake.props[g][sid] for g, e in fake.elements.items() if e["type"] == "Slab" and g in fake.props}
    assert "LoD2 (LGL-BW LoD2)" in src and any(s.startswith("OSM /") for s in src)
    pid = fake.prop_ids[0]
    ids = {fake.props[g][pid] for g, e in fake.elements.items() if e["type"] == "Slab" and g in fake.props}
    assert C.osm_id(world.big[0]) in ids and "lod2/" + ID_B in ids


# ------------------------------------------------------------------ limits / server
def test_limits_estimate_uses_real_lod2_elements(world):
    recs = json.loads((world.cache / "lod2.json").read_text(encoding="utf-8"))["buildings"]
    est = limits.lod2_estimate({"include": {}}, records=recs)
    site = {"buildings": [{"osm_id": f"lod2/{r['id']}", "source": r["source"], "footprint": r["footprint"],
                           "holes": r["holes"], "base_z": 0.0, "height": r["height"], "tags": {}, "lod2": r} for r in recs]}
    want = bp.lod2_counts(site)
    assert est["elements"] == want["elements"] and est["buildings"] == 4 and est["basis"] == "records"
    assert est["elements"] > est["buildings"]                      # roofs count, not 1 per building
    blocks = limits.lod2_estimate({"include": {"lod2_roofs": False}}, records=recs)
    assert blocks["mode"] == "blocks" and blocks["elements"] == 4
    capped = limits.lod2_estimate({"include": {}, "lod2_max_roofs": 2}, records=recs)
    assert capped["elements"] < est["elements"]
    assert limits.lod2_estimate({"include": {"lod2": False}}, records=recs) is None
    assert limits.lod2_estimate({"include": {"buildings": False}}, records=recs) is None
    assert limits.lod2_estimate({}, records=[]) is None
    tiles = {"source": "x", "dgm1_total": 0, "dgm1_missing": 0, "ndom_total": 0, "ndom_missing": 0, "copernicus_tiles": 0}
    counts = {"buildings": 10, "roads": 0, "trees": 0, "water": 0}
    e0 = limits.estimate(1.0, counts, "cache", tiles)["elements"]
    e1 = limits.estimate(1.0, counts, "cache", tiles, lod2=est)
    assert e1["elements"]["lod2"] == est["elements"] and e1["elements"]["buildings"] == 6
    assert e1["elements"]["total"] == e0["total"] - 4 + est["elements"]
    assert any("LoD2 3D buildings" in n for n in e1["notes"])


def test_limits_prior_by_region():
    tue = (48.51, 9.05, 48.52, 9.06)
    est = limits.lod2_estimate({"include": {}}, bb=tue)
    assert est["basis"] == "prior" and est["region"] == "bw" and est["buildings"] > 0
    assert est["elements"] <= est["buildings"] + 400
    assert limits.lod2_estimate({"include": {"lod2_roofs": False}}, bb=tue)["elements"] == est["buildings"]
    assert limits.lod2_estimate({"include": {}}, bb=(-6.16, 106.04, -6.15, 106.05)) is None      # Jakarta: no source


def test_server_credit_count_and_fetch_flags(world, monkeypatch):
    import server
    li = world.site["report"]["info"]["lod2"]
    d = server.disclaimer(None, None, None, {"applied": False}, li)
    assert "3D buildings (LoD2): Datengrundlage: LGL, www.lgl-bw.de (DL-DE BY 2.0)" in d["credits"]
    assert not any("LoD2" in c for c in server.disclaimer(None, None, None, {"applied": False}, {"applied": False})["credits"])
    assert not any("LoD2" in c for c in server.disclaimer()["credits"])
    recs = server.count_lod2(world.cache)
    assert len(recs) == 4
    cx, cy = centre(recs[0])
    lon, lat = world.cv.f.inv.transform(cx, cy)
    assert len(server.count_lod2(world.cache, (lat - 1e-4, lon - 1e-4, lat + 1e-4, lon + 1e-4))) >= 1
    assert server.count_lod2(world.cache, (0, 0, 1, 1)) == []
    assert server.count_lod2(world.tmp) is None
    seen = {}
    monkeypatch.setattr(FE, "fetch", lambda b, **kw: seen.update(kw) or world.cache)
    server._jobs["j1"] = {}
    server._run_fetch_job("j1", {}, "auto", False, True, False, True)
    assert seen["lod2"] is False and seen["lod2_allow_large"] is True
    seen.clear()
    server._jobs["j2"] = {}
    server._run_fetch_job("j2", {}, "auto")
    assert "lod2" not in seen and "lod2_allow_large" not in seen        # defaults: fetch() keeps lod2=True / no large


def test_ui_has_the_lod2_controls():
    html = (HERE.parent / "ui" / "index.html").read_text(encoding="utf-8")
    js = (HERE.parent / "ui" / "app.js").read_text(encoding="utf-8")
    for needle in ('id="i_lod2"', 'id="i_lod2_roofs"', 'id="s_lod2_large"', "Use LoD2 3D buildings where available (BW, NL, CH, JP)",
                   "LoD2 roofs as Roof elements", "Allow large LoD2 downloads (Japan, Switzerland"):
        assert needle in html
    assert "lod2: true, lod2_roofs: true" in js and "lod2_allow_large: false" in js and "s_lod2_large" in js
    assert "l2.summary" in js


# ------------------------------------------------------------------ fetch wiring (mocked)
def _cache_copy(tmp_path):
    d = tmp_path / "cache" / "k"
    shutil.copytree(STATION, d)
    return d


def _stub_fetch(monkeypatch, calls, result=None, region="bw"):
    monkeypatch.setattr(FE, "refresh_osm", lambda d: None)
    monkeypatch.setattr(FE.L2, "detect_region", lambda bb, head=None: region)

    def fake(bb, root, out, timeout=None, **kw):
        calls.append(kw)
        if (result or {}).get("status") != "failed":
            Path(out).write_text("{}", encoding="utf-8")
        return result or {"status": "ok", "region": "bw", "dataset": "LGL-BW LoD2", "count": 3, "warnings": ["w1"]}
    monkeypatch.setattr(FE.L2, "try_fetch", fake)


def test_fetch_cached_adds_lod2_once(tmp_path, monkeypatch):
    d = _cache_copy(tmp_path)
    calls = []
    _stub_fetch(monkeypatch, calls)
    FE._fetch_cached(d, d.parent, False, False, lod2=True, lod2_allow_large=True)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["lod2"]["status"] == "ok" and "LoD2: w1" in meta["warnings"] and (d / "lod2.json").exists()
    assert calls == [{"region": "bw", "allow_large": True}]
    FE._fetch_cached(d, d.parent, False, False, lod2=True)            # valid lod2.json + ok block: no new download
    assert len(calls) == 1


def test_fetch_cached_off_uncovered_and_failed(tmp_path, monkeypatch):
    d = _cache_copy(tmp_path)
    calls = []
    _stub_fetch(monkeypatch, calls)
    FE._fetch_cached(d, d.parent, False, False, lod2=False)
    assert not calls and "lod2" not in json.loads((d / "meta.json").read_text(encoding="utf-8"))
    _stub_fetch(monkeypatch, calls, region=None)                      # no LoD2 source here: nothing recorded
    FE._fetch_cached(d, d.parent, False, False, lod2=True)
    assert not calls and "lod2" not in json.loads((d / "meta.json").read_text(encoding="utf-8"))
    _stub_fetch(monkeypatch, calls, result={"status": "failed", "error": "LargeDownload: big"})
    FE._fetch_cached(d, d.parent, False, False, lod2=True)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["lod2"]["status"] == "failed" and any(w.startswith("LoD2 buildings not fetched") for w in meta["warnings"])
    assert L2.needs_fetch(d, meta)                                    # retried on the next fetch (e.g. large allowed)


def test_fetch_fresh_runs_lod2_beside_osm(tmp_path, monkeypatch):
    """_fetch_fresh with every network piece mocked: the LoD2 block lands in meta.json, OSM does not wait for it."""
    from osmctx.boundary import square_boundary
    calls = []
    _stub_fetch(monkeypatch, calls)
    monkeypatch.setattr(FE, "overpass", lambda q: ({"elements": []}, "mock"))
    monkeypatch.setattr(FE, "post_process_osm", lambda d, data, url, meta: None)

    def fake_dem(*a, **k):
        shutil.copy(STATION / "dem.tif", a[1])
        return {"complete": True, "warnings": [], "tiles": 1, "dates": []}
    monkeypatch.setattr(FE, "fetch_bw_dgm1", fake_dem)
    monkeypatch.setattr(FE, "in_bw_envelope", lambda bb: True)
    d = FE.fetch(square_boundary(48.5157, 9.0555, 100), tmp_path / "cache", dem_source="bw_dgm1", ndom=False,
                 overture=False, lod2=True, lod2_allow_large=False)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["lod2"]["count"] == 3 and (d / "lod2.json").exists() and calls and calls[0]["allow_large"] is False
