"""Site-plan contours (osmctx/contours.py), single-point mesh sublines, contour grouping/replace/pens/UI wiring."""
import math

import pytest
from fastapi.testclient import TestClient
from shapely.geometry import LineString

import server
from osmctx import build_ac, build_plan as bp, contours as ct
from tests.test_build_plan import mini_site


def grid_site(f, n=50.0, s=2.0, e0=100.0):
    """Square site [-n, n]^2: densified outline ring + interior grid like the converter (interior stays 0.3 s off the edge)."""
    m = int(round(2 * n / s))
    ring = ([[-n + i * s, -n] for i in range(m)] + [[n, -n + i * s] for i in range(m)] +
            [[n - i * s, n] for i in range(m)] + [[-n, n - i * s] for i in range(m)])
    xs = [-n + s / 2 + i * s for i in range(m)]
    inner = [[x, y] for x in xs for y in xs if abs(x) < n - 0.3 * s and abs(y) < n - 0.3 * s]
    pts = [[x, y, f(x, y)] for x, y in ring + inner]
    return {"origin": {"elevation": e0}, "boundary": ring, "terrain": {"boundary": ring, "points": pts}}


PLANE = lambda x, y: 0.1 * x + 3.0          # noqa: E731   z abs 103 +- 5 -> 98..108
CONE = lambda x, y: 10.0 - 0.2 * math.hypot(x, y)   # noqa: E731
SADDLE = lambda x, y: 0.3 + (x * x - y * y) / 60.0  # noqa: E731


def all_lines(r):
    return r["minor"] + r["major"]


# ---------------------------------------------------------------- interval
def test_auto_interval_choice():
    assert ct.auto_interval(0.2) == 0.25
    assert ct.auto_interval(9.9) == 0.5      # 19.8 lines
    assert ct.auto_interval(20) == 1.0
    assert ct.auto_interval(24) == 1.0
    assert ct.auto_interval(40) == 2.0
    assert ct.auto_interval(100) == 5.0
    assert ct.auto_interval(240) == 10.0     # 24 lines at 10 m
    assert ct.auto_interval(9000) == 10.0    # capped at the largest
    for relief in (3, 7, 12, 30, 80, 150):   # 15-25 lines on typical sites (within the 2x step grid: >= 10)
        iv = ct.auto_interval(relief)
        assert 10 <= relief / iv <= 25 or iv in (0.25, 10.0)


def test_resolve_interval_user_scale_and_cap():
    s = grid_site(CONE)
    a = ct.resolve_interval(s)
    assert a["source"] == "auto" and a["interval"] == a["auto_interval"] == 1.0
    assert ct.resolve_interval(s, 2)["interval"] == 2.0 and ct.resolve_interval(s, 2)["source"] == "user"
    assert ct.resolve_interval(s, None, 2.0)["interval"] == 2.0          # super-light: auto doubled
    big = grid_site(lambda x, y: 4000 * (x / 50.0))
    r = ct.resolve_interval(big, 0.25)
    assert r["interval"] > 0.25 and "raised" in r["note"]


# ---------------------------------------------------------------- geometry
def test_plane_contours_straight_at_correct_z():
    s = grid_site(PLANE)
    r = ct.build_contours(s)
    assert r["interval"] == 0.5      # relief 10 m -> 20 minor lines
    assert r["counts"]["minor"] + r["counts"]["major"] > 15
    for m in all_lines(r):
        # abs z = 100 + 0.1 x + 3  ->  x = (z_abs - 103) * 10
        want_x = (m["z_abs"] - 103.0) * 10.0
        assert all(abs(p[0] - want_x) < 1e-3 for p in m["points"]), m["z_abs"]
        assert (m["z_abs"] / 0.5) % 1 == pytest.approx(0, abs=1e-6)
    # a straight line collapses to 2 points after Douglas-Peucker; clipped to the boundary
    assert all(len(m["points"]) == 2 for m in all_lines(r))
    assert all(abs(abs(m["points"][0][1]) - 50) < 1e-6 for m in all_lines(r))


def test_cone_contours_are_circles_with_correct_z():
    s = grid_site(CONE, s=2.0)
    r = ct.build_contours(s)
    assert r["interval"] == 1.0
    closed = [m for m in all_lines(r) if m["closed"]]
    assert closed, "cone must give closed loops"
    for m in closed:
        want_r = (100.0 + 10.0 - m["z_abs"]) / 0.2           # z_abs = 110 - 0.2 r
        radii = [math.hypot(*p) for p in m["points"]]
        assert max(abs(rr - want_r) for rr in radii) < 0.35, (m["z_abs"], want_r)
        assert m["points"][0] == m["points"][-1]
    # z on the same TIN the mesh uses equals the level at every contour vertex
    for m in all_lines(r)[:6]:
        for p in m["points"][::3]:
            z = ct.tin_z(s, p[0], p[1])
            assert z is not None and abs(100.0 + z - m["z_abs"]) < 0.02


def _segments_cross(a, b):
    la, lb = LineString(a["points"]), LineString(b["points"])
    inter = la.intersection(lb)
    return not inter.is_empty


def test_saddle_no_crossings_between_levels_and_lines_are_simple():
    s = grid_site(SADDLE, s=2.0, e0=100.0)
    r = ct.build_contours(s, 0.5, simplify=0.0)
    lines = all_lines(r)
    assert len(lines) > 10
    for m in lines:
        ls = LineString(m["points"])
        assert ls.is_simple or m["closed"], m["z_abs"]
    byz = {}
    for m in lines:
        byz.setdefault(m["z_abs"], []).append(m)
    zs = sorted(byz)
    for i, z1 in enumerate(zs):
        for z2 in zs[i + 1:i + 3]:
            for a in byz[z1]:
                for b in byz[z2]:
                    assert not _segments_cross(a, b), (z1, z2)
    # saddle: the lines at a level above and below the pass are hyperbola branches on both sides (>= 2 pieces each)
    above = [m for m in lines if m["z_abs"] > 100.3 + 0.4]
    below = [m for m in lines if m["z_abs"] < 100.3 - 0.4]
    assert len({round(m["points"][0][0] > 0) for m in above}) == 2 or len(above) >= 2
    assert len(below) >= 2


def test_clipped_to_boundary_and_tiny_loops_dropped():
    s = grid_site(CONE)
    r = ct.build_contours(s)
    from shapely.geometry import Polygon
    poly = Polygon([(p[0], p[1]) for p in s["terrain"]["boundary"]]).buffer(1e-6)
    for m in all_lines(r):
        assert poly.covers(LineString(m["points"]))
        if m["closed"]:
            assert LineString(m["points"]).length >= ct.MIN_LOOP_M
    # peak z abs 110: a level just under the peak would be a loop of radius < 0.3 m -> dropped
    s2 = grid_site(lambda x, y: 10.0 - 0.2 * math.hypot(x, y) + (0.0 if math.hypot(x, y) > 1 else 0.0))
    r2 = ct.build_contours(s2, 0.25)
    assert all(LineString(m["points"]).length >= ct.MIN_LOOP_M for m in all_lines(r2) if m["closed"])
    # clipping: a site whose terrain boundary is smaller than the TIN hull keeps lines inside the boundary
    s3 = grid_site(CONE)
    s3["terrain"]["boundary"] = [[-20, -20], [20, -20], [20, 20], [-20, 20]]
    r3 = ct.build_contours(s3)
    assert all(abs(p[0]) <= 20 + 1e-6 and abs(p[1]) <= 20 + 1e-6 for m in all_lines(r3) for p in m["points"])


def test_major_every_fifth_and_levels_absolute():
    s = grid_site(PLANE)
    r = ct.build_contours(s, 1.0)
    assert {m["z_abs"] for m in r["major"]} == {100.0, 105.0}            # multiples of 5 x 1.0 inside 98..108
    assert all((m["z_abs"] % 5) != 0 for m in r["minor"])
    r2 = ct.build_contours(s, 0.5)
    assert {m["z_abs"] for m in r2["major"]} == {100.0, 102.5, 105.0, 107.5}
    # shifting origin.elevation shifts the relative z but not the absolute levels
    s["origin"]["elevation"] = 100.25
    r3 = ct.build_contours(s, 1.0)
    assert {m["z_abs"] for m in r3["major"]} == {100.0, 105.0}
    assert all(abs(p[0] - (m["z_abs"] - 100.25 - 3.0) * 10) < 1e-3 for m in all_lines(r3) for p in m["points"])


def test_flat_site_no_lines_and_few_points_safe():
    flat = grid_site(lambda x, y: 5.0)
    r = ct.build_contours(flat)
    assert r["counts"]["minor"] + r["counts"]["major"] == 0 and r["interval"] == 0.25
    tiny = {"origin": {"elevation": 0}, "terrain": {"boundary": [[0, 0], [1, 0]], "points": [[0, 0, 1], [1, 0, 2]]}}
    assert ct.build_contours(tiny)["warnings"]


# ---------------------------------------------------------------- labels
def test_labels_on_majors_spaced_and_rotated():
    s = grid_site(PLANE, n=100.0, s=4.0)
    r = ct.build_contours(s, 1.0)
    assert r["labels"]
    texts = {lb["text"] for lb in r["labels"]}
    assert texts <= {f"{z:.1f}" for z in {m["z_abs"] for m in r["major"]}}
    pts = [(lb["x"], lb["y"]) for lb in r["labels"]]
    for i, a in enumerate(pts):
        for b in pts[i + 1:]:
            assert math.hypot(a[0] - b[0], a[1] - b[1]) >= ct.LABEL_SPACING_M - 1e-6
    for lb in r["labels"]:        # vertical lines: angle +-90 deg, kept readable in (-90, 90]
        assert -math.pi / 2 < lb["angle"] <= math.pi / 2 + 1e-3
        assert abs(abs(lb["angle"]) - math.pi / 2) < 0.05
    assert max(math.hypot(*p) for p in pts) <= 100 * 2 ** 0.5


def test_label_skips_short_lines_and_formats_decimals():
    short = [{"z_abs": 105.0, "points": [(0, 0), (10, 0)], "closed": False}]
    assert ct.place_labels(short, 1.0) == []                                  # 10 m < 15 m
    longl = [{"z_abs": 105.0, "points": [(0, 0), (60, 0)], "closed": False}]
    lb = ct.place_labels(longl, 1.0)
    assert lb and lb[0]["x"] == pytest.approx(30.0) and lb[0]["text"] == "105.0"   # midpoint
    assert ct.place_labels([{"z_abs": 101.25, "points": [(0, 0), (60, 0)], "closed": False}], 0.25)[0]["text"] == "101.25"


# ---------------------------------------------------------------- mesh points
def test_terrain_mesh_single_point_sublines():
    s = mini_site()
    m = bp.terrain_mesh(s)                       # default = points
    assert len(m["sublines"]) == 9 and all(len(x["coordinates"]) == 1 for x in m["sublines"])
    assert m["ridges"] == "UserDefined" and m["showLines"] is False
    # heights kept, z relative to the level (outline min)
    assert m["level"] == pytest.approx(0.0)
    zs = sorted(x["coordinates"][0]["z"] + m["level"] for x in m["sublines"])
    assert zs[0] == pytest.approx(4.5)  # relative to level 0 and zs[-1] == pytest.approx(5.5)
    rows = bp.terrain_mesh(s, terrain_points="rows")
    assert len(rows["sublines"]) == 3
    assert bp.terrain_mesh(s, dz=3.0)["level"] == pytest.approx(m["level"] + 3.0)
    with pytest.raises(ValueError):
        bp.terrain_mesh(s, terrain_points="nope")


def test_land_cover_single_points():
    from tests.test_m4c import site_with_lc
    ms = bp.land_cover_meshes(site_with_lc())
    assert ms and all(len(x["coordinates"]) == 1 for m in ms for x in m["item"]["sublines"])
    bb = bp.land_cover_expected_bbox(ms[0])
    assert bb["zMax"] >= bb["zMin"]


# ---------------------------------------------------------------- build helpers
PENS = [{"index": 1, "weight": 0.13, "color": {"red": 0, "green": 0, "blue": 0}, "description": "k"},
        {"index": 7, "weight": 0.35, "color": {"red": 0, "green": 0, "blue": 0}, "description": "k2"},
        {"index": 12, "weight": 0.18, "color": {"red": 0.5, "green": 0.25, "blue": 0.25}, "description": "roof"},
        {"index": 32, "weight": 0.35, "color": {"red": 0.5, "green": 0.25, "blue": 0.25}, "description": "roof cut"},
        {"index": 19, "weight": 0.0, "color": {"red": 1, "green": 1, "blue": 1}, "description": "bg"}]


def test_pick_contour_pens_same_colour_family():
    p = build_ac.pick_contour_pens(PENS)
    assert (p["minor"], p["major"], p["label"], p["fill"]) == (12, 32, 32, 19)
    q = build_ac.pick_contour_pens([PENS[0], PENS[1]])      # only black: still one family, thin + thick
    assert (q["minor"], q["major"]) == (1, 7)
    assert build_ac.pick_contour_pens([PENS[0]])["note"].startswith("no thin+thick")


def test_contour_items_payloads():
    s = grid_site(PLANE, n=100.0, s=4.0)
    con = ct.build_contours(s, 1.0)
    pens = build_ac.pick_contour_pens(PENS)
    it = build_ac.contour_items(con, 7, pens)
    assert len(it["minor"]) == len(con["minor"]) and len(it["major"]) == len(con["major"]) and len(it["label"]) == len(con["labels"])
    a = it["minor"][0]
    assert a["layerIndex"] == 7 and a["floorInd"] == 0 and a["linePenIndex"] == 12 and len(a["coordinates"]) >= 2
    assert it["major"][0]["linePenIndex"] == 32
    t = it["label"][0]
    assert t["pen"] == 32 and t["style"]["fixedSize"] is False and t["style"]["anchor"] == "MiddleMiddle" and t["style"]["fillPenIndex"] == 19
    assert t["justification"] == "Center" and t["floorIndex"] == 0 and "angle" in t


def test_group_plan_contours_group():
    created = {"terrain": [("tm", "terrain")], "buildings": [("b1", "w/1"), ("b2", "w/2")],
               "contours_minor": [("m1", "minor"), ("m2", "minor")], "contours_major": [("j1", "major"), ("j2", "major")],
               "contour_labels": [("l1", "label"), ("l2", "label")]}
    node = bp.group_plan(created)
    con = next(k for k in node["children"] if isinstance(k, dict) and k["name"] == "Contours")
    assert [k["name"] for k in con["children"]] == ["Contours minor", "Contours major", "Contour labels"]
    assert node["name"] == "OSM Site Context"
    # a single major line dissolves into the Contours group (>= 2 items rule)
    created["contours_major"] = [("j1", "major")]
    created["contour_labels"] = []
    con = next(k for k in bp.group_plan(created)["children"] if isinstance(k, dict) and k["name"] == "Contours")
    assert "j1" in con["children"] and len(con["children"]) == 2
    # no contours -> no group
    for k in ("contours_minor", "contours_major", "contour_labels"):
        created.pop(k)
    assert all(not (isinstance(k, dict) and k["name"] == "Contours") for k in bp.group_plan(created)["children"])


class ContourAC:
    """Fake client for delete_old_contours / build_contour_elements."""

    def __init__(self, old_polys=3, old_texts=2, foreign=2):
        self.polys = {f"P{i}": True for i in range(old_polys)} | {f"F{i}": False for i in range(foreign)}   # guid -> ours
        self.texts = {f"T{i}": True for i in range(old_texts)} | {"TX": False}
        self.deleted, self.calls, self.window = [], [], None
        self.ud = {}

    def call(self, cmd, p=None):
        if cmd == "API.GetActivePenTables":
            return {"modelViewPenTableId": {"attributeId": {"guid": "pt"}}}
        if cmd == "API.GetPenTableAttributes":
            return {"attributes": [{"penTableAttribute": {"name": "T", "pens": [{"pen": x} for x in PENS]}}]}
        raise AssertionError(cmd)

    def tapir(self, cmd, p=None):
        p = p or {}
        self.calls.append(cmd)
        if cmd == "ChangeWindow":
            self.window = p
            return {"success": True}
        if cmd == "GetElementsByType":
            src = {"PolyLine": self.polys, "Text": self.texts}[p["elementType"]]
            return {"elements": [{"elementId": {"guid": g}} for g in src]}
        if cmd == "GetElementUserData":
            allv = {**self.polys, **self.texts}
            return {"elementUserData": [{"userData": [{"key": "osmctx:contour", "value": "minor"}]}
                                        if allv.get(e["elementId"]["guid"]) or self.ud.get(e["elementId"]["guid"]) else {"userData": []}
                                        for e in p["elements"]]}
        if cmd == "DeleteElements":
            self.deleted += [e["elementId"]["guid"] for e in p["elements"]]
            return {"executionResults": [{"success": True} for _ in p["elements"]]}
        if cmd in ("CreatePolylines", "CreateTexts"):
            key = "polylinesData" if cmd == "CreatePolylines" else "textsData"
            out = []
            for i, _ in enumerate(p[key]):
                g = f"N{len(self.ud)}"
                self.ud[g] = True
                out.append({"elementId": {"guid": g}})
            return {"elements": out}
        if cmd == "SetElementUserData":
            return {"executionResults": [{"success": True} for _ in p["elementUserData"]]}
        raise AssertionError(cmd)


def test_replace_removes_only_our_contours():
    ac = ContourAC()
    n = build_ac.delete_old_contours(ac, lambda *_: None)
    assert n == {"PolyLine": 3, "Text": 2}
    assert sorted(ac.deleted) == ["P0", "P1", "P2", "T0", "T1"]       # foreign polylines/text untouched
    assert ac.window == {"windowType": "FloorPlan", "storyIndex": 0}    # active window must be the plan


def test_build_contour_elements_creates_tags_and_reports():
    ac = ContourAC()
    site = grid_site(CONE, n=60.0, s=3.0)
    rep = {"layer_names": {"contours": "Site & Landscape - General"}}
    created = build_ac.build_contour_elements(ac, lambda *_: None, site, rep, {"contours": 9}, None, 1.0, None)
    r = rep["contours"]
    assert r["interval"] == 1.0 and r["pens"]["minor"] == 12 and r["pens"]["major"] == 32
    assert len(created["contours_minor"]) == r["counts_planned"]["minor"] and r["counts_planned"]["vertices"] > 0
    assert len(created["contours_major"]) == r["counts_planned"]["major"]
    assert len(created["contour_labels"]) == r["counts_planned"]["labels"] > 0
    assert ac.window == {"windowType": "FloorPlan", "storyIndex": 0}
    assert all(v == {"errors": 0, "first": None, "tag_failures": 0} for v in r["errors"].values())
    # explicit pens override
    rep2 = {}
    build_ac.build_contour_elements(ContourAC(), lambda *_: None, site, rep2, {"contours": 9}, 2.0, 1.0, {"minor": 3, "major": 4, "label": 5})
    assert rep2["contours"]["interval"] == 2.0 and rep2["contours"]["pens"]["minor"] == 3 and rep2["contours"]["pens"]["label"] == 5


def test_per_type_creates_contour_layer_single_does_not():
    from tests.test_layers_any_project import FakeAC
    ac = FakeAC(["Archicad Layer", "Site & Landscape - Terrain"])
    build_ac.ensure_layers(ac, lambda *_: None, {"gaps": []}, contours=True)
    assert "Site-Contours" in ac.created_layers
    ac2 = FakeAC(["Archicad Layer", "Site & Landscape - Terrain"])
    build_ac.ensure_layers(ac2, lambda *_: None, {"gaps": []}, contours=False)
    assert "Site-Contours" not in ac2.created_layers


def test_cli_flags(monkeypatch):
    seen = {}
    monkeypatch.setattr(build_ac, "build", lambda *a, **k: seen.update(k) or {"created": {}, "expected": {}})
    build_ac.main(["x.json"])
    assert seen["contours"] is True and seen["contour_interval"] is None and seen["terrain_points"] == "points"
    build_ac.main(["x.json", "--no-contours", "--contour-interval", "2", "--terrain-points", "rows"])
    assert seen["contours"] is False and seen["contour_interval"] == 2.0 and seen["terrain_points"] == "rows"


# ---------------------------------------------------------------- server / UI wiring
def test_import_passes_contour_settings(monkeypatch, tmp_path):
    import json
    seen = {}
    monkeypatch.setattr(build_ac, "build", lambda *a, **k: seen.update(k) or {"created": {}})
    monkeypatch.setattr(server, "OUT_DIR", tmp_path)
    (tmp_path / "site.json").write_text(json.dumps({"boundary": [[0, 0], [100, 0], [100, 100], [0, 100]]}), encoding="utf-8")
    cl = TestClient(server.app)
    r = cl.post("/api/import", json={"settings": {"contours": False, "contour_interval": 2}})
    assert r.status_code == 200, r.text
    assert seen["contours"] is False and seen["contour_interval"] == 2 and seen["contour_scale"] == 1.0
    r = cl.post("/api/import", json={"settings": {}})
    assert seen["contours"] is True and seen["contour_interval"] is None
    r = cl.post("/api/import", json={"settings": {}, "terrain_points": "bad"})
    assert r.status_code == 400


def test_super_light_doubles_interval():
    from osmctx import limits
    g = limits.resolve(10.0, {"contour_interval": 1}, False)
    assert g["preset"] and g["settings"]["contour_interval_scale"] == 2.0 and g["settings"]["contour_interval"] == 1


def test_ui_has_contour_controls():
    from pathlib import Path
    html = (Path(server.ROOT) / "ui" / "index.html").read_text(encoding="utf-8")
    js = (Path(server.ROOT) / "ui" / "app.js").read_text(encoding="utf-8")
    for needle in ('id="s_contours"', 'id="s_cint"', 'id="cintAuto"'):
        assert needle in html
    for opt in ('value="">Auto', 'value="0.25"', 'value="10"'):
        assert opt in html
    assert "contour_interval" in js and "s_contours" in js
