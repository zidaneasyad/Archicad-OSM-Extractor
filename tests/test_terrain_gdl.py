"""GDL terrain library part (osmctx/terrain_gdl.py): payload shape, stable name, script data, closed solid (replayed
from the script data), CreateObjects item, and an offline compile + interpret check with Graphisoft's LP_XMLConverter
(skipped when absent). Compiling/interpreting proves syntax and that the scripts run with default parameters; it
does not build or show the 3D model (an EDGE to a missing vertex passes), hence the geometry replay test."""
import math
import re
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from osmctx import build_plan as bp
from osmctx import contours as ct
from osmctx import terrain_gdl as tg

CONVERTERS = [Path(r"C:\Program Files\Graphisoft\Archicad 30\LP_XMLConverter.exe"),
              Path(r"C:\Program Files\Graphisoft\Archicad 29\LP_XMLConverter.exe")]
CONVERTER = next((p for p in CONVERTERS if p.exists()), None)
TAPIR_TYPES = {"Integer", "Length", "Angle", "RealNumber", "LightSwitch", "RGBColor", "Intensity", "LineType",
               "Material", "FillPattern", "PenColor", "String", "Boolean", "Separator", "Title", "BuildingMaterial",
               "Profile"}


def grid_site(f, n=50.0, s=2.0, e0=100.0, inside=None, lat=48.5, lon=9.05):
    """Square site [-n, n]^2 (densified outline ring + interior grid, like the converter); `inside` = optional
    (ring, predicate) for a non-square outline."""
    m = int(round(2 * n / s))
    ring = ([[-n + i * s, -n] for i in range(m)] + [[n, -n + i * s] for i in range(m)] +
            [[n - i * s, n] for i in range(m)] + [[-n, n - i * s] for i in range(m)])
    xs = [-n + s / 2 + i * s for i in range(m)]
    inner = [[x, y] for x in xs for y in xs if abs(x) < n - 0.3 * s and abs(y) < n - 0.3 * s]
    if inside:
        ring, keep = inside
        inner = [p for p in inner if keep(*p)]
    pts = [[x, y, f(x, y)] for x, y in ring + inner]
    return {"origin": {"lat": lat, "lon": lon, "elevation": e0}, "boundary": ring,
            "terrain": {"boundary": ring, "points": pts}}


def l_site():
    """L-shaped outline (6 corners, one concave) with a sloped hill."""
    corners = [(-40, -40), (40, -40), (40, 0), (0, 0), (0, 40), (-40, 40)]
    ring = []
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
        k = int(max(abs(x1 - x0), abs(y1 - y0)) / 4)
        ring += [[x0 + (x1 - x0) * i / k, y0 + (y1 - y0) * i / k] for i in range(k)]
    keep = lambda x, y: -39 < x < 39 and -39 < y < 39 and not (x > -1 and y > -1)  # noqa: E731
    return grid_site(lambda x, y: 5 + 0.08 * x + 6 * math.exp(-((x + 20) ** 2 + (y + 20) ** 2) / 300), n=40, s=2,
                     inside=(ring, keep))


CONE = lambda x, y: 10.0 - 0.2 * math.hypot(x, y)  # noqa: E731


@pytest.fixture(scope="module")
def cone():
    site = grid_site(CONE)
    part, info = tg.terrain_gdl_part(site)
    return site, part, info


# ---------------------------------------------------------------- payload
def test_part_payload_shape(cone):
    site, part, info = cone
    assert set(part) <= {"name", "parentName", "parentId", "type", "path", "a", "b", "masterScript", "script2D",
                         "script3D", "parameterScript", "uiScript", "propertiesScript", "parameters"}
    assert part["parentName"] == "Model Element"           # Tapir refuses the General GDL Object root as parent
    assert part["type"] == "Object"
    assert part["path"] == f"OSM Context/{part['name']}.gsm"
    assert part["a"] == pytest.approx(100.0) and part["b"] == pytest.approx(100.0)
    names = [p["name"] for p in part["parameters"]]
    assert len(names) == len(set(names))
    for p in part["parameters"]:
        assert set(p) <= {"name", "type", "value", "description", "isHidden"}
        assert p["type"] in TAPIR_TYPES
        assert 1 <= len(p["name"]) <= 31 and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p["name"])
        assert p["name"].upper() not in {"A", "B", "ZZYZX"}
        if p["type"] == "Boolean":
            assert isinstance(p["value"], bool)
        if p["type"] in ("Title", "Separator"):
            assert "value" not in p
        if p["type"] in ("PenColor", "LineType", "Material", "BuildingMaterial", "Integer"):
            assert isinstance(p["value"], int)
    by = {p["name"]: p for p in part["parameters"]}
    for k in ("showContours3D", "showContours2D", "showLabels", "penMinor", "penMajor", "terrainMat"):
        assert k in by
    assert by["terrainMat"]["type"] == "Material" and by["skirtDepth"]["value"] == tg.TERRAIN_SKIRT
    assert all(by[k].get("isHidden") for k in ("infoInterval", "infoSource", "infoCounts"))
    assert "1 m" in by["infoInterval"]["value"]
    wrap = tg.create_library_parts_params([part])
    assert wrap == {"libraryParts": [part], "overwriteExisting": True}


def test_stable_name():
    a = grid_site(CONE)
    n1 = tg.part_name(a)
    assert n1 == tg.part_name(grid_site(CONE)) and re.fullmatch(r"OSM Terrain [0-9a-f]{8}", n1)
    assert tg.terrain_gdl_part(a, interval=2.0)[0]["name"] == n1      # other interval -> same part (overwritten)
    assert tg.part_name(grid_site(CONE, lat=48.6)) != n1
    assert tg.part_name(grid_site(CONE, e0=200.0)) != n1
    assert tg.part_name(grid_site(CONE, n=60.0)) != n1
    assert tg.terrain_gdl_part(a, name="X")[0]["name"] == "X"


def test_script_lines_short_and_counts_in_script(cone):
    site, part, info = cone
    for key in ("script2D", "script3D", "parameterScript"):
        assert max(len(ln) for ln in part[key].splitlines()) <= 256, key
    c = info["counts"]
    P, T = ct._tin(site)
    assert c["triangles"] == len(T) and c["vertices"] == len(set(T.ravel().tolist()))
    assert c["edges"] == c["vertices"] + c["triangles"] - 1        # Euler, a disc: V - E + F = 1
    s3 = part["script3D"]
    assert f"nV = {c['vertices']} : nE = {c['edges']} : nT = {c['triangles']}" in s3
    data = _main_data(s3)
    assert len(data) == 3 * c["vertices"] + 2 * c["edges"] + 3 * c["triangles"]
    assert info["script_bytes"]["script3D"] == len(s3.encode("utf-8"))
    assert c["contour_minor"] > 0 and c["contour_major"] > 0 and c["contour_labels"] > 0
    assert s3.count("BODY -1") == 3                                  # solid + minor + major contour wire bodies
    assert part["script2D"].count("TEXT2 ") == c["contour_labels"]


# ---------------------------------------------------------------- geometry replay
def _main_data(s3: str) -> list[float]:
    vals = []
    for ln in s3.splitlines():
        if ln.startswith("oE = "):
            break
        if ln.startswith("PUT "):
            vals += [float(v) for v in ln[4:].split(":")[0].split(",")]
    return vals


def _replay_solid(part, skirt=tg.TERRAIN_SKIRT, edge_mode=1):
    """Rebuild the body exactly as the 3D script does (same loops, same index arithmetic)."""
    s3 = part["script3D"]
    g = dict(re.findall(r"\b(nV|nE|nT|nVb|nVc|nEb) = (\d+)", s3[:s3.index("zBot = ")]))
    nV, nE, nT, nVb, nVc, nEb = (int(g[k]) for k in ("nV", "nE", "nT", "nVb", "nVc", "nEb"))
    zmin = float(re.search(r"zBot = (-?[\d.]+) - skirtDepth", s3).group(1))
    zbot = zmin - skirt
    dv = [None] + _main_data(s3)                                       # 1-based like GDL
    oE, oT = 3 * nV, 3 * nV + 2 * nE
    V = [None] + [(dv[3 * i - 2], dv[3 * i - 1], dv[3 * i]) for i in range(1, nV + 1)]
    V += [(dv[3 * i - 2], dv[3 * i - 1], zbot) for i in range(1, nV + 1)]
    E = [None]
    est = {0: 0, 1: 1, 2: 2}[edge_mode]
    for i in range(1, nE + 1):
        E.append((int(dv[oE + 2 * i - 1]), int(dv[oE + 2 * i]), 0 if i <= nEb else est))
    for i in range(1, nE + 1):
        E.append((int(dv[oE + 2 * i - 1]) + nV, int(dv[oE + 2 * i]) + nV, 0 if i <= nEb else 1))
    for i in range(1, nVb + 1):
        E.append((i + nV, i, 0 if i <= nVc else 1))
    sgn = lambda t: (t > 0) - (t < 0)  # noqa: E731
    PG = []
    for i in range(1, nT + 1):
        PG.append([int(dv[oT + 3 * i - 2]), int(dv[oT + 3 * i - 1]), int(dv[oT + 3 * i])])
    for i in range(1, nT + 1):
        t1, t2, t3 = (int(dv[oT + 3 * i - 2]), int(dv[oT + 3 * i - 1]), int(dv[oT + 3 * i]))
        PG.append([-(t3 + sgn(t3) * nE), -(t2 + sgn(t2) * nE), -(t1 + sgn(t1) * nE)])
    for i in range(1, nEb + 1):
        PG.append([i + nE, 2 * nE + int(dv[oE + 2 * i]), -i, -(2 * nE + int(dv[oE + 2 * i - 1]))])
    return V, E, PG, zbot, (nV, nE, nT, nVb, nVc, nEb)


def _check_closed(V, E, PG):
    use = Counter()
    vol = 0.0
    for pg in PG:
        chain = []
        for e in pg:
            assert e != 0 and 1 <= abs(e) < len(E)
            a, b, _ = E[abs(e)]
            chain.append((a, b) if e > 0 else (b, a))
            use[e] += 1
        for (a0, b0), (a1, b1) in zip(chain, chain[1:] + chain[:1]):
            assert b0 == a1, "polygon edges not a closed chain"
        p0 = V[chain[0][0]]
        for a, b in chain[1:-1]:                                       # fan -> signed volume (divergence theorem)
            p1, p2 = V[a], V[b]
            vol += (p0[0] * (p1[1] * p2[2] - p1[2] * p2[1]) - p0[1] * (p1[0] * p2[2] - p1[2] * p2[0])
                    + p0[2] * (p1[0] * p2[1] - p1[1] * p2[0])) / 6.0
    for i in range(1, len(E)):                                         # every edge: once each way
        assert use[i] == 1 and use[-i] == 1, f"edge {i}: +{use[i]} -{use[-i]}"
    return vol


@pytest.mark.parametrize("make", [lambda: grid_site(CONE), l_site,
                                  lambda: grid_site(lambda x, y: 0.3 + (x * x - y * y) / 60.0, n=30, s=3)])
def test_solid_closed_oriented_and_volume(make):
    site = make()
    part, info = tg.terrain_gdl_part(site)
    V, E, PG, zbot, (nV, nE, nT, nVb, nVc, nEb) = _replay_solid(part)
    assert len(PG) == info["counts"]["body_polygons"] and len(E) - 1 == info["counts"]["body_edges"]
    vol = _check_closed(V, E, PG)
    P, T = ct._tin(site)                                               # independent: prisms under the TIN
    want = 0.0
    for a, b, c in T.tolist():
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = P[a], P[b], P[c]
        area = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / 2
        want += area * ((z0 + z1 + z2) / 3 - zbot)
    assert vol == pytest.approx(want, rel=1e-4)                        # > 0: normals point outwards
    for pg in PG[:nT]:                                                 # top triangles CCW from above
        pts = [V[E[abs(e)][0] if e > 0 else E[abs(e)][1]] for e in pg]
        cz = (pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1]) - (pts[1][1] - pts[0][1]) * (pts[2][0] - pts[0][0])
        assert cz > -1e-9
    # visibility: TIN interior edges hidden, outline visible, verticals visible only at corners
    assert all(E[i][2] == 0 for i in range(1, nEb + 1)) and all(E[i][2] == 1 for i in range(nEb + 1, nE + 1))
    assert sum(1 for i in range(2 * nE + 1, 2 * nE + nVb + 1) if E[i][2] == 0) == nVc


def test_corners_of_l_site():
    tin = tg.tin_solid_data(l_site())
    assert tin["n_vc"] == 6 and len(tin["loops"]) == 1
    xy = tin["xyz"][: tin["n_vc"], :2].round(3).tolist()
    assert sorted(map(tuple, xy)) == sorted([(-40, -40), (40, -40), (40, 0), (0, 0), (0, 40), (-40, 40)])


def test_contour_data_round_trip(cone):
    site, part, info = cone
    con = ct.build_contours(site)
    s3 = part["script3D"].split("if showContours3D then", 1)[1].split("endif", 1)[0]
    vals = [float(v) for ln in s3.splitlines() if ln.strip().startswith("PUT ")
            for v in ln.strip()[4:].split(":")[0].split(",")]
    k, got = 0, []
    while k < len(vals):
        n, z = int(vals[k]), vals[k + 1]
        got.append((n, z))
        k += 2 + 2 * n
    want = [(len(m["points"]), round(m["z_abs"] - 100.0, 3)) for m in con["minor"] + con["major"]]
    assert got == [(n, pytest.approx(z, abs=1e-3)) for n, z in want]
    assert f"for cL = 1 to {len(con['minor'])}" in s3 and f"for cL = 1 to {len(con['major'])}" in s3


# ---------------------------------------------------------------- CreateObjects item
def test_object_item_matches_object_format():
    site = grid_site(CONE)
    name = tg.part_name(site)
    it = tg.terrain_gdl_object(site, name, dz=1.5, layer_index=7, params={"penMinor": 3, "penMajor": 5})
    tree = bp.tree_object({"x": 1, "y": 2, "z": 3, "height": 8}, "Tree", 0.0)
    assert set(tree) <= set(it)
    assert set(it) == set(tree) | {"parameters", "layerIndex"}
    assert it["libraryPartName"] == name and it["floorIndex"] == 0 and it["angle"] == 0.0
    assert it["coordinates"] == {"x": 0.0, "y": 0.0, "z": 1.5}
    assert set(it["dimensions"]) == {"x", "y", "z"} and it["dimensions"]["x"] == pytest.approx(100.0)
    part, _ = tg.terrain_gdl_part(site)
    assert (it["dimensions"]["x"], it["dimensions"]["y"]) == (part["a"], part["b"])   # ratio 1 to the part's A/B
    assert it["parameters"] == [{"name": "penMinor", "value": 3}, {"name": "penMajor", "value": 5}]
    assert it["layerIndex"] == 7
    plain = tg.terrain_gdl_object(site, name)
    assert "layerIndex" not in plain and "parameters" not in plain


# ---------------------------------------------------------------- offline compile + interpret
def _stub(guid, name, ancestry=()):
    part = {"a": 1, "b": 1, "parameters": []}
    return tg.libpart_xml(part, guid, ancestry=ancestry, placeable=False), name


def _compile(tmp_path: Path, parts: dict[str, str]):
    src, dst = tmp_path / "src", tmp_path / "lib"
    src.mkdir()
    stubs = [_stub(tg.GUID_GENERAL_GDL_OBJECT, "General GDL Object"),
             _stub(tg.GUID_MODEL_ELEMENT, "Model Element", (tg.GUID_GENERAL_GDL_OBJECT,))]
    for xml, name in stubs:
        (src / f"{name}.xml").write_text(xml, encoding="utf-8")
    for name, xml in parts.items():
        (src / f"{name}.xml").write_text(xml, encoding="utf-8")
    r = subprocess.run([str(CONVERTER), "convertlibrary", "-l", "UTF8", "-reportlevel", "2", "-interpret",
                        "-checkinit", str(src), str(dst)], capture_output=True, text=True, timeout=600)
    out = (r.stdout or "") + (r.stderr or "")
    problems = [ln for ln in out.splitlines() if re.search(r"\b(error|warning)\b", ln, re.I)]
    return r.returncode, problems, out, dst


def _with_defaults(part, **over):
    p = dict(part)
    p["parameters"] = [dict(q, value=over[q["name"]]) if q["name"] in over else q for q in part["parameters"]]
    return p


@pytest.mark.skipif(CONVERTER is None, reason="Graphisoft LP_XMLConverter not installed")
def test_compiles_and_interprets_clean(tmp_path):
    cone_part, _ = tg.terrain_gdl_part(grid_site(CONE))
    l_part, _ = tg.terrain_gdl_part(l_site())
    flipped = _with_defaults(cone_part, use2DProjection=True, edgeMode=2, showLabels=False, showOutline2D=False)
    off = _with_defaults(l_part, showContours3D=False, showContours2D=False, edgeMode=0)
    parts = {"cone": tg.libpart_xml(cone_part, "7E3B1C2A-1111-4A2B-9C3D-0000000000A1"),
             "lshape": tg.libpart_xml(l_part, "7E3B1C2A-1111-4A2B-9C3D-0000000000A2"),
             "cone_flipped": tg.libpart_xml(flipped, "7E3B1C2A-1111-4A2B-9C3D-0000000000A3"),
             "l_off": tg.libpart_xml(off, "7E3B1C2A-1111-4A2B-9C3D-0000000000A4")}
    rc, problems, out, dst = _compile(tmp_path, parts)
    assert rc == 0 and not problems, out
    for name in parts:
        assert (dst / f"{name}.gsm").exists()
    assert out.count("Interpret ...") >= len(parts)


@pytest.mark.skipif(CONVERTER is None, reason="Graphisoft LP_XMLConverter not installed")
def test_compile_check_catches_errors(tmp_path):
    """The check is real: a runtime buffer error in the 3D script fails it."""
    part, _ = tg.terrain_gdl_part(grid_site(CONE, n=10, s=2))
    bad = dict(part, script3D=part["script3D"].replace("nD = 0", "nD = 0\nqq = GET(1)", 1))
    rc, problems, out, _ = _compile(tmp_path, {"bad": tg.libpart_xml(bad, "7E3B1C2A-1111-4A2B-9C3D-0000000000A5")})
    assert rc != 0 and any("buffer" in p for p in problems), out
