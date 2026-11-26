"""Terrain as ONE generated GDL object ("gdl" variant of the third terrain type, M11-B). Pure payloads, no Archicad.

What it makes
  terrain_gdl_part()   -> one Tapir `CreateLibraryParts` item: a GDL object whose
     3D script = closed solid terrain (TIN top + vertical skirt walls + flat bottom, like the Mesh tool's
                 "SolidBodyWithSkirt") + optional 3D contour lines (wire bodies, lifted `contourLift` above the TIN),
     2D script = contour polylines (minor/major pens + line types), elevation labels, TIN outline, optional PROJECT2.
  terrain_gdl_object() -> the `CreateObjects` item that places it (same item format as tree/shrub/hedge objects).

Surface = the SAME TIN the contours are cut from (`contours._tin`: Delaunay through terrain.points, triangles kept
inside the site boundary), so contour lines lie exactly on the facets (3D lines are lifted by `contourLift`).

Coordinates: the object is placed at project (0, 0) with angle 0, so GDL local x/y = site x/y (= project metres).
Local z = site z (relative to origin.elevation, the convention of build_plan) and the object's placement
z = dz, i.e. absolute z of a terrain point = dz + z_rel, the same as terrain_mesh / trees (`coordinates.z` is
the absolute elevation; Tapir turns it into floor + offset). UNVERIFIED LIVE (M11 is offline-only).

GDL approach (why, see the M11 notes in MILESTONES.md):
  * Only simple (non-array) GDL parameters can be created through Tapir, so all geometry is baked into the script
    text. To keep it compact every number is written ONCE in `PUT ... : GOSUB 9000` lines (fixed decimals, trailing
    zeros cut, <= 250 chars: the compiler rejects lines > 256) whose subroutine drains the buffer into one array
    dv[]; short loops then index dv[] (GET on a big buffer is quadratic). No thousands of repeated VERT/EDGE/PGON.
  * Solid = one BODY of VERT / EDGE / PGON primitives. Top vertices 1..nV (boundary vertices first, corner vertices
    first of those), bottom copies nV+1..2nV at zBot = zMin - skirtDepth. Edges: top edges 1..nE (boundary edges
    first, directed with the terrain on their left), bottom copies nE+1..2nE, verticals 2nE+1..2nE+nVb (bottom -> top,
    one per boundary vertex). Polygons: top triangles (CCW from above), bottom triangles (the same, reversed),
    one quad per boundary edge (wall). Every edge is shared by exactly two polygons with opposite direction
    (closed, consistently oriented; checked offline in tests by replaying the script data).
  * Edge visibility: EDGE status bit 1 = invisible (GDL reference). TIN interior edges use status `edgeMode`
    (1 invisible = default, 2 = "edge of a curved surface", 0 = visible for debugging); terrain outline (top and
    bottom) visible; vertical skirt edges visible only at outline corners (turn > CORNER_DEG).
  * 3D contour lines = wire bodies (VERT + standalone EDGE with pgon 0, 0), one body per class so each gets its pen.
  * Every script PUTs and GETs its own data (balanced buffer), so PROJECT2 (2D script running the 3D script) cannot
    mix buffers. Contour data is therefore in both the 2D and 3D script (size cost reported in `info`).

The parent subtype is "Model Element": Tapir (main, 2026) refuses the General GDL Object root as parent
("Archicad cannot build the parameter section of a part directly below it", Tapir #2622).
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from . import contours as ct

TERRAIN_SKIRT = 5.0              # == build_plan.TERRAIN_SKIRT (not imported: build_plan is edited in parallel / heavier)
PART_PARENT = "Model Element"
PART_FOLDER = "OSM Context"
PART_PREFIX = "OSM Terrain "
CORNER_DEG = 30.0                # outline vertices turning more than this get a visible vertical skirt edge
CONTOUR_LIFT = 0.05              # m, 3D contour lines above the TIN (no z-fighting with the facets)
LABEL_HEIGHT_MM = 3.0            # == build_ac.CONTOUR_LABEL_HEIGHT_MM (paper mm)
DEFAULT_PENS = {"minor": 1, "major": 7, "label": 7, "outline": 7}   # == build_ac.pick_contour_pens fallback
XY_DEC, Z_DEC = 3, 3             # decimals in the script (mm)
MAX_LINE = 250                   # GDL script line limit is 256 chars (LP_XMLConverter: "Script line is too long")
MAX_PARAM_NAME = 31


# ------------------------------------------------------------------ number formatting
def _f(v: float, nd: int = 3) -> str:
    """Compact fixed-decimal number: 12.300 -> '12.3', 5.000 -> '5', -0.000 -> '0'."""
    s = f"{float(v):.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _gdl_str(s: str) -> str:
    """GDL string literal (double quotes; a double quote inside is turned into a single one)."""
    return '"' + str(s).replace('"', "'").replace("\n", " ") + '"'


# ------------------------------------------------------------------ naming
def part_name(site: dict) -> str:
    """Stable library part name from the site origin + terrain extent ('OSM Terrain 1a2b3c4d'): a re-import of the
    same site overwrites the same part (CreateLibraryParts overwriteExisting)."""
    o = site.get("origin") or {}
    xs, ys = _site_xy(site)
    key = "|".join([f"{float(o.get('lat') or 0):.6f}", f"{float(o.get('lon') or 0):.6f}",
                    f"{float(o.get('elevation') or 0):.2f}",
                    f"{min(xs):.1f},{min(ys):.1f},{max(xs):.1f},{max(ys):.1f}"])
    return PART_PREFIX + hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


def _site_xy(site: dict):
    pts = site["terrain"]["points"]
    return [float(p[0]) for p in pts], [float(p[1]) for p in pts]


def site_extent(site: dict) -> dict:
    """xy extent of terrain.points + z range (rel): {xmin, ymin, xmax, ymax, zmin, zmax} (used for A/B and hotspots)."""
    pts = site["terrain"]["points"]
    xs, ys, zs = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
    return {"xmin": float(min(xs)), "ymin": float(min(ys)), "xmax": float(max(xs)), "ymax": float(max(ys)),
            "zmin": float(min(zs)), "zmax": float(max(zs))}


# ------------------------------------------------------------------ TIN topology
def tin_solid_data(site: dict) -> dict:
    """TIN of the site in the numbering the 3D script uses.
    -> {xyz: Nx3 (top vertices, 1-based order = row order), edges: Ex2 int (1-based; boundary edges first, each
        directed with the terrain on its left; interior edges low->high), tris: Tx3 int (signed 1-based edge
        indices, CCW from above), n_vb (boundary vertices = rows 0..n_vb-1), n_vc (corner vertices = rows 0..n_vc-1),
        n_eb (boundary edges), loops: [[vertex index 1-based, ...] closed outline loops], warnings}"""
    P, T = ct._tin(site)
    warnings: list[str] = []
    if len(T) == 0:
        raise ValueError("terrain has too few points for a TIN")
    T = np.asarray(T, dtype=np.int64).copy()
    a, b, c = P[T[:, 0], :2], P[T[:, 1], :2], P[T[:, 2], :2]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    flip = cross < 0
    T[flip] = T[flip][:, [0, 2, 1]]                                       # all CCW from above
    n_tiny = int((np.abs(cross) < 1e-8).sum())
    if n_tiny:
        warnings.append(f"{n_tiny} near-zero-area TIN triangle(s) kept (topology must stay closed)")

    N = len(P)
    d = T[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)                           # directed tri edges, 3 per triangle
    key = np.minimum(d[:, 0], d[:, 1]) * N + np.maximum(d[:, 0], d[:, 1])
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    if (cnt > 2).any():
        warnings.append(f"{int((cnt > 2).sum())} non-manifold TIN edge(s) (shared by > 2 triangles)")
    is_b = cnt == 1
    # the single directed copy of every boundary edge (terrain on its left, the triangles being CCW)
    first = np.full(len(uk), -1, dtype=np.int64)
    first[inv[::-1]] = np.arange(len(inv))[::-1]                          # first occurrence of each undirected edge
    bdir = d[first[is_b]]                                                 # (a, b) original vertex indices

    # outline loops (follow a -> b), corners by turning angle
    nxt: dict[int, list[int]] = {}
    for u, v in bdir.tolist():
        nxt.setdefault(u, []).append(v)
    loops, seen_e = [], set()
    for u0, v0 in bdir.tolist():
        if (u0, v0) in seen_e:
            continue
        loop, u, v = [u0], u0, v0
        while (u, v) not in seen_e:
            seen_e.add((u, v))
            loop.append(v)
            cand = [w for w in nxt.get(v, []) if (v, w) not in seen_e]
            if not cand:
                break
            u, v = v, cand[0]
        loops.append(loop)
    corner = set()
    cos_lim = math.cos(math.radians(CORNER_DEG))
    pinch = {u for u, vs in nxt.items() if len(vs) > 1}
    for loop in loops:
        ring = loop[:-1] if len(loop) > 1 and loop[0] == loop[-1] else loop
        m = len(ring)
        for i in range(m):
            p, q, r = P[ring[i - 1], :2], P[ring[i], :2], P[ring[(i + 1) % m], :2]
            u1, u2 = q - p, r - q
            n1, n2 = float(np.hypot(*u1)), float(np.hypot(*u2))
            if n1 < 1e-12 or n2 < 1e-12 or float(u1 @ u2) / (n1 * n2) < cos_lim:
                corner.add(int(ring[i]))
    corner |= pinch

    # vertex order: corners, other boundary vertices, interior (each in original order); unused points dropped
    bverts = sorted(set(bdir.ravel().tolist()))
    used = sorted(set(T.ravel().tolist()))
    order = ([v for v in bverts if v in corner] + [v for v in bverts if v not in corner] +
             [v for v in used if v not in set(bverts)])
    new = np.full(N, -1, dtype=np.int64)
    new[np.array(order, dtype=np.int64)] = np.arange(1, len(order) + 1)

    # edge order: boundary edges (directed) first, then interior edges (low -> high new index)
    ea = new[d[first, 0]]
    eb = new[d[first, 1]]
    lo, hi = np.minimum(ea, eb), np.maximum(ea, eb)
    ea = np.where(is_b, ea, lo)
    eb = np.where(is_b, eb, hi)
    eorder = np.concatenate([np.flatnonzero(is_b), np.flatnonzero(~is_b)])
    eid = np.empty(len(uk), dtype=np.int64)
    eid[eorder] = np.arange(1, len(uk) + 1)
    edges = np.stack([ea[eorder], eb[eorder]], axis=1)
    # signed edge index per directed triangle edge
    start_new = new[d[:, 0]]
    sign = np.where(start_new == ea[inv], 1, -1)
    tris = (eid[inv] * sign).reshape(-1, 3)
    return {"xyz": P[np.array(order, dtype=np.int64)], "edges": edges, "tris": tris,
            "n_vb": len(bverts), "n_vc": sum(1 for v in bverts if v in corner), "n_eb": int(is_b.sum()),
            "loops": [[int(new[v]) for v in loop] for loop in loops], "warnings": warnings}


# ------------------------------------------------------------------ parameters
def part_parameters(info_interval: str = "", info_source: str = "", info_counts: str = "",
                    skirt: float = TERRAIN_SKIRT) -> list[dict]:
    """Tapir parameter definitions (simple types only). A and B are written by CreateLibraryParts from a / b."""
    P = [
        {"name": "sepDisplay", "type": "Title", "description": "Display"},
        {"name": "showContours3D", "type": "Boolean", "value": True, "description": "Contour lines in 3D"},
        {"name": "showContours2D", "type": "Boolean", "value": True, "description": "Contour lines on the plan"},
        {"name": "showLabels", "type": "Boolean", "value": True, "description": "Contour elevation labels (plan)"},
        {"name": "showOutline2D", "type": "Boolean", "value": True, "description": "Terrain outline (plan)"},
        {"name": "use2DProjection", "type": "Boolean", "value": False,
         "description": "Plan: project the 3D model (PROJECT2) under the contours"},
        {"name": "edgeMode", "type": "Integer", "value": 1,
         "description": "TIN edges in 3D: 0 visible, 1 hidden, 2 curved-surface edge"},
        {"name": "contourLift", "type": "Length", "value": CONTOUR_LIFT, "description": "3D contour lines above the surface"},
        {"name": "skirtDepth", "type": "Length", "value": float(skirt), "description": "Solid depth below the lowest point"},
        {"name": "sepPens", "type": "Title", "description": "Pens and line types"},
        {"name": "penMinor", "type": "PenColor", "value": DEFAULT_PENS["minor"], "description": "Minor contour pen"},
        {"name": "penMajor", "type": "PenColor", "value": DEFAULT_PENS["major"], "description": "Major contour pen"},
        {"name": "penLabel", "type": "PenColor", "value": DEFAULT_PENS["label"], "description": "Label pen"},
        {"name": "penOutline", "type": "PenColor", "value": DEFAULT_PENS["outline"], "description": "Outline / 3D edge pen"},
        {"name": "ltMinor", "type": "LineType", "value": 1, "description": "Minor contour line type"},
        {"name": "ltMajor", "type": "LineType", "value": 1, "description": "Major contour line type"},
        {"name": "labelSize", "type": "RealNumber", "value": LABEL_HEIGHT_MM, "description": "Label height (paper mm)"},
        {"name": "labelFont", "type": "String", "value": "Arial", "description": "Label font"},
        {"name": "sepMat", "type": "Title", "description": "Material"},
        {"name": "terrainMat", "type": "Material", "value": 0, "description": "Surface (0 = from building material)"},
        {"name": "terrainBMat", "type": "BuildingMaterial", "value": 0, "description": "Building material (0 = default)"},
        {"name": "infoInterval", "type": "String", "value": info_interval, "description": "Contour interval", "isHidden": True},
        {"name": "infoSource", "type": "String", "value": info_source, "description": "Source", "isHidden": True},
        {"name": "infoCounts", "type": "String", "value": info_counts, "description": "TIN / contour counts", "isHidden": True},
    ]
    for p in P:
        if len(p["name"]) > MAX_PARAM_NAME:
            raise ValueError(f"GDL parameter name too long: {p['name']}")
    return P


PARAMETER_SCRIPT = 'VALUES "edgeMode" 0, 1, 2\nVALUES "skirtDepth" RANGE [0.1, ]\nVALUES "labelSize" RANGE (0, ]\n'


# ------------------------------------------------------------------ scripts
def _hotspots_3d(ext: dict) -> list[str]:
    out = []
    for z in ("zBot", _f(ext["zmax"], Z_DEC)):
        for x, y in ((ext["xmin"], ext["ymin"]), (ext["xmax"], ext["ymin"]), (ext["xmax"], ext["ymax"]),
                     (ext["xmin"], ext["ymax"])):
            out.append(f"HOTSPOT {_f(x, XY_DEC)}, {_f(y, XY_DEC)}, {z}")
    return out


DRAIN_LABEL = 9000
# Drains the PUT buffer into the flat array dv[] (nD = values read so far). The buffer never holds more than one PUT
# line: GET on a big buffer is quadratic (LP_XMLConverter interpreter: 120k values PUT + GET(1) loop 33 s,
# chunked PUT + drain 5 s incl. start-up); arrays index in O(1).
DRAIN_SUB = ["END", "", f"{DRAIN_LABEL}:", "while NSP > 0 do", "\tnD = nD + 1", "\tdv[nD] = GET(1)", "endwhile",
             "RETURN"]


def _data_lines(values, indent: str = "") -> list[str]:
    """`PUT ... : GOSUB 9000` lines, each <= MAX_LINE chars (the GDL compiler rejects script lines > 256 chars): the
    values land in dv[] in order. `values` are formatted strings (one number or a comma-joined group)."""
    tail = f" : GOSUB {DRAIN_LABEL}"
    out, cur = [], []
    n = len(indent) + 4 + len(tail)
    for v in values:
        if cur and n + 1 + len(v) > MAX_LINE:
            out.append(indent + "PUT " + ",".join(cur) + tail)
            cur, n = [], len(indent) + 4 + len(tail)
        cur.append(v)
        n += len(v) + 1
    if cur:
        out.append(indent + "PUT " + ",".join(cur) + tail)
    return out


def _contour_vals_3d(lines: list[dict], e0: float) -> list[str]:
    vals = []
    for m in lines:
        vals += [str(len(m["points"])), _f(m["z_abs"] - e0, Z_DEC)]
        for x, y in m["points"]:
            vals += [_f(x, XY_DEC), _f(y, XY_DEC)]
    return vals


def _contour_loop_3d(n_lines: int, pen: str) -> list[str]:
    """Wire body of one contour class from dv[] at pointer cP: n, z_rel, x, y, ... -> VERT + standalone EDGE."""
    return [f"\tPEN {pen}", "\tcK = 0", f"\tfor cL = 1 to {n_lines}", "\t\tcN = dv[cP + 1]",
            "\t\tcZ = dv[cP + 2] + contourLift", "\t\tcP = cP + 2", "\t\tfor i = 1 to cN",
            "\t\t\tVERT dv[cP + 1], dv[cP + 2], cZ", "\t\t\tcP = cP + 2", "\t\t\tcK = cK + 1",
            "\t\t\tif i > 1 then EDGE cK - 1, cK, 0, 0, 0", "\t\tnext i", "\tnext cL", "\tBODY -1"]


def _lines_vals_2d(lines_xy: list[list]) -> list[str]:
    vals = []
    for pts in lines_xy:
        vals.append(str(len(pts)))
        for x, y in pts:
            vals += [_f(x, XY_DEC), _f(y, XY_DEC)]
    return vals


def _lines_loop_2d(n_lines: int, pen: str, ltype: str | None) -> list[str]:
    """2D polylines from dv[] at pointer cP: n, x, y, ... -> LINE2 per segment."""
    out = [f"\tPEN {pen}"] + ([f"\tLINE_TYPE {ltype}"] if ltype else [])
    return out + [f"\tfor cL = 1 to {n_lines}", "\t\tcN = dv[cP + 1]", "\t\tcX = dv[cP + 2]", "\t\tcY = dv[cP + 3]",
                  "\t\tcP = cP + 3", "\t\tfor i = 2 to cN", "\t\t\tLINE2 cX, cY, dv[cP + 1], dv[cP + 2]",
                  "\t\t\tcX = dv[cP + 1]", "\t\t\tcY = dv[cP + 2]", "\t\t\tcP = cP + 2", "\t\tnext i", "\tnext cL"]


def script_3d(tin: dict, con: dict, e0: float, ext: dict, header: str) -> str:
    xyz, E, Tr = tin["xyz"], tin["edges"], tin["tris"]
    nV, nE, nT = len(xyz), len(E), len(Tr)
    L = [header,
         f"! TIN: {nV} vertices ({tin['n_vb']} on the outline, {tin['n_vc']} corners), {nE} edges "
         f"({tin['n_eb']} outline), {nT} triangles",
         f"nV = {nV} : nE = {nE} : nT = {nT} : nVb = {tin['n_vb']} : nVc = {tin['n_vc']} : nEb = {tin['n_eb']}",
         f"zBot = {_f(ext['zmin'], Z_DEC)} - skirtDepth",
         "eSt = 1",
         "if edgeMode = 0 then eSt = 0",
         "if edgeMode = 2 then eSt = 2",
         "DIM dv[]",
         "nD = 0"]
    L += _hotspots_3d(ext)
    L.append("! data -> dv[]: vertices x, y, z (3 nV) | edges v1, v2 (2 nE) | triangles e1, e2, e3 (3 nT, signed)")
    L += _data_lines(f"{_f(x, XY_DEC)},{_f(y, XY_DEC)},{_f(z, Z_DEC)}" for x, y, z in xyz.tolist())
    L += _data_lines(f"{a},{b}" for a, b in E.tolist())
    L += _data_lines(f"{a},{b},{c}" for a, b, c in Tr.tolist())
    L += ["oE = 3 * nV",
          "oT = oE + 2 * nE",
          "BASE",
          "if terrainBMat > 0 then BUILDING_MATERIAL terrainBMat",
          "if terrainMat > 0 then MATERIAL terrainMat",
          "PEN penOutline",
          "! top vertices 1..nV, their copies at zBot nV+1..2nV",
          "for i = 1 to nV",
          "\tVERT dv[3 * i - 2], dv[3 * i - 1], dv[3 * i]",
          "next i",
          "for i = 1 to nV",
          "\tVERT dv[3 * i - 2], dv[3 * i - 1], zBot",
          "next i",
          "! top edges (outline visible, TIN edges by edgeMode), bottom copies, vertical skirt edges (corners visible)",
          "for i = 1 to nE",
          "\tst = eSt",
          "\tif i <= nEb then st = 0",
          "\tEDGE dv[oE + 2 * i - 1], dv[oE + 2 * i], -1, -1, st",
          "next i",
          "for i = 1 to nE",
          "\tst = 1",
          "\tif i <= nEb then st = 0",
          "\tEDGE dv[oE + 2 * i - 1] + nV, dv[oE + 2 * i] + nV, -1, -1, st",
          "next i",
          "for i = 1 to nVb",
          "\tst = 1",
          "\tif i <= nVc then st = 0",
          "\tEDGE i + nV, i, -1, -1, st",
          "next i",
          "! top triangles (CCW from above), bottom triangles (reversed), skirt walls (one quad per outline edge)",
          "for i = 1 to nT",
          "\tPGON 3, 0, 0, dv[oT + 3 * i - 2], dv[oT + 3 * i - 1], dv[oT + 3 * i]",
          "next i",
          "for i = 1 to nT",
          "\tt1 = dv[oT + 3 * i - 2]",
          "\tt2 = dv[oT + 3 * i - 1]",
          "\tt3 = dv[oT + 3 * i]",
          "\tPGON 3, 0, 0, -(t3 + SGN(t3) * nE), -(t2 + SGN(t2) * nE), -(t1 + SGN(t1) * nE)",
          "next i",
          "for i = 1 to nEb",
          "\tPGON 4, 0, 0, i + nE, 2 * nE + dv[oE + 2 * i], -i, -(2 * nE + dv[oE + 2 * i - 1])",
          "next i",
          "BODY -1"]
    if con["minor"] or con["major"]:
        L += ["! contour lines (wire bodies, lifted by contourLift)", "if showContours3D then", "\tcP = nD"]
        L += _data_lines(_contour_vals_3d(con["minor"], e0) + _contour_vals_3d(con["major"], e0), "\t")
        if con["minor"]:
            L += _contour_loop_3d(len(con["minor"]), "penMinor")
        if con["major"]:
            L += _contour_loop_3d(len(con["major"]), "penMajor")
        L.append("endif")
    return "\n".join(L + DRAIN_SUB) + "\n"


def script_2d(tin: dict, con: dict, ext: dict, header: str) -> str:
    xyz = tin["xyz"]
    L = [header,
         "if use2DProjection then PROJECT2 3, 270, 2",
         f"HOTSPOT2 {_f(ext['xmin'], XY_DEC)}, {_f(ext['ymin'], XY_DEC)}",
         f"HOTSPOT2 {_f(ext['xmax'], XY_DEC)}, {_f(ext['ymin'], XY_DEC)}",
         f"HOTSPOT2 {_f(ext['xmax'], XY_DEC)}, {_f(ext['ymax'], XY_DEC)}",
         f"HOTSPOT2 {_f(ext['xmin'], XY_DEC)}, {_f(ext['ymax'], XY_DEC)}",
         "DIM dv[]",
         "nD = 0"]
    outline = [[(xyz[v - 1, 0], xyz[v - 1, 1]) for v in loop] for loop in tin["loops"] if len(loop) >= 2]
    if outline:
        L += ["if showOutline2D then", "\tcP = nD"] + _data_lines(_lines_vals_2d(outline), "\t")
        L += _lines_loop_2d(len(outline), "penOutline", None) + ["endif"]
    minor = [m["points"] for m in con["minor"]]
    major = [m["points"] for m in con["major"]]
    if minor or major:
        L += ["if showContours2D then", "\tcP = nD"] + _data_lines(_lines_vals_2d(minor) + _lines_vals_2d(major), "\t")
        if minor:
            L += _lines_loop_2d(len(minor), "penMinor", "ltMinor")
        if major:
            L += _lines_loop_2d(len(major), "penMajor", "ltMajor")
        L.append("endif")
    if con.get("labels"):
        L += ["if showLabels then",
              '\tDEFINE STYLE "osmCtxContour" labelFont, labelSize, 5, 0',
              '\tSET STYLE "osmCtxContour"',
              "\tPEN penLabel"]
        for lb in con["labels"]:
            L.append(f"\tADD2 {_f(lb['x'], XY_DEC)}, {_f(lb['y'], XY_DEC)} : ROT2 {_f(math.degrees(lb['angle']), 2)} : "
                     f"TEXT2 0, 0, {_gdl_str(lb['text'])} : DEL 2")
        L.append("endif")
    return "\n".join(L + DRAIN_SUB) + "\n"


# ------------------------------------------------------------------ public
def terrain_gdl_part(site: dict, interval: float | None = None, scale: float = 1.0, skirt: float = TERRAIN_SKIRT,
                     name: str | None = None, source: str = "") -> tuple[dict, dict]:
    """-> (CreateLibraryParts item, info). `interval`/`scale` as contours.build_contours (None = auto);
    `skirt` = default skirtDepth (m below the lowest terrain point); `name` default = part_name(site)."""
    name = name or part_name(site)
    e0 = float((site.get("origin") or {}).get("elevation") or 0.0)
    ext = site_extent(site)
    tin = tin_solid_data(site)
    con = ct.build_contours(site, interval=interval, scale=scale, labels=True)
    nV, nE, nT = len(tin["xyz"]), len(tin["edges"]), len(tin["tris"])
    iv_txt = f"{con['interval']:g} m ({con['interval_source']}), major every {con['major_every']}"
    counts_txt = (f"TIN {nV} v / {nT} t; contours {con['counts']['minor']} minor, {con['counts']['major']} major, "
                  f"{con['counts']['labels']} labels")
    header = (f"! {name}: OSM Context terrain, generated by osmctx/terrain_gdl.py (re-import regenerates it).\n"
              f"! local x/y = site x/y (m), local z = elevation - {_f(e0, 3)} m (site origin elevation)")
    s3 = script_3d(tin, con, e0, ext, header)
    s2 = script_2d(tin, con, ext, header)
    a = max(ext["xmax"] - ext["xmin"], 0.01)
    b = max(ext["ymax"] - ext["ymin"], 0.01)
    part = {"name": name,
            "parentName": PART_PARENT,
            "type": "Object",
            "path": f"{PART_FOLDER}/{name}.gsm",
            "a": round(a, 4),
            "b": round(b, 4),
            "masterScript": "",
            "script2D": s2,
            "script3D": s3,
            "parameterScript": PARAMETER_SCRIPT,
            "parameters": part_parameters(iv_txt, source or "OSM Context", counts_txt, skirt)}
    sizes = {"script3D": len(s3.encode("utf-8")), "script2D": len(s2.encode("utf-8")),
             "parameterScript": len(PARAMETER_SCRIPT)}
    sizes["total"] = sum(sizes.values())
    info = {"name": name, "path": part["path"], "parent": PART_PARENT,
            "interval": con["interval"], "interval_source": con["interval_source"],
            "counts": {"vertices": nV, "edges": nE, "triangles": nT, "boundary_vertices": tin["n_vb"],
                       "corners": tin["n_vc"], "boundary_edges": tin["n_eb"], "outline_loops": len(tin["loops"]),
                       "body_vertices": 2 * nV, "body_edges": 2 * nE + tin["n_vb"], "body_polygons": 2 * nT + tin["n_eb"],
                       **{f"contour_{k}": v for k, v in con["counts"].items()}},
            "script_bytes": sizes,
            "extent": {k: round(v, 3) for k, v in ext.items()},
            "z_bottom_rel": round(ext["zmin"] - skirt, 3),
            "z_convention": "local z = site z rel. origin.elevation; place at z = dz (UNVERIFIED live)",
            "warnings": tin["warnings"] + list(con.get("warnings") or [])}
    return part, info


def terrain_gdl_object(site: dict, part_name: str, dz: float = 0.0, layer_index: int | None = None,
                       params: dict | None = None) -> dict:
    """CreateObjects item placing the terrain part: origin (0, 0), angle 0 -> local coords = site coords;
    coordinates.z = dz (absolute; local z is site z rel. origin.elevation, so a point lands at dz + z_rel, like
    terrain_mesh; UNVERIFIED live). dimensions = the part's own A/B (ratio 1). `params` = {gdl name: value}
    overrides, e.g. {"penMinor": 3, "penMajor": 5} from build_ac.pick_contour_pens."""
    ext = site_extent(site)
    item = {"floorIndex": 0,
            "libraryPartName": part_name,
            "coordinates": {"x": 0.0, "y": 0.0, "z": round(float(dz), 4)},
            "dimensions": {"x": round(max(ext["xmax"] - ext["xmin"], 0.01), 4),
                           "y": round(max(ext["ymax"] - ext["ymin"], 0.01), 4),
                           "z": round(ext["zmax"] - ext["zmin"] + TERRAIN_SKIRT, 4)},
            "angle": 0.0}
    if params:
        item["parameters"] = [{"name": k, "value": v} for k, v in params.items()]
    if layer_index is not None:
        item["layerIndex"] = int(layer_index)
    return item


def create_library_parts_params(parts: list[dict], overwrite: bool = True) -> dict:
    """Full CreateLibraryParts parameters (overwriteExisting needs AC27+)."""
    return {"libraryParts": list(parts), "overwriteExisting": bool(overwrite)}


# ------------------------------------------------------------------ offline LP_XMLConverter source (tests / debugging)
_XML_TYPE = {"Integer": "Integer", "Length": "Length", "Angle": "Angle", "RealNumber": "RealNum",
             "LightSwitch": "LightSwitch", "RGBColor": "ColorRGB", "Intensity": "Intensity", "LineType": "LineType",
             "Material": "Material", "FillPattern": "FillPattern", "PenColor": "PenColor", "String": "String",
             "Boolean": "Boolean", "Separator": "Separator", "Title": "Title",
             "BuildingMaterial": "BuildingMaterial", "Profile": "Profile"}
GUID_GENERAL_GDL_OBJECT = "F938E33A-329D-4A36-BE3E-85E126820996"
GUID_MODEL_ELEMENT = "103E8D2C-8230-42E1-9597-46F84CCE28C0"


def _cdata(s: str) -> str:
    return "<![CDATA[" + s.replace("]]>", "]]]]><![CDATA[>") + "]]>"


def _xml_value(p: dict) -> str:
    t, v = p["type"], p.get("value")
    if t in ("Separator", "Title"):
        return ""
    if t == "String":
        return f"<Value>{_cdata(_gdl_str(v if v is not None else ''))}</Value>"
    if t == "Boolean":
        return f"<Value>{1 if v else 0}</Value>"
    if t == "LightSwitch":
        return f"<Value>{1 if v == 'On' else 0}</Value>"
    return f"<Value>{_f(v or 0, 6)}</Value>"


def _xml_param(name, typ, desc, value_xml, fix=False, hidden=False) -> str:
    flags = "<Flags><ParFlg_Hidden/></Flags>" if hidden else ""
    return (f'\t\t<{typ} Name="{name}"><Description>{_cdata(_gdl_str(desc))}</Description>'
            f'{"<Fix/>" if fix else ""}{flags}{value_xml}</{typ}>')


def libpart_xml(part: dict, main_guid: str, ancestry=(GUID_GENERAL_GDL_OBJECT, GUID_MODEL_ELEMENT),
                placeable: bool = True) -> str:
    """The part as LP_XMLConverter source XML (xml2libpart / convertlibrary), for the offline compile check. Mirrors
    what CreateLibraryParts writes: A/B (+ the fixed object parameters), the parameters, master/2D/3D/parameter
    scripts. Not used by the importer itself."""
    rows = [_xml_param("A", "Length", "Width", f"<Value>{_f(part.get('a', 1), 6)}</Value>", fix=True),
            _xml_param("B", "Length", "Depth", f"<Value>{_f(part.get('b', 1), 6)}</Value>", fix=True),
            _xml_param("ZZYZX", "Length", "Height", "<Value>1</Value>", fix=True),
            _xml_param("AC_show2DHotspotsIn3D", "Boolean", "Show 2D Hotspots in 3D", "<Value>0</Value>", fix=True),
            _xml_param("ac_bottomlevel", "Length", "Bottom Level", "<Value>1</Value>", fix=True),
            _xml_param("ac_toplevel", "Length", "Top Level", "<Value>0</Value>", fix=True)]
    for p in part.get("parameters") or []:
        rows.append(_xml_param(p["name"], _XML_TYPE[p["type"]], p.get("description") or p["name"], _xml_value(p),
                               hidden=bool(p.get("isHidden"))))
    anc = "".join(f"\n\t<MainGUID>{g}</MainGUID>" for g in ancestry)

    def sect(tag, text):
        return f'<{tag} SectVersion="20" SectionFlags="0" SubIdent="0">\n{_cdata(chr(10) + (text or "") + chr(10))}\n</{tag}>\n'

    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<Symbol IsArchivable="false" IsPlaceable="{"true" if placeable else "false"}" MainGUID="{main_guid}" '
            f'MigrationValue="Normal" Owner="0" Signature="0" Version="46">\n'
            f'<Ancestry SectVersion="1" SectionFlags="0" SubIdent="0" Template="{"false" if placeable else "true"}">'
            f'{anc}\n</Ancestry>\n'
            f'<CalledMacros SectVersion="2" SectionFlags="0" SubIdent="0">\n</CalledMacros>\n'
            f'<ParamSection SectVersion="27" SectionFlags="0" SubIdent="0">\n\t<ParamSectHeader>\n'
            f'\t\t<AutoHotspots>false</AutoHotspots>\n\t\t<StatBits>\n\t\t\t<STBit_FixSize/>\n\t\t</StatBits>\n'
            f'\t\t<WDLeftFrame>0</WDLeftFrame>\n\t\t<WDRightFrame>0</WDRightFrame>\n\t\t<WDTopFrame>0</WDTopFrame>\n'
            f'\t\t<WDBotFrame>0</WDBotFrame>\n\t\t<LayFlags>65535</LayFlags>\n'
            f'\t\t<WDMirrorThickness>0</WDMirrorThickness>\n\t\t<WDWallInset>0</WDWallInset>\n\t</ParamSectHeader>\n'
            f'\t<Parameters>\n' + "\n".join(rows) + '\n\t</Parameters>\n</ParamSection>\n'
            + sect("Script_1D", part.get("masterScript"))
            + sect("Script_2D", part.get("script2D"))
            + sect("Script_3D", part.get("script3D"))
            + sect("Script_VL", part.get("parameterScript"))
            + '</Symbol>\n')
