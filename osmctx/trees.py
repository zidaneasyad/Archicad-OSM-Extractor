"""Tree size data + helpers (pure, offline).

GENUS_TABLE holds TYPICAL mature heights (m) and crown-diameter / height ratios per genus. They are
rule-of-thumb values from general dendrology references for open-grown/urban trees, NOT measurements of any
individual tree: young, pruned, shaded or old trees differ by a lot (tens of percent). They are only used
when OSM has no `height` tag and no measured height is available, and every tree records which source
produced its size (`height_source`, `crown_source`).

Entry: genus -> (typical_height_m, crown_ratio, leaf_type)
"""
from __future__ import annotations

import math
import re

import numpy as np

B, N = "broadleaved", "needleleaved"

GENUS_TABLE: dict[str, tuple[float, float, str]] = {
    # Central Europe
    "Platanus": (25.0, 0.60, B), "Tilia": (20.0, 0.60, B), "Acer": (15.0, 0.60, B),
    "Quercus": (22.0, 0.70, B), "Fagus": (25.0, 0.65, B), "Aesculus": (20.0, 0.65, B),
    "Betula": (18.0, 0.40, B), "Fraxinus": (22.0, 0.60, B), "Carpinus": (15.0, 0.50, B),
    "Prunus": (8.0, 0.80, B), "Malus": (6.0, 0.90, B), "Robinia": (18.0, 0.55, B),
    "Populus": (28.0, 0.40, B), "Salix": (15.0, 0.70, B), "Ulmus": (22.0, 0.65, B),
    "Alnus": (18.0, 0.50, B), "Sorbus": (10.0, 0.60, B), "Juglans": (18.0, 0.80, B),
    "Pinus": (20.0, 0.35, N), "Picea": (25.0, 0.30, N), "Abies": (28.0, 0.30, N),
    "Larix": (28.0, 0.30, N), "Taxus": (10.0, 0.60, N), "Thuja": (12.0, 0.30, N),
    "Cedrus": (25.0, 0.50, N), "Pseudotsuga": (35.0, 0.30, N),
    # Indonesia / tropical
    "Ficus": (20.0, 1.00, B), "Mangifera": (15.0, 0.90, B), "Swietenia": (30.0, 0.60, B),
    "Tectona": (28.0, 0.45, B), "Cocos": (20.0, 0.30, B), "Samanea": (20.0, 1.30, B),
    "Pterocarpus": (25.0, 0.90, B), "Delonix": (12.0, 1.30, B),
}
# a few species that differ strongly from their genus value: "Genus species" -> (height_m, crown_ratio)
SPECIES_TABLE: dict[str, tuple[float, float]] = {
    "Acer platanoides": (20.0, 0.60), "Acer pseudoplatanus": (25.0, 0.60), "Acer campestre": (12.0, 0.65),
    "Tilia cordata": (22.0, 0.55), "Tilia platyphyllos": (25.0, 0.60), "Quercus robur": (25.0, 0.70),
    "Pinus sylvestris": (22.0, 0.30), "Pinus nigra": (20.0, 0.35), "Picea abies": (30.0, 0.25),
    "Prunus avium": (15.0, 0.65), "Betula pendula": (20.0, 0.35),
}
CONIFER_GENERA = frozenset(g for g, v in GENUS_TABLE.items() if v[2] == N)
DEFAULT_CROWN_RATIO_UNKNOWN = 0.6
MEASURE_RADIUS_M = 2.0
MEASURE_ACCEPT_M = (2.0, 45.0)

_WORD = re.compile(r"[A-Za-z]+")


def taxon(tags: dict) -> tuple[str | None, str | None]:
    """(Genus, 'Genus species') from OSM genus/species/taxon tags (first of `a;b`, hybrid signs ignored)."""
    genus = species = None
    for k in ("species", "taxon", "genus"):
        v = tags.get(k)
        if not v:
            continue
        words = _WORD.findall(str(v).split(";")[0])
        if not words:
            continue
        g = words[0].capitalize()
        if k in ("species", "taxon") and len(words) >= 2:
            sp = words[1].lower()
            if sp != "x" and species is None:
                species = f"{g} {sp}"
        if genus is None:
            genus = g
    return genus, species


def is_conifer(tags: dict) -> bool:
    """True when leaf_type says needleleaved, or (no leaf_type) the genus is a known conifer genus."""
    lt = str(tags.get("leaf_type") or "").lower()
    if lt:
        return lt == N
    g, _ = taxon(tags)
    return g in CONIFER_GENERA


def lookup(tags: dict) -> tuple[float, float] | None:
    """(typical_height, crown_ratio) from the species then genus table, or None."""
    g, sp = taxon(tags)
    if sp in SPECIES_TABLE:
        return SPECIES_TABLE[sp]
    if g in GENUS_TABLE:
        h, r, _ = GENUS_TABLE[g]
        return h, r
    return None


def measure_height(cx, cy, vals, x: float, y: float, r: float = MEASURE_RADIUS_M) -> float | None:
    """Height above ground at (x, y): max of nDOM cell values (cell centres cx, cy, same CRS/units as x, y)
    within r. NaN cells are ignored. Accepted only inside MEASURE_ACCEPT_M, else None."""
    cx, cy, vals = np.asarray(cx, float), np.asarray(cy, float), np.asarray(vals, float)
    m = ((cx - x) ** 2 + (cy - y) ** 2 <= r * r) & np.isfinite(vals)
    if not m.any():
        return None
    h = float(vals[m].max())
    lo, hi = MEASURE_ACCEPT_M
    return h if lo <= h <= hi else None


def size_tree(tags: dict, tag_height, tag_crown, measured, s: dict) -> dict:
    """Fallback chain -> {height, height_source, crown_diameter, crown_source}. tag_height/tag_crown are
    already parsed metres (or None); measured = nDOM height or None; s = converter settings."""
    use_g = bool(s.get("use_genus_table", True))
    tbl = lookup(tags) if use_g else None
    conifer = is_conifer(tags)
    lt = str(tags.get("leaf_type") or "").lower()
    if tag_height is not None:
        h, hs = tag_height, "tag"
    elif measured is not None:
        h, hs = measured, "measured"
    elif tbl is not None:
        h, hs = tbl[0], "genus"
    elif lt in (B, N):
        h, hs = float(s["conifer_height"] if lt == N else s["broadleaf_height"]), "leaf_type"
    else:
        h, hs = float(s["tree_height"]), "default"
    if tag_crown is not None:
        c, cs = tag_crown, "tag"
    else:
        if tbl is not None:
            ratio = tbl[1]
        elif lt in (B, N) or conifer:
            ratio = float(s["conifer_crown_ratio"] if conifer else s["broadleaf_crown_ratio"])
        else:
            ratio = DEFAULT_CROWN_RATIO_UNKNOWN
        c, cs = h * ratio, "ratio"
    return {"height": round(h, 3), "height_source": hs, "crown_diameter": round(max(c, 0.5), 3), "crown_source": cs}


def trunk_diameter(circumference_m: float | None) -> float | None:
    return round(circumference_m / math.pi, 3) if circumference_m else None
