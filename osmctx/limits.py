"""Area tiers, super-light preset and run-time estimate. ONE place for the thresholds:
server.py exposes them via GET /api/limits, the UI reads them there (never hard-codes them)."""
from __future__ import annotations

import math
from pathlib import Path

# ---- thresholds (km2) ------------------------------------------------------------------------------------
NORMAL_MAX_KM2 = 0.25        # <= this: normal, no warning
SUPER_LIGHT_ABOVE_KM2 = 5.0  # > this: super-light preset applied automatically (warning + confirm from NORMAL_MAX)
BLOCK_ABOVE_KM2 = 25.0       # > this: Fetch blocked (public Overpass / Archicad limits)

THRESHOLDS = {"normal_max_km2": NORMAL_MAX_KM2, "super_light_above_km2": SUPER_LIGHT_ABOVE_KM2,
              "block_above_km2": BLOCK_ABOVE_KM2}

# UI-level settings bundle (keys as sent by the UI; server.map_settings passes terrain_min_spacing and
# road_classes_exclude on to the converter, tree_mode goes to the builder)
SUPER_LIGHT_NAME = "super_light"
SUPER_LIGHT = {
    "tree_mode": "2d",                      # 2D tree symbols only (no 3D GPU load)
    "terrain_min_spacing": 20.0,            # m, terrain grid never finer than this (auto cap 2500 pts still applies)
    "road_classes_exclude": ["footway", "path", "cycleway", "steps", "service"],
    "include": {"shrubs": False,            # no shrubs/hedges (merged into the UI `include`, other keys kept)
                "railways": True, "walls": True,   # super-light keeps railways + walls, the other feature types are off
                "building_parts": False, "ground": False, "piers": False, "power": False},
    "forest_scatter": False,                # no tree scatter in woods/forests
    "tree_scatter": {"mangrove": False, "orchard": False, "plant_nursery": False, "park": False, "cemetery": False,
                     "forest": False},      # no estimated area trees at all
    "skip_ndom": True,                      # fetch must not download the nDOM (tree heights fall back down the chain)
    "contour_interval_scale": 2.0,          # contours stay on, interval doubled (fewer lines/vertices)
}

TIER_MESSAGES = {
    "normal": "",
    "large": "Large area - this will take a while.",
    "super_light": "Very large area - this will take a long time. Super-light preset applied automatically.",
    "blocked": ("Area is over {block:g} km2. Fetch is blocked (public Overpass and Archicad limits). "
                "Split it into smaller areas and import them one by one."),
}


def tier_of(area_km2: float) -> str:
    """normal (<=0.25) | large (<=5) | super_light (<=25) | blocked (>25). Boundaries belong to the lower tier."""
    if area_km2 <= NORMAL_MAX_KM2:
        return "normal"
    if area_km2 <= SUPER_LIGHT_ABOVE_KM2:
        return "large"
    if area_km2 <= BLOCK_ABOVE_KM2:
        return "super_light"
    return "blocked"


def limits_payload() -> dict:
    return {**THRESHOLDS, "super_light_name": SUPER_LIGHT_NAME, "super_light": SUPER_LIGHT,
            "messages": {k: v.format(block=BLOCK_ABOVE_KM2) for k, v in TIER_MESSAGES.items()}}


def resolve(area_km2: float, ui_settings: dict | None, override: bool = False) -> dict:
    """Tier + effective UI settings. super_light bundle is laid over the settings unless `override`."""
    tier = tier_of(area_km2)
    preset = tier == "super_light" and not override
    eff = dict(ui_settings or {})
    if preset:
        for k, v in SUPER_LIGHT.items():
            if isinstance(v, dict):   # e.g. include: merge, do not replace the user's other include flags
                eff[k] = {**(eff.get(k) or {}), **v}
            else:
                eff[k] = list(v) if isinstance(v, list) else v
    return {"area_km2": round(area_km2, 4), "tier": tier, "preset": SUPER_LIGHT_NAME if preset else None,
            "override": bool(override) and tier == "super_light", "settings": eff,
            "blocked": tier == "blocked", "message": TIER_MESSAGES[tier].format(block=BLOCK_ABOVE_KM2)}


def boundary_area_km2(boundary: dict) -> float:
    from .boundary import area_m2
    return area_m2(boundary) / 1e6


def guard(boundary: dict, ui_settings: dict | None, override: bool = False) -> dict:
    return resolve(boundary_area_km2(boundary), ui_settings, override)


# ---- estimate --------------------------------------------------------------------------------------------
DGM1_TILE_S = 70.0           # first download of one 1 km DGM1 sub-tile (LGL ~25-40 kB/s)
NDOM_TILE_S = 70.0           # same for nDOM1 (separate zips)
COPERNICUS_TILE_S = 4.0      # one GLO-30 1 deg tile via vsicurl
OVERPASS_BASE_S, OVERPASS_PER_KM2_S = 8.0, 6.0
CONVERT_BASE_S, CONVERT_PER_KM2_S = 5.0, 4.0
BUILD_PER_ELEMENT_S = 0.3    # Tuebingen: 142 elements ~ 45 s incl. verify
ROAD_BEAMS_PER_WAY = 3       # roads are built as one beam per segment
SUPER_LIGHT_ROAD_FACTOR = 0.5  # ASSUMED share of road ways left after dropping footway/path/cycleway/steps/service
LOW_FACTOR, HIGH_FACTOR = 0.6, 1.7
# per km2, used only when no OSM counts are available (rough mixed-urban guess, flagged in the result)
AREA_DENSITY = {"buildings": 600, "roads": 120, "trees": 300, "water": 10}
HEDGE_SEGMENTS_PER_WAY = 6   # ASSUMED: ~35 m hedge way / 6 m segments (real length only known after convert)
BARRIER_ELEMENTS_PER_WAY = 6   # ASSUMED: ~30 m wall/fence way / 5 m segments
RAIL_BEAMS_PER_WAY = 3         # like roads
SCATTER_FILL_FACTOR = 0.5    # ASSUMED share of an element's bounding box that is actually the area


def dem_tiles(boundary: dict, dem_source: str, cache_dir) -> dict:
    """DEM tiles a fetch would need and how many are not yet in cache/_tiles."""
    from . import fetch as F
    from .boundary import bbox, to_polygon
    poly = to_polygon(boundary)
    cache_dir = Path(cache_dir)
    if dem_source.startswith("file:"):
        return {"source": "file", "dgm1_total": 0, "dgm1_missing": 0, "ndom_total": 0, "ndom_missing": 0,
                "copernicus_tiles": 0}
    use_bw = dem_source == "bw_dgm1"
    if dem_source in ("auto", "layered"):
        try:
            use_bw = F.in_bw_envelope(bbox(poly, F.DEM_MARGIN_M))
        except Exception:  # noqa: BLE001
            use_bw = False
    if use_bw:
        subs = F.bw_subtiles(F.bbox_25832(bbox(poly, F.DEM_MARGIN_M)))
        dg = cache_dir / "_tiles" / "bw_dgm1"
        nd = cache_dir / "_tiles" / "bw_ndom1"
        dg_miss = sum(1 for e, n in subs
                      if not (dg / f"dgm1_32_{e}_{n}.tif").exists() and not (dg / f"dgm1_32_{e}_{n}.json").exists())
        nd_miss = sum(1 for e, n in subs
                      if not (nd / f"ndom1_32_{e}_{n}.tif").exists() and not (nd / f"ndom1_32_{e}_{n}.missing").exists())
        return {"source": "bw_dgm1", "dgm1_total": len(subs), "dgm1_missing": dg_miss,
                "ndom_total": len(subs), "ndom_missing": nd_miss, "copernicus_tiles": 0}
    if dem_source not in ("auto", "copernicus"):
        return {"source": dem_source, "dgm1_total": 0, "dgm1_missing": 0, "ndom_total": 0, "ndom_missing": 0,
                "copernicus_tiles": 0}
    n = len(F.copernicus_tile_urls(bbox(poly, F.DSM_MARGIN_M)))
    return {"source": "copernicus", "dgm1_total": 0, "dgm1_missing": 0, "ndom_total": 0, "ndom_missing": 0,
            "copernicus_tiles": n}


def format_range(low_s: float, high_s: float) -> str:
    if high_s < 90:
        return f"about {int(5 * math.floor(low_s / 5)) or 5}-{int(5 * math.ceil(high_s / 5))} s"
    lo, hi = max(1, math.floor(low_s / 60)), max(2, math.ceil(high_s / 60))
    if hi <= lo:
        hi = lo + 1
    if hi >= 120:
        return f"about {lo / 60:.1f}-{hi / 60:.1f} h"
    return f"about {lo}-{hi} min"


def vegetation_counts(counts: dict | None, settings: dict | None, preset: bool = False) -> dict:
    """Rough shrub / hedge-segment / forest-scatter / area-scatter object counts from the OSM counts (keys hedges,
    shrubs, shrub_m2, forest_m2, mangrove_m2, orchard_m2, plant_nursery_m2, park_m2, cemetery_m2 are optional, added
    by server.count_from_osm). settings = effective UI settings."""
    from . import vegetation as veg
    c, s = counts or {}, settings or {}
    inc = s.get("include") or {}
    shrubs_on = bool(inc.get("shrubs", True)) and not preset
    scatter_on = bool(s.get("forest_scatter", False) or (s.get("tree_scatter") or {}).get("forest")) and not preset
    shrubs = int(c.get("shrubs", 0) + c.get("shrub_m2", 0) * SCATTER_FILL_FACTOR / 10.0) if shrubs_on else 0
    hedges = int(c.get("hedges", 0) * HEDGE_SEGMENTS_PER_WAY) if shrubs_on else 0
    scatter = int(c.get("forest_m2", 0) * SCATTER_FILL_FACTOR / float(s.get("forest_area_per_tree") or 40.0)) if scatter_on else 0
    ts = {**veg.VEG_DEFAULTS["tree_scatter"], **(s.get("tree_scatter") or {})}
    by_type = {}
    for typ in veg.TREE_AREA_ORDER:
        if preset or not ts.get(typ):
            continue
        n = int(c.get(f"{typ}_m2", 0) * SCATTER_FILL_FACTOR / veg.scatter_params(s, typ)["area_per_tree"])
        if n:
            by_type[typ] = n
    return {"shrubs": shrubs, "hedge_segments": hedges, "scatter_trees": scatter + sum(by_type.values()),
            "scatter_by_type": ({"forest": scatter, **by_type} if scatter else by_type)}


def feature_counts(counts: dict | None, settings: dict | None, preset: bool = False) -> dict:
    """Rough element counts of the "More features" types from the OSM counts (keys railways, platforms, barriers,
    building_parts, ground, piers, power_towers optional, added by server.count_from_osm). Types switched off in the
    effective settings (include.*) give 0; defaults on except power. Preset: super-light include flags apply."""
    c = counts or {}
    inc = {"railways": True, "walls": True, "building_parts": True, "ground": True, "piers": True, "power": False}
    inc.update((settings or {}).get("include") or {})
    if preset:
        inc.update(SUPER_LIGHT["include"])
    on = lambda k: bool(inc.get(k))  # noqa: E731
    return {"railway_beams": int(c.get("railways", 0) * RAIL_BEAMS_PER_WAY) if on("railways") else 0,
            "platforms": int(c.get("platforms", 0)) if on("railways") else 0,
            "barrier_elements": int(c.get("barriers", 0) * BARRIER_ELEMENTS_PER_WAY) if on("walls") else 0,
            "building_parts": int(c.get("building_parts", 0)) if on("building_parts") else 0,
            "ground": int(c.get("ground", 0)) if on("ground") else 0,
            "piers": int(c.get("piers", 0)) if on("piers") else 0,
            "power_towers": int(c.get("power_towers", 0)) if on("power") else 0}


def estimate(area_km2: float, counts: dict | None, counts_source: str, tiles: dict, preset: bool = False,
             settings: dict | None = None) -> dict:
    """Time range + expected element count. counts: {buildings, roads, trees, water} or None (area-only guess).
    settings (effective UI settings) switches on the shrub/hedge/forest-scatter objects (see vegetation_counts)."""
    guessed = counts is None
    c = dict(counts) if counts else {k: int(round(v * area_km2)) for k, v in AREA_DENSITY.items()}
    roads = c.get("roads", 0) * (SUPER_LIGHT_ROAD_FACTOR if preset else 1.0)
    ov_on = bool(((settings or {}).get("include") or {}).get("overture_buildings", True))
    elements = {"buildings": c.get("buildings", 0) + (c.get("overture_buildings", 0) if ov_on and not guessed else 0),
                "buildings_overture": c.get("overture_buildings", 0) if ov_on and not guessed else 0,
                "roads_ways": int(round(roads)),
                "road_beams": int(round(roads * ROAD_BEAMS_PER_WAY)), "trees": c.get("trees", 0),
                "water": c.get("water", 0), "terrain": 1}
    elements.update(vegetation_counts(None if guessed else c, settings, preset))
    elements.update(feature_counts(None if guessed else c, settings, preset))
    elements["trees"] += elements["scatter_trees"]
    total = (elements["buildings"] + elements["road_beams"] + elements["trees"] + elements["water"] + 1 +
             elements["shrubs"] + elements["hedge_segments"] + sum(
                 elements[k] for k in ("railway_beams", "platforms", "barrier_elements", "building_parts", "ground",
                                       "piers", "power_towers")))
    elements["total"] = total
    dem_s = (tiles["dgm1_missing"] * DGM1_TILE_S + tiles["ndom_missing"] * NDOM_TILE_S +
             tiles["copernicus_tiles"] * COPERNICUS_TILE_S)
    parts = {"overpass_s": OVERPASS_BASE_S + OVERPASS_PER_KM2_S * area_km2, "dem_s": dem_s,
             "convert_s": CONVERT_BASE_S + CONVERT_PER_KM2_S * area_km2, "build_s": total * BUILD_PER_ELEMENT_S}
    mid = sum(parts.values())
    low, high = mid * LOW_FACTOR, mid * HIGH_FACTOR
    notes = []
    if guessed:
        notes.append("OSM counts unavailable (Overpass slow/down): element count and time estimated from area only")
    if ov_on and not guessed and "overture_buildings" not in c:
        notes.append("Overture Maps buildings (gap fill) are only counted after the first fetch (rough: not conflated "
                     "against OSM); they add to the building count")
    if tiles["dgm1_missing"] or tiles["ndom_missing"]:
        notes.append(f"{tiles['dgm1_missing']} DGM1 + {tiles['ndom_missing']} nDOM 1 km tiles to download "
                     f"(~{int(DGM1_TILE_S)} s each, cached afterwards)")
    if preset:
        notes.append("super-light: roads ~half of counted ways assumed (footway/path/cycleway/steps/service dropped)")
    return {"area_km2": round(area_km2, 3), "seconds": {"low": round(low), "mid": round(mid), "high": round(high)},
            "text": format_range(low, high), "elements": elements,
            "counts_source": "area" if guessed else counts_source,
            "tiles": tiles, "breakdown_s": {k: round(v) for k, v in parts.items()}, "notes": notes}
