"""Map UI server: static ui/ + /api/*.  Run: python server.py  -> http://127.0.0.1:8765"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path
from urllib.parse import unquote

import requests
from fastapi import Body, FastAPI, Query
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from osmctx import limits, paths
from osmctx.overpass import MIRRORS as OVERPASS
from osmctx.overpass import race as overpass_race

ROOT = paths.RESOURCE_DIR   # read-only assets (ui/); dev = project root, frozen = bundle
UA = "osm-context-archicad/0.1"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
COUNT_TIMEOUT_S = 15.0
CACHE_DIR = paths.CACHE_DIR
OUT_DIR = paths.OUT_DIR

# ---------------------------------------------------------------- coordinate parsing
_NUM = r"[-+]?\d+(?:\.\d+)?"
_DEC_RE = re.compile(rf"^\s*({_NUM})\s*(?:[,;]\s*|\s+)({_NUM})\s*$")
_DMS_RE = re.compile(
    r"(?:([NSEW])\s*)?(\d+(?:\.\d+)?)\s*(?:°|º|˚|d|deg)\s*"
    r"(?:(\d+(?:\.\d+)?)\s*(?:'|′|’|m|min)\s*)?"
    r"(?:(\d+(?:\.\d+)?)\s*(?:\"|″|”|''|′′|s|sec)\s*)?"
    r"([NSEW])?",
    re.IGNORECASE,
)


def _orient(a: float, b: float):
    """(a, b) assumed lat, lon. If lat out of range but swap valid -> swap."""
    if abs(a) > 90 and abs(b) <= 90:
        a, b = b, a
    if abs(a) <= 90 and abs(b) <= 180:
        return a, b
    return None


def parse_coords(text: str):
    """Parse a coordinate string -> (lat, lon) or None.
    Accepts decimal pairs, DMS with N/S/E/W, Google Maps URLs. Swapped (lon, lat) auto-detected
    when |lat| > 90."""
    if not text or not isinstance(text, str):
        return None
    s = unquote(text.strip())
    if re.match(r"^https?://", s, re.I) or "google." in s.lower() or "goo.gl" in s.lower():
        for pat in (rf"@({_NUM}),({_NUM})", rf"!3d({_NUM})!4d({_NUM})",
                    rf"[?&](?:q|ll|query|center)=({_NUM})[, +]+({_NUM})"):
            m = re.search(pat, s)
            if m:
                return _orient(float(m.group(1)), float(m.group(2)))
        return None
    # DMS
    hits = []
    for m in _DMS_RE.finditer(s):
        pre, d, mi, se, post = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        hemi = (post or pre or "").upper()
        if not hemi:
            continue
        val = float(d) + (float(mi) / 60 if mi else 0) + (float(se) / 3600 if se else 0)
        if hemi in "SW":
            val = -val
        hits.append((hemi, val))
    if len(hits) == 2:
        lat = next((v for h, v in hits if h in "NS"), None)
        lon = next((v for h, v in hits if h in "EW"), None)
        if lat is not None and lon is not None and abs(lat) <= 90 and abs(lon) <= 180:
            return lat, lon
        return None
    # decimal with optional trailing hemisphere letters, e.g. "48.5193 N, 9.0565 E"
    m = re.match(rf"^\s*({_NUM})\s*°?\s*([NSEW])?\s*[,;\s]\s*({_NUM})\s*°?\s*([NSEW])?\s*$", s, re.I)
    if m and (m.group(2) or m.group(4)):
        a, ha, b, hb = float(m.group(1)), (m.group(2) or "").upper(), float(m.group(3)), (m.group(4) or "").upper()
        vals = {}
        for v, h in ((a, ha), (b, hb)):
            if h in ("S", "W"):
                v = -abs(v)
            vals["lat" if h in ("N", "S") else "lon"] = v
        if "lat" in vals and "lon" in vals and abs(vals["lat"]) <= 90 and abs(vals["lon"]) <= 180:
            return vals["lat"], vals["lon"]
    m = _DEC_RE.match(s)
    if m:
        return _orient(float(m.group(1)), float(m.group(2)))
    return None


# ---------------------------------------------------------------- boundary bbox
def boundary_bbox(b: dict):
    """Return (south, west, north, east) for a boundary dict."""
    t = b.get("type")
    if t == "circle":
        lon, lat = b["center"]
        r = float(b["radius_m"])
        dlat = r / 111320.0
        dlon = r / (111320.0 * max(math.cos(math.radians(lat)), 1e-6))
        return lat - dlat, lon - dlon, lat + dlat, lon + dlon
    pts = b["coordinates"]
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    s, w, n, e = min(lats), min(lons), max(lats), max(lons)
    if t == "polyline":
        r = float(b.get("buffer_m", 0))
        lat0 = (s + n) / 2
        dlat = r / 111320.0
        dlon = r / (111320.0 * max(math.cos(math.radians(lat0)), 1e-6))
        s, n, w, e = s - dlat, n + dlat, w - dlon, e + dlon
    return s, w, n, e


def count_query(bbox) -> str:
    s, w, n, e = bbox
    bb = f"({s:.6f},{w:.6f},{n:.6f},{e:.6f})"
    return (
        "[out:json][timeout:25];\n"
        f'nwr["building"]{bb}->.b; .b out count;\n'
        f'way["highway"]{bb}->.h; .h out count;\n'
        f'(node["natural"="tree"]{bb};way["natural"="tree_row"]{bb};)->.t; .t out count;\n'
        f'(way["natural"="water"]{bb};relation["natural"="water"]{bb};way["waterway"]{bb};)->.w; .w out count;\n'
    )


def parse_counts(data: dict) -> dict:
    counts = [int(el.get("tags", {}).get("total", 0)) for el in data.get("elements", []) if el.get("type") == "count"]
    counts += [0] * (4 - len(counts))
    return dict(zip(("buildings", "roads", "trees", "water"), counts[:4]))


# ---------------------------------------------------------------- counting (cache + racing mirrors)
def _in_bbox(el: dict, bb) -> bool:
    """Element (out geom: node lat/lon, way/relation 'bounds') intersects bbox (s, w, n, e)."""
    s, w, n, e = bb
    if "lat" in el and "lon" in el:
        return s <= el["lat"] <= n and w <= el["lon"] <= e
    b = el.get("bounds")
    if not b:
        return True
    return not (b["maxlat"] < s or b["minlat"] > n or b["maxlon"] < w or b["minlon"] > e)


def _bounds_m2(b: dict, bb=None) -> int:
    """Approximate m2 of an element's bounding box (clipped to bb=(s, w, n, e) when given)."""
    s, w, n, e = (b["minlat"], b["minlon"], b["maxlat"], b["maxlon"])
    if bb is not None:
        s, w, n, e = max(s, bb[0]), max(w, bb[1]), min(n, bb[2]), min(e, bb[3])
    if n <= s or e <= w:
        return 0
    lat = (s + n) / 2
    return int((n - s) * 111320.0 * (e - w) * 111320.0 * math.cos(math.radians(lat)))


def count_from_osm(osm: dict, bb=None) -> dict:
    """Same four categories as count_query, from a cached raw Overpass JSON (fetch.py cache).
    The cache covers boundary bbox + DEM margin, so pass bb=(s, w, n, e) to count only the boundary bbox."""
    from osmctx.vegetation import area_tree_type
    c = {"buildings": 0, "roads": 0, "trees": 0, "water": 0}
    for el in osm.get("elements", []):
        if bb is not None and not _in_bbox(el, bb):
            continue
        t = el.get("tags", {})
        typ = el.get("type")
        if "building" in t:
            c["buildings"] += 1
        if "highway" in t and typ == "way":
            c["roads"] += 1
        if t.get("natural") == "tree" or (t.get("natural") == "tree_row" and typ == "way"):
            c["trees"] += 1
        # vegetation extras (cache only; rough, used by the estimate): hedges, shrub nodes, area m2 from bounds
        if t.get("barrier") == "hedge" and typ == "way":
            c["hedges"] = c.get("hedges", 0) + 1
        if t.get("natural") == "shrub" and typ == "node":
            c["shrubs"] = c.get("shrubs", 0) + 1
        veg_key = ("forest_m2" if (t.get("natural") == "wood" or t.get("landuse") == "forest") else
                   "shrub_m2" if t.get("natural") in ("shrubbery", "scrub") else None)
        if not veg_key:
            area_typ = area_tree_type(t)
            veg_key = f"{area_typ}_m2" if area_typ else None
        if veg_key and typ in ("way", "relation") and el.get("bounds"):
            c[veg_key] = c.get(veg_key, 0) + _bounds_m2(el["bounds"], bb)
        if ("waterway" in t and typ == "way") or (t.get("natural") == "water" and typ in ("way", "relation")):
            c["water"] += 1
        # more features (rough, used by the estimate; keys optional)
        if t.get("railway") in ("rail", "light_rail", "tram", "subway", "narrow_gauge") and typ == "way":
            c["railways"] = c.get("railways", 0) + 1
        if (t.get("railway") == "platform" or t.get("public_transport") == "platform") and typ in ("way", "relation"):
            c["platforms"] = c.get("platforms", 0) + 1
        if t.get("barrier") in ("wall", "retaining_wall", "city_wall", "fence", "guard_rail") and typ == "way":
            c["barriers"] = c.get("barriers", 0) + 1
        if "building:part" in t and typ != "node":
            c["building_parts"] = c.get("building_parts", 0) + 1
        if (t.get("amenity") == "parking" or t.get("leisure") in ("pitch", "playground") or "area:highway" in t
                or t.get("place") == "square") and typ in ("way", "relation"):
            c["ground"] = c.get("ground", 0) + 1
        if t.get("man_made") in ("pier", "bridge") and typ != "node":
            c["piers"] = c.get("piers", 0) + 1
        if t.get("power") in ("tower", "pole") and typ == "node":
            c["power_towers"] = c.get("power_towers", 0) + 1
    return c


def count_overture(cache: Path, bb=None) -> int | None:
    """Overture buildings in a fetch cache that are NOT sourced from OSM (the gap fill), counted by first vertex inside
    bb=(s, w, n, e); None when the cache has no overture_buildings.geojson (not fetched / failed). Rough: the
    conflation against OSM (overlap rule) is not applied, so the build count can be lower."""
    p = Path(cache) / "overture_buildings.geojson"
    if not p.exists():
        return None
    try:
        from osmctx import overture as ovt
        n = 0
        for ft in ovt.load(p):
            if ovt.is_osm_sourced(ft.get("properties") or {}):
                continue
            g = ft.get("geometry") or {}
            c = g.get("coordinates") or []
            while c and isinstance(c[0], list):
                c = c[0]
            if bb is None or (c and bb[0] <= c[1] <= bb[2] and bb[1] <= c[0] <= bb[3]):
                n += 1
        return n
    except Exception:  # noqa: BLE001
        return None


def find_cache(boundary: dict, cache_dir: Path = CACHE_DIR, dem_source: str = "auto"):
    """Existing fetch cache dir for this boundary (any DEM source), or None."""
    try:
        from osmctx.fetch import cache_key
    except ImportError:
        return None
    try:
        d = Path(cache_dir) / cache_key(boundary, dem_source)
    except Exception:  # noqa: BLE001
        return None
    return d if (d / "osm.json").exists() and (d / "meta.json").exists() else None


def _post_overpass(url: str, q: str) -> dict:
    r = requests.post(url, data={"data": q}, headers={"User-Agent": UA}, timeout=(5, COUNT_TIMEOUT_S))
    r.raise_for_status()
    return r.json()


def race_overpass(q: str, urls=None, timeout: float = COUNT_TIMEOUT_S, post=None):
    """Send q to all mirrors in parallel (osmctx.overpass.race, shared with fetch); first valid answer wins
    (counts: no wait for a fresher mirror). Returns (counts, url, errors)."""
    def validate(data):
        if sum(1 for el in data.get("elements", []) if el.get("type") == "count") < 4:
            raise ValueError("no count elements (remark: %s)" % str(data.get("remark", ""))[:80])
    data, url, info = overpass_race(q, urls or OVERPASS, timeout, post=post or _post_overpass, validate=validate,
                                    fresh_wait=0.0)
    if data is None:
        return None, None, info["errors"]
    return parse_counts(data), url, info["errors"]


# ---------------------------------------------------------------- settings mapping / accuracy text
_NON_CONVERT_KEYS = frozenset(("dem_source", "dem_file", "tree_mode", "tree_lib", "override", "skip_ndom", "hillshade",
                               "contour_type"))   # fetch-only / builder-only / UI keys


def map_settings(ui: dict | None) -> dict:
    """UI settings -> osmctx.convert settings (floor_height_m->floor_height, tree_height_m->tree_height,
    ground_mode ''/'auto'->None). Fetch-only keys (dem_source, dem_file) and builder-only keys (tree_mode, tree_lib) are dropped."""
    out = {}
    for k, v in (ui or {}).items():
        if k in _NON_CONVERT_KEYS:
            continue
        if k == "floor_height_m":
            out["floor_height"] = v
        elif k == "tree_height_m":
            out["tree_height"] = v
        elif k == "ground_mode":
            out[k] = None if v in (None, "", "auto") else v
        else:
            out[k] = v
    return out


def dem_source_of(ui: dict | None) -> str:
    """UI terrain source -> osmctx.fetch dem_source: auto | copernicus | bw_dgm1 | file:<path>."""
    ui = ui or {}
    src = ui.get("dem_source") or "auto"
    if src == "file":
        f = (ui.get("dem_file") or "").strip()
        if not f:
            raise ValueError("terrain source 'Local file' needs a path")
        return "file:" + f
    return src


GLO30_ACC = {"kind": "DSM", "vertical_m": 5.0,
             "source": "Copernicus GLO-30 (copernicus-dem-30m.s3.amazonaws.com)"}
GLO30_ATTR = ("Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 "
              "provided under COPERNICUS by the European Union and ESA; all rights reserved")
GLO30_LICENCE = "Copernicus DEM licence (free use incl. commercial, attribution required)"
# buildings theme licence = ODbL (docs.overturemaps.org/attribution); per-source licences listed there
OVERTURE_CREDIT_GENERIC = ("Buildings: © OpenStreetMap contributors (ODbL); Overture Maps Foundation, buildings theme (ODbL), "
                           "where the fill is used incl. Google Open Buildings (CC BY 4.0), Global ML Building Footprints, "
                           "Microsoft (ODbL), Esri Community Maps contributors (CC BY 4.0)")


def disclaimer(acc: dict | None = None, attribution: str | None = None, licence: str | None = None,
               overture: dict | None = None) -> dict:
    """Accuracy disclaimer. acc = report.dem_accuracy (None before fetch -> GLO-30 defaults).
    overture = report.info.overture (None before convert -> generic Overture line, as the checkbox is on by default)."""
    a = dict(GLO30_ACC)
    a.update({k: v for k, v in (acc or {}).items() if v is not None})
    src = a.get("source") or GLO30_ACC["source"]
    vert = a.get("vertical_m")
    if a.get("kind") == "DSM" or not acc:
        v = vert if isinstance(vert, (int, float)) else 5.0
        terrain = (f"Terrain from {src} (~30 m grid, a surface model incl. buildings/trees): expect "
                   f"±{v:g} m vertical in open ground, ±{2 * v:g} m or worse under buildings/forest and "
                   f"on steep slopes; contours are interpolated.")
    elif isinstance(vert, (int, float)):
        terrain = (f"Terrain from {src} ({a.get('kind', 'terrain model')}): expect about ±{vert:g} m vertical "
                   f"in open ground; contours are interpolated.")
    else:
        terrain = f"Terrain from {src}: vertical accuracy unknown, check the data provider's metadata."
    text = ("Site context only — not a survey. " + terrain +
            " Map data © OpenStreetMap contributors (ODbL): positions typically ±1–5 m; building "
            "heights estimated from floor counts or defaults. Use a topographic survey for design levels.")
    attr = attribution or (GLO30_ATTR if not acc else None)
    lic = licence or (GLO30_LICENCE if not acc else None)
    credits = ["© OpenStreetMap contributors (ODbL)"]
    if overture is None:      # before convert: the fill is on by default, say which datasets it may use
        credits.append(OVERTURE_CREDIT_GENERIC)
    elif overture.get("applied") and overture.get("attribution"):
        credits.append(overture["attribution"])
    if attr:
        credits.append(attr + (f" — licence: {lic}" if lic else ""))
    return {"text": text, "credits": credits}


STALE_OSM_DAYS = 7.0
SLOPE_NOTE_MIN_PCT = 30.0   # show the slope note from this max slope (or any steep cells / mapped cliffs)


def accuracy_notes(site: dict) -> list[dict]:
    """Extra accuracy-panel lines from a converted site.json: [{kind, level 'info'|'warn', text}].
    Kinds: slope (report.dem_accuracy.slope), poi (peak DEM vs OSM ele), stale_osm (report.info.osm.osm_age_days),
    buildings (steep_footprint / in_water counts)."""
    rep = (site or {}).get("report") or {}
    counts, acc = rep.get("counts") or {}, rep.get("dem_accuracy") or {}
    out: list[dict] = []
    sl = acc.get("slope")
    if sl:
        cliffs = len((site or {}).get("cliffs") or [])
        steep = (sl.get("cells_steep_pct") or 0) > 0
        if steep or cliffs or (sl.get("max_pct") or 0) >= SLOPE_NOTE_MIN_PCT:
            txt = (f"Slope: max {sl.get('max_pct', 0):.0f} %, p95 {sl.get('p95_pct', 0):.0f} %; "
                   f"{sl.get('cells_steep_pct', 0):.0f} % of DEM cells steeper than {sl.get('cells_steeper_than_deg', 45):g} deg"
                   f"{f', {cliffs} mapped cliff line(s)' if cliffs else ''}. One cell of horizontal error is about "
                   f"{sl.get('vertical_error_one_cell_m_p95', 0):.0f} m vertical here (p95); cliffs and steep slopes are smoothed "
                   "by the ~30 m grid.")
            out.append({"kind": "slope", "level": "warn" if (steep or cliffs) else "info", "text": txt})
    for p in (site or {}).get("points_of_interest") or []:
        d = p.get("dem_minus_ele")
        if d is not None and abs(d) >= 5:
            out.append({"kind": "poi", "level": "warn", "text":
                        f"{p.get('kind', 'peak').capitalize()} {p.get('name') or p.get('osm_id')}: DEM {p.get('dem_z_abs', 0):.0f} m "
                        f"vs OSM ele {p.get('ele', 0):.0f} m ({d:+.0f} m). The DEM is not reshaped; no element is built for the "
                        "peak (data only)."})
    age = (((rep.get("info") or {}).get("osm") or {}).get("osm_age_days"))
    if isinstance(age, (int, float)) and age > STALE_OSM_DAYS:
        out.append({"kind": "stale_osm", "level": "warn", "text":
                    f"OSM data is {age:.0f} days old (Overpass mirror out of date): recent mapping edits are missing; try fetching again later."})
    nst, nwa = counts.get("buildings_steep_footprint", 0) or 0, counts.get("buildings_in_water", 0) or 0
    if nst or nwa:
        bits = []
        if nst:
            bits.append(f"{nst} on steep ground (>3 m terrain spread under the footprint; base = lowest point, tag osmctx:steep_footprint)")
        if nwa:
            bits.append(f"{nwa} mostly in water (stilt houses/piers; tag osmctx:in_water)")
        out.append({"kind": "buildings", "level": "warn", "text": "Flagged buildings: " + "; ".join(bits) + "."})
    return out


# ---------------------------------------------------------------- site.json -> GeoJSON
def site_to_geojson(site: dict) -> dict:
    """Features in local UTM-offset metres -> WGS84 GeoJSON (origin easting/northing + x/y)."""
    from pyproj import Transformer
    o = site["origin"]
    tr = Transformer.from_crs(int(o["epsg"]), 4326, always_xy=True)
    e0, n0 = float(o["easting"]), float(o["northing"])

    def ll(p):
        lon, lat = tr.transform(e0 + p[0], n0 + p[1])
        return [round(lon, 7), round(lat, 7)]

    def ring(r):
        pts = [ll(p) for p in r]
        if pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        return pts

    feats = []

    def add(kind, geom, **props):
        feats.append({"type": "Feature", "geometry": geom, "properties": {"kind": kind, **props}})

    if site.get("boundary"):
        add("boundary", {"type": "Polygon", "coordinates": [ring(site["boundary"])]})
    for b in site.get("buildings", []):
        add("building", {"type": "Polygon", "coordinates": [ring(b["footprint"])] + [ring(h) for h in b.get("holes", [])]},
            id=b.get("osm_id"), height=b.get("height"), source=b.get("height_source"))
    for w in site.get("water", []):
        add("water", {"type": "Polygon", "coordinates": [ring(w["polygon"])] + [ring(h) for h in w.get("holes", [])]},
            id=w.get("osm_id"), water_kind=w.get("kind"))
    for r in site.get("roads", []):
        add("road", {"type": "LineString", "coordinates": [ll(p) for p in r["points"]]},
            id=r.get("osm_id"), road_class=r.get("class"), width=r.get("width"),
            bridge=r.get("bridge"), tunnel=r.get("tunnel"))
    for t in site.get("trees", []):
        add("tree", {"type": "Point", "coordinates": ll([t["x"], t["y"]])}, id=t.get("osm_id"), height=t.get("height"))
    # more features (M6)
    for r in site.get("railways", []):
        add("railway", {"type": "LineString", "coordinates": [ll(p) for p in r["points"]]}, id=r.get("osm_id"),
            railway_class=r.get("class"), width=r.get("width"), bridge=r.get("bridge"), tunnel=r.get("tunnel"))
    for b in site.get("barriers", []):
        add("barrier", {"type": "LineString", "coordinates": [ll(p) for p in b["points"]]}, id=b.get("osm_id"),
            barrier=b.get("kind"), height=b.get("height"))
    for p in site.get("building_parts", []):
        add("building", {"type": "Polygon", "coordinates": [ring(p["footprint"])] + [ring(h) for h in p.get("holes", [])]},
            id=p.get("osm_id"), height=p.get("height"), source=p.get("height_source"), part=True)
    for key, kind in (("railway_platforms", "platform"), ("ground_surfaces", "ground"), ("piers", "pier")):
        for g in site.get(key, []):
            add(kind, {"type": "Polygon", "coordinates": [ring(g["polygon"])] + [ring(h) for h in g.get("holes", [])]},
                id=g.get("osm_id"), area_kind=g.get("kind"))
    return {"type": "FeatureCollection", "features": feats}


def _read_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


_osm_memo: tuple | None = None       # (path, mtime_ns, size, parsed): one big osm.json, re-parsed only when it changes
_geojson_memo: tuple | None = None   # same idea for the site.json -> GeoJSON result


def _stamp(p: Path):
    st = p.stat()
    return str(p), st.st_mtime_ns, st.st_size


def _load_osm(p: Path) -> dict:
    global _osm_memo
    key = _stamp(p)
    memo = _osm_memo
    if memo is not None and memo[0] == key:
        return memo[1]
    data = _read_json(p)
    _osm_memo = (key, data)
    return data


def _safe_cache(path: str) -> Path:
    p = Path(path).resolve()
    if CACHE_DIR.resolve() not in p.parents:
        raise ValueError("cache path outside cache/")
    return p


# ---------------------------------------------------------------- app
app = FastAPI(title="OSM Context Archicad")

_geo_lock = threading.Lock()
_geo_last = 0.0
_geo_cache: dict = {}


@app.get("/api/parse")
def api_parse(q: str = Query("")):
    r = parse_coords(q)
    if r is None:
        return {"ok": False}
    return {"ok": True, "lat": r[0], "lon": r[1]}


@app.get("/api/geocode")
def api_geocode(q: str = Query(..., min_length=1)):
    global _geo_last
    key = q.strip().lower()
    if key in _geo_cache:
        return _geo_cache[key]
    with _geo_lock:
        wait = 1.0 - (time.monotonic() - _geo_last)
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(NOMINATIM, params={"q": q, "format": "jsonv2", "limit": 5},
                             headers={"User-Agent": UA}, timeout=20)
            _geo_last = time.monotonic()
            r.raise_for_status()
            raw = r.json()
        except Exception as ex:  # noqa: BLE001
            _geo_last = time.monotonic()
            return JSONResponse({"error": f"geocoder failed: {ex}"}, status_code=502)
    res = []
    for it in raw:
        bb = it.get("boundingbox")
        res.append({"name": it.get("display_name"), "lat": float(it["lat"]), "lon": float(it["lon"]),
                    "type": it.get("type"),
                    "bbox": [float(x) for x in bb] if bb else None})  # [south, north, west, east]
    _geo_cache[key] = res
    return res


def get_counts(b: dict, settings: dict | None = None) -> tuple[dict | None, dict]:
    """OSM element counts for boundary b: fetch cache first, else Overpass race. -> (counts|None, info)."""
    t0 = time.monotonic()
    bb = boundary_bbox(b)
    try:
        cached = find_cache(b, dem_source=dem_source_of(settings))
    except ValueError:
        cached = None
    if cached is not None:
        try:
            out = count_from_osm(_load_osm(cached / "osm.json"), bb)
            ovn = count_overture(cached, bb)
            if ovn is not None:
                out["overture_buildings"] = ovn
            return out, {"source": "cache", "cache": str(cached), "seconds": round(time.monotonic() - t0, 3)}
        except Exception:  # noqa: BLE001
            pass  # fall through to Overpass
    out, url, errs = race_overpass(count_query(bb))
    if out is None:
        return None, {"error": "overpass failed: " + " | ".join(errs)}
    return out, {"source": "overpass", "overpass_url": url, "seconds": round(time.monotonic() - t0, 2)}


@app.post("/api/count")
def api_count(body: dict = Body(...)):
    b = body.get("boundary", body)
    try:
        counts, info = get_counts(b, body.get("settings"))
    except Exception as ex:  # noqa: BLE001
        return JSONResponse({"error": f"bad boundary: {ex}"}, status_code=400)
    if counts is None:
        return JSONResponse({"error": info["error"]}, status_code=502)
    return {**counts, **info}


@app.get("/api/limits")
def api_limits():
    return limits.limits_payload()


def _blocked_response(g: dict) -> JSONResponse:
    return JSONResponse({"error": g["message"], "tier": g["tier"], "area_km2": g["area_km2"]}, status_code=400)


def _guard(boundary, settings, override):
    """(guard dict | None, error response | None). Unreadable boundary -> no guard (the fetcher will fail itself)."""
    try:
        g = limits.guard(boundary, settings, bool(override))
    except Exception:  # noqa: BLE001
        return None, None
    return g, (_blocked_response(g) if g["blocked"] else None)


@app.post("/api/estimate")
def api_estimate(body: dict = Body(...)):
    b = body.get("boundary")
    if not b:
        return JSONResponse({"error": "boundary missing"}, status_code=400)
    try:
        g = limits.guard(b, body.get("settings"), bool(body.get("override")))
        src = dem_source_of(body.get("settings"))
    except Exception as ex:  # noqa: BLE001
        return JSONResponse({"error": f"bad boundary/settings: {ex}"}, status_code=400)
    counts, info = (None, {}) if g["blocked"] else _safe_counts(b, body.get("settings"))
    try:
        tiles = limits.dem_tiles(b, src, CACHE_DIR)
    except Exception:  # noqa: BLE001
        tiles = {"source": "unknown", "dgm1_total": 0, "dgm1_missing": 0, "ndom_total": 0, "ndom_missing": 0,
                 "copernicus_tiles": 0}
    est = limits.estimate(g["area_km2"], counts, info.get("source", "area"), tiles, preset=bool(g["preset"]),
                          settings=g["settings"])
    return {**est, "tier": g["tier"], "preset": g["preset"], "override": g["override"], "blocked": g["blocked"],
            "message": g["message"], "counts": counts, "counts_info": info}


def _safe_counts(b, settings):
    try:
        return get_counts(b, settings)
    except Exception as ex:  # noqa: BLE001
        return None, {"error": str(ex)}


_jobs: dict = {}
_jobs_lock = threading.Lock()
_last_cache: Path | None = None


def _dem_wgs84_bounds(cache: Path, meta: dict | None = None) -> list[float] | None:
    """WGS84 [s, w, n, e] of cache/dem.tif, else of the dem_bbox/bbox in meta.json (pass meta when already read)."""
    dem_tif = cache / "dem.tif"
    if dem_tif.exists():
        try:
            import rasterio
            from rasterio.warp import transform_bounds
            with rasterio.open(dem_tif) as ds:
                w, s, e, n = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
                if ds.crs is not None and ds.crs.to_epsg() != 4326:
                    w, s, e, n = transform_bounds(ds.crs, "EPSG:4326", w, s, e, n, densify_pts=21)
                return [float(s), float(w), float(n), float(e)]
        except Exception:  # noqa: BLE001
            pass
    try:
        if meta is None:
            meta = _read_json(cache / "meta.json")
        db = meta.get("dem_bbox") or meta.get("bbox")
        if db:
            return [float(db["south"]), float(db["west"]), float(db["north"]), float(db["east"])]
    except Exception:  # noqa: BLE001
        pass
    return None


def _run_fetch_job(jid: str, boundary: dict, dem_source: str, skip_ndom: bool = False, overture: bool = True):
    from osmctx.fetch import fetch
    job = _jobs[jid]
    t0 = time.monotonic()
    stop = threading.Event()

    def ticker():  # fetch() has no progress hook: report elapsed time + what is probably going on
        while not stop.wait(1.0):
            el = time.monotonic() - t0
            hint = " (first DGM1 download can take ~5 min)" if dem_source in ("auto", "bw_dgm1") else ""
            job["progress"] = f"fetching OSM + DEM ({dem_source}) {el:.0f} s{hint}"
    threading.Thread(target=ticker, daemon=True).start()
    try:
        path = fetch(boundary, cache_dir=str(CACHE_DIR), dem_source=dem_source,
                     **({"ndom": False} if skip_ndom else {}), **({} if overture else {"overture": False}))
        job.update(state="done", cache=str(path), progress=f"fetched in {time.monotonic() - t0:.1f} s")
    except Exception as ex:  # noqa: BLE001
        job.update(state="error", error=f"{type(ex).__name__}: {ex}", progress="failed")
    finally:
        stop.set()


@app.post("/api/fetch")
def api_fetch(body: dict = Body(...)):
    """Starts a background fetch job -> {job}. Poll GET /api/job/<id> (state running|done|error, progress, cache)."""
    try:
        import osmctx.fetch  # noqa: F401  lazy: written by another worker
    except ImportError:
        return JSONResponse({"error": "fetcher not ready"}, status_code=501)
    try:
        src = dem_source_of(body.get("settings"))
    except ValueError as ex:
        return JSONResponse({"error": str(ex)}, status_code=400)
    g, err = _guard(body.get("boundary"), body.get("settings"), body.get("override"))
    if err is not None:
        return err
    jid = "%08x" % (int(time.time() * 1000) & 0xFFFFFFFF)
    with _jobs_lock:
        _jobs[jid] = {"state": "running", "progress": "starting", "started": time.time()}
    skip_ndom = bool(g and (g.get("settings") or {}).get("skip_ndom"))  # super-light preset: no nDOM download
    overture = bool((((g and g.get("settings")) or body.get("settings") or {}).get("include") or {}).get("overture_buildings", True))
    threading.Thread(target=_run_fetch_job, args=(jid, body["boundary"], src, skip_ndom, overture), daemon=True).start()
    return {"job": jid, "tier": g and g["tier"], "preset": g and g["preset"], "skip_ndom": skip_ndom,
            "overture": overture}


@app.get("/api/job/{jid}")
def api_job(jid: str):
    j = _jobs.get(jid)
    if j is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return dict(j)


@app.post("/api/convert")
def api_convert(body: dict = Body(default={})):
    global _last_cache
    try:
        from osmctx.convert import convert  # lazy: written by another worker
    except ImportError:
        return JSONResponse({"error": "converter not ready"}, status_code=501)
    try:
        cache = _safe_cache(body["cache"])
    except (KeyError, ValueError) as ex:
        return JSONResponse({"error": f"bad cache: {ex}"}, status_code=400)
    boundary = body.get("boundary")
    meta = None
    if not boundary:
        try:
            meta = _read_json(cache / "meta.json")
            boundary = meta.get("boundary")
        except Exception:  # noqa: BLE001
            boundary = None
    g, err = _guard(boundary, body.get("settings"), body.get("override"))
    if err is not None:
        return err
    ui_settings = g["settings"] if g else body.get("settings")
    try:
        site = convert(cache, map_settings(ui_settings), out_dir=str(OUT_DIR))
    except Exception as ex:  # noqa: BLE001
        return JSONResponse({"error": f"{type(ex).__name__}: {ex}"}, status_code=500)
    _last_cache = cache
    rep = site.get("report", {})
    dem = (rep.get("info") or {}).get("dem") or {}
    try:   # auto contour interval for the UI (Terrain panel)
        from osmctx import contours as _ct
        con = _ct.preview(site, (ui_settings or {}).get("contour_interval") or None)
    except Exception:  # noqa: BLE001
        con = None
    return {"report": rep, "origin": site.get("origin"), "contours": con,
            "cache": str(cache), "dem_bounds": _dem_wgs84_bounds(cache, meta),
            "tier": g and g["tier"], "preset": g and g["preset"], "area_km2": g and g["area_km2"],
            "accuracy_notes": accuracy_notes(site),
            "disclaimer": disclaimer(rep.get("dem_accuracy"), dem.get("attribution"), dem.get("licence"),
                                     (rep.get("info") or {}).get("overture") or {"applied": False})}


@app.get("/api/disclaimer")
def api_disclaimer():
    return disclaimer()


@app.get("/api/site_geojson")
def api_site_geojson():
    global _geojson_memo
    p = OUT_DIR / "site.json"
    if not p.exists():
        return JSONResponse({"error": "no out/site.json yet; run Convert"}, status_code=404)
    try:
        key = _stamp(p)
        memo = _geojson_memo
        if memo is not None and memo[0] == key:
            return memo[1]
        gj = site_to_geojson(_read_json(p))
        _geojson_memo = (key, gj)
        return gj
    except Exception as ex:  # noqa: BLE001
        return JSONResponse({"error": f"{type(ex).__name__}: {ex}"}, status_code=500)


@app.get("/api/hillshade.png")
def api_hillshade(cache: str | None = Query(default=None)):
    from osmctx import dem_sources as ds
    dem_path = None
    if cache:
        try:
            dem_path = _safe_cache(cache) / "dem.tif"
        except ValueError as ex:
            return JSONResponse({"error": f"bad cache: {ex}"}, status_code=400)
    elif _last_cache is not None and (_last_cache / "dem.tif").exists():
        dem_path = _last_cache / "dem.tif"
    else:
        tifs = sorted(CACHE_DIR.glob("*/dem.tif"), key=lambda p: p.stat().st_mtime, reverse=True)
        if tifs:
            dem_path = tifs[0]
    if dem_path is None or not dem_path.exists():
        return JSONResponse({"error": "no cached dem.tif yet; run Fetch + Convert"}, status_code=404)
    try:
        return Response(content=ds.hillshade_png(dem_path), media_type="image/png")
    except Exception as ex:  # noqa: BLE001
        return JSONResponse({"error": f"{type(ex).__name__}: {ex}"}, status_code=500)


@app.get("/api/sun_preview")
def api_sun_preview(lat: float, lon: float, date: str, time: str = "12:00"):
    from osmctx import sun
    try:
        return sun.preview(lat, lon, date, time)
    except Exception as ex:  # noqa: BLE001
        return JSONResponse({"error": f"{type(ex).__name__}: {ex}"}, status_code=400)


@app.get("/api/layers")
def api_layers():
    """READ-only: layers of the open Archicad project (for the Import panel's target-layer select)."""
    from osmctx.archicad_http import ArchicadError, Client
    from osmctx import build_ac
    try:
        with Client(retries=0, timeout=15.0) as c:
            layers = build_ac.list_layers(c)
    except ArchicadError as ex:
        return JSONResponse({"error": f"Archicad not reachable or no project open: {ex}"}, status_code=503)
    except Exception as ex:  # noqa: BLE001
        return JSONResponse({"error": f"{type(ex).__name__}: {ex}"}, status_code=500)
    return {"layers": layers, "default": build_ac.default_single_layer(layers)}


@app.post("/api/import")
def api_import(body: dict = Body(default={})):
    """WRITES to the live Archicad model (builder + sun/place settings)."""
    try:
        from osmctx.build_ac import build  # lazy: written by another worker
    except ImportError:
        return JSONResponse({"error": "builder not ready"}, status_code=501)
    site_path = OUT_DIR / "site.json"
    if not site_path.exists():
        return JSONResponse({"error": "no out/site.json; run Convert first"}, status_code=400)
    site = None

    def _site():
        nonlocal site
        if site is None:
            site = _read_json(site_path)
        return site

    boundary = body.get("boundary")
    if not boundary:  # fall back to the converted site's own (local-metre) boundary ring
        try:
            ring = _site().get("boundary") or []
            a = abs(sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(ring, ring[1:] + ring[:1]))) / 2
            g, err = limits.resolve(a / 1e6, body.get("settings"), bool(body.get("override"))), None
            if g["blocked"]:
                err = _blocked_response(g)
        except Exception:  # noqa: BLE001
            g, err = None, None
    else:
        g, err = _guard(boundary, body.get("settings"), body.get("override"))
    if err is not None:
        return err
    tree_mode = str(body.get("tree_mode") or "full3d")
    if g and g["preset"]:
        tree_mode = limits.SUPER_LIGHT["tree_mode"]
    result: dict = {}
    if g:
        result["tier"], result["preset"], result["area_km2"] = g["tier"], g["preset"], g["area_km2"]
    st = (g["settings"] if g else body.get("settings")) or {}   # effective settings (super-light preset applied)
    # validate BEFORE the sun write: a bad request must not touch the live model
    mode = body.get("layer_mode") or "single"
    tp = body.get("terrain_points") or "points"
    ctype = str(body.get("contour_type") or st.get("contour_type") or "mesh")
    for name, v, allowed in (("layer_mode", mode, ("single", "per_type")), ("terrain_points", tp, ("points", "rows")),
                             ("contour_type", ctype, ("mesh", "stepped"))):
        if v not in allowed:
            return JSONResponse({"error": f"{name} must be {'|'.join(allowed)}, got {v!r}", **result}, status_code=400)
    # sun first: the builder saves the project at its end, so the saved file already has the sun settings
    sun_cfg = body.get("sun")
    if sun_cfg:
        try:
            from osmctx import sun
            sdata = _site()
            o = sdata["origin"]
            conv = (((sdata.get("report") or {}).get("info") or {}).get("crs") or {}).get("meridian_convergence_deg", 0.0)
            payload = sun.place_settings(o["lat"], o["lon"], sun_cfg["date"], sun_cfg.get("time", "12:00"),
                                         o.get("elevation", 0.0), conv)
            result["sun"] = {"payload": payload["placeInfo"], "response": sun.apply(payload)}
        except Exception as ex:  # noqa: BLE001
            result["sun_error"] = f"{type(ex).__name__}: {ex}"
    try:
        result["build"] = build(str(site_path), tree_mode=tree_mode, land_cover=not body.get("no_land_cover"),
                                layer_mode=mode, target_layer=body.get("target_layer") or None,
                                terrain_points=tp,
                                contours=st.get("contours") is not False,
                                contour_interval=st.get("contour_interval") or None,
                                contour_scale=float(st.get("contour_interval_scale") or 1.0),
                                fence_tool=body.get("fence_tool") or "railing", site_data=_site(),
                                contour_type=ctype)
    except Exception as ex:  # noqa: BLE001
        return JSONResponse({"error": f"{type(ex).__name__}: {ex}", **result}, status_code=500)
    return result


# ---------------------------------------------------------------- app status (first-run checks)
TAPIR_URL = "https://github.com/ENZYME-APD/tapir-archicad-automation/releases"
QUIT_HOOK = None   # set by app_main: callable that stops the server


def app_version() -> str:
    try:
        return (paths.RESOURCE_DIR / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return "dev"


def archicad_status() -> dict:
    """Is Archicad reachable (ports 19723-19743) and is Tapir installed (GetAddOnVersion)? READ-only. Re-points the
    default Archicad client at the instance that has Tapir."""
    from osmctx import archicad_http
    d = archicad_http.discover()
    sel = d["selected"]
    if sel:
        archicad_http.use(sel["url"])
    if not d["archicad"]:
        msg = "Archicad not found on ports 19723-19743. Start Archicad 30 and open a project, then press Re-check."
        level = "error"
    elif not (d["tapir"] or {}).get("present"):
        msg = ("Archicad found (port %s) but the Tapir add-on is missing - install Tapir from its GitHub releases (%s), "
               "restart Archicad, then press Re-check." % (sel["port"], TAPIR_URL))
        level = "error"
    else:
        msg = "Archicad %s on port %s, Tapir %s." % (sel.get("version"), sel["port"], d["tapir"].get("version"))
        level = "ok"
    return {"level": level, "message": msg, "tapir_url": TAPIR_URL, **d}


@app.get("/api/app/status")
def api_app_status():
    return {"version": app_version(), "data_dir": str(paths.DATA_DIR), "frozen": paths.FROZEN,
            "archicad": archicad_status()}


@app.post("/api/app/quit")
def api_app_quit():
    if QUIT_HOOK is None:
        return JSONResponse({"error": "not running under the launcher"}, status_code=400)
    threading.Thread(target=QUIT_HOOK, daemon=True).start()
    return {"ok": True}


app.mount("/", StaticFiles(directory=str(paths.UI_DIR), html=True), name="ui")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
