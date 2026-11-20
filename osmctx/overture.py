"""Overture Maps buildings as a second source where OSM is thin.

Fetch (``fetch_buildings``): latest Overture release (STAC catalog), buildings theme GeoParquet on public S3
(no key), read with DuckDB (httpfs + spatial). The 512 parquet files of a release are sorted by bbox.xmin
(longitude strips, not globally) and every file is split in 128-256 row groups with min/max statistics. The STAC
collection of the release lists every file's bbox (one 110 KB request, cached in cache/_overture/), which picks the
few files that can hold the bbox; the bbox filter on the parquet bbox struct then downloads only the matching row
groups (a few MB). Reading all 512 footers instead costs ~340 MB / 150 s.
Never raises into the OSM path: ``ensure(d)`` records the failure in meta.json and returns.

Conflation (``conflate``, pure shapely, offline): keep every OSM building; keep an Overture building only if it is
not itself sourced from OpenStreetMap and overlaps OSM buildings by < ``max_overlap`` of its own area.
Height chain: Overture ``height`` -> ``num_floors`` x floor_height -> default.

Licences (docs.overturemaps.org/attribution, checked 2026-11-17): buildings theme = ODbL; OSM = ODbL;
Microsoft Global ML Building Footprints = ODbL; Google Open Buildings = CC BY 4.0; Esri Community Maps = CC BY 4.0;
East-Asia footprints (Qian Shi et al.) = CC BY 4.0; BTN 2024 (IGN Spain) = CC BY 4.0; USGS 3DEP = public domain.
(Not CDLA-Permissive: that is the Places / Addresses licence.)
"""
from __future__ import annotations

import contextlib
import json
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

from .overpass import UA

STAC_CATALOG = "https://stac.overturemaps.org/catalog.json"
S3_BASE = "s3://overturemaps-us-west-2/release/{release}/theme=buildings/type=building/*"
TIME_BOX_S = 60.0
OVERLAP_MAX = 0.20             # Overture building dropped when >= this share of its area lies on OSM buildings
FILE = "overture_buildings.geojson"

# source dataset -> (credit text, licence). Unknown datasets use the licence of the record, else "see Overture".
DATASET_CREDITS = {
    "OpenStreetMap": ("© OpenStreetMap contributors", "ODbL-1.0"),
    "Microsoft ML Buildings": ("Global ML Building Footprints, Microsoft", "ODbL-1.0"),
    "Google Open Buildings": ("Google Open Buildings", "CC BY 4.0"),
    "Esri Community Maps": ("Esri Community Maps contributors", "CC BY 4.0"),
}
ML_PATTERNS = ("ml building", "open buildings", "qian shi", "global ml", "microsoft")
THEME_LICENCE = "ODbL-1.0"


class OvertureError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------ release / DuckDB
def latest_release(timeout: float = 15.0) -> str:
    r = requests.get(STAC_CATALOG, timeout=timeout, headers={"User-Agent": UA})
    r.raise_for_status()
    j = r.json()
    rel = j.get("latest")
    if not rel:
        raise OvertureError("STAC catalog has no 'latest' release")
    return str(rel)


def _connect(spatial: bool = False, threads: int = 8):
    import duckdb
    c = duckdb.connect()
    from . import paths
    if paths.DUCKDB_EXT_DIR.is_dir():   # frozen build ships httpfs/spatial pre-installed (works without extension download)
        c.execute(f"SET extension_directory='{paths.DUCKDB_EXT_DIR.as_posix()}'")
    c.execute("INSTALL httpfs; LOAD httpfs;")
    if spatial:
        c.execute("INSTALL spatial; LOAD spatial;")
    c.execute(f"SET s3_region='us-west-2'; SET threads={int(threads)}; SET enable_http_metadata_cache=true;")
    return c


def _list_files(release: str) -> list[str]:
    c = _connect()
    try:
        rows = c.execute("select file from glob(?) order by file", [S3_BASE.format(release=release)]).fetchall()
    finally:
        c.close()
    return [r[0] for r in rows]


def stac_bboxes(release: str, timeout: float = 20.0) -> list[list[float]]:
    """Per-file bbox [xmin, ymin, xmax, ymax] of the buildings parquet files, from the release's STAC collection
    (one 110 KB request; entry i = part-<i>). Replaces reading the 512 parquet footers (~0.7 MB each)."""
    url = f"https://stac.overturemaps.org/{release}/buildings/building/collection.json"
    r = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
    r.raise_for_status()
    bb = r.json()["extent"]["spatial"]["bbox"]
    return [list(map(float, x)) for x in bb[1:]]   # [0] = union of all files


def select_files(files: list[str], boxes: list[list[float]], bb: tuple) -> list[str]:
    """Parquet files whose overall bbox touches bb = (south, west, north, east)."""
    s, w, n, e = bb
    by_part = {}
    for f in files:
        m = re.search(r"part-(\d+)-", f)
        if m:
            by_part[int(m.group(1))] = f
    if len(boxes) != len(by_part):
        raise OvertureError(f"STAC lists {len(boxes)} files, bucket has {len(by_part)}")
    return [by_part[i] for i, (x0, y0, x1, y1) in enumerate(boxes) if x0 <= e and x1 >= w and y0 <= n and y1 >= s]


# ------------------------------------------------------------------------------------------ fetch
def _cached_json(path: Path, producer):
    """JSON of `path` if present, else producer() written to it."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = producer()
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def _row_group_bytes(c, files: list[str], bb: tuple) -> int | None:
    """Compressed bytes of the columns read, summed over the row groups that pass the bbox filter (estimate of the
    download; footers are not counted)."""
    s, w, n, e = bb
    try:
        q = """
        select coalesce(sum(sz), 0) from (
          select file_name, row_group_id,
            min(case when path_in_schema like '%xmin' then stats_min_value::double end) a,
            max(case when path_in_schema like '%xmax' then stats_max_value::double end) b,
            min(case when path_in_schema like '%ymin' then stats_min_value::double end) cc,
            max(case when path_in_schema like '%ymax' then stats_max_value::double end) d,
            sum(case when regexp_matches(path_in_schema, '^(id|height|num_floors|class|subtype|geometry|bbox|sources)')
                     then total_compressed_size else 0 end) sz
          from parquet_metadata(?) group by 1, 2)
        where a <= ? and b >= ? and cc <= ? and d >= ?"""
        return int(c.execute(q, [files, e, w, n, s]).fetchone()[0])
    except Exception:  # noqa: BLE001
        return None


def fetch_buildings(bb: tuple, cache_root: Path, out_path: Path, timeout: float = TIME_BOX_S) -> dict:
    """bb = (south, west, north, east). Writes `out_path` (GeoJSON FeatureCollection, WGS84) and returns the meta block.
    Raises OvertureError / TimeoutError (callers: use ``ensure``)."""
    t0 = time.monotonic()
    deadline = t0 + timeout
    s, w, n, e = bb
    release = latest_release()
    root = Path(cache_root) / "_overture"
    root.mkdir(parents=True, exist_ok=True)
    flist = root / f"{release}_files.json"
    fbox = root / f"{release}_stac_bboxes.json"
    if flist.exists() and fbox.exists():
        files, boxes = _cached_json(flist, None), _cached_json(fbox, None)
    else:   # first run per release: two independent network calls
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_files = pool.submit(_cached_json, flist, lambda: _list_files(release))
            f_boxes = pool.submit(_cached_json, fbox, lambda: stac_bboxes(release))
            files, boxes = f_files.result(), f_boxes.result()
    sel = select_files(files, boxes, bb)
    if not sel:
        raise OvertureError("no Overture file matches the bbox")
    c = _connect(spatial=True)   # I/O bound (S3 range reads): more threads than cores still pays
    holder: dict = {}
    rows_ready = threading.Event()

    def run():
        try:
            holder["rows"] = c.execute(
                "select id, height, num_floors, class, subtype, sources, ST_AsGeoJSON(geometry) from read_parquet(?) "
                "where bbox.xmin <= ? and bbox.xmax >= ? and bbox.ymin <= ? and bbox.ymax >= ?",
                [sel, e, w, n, s]).fetchall()
        except Exception as ex:  # noqa: BLE001
            holder["err"] = ex
        finally:
            rows_ready.set()
        if "rows" in holder and time.monotonic() < deadline:
            holder["est"] = _row_group_bytes(c, sel, bb)   # informational; bounded by the same time box below

    th = threading.Thread(target=run, daemon=True)
    th.start()
    if not rows_ready.wait(max(1.0, deadline - time.monotonic())):
        with contextlib.suppress(Exception):
            c.interrupt()
        raise TimeoutError(f"Overture query exceeded the {timeout:g} s time box")
    if "err" in holder:
        th.join()
        c.close()
        raise OvertureError(f"{type(holder['err']).__name__}: {holder['err']}")
    rows = holder["rows"]
    th.join(max(0.0, deadline - time.monotonic()))
    if th.is_alive():   # estimate still running past the time box: drop it
        with contextlib.suppress(Exception):
            c.interrupt()
        th.join(5.0)
    est = holder.get("est")
    if not th.is_alive():
        c.close()
    feats, ds = [], Counter()
    for fid, height, nfl, cls, sub, sources, geom in rows:
        srcs = [{"dataset": x.get("dataset"), "license": x.get("license"), "record_id": x.get("record_id"),
                 "confidence": x.get("confidence")} for x in (sources or [])]
        for x in srcs:
            ds[x["dataset"] or "?"] += 1
        feats.append({"type": "Feature", "geometry": json.loads(geom),
                      "properties": {"id": fid, "height": height, "num_floors": nfl, "class": cls, "subtype": sub,
                                     "sources": srcs}})
    fc = {"type": "FeatureCollection", "overture_release": release, "features": feats}
    out_path.write_text(json.dumps(fc), encoding="utf-8")
    prim = Counter((f["properties"]["sources"][0]["dataset"] if f["properties"]["sources"] else "?") for f in feats)
    return {"status": "ok", "release": release, "count": len(feats), "datasets": dict(prim),
            "files_read": [Path(f).name for f in sel], "files_total": len(files),
            "download_est_bytes": est, "elapsed_s": round(time.monotonic() - t0, 1),
            "fetched_at": datetime.now(timezone.utc).isoformat(), **attribution_block(prim)}


def meta_bbox(meta: dict) -> tuple:
    """(south, west, north, east) Overture is fetched for = the OSM fetch bbox of a cache."""
    bb = meta.get("osm_bbox") or meta["bbox"]
    return bb["south"], bb["west"], bb["north"], bb["east"]


def needs_fetch(d: Path, meta: dict, force: bool = False) -> bool:
    ov = meta.get("overture") or {}
    return force or not (ov.get("status") == "ok" and (Path(d) / FILE).exists())


def try_fetch(bb: tuple, cache_root: Path, out_path: Path, timeout: float = TIME_BOX_S) -> dict:
    """fetch_buildings that never raises: returns the meta block, ``{"status": "failed", ...}`` on any error.
    Touches no meta.json, so it can run concurrently with the OSM/DEM fetch (apply the block with ``apply``)."""
    try:
        return fetch_buildings(bb, cache_root, out_path, timeout)
    except Exception as ex:  # noqa: BLE001  never block the OSM path
        return {"status": "failed", "error": f"{type(ex).__name__}: {ex}",
                "fetched_at": datetime.now(timezone.utc).isoformat()}


def apply(meta: dict, ov: dict) -> None:
    """Record a ``try_fetch`` result in a meta dict: ``overture`` block + (on failure) a warning."""
    warns = meta.setdefault("warnings", [])
    warns[:] = [x for x in warns if not x.startswith("Overture")]
    if ov.get("status") == "failed":
        warns.append(f"Overture buildings not fetched ({ov['error'][:160]}): OSM buildings only")
    meta["overture"] = ov


def ensure(d: Path, cache_root: Path | None = None, timeout: float = TIME_BOX_S, force: bool = False) -> dict | None:
    """Fetch Overture buildings into cache dir `d` unless already there. Updates meta.json (``overture`` block +
    a warning on failure). Never raises. Returns the meta block."""
    d = Path(d)
    mp = d / "meta.json"
    meta = json.loads(mp.read_text(encoding="utf-8"))
    if not needs_fetch(d, meta, force):
        return meta["overture"]
    ov = try_fetch(meta_bbox(meta), Path(cache_root) if cache_root else d.parent, d / FILE, timeout)
    apply(meta, ov)
    mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return ov


# ------------------------------------------------------------------------------------------ attribution
def credit_for(dataset: str, licence: str | None = None) -> str:
    cr, lic = DATASET_CREDITS.get(dataset, (dataset, licence or "see Overture attribution page"))
    return f"{cr} ({lic})"


def attribution_block(datasets) -> dict:
    """Credit lines for the source datasets present in the data (OSM always listed first)."""
    names = [k for k in datasets if k]
    ordered = (["OpenStreetMap"] if "OpenStreetMap" in names else []) + sorted(k for k in names if k != "OpenStreetMap")
    return {"attribution": buildings_credit(ordered), "licence": THEME_LICENCE,
            "source_credits": [credit_for(k) for k in ordered]}


def buildings_credit(datasets=()) -> str:
    """One-line credit for the buildings layer. `datasets` = Overture source datasets actually used (may be empty)."""
    used = [x for x in datasets if x and x != "OpenStreetMap"]
    s = "Buildings: © OpenStreetMap contributors (ODbL); Overture Maps Foundation, buildings theme (ODbL)"
    if used:
        s += ", incl. " + ", ".join(credit_for(x) for x in used)
    return s


# ------------------------------------------------------------------------------------------ conflation
def load(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8")).get("features", [])


def source_datasets(props: dict) -> list[str]:
    return [s.get("dataset") for s in props.get("sources") or [] if s.get("dataset")]


def is_osm_sourced(props: dict) -> bool:
    return any(str(d).lower().startswith("openstreetmap") for d in source_datasets(props))


def primary_dataset(props: dict) -> str:
    ds = source_datasets(props)
    return ds[0] if ds else "unknown"


def is_estimated(props: dict) -> bool:
    """ML-derived footprints (Microsoft, Google, East-Asia ML) -> estimated; Esri community/municipal/USGS/IGN -> not."""
    d = primary_dataset(props).lower()
    return any(p in d for p in ML_PATTERNS)


def height_for(props: dict, floor_height: float, default_levels: float) -> tuple[float, str]:
    """Overture height -> num_floors x floor_height -> default. -> (metres, source)."""
    h = props.get("height")
    if isinstance(h, (int, float)) and h > 0.5:
        return float(h), "overture_height"
    nf = props.get("num_floors")
    if isinstance(nf, (int, float)) and nf > 0:
        return float(nf) * floor_height, "overture_floors"
    return default_levels * floor_height, "default"


def source_tag(props: dict) -> str:
    return "overture:" + primary_dataset(props)


def conflate(cands: list[tuple], osm_polys: list, max_overlap: float = OVERLAP_MAX) -> tuple[list[tuple], dict]:
    """cands = [(shapely polygon in the site frame, properties)]; osm_polys = shapely polygons of ALL OSM buildings
    (and parts) in the same frame. -> (kept [(poly, props)], stats). Rules: skip records sourced from OSM
    (duplicates), skip records whose area overlaps OSM buildings by >= max_overlap of their own area."""
    from shapely.strtree import STRtree
    st = Counter()
    tree = STRtree(osm_polys) if osm_polys else None
    kept = []
    for poly, props in cands:
        st["candidates"] += 1
        if is_osm_sourced(props):
            st["skipped_osm_source"] += 1
            continue
        a = poly.area
        if a <= 0:
            st["skipped_degenerate"] += 1
            continue
        ov = 0.0
        if tree is not None:
            for i in tree.query(poly):
                ov += poly.intersection(osm_polys[i]).area
        if ov / a >= max_overlap:
            st["skipped_overlap"] += 1
            continue
        kept.append((poly, props))
        st["kept"] += 1
    return kept, dict(st)
