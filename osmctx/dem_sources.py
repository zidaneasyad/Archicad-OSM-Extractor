"""Layered terrain/DEM source stack and raster utilities (M9a).

Implements the priority elevation cascade from ``research/osm_layered_sources.md``:
  Tier 1 (regional 0.5-10 m): bw_dgm1, nl_ahn, fr_ign, uk_ea, no_kartverket, us_3dep, id_demnas
  Tier 2 (polar 2 m):         arcticdem (>60 N), rema (<-60 S) with EGM2008 geoid correction
  Tier 3 (global 0.5-30 m):   mapterhorn (Terrarium tiles, AWS Terrarium fallback)
  Tier 4 (global 30 m DTM):   gedtm30 / fabdem
  Tier 5 (global fallback):   copernicus (delegated to fetch.dem_from_sources)

Includes Aleph-inspired utilities (atomic file writes, bounded parallel downloads,
Terrarium decoding, and pure-numpy northwest hillshade PNG rendering).
"""
from __future__ import annotations

import math
import os
import struct
import tempfile
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import requests

UA = "osm-context-archicad/0.1"
NODATA = -9999.0

# --------------------------------------------------------------------------- endpoints & envelopes

# (south, west, north, east) geographic envelopes for Tier 1/2 matching
ENVELOPES = {
    "id_demnas": (-11.2, 94.9, 6.2, 141.1),
    "us_3dep": (17.5, -179.9, 71.6, -64.5),
    "nl_ahn": (50.7, 3.3, 53.7, 7.3),
    "fr_ign": (41.3, -5.2, 51.1, 9.6),
    "uk_ea": (49.8, -6.5, 55.9, 1.8),
    "no_kartverket": (57.9, 4.5, 71.3, 31.3),
}

ENDPOINTS = {
    "id_demnas": [
        "https://geoservices.big.go.id/raster/rest/services/DEMNAS/DEMNAS/ImageServer",
        "https://geoservices.big.go.id/raster/rest/services/DEMNAS/ImageServer",
    ],
    "us_3dep": ["https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"],
    "nl_ahn": ("https://service.pdok.nl/rws/ahn/wcs/v1_0", "dtm_05m", 28992, 0.5),
    "fr_ign": ("https://data.geopf.fr/wcs", "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES", 2154, 1.0),
    "uk_ea": ("https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs",
              "1", 27700, 1.0),
    "no_kartverket": ["https://hoydedata.no/arcgis/rest/services/NHM_DTM_25833/ImageServer"],
    "arcticdem": ["https://di-pgc.img.arcgis.com/arcgis/rest/services/arcticdem_mosaic/ImageServer"],
    "rema": ["https://di-pgc.img.arcgis.com/arcgis/rest/services/rema_mosaic/ImageServer"],
    "mapterhorn_xyz": [
        "https://tiles.mapterhorn.com/{z}/{x}/{y}.webp",
        "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
    ],
    "gedtm30": [
        "https://s3.opengeohub.org/global/dtm/gedtm_rf_30m_20060101_20221231_go_epsg.4326_v20250212.tif",
    ],
    "fabdem": [
        "https://opentopography.s3.sdsc.edu/raster/FABDEM/FABDEM.vrt",
    ],
}

LAYERED_DEM_INFO = {
    "layered": {
        "name": "Layered terrain stack (Tier 1 Regional -> Polar -> Mapterhorn -> GEDTM30 -> Copernicus GLO-30)",
        "attribution": "Multi-tier elevation stack (see selected source attribution)",
        "licence": "Varies by selected source",
        "vertical_datum": "Varies (MSL / EGM2008)",
        "accuracy": {
            "horizontal_m": None,
            "vertical_m": None,
            "kind": "DTM",
            "note": "Selects highest-resolution open terrain source intersecting the site boundary.",
        },
    },
    "id_demnas": {
        "name": "Indonesia BIG DEMNAS (geoservices.big.go.id)",
        "attribution": "Badan Informasi Geospasial (BIG) Indonesia - DEMNAS",
        "licence": "BIG open geospatial data (Indonesia)",
        "vertical_datum": "EGM2008",
        "accuracy": {
            "horizontal_m": 8.0,
            "vertical_m": 2.5,
            "kind": "DTM",
            "note": "BIG DEMNAS national ~8 m elevation model (0.27 arc-second) for Indonesia.",
        },
    },
    "us_3dep": {
        "name": "USGS 3DEP Elevation (elevation.nationalmap.gov)",
        "attribution": "U.S. Geological Survey 3D Elevation Program (3DEP)",
        "licence": "Public domain (CC0 / US Gov)",
        "vertical_datum": "NAVD88",
        "accuracy": {
            "horizontal_m": 1.0,
            "vertical_m": 0.5,
            "kind": "DTM",
            "note": "USGS 3DEP best-available LiDAR/bare-earth DTM (1-10 m across US, AK, HI, PR).",
        },
    },
    "nl_ahn": {
        "name": "Netherlands PDOK AHN DTM (service.pdok.nl)",
        "attribution": "PDOK / Rijkswaterstaat Actueel Hoogtebestand Nederland (AHN)",
        "licence": "CC0 1.0",
        "vertical_datum": "NAP",
        "accuracy": {
            "horizontal_m": 0.5,
            "vertical_m": 0.15,
            "kind": "DTM",
            "note": "PDOK AHN 0.5 m airborne LiDAR bare-earth DTM.",
        },
    },
    "fr_ign": {
        "name": "France IGN LiDAR HD / RGE ALTI (data.geopf.fr)",
        "attribution": "IGN France - Géoplateforme RGE ALTI / LiDAR HD",
        "licence": "Licence Ouverte / Etalab 2.0",
        "vertical_datum": "IGN69",
        "accuracy": {
            "horizontal_m": 0.5,
            "vertical_m": 0.2,
            "kind": "DTM",
            "note": "IGN France high-resolution LiDAR DTM (0.5-1 m).",
        },
    },
    "uk_ea": {
        "name": "UK Environment Agency LiDAR DTM (environment.data.gov.uk)",
        "attribution": "Environment Agency (England) / Natural Resources Wales LiDAR DTM",
        "licence": "Open Government Licence v3.0",
        "vertical_datum": "ODN",
        "accuracy": {
            "horizontal_m": 1.0,
            "vertical_m": 0.15,
            "kind": "DTM",
            "note": "UK Environment Agency 1 m composite LiDAR DTM (~75% England, partial Wales).",
        },
    },
    "no_kartverket": {
        "name": "Norway Kartverket NDH DTM (hoydedata.no)",
        "attribution": "Kartverket - Nasjonal detaljert høydemodell (NDH)",
        "licence": "CC BY 4.0",
        "vertical_datum": "NN2000",
        "accuracy": {
            "horizontal_m": 1.0,
            "vertical_m": 0.2,
            "kind": "DTM",
            "note": "Kartverket 1 m LiDAR DTM (mainland Norway, excl. Svalbard).",
        },
    },
    "arcticdem": {
        "name": "PGC ArcticDEM 2 m Mosaic (di-pgc.img.arcgis.com)",
        "attribution": "Polar Geospatial Center (PGC), University of Minnesota - ArcticDEM",
        "licence": "CC BY 4.0",
        "vertical_datum": "EGM2008 (geoid-corrected from WGS84)",
        "accuracy": {
            "horizontal_m": 2.0,
            "vertical_m": 1.0,
            "kind": "DSM",
            "note": "PGC ArcticDEM 2 m strip/mosaic surface model (>60 N), converted to EGM2008 MSL.",
        },
    },
    "rema": {
        "name": "PGC REMA 2 m Mosaic (di-pgc.img.arcgis.com)",
        "attribution": "Polar Geospatial Center (PGC), University of Minnesota - REMA",
        "licence": "CC BY 4.0",
        "vertical_datum": "EGM2008 (geoid-corrected from WGS84)",
        "accuracy": {
            "horizontal_m": 2.0,
            "vertical_m": 1.0,
            "kind": "DSM",
            "note": "PGC Reference Elevation Model of Antarctica 2 m surface model, converted to EGM2008 MSL.",
        },
    },
    "mapterhorn": {
        "name": "Mapterhorn / Terrarium Global Terrain (tiles.mapterhorn.com)",
        "attribution": "Mapterhorn & national open LiDAR / AWS Terrain Tiles contributors",
        "licence": "Open national LiDAR / Copernicus / CC BY",
        "vertical_datum": "MSL / EGM2008",
        "accuracy": {
            "horizontal_m": 2.0,
            "vertical_m": 1.5,
            "kind": "DTM",
            "note": "Mapterhorn unified global terrain tiles (0.5-30 m, falls back to global 30 m).",
        },
    },
    "gedtm30": {
        "name": "OpenGeoHub GEDTM30 Bare-Earth DTM (s3.opengeohub.org)",
        "attribution": "OpenGeoHub Foundation - Global Ensemble Digital Terrain Model 30 m",
        "licence": "CC BY 4.0",
        "vertical_datum": "EGM2008",
        "accuracy": {
            "horizontal_m": 30.0,
            "vertical_m": 3.0,
            "kind": "DTM",
            "note": "ML-filtered 30 m global bare-earth DTM removing forest canopy and building bias.",
        },
    },
    "fabdem": {
        "name": "FABDEM 30 m Bare-Earth DTM (opentopography.s3.sdsc.edu)",
        "attribution": "Hawker et al. / University of Bristol - FABDEM V1-2",
        "licence": "CC BY-NC-SA 4.0 (non-commercial by default; check licence before commercial use)",
        "vertical_datum": "EGM2008",
        "accuracy": {
            "horizontal_m": 30.0,
            "vertical_m": 2.5,
            "kind": "DTM",
            "note": "Forest And Buildings removed Copernicus DEM (~30 m bare-earth DTM; non-commercial default licence).",
        },
    },
}

# Order checked by "layered" cascade
TIER1_ORDER = ("bw_dgm1", "nl_ahn", "fr_ign", "uk_ea", "no_kartverket", "us_3dep", "id_demnas")

# --------------------------------------------------------------------------- Aleph-inspired core utilities

@contextmanager
def atomic_path(path: Path):
    """Write to a sibling temporary file and replace atomically on exit (adapted from Belluxx/Aleph)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        yield tmp
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def parallel_fetch(fn, items, workers: int = 8):
    """Run ``fn(item)`` over ``items`` with at most ``2 * workers`` futures queued, yielding in input order."""
    items = list(items)
    if not items:
        return
    workers = max(1, int(workers))
    if workers == 1 or len(items) == 1:
        for item in items:
            yield fn(item)
        return
    pending: deque = deque()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            for item in items:
                pending.append(pool.submit(fn, item))
                if len(pending) > 2 * workers:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for fut in pending:
                fut.cancel()


def _split_rgb(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(rgb, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError(f"expected 3D RGB array, got shape {a.shape}")
    if a.shape[0] >= 3 and a.shape[0] <= 4 and a.shape[2] not in (3, 4):
        return a[0], a[1], a[2]
    if a.shape[2] >= 3:
        return a[:, :, 0], a[:, :, 1], a[:, :, 2]
    raise ValueError(f"unsupported RGB channel layout {a.shape}")


def decode_terrarium(rgb: np.ndarray) -> np.ndarray:
    """Decode Mapzen/Mapterhorn Terrarium RGB -> float32 elevation in metres."""
    r, g, b = _split_rgb(rgb)
    return ((r * 256.0 + g + b / 256.0) - 32768.0).astype(np.float32)


def encode_rgba_png(rgba: np.ndarray) -> bytes:
    """Minimal pure-Python RGBA PNG encoder for uint8 (H, W, 4) arrays (no Pillow needed)."""
    arr = np.asarray(rgba, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"expected (H, W, 4) RGBA array, got {arr.shape}")
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b"")


def hillshade_png(dem_path: Path, exaggeration: float = 1.0, max_dim: int = 1024) -> bytes:
    """Render northwest-lit slope shading RGBA PNG from a DEM GeoTIFF (adapted from Belluxx/Aleph)."""
    import rasterio
    with rasterio.open(dem_path) as ds:
        arr = ds.read(1).astype(np.float32)
        t = ds.transform
        crs = ds.crs
        nodata = ds.nodata
        bounds = ds.bounds
    bad = ~np.isfinite(arr)
    if nodata is not None:
        bad |= arr == float(nodata)
    h, w = arr.shape
    stride = max(1, int(math.ceil(max(h, w) / float(max_dim))))
    if stride > 1:
        arr = arr[::stride, ::stride]
        bad = bad[::stride, ::stride]
    valid = ~bad
    fill = float(np.mean(arr[valid])) if valid.any() else 0.0
    clean = np.where(valid, arr, fill)
    if crs is not None and not crs.is_projected:
        lat_mid = (bounds.bottom + bounds.top) / 2.0
        mpp_x = abs(t.a) * 111320.0 * max(math.cos(math.radians(lat_mid)), 1e-4) * stride
        mpp_y = abs(t.e) * 111320.0 * stride   # a degree of latitude is ~111 km everywhere, only longitude shrinks
    else:
        mpp_x, mpp_y = abs(t.a) * stride, abs(t.e) * stride
    mpp_x, mpp_y = max(float(mpp_x), 0.05), max(float(mpp_y), 0.05)

    padded = np.pad(clean, 1, mode="reflect", reflect_type="odd")
    east = (padded[1:-1, 2:] - padded[1:-1, :-2]) * (float(exaggeration) / (2.0 * mpp_x))
    south = (padded[2:, 1:-1] - padded[:-2, 1:-1]) * (float(exaggeration) / (2.0 * mpp_y))
    shade = (east * 0.5 + south * 0.5 + 0.7071) / np.sqrt(east * east + south * south + 1.0) - 0.7071
    alpha = np.minimum(np.maximum(-shade, 0.0) * 160.0, 120.0) + np.minimum(np.maximum(shade, 0.0) * 400.0, 120.0)
    tone = np.where(shade > 0.0, 255, 0).astype(np.uint8)
    a_u8 = np.where(valid, np.clip(alpha, 0.0, 255.0), 0.0).astype(np.uint8)
    rgba = np.dstack((tone, tone, tone, a_u8))
    return encode_rgba_png(rgba)


# --------------------------------------------------------------------------- envelope & CRS helpers

def bbox_intersects(bb: tuple, env: tuple) -> bool:
    """True when WGS84 (s, w, n, e) intersects envelope (es, ew, en, ee)."""
    s, w, n, e = bb
    es, ew, en, ee = env
    return not (n < es or s > en or e < ew or w > ee)


def write_float32_tif(path: Path, arr: np.ndarray, transform, crs, nodata_fill: float | None = None,
                      predictor: int | None = None, tag_area: bool = True) -> None:
    """Atomic float32 GeoTIFF (deflate). Non-finite -> ``nodata_fill`` (no nodata tag) if given, else NODATA tagged as
    nodata. Single writer for fetch.write_dem and every layered source."""
    import rasterio
    fill = NODATA if nodata_fill is None else nodata_fill
    out = np.where(np.isfinite(arr), arr, fill).astype(np.float32)
    kw = {} if nodata_fill is not None else {"nodata": NODATA}
    if predictor:
        kw["predictor"] = predictor
    with atomic_path(path) as tmp:
        with rasterio.open(tmp, "w", driver="GTiff", height=out.shape[0], width=out.shape[1], count=1,
                           dtype="float32", crs=crs, transform=transform, compress="deflate", **kw) as dst:
            dst.write(out, 1)
            if tag_area:
                dst.update_tags(AREA_OR_POINT="Area")


def gdal_env(allowed_ext: str | None = None):
    """rasterio.Env for /vsicurl/ reads (user agent, retries). ``allowed_ext`` limits the extensions vsicurl may probe."""
    import rasterio
    opts = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_USERAGENT": UA,
            "GDAL_HTTP_MAX_RETRY": "3", "GDAL_HTTP_RETRY_DELAY": "2"}
    if allowed_ext:
        opts["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = allowed_ext
    return rasterio.Env(**opts)


def apply_geoid_offset(arr: np.ndarray, lon: float, lat: float) -> tuple[np.ndarray, float, str | None]:
    """Convert WGS84 ellipsoidal elevations to EGM2008 orthometric heights at (lon, lat)."""
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4979", "EPSG:4326+3855", always_xy=True)
        _, _, h_ortho = tr.transform(float(lon), float(lat), 0.0)
        offset = float(h_ortho)
        if not math.isfinite(offset):
            raise ValueError("non-finite geoid offset")
        return (arr + offset).astype(np.float32), offset, None
    except Exception as ex:  # noqa: BLE001
        return arr.astype(np.float32), 0.0, (
            f"EGM2008 geoid grid unavailable ({type(ex).__name__}): polar heights remain WGS84 ellipsoidal"
        )


def _metric_span(bb: tuple) -> tuple[float, float]:
    s, w, n, e = bb
    lat0 = (s + n) / 2.0
    dy_m = max(1.0, (n - s) * 111320.0)
    dx_m = max(1.0, (e - w) * 111320.0 * max(math.cos(math.radians(lat0)), 1e-5))
    return dx_m, dy_m


# --------------------------------------------------------------------------- protocol fetchers

def fetch_arcgis_imageserver(
    service_url: str | list[str],
    bb: tuple,
    out_path: Path,
    target_res_m: float,
    bbox_sr: int = 4326,
    image_sr: int = 4326,
    unit_scale: float = 1.0,
    nodata_val: float = -9999.0,
    timeout: float = 60.0,
) -> dict:
    """Download a Float32 GeoTIFF from an ArcGIS REST ImageServer /exportImage endpoint."""
    import rasterio
    from rasterio.io import MemoryFile

    urls = [service_url] if isinstance(service_url, str) else list(service_url)
    s, w, n, e = bb
    if bbox_sr == 4326:
        req_bbox = f"{w:.7f},{s:.7f},{e:.7f},{n:.7f}"
    else:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{bbox_sr}", always_xy=True)
        x0, y0 = tr.transform(w, s)
        x1, y1 = tr.transform(e, n)
        req_bbox = f"{min(x0, x1):.3f},{min(y0, y1):.3f},{max(x0, x1):.3f},{max(y0, y1):.3f}"
    dx_m, dy_m = _metric_span(bb)
    width = max(16, min(4096, int(math.ceil(dx_m / max(target_res_m, 0.2)))))
    height = max(16, min(4096, int(math.ceil(dy_m / max(target_res_m, 0.2)))))
    params = {
        "bbox": req_bbox,
        "bboxSR": str(bbox_sr),
        "imageSR": str(image_sr),
        "size": f"{width},{height}",
        "format": "tiff",
        "pixelType": "F32",
        "noData": str(nodata_val),
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }
    last_err: Exception | None = None
    for base in urls:
        url = f"{base.rstrip('/')}/exportImage"
        try:
            r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=timeout)
            r.raise_for_status()
            ctype = (r.headers.get("content-type") or "").lower()
            if "json" in ctype or r.content.lstrip()[:1] == b"{":
                raise RuntimeError(f"ImageServer returned error JSON: {r.text[:240]}")
            with MemoryFile(r.content) as mem, mem.open() as ds:
                arr = ds.read(1).astype(np.float32)
                t = ds.transform
                crs = ds.crs or rasterio.CRS.from_epsg(image_sr)
                ds_nodata = ds.nodata
            bad = ~np.isfinite(arr) | (np.abs(arr) >= 1e30)
            if ds_nodata is not None:
                bad |= arr == float(ds_nodata)
            if nodata_val is not None:
                bad |= arr == float(nodata_val)
            arr = np.where(bad, np.nan, arr * float(unit_scale)).astype(np.float32)
            if np.isnan(arr).all():
                raise RuntimeError("ImageServer returned only nodata pixels")
            write_float32_tif(out_path, arr, t, crs)
            return {
                "url": base,
                "crs": crs.to_string(),
                "shape": list(arr.shape),
                "nan_count": int(np.isnan(arr).sum()),
            }
        except Exception as ex:  # noqa: BLE001
            last_err = ex
    raise RuntimeError(f"ArcGIS ImageServer failed: {last_err}")


def fetch_wcs_tiff(
    wcs_url: str,
    coverage_id: str,
    bb: tuple,
    out_path: Path,
    crs_epsg: int,
    res_m: float,
    nodata_cutoff: float = 1e30,
    timeout: float = 60.0,
) -> dict:
    """Fetch Float32 GeoTIFF via OGC WCS 2.0.1 (falling back to WCS 1.0.0)."""
    import rasterio
    from pyproj import Transformer
    from rasterio.io import MemoryFile

    s, w, n, e = bb
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{crs_epsg}", always_xy=True)
    xs, ys = zip(*(tr.transform(lo, la) for lo, la in ((w, s), (e, s), (e, n), (w, n))))
    xmin = math.floor(min(xs) / res_m) * res_m
    ymin = math.floor(min(ys) / res_m) * res_m
    xmax = math.ceil(max(xs) / res_m) * res_m
    ymax = math.ceil(max(ys) / res_m) * res_m
    width = max(16, min(4096, int(round((xmax - xmin) / res_m))))
    height = max(16, min(4096, int(round((ymax - ymin) / res_m))))

    attempts = [
        [
            ("SERVICE", "WCS"), ("VERSION", "2.0.1"), ("REQUEST", "GetCoverage"),
            ("COVERAGEID", coverage_id), ("FORMAT", "image/tiff"),
            ("SUBSET", f"x({xmin:.3f},{xmax:.3f})"), ("SUBSET", f"y({ymin:.3f},{ymax:.3f})"),
        ],
        [
            ("SERVICE", "WCS"), ("VERSION", "2.0.1"), ("REQUEST", "GetCoverage"),
            ("COVERAGEID", coverage_id), ("FORMAT", "image/tiff"),
            ("SUBSET", f"E({xmin:.3f},{xmax:.3f})"), ("SUBSET", f"N({ymin:.3f},{ymax:.3f})"),
        ],
        {
            "SERVICE": "WCS", "VERSION": "1.0.0", "REQUEST": "GetCoverage",
            "COVERAGE": coverage_id, "CRS": f"EPSG:{crs_epsg}", "FORMAT": "GeoTIFF",
            "BBOX": f"{xmin:.3f},{ymin:.3f},{xmax:.3f},{ymax:.3f}",
            "WIDTH": str(width), "HEIGHT": str(height),
        },
    ]
    last_err: Exception | None = None
    for params in attempts:
        try:
            r = requests.get(wcs_url, params=params, headers={"User-Agent": UA}, timeout=timeout)
            r.raise_for_status()
            ctype = (r.headers.get("content-type") or "").lower()
            if "xml" in ctype or "html" in ctype or r.content.lstrip()[:1] == b"<":
                raise RuntimeError(f"WCS returned XML/HTML error: {r.text[:240]}")
            with MemoryFile(r.content) as mem, mem.open() as ds:
                arr = ds.read(1).astype(np.float32)
                t = ds.transform
                crs = ds.crs or rasterio.CRS.from_epsg(crs_epsg)
                ds_nodata = ds.nodata
            bad = ~np.isfinite(arr) | (np.abs(arr) >= float(nodata_cutoff)) | (arr <= -9000.0)
            if ds_nodata is not None:
                bad |= arr == float(ds_nodata)
            arr = np.where(bad, np.nan, arr).astype(np.float32)
            if np.isnan(arr).all():
                raise RuntimeError("WCS returned only nodata pixels")
            write_float32_tif(out_path, arr, t, crs)
            return {
                "url": wcs_url,
                "crs": crs.to_string(),
                "shape": list(arr.shape),
                "nan_count": int(np.isnan(arr).sum()),
            }
        except Exception as ex:  # noqa: BLE001
            last_err = ex
    raise RuntimeError(f"WCS fetch failed ({wcs_url}): {last_err}")


def _latlon_to_tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    lat_c = max(-85.05112878, min(85.05112878, lat))
    n = 1 << z
    x = int(math.floor((lon + 180.0) / 360.0 * n))
    lat_r = math.radians(lat_c)
    y = int(math.floor((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n))
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def _pick_terrarium_zoom(bb: tuple, max_tiles: int = 16) -> int:
    s, w, n, e = bb
    for z in (15, 14, 13, 12, 11):
        x0, y0 = _latlon_to_tile(n, w, z)
        x1, y1 = _latlon_to_tile(s, e, z)
        if (x1 - x0 + 1) * (y1 - y0 + 1) <= max_tiles:
            return z
    return 11


def fetch_mapterhorn(bb: tuple, out_path: Path, cache_dir: Path, zoom: int | None = None, workers: int = 6) -> dict:
    """Fetch Terrarium tiles (Mapterhorn primary, AWS Terrarium fallback) into an EPSG:3857 Float32 GeoTIFF."""
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    z = int(zoom) if zoom is not None else _pick_terrarium_zoom(bb)
    s, w, n, e = bb
    x0, y0 = _latlon_to_tile(n, w, z)
    x1, y1 = _latlon_to_tile(s, e, z)
    coords = [(z, x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    tile_root = Path(cache_dir) / "_tiles" / "mapterhorn"
    tile_root.mkdir(parents=True, exist_ok=True)
    ses = requests.Session()
    ses.headers["User-Agent"] = UA

    def get_tile(xyz):
        tz, tx, ty = xyz
        for idx, tmpl in enumerate(ENDPOINTS["mapterhorn_xyz"]):
            ext = tmpl.rsplit(".", 1)[-1]
            cp = tile_root / f"{tz}_{tx}_{ty}_s{idx}.{ext}"
            cached = cp.exists() and cp.stat().st_size > 0
            if cached:
                raw = cp.read_bytes()
            else:
                url = tmpl.format(z=tz, x=tx, y=ty)
                r = ses.get(url, timeout=30)
                if r.status_code != 200 or len(r.content) < 64:
                    continue
                raw = r.content
            try:
                with MemoryFile(raw) as mem, mem.open() as ds:
                    bands = ds.read([1, 2, 3])
            except Exception:  # noqa: BLE001  undecodable (e.g. no WEBP driver, corrupt cache): next endpoint
                if cached:
                    cp.unlink(missing_ok=True)
                continue
            if not cached:   # only cache tiles that decode
                with atomic_path(cp) as tmp:
                    tmp.write_bytes(raw)
            return xyz, decode_terrarium(bands), idx
        raise RuntimeError(f"no Terrarium tile available for {tz}/{tx}/{ty}")

    try:
        decoded = list(parallel_fetch(get_tile, coords, workers=workers))
    finally:
        ses.close()
    tile_px = decoded[0][1].shape[0]
    cols = x1 - x0 + 1
    rows = y1 - y0 + 1
    mosaic = np.full((rows * tile_px, cols * tile_px), np.nan, dtype=np.float32)
    sources_used = set()
    for (_tz, tx, ty), arr, src_idx in decoded:
        if arr.shape[0] != tile_px:
            raise RuntimeError("mixed Terrarium tile sizes in mosaic")
        r_off = (ty - y0) * tile_px
        c_off = (tx - x0) * tile_px
        mosaic[r_off:r_off + tile_px, c_off:c_off + tile_px] = arr
        sources_used.add("mapterhorn" if src_idx == 0 else "aws_terrarium")

    half = 20037508.342789244
    tile_m = (2.0 * half) / (1 << z)
    px_m = tile_m / float(tile_px)
    west_3857 = -half + x0 * tile_m
    north_3857 = half - y0 * tile_m
    transform = from_origin(west_3857, north_3857, px_m, px_m)
    write_float32_tif(out_path, mosaic, transform, "EPSG:3857")
    return {
        "crs": "EPSG:3857",
        "zoom": z,
        "tile_px": tile_px,
        "tiles": len(coords),
        "sources": sorted(sources_used),
        "shape": list(mosaic.shape),
    }


def fetch_cog_vrt(urls: str | list[str], bb: tuple, out_path: Path, crs_fallback: str = "EPSG:4326") -> dict:
    """Clip a remote COG or VRT over HTTP (/vsicurl/) onto its native source pixel grid."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window

    url_list = [urls] if isinstance(urls, str) else list(urls)
    s, w, n, e = bb
    last_err: Exception | None = None
    with gdal_env():
        for src in url_list:
            vsi = "/vsicurl/" + src if src.startswith("http") else src
            try:
                with rasterio.open(vsi) as ds:
                    crs = ds.crs or rasterio.CRS.from_string(crs_fallback)
                    b = (w, s, e, n) if crs.to_epsg() == 4326 else transform_bounds("EPSG:4326", crs, w, s, e, n, densify_pts=21)
                    win = ds.window(*b).round_offsets().round_lengths()
                    c0 = max(0, int(win.col_off))
                    r0 = max(0, int(win.row_off))
                    c1 = min(ds.width, int(win.col_off + win.width))
                    r1 = min(ds.height, int(win.row_off + win.height))
                    if c1 <= c0 or r1 <= r0:
                        raise RuntimeError("COG/VRT does not overlap bbox")
                    w_sub = Window(c0, r0, c1 - c0, r1 - r0)
                    arr = ds.read(1, window=w_sub).astype(np.float32)
                    t = ds.window_transform(w_sub)
                    if ds.nodata is not None:
                        arr = np.where(arr == float(ds.nodata), np.nan, arr)
                    arr = np.where(np.isfinite(arr) & (np.abs(arr) < 1e30), arr, np.nan).astype(np.float32)
                    scale, offset = (ds.scales[0] or 1.0), (ds.offsets[0] or 0.0)   # packed-integer DEMs (e.g. dm)
                    if scale != 1.0 or offset != 0.0:
                        arr = (arr * np.float32(scale) + np.float32(offset)).astype(np.float32)
                    if np.isnan(arr).all():
                        raise RuntimeError("COG/VRT returned only nodata cells")
                    write_float32_tif(out_path, arr, t, crs)
                    return {"url": src, "crs": crs.to_string(), "shape": list(arr.shape)}
            except Exception as ex:  # noqa: BLE001
                last_err = ex
    raise RuntimeError(f"COG/VRT fetch failed: {last_err}")


# --------------------------------------------------------------------------- seam feathering

def _distance_to_invalid(valid: np.ndarray, max_px: int) -> np.ndarray:
    """Approximate pixel distance from the nearest invalid cell (0 at invalid cells, capped at max_px)."""
    h, w = valid.shape
    dist = np.where(valid, float(max_px), 0.0).astype(np.float32)
    for _ in range(max_px):
        padded = np.pad(dist, 1, mode="edge")
        nbr = np.minimum.reduce([
            padded[:-2, 1:-1],
            padded[2:, 1:-1],
            padded[1:-1, :-2],
            padded[1:-1, 2:],
            padded[:-2, :-2] + 0.4142,
            padded[:-2, 2:] + 0.4142,
            padded[2:, :-2] + 0.4142,
            padded[2:, 2:] + 0.4142,
        ])
        dist = np.where(valid, np.minimum(dist, nbr + 1.0), 0.0)
    return dist


def feather_with_fallback(primary_path: Path, bb: tuple, fallback_fn, feather_px: int = 12) -> list[str]:
    """Fill NaN/nodata holes in ``primary_path`` using ``fallback_fn(bb, tmp_path)`` and blend across seams."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(primary_path) as ds:
        arr = ds.read(1).astype(np.float32)
        t = ds.transform
        crs = ds.crs
        nodata = ds.nodata
    invalid = ~np.isfinite(arr)
    if nodata is not None:
        invalid |= arr == float(nodata)
    if not invalid.any():
        return []

    warnings: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        fb_path = Path(td) / "fallback.tif"
        fb_info = fallback_fn(bb, fb_path)
        if isinstance(fb_info, dict) and fb_info.get("warnings"):
            warnings.extend(fb_info["warnings"])
        fb_reproj = np.full(arr.shape, np.nan, dtype=np.float32)
        with rasterio.open(fb_path) as fds:
            reproject(
                source=rasterio.band(fds, 1),
                destination=fb_reproj,
                src_transform=fds.transform,
                src_crs=fds.crs,
                src_nodata=fds.nodata,
                dst_transform=t,
                dst_crs=crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    valid = ~invalid
    if feather_px > 0 and valid.any() and np.isfinite(fb_reproj).any():
        dist = _distance_to_invalid(valid, feather_px)
        t_ramp = np.clip(dist / float(feather_px), 0.0, 1.0)
        alpha = 0.5 * (1.0 - np.cos(math.pi * t_ramp))
        both = valid & np.isfinite(fb_reproj)
        prim_clean = np.where(valid, arr, fb_reproj)
        fb_clean = np.where(np.isfinite(fb_reproj), fb_reproj, prim_clean)
        blended = np.where(both, alpha * prim_clean + (1.0 - alpha) * fb_clean, np.where(valid, prim_clean, fb_clean))
    else:
        blended = np.where(valid, arr, fb_reproj)

    write_float32_tif(primary_path, blended, t, crs)
    warnings.append(f"Feathered {int(invalid.sum())} missing regional DEM cells with fallback terrain")
    return warnings


# --------------------------------------------------------------------------- unified layered dispatcher

def _run_source(src_id: str, bb: tuple, out_path: Path, cache_dir: Path, bw_fn=None) -> tuple[dict, list[str]]:
    warns: list[str] = []
    if src_id == "bw_dgm1":
        if bw_fn is None:
            raise RuntimeError("bw_dgm1 fetcher not provided")
        info = bw_fn(bb, out_path)
        return {"dem_crs": "EPSG:25832", **info}, list(info.get("warnings") or [])
    if src_id == "id_demnas":
        info = fetch_arcgis_imageserver(ENDPOINTS["id_demnas"], bb, out_path, target_res_m=8.0)
        return {"dem_crs": info["crs"], **info}, warns
    if src_id == "us_3dep":
        info = fetch_arcgis_imageserver(ENDPOINTS["us_3dep"], bb, out_path, target_res_m=1.0)
        return {"dem_crs": info["crs"], **info}, warns
    if src_id == "no_kartverket":
        info = fetch_arcgis_imageserver(ENDPOINTS["no_kartverket"], bb, out_path, target_res_m=1.0, image_sr=25833)
        return {"dem_crs": info["crs"], **info}, warns
    if src_id in ("nl_ahn", "fr_ign", "uk_ea"):
        url, cov, epsg, res_m = ENDPOINTS[src_id]
        info = fetch_wcs_tiff(url, cov, bb, out_path, epsg, res_m)
        return {"dem_crs": info["crs"], **info}, warns
    if src_id in ("arcticdem", "rema"):
        info = fetch_arcgis_imageserver(ENDPOINTS[src_id], bb, out_path, target_res_m=2.0)
        import rasterio
        s, w, n, e = bb
        with rasterio.open(out_path) as ds:
            arr = ds.read(1).astype(np.float32)
            t, crs, nd = ds.transform, ds.crs, ds.nodata
        bad = ~np.isfinite(arr) | ((arr == float(nd)) if nd is not None else False)
        adj, off, gwarn = apply_geoid_offset(np.where(bad, np.nan, arr), (w + e) / 2.0, (s + n) / 2.0)
        if gwarn:
            warns.append(gwarn)
        write_float32_tif(out_path, adj, t, crs)
        info["geoid_offset_m"] = round(off, 3)
        return {"dem_crs": info["crs"], **info}, warns
    if src_id == "mapterhorn":
        info = fetch_mapterhorn(bb, out_path, cache_dir)
        return {"dem_crs": info["crs"], **info}, warns
    if src_id in ("gedtm30", "fabdem"):
        info = fetch_cog_vrt(ENDPOINTS[src_id], bb, out_path)
        return {"dem_crs": info["crs"], **info}, warns
    raise ValueError(f"unknown layered DEM source {src_id!r}")


def fetch_layered_dem(
    bb: tuple,
    out_path: Path,
    cache_dir: Path,
    requested: str = "layered",
    bw_fn=None,
    copernicus_fn=None,
) -> dict:
    """Resolve ``requested`` ('layered' or a specific Tier 1-4 id) into ``out_path``."""
    s, w, n, e = bb

    def run(src_id: str, fallback_reason: str | None) -> dict:
        extra, warns = _run_source(src_id, bb, out_path, cache_dir, bw_fn=bw_fn)
        if copernicus_fn is not None:
            warns.extend(feather_with_fallback(out_path, bb, copernicus_fn))
        return {"used": src_id, "dem_crs": extra.pop("dem_crs", "EPSG:4326"), "warnings": warns,
                "fallback_reason": fallback_reason, "extra": extra}

    if requested != "layered":
        return run(requested, None)

    candidates = [src_id for src_id in TIER1_ORDER
                  if src_id == "bw_dgm1" or bbox_intersects(bb, ENVELOPES[src_id])]
    if n >= 60.0:
        candidates.append("arcticdem")
    if s <= -60.0:
        candidates.append("rema")
    candidates.extend(["mapterhorn", "gedtm30"])

    reasons: list[str] = []
    for src_id in candidates:
        try:
            return run(src_id, "; ".join(reasons) if reasons else None)
        except Exception as ex:  # noqa: BLE001
            reasons.append(f"{src_id}: {type(ex).__name__}: {ex}")

    if copernicus_fn is None:
        raise RuntimeError("all layered DEM tiers failed: " + "; ".join(reasons))
    fb_info = copernicus_fn(bb, out_path)
    return {
        "used": "copernicus",
        "dem_crs": "EPSG:4326",
        "warnings": list(fb_info.get("warnings") or []),
        "fallback_reason": "; ".join(reasons) if reasons else None,
        "extra": {},
    }
