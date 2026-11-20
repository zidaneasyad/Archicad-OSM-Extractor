"""Boundary helpers: UI boundary dict -> shapely Polygon in WGS84."""
from __future__ import annotations

import math

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import LineString, Polygon


def _local_transformers(lon: float, lat: float):
    """Azimuthal equidistant projection centred on (lon, lat): to metric and back."""
    crs = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    return fwd, inv


def transform(fn, geom):
    """Apply a pyproj-style fn(x_array, y_array) -> (x', y') to every vertex (shapely 2 API)."""
    return shapely.transform(geom, lambda c: np.column_stack(fn(c[:, 0], c[:, 1])))


def to_polygon(boundary: dict) -> Polygon:
    t = boundary.get("type")
    if t in ("rectangle", "polygon"):
        coords = [tuple(c) for c in boundary["coordinates"]]
        if len(coords) < 3:
            raise ValueError("need >= 3 coordinates")
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly
    if t == "circle":
        lon, lat = boundary["center"]
        r = float(boundary["radius_m"])
        _fwd, inv = _local_transformers(lon, lat)
        n = 64
        pts = [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n)]
        return transform(inv, Polygon(pts))
    if t == "polyline":
        coords = [tuple(c) for c in boundary["coordinates"]]
        if len(coords) < 2:
            raise ValueError("polyline needs >= 2 coordinates")
        buf = float(boundary.get("buffer_m", 10))
        clon = sum(c[0] for c in coords) / len(coords)
        clat = sum(c[1] for c in coords) / len(coords)
        fwd, inv = _local_transformers(clon, clat)
        line = transform(fwd, LineString(coords))
        return transform(inv, line.buffer(buf))
    raise ValueError(f"unknown boundary type: {t!r}")


def square_boundary(lat: float, lon: float, size_m: float) -> dict:
    """Rectangle boundary of size_m x size_m centred on lat/lon."""
    _fwd, inv = _local_transformers(lon, lat)
    h = size_m / 2.0
    ring = [inv(x, y) for x, y in [(-h, -h), (h, -h), (h, h), (-h, h), (-h, -h)]]
    return {"type": "rectangle", "coordinates": [[round(x, 7), round(y, 7)] for x, y in ring]}


def bbox(poly: Polygon, margin_m: float = 0) -> tuple:
    """(south, west, north, east) in degrees, expanded by margin_m metres."""
    w, s, e, n = poly.bounds
    if margin_m:
        lat0 = (s + n) / 2
        dlat = margin_m / 111320.0
        dlon = margin_m / (111320.0 * max(math.cos(math.radians(lat0)), 1e-6))
        s, n, w, e = s - dlat, n + dlat, w - dlon, e + dlon
    return (s, w, n, e)


def area_m2(boundary: dict) -> float:
    """Boundary area in m2 (metric azimuthal-equidistant projection centred on the boundary)."""
    poly = to_polygon(boundary)
    c = poly.centroid
    fwd, _inv = _local_transformers(c.x, c.y)
    return float(transform(fwd, poly).area)
