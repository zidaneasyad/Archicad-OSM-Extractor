import math

from pyproj import Geod

from osmctx.boundary import bbox, square_boundary, to_polygon
from osmctx.fetch import boundary_hash

G = Geod(ellps="WGS84")


def area_m2(p):
    return abs(G.geometry_area_perimeter(p)[0])


def test_circle_area():
    p = to_polygon({"type": "circle", "center": [9.0565, 48.5193], "radius_m": 50})
    assert len(p.exterior.coords) == 65
    assert abs(area_m2(p) - math.pi * 2500) / (math.pi * 2500) < 0.01


def test_polyline_corridor():
    pts = [[9.0565, 48.5193], [9.0600, 48.5200]]
    length = G.line_length([c[0] for c in pts], [c[1] for c in pts])
    p = to_polygon({"type": "polyline", "coordinates": pts, "buffer_m": 10})
    exact = length * 20 + math.pi * 100  # rectangle + two round caps
    assert abs(area_m2(p) - exact) / exact < 0.05
    assert abs(area_m2(p) - length * 20) / (length * 20) < 0.10


def test_rectangle_size():
    b = square_boundary(48.5193, 9.0565, 100)
    p = to_polygon(b)
    assert b["type"] == "rectangle" and b["coordinates"][0] == b["coordinates"][-1]
    assert abs(area_m2(p) - 10000) < 50
    s, w, n, e = bbox(p)
    assert abs(G.inv(w, 48.5193, e, 48.5193)[2] - 100) < 0.5
    assert abs(G.inv(9.0565, s, 9.0565, n)[2] - 100) < 0.5
    s2, _w2, n2, _e2 = bbox(p, 30)
    assert abs(G.inv(9.0565, s2, 9.0565, n2)[2] - 160) < 1


def test_polygon_valid():
    ring = [[0, 0], [0, 0.001], [0.001, 0.001], [0.001, 0], [0, 0]]
    assert to_polygon({"type": "polygon", "coordinates": ring}).is_valid


def test_hash_stable():
    a = {"type": "circle", "center": [1, 2], "radius_m": 5}
    b = {"radius_m": 5, "center": [1, 2], "type": "circle"}
    assert boundary_hash(a) == boundary_hash(b)
    assert boundary_hash(a) != boundary_hash({**a, "radius_m": 6})
