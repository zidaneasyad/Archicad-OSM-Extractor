import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from server import boundary_bbox, count_query, parse_coords, parse_counts


def close(r, lat, lon):
    assert r is not None
    assert r[0] == pytest.approx(lat, abs=1e-4)
    assert r[1] == pytest.approx(lon, abs=1e-4)


def test_decimal_comma():
    close(parse_coords("48.5193, 9.0565"), 48.5193, 9.0565)


def test_decimal_space():
    close(parse_coords("48.5193 9.0565"), 48.5193, 9.0565)


def test_negative():
    close(parse_coords("-8.829, 115.085"), -8.829, 115.085)


def test_dms():
    close(parse_coords("48°31'09.5\"N 9°03'23.4\"E"), 48.519306, 9.056500)


def test_dms_south_west():
    r = parse_coords("8°49'44.4\"S 115°05'06\"E")
    assert r[0] < 0 and r[1] > 0


def test_dms_lon_first():
    close(parse_coords("9°03'23.4\"E 48°31'09.5\"N"), 48.519306, 9.0565)


def test_google_at():
    close(parse_coords("https://www.google.com/maps/place/Tuebingen/@48.5193,9.0565,17z/data=!3m1"), 48.5193, 9.0565)


def test_google_q():
    close(parse_coords("https://maps.google.com/?q=48.5193,9.0565"), 48.5193, 9.0565)


def test_swapped():
    close(parse_coords("115.085, -8.829"), -8.829, 115.085)


@pytest.mark.parametrize("s", ["", "hello world", "abc, def", "200, 400", "https://example.com", None])
def test_garbage(s):
    assert parse_coords(s) is None


def test_bbox_and_query():
    b = {"type": "circle", "center": [9.0565, 48.5193], "radius_m": 50}
    s, w, n, e = boundary_bbox(b)
    assert n > 48.5193 > s and e > 9.0565 > w
    assert "out count" in count_query((s, w, n, e))


def test_parse_counts():
    data = {"elements": [{"type": "count", "tags": {"total": str(i)}} for i in (1, 2, 3, 4)]}
    assert parse_counts(data) == {"buildings": 1, "roads": 2, "trees": 3, "water": 4}
