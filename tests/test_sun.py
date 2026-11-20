import datetime as dt
import math

import pytest

from osmctx import sun


def test_tuebingen_solstice_noon():
    p = sun.preview(48.5193, 9.0565, "2026-06-21", "12:00")
    assert p["compass_azimuth"] == pytest.approx(138.4, abs=0.3)
    assert p["altitude"] == pytest.approx(59.7, abs=0.3)
    assert p["archicad_azimuth"] == pytest.approx(311.6, abs=0.3)  # live Archicad: 311.63
    assert p["timeZoneInMinutes"] == 60 and p["sumTime"] is True


def test_tuebingen_winter_no_dst():
    p = sun.preview(48.5193, 9.0565, "2026-12-21", "12:00")
    assert p["timeZoneInMinutes"] == 60 and p["sumTime"] is False
    assert p["altitude"] == pytest.approx(18.0, abs=0.7)   # 90 - 48.52 - 23.44
    assert 170 < p["compass_azimuth"] < 190


def test_jakarta_no_dst():
    p = sun.preview(-6.2, 106.8, "2026-06-21", "12:00")
    assert p["timeZoneInMinutes"] == 420 and p["sumTime"] is False
    assert p["time_zone"] == "Asia/Jakarta"
    # June solstice, southern hemisphere: sun in the north, ~ 47 deg altitude at noon
    assert p["compass_azimuth"] < 30 or p["compass_azimuth"] > 330
    assert p["altitude"] == pytest.approx(90 - 6.2 - 23.44, abs=1.0)


def test_southern_dst_sydney_january():
    p = sun.preview(-33.87, 151.2, "2026-01-15", "12:00")
    assert p["timeZoneInMinutes"] == 600 and p["sumTime"] is True
    p = sun.preview(-33.87, 151.2, "2026-07-15", "12:00")
    assert p["timeZoneInMinutes"] == 600 and p["sumTime"] is False


def test_morning_sun_is_east_afternoon_west():
    am = sun.preview(48.5193, 9.0565, "2026-03-20", "09:00")["compass_azimuth"]
    pm = sun.preview(48.5193, 9.0565, "2026-03-20", "15:00")["compass_azimuth"]
    assert 90 < am < 180 < pm < 270


def test_archicad_angle():
    assert sun.archicad_angle(138.4) == pytest.approx(311.6)
    assert sun.archicad_angle(0) == 90 and sun.archicad_angle(90) == 0 and sun.archicad_angle(180) == 270


def test_place_settings_payload():
    p = sun.place_settings(48.5193, 9.0565, dt.date(2026, 6, 21), dt.time(12, 0), 314.5, 0.04233)["placeInfo"]
    assert p["timeZoneInMinutes"] == 60 and p["sumTime"] is True
    assert p["altitude"] == 314.5 and (p["year"], p["month"], p["day"], p["hour"]) == (2026, 6, 21, 12)
    assert p["north"] == pytest.approx(math.pi / 2 + math.radians(0.04233))
    assert p["latitude"] == 48.5193 and p["longitude"] == 9.0565


def test_place_settings_no_correction_north_is_pi_half():
    p = sun.place_settings(-6.2, 106.8, "2026-06-21", "12:00", 5)["placeInfo"]
    assert p["north"] == pytest.approx(math.pi / 2)
    assert p["timeZoneInMinutes"] == 420 and p["sumTime"] is False
