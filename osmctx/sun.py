"""Sun / place settings for Archicad.

- `sun_position(lat, lon, when_utc)` -> (compass_azimuth_deg, altitude_deg)   NOAA solar-position equations,
  no atmospheric refraction (Archicad's sun does not apply it either).
- `place_settings(...)` -> parameters for the Tapir command that sets the project place (MCP name
  project_set_place_settings; native add-on command "SetPlaceSettings", namespace "TapirCommand",
  parameters {"placeInfo": {...}}). Read-back / preview commands: "GetSunSettings", "CalculateSunPosition"
  (both verified live; the native name SetPlaceSettings is inferred from the MCP name, not yet called).
- `apply(payload)` POSTs it to Archicad. WRITES to the live model: only /api/import may call it.

Angle conventions
  compass azimuth : clockwise from true north (N=0, E=90, S=180, W=270).
  Archicad angle  : counter-clockwise from the +x (east) axis = (90 - compass) mod 360   (verified vs
                    CalculateSunPosition on 2026-06-21 12:00 CEST Tuebingen: compass 138.4 <-> 311.63).
  Archicad `north`: direction of TRUE north in project coordinates, radians counter-clockwise from +x.
                    Default pi/2 (true north = +y).  Project x/y are UTM grid offsets (grid north = +y); with
                    meridian convergence g (positive east of the central meridian) true north lies g CCW of
                    grid north, so  north = pi/2 + radians(g).  Tuebingen: g = +0.0423 deg.
                    (Sign derived from geometry, not yet checked against Archicad's sun with a rotated north.)
"""
from __future__ import annotations

import datetime as dt
import math
from zoneinfo import ZoneInfo

SET_COMMAND = "SetPlaceSettings"
_TZF = None


def archicad_angle(compass_deg: float) -> float:
    return (90.0 - compass_deg) % 360.0


# ------------------------------------------------------------------ NOAA solar position
def sun_position(lat: float, lon: float, when_utc: dt.datetime) -> tuple[float, float]:
    """(compass azimuth deg, altitude deg) for a UTC datetime."""
    if when_utc.tzinfo is not None:
        when_utc = when_utc.astimezone(dt.timezone.utc).replace(tzinfo=None)
    y, m = when_utc.year, when_utc.month
    d = when_utc.day + (when_utc.hour + (when_utc.minute + when_utc.second / 60) / 60) / 24
    if m <= 2:
        y, m = y - 1, m + 12
    a = y // 100
    b = 2 - a + a // 4
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5
    t = (jd - 2451545.0) / 36525.0
    l0 = (280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360
    mean_anom = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    ecc = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    mr = math.radians(mean_anom)
    c = (math.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t))
         + math.sin(2 * mr) * (0.019993 - 0.000101 * t) + math.sin(3 * mr) * 0.000289)
    true_long = l0 + c
    omega = 125.04 - 1934.136 * t
    app_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    eps0 = 23 + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60) / 60
    eps = eps0 + 0.00256 * math.cos(math.radians(omega))
    decl = math.asin(math.sin(math.radians(eps)) * math.sin(math.radians(app_long)))
    yy = math.tan(math.radians(eps / 2)) ** 2
    l0r = math.radians(l0)
    eq_time = 4 * math.degrees(yy * math.sin(2 * l0r) - 2 * ecc * math.sin(mr)
                               + 4 * ecc * yy * math.sin(mr) * math.cos(2 * l0r)
                               - 0.5 * yy * yy * math.sin(4 * l0r) - 1.25 * ecc * ecc * math.sin(2 * mr))
    minutes = when_utc.hour * 60 + when_utc.minute + when_utc.second / 60
    tst = (minutes + eq_time + 4 * lon) % 1440          # true solar time (min), lon east positive
    ha = math.radians(tst / 4 - 180)
    latr = math.radians(lat)
    cos_z = math.sin(latr) * math.sin(decl) + math.cos(latr) * math.cos(decl) * math.cos(ha)
    cos_z = max(-1.0, min(1.0, cos_z))
    zen = math.acos(cos_z)
    alt = 90 - math.degrees(zen)
    sz = math.sin(zen)
    if sz < 1e-9 or abs(math.cos(latr)) < 1e-9:
        return 180.0, alt
    cos_az = (math.sin(latr) * cos_z - math.sin(decl)) / (math.cos(latr) * sz)
    az = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    # NOAA: compass = 180 + az (afternoon, ha > 0) or 180 - az (morning)
    compass = (180 + az) % 360 if ha > 0 else (180 - az) % 360
    return compass, alt


# ------------------------------------------------------------------ time zone
def _tz_finder():
    global _TZF
    if _TZF is None:
        from timezonefinder import TimezoneFinder
        _TZF = TimezoneFinder()
    return _TZF


def tz_name(lat: float, lon: float) -> str:
    n = _tz_finder().timezone_at(lat=lat, lng=lon)
    if n:
        return n
    h = int(round(lon / 15.0))                       # open sea: nautical zone
    return "Etc/GMT%+d" % (-h) if h else "Etc/UTC"


def _minutes(d: dt.datetime) -> int:
    return int(d.utcoffset().total_seconds() // 60)


def tz_info(lat: float, lon: float, date: dt.date, time: dt.time) -> dict:
    """{name, std_minutes, dst, total_minutes} for the local wall-clock datetime.
    std = the smaller of the Jan-1 / Jul-1 offsets of that year (handles the southern hemisphere and
    Europe/Dublin's negative-DST encoding); dst = offset on that date is larger than std."""
    name = tz_name(lat, lon)
    z = ZoneInfo(name)
    total = _minutes(dt.datetime.combine(date, time, tzinfo=z))
    std = min(_minutes(dt.datetime(date.year, 1, 1, 12, tzinfo=z)),
              _minutes(dt.datetime(date.year, 7, 1, 12, tzinfo=z)), total)
    return {"name": name, "std_minutes": std, "dst": total > std, "total_minutes": total}


def _as_date(d) -> dt.date:
    return d if isinstance(d, dt.date) else dt.date.fromisoformat(str(d))


def _as_time(t) -> dt.time:
    if isinstance(t, dt.time):
        return t
    p = [int(x) for x in str(t).split(":")]
    return dt.time(*(p + [0, 0])[:3])


def local_to_utc(lat, lon, date, time) -> tuple[dt.datetime, dict]:
    date, time = _as_date(date), _as_time(time)
    tz = tz_info(lat, lon, date, time)
    local = dt.datetime.combine(date, time)
    return local - dt.timedelta(minutes=tz["total_minutes"]), tz


def preview(lat, lon, date, time) -> dict:
    """Sun for a local wall-clock date/time at lat/lon (what the UI shows before import)."""
    utc, tz = local_to_utc(lat, lon, date, time)
    comp, alt = sun_position(lat, lon, utc)
    return {"compass_azimuth": round(comp, 2), "altitude": round(alt, 2),
            "archicad_azimuth": round(archicad_angle(comp), 2), "time_zone": tz["name"],
            "timeZoneInMinutes": tz["std_minutes"], "sumTime": tz["dst"]}


# ------------------------------------------------------------------ Tapir payload
def place_settings(lat: float, lon: float, date, time, origin_elevation: float = 0.0,
                   north_correction_deg: float = 0.0) -> dict:
    """Tapir SetPlaceSettings parameters. north_correction_deg = meridian (grid) convergence from
    site.json report.info.crs.meridian_convergence_deg; north = pi/2 + radians(convergence)."""
    date, time = _as_date(date), _as_time(time)
    tz = tz_info(lat, lon, date, time)
    return {"placeInfo": {
        "longitude": lon, "latitude": lat, "altitude": float(origin_elevation or 0.0),
        "north": math.pi / 2 + math.radians(north_correction_deg or 0.0),
        "year": date.year, "month": date.month, "day": date.day,
        "hour": time.hour, "minute": time.minute, "second": time.second,
        "sumTime": tz["dst"], "timeZoneInMinutes": tz["std_minutes"],
    }}


def apply(payload: dict, url: str | None = None, timeout: float = 20) -> dict:
    """WRITE to Archicad (project place + sun). Only called from /api/import."""
    import requests
    from . import archicad_http
    url = url or archicad_http.DEFAULT_URL
    body = {"command": "API.ExecuteAddOnCommand", "parameters": {
        "addOnCommandId": {"commandNamespace": "TapirCommand", "commandName": SET_COMMAND},
        "addOnCommandParameters": payload}}
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    if not d.get("succeeded"):
        raise RuntimeError(f"Archicad {SET_COMMAND} failed: {d.get('error')}")
    return d.get("result", {}).get("addOnCommandResponse", {})
