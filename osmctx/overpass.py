"""Overpass access shared by fetch (site data) and server (counts): race all mirrors, first valid FRESH answer wins.

Mirrors differ in data age (2026-11-15: private.coffee and kumi.systems served osm_base 2026-06-01 while
overpass-api.de / mail.ru were current). Every answer carries `osm3s.timestamp_osm_base`; an answer older than
`fresh_days` is kept only as a fallback while the race waits (up to `fresh_wait` s) for a fresh mirror. If only stale
mirrors answer, the stale data is returned with info["stale"] = True so callers can warn with the data date.
Answers without a timestamp (unknown age) are accepted immediately.
"""
from __future__ import annotations

import concurrent.futures as cf
import time
from datetime import datetime, timezone

import requests

UA = "osm-context-archicad/0.1"
MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
FRESH_DAYS = 7.0


def osm_base(data: dict) -> datetime | None:
    """`osm3s.timestamp_osm_base` of an Overpass JSON answer (UTC) or None."""
    ts = ((data or {}).get("osm3s") or {}).get("timestamp_osm_base")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def freshness(data: dict, now: datetime | None = None, fresh_days: float = FRESH_DAYS) -> dict:
    """{"osm_base": iso|None, "age_days": float|None, "stale": bool|None} for an answer."""
    b = osm_base(data)
    if b is None:
        return {"osm_base": None, "age_days": None, "stale": None}
    now = now or datetime.now(timezone.utc)
    age = (now - b).total_seconds() / 86400.0
    return {"osm_base": b.isoformat().replace("+00:00", "Z"), "age_days": round(age, 2), "stale": age > fresh_days}


def stale_warning(info: dict, url: str | None = None) -> str | None:
    if not info or not info.get("stale"):
        return None
    return (f"OSM data is {info['age_days']:.0f} days old (osm_base {info['osm_base']}"
            + (f", mirror {url}" if url else "") + "): no up-to-date Overpass mirror answered; recent OSM edits are missing")


def remark_error(data: dict) -> str | None:
    rem = str((data or {}).get("remark", ""))
    low = rem.lower()
    if "runtime error" in low or "timed out" in low or "out of memory" in low:
        return rem[:160]
    return None


def default_post(timeout: float):
    def post(url: str, q: str) -> dict:
        r = requests.post(url, data={"data": q}, headers={"User-Agent": UA}, timeout=(10, timeout))
        r.raise_for_status()
        return r.json()
    return post


def race(q: str, urls=None, timeout: float = 120.0, post=None, validate=None, fresh_days: float = FRESH_DAYS,
         fresh_wait: float = 20.0, now: datetime | None = None):
    """Send q to all `urls` in parallel. Returns (data, url, info) with info = {errors, osm_base, age_days, stale}.
    data/url are None when no valid answer arrived within `timeout`. `post(url, q)` -> JSON dict (raises on error);
    `validate(data)` raises to reject an answer (Overpass runtime-error remarks are always rejected).
    A stale answer (older than fresh_days) is held back up to `fresh_wait` s (bounded by `timeout`) for a fresh one."""
    urls = list(urls or MIRRORS)
    post = post or default_post(timeout)
    errs: list[str] = []
    t0 = time.monotonic()
    ex = cf.ThreadPoolExecutor(max_workers=len(urls))
    futs = {ex.submit(post, u, q): u for u in urls}
    fallback = None  # (data, url, info) of the best stale answer so far
    deadline = t0 + timeout
    try:
        pending = set(futs)
        while pending:
            left = deadline - time.monotonic()
            if fallback is not None:
                left = min(left, fallback[3] - time.monotonic())
            if left <= 0:
                break
            done, pending = cf.wait(pending, timeout=left, return_when=cf.FIRST_COMPLETED)
            if not done:
                break
            for f in done:
                u = futs[f]
                try:
                    data = f.result()
                    err = remark_error(data)
                    if err:
                        raise RuntimeError(f"remark: {err}")
                    if validate is not None:
                        validate(data)
                except Exception as e:  # noqa: BLE001
                    errs.append(f"{u}: {type(e).__name__}: {str(e)[:120]}")
                    continue
                info = freshness(data, now, fresh_days)
                if not info["stale"]:
                    return data, u, {**info, "errors": errs}
                if fallback is None or (info["age_days"] or 0) < (fallback[2]["age_days"] or 0):
                    wait_until = (fallback[3] if fallback else time.monotonic() + fresh_wait)
                    fallback = (data, u, info, wait_until)
        if fallback is None and pending:
            errs.append(f"no valid answer within {timeout:.0f} s")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    if fallback is not None:
        data, u, info, _ = fallback
        return data, u, {**info, "errors": errs}
    return None, None, {"osm_base": None, "age_days": None, "stale": None, "errors": errs}


def query(q: str, urls=None, timeout: float = 150.0, rounds: int = 2, post=None, backoff_s: float = 5.0, **kw):
    """race() with retry rounds; raises RuntimeError when every round fails. Returns (data, url, info)."""
    errs: list[str] = []
    for rnd in range(rounds):
        data, url, info = race(q, urls, timeout, post=post, **kw)
        if data is not None:
            info["errors"] = errs + info["errors"]
            return data, url, info
        errs += info["errors"]
        if rnd + 1 < rounds:
            time.sleep(backoff_s * (rnd + 1))
    raise RuntimeError("Overpass failed on all mirrors: " + " | ".join(errs[-8:]))
