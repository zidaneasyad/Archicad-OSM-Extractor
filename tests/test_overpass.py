"""Offline: shared Overpass race (freshness), fetch post-processing (stale warning, bridge chain completion)."""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from osmctx import fetch as F
from osmctx import overpass as OV

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)


def ans(base: str | None, n: int = 1):
    d = {"elements": [{"type": "node", "id": i} for i in range(n)]}
    if base:
        d["osm3s"] = {"timestamp_osm_base": base}
    return d


def test_freshness_and_warning():
    f = OV.freshness(ans("2026-06-01T08:52:28Z"), NOW)
    assert f["stale"] and f["age_days"] == pytest.approx(167.13, abs=0.01)
    assert "167 days old" in OV.stale_warning(f, "m")
    assert OV.freshness(ans("2026-11-15T07:00:00Z"), NOW)["stale"] is False
    assert OV.freshness(ans(None), NOW)["stale"] is None and OV.stale_warning(OV.freshness(ans(None))) is None


def test_race_prefers_fresh_over_faster_stale():
    def post(url, q):
        if url == "stale":
            return ans("2026-06-01T08:52:28Z")
        time.sleep(0.3)
        return ans("2026-11-15T07:00:00Z")
    data, url, info = OV.race("q", ["stale", "fresh"], timeout=5, post=post, now=NOW, fresh_wait=3)
    assert url == "fresh" and info["stale"] is False


def test_race_accepts_stale_when_nothing_fresh_answers():
    def post(url, q):
        if url == "stale":
            return ans("2026-06-01T08:52:28Z")
        raise RuntimeError("502")
    t0 = time.monotonic()
    data, url, info = OV.race("q", ["stale", "down"], timeout=5, post=post, now=NOW, fresh_wait=3)
    assert url == "stale" and info["stale"] is True and time.monotonic() - t0 < 2
    assert any("502" in e for e in info["errors"])


def test_race_stale_fallback_after_fresh_wait():
    def post(url, q):
        if url == "stale":
            return ans("2026-06-01T08:52:28Z")
        time.sleep(3)
        return ans("2026-11-15T07:00:00Z")
    t0 = time.monotonic()
    data, url, info = OV.race("q", ["stale", "slow"], timeout=10, post=post, now=NOW, fresh_wait=0.5)
    assert url == "stale" and info["stale"] and time.monotonic() - t0 < 2


def test_race_rejects_runtime_error_remark_and_unknown_age_accepted():
    def post(url, q):
        if url == "busy":
            return {"elements": [], "remark": "runtime error: Query timed out in \"query\" at line 3"}
        return ans(None, 3)
    data, url, info = OV.race("q", ["busy", "ok"], timeout=5, post=post, now=NOW)
    assert url == "ok" and len(data["elements"]) == 3 and info["stale"] is None


def test_query_raises_after_rounds():
    def post(url, q):
        raise RuntimeError("down")
    with pytest.raises(RuntimeError, match="all mirrors"):
        OV.query("q", ["a"], timeout=1, rounds=2, post=post, backoff_s=0.01)


def test_fetch_and_server_share_the_mirror_list():
    import server
    assert F.OVERPASS_URLS == OV.MIRRORS == server.OVERPASS and len(OV.MIRRORS) == 4


def test_query_rev3_tags():
    q = F.build_query((0, 0, 1, 1))
    for s in ('"natural"~"^(peak|volcano)$"', '"natural"="cliff"', '"natural"="sand"',
              '"wetland"~"^(tidalflat|saltmarsh|mangrove)$"', '"natural"="beach"'):
        assert s in q
    assert F.OSM_QUERY_REV >= 3


# --------------------------------------------------------------------------- bridge chains

def w(wid, nodes, tags, lonlat=None):
    lonlat = lonlat or [(0.001 * i, 0.0) for i in range(len(nodes))]
    return {"type": "way", "id": wid, "nodes": nodes, "tags": tags,
            "geometry": [{"lon": a, "lat": b} for a, b in lonlat]}


def test_dangling_ends_and_completion():
    data = {"elements": [w(1, [10, 11], {"highway": "primary", "bridge": "yes"}),
                         w(2, [9, 10], {"highway": "primary"})]}
    assert F.dangling_chain_ends(data["elements"]) == [11]
    replies = {11: [w(3, [11, 12], {"highway": "primary", "bridge": "yes"})],
               12: [w(3, [11, 12], {"highway": "primary", "bridge": "yes"}), w(4, [12, 13], {"highway": "primary"})]}
    asked = []

    def qf(q):
        ids = [int(x) for x in q.split("node(id:")[1].split(")")[0].split(",")]
        asked.append(ids)
        return {"elements": [e for i in ids for e in replies.get(i, [])]}, "mock"
    rec = F.complete_chains(data, qf)
    assert asked == [[11], [12]] and rec["rounds"] == 2 and sorted(rec["added_ways"]) == [3, 4]
    assert rec["unresolved_ends"] == [] and F.dangling_chain_ends(data["elements"]) == []


def test_completion_error_is_recorded_not_raised(tmp_path):
    data = {"elements": [w(1, [10, 11], {"highway": "primary", "bridge": "yes"})]}

    def bad(q):
        raise RuntimeError("mirrors down")
    rec = F.complete_chains(data, bad)
    assert rec["errors"] and rec["unresolved_ends"] == [10, 11]


def test_post_process_stale_warning_and_no_dem_ext(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "overpass", lambda q: (_ for _ in ()).throw(AssertionError("no query expected")))
    data = ans("2026-06-01T08:52:28Z")
    meta = {"warnings": [], "dem_bbox": {"south": 0, "west": 0, "north": 1, "east": 1}}
    F.post_process_osm(tmp_path, data, "https://overpass.kumi.systems/api/interpreter", meta)
    assert meta["osm_base"] == "2026-06-01T08:52:28Z" and meta["osm_age_days"] > 100
    assert any(x.startswith("OSM data is ") and "kumi" in x for x in meta["warnings"])
    assert meta["bridge_chains"]["rounds"] == 0 and "dem_ext" not in meta
    json.dumps(meta)
