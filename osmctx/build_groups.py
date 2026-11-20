"""Nested Archicad groups for everything the builder creates (CreateGroups, inner groups first).

Groups are unnamed in the Archicad UI: metadata goes into group user data via SetUserDataOfGroups.
"""
from __future__ import annotations

from . import build_plan as bp
from .archicad_http import ArchicadError, chunks, eid

GROUPS_PER_CALL = 100
USERDATA_PER_CALL = 200


def _nodes_by_height(node: dict) -> dict[int, list[dict]]:
    """Group nodes by tree height (0 = leaf groups holding only elements, higher = parent groups)."""
    levels: dict[int, list[dict]] = {}

    def walk(n: dict) -> int:
        ch_h = [walk(k) for k in n["children"] if isinstance(k, dict)]
        h = (max(ch_h) + 1) if ch_h else 0
        levels.setdefault(h, []).append(n)
        return h

    walk(node)
    return levels


def _create_one(c, items):
    """CreateGroups for one group -> its groupGuids entry, or {"error": ...}."""
    try:
        res = c.tapir("CreateGroups", {"elementGroups": [{"elements": items}]})["groupGuids"]
        return res[0] if res else {"error": "no result"}
    except ArchicadError as e:
        return {"error": str(e)}


def _create_groups(c, log, part):
    """CreateGroups for a chunk of (node, items) -> one result per node (aligned with `part`).
    A failed batch call falls back to one call per group, so one bad group cannot sink its siblings."""
    try:
        res = c.tapir("CreateGroups", {"elementGroups": [{"elements": items} for _, items in part]})["groupGuids"]
    except ArchicadError as e:
        log(f"  CreateGroups batch of {len(part)} FAILED ({e}): retrying one group per call")
        return [_create_one(c, items) for _, items in part]
    return list(res[:len(part)]) + [{"error": "no result"}] * (len(part) - len(res))


def _record_group_results(part, res_list, gid_of, summary, fails, log):
    for (node, items), res in zip(part, res_list):
        if "groupId" not in res:
            fails.append({"group": node["name"], "error": res.get("error")})
            log(f"  group '{node['name']}' FAILED {res.get('error')}")
            continue
        gid_of[id(node)] = res["groupId"]
        summary[node["name"]] = len(items)


def make_groups(c, log, created, report, build_date):
    plan = bp.group_plan(created)
    if not plan:
        report["groups"] = {"skipped": "fewer than 2 top-level items"}
        return
    gid_of, summary, fails = {}, {}, []
    levels = _nodes_by_height(plan)
    for h in sorted(levels):   # children (lower height) exist before the groups that contain them
        pending = []
        for node in levels[h]:
            items = [{"groupId": gid_of[id(k)]} if isinstance(k, dict) else eid(k) for k in node["children"]
                     if not isinstance(k, dict) or id(k) in gid_of]
            if len(items) < 2:
                fails.append({"group": node["name"], "error": f"fewer than 2 valid children ({len(items)})"})
                log(f"  group '{node['name']}' FAILED fewer than 2 valid children ({len(items)})")
                continue
            pending.append((node, items))
        for part in chunks(pending, GROUPS_PER_CALL):
            _record_group_results(part, _create_groups(c, log, part), gid_of, summary, fails, log)
    if gid_of:
        pairs = [{"key": "osmctx:group", "value": plan["name"]},
                 {"key": "osmctx:build", "value": build_date}]
        for part in chunks(list(gid_of.values()), USERDATA_PER_CALL):
            c.tapir("SetUserDataOfGroups", {"groups": [{"groupId": g} for g in part], "pairs": pairs})
    info = {"depth": bp.group_depth(plan), "groups_created": len(gid_of), "failed": fails,
            "direct_children": summary}
    report["groups"] = info
    log(f"groups: {info}")
