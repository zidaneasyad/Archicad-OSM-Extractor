"""TEST HELPER ONLY - not part of the tool (views/sections/layouts are the user's; tool scope = fetch OSM data into Archicad).

Render the named views of the live Archicad project to PNG (Tapir ChangeWindow + SaveCurrentWindowAsImage).

    python -m osmctx.render_views [--plan out/site_plan.png] [--three-d out/site_3d.png]

The saved 3D view's camera cannot be stored through the API (gaps.md 20): the camera is set on the open window
each time (Isometric, azimuth 225, sun 2026-06-21 15:00 summer time).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .archicad_http import Client

PLAN_VIEW, THREED_VIEW = "Site Plan 1:500", "Site 3D"


def find_view(c, name):
    tree = c.tapir("GetNavigatorItemTree", {"navigatorMapId": "PublicViewMap"})

    def walk(n):
        if isinstance(n, dict):
            if n.get("name") == name and "navigatorItemId" in n:
                return n["navigatorItemId"]
            for v in n.values():
                r = walk(v)
                if r:
                    return r
        elif isinstance(n, list):
            for v in n:
                r = walk(v)
                if r:
                    return r
        return None
    return walk(tree)


def save(c, path):
    path = str(Path(path).resolve())
    return c.tapir("SaveCurrentWindowAsImage", {"imageFilePath": path, "overwrite": True})["imageFilePath"]


def render_plan(c, path, view=PLAN_VIEW):
    c.tapir("ChangeWindow", {"navigatorItemId": find_view(c, view)})
    return save(c, path)


def render_3d(c, path, view=THREED_VIEW, azimuth=225):
    c.tapir("ChangeWindow", {"navigatorItemId": find_view(c, view)})
    c.tapir("Set3DProjectionSettings", {"axonometry": {
        "projectionType": "Isometric", "azimuth": azimuth,
        "sun": {"sunPositionOption": "DateAndTime", "year": 2026, "month": 6, "day": 21,
                "hour": 15, "minute": 0, "second": 0, "summerTime": True}}})
    c.tapir("FitInWindow", {})
    return save(c, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default="out/site_plan.png")
    ap.add_argument("--three-d", default="out/site_3d.png")
    a = ap.parse_args(argv)
    with Client() as c:
        if a.plan:
            print(render_plan(c, a.plan))
        if a.three_d:
            print(render_3d(c, a.three_d))


if __name__ == "__main__":
    main()
