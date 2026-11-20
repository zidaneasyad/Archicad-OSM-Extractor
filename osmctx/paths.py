"""One place for every filesystem location (dev workflow vs frozen PyInstaller build).

- RESOURCE_DIR: read-only assets (ui/, defaults/). Dev: project root. Frozen: the bundle (sys._MEIPASS).
- DATA_DIR: writable (cache/, out/, logs/, settings.json). Dev: project root (unchanged workflow).
  Frozen: %LOCALAPPDATA%/OSMContext. Env OSMCTX_DATA_DIR overrides both.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", _PROJECT_ROOT)) if FROZEN else _PROJECT_ROOT


def _data_dir() -> Path:
    env = os.environ.get("OSMCTX_DATA_DIR")
    if env:
        return Path(env).resolve()
    if FROZEN:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "OSMContext"
    return _PROJECT_ROOT


DATA_DIR = _data_dir()
CACHE_DIR = DATA_DIR / "cache"
OUT_DIR = DATA_DIR / "out"
LOG_DIR = DATA_DIR / "logs" if (FROZEN or os.environ.get("OSMCTX_DATA_DIR")) else DATA_DIR / "out"
SETTINGS_FILE = DATA_DIR / "settings.json"
UI_DIR = RESOURCE_DIR / "ui"
# shipped default for the Archicad attribute setup (dev: the project's own out/ copy)
DEFAULT_SETUP = (RESOURCE_DIR / "defaults" / "archicad_setup.json") if FROZEN else (_PROJECT_ROOT / "out" / "archicad_setup.json")
DUCKDB_EXT_DIR = RESOURCE_DIR / "duckdb_extensions"   # pre-installed httpfs/spatial (frozen build only)


def setup_json() -> Path:
    """User copy in out/ wins (editable), else the shipped default."""
    p = OUT_DIR / "archicad_setup.json"
    return p if p.exists() else DEFAULT_SETUP


def ensure_dirs() -> None:
    for d in (CACHE_DIR, OUT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / "archicad_setup.json"
    if not target.exists() and DEFAULT_SETUP.exists():
        shutil.copyfile(DEFAULT_SETUP, target)
