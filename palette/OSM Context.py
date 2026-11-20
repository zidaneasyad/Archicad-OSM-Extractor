"""OSM Context - start the site-context importer from Archicad's Python Palette.

How to use (once):
  Archicad -> Window -> Palettes -> Python Palette -> add (browse) THIS folder.
  Then click "OSM Context" in the palette list.

What it does: starts OSM Context if it isn't running yet; it opens in its own window. Imports go into the open Archicad project through the Tapir add-on.
Standard library only, so any Python 3.8+ the palette uses can run it.

Where it looks for the tool (first match wins):
  1. OSMCONTEXT_HOME environment variable (folder with "OSM Context.exe" or with app_main.py + .venv)
  2. %LOCALAPPDATA%\\OSMContext\\app\\OSM Context.exe   (installed packaged build)
  3. <this file's folder>\\..\\dist\\OSM Context\\OSM Context.exe   (local packaged build)
  4. <this file's folder>\\..\\app_main.py with .venv\\Scripts\\python.exe   (development folder)
"""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORT = int(os.environ.get("OSMCONTEXT_PORT", "8765"))
URL = f"http://127.0.0.1:{PORT}/"
HERE = Path(__file__).resolve().parent


def running() -> bool:
    try:
        with urllib.request.urlopen(URL + "api/limits", timeout=1.5) as r:
            return r.status == 200
    except Exception:
        return False


def find_command():
    roots = []
    if os.environ.get("OSMCONTEXT_HOME"):
        roots.append(Path(os.environ["OSMCONTEXT_HOME"]))
    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots.append(Path(local) / "OSMContext" / "app")
    roots += [HERE.parent / "dist" / "OSM Context", HERE.parent]
    for root in roots:
        exe = root / "OSM Context.exe"
        if exe.is_file():
            return [str(exe)], root
        py = root / ".venv" / "Scripts" / "python.exe"
        if (root / "app_main.py").is_file() and py.is_file():
            return [str(py), "app_main.py"], root
    return None, None


def main():
    if running():
        print("OSM Context is already running - its window is open (check the taskbar).")
        return 0
    cmd, cwd = find_command()
    if not cmd:
        print("OSM Context was not found. Set OSMCONTEXT_HOME to the folder that contains "
              "'OSM Context.exe' (or the project folder with app_main.py and .venv).")
        return 1
    flags = 0
    if sys.platform == "win32":  # detach so the palette returns and the backend keeps running
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
    subprocess.Popen(cmd, cwd=str(cwd), creationflags=flags, close_fds=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Starting OSM Context from {cwd} ...")
    for _ in range(60):  # up to ~30 s on first start
        if running():
            print("OSM Context window opened. Import goes into the open Archicad project via Tapir.")
            return 0
        time.sleep(0.5)
    print("The backend did not answer within 30 s. Check that nothing else uses port "
          f"{PORT}, then try again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
