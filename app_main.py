"""OSM Context launcher (also the PyInstaller entry point).

Picks a free port (default 8765), runs the FastAPI app in-process with uvicorn (background thread), shows the map UI in
its own window (pywebview, Edge WebView2), stops the backend when the window closes (or Ctrl+C / POST /api/app/quit).
Fallback when WebView2/pywebview is unavailable: default browser (message + log).
Dev:  .venv/Scripts/python.exe app_main.py [--port N] [--browser] [--headless]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import socket
import sys
import threading
import time
import webbrowser

DEFAULT_PORT = 8765


def free_port(start: int = DEFAULT_PORT, tries: int = 50) -> int:
    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"no free port in {start}..{start + tries - 1}")


def _load_size(paths):
    try:
        d = json.loads(paths.SETTINGS_FILE.read_text(encoding="utf-8")).get("window", {})
        w, h = int(d["width"]), int(d["height"])
        if 600 <= w <= 8000 and 400 <= h <= 6000:
            return w, h
    except Exception:  # noqa: BLE001
        pass
    return 1400, 900


def _save_size(paths, w, h):
    try:
        cur = json.loads(paths.SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        cur = {}
    cur["window"] = {"width": int(w), "height": int(h)}
    paths.SETTINGS_FILE.write_text(json.dumps(cur), encoding="utf-8")


def main(argv=None) -> int:
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(prog="OSM Context")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--browser", action="store_true", help="use the default browser instead of the app window")
    ap.add_argument("--headless", "--no-browser", dest="headless", action="store_true", help="no UI at all (smoke tests)")
    a = ap.parse_args(argv)

    from osmctx import paths
    paths.ensure_dirs()

    import uvicorn
    import server

    handlers = [logging.FileHandler(paths.LOG_DIR / "osmcontext.log", encoding="utf-8")]
    if sys.stdout is not None:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers)

    port = free_port(a.port)
    url = f"http://127.0.0.1:{port}/"
    cfg = uvicorn.Config(server.app, host="127.0.0.1", port=port, log_config=None, log_level="info")
    srv = uvicorn.Server(cfg)
    server.QUIT_HOOK = lambda: setattr(srv, "should_exit", True)

    print(f"OSM Context {server.app_version()}")
    print(f"  UI:    {url}   (close the OSM Context window to stop)")
    print(f"  Data:  {paths.DATA_DIR}")
    if port != a.port:
        print(f"  Note:  port {a.port} busy, using {port}")

    log = logging.getLogger("osmcontext")
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(150):
        if srv.started or not t.is_alive():
            break
        time.sleep(0.1)
    if not srv.started:
        print("Backend failed to start, see " + str(paths.LOG_DIR / "osmcontext.log"))
        return 1
    st = server.archicad_status()
    print(f"  Archicad: [{st['level']}] {st['message']}")

    def browser_fallback(reason: str):
        msg = f"{reason} - opening the default browser instead ({url}). Keep this console open; close it to stop."
        print("  Note:  " + msg)
        log.warning(msg)
        webbrowser.open(url)
        try:
            while t.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

    try:
        if a.headless:
            while t.is_alive():
                time.sleep(0.5)
        elif a.browser:
            browser_fallback("--browser given")
        else:
            try:
                import webview
            except Exception as ex:  # noqa: BLE001
                browser_fallback(f"pywebview not available ({type(ex).__name__}: {ex})")
            else:
                size = _load_size(paths)
                win = webview.create_window("OSM Context", url, width=size[0], height=size[1], min_size=(900, 600))

                def on_closing():
                    with contextlib.suppress(Exception):
                        _save_size(paths, win.width, win.height)

                win.events.closing += on_closing
                try:
                    (paths.DATA_DIR / "webview").mkdir(parents=True, exist_ok=True)
                    webview.start(gui="edgechromium", storage_path=str(paths.DATA_DIR / "webview"), private_mode=False)
                except Exception as ex:  # noqa: BLE001  (WebView2 runtime missing, etc.)
                    log.exception("webview failed")
                    browser_fallback(f"Edge WebView2 not available ({type(ex).__name__}: {ex})")
    except KeyboardInterrupt:
        pass
    srv.should_exit = True
    t.join(timeout=10)
    print("OSM Context stopped.", flush=True)
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    rc = main()
    logging.shutdown()
    sys.stdout and sys.stdout.flush()
    os._exit(rc)   # hard exit: no stray non-daemon thread (duckdb/requests pools) may keep the console process alive
