# PyInstaller spec - onedir build of OSM Context.  Run via packaging/build.py (prepares VERSION info + duckdb extensions).
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
datas, binaries, hiddenimports = [], [], []

for pkg in ("rasterio", "pyproj", "shapely", "timezonefinder", "timezonefinder_data", "certifi", "duckdb", "webview", "clr_loader", "pythonnet"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

hiddenimports += ["clr", "uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto", "uvicorn.protocols.http.h11_impl",
                  "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on", "multipart"]

datas += [
    (os.path.join(ROOT, "ui"), "ui"),
    (os.path.join(ROOT, "VERSION"), "."),
    (os.path.join(ROOT, "out", "archicad_setup.json"), "defaults"),
    (os.path.join(ROOT, "build", "duckdb_ext"), "duckdb_extensions"),
]

excludes = ["pytest", "onnxruntime", "matplotlib", "tkinter", "pandas", "scipy", "IPython", "PIL", "fitz", "pymupdf", "numba",
            "llvmlite", "setuptools", "pip", "wheel"]

a = Analysis([os.path.join(ROOT, "app_main.py")], pathex=[ROOT], binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             hookspath=[], runtime_hooks=[], excludes=excludes, noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="OSM Context", console=True, icon=os.path.join(SPECPATH, "osmcontext.ico"),
          version=os.path.join(ROOT, "build", "version_info.txt"), upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="OSM Context")
