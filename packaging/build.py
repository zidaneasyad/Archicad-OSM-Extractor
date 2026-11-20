"""Build dist/OSM Context/ + dist/OSMContext-<ver>-win64.zip.  Run: .venv\Scripts\python.exe packaging\build.py
Local only: nothing is uploaded."""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VER = (ROOT / "VERSION").read_text().strip()
B = ROOT / "build"
B.mkdir(exist_ok=True)

# 1) icon + Windows file properties
if not (ROOT / "packaging" / "osmcontext.ico").exists():
    subprocess.check_call([sys.executable, str(ROOT / "packaging" / "make_icon.py")])
v = tuple(int(x) for x in VER.split(".")) + (0,) * (4 - len(VER.split(".")))
(B / "version_info.txt").write_text(f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={v}, prodvers={v}, mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'ENZYME APD'),
      StringStruct('FileDescription', 'OSM Context - OpenStreetMap site context importer for Archicad'),
      StringStruct('FileVersion', '{VER}'),
      StringStruct('InternalName', 'OSM Context'),
      StringStruct('LegalCopyright', 'Map data OpenStreetMap contributors (ODbL), Overture Maps, Copernicus, LGL-BW'),
      StringStruct('OriginalFilename', 'OSM Context.exe'),
      StringStruct('ProductName', 'OSM Context'),
      StringStruct('ProductVersion', '{VER}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ])
""", encoding="utf-8")

# 2) duckdb extensions pre-installed into the bundle (first run needs no extension download)
ext = B / "duckdb_ext"
shutil.rmtree(ext, ignore_errors=True)
import duckdb
c = duckdb.connect()
c.execute(f"SET extension_directory='{ext.as_posix()}'")
for e in ("httpfs", "spatial"):
    c.execute(f"INSTALL {e}")
    c.execute(f"LOAD {e}")
c.close()

# 3) PyInstaller
subprocess.check_call([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", str(ROOT / "dist"),
                       "--workpath", str(B / "pyi"), str(ROOT / "packaging" / "osmcontext.spec")])

# 4) README + zip
dist = ROOT / "dist" / "OSM Context"
shutil.copyfile(ROOT / "packaging" / "README.txt", dist / "README.txt")
shutil.copytree(ROOT / "palette", dist / "palette", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
z = ROOT / "dist" / f"OSMContext-{VER}-win64.zip"
z.unlink(missing_ok=True)
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for p in dist.rglob("*"):
        if p.is_file():
            zf.write(p, Path("OSM Context") / p.relative_to(dist))
print("zip:", z, round(z.stat().st_size / 1e6, 1), "MB")
