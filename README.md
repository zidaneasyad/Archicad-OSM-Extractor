# OSM Context for Archicad

Pull a site's surroundings — terrain, buildings, roads, water, trees — from open data straight into an Archicad project, built with **native Archicad tools** (not imported meshes or DWG lines).

> **Status: v0.1.0, early release.** It works end to end, but some recent features (stepped contour model, plan colours, layered terrain sources) have only been tested offline so far. Save or back up your Archicad project before importing.

## What it does

Draw a boundary on a map, then **Fetch → Convert → Import**:

| Site element | Source | Archicad tool |
|---|---|---|
| Terrain | Best available elevation model for the site (national LiDAR where available, global fallback) | Mesh, or a stepped contour model made of Slabs |
| Contour lines | Derived from the terrain | 2D polylines with height labels on the floor plan |
| Buildings | OpenStreetMap + Overture Maps footprints and heights | Slab masses |
| Roads, rail | OpenStreetMap | Beams |
| Water | OpenStreetMap | Slabs |
| Trees, shrubs, hedges | OpenStreetMap, plus area-based scatter for parks and forests | Library objects |

Imported elements get classifications, an `OSM ID` / `Source` property and groups, so a later re-import replaces them cleanly. It also sets the project location and sun.

## Requirements

- Windows 10 / 11, 64-bit
- Archicad 30 with the [Tapir add-on](https://github.com/ENZYME-APD/tapir-archicad-automation/releases) installed
- An internet connection (map tiles and data downloads)
- For running from source: Python 3.13

## Run from source

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app_main.py
```

`app_main.py` opens the app in its own window. Alternatively, `Start OSM Context.bat` starts the backend and opens the map in your browser at `http://127.0.0.1:8765`.

Then:

1. Open Archicad 30 and the project you want to import into. The side panel shows whether Archicad and Tapir were found.
2. Draw a boundary. Start small for the first try (a 150 m square with the "super-light" preset).
3. **Fetch**, then **Convert**, then **Import**.

You can also launch it from Archicad's Python Palette: add the `palette` folder and click **OSM Context**.

## Build the Windows app

```powershell
.venv\Scripts\python.exe packaging\build.py
```

This produces `dist\OSM Context\OSM Context.exe` plus a zip. See `packaging/README.txt` for end-user instructions. The exe is not code-signed, so Windows SmartScreen may ask you to confirm ("More info → Run anyway").

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

The suite runs offline and uses a fake Archicad, so no Archicad instance is needed.

## Data credits

Keep these credits on anything you publish that uses imported data. The importer also writes the credits for the sources it used into the project.

- Map data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, ODbL
- Buildings: [Overture Maps Foundation](https://overturemaps.org) (ODbL parts)
- Elevation, depending on the site and the source chosen:
  - Copernicus GLO-30 DEM © DLR / ESA
  - LGL Baden-Württemberg, Datenlizenz Deutschland – Namensnennung – Version 2.0
  - USGS 3DEP (public domain), PDOK AHN (CC0 1.0), IGN France (Licence Ouverte 2.0), Environment Agency / NRW (OGL v3.0), Kartverket (CC BY 4.0), BIG Indonesia DEMNAS, PGC ArcticDEM / REMA (CC BY 4.0), Mapterhorn, OpenGeoHub GEDTM30 (CC BY 4.0)
  - FABDEM is **CC BY-NC-SA 4.0 (non-commercial)**: check its licence before commercial use

This is site context, not a survey: use a topographic survey for design levels.

## License

[MIT](LICENSE) © 2026 ENZYME APD
