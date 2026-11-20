OSM Context 0.1.0 - OpenStreetMap site context for Archicad
===========================================================

Requirements
  - Windows 10 / 11, 64-bit
  - Archicad 30 with the Tapir add-on installed (Tapir: https://github.com/ENZYME-APD/tapir-archicad-automation/releases ,
    "install Tapir from its GitHub releases")
  - Internet connection (map tiles, OpenStreetMap / Overture / elevation downloads)
  - No Python needed.

Start
  1. Unzip the whole "OSM Context" folder anywhere (e.g. C:\Tools\OSM Context).
  2. Open Archicad 30 and the project you want to import into.
  3. Double-click "OSM Context.exe". A small console window opens and OSM Context appears in its own window
     (internal port 8765, the next free port is used if busy). If Edge WebView2 is missing, your default browser is
     used instead (WebView2 ships with Windows 10/11 and Microsoft Edge).
  4. The top of the side panel shows whether Archicad and Tapir were found (ports 19723-19743).
     Fix it and press "Re-check" if not.
  5. Draw a boundary, Fetch, Convert, Import. Start small (150 m square, "super-light") for the first try.
  Close the OSM Context window to stop (the console closes with it).

Optional: start it from Archicad's Python Palette
  Archicad -> Window -> Palettes -> Python Palette -> add (browse) the "palette" folder next to this README, then click
  "OSM Context" in the list. It looks for OSM Context.exe at %LOCALAPPDATA%\OSMContext\app\ , in ..\dist\OSM Context\ ,
  or in the folder named by the OSMCONTEXT_HOME environment variable - so either copy the "OSM Context" folder to
  %LOCALAPPDATA%\OSMContext\app\ or set OSMCONTEXT_HOME to it.

Windows SmartScreen
  The exe has no purchased code-signing certificate, so Windows may say "Windows protected your PC / unknown publisher".
  Click "More info" -> "Run anyway".

Where your data goes
  %LOCALAPPDATA%\OSMContext\   (cache\ = downloaded map/elevation data, out\ = converted site + build logs,
  logs\ = application log). Nothing is written next to the exe. You can delete cache\ any time.
  Imports go into the OPEN Archicad project through Tapir; save/backup your project first.

Data credits (keep them on anything you publish)
  - Map data (c) OpenStreetMap contributors, ODbL - https://www.openstreetmap.org/copyright
  - Buildings: Overture Maps Foundation (ODbL parts) - https://overturemaps.org
  - Elevation: Copernicus GLO-30 DEM (c) DLR / ESA; in Baden-Wuerttemberg: LGL-BW (2026),
    Datenlizenz Deutschland - Namensnennung - Version 2.0, www.lgl-bw.de
  Site context only, not a survey: use a topographic survey for design levels.
