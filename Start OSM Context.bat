@echo off
rem OSM Context for Archicad - start the local backend and open the map UI.
rem Imports go into the open Archicad project through the Tapir add-on (until the OSM Context .apx has a registered ID).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found in "%~dp0.venv". See README.md (Run from source).
  pause
  exit /b 1
)
rem Stop an older OSM Context backend still holding port 8765 (otherwise the browser keeps talking to old code).
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { $p = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $_.OwningProcess); if ($p.CommandLine -match 'server\.py') { Stop-Process -Id $p.ProcessId -Force; Write-Host ('Stopped old backend PID ' + $p.ProcessId) } }"
start "" "http://127.0.0.1:8765/"
echo OSM Context backend running on http://127.0.0.1:8765  (close this window to stop it)
".venv\Scripts\python.exe" server.py
