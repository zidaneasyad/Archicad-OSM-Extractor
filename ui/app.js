/* OSM Context for Archicad - map UI. No build step. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const R_EARTH = 6371008.8;
  // Size thresholds come from the server (GET /api/limits, osmctx/limits.py) so UI and server agree.
  let limits = null;
  fetch("/api/limits").then((r) => r.json()).then((d) => { limits = d; refresh(); }).catch(() => { /* server guard still applies */ });
  // First-run check: Archicad reachable + Tapir present (GET /api/app/status)
  function checkArchicad() {
    const t = $("acStatusTxt"), box = $("acStatus");
    if (!t) return;
    t.textContent = "Checking Archicad...";
    fetch("/api/app/status").then((r) => r.json()).then((d) => {
      const a = d.archicad || {};
      t.textContent = (a.message || "") + (d.version ? "  [v" + d.version + "]" : "");
      box.className = "msg " + (a.level === "ok" ? "ok" : "err");
    }).catch(() => { t.textContent = "Backend not answering."; box.className = "msg err"; });
  }
  if ($("acRecheck")) $("acRecheck").addEventListener("click", checkArchicad);
  checkArchicad();
  function tierOf(km2) {
    if (!limits || km2 == null) return "normal";
    if (km2 <= limits.normal_max_km2) return "normal";
    if (km2 <= limits.super_light_above_km2) return "large";
    if (km2 <= limits.block_above_km2) return "super_light";
    return "blocked";
  }
  // explicit "Use normal settings" choice is remembered per boundary (rounded bbox signature)
  let overrides = new Set();
  try { overrides = new Set(JSON.parse(localStorage.getItem("osmctx.overrides") || "[]")); } catch (e) { /* ignore */ }
  function saveOverrides() { try { localStorage.setItem("osmctx.overrides", JSON.stringify(Array.from(overrides).slice(-50))); } catch (e) { /* ignore */ } }
  function boundarySig() {
    if (!current) return null;
    const lyr = current.layer;
    if (!lyr.getBounds) { const c = lyr.getLatLng(); return [current.type, c.lat.toFixed(4), c.lng.toFixed(4), Math.round(lyr.getRadius())].join("|"); }
    const b = lyr.getBounds();
    return [current.type, b.getSouth().toFixed(4), b.getWest().toFixed(4), b.getNorth().toFixed(4), b.getEast().toFixed(4)].join("|");
  }
  function overrideOn() { const sg = boundarySig(); return !!sg && overrides.has(sg); }

  // ---------------------------------------------------------------- settings
  const ROAD_CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "service", "footway", "path", "cycleway", "default"];
  const DEFAULTS = {
    floor_height_m: 3.2, default_levels: 3, tree_height_m: 8, broadleaf_height: 12, conifer_height: 15, broadleaf_crown_ratio: 0.65, conifer_crown_ratio: 0.35, use_genus_table: true, lod2_allow_large: false, forest_scatter: false, tree_scatter: { mangrove: true, orchard: true, plant_nursery: true, park: true }, tree_mode: "full3d", tree_lib: "", dem_source: "auto", dem_file: "", ground_mode: "auto", hillshade: false, contour_type: "mesh", plan_style: "colour", contours: true, contour_interval: null,
    include: { terrain: true, buildings: true, overture_buildings: true, lod2: true, lod2_roofs: true, roads: true, trees: true, water: true, shrubs: true, railways: true, walls: true, building_parts: true, ground: true, piers: true, power: false },
    road_widths: { motorway: 12, trunk: 10, primary: 8, secondary: 7, tertiary: 6, residential: 5, service: 3.5, footway: 2, path: 2, cycleway: 2, default: 4 },
  };
  const clone = (o) => JSON.parse(JSON.stringify(o));
  let settings = clone(DEFAULTS);
  try {
    const raw = localStorage.getItem("osmctx.settings");
    if (raw) {
      const s = JSON.parse(raw);
      settings = Object.assign(clone(DEFAULTS), s);
      settings.include = Object.assign({}, DEFAULTS.include, s.include || {});
      settings.tree_scatter = Object.assign({}, DEFAULTS.tree_scatter, s.tree_scatter || {});
      settings.road_widths = Object.assign({}, DEFAULTS.road_widths, s.road_widths || {});
    }
  } catch (e) { /* storage unavailable */ }
  function saveSettings() { try { localStorage.setItem("osmctx.settings", JSON.stringify(settings)); } catch (e) { /* ignore */ } }

  function buildSettingsUI() {
    const rw = $("roadWidths");
    rw.innerHTML = "";
    ROAD_CLASSES.forEach((c) => {
      const l = document.createElement("label");
      l.textContent = c === "footway" ? "foot/path/cycle" : c;
      const i = document.createElement("input");
      i.type = "number"; i.step = "0.5"; i.min = "0.5"; i.id = "rw_" + c;
      i.addEventListener("change", () => { const v = parseFloat(i.value); if (v > 0) { settings.road_widths[c] = v; if (c === "footway") { settings.road_widths.path = v; settings.road_widths.cycleway = v; $("rw_path").value = v; $("rw_cycleway").value = v; } saveSettings(); } });
      l.appendChild(i); rw.appendChild(l);
    });
    [["s_floor", "floor_height_m"], ["s_levels", "default_levels"], ["s_tree", "tree_height_m"], ["s_bh", "broadleaf_height"], ["s_ch", "conifer_height"], ["s_bcr", "broadleaf_crown_ratio"], ["s_ccr", "conifer_crown_ratio"]].forEach(([id, key]) => {
      $(id).addEventListener("change", () => { const v = parseFloat($(id).value); if (v > 0) { settings[key] = v; saveSettings(); } });
    });
    [["s_dem", "dem_source"], ["s_ground", "ground_mode"]].forEach(([id, key]) => {
      $(id).addEventListener("change", () => { settings[key] = $(id).value; saveSettings(); $("demFileRow").classList.toggle("hidden", settings.dem_source !== "file"); });
    });
    $("s_treemode").addEventListener("change", () => { settings.tree_mode = $("s_treemode").value; saveSettings(); $("treeLibRow").classList.toggle("hidden", settings.tree_mode !== "lib"); });
    const syncTerrainSub = () => { $("s_tsub_row").style.display = ["morph", "gdl"].includes(settings.contour_type) ? "" : "none"; };
    $("s_tsub").addEventListener("change", () => { settings.contour_type = $("s_tsub").value; syncTerrainSub(); saveSettings(); });
    $("s_pstyle").addEventListener("change", () => { settings.plan_style = $("s_pstyle").value; saveSettings(); });
    $("s_ctype").addEventListener("change", () => { settings.contour_type = $("s_ctype").value === "morph" ? $("s_tsub").value : $("s_ctype").value; syncTerrainSub(); $("s_cint").disabled = !settings.contours && !["stepped", "morph", "gdl"].includes(settings.contour_type); saveSettings(); });
    $("s_contours").addEventListener("change", () => { settings.contours = $("s_contours").checked; $("s_cint").disabled = !settings.contours && !["stepped", "morph", "gdl"].includes(settings.contour_type); saveSettings(); });
    $("s_hillshade").addEventListener("change", () => { settings.hillshade = $("s_hillshade").checked; saveSettings(); updateHillshadeOverlay(); });
    $("s_cint").addEventListener("change", () => { const v = parseFloat($("s_cint").value); settings.contour_interval = v > 0 ? v : null; saveSettings(); });
    $("s_genus").addEventListener("change", () => { settings.use_genus_table = $("s_genus").checked; saveSettings(); });
    $("s_forest").addEventListener("change", () => { settings.forest_scatter = $("s_forest").checked; saveSettings(); });
    Object.keys(DEFAULTS.tree_scatter).forEach((k) => $("s_ts_" + k).addEventListener("change", () => { settings.tree_scatter[k] = $("s_ts_" + k).checked; saveSettings(); }));
    $("s_treelib").addEventListener("change", () => { settings.tree_lib = $("s_treelib").value.trim(); saveSettings(); });
    $("s_demfile").addEventListener("change", () => { settings.dem_file = $("s_demfile").value.trim(); saveSettings(); });
    Object.keys(DEFAULTS.include).forEach((k) => $("i_" + k).addEventListener("change", () => { settings.include[k] = $("i_" + k).checked; saveSettings(); }));
    $("s_lod2_large").addEventListener("change", () => { settings.lod2_allow_large = $("s_lod2_large").checked; saveSettings(); });
    $("resetSettings").addEventListener("click", () => { settings = clone(DEFAULTS); saveSettings(); fillSettingsUI(); });
    fillSettingsUI();
  }
  function fillSettingsUI() {
    $("s_floor").value = settings.floor_height_m; $("s_levels").value = settings.default_levels; $("s_tree").value = settings.tree_height_m;
    $("s_bh").value = settings.broadleaf_height; $("s_ch").value = settings.conifer_height;
    $("s_bcr").value = settings.broadleaf_crown_ratio; $("s_ccr").value = settings.conifer_crown_ratio; $("s_genus").checked = !!settings.use_genus_table;
    $("s_forest").checked = !!settings.forest_scatter;
    { const ct = settings.contour_type || "mesh"; const sub = ["morph", "gdl"].includes(ct);
      $("s_ctype").value = sub ? "morph" : ct; if (sub && [...$("s_tsub").options].some(o => o.value === ct)) $("s_tsub").value = ct;
      $("s_tsub_row").style.display = sub ? "" : "none"; }
    $("s_pstyle").value = settings.plan_style || "colour";
    $("s_contours").checked = settings.contours !== false; $("s_cint").value = settings.contour_interval ? String(settings.contour_interval) : ""; $("s_cint").disabled = settings.contours === false && !["stepped", "morph", "gdl"].includes(settings.contour_type);
    $("s_hillshade").checked = !!settings.hillshade;
    Object.keys(DEFAULTS.tree_scatter).forEach((k) => { $("s_ts_" + k).checked = !!settings.tree_scatter[k]; });
    $("s_dem").value = settings.dem_source; $("s_ground").value = settings.ground_mode; $("s_demfile").value = settings.dem_file;
    $("demFileRow").classList.toggle("hidden", settings.dem_source !== "file");
    $("s_treemode").value = settings.tree_mode; $("s_treelib").value = settings.tree_lib;
    $("treeLibRow").classList.toggle("hidden", settings.tree_mode !== "lib");
    Object.keys(DEFAULTS.include).forEach((k) => { $("i_" + k).checked = !!settings.include[k]; });
    $("s_lod2_large").checked = !!settings.lod2_allow_large;
    ROAD_CLASSES.forEach((c) => { $("rw_" + c).value = settings.road_widths[c]; });
  }

  // ---------------------------------------------------------------- map
  const map = L.map("map", { zoomControl: true }).setView([48.5193, 9.0565], 16);
  const osm = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "&copy; OpenStreetMap contributors" });
  const sat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 19, attribution: "Tiles &copy; Esri, Maxar, Earthstar Geographics" });
  osm.addTo(map);
  L.control.layers({ "OpenStreetMap": osm, "Satellite (Esri)": sat }, null, { position: "topright" }).addTo(map);
  L.control.scale({ metric: true, imperial: false }).addTo(map);

  // Geoman default toolbar is not added (and hidden by CSS); the #toolbox strip drives map.pm.enableDraw().
  const STYLE = { color: "#2b6cd4", weight: 2, fillColor: "#2b6cd4", fillOpacity: 0.15 };
  map.pm.setGlobalOptions({ snappable: false, allowSelfIntersection: true, pathOptions: STYLE });

  // ---------------------------------------------------------------- geometry helpers (local metres)
  function proj(ref) {
    const k = Math.cos(ref.lat * Math.PI / 180);
    return {
      to: (ll) => [(ll.lng - ref.lng) * Math.PI / 180 * R_EARTH * k, (ll.lat - ref.lat) * Math.PI / 180 * R_EARTH],
      from: (p) => L.latLng(ref.lat + p[1] / R_EARTH * 180 / Math.PI, ref.lng + p[0] / (R_EARTH * k) * 180 / Math.PI),
    };
  }
  function ringArea(pts) { let a = 0; for (let i = 0; i < pts.length; i++) { const p = pts[i], q = pts[(i + 1) % pts.length]; a += p[0] * q[1] - q[0] * p[1]; } return Math.abs(a) / 2; }
  function pathLen(pts, closed) { let d = 0; const n = pts.length; for (let i = 0; i < n - (closed ? 0 : 1); i++) { const p = pts[i], q = pts[(i + 1) % n]; d += Math.hypot(q[0] - p[0], q[1] - p[1]); } return d; }
  const r6 = (v) => Math.round(v * 1e6) / 1e6;

  // ---------------------------------------------------------------- boundary state
  let current = null; // {layer, type}
  let corridor = L.layerGroup().addTo(map);
  let marker = null;

  function shapeOf(e) { return { Rectangle: "rectangle", Circle: "circle", Polygon: "polygon", Line: "polyline" }[e.shape]; }

  function setBoundary(layer, type) {
    if (current && current.layer !== layer) { try { current.layer.pm.disable(); } catch (e) { /* ignore */ } map.removeLayer(current.layer); }
    corridor.clearLayers();
    current = { layer, type };
    if (!map.hasLayer(layer)) layer.addTo(map);
    layer.setStyle(STYLE);
    layer.pm.enable({ allowSelfIntersection: true, snappable: false });
    ["pm:edit", "pm:markerdrag", "pm:dragend", "pm:drag", "pm:vertexadded", "pm:vertexremoved", "pm:centerplaced", "pm:update"].forEach((ev) => layer.on(ev, refresh));
    $("corridor").classList.toggle("hidden", type !== "polyline");
    refresh();
  }

  function clearBoundary() {
    if (current) { try { current.layer.pm.disable(); } catch (e) { /* ignore */ } map.removeLayer(current.layer); }
    current = null; corridor.clearLayers(); $("corridor").classList.add("hidden"); refresh();
  }

  // ---------------------------------------------------------------- toolbox strip
  const TOOLS = { tRect: "Rectangle", tCircle: "Circle", tPoly: "Polygon", tLine: "Line" };
  function toolHighlight(activeId) { ["tArrow"].concat(Object.keys(TOOLS)).forEach((id) => $(id).classList.toggle("on", id === activeId)); }
  function pickTool(id) {
    map.pm.disableDraw();
    if (TOOLS[id]) { map.pm.enableDraw(TOOLS[id], { pathOptions: STYLE, snappable: false, allowSelfIntersection: true }); toolHighlight(id); }
    else toolHighlight("tArrow");
  }
  Object.keys(TOOLS).forEach((id) => $(id).addEventListener("click", () => pickTool(id)));
  $("tArrow").addEventListener("click", () => pickTool("tArrow"));
  $("tDelete").addEventListener("click", () => { map.pm.disableDraw(); clearBoundary(); toolHighlight("tArrow"); });
  map.on("pm:drawend", () => toolHighlight("tArrow"));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !/INPUT|SELECT|TEXTAREA/.test((e.target || {}).tagName || "")) pickTool("tArrow"); });
  toolHighlight("tArrow");

  // ---------------------------------------------------------------- theme (light default, follows OS unless toggled)
  function currentTheme() {
    const t = document.documentElement.getAttribute("data-theme");
    return t || (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  }
  $("themeBtn").addEventListener("click", () => {
    const n = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", n);
    try { localStorage.setItem("osmctx.theme", n); } catch (e) { /* ignore */ }
  });

  map.on("pm:create", (e) => {
    const type = shapeOf(e);
    if (!type) return;
    setBoundary(e.layer, type);
  });

  function bufferM() { const v = parseFloat($("buf").value); return v > 0 ? v : 10; }

  function currentBoundary() {
    if (!current) return null;
    const { layer, type } = current;
    if (type === "circle") {
      const c = layer.getLatLng();
      return { type, coordinates: [[r6(c.lng), r6(c.lat)]], center: [r6(c.lng), r6(c.lat)], radius_m: Math.round(layer.getRadius() * 100) / 100 };
    }
    let ll = layer.getLatLngs();
    if (type !== "polyline") {
      ll = Array.isArray(ll[0]) ? ll[0] : ll;
      const pts = ll.map((p) => [r6(p.lng), r6(p.lat)]);
      if (pts.length && (pts[0][0] !== pts[pts.length - 1][0] || pts[0][1] !== pts[pts.length - 1][1])) pts.push(pts[0].slice());
      return { type, coordinates: pts };
    }
    return { type, coordinates: ll.map((p) => [r6(p.lng), r6(p.lat)]), buffer_m: bufferM() };
  }

  function metrics() {
    if (!current) return null;
    const { layer, type } = current;
    if (type === "circle") {
      const r = layer.getRadius(), c = layer.getLatLng();
      return { area: Math.PI * r * r, perim: 2 * Math.PI * r, centre: c };
    }
    let ll = layer.getLatLngs();
    if (type !== "polyline") ll = Array.isArray(ll[0]) ? ll[0] : ll;
    const b = L.latLngBounds(ll), ref = b.getCenter(), P = proj(ref);
    const pts = ll.map(P.to);
    if (type === "polyline") {
      const len = pathLen(pts, false), w = bufferM();
      return { area: len * 2 * w + Math.PI * w * w, perim: len, centre: ref, note: "corridor area approx." };
    }
    return { area: ringArea(pts), perim: pathLen(pts, true), centre: ref };
  }

  function drawCorridor() {
    corridor.clearLayers();
    if (!current || current.type !== "polyline") return;
    const ll = current.layer.getLatLngs();
    if (ll.length < 1) return;
    const w = bufferM(), P = proj(L.latLngBounds(ll).getCenter());
    const st = { stroke: false, fillColor: "#3b9eff", fillOpacity: 0.18, interactive: false };
    ll.forEach((p) => L.circle(p, Object.assign({ radius: w }, st)).addTo(corridor));
    for (let i = 0; i < ll.length - 1; i++) {
      const a = P.to(ll[i]), b = P.to(ll[i + 1]), dx = b[0] - a[0], dy = b[1] - a[1], d = Math.hypot(dx, dy);
      if (d < 1e-6) continue;
      const nx = -dy / d * w, ny = dx / d * w;
      L.polygon([[a[0] + nx, a[1] + ny], [b[0] + nx, b[1] + ny], [b[0] - nx, b[1] - ny], [a[0] - nx, a[1] - ny]].map(P.from), Object.assign({}, st)).addTo(corridor);
    }
  }

  const fmt = (v, d) => v.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
  let refresh = function () {
    const b = currentBoundary(), m = metrics();
    drawCorridor();
    if (!b) {
      ["iShape", "iArea", "iPerim", "iCentre"].forEach((i) => { $(i).textContent = "-"; });
      $("json").textContent = "(no boundary)";
    } else {
      $("iShape").textContent = b.type;
      $("iArea").textContent = fmt(m.area, 0) + " m² (" + fmt(m.area / 1e4, 3) + " ha)" + (m.note ? " *" : "");
      $("iPerim").textContent = fmt(m.perim, 1) + " m";
      $("iCentre").textContent = m.centre.lat.toFixed(5) + ", " + m.centre.lng.toFixed(5);
      $("json").textContent = JSON.stringify(b);
    }
    const km2 = m ? m.area / 1e6 : null, tier = tierOf(km2);
    const fk = (v) => (v < 10 ? v.toFixed(2) : v.toFixed(1));
    let msg = "";
    if (tier === "blocked") msg = "Area is " + fk(km2) + " km² (over " + limits.block_above_km2 + " km²). Fetch is blocked (public Overpass and Archicad limits). Split it into smaller areas and import them one by one.";
    else if (tier === "large") msg = "Large area (" + fk(km2) + " km²) - this will take a while. Fetch will show an estimate and ask you to confirm.";
    else if (tier === "super_light") msg = "Very large area (" + fk(km2) + " km²) - this will take a long time. Fetch will show an estimate and ask you to confirm.";
    $("warn").textContent = msg;
    $("warn").classList.toggle("hidden", !msg);
    $("warn").classList.toggle("blocked", tier === "blocked");
    const sl = tier === "super_light", ov = sl && overrideOn();
    $("slBadge").classList.toggle("hidden", !(sl && !ov));
    $("slOff").classList.toggle("hidden", !ov);
    $("fetch").disabled = !b || tier === "blocked";
    $("count").disabled = !b;
    if (!b || (tier !== "large" && tier !== "super_light")) hideBigConfirm();
  };

  $("buf").addEventListener("input", refresh);
  $("clear").addEventListener("click", clearBoundary);
  $("copy").addEventListener("click", async () => {
    const b = currentBoundary(); if (!b) return;
    const txt = JSON.stringify(b, null, 2);
    try { await navigator.clipboard.writeText(txt); status("Boundary JSON copied.", "ok"); }
    catch (e) { const r = document.createRange(); r.selectNodeContents($("json")); const s = getSelection(); s.removeAllRanges(); s.addRange(r); status("Select-copy the JSON manually.", ""); }
  });

  $("sq100").addEventListener("click", () => {
    const c = map.getCenter(), h = 50;
    const dlat = h / R_EARTH * 180 / Math.PI, dlon = dlat / Math.cos(c.lat * Math.PI / 180);
    const rect = L.rectangle([[c.lat - dlat, c.lng - dlon], [c.lat + dlat, c.lng + dlon]], STYLE);
    setBoundary(rect, "rectangle");
    map.fitBounds(rect.getBounds().pad(1.5));
  });

  // ---------------------------------------------------------------- search
  function flyTo(lat, lon, zoom, bbox) {
    if (marker) map.removeLayer(marker);
    marker = L.marker([lat, lon]).addTo(map);
    if (bbox) map.flyToBounds([[bbox[0], bbox[2]], [bbox[1], bbox[3]]], { maxZoom: 18 });
    else map.flyTo([lat, lon], zoom || 17);
  }
  function searchMsg(t, cls) { $("searchMsg").textContent = t; $("searchMsg").className = "msg " + (cls || ""); }

  async function doSearch() {
    const q = $("q").value.trim();
    $("results").innerHTML = ""; searchMsg("");
    if (!q) return;
    const c = window.parseCoords(q);
    if (c) { flyTo(c.lat, c.lon, 17); searchMsg("Coordinates: " + c.lat.toFixed(5) + ", " + c.lon.toFixed(5), "ok"); return; }
    searchMsg("Searching...");
    try {
      const r = await fetch("/api/geocode?q=" + encodeURIComponent(q));
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || r.status);
      if (!d.length) { searchMsg("No results."); return; }
      searchMsg("");
      const seen = new Set();  // Nominatim can return the same place twice (e.g. node + relation): dedupe by rounded lat/lon + name
      d.filter((it) => {
        const k = it.lat.toFixed(5) + "|" + it.lon.toFixed(5) + "|" + (it.name || "");
        if (seen.has(k)) return false;
        seen.add(k);
        return true;
      }).forEach((it) => {
        const li = document.createElement("li");
        li.textContent = it.name;
        li.addEventListener("click", () => { flyTo(it.lat, it.lon, 17, it.bbox); $("results").innerHTML = ""; });
        $("results").appendChild(li);
      });
    } catch (e) { searchMsg("Search failed: " + e.message, "err"); }
  }
  $("go").addEventListener("click", doSearch);
  $("q").addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });

  // ---------------------------------------------------------------- count / fetch / convert
  function jobLine(t, cls) { $("jobStatus").textContent = t || "idle"; $("jobCell").className = "cell job grow " + (cls || ""); }
  function status(t, cls) { $("status").textContent = t; $("status").className = "msg " + (cls || ""); jobLine(t, cls); }
  async function post(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    let d = {}; try { d = await r.json(); } catch (e) { /* non-json */ }
    if (!r.ok) throw new Error(d.error || ("HTTP " + r.status));
    return d;
  }
  const esc = (t) => String(t).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

  // accuracy disclaimer + credits: text is built server-side from report.dem_accuracy (GLO-30 defaults before fetch)
  fetch("/api/disclaimer").then((r) => r.json()).then((d) => {
    $("disclaimer").textContent = d.text;
    $("credits").innerHTML = d.credits.map((c) => "<div>" + esc(c) + "</div>").join("");
  }).catch(() => { $("disclaimer").textContent = "Site context only, not a survey. Map data © OpenStreetMap contributors."; });

  $("count").addEventListener("click", async () => {
    const b = currentBoundary(); if (!b) return;
    $("count").disabled = true; status("Counting...");
    try {
      const d = await post("/api/count", { boundary: b, settings });
      $("cB").textContent = d.buildings; $("cR").textContent = d.roads; $("cT").textContent = d.trees; $("cW").textContent = d.water;
      $("counts").classList.remove("hidden");
      const via = d.source === "cache" ? "from fetch cache" : "Overpass " + (d.overpass_url || "").replace(/^https?:\/\/([^/]+).*/, "$1");
      status("Counted over boundary bbox (" + via + ", " + d.seconds + " s).", "ok");
    } catch (e) { status("Count failed: " + e.message, "err"); }
    $("count").disabled = !current;
  });

  // ---- converted-site overlay on the map
  let siteLayer = null, hillshadeLayer = null, lastDemBounds = null, lastCache = null;
  function updateHillshadeOverlay() {
    if (hillshadeLayer) { map.removeLayer(hillshadeLayer); hillshadeLayer = null; }
    if (!settings.hillshade || !lastDemBounds || lastDemBounds.length !== 4) return;
    const [s, w, n, e] = lastDemBounds;
    const url = "/api/hillshade.png?t=" + Date.now() + (lastCache ? "&cache=" + encodeURIComponent(lastCache) : "");
    hillshadeLayer = L.imageOverlay(url, [[s, w], [n, e]], { opacity: 0.48, interactive: false }).addTo(map);
  }
  const roadCol = (w) => (w >= 8 ? "#ff7043" : w >= 5 ? "#f0a030" : w >= 3 ? "#e0c060" : "#c9b88a");
  async function showSite() {
    const r = await fetch("/api/site_geojson");
    const g = await r.json();
    if (!r.ok) throw new Error(g.error || r.status);
    if (siteLayer) map.removeLayer(siteLayer);
    siteLayer = L.geoJSON(g, {
      filter: (f) => f.properties.kind !== "boundary",
      style: (f) => {
        const p = f.properties;
        if (p.kind === "building") return { color: "#6b7178", weight: 1, fillColor: "#9aa0a6", fillOpacity: 0.75 };
        if (p.kind === "water") return { color: "#2a7fd6", weight: 1, fillColor: "#3b9eff", fillOpacity: 0.55 };
        if (p.kind === "railway") return { color: "#7a3f98", weight: 3, opacity: 0.9, dashArray: p.bridge ? "6 4" : "2 5" };
        if (p.kind === "barrier") return { color: "#8a5a2b", weight: 2, opacity: 0.9 };
        if (p.kind === "ground" || p.kind === "platform" || p.kind === "pier") return { color: "#9a7b3b", weight: 1, fillColor: "#d9c27a", fillOpacity: 0.5 };
        return { color: roadCol(p.width || 4), weight: Math.max(2, Math.min(10, p.width || 4)), opacity: 0.9, dashArray: p.bridge ? "6 4" : null };
      },
      pointToLayer: (f, ll) => L.circleMarker(ll, { radius: 3, color: "#2f9e6b", weight: 1, fillColor: "#4cc38a", fillOpacity: 0.9 }),
      onEachFeature: (f, l) => {
        const p = f.properties;
        l.bindTooltip(p.kind + (p.id ? " " + p.id : "") + (p.height ? " h=" + p.height + " m" : "") + (p.width ? " w=" + p.width + " m" : ""), { sticky: true });
      },
    }).addTo(map);
    return g;
  }

  // extra accuracy-panel lines from the convert result: slope note, peak DEM-vs-ele mismatch, stale OSM, flagged buildings
  function showAccuracyNotes(notes) {
    const el = $("accNotes");
    el.innerHTML = notes.map((n) => "<div class='" + (n.level === "warn" ? "approx" : "hint") + "' data-kind='" + esc(n.kind) + "'>" + esc(n.text) + "</div>").join("");
    el.classList.toggle("hidden", !notes.length);
  }

  function showReport(d) {
    const rep = d.report, c = rep.counts || {}, acc = rep.dem_accuracy || {}, dem = (rep.info || {}).dem || {};
    const dflt = rep.defaults_used || {};
    const n = (v) => (v == null ? 0 : v);
    const l2 = (rep.info || {}).lod2 || {};
    // building sources: "N OSM + M Overture + K LoD2 (R replaced)" when LoD2 was used, else the Overture split
    const bsrc = l2.applied && l2.added ? " (" + esc(l2.summary) + (c.buildings_overture ? "; Overture estimated" : "") + ")"
      : (c.buildings_overture ? " (" + n(c.buildings_osm) + " OSM + " + c.buildings_overture + " Overture Maps, estimated)" : "");
    $("repSummary").innerHTML =
      "<b>" + n(c.buildings) + "</b> buildings" + bsrc + ", <b>" + n(c.roads) + "</b> roads, <b>" + n(c.trees) + "</b> trees, <b>" + n(c.water) + "</b> water<br>" +
      "Terrain: <b>" + n(c.terrain_points) + "</b> points; ground " + esc(acc.ground_mode || "?") + "; source " + esc(dem.source || acc.source || "?") + "<br>" +
      "Defaults used: heights " + esc(JSON.stringify(dflt.heights || {})) + "; warnings <b>" + (rep.warnings || []).length + "</b>" +
      (acc.vertical_m != null ? "<br>DEM vertical accuracy: ±" + acc.vertical_m + " m (" + esc(acc.kind || "") + ")" : "");
    const apx = [];
    if ((acc.ground_mode || "") === "bare_earth_approx" || /approximat/i.test(acc.note || "")) apx.push("Ground is an <b>APPROXIMATION</b> (" + esc(acc.ground_mode || "") + "): " + esc(acc.note || ""));
    (rep.warnings || []).filter((w) => /approx/i.test(w)).forEach((w) => apx.push("<b>APPROXIMATION</b>: " + esc(w)));
    $("repApprox").innerHTML = apx.join("<br>"); $("repApprox").classList.toggle("hidden", !apx.length);
    showAccuracyNotes(d.accuracy_notes || []);
    if (d.contours) $("s_cint").options[0].textContent = "Auto (" + d.contours.auto_interval + " m)";
    if (d.contours) $("cintAuto").textContent = "Auto interval for this site: " + d.contours.auto_interval + " m (relief " + d.contours.relief_m + " m, about " + d.contours.est_lines + " lines). Major = every 5th, labelled. Drawn as 2D lines on every import.";
    $("repDisc").innerHTML = esc(d.disclaimer.text) + d.disclaimer.credits.map((x) => "<div class='credits'>" + esc(x) + "</div>").join("");
    $("repDetail").textContent = JSON.stringify({ defaults_used: dflt, warnings: rep.warnings, dem_accuracy: acc }, null, 1);
    $("reportSec").classList.remove("hidden");
  }

  // ---- large-area confirm (inline, with estimate from /api/estimate)
  function hideBigConfirm() { $("bigConfirm").classList.add("hidden"); }
  let lastEstimate = null;
  async function showBigConfirm(b) {
    $("bigConfirm").classList.remove("hidden");
    $("bigText").textContent = "Estimating...";
    $("bigGo").disabled = true;
    const sigAtStart = boundarySig();
    try {
      const d = await post("/api/estimate", { boundary: b, settings, override: overrideOn() });
      if (sigAtStart !== boundarySig()) return;
      lastEstimate = d;
      const el = d.elements || {};
      let t = "Large area (" + d.area_km2 + " km²) - this will take a while. Estimated " + d.text + ", about " + el.total + " Archicad elements (" +
        el.buildings + " buildings, " + el.road_beams + " road beams, " + el.trees + " trees" + (el.scatter_trees ? " (" + el.scatter_trees + " estimated, scattered in mapped areas)" : "") + ", " + el.water + " water" + (el.shrubs || el.hedge_segments ? ", " + (el.shrubs || 0) + " shrubs, " + (el.hedge_segments || 0) + " hedge segments" : "") + ").";
      if (d.preset) t += " Super-light preset will be used (2D trees, no shrubs/hedges/tree scatter, coarse terrain, no footways/paths/cycleways/steps/service roads; railways + walls kept, building parts/roofs, ground surfaces, piers and power off).";
      if (d.counts_source === "area") t += " (Rough guess from area only: element counts unavailable.)";
      else if (d.counts_source) t += " (Counts from " + d.counts_source + ".)";
      const nn = (d.notes || []).filter((n) => !/area only/.test(n));
      if (nn.length) t += " " + nn.join("; ") + ".";
      $("bigText").textContent = t;
      $("bigGo").disabled = false;
    } catch (e) { $("bigText").textContent = "Could not estimate (" + e.message + "). You can still continue."; $("bigGo").disabled = false; }
  }
  $("bigCancel").addEventListener("click", () => { hideBigConfirm(); status("Cancelled."); });
  $("bigGo").addEventListener("click", () => { hideBigConfirm(); runFetch(); });
  $("slOffBtn").addEventListener("click", (e) => { e.preventDefault(); const sg = boundarySig(); if (sg) { overrides.add(sg); saveOverrides(); } refresh(); });
  $("slOnBtn").addEventListener("click", (e) => { e.preventDefault(); const sg = boundarySig(); if (sg) { overrides.delete(sg); saveOverrides(); } refresh(); });

  let converted = false;
  $("fetch").addEventListener("click", () => {
    const b = currentBoundary(); if (!b) return;
    const m = metrics(), tier = tierOf(m ? m.area / 1e6 : null);
    if (tier === "large" || tier === "super_light") showBigConfirm(b); else runFetch();
  });
  async function runFetch() {
    const b = currentBoundary(); if (!b) return;
    $("fetch").disabled = true; $("import").disabled = true; converted = false;
    try {
      status("Fetching OSM + DEM...");
      const ov = overrideOn();
      const j = await post("/api/fetch", { boundary: b, settings, override: ov });
      let f;
      for (;;) {
        const r = await fetch("/api/job/" + j.job); const d = await r.json();
        if (!r.ok) throw new Error(d.error || r.status);
        if (d.state === "error") throw new Error(d.error);
        if (d.state === "done") { f = d; break; }
        status(d.progress || "Fetching...");
        await new Promise((res) => setTimeout(res, 1000));
      }
      status("Converting...");
      const d = await post("/api/convert", { cache: f.cache, boundary: b, settings, override: ov });
      lastDemBounds = d.dem_bounds || null;
      lastCache = d.cache || f.cache || null;
      updateHillshadeOverlay();
      showReport(d);
      if (d.preset) $("repSummary").innerHTML += "<br><b>Super-light preset applied</b> (" + d.area_km2 + " km²)";
      await showSite();
      converted = true;
      status("Fetched + converted. Cache: " + f.cache, "ok");
    } catch (e) { status("Failed: " + e.message, "err"); }
    refresh();
    $("import").disabled = !converted;
  }

  // ---------------------------------------------------------------- sun
  const PRESETS = [["Jun 21", "2026-06-21"], ["Dec 21", "2026-12-21"], ["Mar 20", "2026-03-20"]];
  const TIMES = ["09:00", "12:00", "15:00"];
  function sunPlace() { const m = metrics(); const c = m ? m.centre : map.getCenter(); return { lat: c.lat, lon: c.lng }; }
  let sunReq = 0;
  async function sunUpdate() {
    const id = ++sunReq, p = sunPlace();
    const q = new URLSearchParams({ lat: p.lat.toFixed(6), lon: p.lon.toFixed(6), date: $("sunDate").value, time: $("sunTime").value });
    document.querySelectorAll("#sunPresets button").forEach((b) => b.classList.toggle("on", b.dataset.d === $("sunDate").value && b.dataset.t === $("sunTime").value));
    try {
      const r = await fetch("/api/sun_preview?" + q);
      const d = await r.json();
      if (id !== sunReq) return;
      if (!r.ok) throw new Error(d.error || r.status);
      $("sunAz").textContent = d.compass_azimuth.toFixed(1) + "°";
      $("sunAlt").textContent = d.altitude.toFixed(1) + "°" + (d.altitude < 0 ? " (below horizon)" : "");
      $("sunAc").textContent = d.archicad_azimuth.toFixed(1) + "°";
      $("sunTz").textContent = d.time_zone + ", UTC" + (d.timeZoneInMinutes >= 0 ? "+" : "-") + (Math.abs(d.timeZoneInMinutes) / 60) + (d.sumTime ? " + summer time" : "");
    } catch (e) { $("sunAz").textContent = "error: " + e.message; }
  }
  PRESETS.forEach(([nm, d]) => TIMES.forEach((t) => {
    const b = document.createElement("button");
    b.textContent = nm + " " + t; b.dataset.d = d; b.dataset.t = t;
    b.addEventListener("click", () => { $("sunDate").value = d; $("sunTime").value = t; sunUpdate(); });
    $("sunPresets").appendChild(b);
  }));
  $("sunDate").addEventListener("change", sunUpdate);
  $("sunTime").addEventListener("change", sunUpdate);
  map.on("moveend", () => { if (!current) sunUpdate(); });
  const baseRefresh = refresh;
  refresh = function () { baseRefresh(); sunUpdate(); };

  // ---------------------------------------------------------------- target layer (READ from Archicad)
  // select value: "one:<layer name>" | "per_type"
  let layerList = [], layerDefault = null;
  function layerPref() { try { return localStorage.getItem("osmctx.layer") || ""; } catch (e) { return ""; } }
  function layerSave(v) { try { localStorage.setItem("osmctx.layer", v); } catch (e) { /* ignore */ } }
  function layerChoice() {
    const v = $("layerSel").value || "";
    return v === "per_type" ? { layer_mode: "per_type" } : v.indexOf("one:") === 0 ? { layer_mode: "single", target_layer: v.slice(4) } : { layer_mode: "single" };
  }
  function layerWarn() {
    const c = layerChoice(), w = $("layerWarn"), l = layerList.find((x) => x.name === c.target_layer);
    let t = "";
    if (c.layer_mode === "single" && l && l.locked) t = "Layer \"" + l.name + "\" is locked in the current layer combination: the import will be refused. Unlock it in Archicad or pick another layer.";
    else if (c.layer_mode === "single" && l && l.hidden) t = "Layer \"" + l.name + "\" is hidden in the current layer combination: the import will work but you will not see it until you show the layer.";
    w.textContent = t; w.classList.toggle("hidden", !t);
  }
  async function layersLoad() {
    const sel = $("layerSel"), keep = layerPref() || sel.value;
    sel.innerHTML = ""; $("layerWarn").classList.add("hidden");
    try {
      const d = await (await fetch("/api/layers")).json();
      if (d.error) throw new Error(d.error);
      layerList = d.layers || []; layerDefault = d.default;
      const one = (n, label) => { const o = document.createElement("option"); o.value = "one:" + n; o.textContent = label; sel.appendChild(o); };
      if (layerDefault) one(layerDefault, "One layer: " + layerDefault + " (default)");
      layerList.forEach((l) => { if (l.name !== layerDefault) one(l.name, "One layer: " + l.name + (l.locked ? " [locked]" : l.hidden ? " [hidden]" : "")); });
      const pt = document.createElement("option"); pt.value = "per_type"; pt.textContent = "Separate layers per type (Site-*)"; sel.appendChild(pt);
      if (Array.from(sel.options).some((o) => o.value === keep)) sel.value = keep;
    } catch (e) {
      layerList = []; layerDefault = null;
      const o = document.createElement("option"); o.value = ""; o.textContent = "One layer (auto) - Archicad not reachable"; sel.appendChild(o);
      const pt = document.createElement("option"); pt.value = "per_type"; pt.textContent = "Separate layers per type (Site-*)"; sel.appendChild(pt);
      if (keep === "per_type") sel.value = keep;
      $("layerWarn").textContent = "Could not read layers: " + e.message; $("layerWarn").classList.remove("hidden");
      return;
    }
    layerWarn();
  }
  $("layerSel").addEventListener("change", () => { layerSave($("layerSel").value); layerWarn(); });
  $("layerRefresh").addEventListener("click", layersLoad);
  layersLoad();

  // ---------------------------------------------------------------- import (WRITES to Archicad)
  $("import").addEventListener("click", async () => {
    const big = lastEstimate && lastEstimate.tier && lastEstimate.tier !== "normal" && lastEstimate.tier !== "blocked";
    const pre = big ? "Large area (" + lastEstimate.area_km2 + " km²): the build can take " + lastEstimate.text + (lastEstimate.preset ? " (super-light preset)" : "") + ".\n\n" : "";
    if (!confirm(pre + "Write the converted site into the open Archicad project?")) return;
    $("import").disabled = true; $("importStatus").className = "msg"; $("importStatus").textContent = "Importing to Archicad (can take a while)..."; jobLine("Importing to Archicad...");
    try {
      const tm = settings.tree_mode === "lib" ? (settings.tree_lib ? "lib:" + settings.tree_lib : "full3d") : (settings.tree_mode || "full3d");
      const d = await post("/api/import", { sun: { date: $("sunDate").value, time: $("sunTime").value }, tree_mode: tm, contour_type: settings.contour_type || "mesh", plan_style: settings.plan_style || "colour", boundary: currentBoundary(), settings, override: overrideOn(), ...layerChoice() });
      $("importStatus").className = "msg ok";
      $("importStatus").textContent = "Import done." + (d.preset ? " Super-light preset used (2D trees)." : "") + (d.sun_error ? " Sun failed: " + d.sun_error : d.sun ? " Sun set." : "");
      const tw = ((d.build || {}).tree_mode || {}).warnings || [];
      const bw = (d.build || {}).warnings || [];
      if (bw.length) $("importStatus").textContent += " Warning: " + bw.join("; ");
      if ((d.build || {}).layer_note) $("importStatus").textContent += " " + d.build.layer_note;
      if (tw.length) $("importStatus").textContent += " Tree warning: " + tw.join("; ");
      jobLine($("importStatus").textContent, "ok");
    } catch (e) { $("importStatus").className = "msg err"; $("importStatus").textContent = "Import failed: " + e.message; jobLine("Import failed: " + e.message, "err"); }
    $("import").disabled = !converted;
  });

  buildSettingsUI();
  refresh();
  // Running inside the Archicad "OSM Context" palette? (JS bridge injected by the add-on)
  if (window.ACAPI && typeof window.ACAPI.GetVersion === "function") {
    window.ACAPI.GetVersion().then(function (v) { jobLine("Running inside Archicad" + (v ? " (" + v + ")" : ""), "ok"); },
      function () { jobLine("Running inside Archicad", "ok"); });
  }
})();
