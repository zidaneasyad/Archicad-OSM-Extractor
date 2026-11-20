/* Coordinate parser, mirrors server.py parse_coords. Returns {lat, lon} or null. */
(function (root) {
  const NUM = "[-+]?\\d+(?:\\.\\d+)?";
  const DEC = new RegExp("^\\s*(" + NUM + ")\\s*(?:[,;]\\s*|\\s+)(" + NUM + ")\\s*$");
  const DEC_H = new RegExp("^\\s*(" + NUM + ")\\s*°?\\s*([NSEW])?\\s*[,;\\s]\\s*(" + NUM + ")\\s*°?\\s*([NSEW])?\\s*$", "i");
  const DMS = /(?:([NSEW])\s*)?(\d+(?:\.\d+)?)\s*(?:°|º|˚|d|deg)\s*(?:(\d+(?:\.\d+)?)\s*(?:'|′|’|m|min)\s*)?(?:(\d+(?:\.\d+)?)\s*(?:"|″|”|''|′′|s|sec)\s*)?([NSEW])?/gi;

  function orient(a, b) {
    if (Math.abs(a) > 90 && Math.abs(b) <= 90) { const t = a; a = b; b = t; }
    if (Math.abs(a) <= 90 && Math.abs(b) <= 180) return { lat: a, lon: b };
    return null;
  }

  function parseCoords(text) {
    if (!text || typeof text !== "string") return null;
    let s = text.trim();
    try { s = decodeURIComponent(s); } catch (e) { /* keep raw */ }
    if (/^https?:\/\//i.test(s) || /google\.|goo\.gl/i.test(s)) {
      const pats = [new RegExp("@(" + NUM + "),(" + NUM + ")"), new RegExp("!3d(" + NUM + ")!4d(" + NUM + ")"),
        new RegExp("[?&](?:q|ll|query|center)=(" + NUM + ")[, +]+(" + NUM + ")")];
      for (const p of pats) { const m = s.match(p); if (m) return orient(parseFloat(m[1]), parseFloat(m[2])); }
      return null;
    }
    const hits = [];
    let m;
    DMS.lastIndex = 0;
    while ((m = DMS.exec(s)) !== null) {
      if (m[0] === "") { DMS.lastIndex++; continue; }
      const h = (m[5] || m[1] || "").toUpperCase();
      if (!h) continue;
      let v = parseFloat(m[2]) + (m[3] ? parseFloat(m[3]) / 60 : 0) + (m[4] ? parseFloat(m[4]) / 3600 : 0);
      if (h === "S" || h === "W") v = -v;
      hits.push([h, v]);
    }
    if (hits.length === 2) {
      const la = hits.find(x => "NS".includes(x[0])), lo = hits.find(x => "EW".includes(x[0]));
      if (la && lo && Math.abs(la[1]) <= 90 && Math.abs(lo[1]) <= 180) return { lat: la[1], lon: lo[1] };
      return null;
    }
    m = s.match(DEC_H);
    if (m && (m[2] || m[4])) {
      const v = {};
      [[parseFloat(m[1]), (m[2] || "").toUpperCase()], [parseFloat(m[3]), (m[4] || "").toUpperCase()]].forEach(([x, h]) => {
        if (h === "S" || h === "W") x = -Math.abs(x);
        if (h) v[h === "N" || h === "S" ? "lat" : "lon"] = x;
      });
      if ("lat" in v && "lon" in v && Math.abs(v.lat) <= 90 && Math.abs(v.lon) <= 180) return v;
    }
    m = s.match(DEC);
    if (m) return orient(parseFloat(m[1]), parseFloat(m[2]));
    return null;
  }
  root.parseCoords = parseCoords;
})(window);
