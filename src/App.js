import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";
import { Document, Packer, Paragraph, Table, TableRow, TableCell, TextRun, ImageRun, WidthType, AlignmentType, HeadingLevel } from "docx";
import { saveAs } from "file-saver";

/* ══════════════════════════════════════════════════════════════
   CONSTANTS & PHYSICS
   ══════════════════════════════════════════════════════════════ */
const DEG = Math.PI / 180, RAD = 180 / Math.PI;
const R_E = 6371, MU = 398600.4418, J2 = 1.08263e-3, OMEGA_E = 7.2921159e-5;

function orbPeriod(alt) { const a = R_E + alt; return 2 * Math.PI * Math.sqrt(a * a * a / MU); }
function nodalPrec(alt, inc, ecc = 0) {
  const a = R_E + alt, n = Math.sqrt(MU / (a * a * a)), p = a * (1 - ecc * ecc);
  return -1.5 * n * J2 * Math.pow(R_E / p, 2) * Math.cos(inc * DEG) * RAD * 86400;
}
function effSwathKm(alt, swHalf) {
  // swHalf is the sensor half-angle FOV; swath = 2 × alt × tan(halfAngle)
  return 2 * alt * Math.tan(swHalf * DEG);
}

/* Maximum straight-line (diagonal) distance across the country bounding box */
function countryDiagonal(cty) {
  const dLat = cty.latMax - cty.latMin;
  const dLon = cty.lonMax - cty.lonMin;
  const cosLat = Math.cos(((cty.latMin + cty.latMax) / 2) * DEG);
  return Math.sqrt(Math.pow(dLat * 111.32, 2) + Math.pow(dLon * 111.32 * cosLat, 2));
}

/* Haversine track length over a list of {lat, lon} points, capped at country diagonal.
   Uses ALL buffer-zone points in the detected pass — no strict-bounds clipping needed. */
function passTrackLength(passPoints, cty) {
  let totalKm = 0;
  for (let i = 1; i < passPoints.length; i++) {
    const dLatR = (passPoints[i].lat - passPoints[i - 1].lat) * DEG;
    const dLonR = (passPoints[i].lon - passPoints[i - 1].lon) * DEG;
    const a = Math.pow(Math.sin(dLatR / 2), 2) +
              Math.cos(passPoints[i - 1].lat * DEG) * Math.cos(passPoints[i].lat * DEG) *
              Math.pow(Math.sin(dLonR / 2), 2);
    totalKm += R_E * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }
  return Math.min(totalKm, countryDiagonal(cty));
}
function groundTrack(sat, numOrbits, ptsPerOrbit = 300) {
  const { altitude, inclination, eccentricity: e = 0, raan = 0, argPerigee = 0 } = sat;
  const a = R_E + altitude, T = orbPeriod(altitude), n = 2 * Math.PI / T;
  const omDot = nodalPrec(altitude, inclination, e) / 86400 * DEG;
  const total = numOrbits * ptsPerOrbit, tTotal = numOrbits * T, pts = [];
  for (let k = 0; k < total; k++) {
    const t = (k / total) * tTotal, M = n * t, nu = M + 2 * e * Math.sin(M);
    const r = a * (1 - e * e) / (1 + e * Math.cos(nu)), u = argPerigee * DEG + nu;
    const iR = inclination * DEG, Om = raan * DEG + omDot * t;
    const x = r * (Math.cos(Om) * Math.cos(u) - Math.sin(Om) * Math.sin(u) * Math.cos(iR));
    const y = r * (Math.sin(Om) * Math.cos(u) + Math.cos(Om) * Math.sin(u) * Math.cos(iR));
    const z = r * Math.sin(u) * Math.sin(iR);
    const lat = Math.asin(z / r) * RAD;
    // Proper longitude wrapping for arbitrary time spans
    let lon = Math.atan2(y, x) * RAD - OMEGA_E * t * RAD;
    lon = lon % 360;
    if (lon > 180) lon -= 360;
    if (lon < -180) lon += 360;
    pts.push({ lat, lon, t });
  }
  return pts;
}

/* ══════════════════════════════════════════════════════════════
   SUNLIT IMAGING WINDOW
   ══════════════════════════════════════════════════════════════ */
function isInImagingWindow(tSeconds, latDeg, lonDeg, dayOfYear = 80) {
  const decl = 23.44 * Math.sin((360 / 365) * (dayOfYear - 81) * DEG);
  const cosH = -Math.tan(latDeg * DEG) * Math.tan(decl * DEG);
  if (cosH > 1) return false;  // polar night
  if (cosH < -1) return true;  // midnight sun
  const halfDay = Math.acos(Math.max(-1, Math.min(1, cosH))) / (15 * DEG);
  const sunrise = 12 - halfDay;
  const sunset = 12 + halfDay;
  let lst = ((tSeconds / 3600) + lonDeg / 15) % 24;
  if (lst < 0) lst += 24;
  return lst >= sunrise + 1 && lst <= sunset - 1;
}

/* Generate merged ground track for a constellation of N satellites
   evenly spaced in the same orbital plane (different argPerigee offsets).
   This correctly models revisit improvement from multiple co-planar sats. */
function constellationTrack(sat, numOrbits, ptsPerOrbit) {
  const count = sat.count || 1;
  if (count <= 1) return groundTrack(sat, numOrbits, ptsPerOrbit);
  
  const allPts = [];
  for (let s = 0; s < count; s++) {
    const offset = (360 / count) * s; // evenly space in true anomaly
    const shifted = { ...sat, argPerigee: (sat.argPerigee || 0) + offset };
    const pts = groundTrack(shifted, numOrbits, ptsPerOrbit);
    allPts.push(...pts);
  }
  // Sort by time so pass detection works correctly
  allPts.sort((a, b) => a.t - b.t);
  return allPts;
}

/* Module-level cache: avoids re-running the expensive search across renders */
const _raanCache = new Map();

/* Compute optimal RAAN by longitude-matching: shift the orbit so its first ascending
   node crossing at the target latitude aligns with the anchor country's centre longitude.
   This guarantees the first pass is within ~1 orbit period (~1 hr for LEO) for all
   inclinations. Sunlit timing varies by orbit type and is visible in the revisit table.
   Results are cached by sat fingerprint + country id so subsequent calls are O(1). */
function computeOptimalRaan(sat, anchorCty) {
  const key = `${sat.altitude}|${sat.inclination}|${sat.eccentricity||0}|${sat.swathAngle||10}|${anchorCty.id}`;
  if (_raanCache.has(key)) return _raanCache.get(key);

  const targetLon = (anchorCty.lonMin + anchorCty.lonMax) / 2;
  const targetLat = (anchorCty.latMin + anchorCty.latMax) / 2;
  const testPts = groundTrack({ ...sat, raan: 0, eccentricity: sat.eccentricity || 0, count: 1 }, 1, 2000);
  let bestShift = 0, bestErr = 999;
  for (let i = 0; i < testPts.length; i++) {
    if (Math.abs(testPts[i].lat - targetLat) < 3) {
      const err = Math.abs(testPts[i].lat - targetLat);
      if (err < bestErr) {
        bestErr = err;
        bestShift = targetLon - testPts[i].lon;
      }
    }
  }
  const bestRaan = ((bestShift % 360) + 360) % 360;
  _raanCache.set(key, bestRaan);
  return bestRaan;
}

/* Create a copy of a satellite with RAAN optimized for an anchor country */
function satWithAnchorRaan(sat, anchorCty) {
  const optRaan = computeOptimalRaan(sat, anchorCty);
  return { ...sat, raan: optRaan };
}

/* ══════════════════════════════════════════════════════════════
   CLOUD COVER DATABASE (annual avg % cloud cover by region)
   Sources: NASA CERES / MODIS climatology approximations
   ══════════════════════════════════════════════════════════════ */
const CLOUD_DATA = {
  cz: 65, us: 52, de: 66, fr: 58, gb: 72, ua: 60, pl: 64, es: 40, it: 45, cn: 55,
  in: 48, ru: 68, br: 62, au: 38, jp: 60, kr: 58, sa: 18, eg: 15, za: 35, ng: 55,
  il: 22, tr: 48, ar: 50, ca: 62, mx: 45, id: 70, se: 68, no: 72, ke: 50, co: 65,
  rw: 55, dz: 25, ao: 50, bj: 55, bw: 25, bf: 45, bi: 60, cm: 60, cf: 55, td: 30,
  km: 55, cd: 65, cg: 65, ci: 60, dj: 20, gq: 70, er: 25, sz: 45, et: 45, ga: 70,
  gm: 40, gh: 55, gn: 60, gw: 55, ls: 40, lr: 65, ly: 15, mg: 50, mw: 45, ml: 35,
  mr: 20, mu: 50, mz: 40, na: 20, ne: 30, sn: 40, sl: 60, so: 25, ss: 55, sd: 25,
  tz: 50, tg: 55, tn: 35, ug: 55, zm: 45, zw: 40,
};

/* ══════════════════════════════════════════════════════════════
   COUNTRIES
   ══════════════════════════════════════════════════════════════ */
const COUNTRIES = [
  { id:"cz", name:"Czech Republic", lat:49.8, lon:15.5, latMin:48.5, latMax:51.1, lonMin:12.1, lonMax:18.9, area:78867 },
  { id:"us", name:"United States", lat:39.8, lon:-98.6, latMin:24.5, latMax:49.4, lonMin:-124.8, lonMax:-66.9, area:9833520 },
  { id:"de", name:"Germany", lat:51.2, lon:10.4, latMin:47.3, latMax:55.1, lonMin:5.9, lonMax:15.0, area:357022 },
  { id:"fr", name:"France", lat:46.2, lon:2.2, latMin:41.4, latMax:51.1, lonMin:-5.1, lonMax:9.6, area:640679 },
  { id:"gb", name:"United Kingdom", lat:55.4, lon:-3.4, latMin:49.9, latMax:58.7, lonMin:-8.2, lonMax:1.8, area:243610 },
  { id:"ua", name:"Ukraine", lat:48.4, lon:31.2, latMin:44.4, latMax:52.4, lonMin:22.1, lonMax:40.2, area:603500 },
  { id:"pl", name:"Poland", lat:51.9, lon:19.1, latMin:49.0, latMax:54.8, lonMin:14.1, lonMax:24.1, area:312685 },
  { id:"es", name:"Spain", lat:40.5, lon:-3.7, latMin:36.0, latMax:43.8, lonMin:-9.3, lonMax:4.3, area:505990 },
  { id:"it", name:"Italy", lat:41.9, lon:12.6, latMin:36.6, latMax:47.1, lonMin:6.6, lonMax:18.5, area:301340 },
  { id:"cn", name:"China", lat:35.9, lon:104.2, latMin:18.2, latMax:53.6, lonMin:73.7, lonMax:134.8, area:9596960 },
  { id:"in", name:"India", lat:20.6, lon:79.0, latMin:6.7, latMax:35.5, lonMin:68.2, lonMax:97.4, area:3287263 },
  { id:"ru", name:"Russia", lat:61.5, lon:105.3, latMin:41.2, latMax:81.9, lonMin:27.0, lonMax:180.0, area:17098242 },
  { id:"br", name:"Brazil", lat:-14.2, lon:-51.9, latMin:-33.7, latMax:5.3, lonMin:-73.9, lonMax:-34.8, area:8515767 },
  { id:"au", name:"Australia", lat:-25.3, lon:133.8, latMin:-43.6, latMax:-10.1, lonMin:113.2, lonMax:153.6, area:7692024 },
  { id:"jp", name:"Japan", lat:36.2, lon:138.3, latMin:24.4, latMax:45.5, lonMin:122.9, lonMax:153.0, area:377975 },
  { id:"kr", name:"South Korea", lat:35.9, lon:127.8, latMin:33.1, latMax:38.6, lonMin:125.1, lonMax:131.9, area:100210 },
  { id:"sa", name:"Saudi Arabia", lat:23.9, lon:45.1, latMin:16.4, latMax:32.2, lonMin:34.6, lonMax:55.7, area:2149690 },
  { id:"il", name:"Israel", lat:31.0, lon:34.9, latMin:29.5, latMax:33.3, lonMin:34.3, lonMax:35.9, area:20770 },
  { id:"tr", name:"Turkey", lat:38.9, lon:35.2, latMin:36.0, latMax:42.1, lonMin:26.0, lonMax:44.8, area:783562 },
  { id:"ar", name:"Argentina", lat:-38.4, lon:-63.6, latMin:-55.1, latMax:-21.8, lonMin:-73.6, lonMax:-53.6, area:2780400 },
  { id:"ca", name:"Canada", lat:56.1, lon:-106.3, latMin:41.7, latMax:83.1, lonMin:-141.0, lonMax:-52.6, area:9984670 },
  { id:"mx", name:"Mexico", lat:23.6, lon:-102.6, latMin:14.5, latMax:32.7, lonMin:-118.4, lonMax:-86.7, area:1964375 },
  { id:"id", name:"Indonesia", lat:-0.8, lon:113.9, latMin:-11.0, latMax:6.1, lonMin:95.0, lonMax:141.0, area:1904569 },
  { id:"se", name:"Sweden", lat:60.1, lon:18.6, latMin:55.3, latMax:69.1, lonMin:11.1, lonMax:24.2, area:450295 },
  { id:"no", name:"Norway", lat:60.5, lon:8.5, latMin:58.0, latMax:71.2, lonMin:4.5, lonMax:31.2, area:385207 },
  // ── Africa ──
  { id:"rw", name:"Rwanda", lat:-1.9, lon:29.9, latMin:-2.8, latMax:-1.1, lonMin:28.9, lonMax:30.9, area:26338 },
  { id:"eg", name:"Egypt", lat:26.8, lon:30.8, latMin:22.0, latMax:31.7, lonMin:24.7, lonMax:36.9, area:1002450 },
  { id:"za", name:"South Africa", lat:-30.6, lon:22.9, latMin:-34.8, latMax:-22.1, lonMin:16.5, lonMax:32.9, area:1219090 },
  { id:"ng", name:"Nigeria", lat:9.1, lon:8.7, latMin:4.3, latMax:13.9, lonMin:2.7, lonMax:14.7, area:923768 },
  { id:"dz", name:"Algeria", lat:28.0, lon:1.7, latMin:18.9, latMax:37.1, lonMin:-8.7, lonMax:12.0, area:2381741 },
  { id:"ao", name:"Angola", lat:-11.2, lon:17.9, latMin:-18.0, latMax:-4.4, lonMin:11.7, lonMax:24.1, area:1246700 },
  { id:"bj", name:"Benin", lat:9.3, lon:2.3, latMin:6.2, latMax:12.4, lonMin:0.8, lonMax:3.8, area:112622 },
  { id:"bw", name:"Botswana", lat:-22.3, lon:24.7, latMin:-26.9, latMax:-17.8, lonMin:19.9, lonMax:29.4, area:581730 },
  { id:"bf", name:"Burkina Faso", lat:12.4, lon:-1.6, latMin:9.4, latMax:15.1, lonMin:-5.5, lonMax:2.4, area:274200 },
  { id:"bi", name:"Burundi", lat:-3.4, lon:29.9, latMin:-4.5, latMax:-2.3, lonMin:29.0, lonMax:30.8, area:27834 },
  { id:"cm", name:"Cameroon", lat:7.4, lon:12.4, latMin:1.7, latMax:13.1, lonMin:8.5, lonMax:16.2, area:475440 },
  { id:"cf", name:"Central African Rep.", lat:6.6, lon:20.9, latMin:2.2, latMax:11.0, lonMin:14.4, lonMax:27.5, area:622984 },
  { id:"td", name:"Chad", lat:15.5, lon:18.7, latMin:7.4, latMax:23.5, lonMin:13.5, lonMax:24.0, area:1284000 },
  { id:"cd", name:"DR Congo", lat:-4.0, lon:21.8, latMin:-13.5, latMax:5.4, lonMin:12.2, lonMax:31.3, area:2344858 },
  { id:"cg", name:"Congo", lat:-0.2, lon:15.8, latMin:-5.0, latMax:3.7, lonMin:11.2, lonMax:18.6, area:342000 },
  { id:"ci", name:"Côte d'Ivoire", lat:7.5, lon:-5.5, latMin:4.4, latMax:10.7, lonMin:-8.6, lonMax:-2.5, area:322463 },
  { id:"dj", name:"Djibouti", lat:11.6, lon:43.1, latMin:10.9, latMax:12.7, lonMin:41.8, lonMax:43.4, area:23200 },
  { id:"gq", name:"Equatorial Guinea", lat:1.6, lon:10.3, latMin:-1.5, latMax:3.8, lonMin:5.6, lonMax:11.3, area:28051 },
  { id:"er", name:"Eritrea", lat:15.2, lon:39.8, latMin:12.4, latMax:18.0, lonMin:36.4, lonMax:43.1, area:117600 },
  { id:"sz", name:"Eswatini", lat:-26.5, lon:31.5, latMin:-27.3, latMax:-25.7, lonMin:30.8, lonMax:32.1, area:17364 },
  { id:"et", name:"Ethiopia", lat:9.1, lon:40.5, latMin:3.4, latMax:15.0, lonMin:33.0, lonMax:48.0, area:1104300 },
  { id:"ga", name:"Gabon", lat:-0.8, lon:11.6, latMin:-3.9, latMax:2.3, lonMin:8.7, lonMax:14.5, area:267668 },
  { id:"gm", name:"Gambia", lat:13.4, lon:-15.3, latMin:13.1, latMax:13.8, lonMin:-16.8, lonMax:-13.8, area:11300 },
  { id:"gh", name:"Ghana", lat:7.9, lon:-1.0, latMin:4.7, latMax:11.2, lonMin:-3.3, lonMax:1.2, area:238535 },
  { id:"gn", name:"Guinea", lat:9.9, lon:-11.8, latMin:7.2, latMax:12.7, lonMin:-15.1, lonMax:-7.6, area:245857 },
  { id:"gw", name:"Guinea-Bissau", lat:12.0, lon:-15.2, latMin:10.9, latMax:12.7, lonMin:-16.7, lonMax:-13.6, area:36125 },
  { id:"ke", name:"Kenya", lat:-0.0, lon:38.0, latMin:-4.7, latMax:5.0, lonMin:33.9, lonMax:41.9, area:580367 },
  { id:"ls", name:"Lesotho", lat:-29.6, lon:28.2, latMin:-30.7, latMax:-28.6, lonMin:27.0, lonMax:29.5, area:30355 },
  { id:"lr", name:"Liberia", lat:6.4, lon:-9.4, latMin:4.4, latMax:8.6, lonMin:-11.5, lonMax:-7.4, area:111369 },
  { id:"ly", name:"Libya", lat:26.3, lon:17.2, latMin:19.5, latMax:33.2, lonMin:9.4, lonMax:25.1, area:1759540 },
  { id:"mg", name:"Madagascar", lat:-18.8, lon:46.9, latMin:-25.6, latMax:-11.9, lonMin:43.2, lonMax:50.5, area:587041 },
  { id:"mw", name:"Malawi", lat:-13.3, lon:34.3, latMin:-17.1, latMax:-9.4, lonMin:32.7, lonMax:35.9, area:118484 },
  { id:"ml", name:"Mali", lat:17.6, lon:-4.0, latMin:10.2, latMax:25.0, lonMin:-12.2, lonMax:4.3, area:1240192 },
  { id:"mr", name:"Mauritania", lat:21.0, lon:-10.9, latMin:14.7, latMax:27.3, lonMin:-17.1, lonMax:-4.8, area:1030700 },
  { id:"mz", name:"Mozambique", lat:-18.7, lon:35.5, latMin:-26.9, latMax:-10.5, lonMin:30.2, lonMax:40.8, area:801590 },
  { id:"na", name:"Namibia", lat:-22.0, lon:18.5, latMin:-28.9, latMax:-16.9, lonMin:11.7, lonMax:25.3, area:824292 },
  { id:"ne", name:"Niger", lat:17.6, lon:8.1, latMin:11.7, latMax:23.5, lonMin:0.2, lonMax:16.0, area:1267000 },
  { id:"sn", name:"Senegal", lat:14.5, lon:-14.5, latMin:12.3, latMax:16.7, lonMin:-17.5, lonMax:-11.4, area:196722 },
  { id:"sl", name:"Sierra Leone", lat:8.5, lon:-11.8, latMin:6.9, latMax:10.0, lonMin:-13.3, lonMax:-10.3, area:71740 },
  { id:"so", name:"Somalia", lat:5.2, lon:46.2, latMin:-1.7, latMax:12.0, lonMin:40.9, lonMax:51.4, area:637657 },
  { id:"ss", name:"South Sudan", lat:6.9, lon:31.3, latMin:3.5, latMax:12.2, lonMin:24.1, lonMax:35.9, area:644329 },
  { id:"sd", name:"Sudan", lat:12.9, lon:30.2, latMin:8.7, latMax:22.2, lonMin:21.8, lonMax:38.6, area:1861484 },
  { id:"tz", name:"Tanzania", lat:-6.4, lon:34.9, latMin:-11.7, latMax:-1.0, lonMin:29.3, lonMax:40.4, area:945087 },
  { id:"tg", name:"Togo", lat:8.6, lon:1.2, latMin:6.1, latMax:11.1, lonMin:-0.1, lonMax:1.8, area:56785 },
  { id:"tn", name:"Tunisia", lat:33.9, lon:9.5, latMin:30.2, latMax:37.3, lonMin:7.5, lonMax:11.6, area:163610 },
  { id:"ug", name:"Uganda", lat:1.4, lon:32.3, latMin:-1.5, latMax:4.2, lonMin:29.6, lonMax:35.0, area:241038 },
  { id:"zm", name:"Zambia", lat:-13.1, lon:27.8, latMin:-18.1, latMax:-8.2, lonMin:22.0, lonMax:33.7, area:752618 },
  { id:"zw", name:"Zimbabwe", lat:-20.0, lon:30.0, latMin:-22.4, latMax:-15.6, lonMin:25.2, lonMax:33.1, area:390757 },
];

const SAT_COLORS = [
  "#00E5FF","#76FF03","#FFEA00","#E040FB","#FF6D00","#FF4081","#64FFDA",
  "#448AFF","#FF1744","#B388FF","#F4FF81","#FF80AB","#84FFFF","#CCFF90"
];

/* ══════════════════════════════════════════════════════════════
   PRESETS
   ══════════════════════════════════════════════════════════════ */
const PRESETS = [
  // ── TRL Space Satellites ──
  { id:"troll_ng", name:"TROLL NG", altitude:520, inclination:97.5, eccentricity:0.0001, raan:0, argPerigee:0, swathAngle:1.07, offNadir:0, dataCapacity:70000, lifetime:5, type:"Hyperspectral", gsd:"4.75m", mass:"<16 kg", desc:"Hyperspectral 32-band nanosatellite (8U CubeSat)" },
  { id:"trap", name:"TRAP", altitude:510, inclination:97.4, eccentricity:0.0001, raan:0, argPerigee:0, swathAngle:0.56, offNadir:30, dataCapacity:8000, lifetime:7, type:"Defence EO", gsd:"0.9m (0.7m proc.)", mass:"<100 kg", desc:"Defence EO microsatellite, PAN+RGB, on-board AI" },
  { id:"trace", name:"TRACE", altitude:500, inclination:97.4, eccentricity:0.0001, raan:0, argPerigee:0, swathAngle:0.63, offNadir:45, dataCapacity:20000, lifetime:7, type:"VHR MS", gsd:"50cm PAN (40cm proc.)", mass:"<200 kg", desc:"VHR multispectral surveillance, PAN+MS (6 bands), on-board AI" },
  // ── Reference Satellites ──
  { id:"sentinel2", name:"Sentinel-2A", altitude:786, inclination:98.5, eccentricity:0.0001, raan:160, argPerigee:0, swathAngle:10.3, offNadir:0, dataCapacity:1600000, lifetime:7.25, type:"EO", gsd:"10m MS", mass:"1140 kg", desc:"ESA Copernicus wide-swath multispectral" },
  { id:"worldview3", name:"WorldView-3", altitude:617, inclination:97.2, eccentricity:0.0001, raan:50, argPerigee:0, swathAngle:1.0, offNadir:45, dataCapacity:680000, lifetime:7, type:"VHR", gsd:"31cm PAN", mass:"2800 kg", desc:"Maxar VHR commercial imaging" },
  { id:"planet", name:"PlanetScope", altitude:475, inclination:97.4, eccentricity:0.0001, raan:100, argPerigee:0, swathAngle:12.3, offNadir:0, dataCapacity:200000, lifetime:3, type:"EO", gsd:"3m", mass:"5 kg", desc:"Planet Labs Dove nanosatellite" },
];

/* ══════════════════════════════════════════════════════════════
   COVERAGE SIMULATION (returns timeline + pass lines)
   ══════════════════════════════════════════════════════════════ */
function simulateCoverage(sat, cty, tfDays) {
  const gridRes = Math.max(0.5, Math.min(2.5, Math.sqrt(cty.area / 40000)));
  const latN = Math.max(4, Math.ceil((cty.latMax - cty.latMin) / gridRes));
  const lonN = Math.max(4, Math.ceil((cty.lonMax - cty.lonMin) / gridRes));
  const cells = latN * lonN;
  const covered = new Uint8Array(cells);
  const T = orbPeriod(sat.altitude);
  const orbits = Math.ceil((86400 / T) * tfDays);
  const sw = effSwathKm(sat.altitude, sat.swathAngle || 10);
  const swLat = (sw / (2 * Math.PI * R_E)) * 360;
  const maxPerDay = (sat.dataCapacity || 1e9) * (sat.count || 1);
  let dayUsed = 0, curDay = 0;
  const _ls = cty.latMax - cty.latMin;
  const _fl = 2 * Math.min(sat.inclination, 180 - sat.inclination);
  const _mp = Math.ceil((_fl / Math.max(_ls, 0.5)) * 4);
  const _pp = Math.max(_mp, 500);
  const _sp = orbits * _pp > 500000 ? Math.max(500, Math.round(500000 / orbits)) : _pp;
  const track = constellationTrack(sat, orbits, _sp);
  const timeline = [];
  let lastPct = -1;
  const passLines = []; // segments over country
  let inPass = false, passStart = null;

  for (let p = 0; p < track.length; p++) {
    const pt = track[p];
    const day = pt.t / 86400;
    if (Math.floor(day) > curDay) { curDay = Math.floor(day); dayUsed = 0; }
    const swLon = swLat / Math.max(Math.cos(pt.lat * DEG), 0.08);
    const overCountry = pt.lat + swLat / 2 >= cty.latMin && pt.lat - swLat / 2 <= cty.latMax &&
                        pt.lon + swLon / 2 >= cty.lonMin && pt.lon - swLon / 2 <= cty.lonMax;

    if (overCountry) {
      if (!inPass) { passStart = { lat: pt.lat, lon: pt.lon }; inPass = true; }
      for (let i = 0; i < latN; i++) {
        const cLat = cty.latMin + (i + 0.5) * (cty.latMax - cty.latMin) / latN;
        if (cLat < pt.lat - swLat / 2 || cLat > pt.lat + swLat / 2) continue;
        for (let j = 0; j < lonN; j++) {
          const idx = i * lonN + j;
          if (covered[idx]) continue;
          const cLon = cty.lonMin + (j + 0.5) * (cty.lonMax - cty.lonMin) / lonN;
          if (cLon < pt.lon - swLon / 2 || cLon > pt.lon + swLon / 2) continue;
          const ca = cty.area / cells;
          if (dayUsed + ca > maxPerDay) continue;
          covered[idx] = 1; dayUsed += ca;
        }
      }
    } else if (inPass) {
      passLines.push({ from: passStart, to: { lat: track[p - 1].lat, lon: track[p - 1].lon } });
      inPass = false;
    }
    const cnt = covered.reduce((s, v) => s + v, 0);
    const pct = Math.round((cnt / cells) * 100);
    if (pct !== lastPct) { timeline.push({ day, pct }); lastPct = pct; if (pct >= 100) break; }
  }
  if (inPass && track.length > 0) {
    passLines.push({ from: passStart, to: { lat: track[track.length - 1].lat, lon: track[track.length - 1].lon } });
  }
  const finalPct = covered.reduce((s, v) => s + v, 0) / cells * 100;
  if (timeline.length === 0 || timeline[timeline.length - 1].pct < finalPct) {
    timeline.push({ day: tfDays, pct: Math.round(finalPct) });
  }
  return { timeline, passLines, finalPct: Math.round(finalPct), fullCovDay: (timeline.find(d => d.pct >= 100) || {}).day || null };
}

/* ══════════════════════════════════════════════════════════════
   CONTINENT OUTLINES
   ══════════════════════════════════════════════════════════════ */
const COASTS = [
  [[-130,50],[-125,50],[-120,48],[-124,42],[-118,34],[-105,30],[-100,28],[-97,26],[-82,25],[-80,30],[-75,35],[-70,42],[-67,45],[-65,48],[-55,50],[-60,54],[-70,63],[-90,68],[-110,72],[-130,70],[-150,62],[-168,65],[-165,72],[-130,50]],
  [[-80,10],[-75,12],[-60,5],[-50,0],[-35,-5],[-38,-15],[-48,-28],[-56,-36],[-68,-52],[-75,-45],[-75,-35],[-70,-18],[-80,-5],[-80,10]],
  [[-10,36],[3,43],[0,46],[5,44],[15,45],[20,40],[28,37],[30,45],[20,48],[18,55],[10,55],[0,50],[-10,36]],
  [[20,55],[28,60],[32,65],[30,70],[18,70],[10,60],[20,55]],
  [[-15,12],[-17,15],[-15,28],[-5,36],[10,37],[25,32],[35,30],[42,12],[50,12],[45,2],[40,-8],[28,-33],[18,-34],[12,-25],[5,8],[-8,5],[-15,12]],
  [[30,40],[50,38],[60,40],[68,25],[78,8],[85,22],[100,15],[115,22],[128,35],[142,43],[138,52],[120,53],[80,50],[55,52],[40,42],[30,40]],
  [[115,-15],[140,-18],[154,-28],[148,-35],[130,-32],[115,-33],[114,-22],[115,-15]],
];

/* ══════════════════════════════════════════════════════════════
   UI PRIMITIVES
   ══════════════════════════════════════════════════════════════ */
const P = ({ children, style, ...p }) => <div style={{ background:"rgba(8,12,24,0.85)", border:"1px solid rgba(70,140,200,0.12)", borderRadius:6, ...style }} {...p}>{children}</div>;
const Lbl = ({ children }) => <div style={{ fontSize:9, color:"#3a6888", letterSpacing:2, textTransform:"uppercase", marginBottom:6, fontWeight:600 }}>{children}</div>;
const Btn = ({ children, on, sm, accent, style, ...p }) => <button style={{
  background: on ? "rgba(70,160,220,0.18)" : "rgba(70,160,220,0.04)",
  border: `1px solid ${on ? "rgba(70,160,220,0.4)" : "rgba(70,160,220,0.1)"}`,
  color: on ? "#a0d8f0" : "#3a6888", padding: sm ? "3px 7px" : "5px 11px",
  borderRadius:3, cursor:"pointer", fontSize: sm ? 9 : 10, fontFamily:"inherit", letterSpacing:1, transition:"all .15s", ...style
}} {...p}>{children}</button>;
const Inp = ({ label, ...p }) => <div style={{ marginBottom:7 }}>
  {label && <div style={{ fontSize:8, color:"#3a6888", letterSpacing:1, marginBottom:2 }}>{label}</div>}
  <input style={{ width:"100%", background:"rgba(4,8,16,0.9)", border:"1px solid rgba(70,140,200,0.16)", color:"#b0d0e8", padding:"5px 7px", borderRadius:3, fontSize:10, fontFamily:"inherit", boxSizing:"border-box" }} {...p} />
</div>;


/* ══════════════════════════════════════════════════════════════
   COUNTRY GROUND TRACK MAP
   - Swath bands perpendicular to track heading
   - Coverage correctly scales with timeframe
   - Mercator projection for proper shape
   ══════════════════════════════════════════════════════════════ */
const TRACK_TF_OPTS = [
  { l: "1 day", d: 1 }, { l: "3 days", d: 3 }, { l: "5 days", d: 5 },
  { l: "1 week", d: 7 }, { l: "14 days", d: 14 }, { l: "30 days", d: 30 },
];

const BORDERS = {"id":[[140.9745,-2.6005],[140.8492,-6.7028],[140.977,-9.1061],[139.9881,-8.1925],[140.1517,-7.8847],[139.9736,-8.0969],[139.3952,-8.2009],[139.2171,-7.9513],[138.9144,-8.2921],[139.093,-7.56],[138.6675,-7.1978],[139.24,-7.1511],[138.8425,-7.1456],[138.5615,-6.9186],[139.1922,-6.9729],[138.6834,-6.7145],[138.8291,-6.706],[138.2625,-5.9147],[138.404,-5.8357],[138.1576,-5.7737],[138.363,-5.6788],[138.0711,-5.7286],[138.0614,-5.4037],[137.2788,-4.9366],[135.1856,-4.4489],[134.6482,-4.1187],[134.6885,-3.9474],[134.9678,-3.9368],[134.5252,-4.0265],[134.3335,-3.8688],[134.3153,-4.0203],[134.1275,-3.7453],[133.971,-3.8503],[133.6414,-3.4837],[133.6909,-3.2161],[133.9295,-3.1147],[133.8605,-2.9229],[133.514,-3.3784],[133.4534,-3.8654],[132.9077,-4.0881],[132.731,-3.6823],[132.9318,-3.5541],[132.8117,-3.2817],[131.958,-2.7882],[132.3118,-2.6764],[132.7283,-2.8132],[133.2396,-2.4168],[133.3807,-2.6764],[133.4456,-2.4947],[133.6824,-2.7175],[133.7472,-2.4379],[134.0051,-2.3884],[133.95,-2.2035],[133.792,-2.2602],[133.9404,-2.1076],[132.3057,-2.2656],[131.9419,-1.8951],[131.9822,-1.5388],[130.9483,-1.4122],[131.2362,-1.1173],[131.243,-0.8194],[132.3967,-0.3534],[133.3916,-0.7232],[133.9772,-0.7232],[134.282,-1.3502],[134.0871,-1.6828],[134.1934,-2.3791],[134.4639,-2.8621],[134.4844,-2.5393],[134.6384,-2.5112],[134.8989,-3.2628],[135.3408,-3.3897],[135.9099,-2.9858],[136.3809,-2.2341],[137.1785,-2.1061],[137.1235,-1.7975],[137.8391,-1.4711],[139.7808,-2.3547],[140.9745,-2.6005]],"ar":[[-67.1939,-22.8222],[-66.2225,-21.7869],[-64.5962,-22.2066],[-64.3253,-22.8719],[-63.9476,-22.0076],[-62.8044,-22.0041],[-61.0063,-23.8055],[-57.7541,-25.1809],[-57.5753,-25.5644],[-58.6042,-27.3163],[-55.7547,-27.4437],[-54.7931,-26.645],[-54.6002,-25.5749],[-53.91,-25.6292],[-53.6616,-26.2598],[-53.83,-27.1568],[-55.7725,-28.232],[-57.8898,-30.5508],[-58.5704,-34.2882],[-57.2488,-35.2488],[-57.3664,-35.9879],[-56.6723,-36.6005],[-57.5584,-38.1209],[-59.7969,-38.8385],[-62.3798,-38.7984],[-62.1815,-40.6266],[-62.7294,-41.0411],[-65.1317,-40.8424],[-65.005,-42.0932],[-64.4451,-42.4467],[-63.7686,-42.0777],[-63.6203,-42.7512],[-64.946,-42.6542],[-64.3021,-42.9809],[-65.3308,-43.6633],[-65.5214,-44.9319],[-66.9293,-45.2565],[-67.5785,-45.9929],[-66.7864,-47.0066],[-65.7449,-47.2042],[-65.8456,-47.7422],[-66.3891,-47.8639],[-65.762,-47.9486],[-67.5565,-49.0158],[-67.8949,-49.9969],[-69.0135,-50.0077],[-68.3499,-50.1488],[-69.4107,-51.0839],[-68.9626,-51.5602],[-69.6168,-51.6252],[-68.9735,-51.62],[-68.4347,-52.3907],[-71.9652,-51.9706],[-72.4501,-51.5527],[-72.3028,-50.6489],[-73.1777,-50.7495],[-73.5727,-49.9324],[-72.3023,-48.3475],[-72.5439,-47.9148],[-71.6872,-46.6901],[-71.7651,-45.5724],[-71.3172,-45.2672],[-72.0887,-44.7828],[-71.1224,-44.5494],[-71.8606,-44.3771],[-71.743,-43.1901],[-72.1484,-42.5926],[-71.7363,-42.0839],[-71.7144,-39.6014],[-70.8376,-38.5845],[-71.1952,-36.8391],[-70.3869,-36.0587],[-70.5789,-35.2597],[-69.8289,-34.2333],[-70.569,-31.3042],[-69.8358,-30.1619],[-69.653,-28.3979],[-68.8143,-27.1203],[-68.3305,-27.045],[-68.5724,-24.7699],[-67.3398,-24.0006],[-67.1939,-22.8222]],"in":[[77.8003,35.4954],[79.6139,32.765],[78.3848,32.5475],[78.7451,31.3081],[81.0033,30.2127],[80.0364,28.837],[82.7521,27.495],[87.9755,26.3666],[88.0999,27.9283],[88.6105,28.1058],[89.0965,26.8214],[91.8787,26.803],[91.5644,27.8514],[92.575,27.8478],[94.5999,29.3166],[96.0745,29.3693],[96.5926,28.7579],[96.3019,28.4207],[97.3623,27.9949],[96.8621,27.5993],[97.1192,27.0873],[96.1426,27.2575],[95.1228,26.6116],[94.1319,23.857],[93.3097,24.064],[92.8775,21.957],[92.2593,23.7065],[91.5861,22.9445],[91.1363,23.6295],[92.4838,24.9282],[89.819,25.285],[89.653,26.2227],[88.3725,26.6116],[88.0876,25.831],[88.9849,25.285],[88.0218,24.6456],[88.7375,24.2871],[89.0827,21.6173],[88.6602,22.205],[88.2544,21.5547],[87.9066,22.4248],[88.1643,22.0898],[86.9177,21.3274],[86.4263,19.9992],[85.2681,19.78],[82.3074,16.5798],[80.2796,15.7007],[79.8567,10.2933],[78.9133,9.4743],[79.4536,9.159],[78.4121,9.0998],[77.5109,8.076],[76.544,8.9125],[73.447,16.0689],[72.6492,19.852],[72.5646,21.3858],[73.132,21.7578],[72.5017,21.976],[72.9175,22.2714],[72.1488,22.2779],[72.1047,21.2051],[70.7402,20.7185],[68.9446,22.2951],[70.5029,23.1044],[69.5857,22.7612],[68.4237,23.4458],[68.8216,23.8784],[68.1434,23.6127],[68.7473,24.3312],[71.0828,24.4115],[69.5344,27.1256],[70.3419,28.0115],[71.8744,27.9597],[74.6588,31.0838],[74.5504,31.827],[75.359,32.2617],[74.0023,33.1777],[73.9193,34.6425],[75.7958,34.5079],[77.8003,35.4954]],"cn":[[78.9177,33.3863],[78.0443,35.4916],[74.3826,37.1266],[75.1641,37.4006],[74.8358,38.4552],[73.6326,39.4483],[80.2312,42.0337],[80.7934,43.1495],[79.8582,44.9037],[82.5397,45.1237],[83.0215,47.2059],[85.4986,47.0518],[86.8595,49.1053],[90.3448,47.6586],[90.8732,45.1862],[95.3977,44.2805],[96.3578,42.7245],[104.9738,41.5861],[110.4067,42.7686],[111.9334,43.6966],[111.9583,45.0845],[113.6048,44.7397],[117.3938,46.5714],[119.8544,46.6597],[118.5423,47.9662],[115.5142,48.1316],[116.6843,49.8233],[119.3162,50.0927],[120.8743,53.2802],[125.6214,53.0621],[127.5387,49.7899],[130.6743,48.8708],[131.0234,47.6823],[134.7187,48.2634],[133.0988,45.1078],[130.9334,44.8417],[130.5308,42.5305],[129.8798,42.996],[128.1461,41.3763],[126.6795,41.736],[121.1795,38.7209],[121.8679,40.9958],[117.7154,39.1114],[119.1455,37.1789],[122.6882,37.4098],[119.1929,35.0004],[121.9221,31.7544],[119.6129,32.3546],[121.9508,30.9821],[120.1471,30.1989],[122.1291,29.9038],[121.1423,28.8459],[121.6563,28.3393],[120.5596,28.1128],[119.0935,26.1458],[119.6546,25.3581],[116.4947,22.9394],[113.4175,23.097],[113.4837,22.1551],[110.3585,21.4357],[110.2803,20.253],[109.5945,21.7464],[107.3482,21.5994],[105.3122,23.3658],[101.6891,22.4789],[101.756,21.1433],[100.1625,21.4364],[98.8657,24.1457],[97.5165,23.9428],[98.6793,27.5773],[96.3019,28.4207],[96.142,29.3685],[91.9522,27.7248],[85.9803,27.8852],[78.7451,31.3081],[78.3848,32.5475],[79.6207,32.7287],[78.9177,33.3863]],"il":[[34.2484,31.2114],[34.3509,31.2893],[34.3673,31.3928],[34.556,31.5398],[34.4812,31.5831],[34.7119,31.9516],[34.8377,32.2807],[34.9475,32.8142],[35.0628,32.8583],[35.0996,33.0876],[35.2837,33.1012],[35.3452,33.0556],[35.4801,33.0874],[35.5495,33.281],[35.6039,33.2401],[35.6985,33.3227],[35.7694,33.3426],[35.8211,33.4067],[35.8099,33.36],[35.7638,33.3344],[35.8021,33.3125],[35.7686,33.2727],[35.8302,33.19],[35.8115,33.1119],[35.8489,33.0987],[35.8881,32.9449],[35.8342,32.8279],[35.7576,32.7443],[35.5605,32.6409],[35.58,32.5604],[35.552,32.5256],[35.58,32.4977],[35.5451,32.4096],[35.561,32.3847],[35.4069,32.4148],[35.3929,32.4945],[35.3631,32.5101],[35.1908,32.5417],[35.0447,32.4341],[34.9966,32.323],[35.0076,32.2444],[34.9614,32.2016],[35.0072,32.0208],[34.9867,31.9672],[35.039,31.9086],[35.0343,31.8611],[34.9774,31.8334],[35.0763,31.8542],[35.1862,31.8087],[35.2158,31.8201],[35.222,31.8792],[35.2649,31.8263],[35.2628,31.7486],[35.238,31.7093],[35.1257,31.7331],[34.9736,31.6304],[34.8672,31.3964],[34.8788,31.3628],[35.1649,31.3623],[35.3901,31.4871],[35.4585,31.4916],[35.3957,31.2577],[35.4385,31.1037],[35.3853,30.9633],[35.3222,30.89],[35.3166,30.8228],[35.14,30.4302],[35.1621,30.3614],[35.1252,30.2447],[35.1453,30.1234],[35.0863,30.034],[34.9513,29.5456],[34.903,29.4897],[34.8553,29.5457],[34.8244,29.7417],[34.7332,30.0126],[34.5995,30.3445],[34.5269,30.4096],[34.5362,30.4822],[34.4804,30.6512],[34.2484,31.2114]],"et":[[34.0707,9.4546],[34.3259,10.2236],[34.2795,10.5655],[34.5748,10.8841],[34.7544,10.6841],[34.8385,10.7291],[34.9583,10.9034],[35.0727,11.8189],[35.3174,12.0237],[35.6877,12.6592],[36.115,12.7018],[36.1369,13.0324],[36.3913,13.5994],[36.5264,14.2635],[36.7481,14.3321],[37.0602,14.2768],[37.1211,14.4208],[37.2938,14.4574],[37.5537,14.1055],[37.8915,14.8795],[38.0067,14.7244],[38.2293,14.6797],[38.4268,14.4172],[38.8661,14.4937],[39.0097,14.6512],[39.2082,14.4402],[39.515,14.5678],[39.8923,14.4266],[40.1046,14.466],[40.8332,14.106],[41.192,13.6159],[41.7118,13.2479],[41.9466,12.8762],[42.1849,12.733],[42.3795,12.4659],[41.7924,11.7045],[41.7613,10.9962],[41.9375,10.9298],[42.6156,11.0897],[42.9237,10.9988],[42.6472,10.6322],[42.8363,10.2081],[43.1872,9.8833],[43.419,9.413],[43.9848,9.0083],[46.9792,7.9966],[47.9792,7.9966],[44.9415,4.9115],[43.969,4.954],[43.5285,4.8416],[43.1193,4.6477],[42.8316,4.3023],[42.1029,4.1919],[41.8359,3.9494],[41.163,3.9427],[40.7637,4.2849],[39.8485,3.8673],[39.5044,3.4033],[38.509,3.6508],[38.1019,3.6126],[37.0157,4.3706],[35.9405,4.508],[35.7516,4.8544],[35.8042,5.318],[35.4693,5.4308],[35.2875,5.3741],[34.9744,5.8631],[34.7035,6.6849],[34.2807,6.9801],[34.0336,7.2503],[34.0065,7.4099],[33.7161,7.6572],[33.0583,7.798],[32.9898,7.9172],[33.1851,8.1411],[33.2352,8.4556],[33.6204,8.4607],[33.741,8.3671],[34.0902,8.5565],[34.0707,9.4546]],"ss":[[35.9208,4.6193],[35.6106,4.62],[35.5708,4.9043],[35.3961,4.9264],[35.4116,5.0304],[35.2457,4.9817],[34.3812,4.6202],[33.4906,3.7497],[32.9972,3.8855],[32.3718,3.7311],[32.168,3.5122],[31.93,3.5984],[31.7779,3.8164],[31.5237,3.6561],[31.1415,3.7851],[30.8395,3.4902],[30.7469,3.6749],[30.5535,3.6045],[30.5252,3.8675],[30.1894,3.9561],[29.7878,4.3688],[29.7967,4.556],[29.535,4.6607],[29.2213,4.3407],[28.7554,4.554],[28.3574,4.2829],[27.7724,4.5958],[27.7526,4.7777],[27.2645,5.2601],[27.2611,5.5776],[27.1238,5.7687],[26.4248,6.0724],[26.509,6.2055],[26.2702,6.4657],[26.378,6.6533],[25.1904,7.5012],[25.2169,7.8641],[24.8002,8.1803],[24.1529,8.3179],[24.2505,8.5796],[24.177,8.7099],[24.5584,8.8867],[24.7964,9.8189],[25.0769,10.2869],[25.8285,10.4195],[25.9054,10.1813],[26.6257,9.4807],[27.0804,9.6068],[27.8951,9.5954],[28.0451,9.3314],[28.8438,9.3245],[28.9138,9.529],[29.4831,9.7617],[29.6154,10.0581],[30.013,10.2705],[30.7653,9.7243],[31.1644,9.764],[32.4307,11.0822],[32.3483,11.7032],[32.0822,11.9998],[32.7467,12.0027],[32.7318,12.2162],[33.2092,12.2104],[33.0832,11.5992],[33.14,10.739],[33.917,10.1745],[33.8814,9.4992],[34.0707,9.4546],[34.1119,8.6266],[33.7519,8.3688],[33.2352,8.4556],[33.1851,8.1411],[32.9898,7.9172],[33.0583,7.798],[33.7161,7.6572],[34.2807,6.9801],[34.7035,6.6849],[34.9744,5.8631],[35.2875,5.3741],[35.8042,5.318],[35.7516,4.8544],[35.9208,4.6193]],"so":[[46.467,6.5383],[47.9792,7.9966],[48.9391,9.4512],[48.9391,11.2491],[49.4192,11.3448],[49.5768,11.46],[49.8512,11.4649],[50.2683,11.5893],[50.437,11.6906],[50.5612,11.9079],[50.6411,11.9544],[50.7979,11.9891],[51.0488,11.8786],[51.2927,11.8331],[51.2478,11.6524],[51.1209,11.5053],[51.0813,11.3415],[51.0859,11.188],[51.1895,11.1383],[51.1151,10.9934],[51.1607,10.5985],[51.1074,10.5766],[51.0923,10.4822],[51.0101,10.434],[51.2204,10.442],[51.1761,10.5628],[51.2925,10.4801],[51.3952,10.479],[51.3975,10.3958],[51.3674,10.3677],[51.2729,10.3779],[51.241,10.4256],[51.0901,10.4073],[50.9274,10.3278],[50.9017,10.0117],[50.8062,9.6245],[50.8303,9.4196],[50.647,9.2114],[50.6336,9.0732],[50.4397,8.8882],[50.3035,8.5046],[50.1564,8.3206],[50.0911,8.161],[49.8242,7.9339],[49.8193,7.7399],[49.2473,6.8106],[49.0747,6.4153],[49.0713,6.2199],[48.6467,5.48],[48.2024,4.9089],[47.9485,4.4571],[46.8342,3.2325],[46.027,2.4381],[45.0015,1.8664],[44.5501,1.5591],[43.9326,1.0098],[43.7819,0.9216],[42.5803,-0.2975],[42.1665,-0.8027],[42.0715,-0.8287],[42.0164,-0.968],[41.9719,-0.9998],[41.9528,-0.8945],[41.9717,-1.0236],[41.8854,-1.2096],[41.8291,-1.1689],[41.8366,-1.2623],[41.5351,-1.6963],[41.5229,-1.5728],[40.9798,-0.8708],[40.9798,2.8419],[41.3427,3.2011],[41.9235,4.0706],[42.0684,4.1746],[42.8316,4.3023],[42.8996,4.3611],[42.9604,4.5174],[43.1193,4.6477],[43.5285,4.8416],[43.9323,4.9452],[44.9415,4.9115],[46.467,6.5383]],"ke":[[35.7058,4.6194],[35.7394,4.6811],[35.9208,4.6193],[36.0413,4.4437],[37.0157,4.3706],[38.1019,3.6126],[38.509,3.6508],[39.5044,3.4033],[39.8485,3.8673],[40.7637,4.2849],[41.163,3.9427],[41.885,3.9772],[41.3427,3.2011],[40.9654,2.8141],[40.9798,-0.8708],[41.5229,-1.5728],[41.5351,-1.6963],[41.2827,-1.9685],[41.219,-1.9225],[41.013,-2.0424],[41.0061,-1.9018],[40.9453,-2.0732],[40.861,-1.9635],[40.9448,-2.3093],[40.863,-2.2314],[40.7739,-2.2792],[40.8169,-2.3979],[40.6506,-2.5393],[40.4909,-2.5285],[40.2321,-2.669],[40.1626,-2.9343],[40.2329,-2.9819],[40.1229,-3.2673],[39.9891,-3.3732],[39.9661,-3.3212],[39.866,-3.6165],[39.7809,-3.5666],[39.867,-3.6949],[39.7864,-3.9172],[39.6838,-3.9172],[39.7453,-3.9687],[39.6995,-4.0484],[39.6605,-3.9708],[39.5606,-4.0398],[39.6659,-4.1121],[39.3962,-4.629],[39.3005,-4.595],[39.1906,-4.6775],[37.5991,-3.5134],[37.5852,-3.413],[37.7001,-3.3144],[37.6449,-3.046],[33.9042,-1.0026],[33.9535,-0.1544],[33.8936,0.1098],[34.0852,0.347],[34.1496,0.6036],[34.388,0.8158],[34.487,1.0825],[34.7979,1.2319],[34.779,1.3885],[34.9727,1.6542],[35.0065,1.9169],[34.8656,2.3475],[34.8756,2.5912],[34.434,3.182],[34.3813,3.4769],[34.4397,3.6677],[34.1525,3.7756],[34.205,3.8745],[34.0848,3.8773],[33.9771,4.2197],[34.3812,4.6202],[35.4116,5.0304],[35.3961,4.9264],[35.5708,4.9043],[35.5222,4.7805],[35.6106,4.62],[35.7058,4.6194]],"mw":[[34.9646,-11.5736],[34.6167,-11.5783],[34.5638,-11.8309],[34.3588,-12.1544],[34.4945,-12.6874],[34.5453,-13.3257],[34.6406,-13.4924],[34.8631,-13.5158],[35.0841,-13.7042],[35.8472,-14.6504],[35.9043,-14.8869],[35.7842,-15.1718],[35.8405,-15.4172],[35.7851,-16.0423],[35.5146,-16.1714],[35.3746,-16.141],[35.2588,-16.2715],[35.2282,-16.4703],[35.1319,-16.5376],[35.3102,-16.8506],[35.305,-17.1042],[35.2348,-17.1353],[35.066,-17.1098],[35.1183,-16.8334],[34.9055,-16.7408],[34.5247,-16.3008],[34.4265,-16.2739],[34.4045,-16.0597],[34.2332,-15.8898],[34.5652,-15.2973],[34.5018,-14.5757],[34.3441,-14.3874],[34.0508,-14.4952],[33.7204,-14.4897],[33.6449,-14.6017],[33.4392,-14.3807],[33.2802,-14.0608],[33.1403,-13.925],[32.9998,-14.0505],[32.8741,-13.8206],[32.763,-13.7772],[32.8134,-13.7112],[32.7699,-13.6469],[32.6633,-13.5963],[32.8131,-13.5282],[32.9982,-13.2024],[33.0136,-12.9172],[32.9411,-12.7699],[33.0248,-12.6127],[33.3504,-12.5327],[33.5271,-12.3556],[33.3495,-12.3279],[33.2527,-12.1395],[33.314,-11.7722],[33.2968,-11.5936],[33.2126,-11.5638],[33.2303,-11.4166],[33.3909,-11.1648],[33.2326,-10.8833],[33.501,-10.7692],[33.6742,-10.577],[33.5333,-10.2312],[33.3041,-10.0638],[33.3435,-9.8314],[33.1754,-9.6023],[32.9869,-9.6285],[32.9363,-9.3917],[33.1729,-9.511],[33.3003,-9.4922],[33.439,-9.6215],[33.7346,-9.5841],[33.9119,-9.718],[34.0127,-9.4775],[34.3179,-9.7192],[34.5364,-10.043],[34.5811,-10.5397],[34.675,-10.7524],[34.6135,-11.1022],[34.7656,-11.3451],[34.8866,-11.3754],[34.9646,-11.5736]],"tz":[[32.9209,-9.4079],[32.4234,-9.1438],[31.9381,-9.0618],[31.9366,-8.9326],[31.6727,-8.913],[31.3533,-8.5872],[30.9595,-8.5505],[30.2945,-7.1494],[29.707,-6.6193],[29.5255,-6.2731],[29.4761,-6.0021],[29.6124,-5.7044],[29.3238,-4.9202],[29.4042,-4.4498],[29.7328,-4.4633],[30.003,-4.2719],[30.3816,-3.7883],[30.4323,-3.5519],[30.8322,-3.1728],[30.8255,-2.9786],[30.4158,-2.8517],[30.5081,-2.4635],[30.8344,-2.3453],[30.8878,-2.0825],[30.831,-1.5942],[30.5553,-1.3184],[30.481,-1.0591],[34,-1.0026],[37.6449,-3.046],[37.7001,-3.3144],[37.5991,-3.5134],[39.2151,-4.6887],[38.7791,-6.0549],[38.915,-6.4367],[39.5537,-7.0007],[39.3278,-7.3093],[39.2526,-7.8233],[39.3278,-7.7263],[39.4435,-7.8326],[39.28,-8.3093],[39.3543,-8.714],[39.5537,-8.9223],[39.3887,-8.8989],[39.6438,-9.1891],[39.5674,-9.4475],[39.8006,-9.8238],[39.6908,-10.0434],[39.8269,-9.9903],[39.9851,-10.221],[40.0198,-10.1329],[40.1206,-10.2712],[40.2321,-10.2966],[40.2321,-10.2005],[40.3488,-10.3581],[40.427,-10.3001],[40.3892,-10.5353],[39.7858,-10.9326],[39.2382,-11.1742],[38.8963,-11.1677],[38.4923,-11.4135],[38.2566,-11.2771],[37.9381,-11.284],[37.8169,-11.5336],[37.4697,-11.7195],[37.0324,-11.5649],[36.6257,-11.7214],[36.1997,-11.7015],[36.1681,-11.5787],[35.8264,-11.4135],[35.5763,-11.6052],[34.9646,-11.5736],[34.6135,-11.1022],[34.675,-10.7524],[34.5364,-10.043],[34.3179,-9.7192],[34.0127,-9.4775],[33.9119,-9.718],[33.7346,-9.5841],[33.439,-9.6215],[32.9209,-9.4079]],"kr":[[128.3649,38.6243],[129.4302,37.0731],[129.3809,36.0415],[129.5873,36.0169],[129.4546,35.5133],[129.0004,35.046],[128.6021,35.2139],[128.6521,35.0613],[128.3714,35.0136],[128.4881,35.0512],[128.4353,34.8432],[128.0557,34.9316],[128.0693,35.0825],[127.7192,34.9901],[127.5753,34.9136],[127.7756,34.8491],[127.6407,34.618],[127.5213,34.8898],[127.3653,34.824],[127.5036,34.5913],[127.3494,34.5858],[127.4392,34.5407],[127.3436,34.445],[127.1245,34.5407],[127.3579,34.6982],[127.2481,34.7665],[126.898,34.4177],[126.7778,34.5865],[126.5213,34.2942],[126.5396,34.563],[126.2953,34.5891],[126.2903,34.7502],[126.4646,34.5847],[126.3786,34.7112],[126.6236,34.63],[126.364,34.7492],[126.6653,34.8081],[126.5676,34.9014],[126.3735,34.7907],[126.4364,34.9814],[126.3152,34.9379],[126.2542,35.1234],[126.4251,35.0198],[126.3089,35.2333],[126.4927,35.5253],[126.6925,35.5344],[126.4676,35.6396],[126.8375,35.8933],[126.5214,35.9705],[126.8707,36.062],[126.6925,35.9999],[126.4934,36.1308],[126.6033,36.1718],[126.4988,36.4201],[126.6174,36.4735],[126.4977,36.4484],[126.4796,36.7471],[126.2953,36.5827],[126.1282,36.7613],[126.3089,36.9669],[126.3567,36.7888],[126.353,37],[126.4839,36.8415],[126.5826,36.898],[126.5007,37.055],[126.6311,36.9116],[126.7962,36.9558],[126.8508,36.754],[127.042,36.9936],[126.7552,37.0488],[126.8645,37.1791],[126.6856,37.1301],[126.6585,37.2543],[126.8707,37.2679],[126.6151,37.3797],[126.5349,37.7682],[126.7552,37.6379],[126.6658,37.9376],[127.0905,38.2859],[128.0399,38.3043],[128.3649,38.6243]],"cg":[[18.6264,3.4769],[18.6131,3.1282],[18.0724,2.16],[18.0714,1.5515],[17.8666,1.0166],[17.9455,0.3773],[17.7164,-0.1958],[17.7495,-0.5234],[17.308,-1.015],[16.8332,-1.269],[16.514,-1.8874],[16.1946,-2.182],[16.2256,-3.3132],[15.9121,-3.9174],[15.5338,-4.058],[14.6727,-4.8999],[14.4007,-4.8863],[14.3871,-4.2771],[13.9445,-4.4981],[13.715,-4.453],[13.7125,-4.6857],[13.3926,-4.8853],[12.7617,-4.3912],[12.387,-4.6055],[12.3219,-4.7782],[12.1922,-4.7635],[12.0096,-5.0196],[11.8064,-4.5771],[11.114,-3.9369],[11.2125,-3.6973],[11.4826,-3.5093],[11.6876,-3.6987],[11.9055,-3.6435],[11.825,-3.5712],[11.9444,-3.3032],[11.6875,-3.171],[11.7784,-3.0057],[11.6368,-2.8333],[11.5307,-2.8667],[11.5662,-2.3297],[11.7561,-2.4151],[12.4593,-2.3299],[12.4269,-1.8841],[12.6324,-1.8252],[12.8046,-1.9191],[12.969,-2.3724],[13.4589,-2.441],[13.7432,-2.0971],[13.9062,-2.3598],[13.8458,-2.4666],[14.09,-2.4986],[14.4488,-1.6863],[14.3648,-1.612],[14.4877,-1.4094],[14.3918,-1.0233],[14.49,-0.6009],[13.8312,-0.2054],[13.9452,0.3504],[14.3288,0.6212],[14.4681,0.9132],[14.1445,1.3916],[13.7947,1.4344],[13.1502,1.2566],[13.2513,1.3376],[13.1296,1.5925],[13.1622,1.8998],[13.2946,2.1611],[14.5621,2.2088],[14.8925,2.0064],[15.7068,1.932],[16.0699,1.6546],[16.159,1.7348],[16.0858,2.1492],[16.4791,2.8363],[16.5985,3.5017],[17.4589,3.7083],[18.1816,3.4774],[18.4721,3.6321],[18.6264,3.4769]],"cd":[[18.6264,3.4769],[18.5419,4.3279],[19.409,5.1304],[19.8247,5.097],[20.5803,4.415],[22.3033,4.1287],[22.8984,4.8236],[23.4145,4.5908],[24.3964,5.1223],[24.7306,4.9175],[25.3076,5.0323],[25.5437,5.3753],[26.5025,5.0475],[27.3019,5.132],[28.3815,4.2753],[28.7554,4.554],[29.2213,4.3407],[29.6185,4.6416],[30.5252,3.8675],[30.543,3.6129],[30.8964,3.52],[30.7106,2.4451],[31.2715,2.103],[29.9473,0.8246],[29.5779,-1.3884],[28.8572,-2.5183],[29.2346,-3.0466],[29.5377,-6.3124],[30.2945,-7.1494],[30.7521,-8.1941],[28.8892,-8.4831],[28.8931,-8.7658],[28.3525,-9.272],[28.6985,-9.792],[28.6996,-10.6432],[28.3603,-11.5336],[29.0351,-12.3789],[29.4997,-12.4531],[29.4743,-12.2427],[29.7993,-12.1541],[29.7822,-13.4557],[29.5744,-13.2254],[28.994,-13.3959],[28.4227,-12.5213],[27.6382,-12.2936],[27.1801,-11.5696],[26.7174,-12.013],[25.9811,-11.903],[25.3449,-11.6422],[25.3103,-11.1948],[24.2939,-11.3992],[24.3705,-11.1002],[24.0015,-10.8734],[22.2376,-11.2495],[22.3134,-10.3686],[21.792,-9.4061],[21.785,-7.2834],[20.5205,-7.2864],[20.6115,-6.916],[19.5218,-7.0019],[19.3555,-8.002],[17.6002,-8.0985],[16.9457,-7.2046],[16.5529,-5.9024],[12.8962,-5.8145],[12.4495,-6.0519],[12.2493,-5.8288],[12.509,-5.7263],[12.4444,-5.0551],[13.088,-4.5795],[13.3926,-4.8853],[13.715,-4.453],[14.3871,-4.2771],[14.4007,-4.8863],[14.6727,-4.8999],[15.9121,-3.9174],[16.2256,-3.3132],[16.1946,-2.182],[16.8332,-1.269],[17.7495,-0.5234],[18.0724,2.16],[18.6264,3.4769]],"ua":[[31.7643,52.1006],[33.8041,52.3546],[34.4127,51.7777],[34.0891,51.6666],[34.186,51.2489],[35.0787,51.2076],[35.6114,50.3517],[36.5944,50.2044],[37.4152,50.4319],[38.0004,49.8998],[38.2999,50.0605],[40.1307,49.5798],[40.1417,49.2458],[39.6811,49.0203],[40.0538,48.8692],[39.6317,48.5869],[39.9932,48.2732],[39.7591,47.8329],[38.8309,47.8482],[38.2934,47.5628],[38.1423,47.0377],[37.5464,47.0816],[36.7693,46.6374],[35.9016,46.6527],[35,46.0736],[35.3421,46.3358],[35.1883,46.5124],[34.8123,46.1767],[34.9757,45.7173],[33.8462,46.1957],[32.5034,46.0761],[31.8011,46.2851],[32.0536,46.406],[31.5083,46.5876],[32.3176,46.4643],[32.6455,46.6496],[32.0228,46.6291],[31.7417,47.2591],[31.8994,46.6496],[31.4407,46.8005],[30.7907,46.5529],[29.8601,45.6689],[29.6382,45.8263],[29.7088,45.2233],[29.3221,45.4438],[28.577,45.2481],[28.2016,45.4689],[28.9569,46.0014],[28.9458,46.4548],[30.1316,46.4228],[29.5587,46.9457],[29.5566,47.324],[29.1556,47.45],[29.124,47.976],[27.583,48.486],[24.8966,47.7101],[24.5421,47.9438],[22.8776,47.9467],[22.1328,48.4048],[22.5396,49.0722],[22.867,49.01],[22.6658,49.5674],[24.1077,50.5408],[23.9576,50.808],[24.1432,50.8564],[23.6352,51.3047],[23.6167,51.6248],[25.7679,51.9285],[27.7143,51.4637],[28.21,51.652],[28.7288,51.4013],[29.0632,51.6306],[29.3199,51.3656],[30.1486,51.4844],[30.5396,51.2352],[30.5151,51.6037],[30.9188,52.0592],[31.7643,52.1006]],"na":[[16.4871,-28.5729],[16.3469,-28.5561],[15.7478,-28.0246],[15.2953,-27.3224],[15.2285,-26.9456],[15.083,-26.6952],[15.1687,-26.5922],[15.1056,-26.4206],[14.9663,-26.3352],[14.979,-26.0588],[14.8403,-25.7517],[14.8533,-25.0642],[14.4592,-24.0983],[14.5139,-23.8923],[14.4343,-23.4152],[14.492,-23.3092],[14.4102,-22.9708],[14.4351,-22.8798],[14.4592,-22.9953],[14.5293,-22.9002],[14.5086,-22.548],[13.3976,-20.8493],[13.1777,-20.1892],[13.0373,-20.0595],[12.45,-18.9046],[12.0319,-18.5054],[11.7969,-17.9561],[11.7176,-17.5463],[11.7551,-17.2667],[12.0881,-17.1392],[12.2426,-17.2248],[12.5672,-17.2346],[13.1663,-16.9511],[13.4798,-17.0103],[13.5213,-17.1219],[13.9806,-17.4236],[18.4536,-17.3899],[18.6428,-17.638],[18.89,-17.7993],[19.771,-17.8903],[20.3362,-17.8549],[20.8062,-18.0314],[21.1613,-17.9306],[21.3868,-18.0145],[24.2205,-17.4795],[25.037,-17.5812],[25.2598,-17.7941],[24.9533,-17.7884],[24.5644,-18.0528],[24.3506,-17.9561],[23.916,-18.2012],[23.6099,-18.4777],[23.2928,-17.999],[21.4757,-18.2995],[20.9751,-18.3193],[20.9848,-21.964],[19.9783,-22.0007],[19.9817,-28.4223],[19.5688,-28.5312],[19.4551,-28.7052],[19.2659,-28.7426],[19.2884,-28.8831],[19.1202,-28.9575],[18.7457,-28.8399],[18.1665,-28.9019],[17.9133,-28.7813],[17.6287,-28.7641],[17.5829,-28.6802],[17.4036,-28.7043],[17.4206,-28.5934],[17.3242,-28.4705],[17.3997,-28.3953],[17.3457,-28.2276],[17.2131,-28.2321],[17.0764,-28.0268],[16.893,-28.0826],[16.7687,-28.2654],[16.7404,-28.4809],[16.4871,-28.5729]],"za":[[19.9814,-24.7525],[20.6565,-25.4678],[20.8414,-26.1313],[20.6053,-26.4933],[20.6908,-26.8918],[21.6645,-26.8631],[22.5451,-26.2074],[23.0307,-25.2994],[23.4593,-25.2822],[23.9248,-25.6292],[24.665,-25.8233],[25.4584,-25.7112],[25.8684,-24.7482],[26.4044,-24.6328],[26.8392,-24.2656],[27.1274,-23.525],[27.7536,-23.2208],[28.3018,-22.6038],[28.9125,-22.4537],[29.0387,-22.2239],[29.6614,-22.1265],[30.335,-22.3447],[31.1526,-22.3164],[31.9866,-24.4231],[31.9958,-25.6369],[31.8919,-25.9838],[31.4015,-25.736],[31.1198,-25.91],[30.8046,-26.3975],[30.8024,-26.8089],[31.142,-27.1965],[31.5265,-27.3109],[31.9683,-27.3163],[31.9901,-26.8083],[32.8931,-26.8461],[32.8652,-27.0173],[32.3921,-28.5353],[31.365,-29.3326],[30.0628,-31.2387],[29.4116,-31.6775],[28.5486,-32.5624],[27.1009,-33.5253],[26.4761,-33.7643],[25.7854,-33.7436],[25.634,-33.8552],[25.6975,-34.0349],[24.9529,-33.9851],[24.8342,-34.2057],[23.6355,-33.9797],[23.4084,-34.11],[22.5837,-33.9941],[22.1692,-34.0832],[21.7887,-34.3813],[20.9134,-34.3604],[20.4827,-34.4705],[20.0156,-34.8189],[19.6532,-34.7741],[19.2969,-34.6178],[19.3602,-34.5004],[19.1399,-34.2944],[18.8176,-34.3769],[18.8001,-34.0895],[18.4784,-34.1069],[18.4701,-34.3474],[18.3163,-34.1575],[18.4844,-33.8439],[17.843,-32.8233],[18.2964,-32.6086],[18.2769,-31.8927],[17.282,-30.3478],[16.8241,-29.101],[16.47,-28.62],[17.0764,-28.0268],[17.3457,-28.2276],[17.4036,-28.7043],[18.1665,-28.9019],[19.1202,-28.9575],[19.5688,-28.5312],[19.9817,-28.4223],[19.9814,-24.7525]],"br":[[-57.6028,-30.1905],[-53.83,-27.1568],[-53.6616,-26.2598],[-54.6002,-25.5749],[-54.2453,-24.0506],[-55.4207,-23.9545],[-55.861,-22.2895],[-57.9862,-22.0745],[-58.1246,-19.7299],[-57.4657,-18.2177],[-58.3991,-17.2374],[-58.3396,-16.2893],[-60.1607,-16.2648],[-60.4726,-13.7979],[-64.9974,-11.9963],[-65.398,-9.6867],[-68.6157,-11.1125],[-70.6413,-11.0108],[-70.525,-9.4309],[-72.1957,-10.0056],[-73.215,-9.409],[-74.0185,-7.5435],[-72.9179,-5.1321],[-69.9718,-4.291],[-69.3995,-1.1827],[-70.0542,0.5881],[-69.1375,0.6501],[-69.8562,1.7077],[-67.4246,2.1381],[-67.0861,1.176],[-65.4978,0.6587],[-63.3726,2.2668],[-64.048,2.4713],[-64.7916,4.2844],[-62.8894,3.5608],[-60.2134,5.2672],[-59.5277,3.9395],[-59.7588,1.8621],[-58.834,1.1858],[-57.3194,1.9771],[-55.9538,1.8533],[-55.971,2.5303],[-52.9597,2.1752],[-51.5165,4.4411],[-50.7797,2.0709],[-49.8876,1.1749],[-52.7081,-1.6],[-50.8396,-0.9128],[-50.8442,-1.9553],[-49.422,-1.779],[-49.6637,-2.6777],[-48.0379,-0.6617],[-46.8192,-0.7117],[-44.598,-1.7448],[-44.796,-3.3002],[-43.4263,-2.3378],[-39.9857,-2.8484],[-37.1829,-4.9109],[-35.5106,-5.1437],[-34.7936,-7.1762],[-35.3051,-9.193],[-38.3204,-12.9352],[-38.9237,-12.7366],[-39.1371,-17.6796],[-40.9953,-22.0143],[-42.0135,-22.9817],[-44.4114,-22.9374],[-48.743,-25.3521],[-48.7561,-28.5284],[-52.0686,-32.1624],[-50.565,-30.4501],[-51.2835,-30.0074],[-51.2697,-30.7818],[-53.3791,-33.7407],[-53.1108,-32.7224],[-53.7573,-32.055],[-56.8313,-30.102],[-57.6028,-30.1905]],"ru":[[87.8163,49.1658],[80.0213,50.7669],[76.8959,54.4639],[68.9273,55.4337],[60.9961,53.9414],[61.3721,50.7825],[47.4698,50.4118],[46.4783,48.4106],[49.306,46.2631],[46.6804,44.5285],[47.8711,41.2082],[39.9626,43.3926],[36.5773,45.1942],[40.1307,49.5798],[31.7643,52.1006],[32.7012,53.4621],[27.3285,57.5706],[30.2461,59.9749],[27.8079,60.553],[31.5695,62.9059],[28.4474,68.5149],[30.841,69.8058],[41.0399,67.6778],[38.5544,66.055],[31.8509,67.1528],[37.4378,63.7923],[36.82,65.1471],[44.1797,65.8643],[43.3172,68.6666],[46.4067,66.7462],[60.9266,69.8675],[68.2881,68.1883],[66.6205,71.0578],[71.5632,72.916],[73.6463,68.4578],[68.965,66.8096],[72.0757,66.2358],[74.6279,68.7746],[79.0525,67.5705],[73.7588,69.1694],[74.7511,72.82],[83.2517,71.7257],[83.1111,70.0695],[80.5358,73.5808],[86.7825,73.0009],[87.0246,75.1642],[104.2903,77.7375],[113.8901,75.854],[105.073,72.7651],[124.3831,73.8064],[131.042,70.7388],[140.7258,72.8986],[159.2378,70.8293],[160.8211,68.5274],[180,68.9811],[180,65.0662],[174.4417,64.6925],[179.563,62.6247],[163.6287,60.0466],[161.9932,58.0874],[163.3661,56.1807],[156.67,50.8639],[156.749,57.7296],[165.6509,62.4592],[157.0276,61.6506],[155.1941,59.1758],[140.9043,58.3828],[135.1734,54.8213],[141.4075,53.3038],[140.177,48.4618],[133.1577,42.6884],[130.6628,42.3206],[130.9334,44.8417],[134.7187,48.2634],[130.9657,47.7039],[123.6143,53.5633],[117.8738,49.5132],[98.9276,52.1296],[97.3433,49.7342],[87.8163,49.1658]],"cz":[[14.8104,50.8584],[14.9821,50.8591],[14.961,50.9927],[15.1444,51.0116],[15.2698,50.9527],[15.3561,50.7755],[15.7922,50.7427],[15.9715,50.6786],[15.982,50.6036],[16.3316,50.644],[16.4259,50.5676],[16.1996,50.4063],[16.3437,50.3699],[16.6606,50.093],[17.0148,50.2185],[16.893,50.4329],[17.1876,50.3785],[17.4242,50.2406],[17.708,50.311],[17.7476,50.2175],[17.5894,50.1632],[17.6328,50.1063],[17.8394,49.9736],[18.0324,50.0028],[18.0024,50.0468],[18.2925,49.9078],[18.5592,49.9072],[18.6177,49.7139],[18.7884,49.6686],[18.8332,49.5103],[18.5356,49.4817],[18.1606,49.2587],[18.0755,49.0472],[17.7272,48.8629],[17.5351,48.813],[17.1675,48.8598],[16.945,48.6042],[16.873,48.719],[16.5195,48.8056],[16.3583,48.7273],[16.0854,48.7429],[15.8185,48.8723],[15.2746,48.9867],[15.1419,48.9375],[15.1374,48.993],[15.004,49.0098],[14.9396,48.7628],[14.8004,48.7765],[14.6756,48.5762],[14.4582,48.6432],[14.3254,48.5586],[14.0408,48.6012],[13.6087,48.9462],[13.4273,48.96],[12.9998,49.2949],[12.6435,49.4295],[12.3836,49.7429],[12.5241,49.905],[12.2469,50.045],[12.0761,50.3152],[12.3004,50.1608],[12.5102,50.3888],[12.9526,50.4042],[13.0096,50.4927],[13.1601,50.497],[13.3688,50.6283],[13.4479,50.5973],[13.5566,50.7067],[14.3465,50.8802],[14.3819,50.9209],[14.2383,50.9825],[14.2875,51.0368],[14.4821,51.0372],[14.5743,50.9755],[14.5504,50.9121],[14.6292,50.9207],[14.6132,50.8456],[14.8104,50.8584]],"de":[[13.8157,48.7664],[13.7165,48.5217],[13.4546,48.5734],[13.4056,48.3766],[12.7389,48.1134],[13.072,47.6595],[13.0019,47.466],[12.2422,47.732],[11.237,47.394],[10.4292,47.577],[10.1599,47.2711],[9.9459,47.5408],[8.5582,47.8012],[8.3913,47.6655],[8.5741,47.5924],[7.6097,47.5647],[7.5789,48.1145],[8.2003,48.9586],[6.7256,49.1556],[6.3453,49.4553],[6.5026,49.7957],[6.0964,50.0486],[6.3745,50.3155],[5.8582,51.0193],[6.2078,51.3877],[5.9315,51.8156],[6.7438,51.9083],[6.6799,52.0605],[7.0263,52.2306],[6.6717,52.5417],[7.0185,52.626],[7.3665,53.3028],[7.0236,53.3764],[7.2261,53.6665],[8.0314,53.7081],[8.2053,53.4107],[8.2707,53.6129],[8.5521,53.5436],[8.5044,53.3581],[8.5876,53.8704],[9.2107,53.872],[9.832,53.5436],[8.916,53.9374],[8.9631,54.3176],[8.5999,54.3381],[9.0116,54.5063],[8.6608,54.8963],[9.9475,54.7795],[10.0269,54.56],[9.8401,54.4753],[10.1434,54.4918],[10.1418,54.3244],[11.1355,54.3859],[10.7526,54.0501],[10.9021,53.9614],[12.115,54.0979],[12.5339,54.4883],[12.9212,54.4335],[12.4124,54.2514],[13.0088,54.4381],[13.4836,54.0913],[13.7119,54.1737],[13.8171,53.8529],[14.2645,53.7516],[14.4416,53.2518],[14.1239,52.8507],[14.6448,52.5769],[14.5454,52.3822],[14.7614,52.0767],[14.5858,51.8039],[14.9554,51.4354],[14.9553,51.064],[14.7592,50.8102],[14.2875,51.0368],[14.3465,50.8802],[12.5102,50.3888],[12.3004,50.1608],[12.0761,50.3152],[12.5241,49.905],[12.3836,49.7429],[12.6435,49.4295],[13.8157,48.7664]],"se":[[20.6232,69.0364],[23.6622,67.9504],[23.4313,67.4855],[23.7652,67.4201],[23.5813,67.1531],[24.004,66.8052],[23.6628,66.3165],[24.1631,65.8227],[22.653,65.9084],[22.2092,65.7612],[22.4053,65.5419],[21.7642,65.7277],[22.203,65.549],[21.2778,65.35],[21.5859,65.0692],[21.0374,64.8327],[21.6065,64.4522],[20.6747,63.7893],[19.4463,63.5609],[18.1951,62.7774],[17.6936,62.9936],[18.0642,62.5958],[17.3308,62.4877],[17.6527,62.232],[17.3446,61.9367],[17.4963,61.6351],[17.0563,61.5822],[17.1867,60.6924],[17.9749,60.5878],[19.0745,59.7497],[18.009,59.4085],[18.6465,59.3303],[17.8928,58.861],[17.6589,59.1689],[17.3578,58.7512],[16.1792,58.6345],[16.9402,58.4911],[16.4126,58.4775],[16.7992,58.3236],[16.7688,57.8878],[16.4603,57.9021],[16.7068,57.7506],[16.4194,57.8947],[16.689,57.4717],[15.8517,56.0864],[14.7249,56.1676],[14.2347,55.8627],[14.1812,55.3908],[12.8328,55.3826],[13.0549,55.694],[12.4514,56.3044],[12.8145,56.2362],[12.6216,56.4193],[12.9344,56.5471],[11.9033,57.3944],[11.6563,57.8394],[11.8828,58.3328],[11.2209,58.3463],[11.1082,58.9536],[11.6644,58.9201],[11.6749,59.6071],[12.6053,60.4192],[12.2533,61.0017],[12.8882,61.3496],[12.1617,61.7248],[12.3143,62.2861],[11.9924,63.2889],[12.7553,63.9914],[14.1478,64.169],[13.6428,64.584],[14.5143,65.3181],[14.5405,66.1254],[15.4819,66.2749],[16.3996,67.036],[16.1807,67.4964],[17.3215,68.1053],[17.921,67.9729],[18.1718,68.5359],[20.0103,68.3639],[20.3565,68.8046],[20.1023,69.0221],[20.6232,69.0364]],"tr":[[43.4404,41.1066],[43.7206,40.7629],[43.6654,40.1102],[44.3256,40.0347],[44.807,39.6399],[44.5657,39.7659],[44.388,39.4145],[44.0149,39.3741],[44.4588,38.3383],[44.2021,37.8972],[44.5961,37.7164],[44.7661,37.1419],[44.2848,36.9692],[44.088,37.3112],[42.7805,37.3755],[42.5452,37.1409],[42.2112,37.3249],[39.2396,36.6613],[38.2243,36.9084],[37.4461,36.6342],[36.6399,36.8281],[36.67,36.2373],[36.373,36.2276],[36.1385,35.8198],[35.7874,36.3043],[36.2112,36.6424],[36.0203,36.9263],[35.346,36.5411],[34.6929,36.8139],[33.959,36.2136],[32.8037,36.0272],[32.0188,36.5465],[30.6865,36.8912],[30.4274,36.2264],[29.6724,36.1234],[28.452,36.8836],[28.0359,36.5648],[28.1067,36.803],[27.4043,36.6652],[28.3294,37.0368],[27.2632,36.9634],[27.616,37.2747],[27.1908,37.3573],[27.0115,37.6657],[27.2391,37.9879],[26.2331,38.2803],[26.513,38.4329],[26.4141,38.6794],[26.6731,38.3072],[27.1642,38.4466],[26.7191,38.6527],[27.0627,38.8869],[26.6092,39.2752],[26.9377,39.5748],[26.07,39.4963],[26.1677,39.9736],[26.7356,40.4036],[29.0516,40.3673],[28.7742,40.5228],[28.9701,40.6395],[29.9393,40.7351],[29.2546,40.8108],[29.0076,41.0305],[29.1697,41.2331],[31.2312,41.092],[32.2634,41.7202],[33.3245,42.019],[35.0239,42.0925],[35.2982,41.7152],[36.0449,41.7015],[36.395,41.2564],[36.7581,41.3663],[38.3309,40.9179],[39.4302,41.1125],[40.1428,40.9228],[41.5208,41.5142],[42.661,41.5883],[43.4404,41.1066]],"es":[[-1.7941,43.386],[-1.4035,43.2433],[-1.4571,43.0451],[-1.286,43.1092],[-0.569,42.7726],[0.2754,42.6687],[0.8129,42.8316],[1.3426,42.7087],[1.4479,42.4346],[1.707,42.5028],[2.0092,42.3471],[3.181,42.4315],[3.3177,42.323],[3.1188,42.2183],[3.1949,41.8883],[2.1305,41.3037],[1.0174,41.0518],[0.7142,40.8152],[0.8581,40.6863],[0.5635,40.59],[-0.3234,39.5162],[-0.158,39.0021],[0.2345,38.7423],[-0.5098,38.3384],[-0.8561,37.7425],[-0.7053,37.6195],[-1.6709,37.3633],[-2.1267,36.7364],[-4.4032,36.7239],[-5.1866,36.4112],[-5.6118,36.0065],[-6.0368,36.1895],[-6.3947,36.6369],[-6.1948,36.9321],[-6.388,36.8082],[-6.8929,37.1666],[-6.8527,37.2952],[-6.9285,37.1723],[-7.4144,37.1928],[-7.5143,37.6014],[-6.9475,38.1966],[-7.3592,38.4464],[-6.9733,39.014],[-7.5573,39.6798],[-7.0211,39.694],[-6.8797,40.0092],[-7.0431,40.1814],[-6.7944,40.3564],[-6.9425,41.016],[-6.2059,41.5703],[-6.5415,41.6589],[-6.6095,41.9623],[-8.1796,41.8107],[-8.0951,42.0409],[-8.2224,42.1536],[-8.8813,41.8933],[-8.8955,42.1191],[-8.6227,42.3475],[-8.8675,42.2564],[-8.6553,42.4414],[-8.9426,42.4693],[-8.7309,42.6884],[-9.0264,42.5431],[-8.8877,42.8311],[-9.1002,42.7575],[-9.2731,42.8933],[-9.1077,43.1411],[-9.2176,43.1553],[-8.83,43.3412],[-8.2102,43.3132],[-8.3127,43.5671],[-7.6855,43.7934],[-7.0446,43.4792],[-5.8522,43.6629],[-4.4817,43.3836],[-3.5794,43.5183],[-3.0082,43.3209],[-2.7686,43.4525],[-2.1278,43.2874],[-1.7941,43.386]],"ly":[[11.5051,33.1812],[12.3483,32.8327],[12.8244,32.8043],[13.3517,32.9042],[13.5715,32.8028],[14.1853,32.7145],[14.4695,32.519],[15.182,32.3955],[15.3645,32.1747],[15.3611,31.9653],[15.4904,31.6649],[15.7615,31.3881],[16.0446,31.2755],[16.9419,31.1833],[17.8486,30.9237],[18.2083,30.7692],[18.6532,30.4223],[19.0568,30.2662],[19.6164,30.4214],[20.0999,30.935],[20.1467,31.2168],[19.9172,31.7592],[20.0715,32.1764],[20.5627,32.5578],[21.051,32.7723],[21.4148,32.7934],[21.6232,32.9368],[22.1496,32.9479],[23.1061,32.6398],[23.1489,32.4781],[23.0806,32.3341],[23.3122,32.1627],[23.2991,32.2242],[23.6697,32.1833],[24.0907,32.0058],[24.7337,32.0331],[25.0179,31.949],[25.1563,31.6631],[24.849,31.3486],[24.9949,30.7851],[24.6883,30.1442],[24.9812,29.1814],[24.9768,19.996],[23.9813,19.9954],[23.9813,19.4961],[15.9851,23.4447],[14.2162,22.6163],[13.4823,23.1797],[11.9689,23.5174],[11.5086,24.3138],[10.7207,24.5523],[10.4105,24.4733],[10.2603,24.5766],[10.1934,24.7499],[10.032,24.8563],[10.0079,25.3314],[9.3778,26.1689],[9.4826,26.3526],[9.8546,26.5244],[9.9106,26.8431],[9.8255,26.9206],[9.7219,27.3085],[9.9359,27.8667],[9.777,28.2676],[9.8261,29.1285],[9.6677,29.6083],[9.2865,30.1171],[9.8713,30.3552],[10.1922,30.7313],[10.2702,30.9156],[10.1062,31.4292],[10.264,31.6805],[10.4989,31.7443],[10.6059,31.9536],[10.7729,32.0045],[10.8732,32.1367],[11.5464,32.4343],[11.4499,32.638],[11.5051,33.1812]],"tn":[[11.5051,33.1812],[11.4499,32.638],[11.5464,32.4343],[10.8732,32.1367],[10.7729,32.0045],[10.6059,31.9536],[10.4989,31.7443],[10.264,31.6805],[10.1062,31.4292],[10.2539,30.8418],[9.8713,30.3552],[9.5197,30.2289],[9.045,32.0718],[8.3313,32.5276],[8.2829,32.8364],[8.0868,33.0943],[7.725,33.2314],[7.4849,33.855],[7.5175,34.095],[7.7653,34.2447],[7.8315,34.4144],[8.2365,34.6477],[8.3005,35.0677],[8.4313,35.2417],[8.2944,35.3251],[8.337,35.5379],[8.2414,35.8277],[8.3577,36.4304],[8.1699,36.5258],[8.4304,36.6626],[8.4131,36.7839],[8.6417,36.8364],[8.6025,36.9395],[8.8244,36.9795],[9.2107,37.2338],[9.7446,37.3452],[9.8589,37.3288],[9.7718,37.2133],[9.8159,37.1518],[9.9143,37.1783],[9.8196,37.2276],[9.8825,37.2636],[10.2721,37.1785],[10.1281,37.1587],[10.2271,37.1177],[10.1824,37.027],[10.3472,36.8803],[10.1885,36.7976],[10.3553,36.7312],[10.5187,36.7558],[10.585,36.8762],[11.051,37.0803],[11.1305,36.854],[10.7999,36.4519],[10.5509,36.3773],[10.4747,36.1125],[10.6221,35.8426],[11.041,35.6377],[11.0422,35.3344],[11.1592,35.2188],[10.597,34.5447],[10.1267,34.3256],[10.0052,34.1706],[10.0713,33.9462],[10.3303,33.7028],[10.4893,33.6471],[10.7166,33.7069],[10.6741,33.5456],[10.7381,33.478],[10.9094,33.5396],[10.9362,33.6363],[11.0428,33.6179],[11.1021,33.3641],[11.2937,33.2868],[11.1336,33.3114],[11.1795,33.2111],[11.4168,33.1838],[11.3519,33.2589],[11.5051,33.1812]],"pl":[[18.8332,49.5103],[18.5592,49.9072],[18.0024,50.0468],[17.8394,49.9736],[17.6328,50.1063],[17.5894,50.1632],[17.7476,50.2175],[17.708,50.311],[17.4242,50.2406],[16.893,50.4329],[17.0148,50.2185],[16.6606,50.093],[16.1996,50.4063],[16.4259,50.5676],[16.3316,50.644],[15.982,50.6036],[15.7922,50.7427],[15.3561,50.7755],[15.1444,51.0116],[14.961,50.9927],[14.9821,50.8591],[14.8104,50.8584],[15.0195,51.2717],[14.9451,51.4492],[14.71,51.5302],[14.7325,51.6583],[14.5858,51.8039],[14.7614,52.0767],[14.5454,52.3822],[14.6448,52.5769],[14.1239,52.8507],[14.4416,53.2518],[14.2639,53.7],[14.5906,53.5984],[14.6306,53.8515],[14.4319,53.9062],[14.3189,53.8182],[14.1753,53.9065],[16.1792,54.263],[16.5696,54.5572],[18.1524,54.8383],[18.7517,54.6901],[18.8176,54.5981],[18.4131,54.7319],[18.5881,54.4337],[18.8859,54.3502],[19.6095,54.4567],[22.8376,54.4009],[23.449,54.1549],[23.5909,53.6113],[23.8937,53.152],[23.9225,52.7426],[23.3923,52.5096],[23.1656,52.2894],[23.6375,52.0845],[23.5434,51.5927],[23.6976,51.4044],[23.6352,51.3047],[24.1432,50.8564],[23.9576,50.808],[24.1077,50.5408],[23.9813,50.4048],[23.6822,50.3682],[22.6658,49.5674],[22.6817,49.1612],[22.8553,48.994],[22.0408,49.1975],[21.6012,49.4265],[21.0688,49.4192],[20.919,49.2903],[20.6895,49.4005],[20.3177,49.3916],[20.0505,49.1732],[19.7607,49.1942],[19.7693,49.3931],[19.627,49.4019],[19.4573,49.5981],[19.1417,49.3942],[18.8332,49.5103]],"gb":[[-2.6654,51.6173],[-3.5392,51.3985],[-5.3109,51.864],[-3.9473,52.5493],[-4.0776,52.9267],[-4.7628,52.7895],[-4.178,53.2176],[-2.6998,53.3518],[-3.1019,53.5581],[-2.796,54.2487],[-3.1454,54.0638],[-3.6342,54.5129],[-3.0218,54.9755],[-4.8591,54.8663],[-4.8591,54.633],[-5.1571,54.8791],[-4.6178,55.4956],[-4.8865,55.9197],[-4.4817,55.9283],[-4.8516,55.988],[-4.7555,56.2065],[-4.9831,55.871],[-5.3383,55.9966],[-4.9205,56.2778],[-5.4444,56.0276],[-5.3178,55.783],[-5.7829,55.3135],[-5.5726,56.3281],[-4.9956,56.716],[-5.6741,56.4969],[-6.0082,56.6423],[-5.5437,56.6955],[-6.2345,56.7286],[-5.4004,57.1064],[-5.7293,57.2982],[-5.4481,57.4217],[-5.8078,57.3549],[-5.5127,57.5415],[-5.8112,57.8332],[-5.0708,57.8332],[-5.447,58.0844],[-4.9336,58.223],[-5.1732,58.3539],[-4.9882,58.6283],[-3.0191,58.6403],[-4.0093,57.8679],[-4.3924,57.9083],[-3.7794,57.8557],[-4.2489,57.4975],[-1.8295,57.6135],[-2.5146,56.5915],[-3.3235,56.3659],[-2.5768,56.2778],[-3.8375,56.1064],[-2.138,55.9146],[-1.6308,55.5851],[-1.2037,54.6257],[-0.0755,54.1121],[0.1312,53.5734],[-0.7265,53.7006],[0.2195,53.4195],[0.3566,53.1457],[0.0103,52.8865],[1.2747,52.9292],[1.7472,52.6252],[1.5823,52.0815],[0.3845,51.4531],[1.4238,51.3921],[1.3845,51.1517],[0.271,50.7474],[-1.4664,50.9182],[-2.4534,50.5277],[-3.4644,50.6785],[-3.6606,50.221],[-4.2006,50.4594],[-5.1912,49.9591],[-5.7051,50.0523],[-4.2285,51.1925],[-3.0218,51.1925],[-2.6654,51.6173]],"zm":[[32.9209,-9.4079],[32.9869,-9.6285],[33.1754,-9.6023],[33.3435,-9.8314],[33.3041,-10.0638],[33.5333,-10.2312],[33.6742,-10.577],[33.2326,-10.8833],[33.3909,-11.1648],[33.2303,-11.4166],[33.2527,-12.1395],[33.3495,-12.3279],[33.5271,-12.3556],[33.0248,-12.6127],[32.9982,-13.2024],[32.6633,-13.5963],[32.9998,-14.0505],[33.1403,-13.925],[33.2027,-14.0139],[30.2145,-14.9815],[30.4104,-15.6304],[29.5872,-15.6557],[28.9324,-15.9637],[28.7611,-16.5323],[27.8169,-16.9596],[27.1154,-17.8822],[26.7,-18.0692],[26.2119,-17.8828],[25.9245,-17.999],[25.6814,-17.8115],[25.3355,-17.8412],[25.037,-17.5812],[24.6844,-17.4924],[23.3817,-17.6411],[22.1081,-16.544],[21.9815,-16.1443],[21.9799,-13.0015],[24.0006,-13.0015],[23.8657,-12.7897],[24.0308,-12.3851],[23.9675,-10.8723],[24.3705,-11.1002],[24.2855,-11.3812],[24.4465,-11.4636],[25.3103,-11.1948],[25.3449,-11.6422],[25.9811,-11.903],[26.7174,-12.013],[26.9477,-11.9091],[27.0238,-11.5956],[27.1801,-11.5696],[27.6382,-12.2936],[27.8567,-12.2624],[28.4227,-12.5213],[28.5428,-12.8856],[28.708,-12.8898],[28.994,-13.3959],[29.1841,-13.4369],[29.5744,-13.2254],[29.6174,-13.4212],[29.7672,-13.4584],[29.7993,-12.1541],[29.4743,-12.2427],[29.4997,-12.4531],[29.0351,-12.3789],[28.5079,-11.8728],[28.3557,-11.5182],[28.6996,-10.6432],[28.5699,-10.2196],[28.6985,-9.792],[28.3525,-9.272],[28.8931,-8.7658],[28.8892,-8.4831],[30.7521,-8.1941],[31.0336,-8.6003],[31.3533,-8.5872],[31.6727,-8.913],[31.9366,-8.9326],[31.9381,-9.0618],[32.9209,-9.4079]],"sl":[[-10.2822,8.4846],[-10.3209,8.1831],[-10.5292,8.1243],[-10.6121,8.0319],[-10.6152,7.769],[-11.3172,7.2136],[-11.4762,6.9194],[-11.8396,7.149],[-12.5066,7.3898],[-12.3695,7.3898],[-12.4892,7.4451],[-12.4283,7.5336],[-12.2951,7.5331],[-12.1838,7.6027],[-12.4637,7.562],[-12.4855,7.6505],[-12.5439,7.6536],[-12.452,7.7741],[-12.6019,7.6838],[-12.7781,7.7558],[-12.7598,7.8218],[-12.8287,7.7983],[-12.9379,7.8864],[-12.9055,7.9341],[-12.8213,7.9174],[-12.9118,8.0307],[-12.8759,8.103],[-12.9811,8.1862],[-12.9721,8.2539],[-13.1639,8.1713],[-13.1639,8.2744],[-13.2868,8.4245],[-13.2868,8.5002],[-13.0336,8.3767],[-13.1161,8.473],[-13.0131,8.5617],[-12.8697,8.5692],[-12.9311,8.5828],[-12.8759,8.6852],[-12.9364,8.607],[-13.0618,8.5832],[-13.0267,8.6511],[-13.0819,8.6169],[-13.1577,8.671],[-13.1075,8.581],[-13.1624,8.5176],[-13.2412,8.6722],[-13.2402,8.8258],[-12.9857,8.8571],[-13.1304,8.8641],[-13.1426,8.9167],[-13.0751,8.9391],[-13.267,8.9521],[-13.3011,9.0415],[-13.1159,9.0439],[-12.9592,9.1778],[-12.9579,9.2724],[-12.8358,9.282],[-12.7642,9.3477],[-12.4723,9.8813],[-12.2767,9.9293],[-12.1413,9.8749],[-11.9218,9.9222],[-11.9101,9.9925],[-11.2473,9.9934],[-10.8863,9.5684],[-10.8318,9.3921],[-10.6751,9.3065],[-10.7463,9.0838],[-10.5941,9.0541],[-10.5997,8.8154],[-10.4819,8.6725],[-10.6319,8.532],[-10.7353,8.2873],[-10.6598,8.3333],[-10.5544,8.3118],[-10.3898,8.4891],[-10.2822,8.4846]],"gn":[[-13.3011,9.0415],[-13.1577,9.1855],[-13.4078,9.287],[-13.4867,9.5691],[-13.7262,9.5111],[-13.5611,9.7882],[-13.7289,9.7441],[-13.6853,9.9465],[-13.8428,9.8608],[-14.0583,10.0253],[-13.9999,10.1997],[-14.1532,10.0557],[-14.6578,10.4796],[-14.5002,10.8911],[-14.6998,10.6431],[-14.6514,10.81],[-14.7603,10.6994],[-14.8197,10.912],[-14.6852,11.0637],[-14.9186,11.029],[-14.9596,10.7683],[-15.0795,10.8587],[-14.7111,11.4976],[-13.7299,11.7098],[-13.7239,12.0028],[-13.9725,12.1516],[-13.695,12.2929],[-13.7283,12.6734],[-13.0765,12.6359],[-13.052,12.4723],[-12.3609,12.3056],[-11.4816,12.4282],[-11.5078,12.1916],[-11.3347,12.0262],[-10.9521,12.2195],[-10.7114,11.8904],[-10.2669,12.2178],[-9.723,12.0254],[-9.3368,12.2698],[-9.4126,12.4554],[-9.2659,12.4952],[-8.9759,12.3687],[-8.8472,11.6579],[-8.3753,11.3677],[-8.7007,10.9785],[-8.3266,11.0239],[-8.2844,10.5112],[-7.9712,10.2818],[-8.1738,9.9419],[-8.1576,9.5309],[-7.8654,9.4099],[-7.9296,9.1839],[-7.7468,9.0764],[-7.9689,8.8128],[-7.6968,8.6248],[-7.6624,8.3749],[-8.2489,8.4536],[-8.266,8.246],[-8.0033,8.184],[-7.9626,8.0152],[-8.2002,7.5763],[-8.6866,7.6944],[-8.8578,7.273],[-9.1254,7.1902],[-9.2343,7.3813],[-9.4907,7.376],[-9.3555,7.7419],[-9.4991,8.3435],[-9.7736,8.563],[-10.7353,8.2873],[-10.4819,8.6725],[-10.5941,9.0541],[-10.7463,9.0838],[-10.6751,9.3065],[-11.1996,9.9822],[-12.4723,9.8813],[-12.7642,9.3477],[-13.3011,9.0415]],"lr":[[-11.4762,6.9194],[-11.3172,7.2136],[-10.6152,7.769],[-10.6121,8.0319],[-10.5292,8.1243],[-10.3209,8.1831],[-10.2822,8.4846],[-10.0911,8.5101],[-10.0617,8.4222],[-9.8068,8.5063],[-9.7736,8.563],[-9.7221,8.4356],[-9.7033,8.4873],[-9.6505,8.4648],[-9.6773,8.3915],[-9.6002,8.4109],[-9.4991,8.3435],[-9.5394,8.3146],[-9.5301,8.1855],[-9.4814,8.1647],[-9.4742,8.0288],[-9.4241,8.0262],[-9.4387,7.8662],[-9.3555,7.7419],[-9.3535,7.6224],[-9.4907,7.376],[-9.4381,7.4216],[-9.4074,7.3781],[-9.3492,7.4216],[-9.2343,7.3813],[-9.1254,7.1902],[-8.9572,7.2865],[-8.9122,7.2502],[-8.7296,7.5087],[-8.6866,7.6944],[-8.5674,7.6882],[-8.5665,7.6234],[-8.4256,7.5019],[-8.2841,7.0179],[-8.3375,6.7773],[-8.6187,6.4928],[-8.4972,6.4314],[-8.4675,6.4871],[-8.4183,6.4512],[-8.4118,6.34],[-8.3529,6.3611],[-8.1706,6.274],[-7.9144,6.2717],[-7.8466,6.1975],[-7.8618,6.0815],[-7.8097,6.0712],[-7.7866,5.9582],[-7.7172,5.9008],[-7.6805,5.9409],[-7.4944,5.8062],[-7.4465,5.8459],[-7.3841,5.569],[-7.45,5.4228],[-7.3898,5.3171],[-7.4188,5.2605],[-7.4757,5.2642],[-7.4889,5.1407],[-7.5758,5.0721],[-7.5495,4.9139],[-7.6055,4.8974],[-7.5792,4.3871],[-7.5407,4.3528],[-7.7127,4.3615],[-7.942,4.5049],[-8.2573,4.58],[-9.105,5.0316],[-9.5937,5.4303],[-9.597,5.4883],[-9.9796,5.7981],[-10.0555,5.9118],[-10.3537,6.1024],[-10.3336,6.1361],[-10.7883,6.293],[-10.7594,6.4042],[-11.3532,6.7012],[-11.3819,6.8368],[-11.4762,6.9194]],"cf":[[22.8611,10.9192],[23.6448,9.8631],[23.6322,9.2776],[23.4357,9.0189],[23.5675,8.9748],[23.5053,8.7107],[24.2177,8.6915],[24.1529,8.3179],[24.8002,8.1803],[25.2169,7.8641],[25.2794,7.6595],[25.1904,7.5012],[26.378,6.6533],[26.2702,6.4657],[26.509,6.2055],[26.4248,6.0724],[27.1238,5.7687],[27.2611,5.5776],[27.2645,5.2601],[27.4413,5.0707],[27.0737,5.2033],[26.8671,5.0375],[26.5025,5.0475],[26.1983,5.2384],[25.9109,5.1701],[25.5437,5.3753],[25.3637,5.3106],[25.3076,5.0323],[24.7306,4.9175],[24.3964,5.1223],[24.2592,4.9299],[23.4145,4.5908],[22.8984,4.8236],[22.5924,4.4737],[22.5105,4.1913],[22.3033,4.1287],[21.7373,4.2934],[21.2109,4.2916],[20.8715,4.4534],[20.5803,4.415],[20.3389,4.7719],[19.8247,5.097],[19.409,5.1304],[19.0833,4.9093],[18.7533,4.4006],[18.5419,4.3279],[18.6465,4.046],[18.6264,3.4769],[18.4721,3.6321],[18.1816,3.4774],[17.4589,3.7083],[16.6394,3.5286],[16.4655,3.1565],[16.4791,2.8363],[16.1797,2.247],[16.0554,2.9398],[15.0849,3.8858],[15.026,4.0261],[15.1898,4.0631],[14.9939,4.4133],[14.7173,4.622],[14.6589,5.1645],[14.5199,5.2906],[14.6174,5.865],[14.3873,6.039],[14.7723,6.3183],[15.2239,7.2475],[15.481,7.5233],[15.9253,7.4882],[16.5487,7.87],[16.7824,7.5411],[17.6399,7.985],[18.5893,8.0479],[19.1241,8.6751],[18.8698,8.8494],[19.0219,8.9852],[20.4352,9.1381],[21.7188,10.2966],[21.7226,10.6367],[22.4471,10.9983],[22.8611,10.9192]],"sd":[[22.8611,10.9192],[22.9537,11.2509],[22.5417,11.633],[22.6124,12.0728],[22.4581,12.0304],[22.4457,12.6111],[22.2049,12.7434],[21.936,12.6395],[21.8094,12.7937],[22.2676,13.3346],[22.075,13.7803],[22.5464,14.1397],[22.3635,14.5435],[22.6766,14.689],[22.6589,14.8574],[22.958,15.2016],[22.9067,15.5414],[23.0946,15.7043],[23.9844,15.7212],[23.9813,19.9954],[24.9805,20.0021],[24.9812,21.9954],[31.2484,21.9944],[31.4233,22.227],[31.4355,21.9954],[33.1811,21.9954],[33.5609,21.7091],[33.9991,21.7674],[34.0842,21.9955],[36.8836,21.9957],[36.8879,21.6501],[37.313,21.0577],[37.1214,21.2243],[37.0864,21.0406],[37.4343,18.8617],[38.6039,18.0139],[38.2334,17.5257],[37.496,17.3122],[37.4034,17.0299],[36.9735,17.0625],[36.9543,16.2783],[36.4251,15.1334],[36.5264,14.2635],[36.1236,12.7214],[35.6877,12.6592],[35.0727,11.8189],[34.8385,10.7291],[34.5748,10.8841],[34.2795,10.5655],[34.3259,10.2236],[34.0707,9.4546],[33.8814,9.4992],[33.917,10.1745],[33.14,10.739],[33.0832,11.5992],[33.2092,12.2104],[32.7318,12.2162],[32.7467,12.0027],[32.0822,11.9998],[32.3483,11.7032],[32.4307,11.0822],[31.1644,9.764],[30.7653,9.7243],[30.013,10.2705],[29.6154,10.0581],[29.4831,9.7617],[28.9138,9.529],[28.8438,9.3245],[28.0451,9.3314],[27.8951,9.5954],[26.6257,9.4807],[25.9054,10.1813],[25.8285,10.4195],[25.0769,10.2869],[24.7964,9.8189],[24.5584,8.8867],[24.114,8.6816],[23.4901,8.7326],[23.6448,9.8631],[22.8611,10.9192]],"dj":[[43.2407,11.4879],[42.9237,10.9988],[42.9081,11.0042],[42.8733,10.9782],[42.7718,10.9965],[42.7296,11.065],[42.6156,11.0897],[42.4797,11.0591],[42.3918,11.0059],[42.3385,11.0155],[42.2216,10.9843],[42.0963,10.9903],[42.0384,10.9504],[41.9375,10.9298],[41.9169,10.9471],[41.7738,10.9801],[41.7613,10.9962],[41.7856,11.0674],[41.788,11.2597],[41.7491,11.538],[41.7924,11.7045],[41.8082,11.7362],[41.8623,11.7631],[41.8778,11.7901],[41.9364,11.8274],[42.0977,12.0702],[42.1323,12.0935],[42.1555,12.1423],[42.2887,12.321],[42.3192,12.3864],[42.4309,12.5203],[42.5216,12.4962],[42.6783,12.36],[42.7499,12.4171],[42.7923,12.4292],[42.7852,12.5172],[42.8288,12.5576],[42.8688,12.6262],[42.9004,12.6164],[43.1177,12.7079],[43.2832,12.4965],[43.313,12.4777],[43.3294,12.4888],[43.3312,12.4367],[43.3605,12.383],[43.3678,12.3069],[43.4143,12.2249],[43.4133,12.0593],[43.3771,12.0048],[43.2883,11.9674],[43.2066,11.9578],[43.0972,11.8445],[43.0692,11.8303],[43.0555,11.8036],[42.9868,11.7961],[42.9426,11.7738],[42.8965,11.7806],[42.7772,11.7322],[42.668,11.5685],[42.6277,11.5591],[42.569,11.5634],[42.5486,11.5838],[42.5151,11.5741],[42.5281,11.5702],[42.5387,11.503],[42.6038,11.4678],[42.6448,11.4882],[42.6817,11.4845],[42.6875,11.4972],[42.6653,11.5156],[42.6941,11.546],[42.8291,11.577],[42.8558,11.5958],[42.8752,11.5837],[43.0818,11.5895],[43.1034,11.577],[43.1519,11.6118],[43.1838,11.5266],[43.2407,11.4879]],"er":[[43.1177,12.7079],[42.8688,12.6262],[42.7923,12.4292],[42.6783,12.36],[42.4757,12.5163],[42.3795,12.4659],[42.1849,12.733],[41.9466,12.8762],[41.7118,13.2479],[41.192,13.6159],[40.8332,14.106],[40.1046,14.466],[39.8923,14.4266],[39.515,14.5678],[39.2082,14.4402],[39.1275,14.6022],[39.0097,14.6512],[38.8661,14.4937],[38.4268,14.4172],[38.2293,14.6797],[38.0067,14.7244],[37.8915,14.8795],[37.5537,14.1055],[37.2938,14.4574],[37.1211,14.4208],[37.0602,14.2768],[36.7481,14.3321],[36.5264,14.2635],[36.4251,15.1334],[36.6279,15.4456],[36.671,15.7231],[36.9543,16.2783],[36.8765,16.5136],[36.9997,16.8421],[36.9735,17.0625],[37.4034,17.0299],[37.4947,17.1746],[37.496,17.3122],[37.7204,17.3838],[37.7989,17.4771],[37.8959,17.4426],[38.0388,17.5482],[38.0765,17.4799],[38.1195,17.5456],[38.2334,17.5257],[38.4593,17.8907],[38.6016,18.0048],[39.0125,17.1654],[39.2941,15.9288],[39.437,15.812],[39.4575,15.5246],[39.6113,15.5],[39.7183,15.2639],[39.7156,15.0941],[39.8113,15.0789],[39.8547,15.1735],[39.7734,15.3905],[39.8753,15.5041],[40.0799,15.3407],[40.0266,15.2372],[40.1594,14.9822],[40.2868,14.9135],[40.3898,15.0032],[40.499,14.9622],[40.5127,15.0175],[40.6277,14.892],[40.6945,14.8962],[40.7664,14.6953],[41.1675,14.6369],[41.6765,13.9403],[41.9697,13.8456],[42.1066,13.6435],[42.2239,13.5539],[42.1848,13.6675],[42.2884,13.5746],[42.3794,13.2226],[42.5339,13.2326],[42.7297,13.0331],[42.7813,12.8519],[42.928,12.7897],[43.001,12.9003],[43.1177,12.7079]],"it":[[7.0221,45.9253],[7.8437,45.9192],[8.3992,46.4522],[8.4381,46.2354],[9.0024,45.8207],[9.2823,46.4974],[10.0757,46.22],[10.0328,46.533],[10.444,46.5377],[10.4538,46.8644],[10.9969,46.7691],[11.1745,46.9639],[12.1806,47.0852],[12.1269,46.9089],[12.405,46.6901],[13.701,46.5197],[13.3653,46.2903],[13.6374,46.1804],[13.4618,46.0064],[13.8478,45.5847],[13.5742,45.7896],[13.1193,45.772],[12.1545,45.3135],[12.5339,44.97],[12.2664,44.8266],[12.3688,44.2507],[13.6156,43.5605],[14.0754,42.5988],[14.7407,42.0846],[16.1405,41.9199],[16.1843,41.7796],[15.9001,41.6147],[15.9611,41.4595],[18.007,40.6507],[18.5161,40.1392],[18.3892,39.8171],[18.0715,39.9115],[17.8631,40.2856],[16.9761,40.4926],[16.4898,39.7752],[17.1587,39.4062],[17.1724,38.9605],[16.5856,38.7978],[16.5696,38.4295],[16.0772,37.9399],[15.6844,37.9535],[15.6328,38.2203],[15.9178,38.5168],[15.8366,38.649],[16.2217,38.8566],[15.6243,40.0778],[15.2612,40.0294],[14.9111,40.2416],[14.9983,40.3976],[14.782,40.6699],[14.3426,40.5706],[14.4741,40.7297],[14.0477,40.7896],[13.7219,41.2524],[13.0446,41.2275],[11.6582,42.2794],[11.1066,42.3913],[11.1585,42.564],[10.7635,42.9183],[10.4995,42.9405],[10.5282,43.2468],[10.0906,44.0254],[8.7624,44.4321],[8.0677,43.8967],[7.5023,43.7922],[7.6556,44.1761],[6.9734,44.2494],[6.8357,44.534],[7.0552,44.6849],[7.0049,44.828],[6.6027,45.1034],[7.1084,45.2592],[7.1607,45.4109],[6.782,45.7775],[7.0221,45.9253]],"ci":[[-7.9897,10.162],[-7.5015,10.4586],[-7.3634,10.2522],[-7.0158,10.1409],[-6.9612,10.3449],[-6.6689,10.361],[-6.6692,10.6542],[-6.446,10.5523],[-6.427,10.6861],[-6.256,10.7265],[-6.24,10.2519],[-6.0159,10.1899],[-5.8781,10.3761],[-5.594,10.4539],[-5.4025,10.3006],[-5.1357,10.3043],[-4.9661,9.901],[-4.6811,9.6821],[-4.51,9.7451],[-4.3712,9.583],[-4.1503,9.8183],[-3.755,9.9359],[-3.2148,9.9156],[-2.8104,9.4111],[-2.6892,9.4887],[-2.7748,9.055],[-2.6194,8.9236],[-2.5063,8.2093],[-2.7906,7.9433],[-3.2625,6.6171],[-3.0298,5.7043],[-2.7856,5.595],[-2.7666,5.161],[-3.1391,5.1423],[-3.1461,5.3688],[-3.2621,5.3409],[-3.3122,5.1199],[-3.987,5.244],[-3.7289,5.2665],[-3.8049,5.376],[-3.744,5.274],[-3.9063,5.3484],[-4.6872,5.2317],[-4.6735,5.3136],[-4.7834,5.1703],[-3.9958,5.2317],[-4.8926,5.1286],[-5.3215,5.2302],[-5.3579,5.1235],[-5.2284,5.2044],[-5.0024,5.1286],[-5.8513,5.03],[-7.5738,4.3754],[-7.5758,5.0721],[-7.3898,5.3171],[-7.4465,5.8459],[-7.7866,5.9582],[-7.9144,6.2717],[-8.4118,6.34],[-8.4183,6.4512],[-8.6187,6.4928],[-8.2866,6.9938],[-8.4854,7.558],[-8.229,7.5443],[-8.0691,8.0293],[-7.9626,8.0152],[-8.0033,8.184],[-8.266,8.246],[-8.2544,8.4465],[-7.9705,8.497],[-7.6624,8.3749],[-7.6968,8.6248],[-7.9689,8.8128],[-7.9318,8.9966],[-7.7468,9.0764],[-7.9296,9.1839],[-7.8654,9.4099],[-8.0541,9.3891],[-8.1576,9.5309],[-8.1738,9.9419],[-7.9897,10.162]],"ml":[[-12.2641,14.7749],[-11.8204,14.9],[-11.7271,15.5412],[-11.5244,15.6435],[-10.9152,15.1029],[-10.7254,15.4331],[-9.4239,15.4405],[-9.3565,15.6979],[-9.3492,15.4956],[-5.511,15.4942],[-5.3533,16.312],[-5.6233,16.5278],[-6.5931,24.9941],[-4.8216,24.9951],[1.1465,21.1017],[1.1546,20.7388],[1.5597,20.5975],[1.7781,20.3043],[2.2008,20.2739],[2.4004,20.0566],[3.1988,19.8205],[3.2608,19.3883],[3.1041,19.1355],[3.2849,18.9957],[4.2286,19.1422],[4.184,16.4161],[3.8712,15.7148],[3.5071,15.354],[1.3315,15.2836],[0.6701,14.9397],[-0.7197,15.0785],[-1.9971,14.4707],[-2.0359,14.1812],[-2.5974,14.2223],[-2.8673,14.0005],[-2.8952,13.6516],[-3.287,13.6978],[-3.2486,13.2928],[-3.4495,13.1688],[-3.9813,13.499],[-4.3511,13.1183],[-4.2119,12.8192],[-4.4853,12.7147],[-4.4061,12.3075],[-4.7613,12.0066],[-5.4124,11.8285],[-5.2183,11.4222],[-5.5042,11.0727],[-5.4817,10.5353],[-6.0159,10.1899],[-6.24,10.2519],[-6.256,10.7265],[-6.446,10.5523],[-6.6692,10.6542],[-6.6689,10.361],[-6.9612,10.3449],[-7.0158,10.1409],[-7.5015,10.4586],[-7.9897,10.162],[-8.2955,10.5374],[-8.3266,11.0239],[-8.6962,10.9627],[-8.3753,11.3677],[-8.8472,11.6579],[-8.8198,12.0127],[-9.1473,12.4655],[-9.3982,12.4732],[-9.3368,12.2698],[-9.723,12.0254],[-10.2669,12.2178],[-10.7114,11.8904],[-10.9521,12.2195],[-11.3347,12.0262],[-11.5078,12.1916],[-11.3888,12.9706],[-11.6131,13.3608],[-11.8281,13.3073],[-12.0976,13.7044],[-11.9976,14.1662],[-12.2217,14.3912],[-12.2641,14.7749]],"sn":[[-12.2641,14.7749],[-12.165,14.6419],[-12.2217,14.3912],[-11.9976,14.1662],[-11.9629,13.8295],[-12.0976,13.7044],[-11.8281,13.3073],[-11.7412,13.4078],[-11.6131,13.3608],[-11.3888,12.9706],[-11.4678,12.5435],[-11.3778,12.4802],[-12.3609,12.3056],[-12.9131,12.5363],[-13.052,12.4723],[-13.0765,12.6359],[-13.7283,12.6734],[-15.1951,12.6794],[-15.677,12.4393],[-16.7284,12.3325],[-16.8008,12.4945],[-16.5797,12.6328],[-15.6448,12.5373],[-15.3916,12.8382],[-15.5495,12.7943],[-15.641,12.5595],[-16.0221,12.729],[-16.371,12.5645],[-16.5013,12.7079],[-16.5082,12.6254],[-16.6347,12.6694],[-16.5969,12.7973],[-16.7546,12.5722],[-16.7084,13.1567],[-15.8333,13.1568],[-15.8187,13.3335],[-15.2771,13.3803],[-15.1377,13.59],[-14.3688,13.2357],[-13.9006,13.3146],[-13.8187,13.4294],[-13.9885,13.5791],[-14.3656,13.4548],[-14.5426,13.6408],[-14.7409,13.6154],[-15.0763,13.819],[-15.3941,13.7715],[-15.5182,13.5831],[-16.5614,13.5869],[-16.5089,13.7402],[-16.5989,13.6542],[-16.6298,13.7605],[-16.487,13.8046],[-16.487,14.0026],[-16.7001,13.769],[-16.7468,13.9514],[-16.3647,14.1678],[-16.6542,14.0028],[-16.7279,14.0647],[-16.7751,13.8388],[-16.9565,14.3901],[-17.2241,14.6904],[-17.4174,14.7377],[-17.4398,14.6538],[-17.536,14.7574],[-17.173,14.9008],[-16.8782,15.2331],[-16.5343,15.7844],[-16.2951,16.5106],[-15.6722,16.4814],[-14.9743,16.6914],[-14.4066,16.6608],[-13.8358,16.1206],[-13.4849,16.155],[-13.2378,15.6234],[-12.9643,15.5092],[-12.8884,15.2509],[-12.2641,14.7749]],"ng":[[3.5964,11.6958],[3.6416,12.518],[4.0888,12.9962],[4.1258,13.473],[5.5218,13.8803],[6.3689,13.6263],[6.9371,12.9954],[7.824,13.3453],[8.6793,12.9236],[9.5903,12.8014],[10.1198,13.2501],[10.675,13.3751],[11.44,13.3644],[12.4727,13.0642],[12.8423,13.4891],[13.6073,13.7041],[14.0649,13.078],[14.1889,12.3704],[14.6699,12.1674],[14.6058,11.5147],[13.7548,11.0167],[13.4349,10.1534],[13.2302,10.0453],[13.1955,9.5422],[12.8554,9.3781],[12.7729,8.7591],[12.227,8.3863],[12.1921,7.9607],[11.7366,7.2633],[11.8722,7.0794],[11.3698,6.4558],[11.1135,6.434],[11.0367,6.7405],[10.5787,7.1308],[10.496,6.8747],[10.1429,7.0077],[8.8559,5.8475],[8.7989,5.1596],[8.5298,4.7089],[8.1817,4.9914],[8.2886,4.5486],[7.5584,4.5253],[7.5448,4.711],[7.281,4.5049],[7.0718,4.7581],[7.1846,4.519],[7.044,4.4366],[6.9627,4.7309],[6.9695,4.3746],[6.7156,4.8345],[6.8708,4.3608],[6.7156,4.3547],[6.7361,4.6079],[6.6877,4.3366],[6.5505,4.5049],[6.5696,4.3349],[6.2705,4.2926],[6.2363,4.498],[6.1128,4.2722],[5.4754,4.8475],[5.3402,5.3439],[5.6399,5.5402],[5.4338,5.3956],[5.198,5.5015],[5.5096,5.5906],[5.3376,5.7246],[5.1292,5.6281],[5.2904,5.9101],[5.0447,5.7675],[4.4055,6.3589],[3.4212,6.414],[3.844,6.6108],[2.7038,6.3684],[2.7692,9.057],[3.076,9.0952],[3.8374,10.5999],[3.4684,11.4195],[3.5964,11.6958]],"bj":[[3.5964,11.6958],[3.4684,11.4195],[3.7221,11.1123],[3.73,10.8077],[3.8229,10.7032],[3.8374,10.5999],[3.7725,10.4076],[3.6521,10.4358],[3.5724,10.2857],[3.6675,10.1466],[3.5895,9.9488],[3.5131,9.8466],[3.3259,9.8017],[3.3316,9.652],[3.2457,9.6337],[3.1326,9.4573],[3.1514,9.2886],[3.076,9.0952],[2.7692,9.057],[2.7232,8.7829],[2.7448,8.4983],[2.6971,8.3207],[2.7344,8.19],[2.6711,7.8979],[2.7272,7.7933],[2.7206,7.5866],[2.7923,7.4776],[2.7412,7.4082],[2.7353,7.096],[2.7797,7.0413],[2.7151,6.9798],[2.7236,6.7997],[2.7786,6.6988],[2.7311,6.6418],[2.7038,6.3684],[1.6196,6.2139],[1.7824,6.2773],[1.7441,6.4257],[1.566,6.6781],[1.5901,6.8673],[1.5319,6.9919],[1.6257,6.9968],[1.6011,8.3716],[1.6442,8.4935],[1.6076,8.5588],[1.6012,9.0495],[1.4012,9.3213],[1.3799,9.4625],[1.3264,9.522],[1.331,9.9965],[0.7688,10.3671],[0.7813,10.693],[0.8629,10.7961],[0.9015,10.9927],[0.9633,10.9741],[0.9814,11.0801],[1.1142,11.0277],[1.0623,11.1421],[1.1584,11.1659],[1.1356,11.2601],[1.2684,11.2479],[1.2733,11.3182],[1.3304,11.2888],[1.3391,11.3661],[1.4396,11.4738],[1.5693,11.4533],[1.6016,11.3887],[1.8406,11.4455],[1.9834,11.414],[2.2736,11.6521],[2.3809,11.9479],[2.4567,11.9791],[2.3612,12.2189],[2.4951,12.2832],[2.6638,12.2916],[2.8443,12.3992],[3.252,12.0241],[3.307,11.9024],[3.4954,11.8375],[3.5964,11.6958]],"ao":[[23.9675,-10.8723],[24.0617,-11.407],[23.9546,-11.6369],[24.0308,-12.3851],[23.8657,-12.7897],[24.0006,-13.0015],[21.9799,-13.0015],[21.9838,-16.1659],[22.1517,-16.5977],[23.3817,-17.6411],[21.3868,-18.0145],[21.1613,-17.9306],[20.8062,-18.0314],[20.3362,-17.8549],[18.89,-17.7993],[18.4536,-17.3899],[13.9806,-17.4236],[13.4798,-17.0103],[13.1663,-16.9511],[12.5672,-17.2346],[12.0881,-17.1392],[11.758,-17.2392],[11.822,-16.4791],[11.7355,-15.8571],[11.992,-15.6173],[12.1453,-15.1686],[12.3288,-14.0922],[12.5025,-13.8552],[12.5153,-13.426],[12.9448,-12.9824],[12.9395,-12.8167],[13.1786,-12.6017],[13.3929,-12.5704],[13.6398,-12.2487],[13.8465,-11.1137],[13.7717,-10.6779],[13.5276,-10.4065],[12.9963,-9.0938],[13.0864,-8.8923],[13.013,-9.0838],[13.2241,-8.7739],[13.3801,-8.7551],[13.3789,-8.3471],[12.85,-7.2659],[12.8335,-6.9123],[12.5506,-6.6326],[12.2903,-6.0966],[12.8174,-6.0317],[13.1849,-5.8564],[16.5151,-5.8871],[16.7268,-6.1892],[16.7061,-6.465],[16.9732,-7.0148],[16.9457,-7.2046],[17.6002,-8.0985],[18.0982,-8.1093],[18.1204,-8.0225],[18.5165,-7.9322],[19.3555,-8.002],[19.3556,-7.5741],[19.5132,-7.4794],[19.5218,-7.0019],[20.6115,-6.916],[20.5205,-7.2864],[21.785,-7.2834],[21.8609,-7.4698],[21.7539,-7.9973],[21.9439,-8.4559],[21.792,-9.4061],[21.8789,-9.6571],[22.1599,-9.9315],[22.3134,-10.3686],[22.3195,-10.7613],[22.1655,-10.8524],[22.2376,-11.2495],[22.501,-11.0424],[23.0143,-11.1025],[23.456,-10.961],[23.8337,-11.0285],[23.9675,-10.8723]],"sa":[[50.8079,24.7466],[50.9789,24.5677],[51.4983,24.5836],[51.2788,24.3078],[51.5693,24.2562],[52.5585,22.9386],[55.1053,22.6209],[55.1868,22.7036],[55.6376,21.979],[54.9784,19.9954],[51.9786,18.9956],[49.0358,18.5797],[48.1619,18.1489],[47.4276,17.0918],[47.1613,16.9474],[46.9703,16.9568],[46.7101,17.2754],[45.4304,17.312],[45.1654,17.4284],[43.9151,17.3018],[43.4129,17.52],[43.165,17.3259],[43.18,16.6646],[42.7895,16.371],[42.5442,17.0043],[42.3567,17.1824],[42.363,17.0248],[42.3015,17.4536],[41.4933,18.2239],[40.7449,19.777],[40.0838,20.2792],[39.7601,20.3508],[39.1743,21.1089],[39.1091,21.7478],[38.9313,22.0061],[39.1428,22.4032],[38.4526,23.784],[37.9479,24.207],[37.4294,24.3673],[37.1486,24.851],[37.2566,24.8712],[37.2422,25.1753],[36.7023,25.78],[36.6961,26.0223],[36.5127,26.111],[35.2163,28.0544],[34.5728,28.1128],[34.8038,28.5271],[34.9494,29.3517],[36.0695,29.2],[36.7287,29.8535],[37.4702,29.9946],[37.6476,30.3309],[37.9814,30.4988],[36.9595,31.491],[39.1577,32.1213],[40.3703,31.9385],[42.0754,31.0799],[44.6918,29.2018],[47.4335,28.9947],[47.6681,28.5335],[48.4946,28.5001],[48.6521,28.0461],[48.8827,27.8267],[48.8318,27.6125],[49.2398,27.5451],[49.3128,27.4486],[49.1198,27.4411],[49.2638,27.4139],[49.3748,27.1491],[49.5721,27.1942],[49.6995,26.9582],[50.1585,26.6649],[49.9837,26.7082],[50.2166,26.3239],[50.1414,26.036],[50.0324,26.2011],[49.99,26.0018],[50.4715,25.4372],[50.8079,24.7466]],"bw":[[25.2598,-17.7941],[25.255,-18.0011],[25.9407,-18.9213],[25.9486,-19.1033],[26.1555,-19.5372],[26.3035,-19.5773],[26.3122,-19.6514],[26.7139,-19.9274],[27.2018,-20.093],[27.2684,-20.4958],[27.6981,-20.5091],[27.6666,-21.0712],[27.885,-21.3102],[27.9904,-21.5519],[28.6157,-21.6471],[29.0387,-21.7979],[29.0405,-22.0209],[29.2393,-22.0726],[29.3501,-22.1867],[29.0387,-22.2239],[28.9304,-22.4409],[28.8168,-22.4929],[28.3018,-22.6038],[27.9375,-22.9638],[27.9301,-23.057],[27.7742,-23.1252],[27.7536,-23.2208],[27.608,-23.2167],[27.5489,-23.3608],[27.35,-23.3915],[27.1274,-23.525],[26.9672,-23.7187],[26.8392,-24.2656],[26.5312,-24.4587],[26.4044,-24.6328],[25.8684,-24.7482],[25.8352,-25.0164],[25.5873,-25.6195],[25.4584,-25.7112],[24.665,-25.8233],[24.3389,-25.7518],[24.1835,-25.6259],[23.9248,-25.6292],[23.4593,-25.2822],[23.0307,-25.2994],[22.8446,-25.4811],[22.7105,-25.9969],[22.5451,-26.2074],[22.2483,-26.3469],[22.0577,-26.6176],[21.7806,-26.6782],[21.7623,-26.8045],[21.6645,-26.8631],[21.1225,-26.8653],[20.9077,-26.8],[20.6908,-26.8918],[20.6147,-26.7532],[20.6053,-26.4933],[20.8414,-26.1313],[20.7941,-25.8939],[20.6379,-25.6203],[20.6565,-25.4678],[20.5422,-25.3822],[20.3649,-25.0332],[19.9814,-24.7525],[19.9783,-22.0007],[20.9848,-21.964],[20.9751,-18.3193],[21.4757,-18.2995],[23.2928,-17.999],[23.3958,-18.1914],[23.5015,-18.2375],[23.6099,-18.4777],[23.916,-18.2012],[24.3506,-17.9561],[24.5644,-18.0528],[24.9533,-17.7884],[25.2598,-17.7941]],"zw":[[25.2598,-17.7941],[25.5165,-17.8623],[25.6814,-17.8115],[25.9245,-17.999],[26.2119,-17.8828],[26.7,-18.0692],[27.0213,-17.9585],[27.5773,-17.3631],[27.8169,-16.9596],[28.8089,-16.4863],[28.8572,-16.0605],[28.9324,-15.9637],[29.5872,-15.6557],[30.3963,-15.636],[30.4026,-16.0012],[30.8574,-15.9981],[30.9898,-16.0643],[31.1141,-15.9969],[31.26,-16.0235],[31.4046,-16.1623],[31.7107,-16.2179],[31.9103,-16.4289],[32.2905,-16.4518],[32.6718,-16.5998],[32.7313,-16.7088],[32.9685,-16.6816],[32.8288,-16.9351],[33.0216,-17.3456],[32.9366,-17.4983],[33.0245,-17.6192],[32.9274,-17.9729],[32.955,-18.2563],[33.0428,-18.352],[32.9964,-18.4671],[32.8707,-18.5358],[32.9025,-18.7745],[32.6914,-18.8342],[32.6811,-18.955],[32.699,-19.0222],[32.8142,-19.0238],[32.8559,-19.0837],[32.7634,-19.464],[32.8322,-19.5009],[32.8196,-19.6743],[32.9432,-19.6493],[33.0328,-19.7842],[33.0073,-20.032],[32.8853,-20.103],[32.853,-20.2739],[32.6718,-20.5318],[32.4816,-20.603],[32.491,-20.9363],[32.3398,-21.1341],[32.4472,-21.3137],[32.4085,-21.2903],[31.2889,-22.3973],[31.1526,-22.3164],[30.8379,-22.2823],[30.335,-22.3447],[29.7589,-22.1309],[29.3569,-22.1909],[29.2393,-22.0726],[29.0405,-22.0209],[29.0387,-21.7979],[28.6157,-21.6471],[27.9904,-21.5519],[27.885,-21.3102],[27.6666,-21.0712],[27.6981,-20.5091],[27.2684,-20.4958],[27.266,-20.2342],[27.2018,-20.093],[26.7139,-19.9274],[26.3122,-19.6514],[26.3035,-19.5773],[26.1555,-19.5372],[25.9486,-19.1033],[25.9407,-18.9213],[25.255,-18.0011],[25.2598,-17.7941]],"td":[[14.0649,13.078],[13.6073,13.7041],[13.4495,14.439],[13.6579,14.5487],[13.8339,15.0196],[14.369,15.7496],[15.4655,16.8947],[15.736,19.9035],[15.9703,20.3363],[15.5702,20.7519],[15.6093,20.9507],[15.1845,21.4912],[15.172,21.9934],[14.9799,22.9957],[15.9851,23.4447],[23.9813,19.4961],[23.9844,15.7212],[23.0946,15.7043],[22.9067,15.5414],[22.958,15.2016],[22.6589,14.8574],[22.6766,14.689],[22.3635,14.5435],[22.429,14.2646],[22.5464,14.1397],[22.2147,13.9566],[22.075,13.7803],[22.2676,13.3346],[21.9641,13.0983],[21.8094,12.7937],[21.936,12.6395],[22.2049,12.7434],[22.4457,12.6111],[22.3728,12.4631],[22.4581,12.0304],[22.6124,12.0728],[22.5417,11.633],[22.7719,11.4032],[22.9148,11.3961],[22.9224,11.0871],[22.8611,10.9192],[22.4471,10.9983],[21.7226,10.6367],[21.7188,10.2966],[21.5132,10.2005],[21.3743,9.9731],[21.2569,9.9757],[20.8261,9.4181],[20.6679,9.302],[20.5418,9.3216],[20.4352,9.1381],[19.0219,8.9852],[18.8698,8.8494],[19.1241,8.6751],[18.5893,8.0479],[17.6399,7.985],[16.8055,7.5437],[16.6142,7.6796],[16.5487,7.87],[16.3922,7.7836],[16.3708,7.6725],[15.9253,7.4882],[15.481,7.5233],[15.5629,7.7924],[15.4407,7.8394],[15.1837,8.4791],[14.3495,9.1684],[13.9476,9.6378],[14.1817,9.9782],[14.7726,9.9217],[15.6812,9.9913],[15.1317,10.5407],[15.0212,11.1825],[15.1356,11.5308],[15.0665,11.6806],[15.1109,11.7829],[15.0494,12.0846],[14.898,12.1528],[14.8308,12.6182],[14.5548,12.7669],[14.4182,13.0811],[14.0649,13.078]],"dz":[[-4.8216,24.9951],[-8.6824,27.2854],[-8.6788,28.6928],[-7.6195,29.3894],[-5.7562,29.6141],[-5.5395,29.5232],[-5.1761,29.9775],[-4.3724,30.5086],[-3.6455,30.7113],[-3.5547,30.9559],[-3.8428,31.1702],[-3.6595,31.6478],[-2.8278,31.7946],[-2.8812,32.0763],[-1.2103,32.0897],[-1.2441,32.3569],[-1.032,32.4944],[-1.4233,32.7424],[-1.6742,33.238],[-1.6696,34.0792],[-1.8096,34.3725],[-1.703,34.4797],[-1.8711,34.5966],[-1.7695,34.7413],[-2.2226,35.0893],[-1.7611,35.1296],[-1.0329,35.6826],[-0.6265,35.723],[-0.4795,35.89],[-0.0831,35.7944],[0.3423,36.206],[1.3467,36.545],[2.6008,36.5963],[2.9331,36.8086],[3.8739,36.9172],[4.7871,36.8954],[5.3044,36.643],[6.4158,37.093],[6.9486,36.89],[7.2329,36.9669],[7.2227,37.0898],[7.9085,36.8496],[8.6025,36.9395],[8.1699,36.5258],[8.3577,36.4304],[8.2414,35.8277],[8.2944,35.3251],[8.4313,35.2417],[8.2365,34.6477],[7.8315,34.4144],[7.4798,33.8939],[7.725,33.2314],[8.0868,33.0943],[8.3313,32.5276],[9.045,32.0718],[9.5197,30.2289],[9.2865,30.1171],[9.8261,29.1285],[9.777,28.2676],[9.9343,27.8274],[9.7213,27.2919],[9.8963,26.6528],[9.3778,26.1689],[10.0079,25.3314],[10.032,24.8563],[10.2422,24.5951],[11.5414,24.2975],[11.9689,23.5174],[7.4827,20.8726],[5.7943,19.4498],[3.3331,18.9756],[3.1041,19.1355],[3.2608,19.3883],[3.1988,19.8205],[2.4004,20.0566],[2.2008,20.2739],[1.7781,20.3043],[1.5597,20.5975],[1.1546,20.7388],[1.1465,21.1017],[-4.8216,24.9951]],"mz":[[32.1139,-26.84],[31.9058,-25.813],[31.9866,-24.4231],[31.2791,-22.4153],[32.4472,-21.3137],[32.4816,-20.603],[33.0073,-20.032],[32.6811,-18.955],[33.0428,-18.352],[32.8288,-16.9351],[32.9685,-16.6816],[31.9103,-16.4289],[31.26,-16.0235],[30.4026,-16.0012],[30.2145,-14.9815],[33.2027,-14.0139],[33.6449,-14.6017],[34.3441,-14.3874],[34.5691,-15.2712],[34.2332,-15.8898],[35.1183,-16.8334],[35.066,-17.1098],[35.2919,-17.1293],[35.1319,-16.5376],[35.3746,-16.141],[35.7851,-16.0423],[35.9043,-14.8869],[35.0841,-13.7042],[34.5453,-13.3257],[34.354,-12.1995],[34.6167,-11.5783],[35.5315,-11.6127],[35.8264,-11.4135],[36.1997,-11.7015],[37.0324,-11.5649],[37.4697,-11.7195],[37.9381,-11.284],[38.4923,-11.4135],[40.5105,-10.4777],[40.655,-10.6869],[40.3488,-11.3171],[40.6468,-12.7663],[40.4109,-12.9421],[40.5952,-12.9757],[40.6294,-14.5663],[40.8014,-14.4107],[40.848,-14.705],[40.528,-15.131],[40.6817,-15.233],[40.5818,-15.4921],[39.7834,-16.3019],[39.856,-16.4298],[39.0905,-16.9838],[38.0808,-17.1907],[36.9841,-18.0124],[36.8328,-17.8776],[36.9772,-18.0466],[36.4075,-18.7848],[36.2381,-18.6941],[36.2669,-18.8909],[35.8553,-18.9513],[35.1268,-19.705],[34.8801,-19.8634],[34.5532,-19.5823],[34.7776,-19.815],[34.6683,-20.5411],[35.1201,-20.9664],[35.3115,-22.413],[35.5514,-22.1791],[35.6061,-22.9147],[35.3403,-23.6762],[35.4968,-24.1013],[35.1065,-24.5979],[32.8727,-25.5439],[32.4768,-25.9846],[32.8425,-26.2887],[32.9544,-26.0789],[32.8931,-26.8461],[32.1139,-26.84]],"sz":[[31.9492,-25.9581],[31.9753,-25.9804],[32.004,-25.9944],[32.0706,-26.0098],[32.0575,-26.0412],[32.0627,-26.0772],[32.0746,-26.1142],[32.0814,-26.1488],[32.0778,-26.1755],[32.0447,-26.2545],[32.0399,-26.2837],[32.0523,-26.3868],[32.0598,-26.4148],[32.1071,-26.5003],[32.1174,-26.5823],[32.1139,-26.84],[32.0978,-26.8335],[32.0732,-26.8114],[32.0571,-26.8086],[31.9901,-26.8083],[31.9922,-26.8383],[31.9753,-26.9266],[31.9597,-27.0088],[31.9421,-27.1007],[31.9474,-27.2555],[31.953,-27.2702],[31.9598,-27.281],[31.9648,-27.2917],[31.9677,-27.303],[31.9683,-27.3163],[31.8786,-27.3152],[31.7825,-27.314],[31.5265,-27.3109],[31.4595,-27.2987],[31.2801,-27.2435],[31.157,-27.2056],[31.142,-27.1965],[31.1265,-27.1826],[31.0788,-27.1182],[30.9761,-27.035],[30.9536,-27.0003],[30.9494,-26.9754],[30.9597,-26.9361],[30.961,-26.9112],[30.9551,-26.8914],[30.9444,-26.8766],[30.9158,-26.8494],[30.9026,-26.8313],[30.8802,-26.7722],[30.8687,-26.7849],[30.8534,-26.7966],[30.8364,-26.8051],[30.8198,-26.808],[30.8024,-26.8089],[30.7957,-26.7855],[30.7857,-26.7169],[30.7841,-26.5788],[30.7829,-26.4724],[30.8046,-26.3975],[30.8973,-26.2917],[30.9699,-26.2091],[31.0378,-26.1001],[31.0916,-25.9836],[31.1069,-25.9309],[31.1198,-25.91],[31.2084,-25.8386],[31.3093,-25.7574],[31.3374,-25.7447],[31.372,-25.7364],[31.4015,-25.736],[31.4269,-25.7436],[31.6388,-25.8675],[31.7306,-25.9212],[31.8346,-25.9821],[31.8634,-25.9899],[31.8919,-25.9838],[31.9492,-25.9581]],"bi":[[30.5546,-2.4006],[30.416,-2.6456],[30.4232,-2.6809],[30.5226,-2.6494],[30.4505,-2.7418],[30.4158,-2.8517],[30.4742,-2.9032],[30.4885,-2.9435],[30.5389,-2.899],[30.6295,-2.9477],[30.6376,-2.9744],[30.6837,-2.97],[30.7321,-2.9935],[30.8255,-2.9786],[30.7783,-3.0474],[30.8329,-3.1477],[30.8148,-3.2477],[30.7757,-3.291],[30.7215,-3.281],[30.6405,-3.3329],[30.6033,-3.3726],[30.6404,-3.3928],[30.6391,-3.4195],[30.4323,-3.5519],[30.4289,-3.6022],[30.3731,-3.7036],[30.3816,-3.7883],[30.337,-3.7738],[30.3117,-3.7899],[30.2088,-3.9305],[30.1498,-4.0868],[30.0515,-4.1802],[30.003,-4.2719],[29.8474,-4.3705],[29.7841,-4.3738],[29.7328,-4.4633],[29.4042,-4.4498],[29.3395,-4.0932],[29.215,-3.8999],[29.2252,-3.5886],[29.2067,-3.3345],[29.2409,-3.2663],[29.2169,-3.2331],[29.2141,-3.1665],[29.2423,-3.1108],[29.2346,-3.0466],[29.1568,-3.0048],[29.1384,-2.9504],[29.0898,-2.9261],[29.0491,-2.8275],[28.9869,-2.7968],[28.9955,-2.7584],[29.0286,-2.7449],[29.0154,-2.7207],[29.0326,-2.6207],[29.0553,-2.5984],[29.1134,-2.5946],[29.2944,-2.6511],[29.3375,-2.7238],[29.3181,-2.7626],[29.3397,-2.8264],[29.4254,-2.8034],[29.5041,-2.8269],[29.6193,-2.7877],[29.7199,-2.8055],[29.7447,-2.76],[29.8424,-2.7524],[29.8979,-2.6711],[29.9317,-2.3169],[30.0133,-2.3483],[30.1168,-2.4315],[30.1562,-2.419],[30.2075,-2.3392],[30.298,-2.3537],[30.3786,-2.3031],[30.4185,-2.3118],[30.4885,-2.3838],[30.5546,-2.4006]],"rw":[[29.0154,-2.7207],[28.8913,-2.6527],[28.8981,-2.5763],[28.8572,-2.5183],[28.8782,-2.4897],[28.8588,-2.4182],[28.8804,-2.3799],[28.9597,-2.3398],[28.9937,-2.2777],[29.0754,-2.2639],[29.1348,-2.1977],[29.1563,-2.1087],[29.1305,-1.8433],[29.2333,-1.6587],[29.3038,-1.5879],[29.3428,-1.5165],[29.4374,-1.5062],[29.5386,-1.4023],[29.6577,-1.3839],[29.7348,-1.3482],[29.7749,-1.3663],[29.8161,-1.3226],[29.8642,-1.3703],[29.8807,-1.4536],[29.9174,-1.4752],[29.9604,-1.4648],[30.1472,-1.3451],[30.1656,-1.2775],[30.2122,-1.2595],[30.3174,-1.137],[30.3375,-1.0662],[30.4718,-1.0668],[30.453,-1.0973],[30.4709,-1.1181],[30.4673,-1.1553],[30.5112,-1.1704],[30.5553,-1.3184],[30.7375,-1.4067],[30.7389,-1.4895],[30.7916,-1.591],[30.831,-1.5942],[30.8383,-1.6154],[30.8242,-1.7307],[30.8379,-1.8368],[30.8242,-1.9021],[30.797,-1.9293],[30.8269,-1.9341],[30.8168,-2.0187],[30.8535,-2.0237],[30.8878,-2.0825],[30.8344,-2.3453],[30.7588,-2.3811],[30.6877,-2.35],[30.6379,-2.397],[30.5217,-2.3994],[30.4185,-2.3118],[30.3786,-2.3031],[30.298,-2.3537],[30.2075,-2.3392],[30.1562,-2.419],[30.1168,-2.4315],[30.0133,-2.3483],[29.9317,-2.3169],[29.8979,-2.6711],[29.8424,-2.7524],[29.7447,-2.76],[29.7199,-2.8055],[29.6193,-2.7877],[29.5041,-2.8269],[29.4254,-2.8034],[29.3397,-2.8264],[29.3181,-2.7626],[29.3375,-2.7238],[29.3066,-2.6601],[29.1134,-2.5946],[29.0553,-2.5984],[29.0326,-2.6207],[29.0154,-2.7207]],"ug":[[30.4718,-1.0668],[30.3375,-1.0662],[30.1472,-1.3451],[29.9174,-1.4752],[29.8251,-1.3239],[29.5779,-1.3884],[29.5547,-0.9286],[29.6108,-0.864],[29.7031,0.0725],[29.7971,0.1648],[29.8396,0.3585],[29.9405,0.4983],[29.9473,0.8246],[30.1548,0.9087],[30.2389,1.136],[30.4783,1.2386],[30.6817,1.5003],[31.0258,1.7782],[31.2715,2.103],[31.1776,2.3029],[31.0552,2.2902],[30.968,2.4054],[30.8718,2.3321],[30.8049,2.4344],[30.7106,2.4451],[30.8534,2.8534],[30.7438,3.0555],[30.91,3.3934],[30.8964,3.52],[30.8395,3.4902],[30.9444,3.6793],[31.2148,3.7924],[31.5237,3.6561],[31.6868,3.7128],[31.7779,3.8164],[31.9019,3.7043],[31.93,3.5984],[32.168,3.5122],[32.1878,3.6192],[32.3718,3.7311],[32.7564,3.769],[32.9972,3.8855],[33.1645,3.7631],[33.4906,3.7497],[34.006,4.2057],[34.1071,3.9595],[34.0848,3.8773],[34.205,3.8745],[34.1525,3.7756],[34.2409,3.7836],[34.2789,3.7097],[34.3371,3.7346],[34.4397,3.6677],[34.4345,3.5262],[34.3813,3.4769],[34.434,3.182],[34.5338,3.1185],[34.5847,2.9287],[34.6405,2.8601],[34.7301,2.8529],[34.8187,2.598],[34.8756,2.5912],[34.9236,2.4773],[34.8656,2.3475],[35.0065,1.9169],[34.9727,1.6542],[34.779,1.3885],[34.7979,1.2319],[34.487,1.0825],[34.388,0.8158],[34.1496,0.6036],[34.0852,0.347],[33.8936,0.1098],[33.9535,-0.1544],[33.9042,-1.0026],[30.7771,-0.9858],[30.6386,-1.0734],[30.4718,-1.0668]],"ls":[[28.9808,-28.909],[29.0405,-28.923],[29.0558,-28.9654],[29.126,-28.9924],[29.2084,-29.0689],[29.3118,-29.0898],[29.3168,-29.1474],[29.3575,-29.1833],[29.4359,-29.3424],[29.3995,-29.4303],[29.2827,-29.4856],[29.2751,-29.5323],[29.2963,-29.5716],[29.2786,-29.6136],[29.1613,-29.6669],[29.1138,-29.7501],[29.1058,-29.8256],[29.133,-29.852],[29.1442,-29.9197],[28.7956,-30.0894],[28.6397,-30.1315],[28.5406,-30.1133],[28.4562,-30.147],[28.3839,-30.1442],[28.2944,-30.2425],[28.2006,-30.2825],[28.2385,-30.3523],[28.1254,-30.4573],[28.1412,-30.5008],[28.0816,-30.5869],[28.0788,-30.6588],[27.9366,-30.6408],[27.8582,-30.6037],[27.7436,-30.6003],[27.5986,-30.4903],[27.5297,-30.3822],[27.4775,-30.3563],[27.4521,-30.3101],[27.3662,-30.311],[27.3437,-30.2129],[27.3814,-30.1427],[27.3128,-30.1316],[27.2659,-30.0333],[27.1988,-29.9785],[27.0801,-29.7351],[27.0022,-29.6665],[27.0224,-29.6162],[27.0424,-29.6281],[27.0622,-29.6002],[27.0859,-29.6144],[27.1281,-29.5786],[27.2631,-29.5411],[27.3502,-29.4803],[27.4176,-29.3966],[27.4063,-29.364],[27.4458,-29.332],[27.4533,-29.2916],[27.5289,-29.261],[27.5215,-29.2165],[27.613,-29.1338],[27.6307,-29.0725],[27.665,-29.0657],[27.6392,-29.0353],[27.7117,-28.991],[27.7474,-28.9086],[27.8613,-28.9154],[27.9262,-28.8534],[28.0136,-28.8784],[28.155,-28.7023],[28.2971,-28.7069],[28.3829,-28.6265],[28.5625,-28.6065],[28.6343,-28.5708],[28.6665,-28.5972],[28.6995,-28.6885],[28.7439,-28.6895],[28.7877,-28.7487],[28.8728,-28.7836],[28.9808,-28.909]],"cm":[[11.3221,2.1658],[9.8109,2.2649],[9.9295,3.2716],[9.6421,3.539],[9.8196,3.6279],[9.6262,3.6],[9.5459,3.8196],[9.7445,3.8196],[9.7644,3.9575],[9.6189,3.9604],[9.7546,4.1357],[9.5794,4.013],[9.4785,4.1114],[9.5325,3.9894],[9.3435,3.9158],[9.3262,4.0319],[9.2107,3.9643],[8.9712,4.1008],[8.847,4.6397],[8.7884,4.5414],[8.6551,4.7383],[8.7227,4.5131],[8.5103,4.527],[8.7989,5.1596],[8.8452,5.831],[10.1429,7.0077],[10.2108,6.8792],[10.496,6.8747],[10.5787,7.1308],[11.0367,6.7405],[11.0971,6.4491],[11.2555,6.43],[11.8722,7.0794],[11.7366,7.2633],[12.1921,7.9607],[12.227,8.3863],[12.4039,8.4965],[12.3688,8.6096],[12.7945,8.7886],[12.8554,9.3781],[13.1955,9.5422],[13.2302,10.0453],[13.4349,10.1534],[13.7548,11.0167],[13.9823,11.2763],[14.1655,11.2383],[14.6162,11.5387],[14.5419,11.7098],[14.6683,12.1781],[14.1889,12.3704],[14.0649,13.078],[14.4182,13.0811],[14.5548,12.7669],[14.8192,12.6388],[14.898,12.1528],[15.0494,12.0846],[15.1382,10.5217],[15.6812,9.9913],[14.1817,9.9782],[13.9459,9.6526],[15.0519,8.6438],[15.5629,7.7924],[14.7723,6.3183],[14.3873,6.039],[14.6174,5.865],[14.5199,5.2906],[14.6589,5.1645],[14.7173,4.622],[15.1898,4.0631],[15.026,4.0261],[15.0849,3.8858],[16.0924,2.8633],[16.1558,1.7191],[16.0699,1.6546],[15.7068,1.932],[14.8925,2.0064],[14.5621,2.2088],[13.2946,2.1611],[13.1638,2.2832],[11.6573,2.3225],[11.3516,2.3006],[11.3221,2.1658]],"ga":[[13.2946,2.1611],[13.1622,1.8998],[13.2513,1.3376],[13.1502,1.2566],[13.7947,1.4344],[14.1445,1.3916],[14.4681,0.9132],[14.3288,0.6212],[13.9452,0.3504],[13.8312,-0.2054],[14.49,-0.6009],[14.4045,-1.8931],[14.2401,-1.9823],[14.148,-2.2248],[14.2344,-2.3549],[14.09,-2.4986],[13.8458,-2.4666],[13.9062,-2.3598],[13.7432,-2.0971],[13.4589,-2.441],[12.969,-2.3724],[12.8046,-1.9191],[12.6324,-1.8252],[12.4269,-1.8841],[12.4593,-2.3299],[11.7561,-2.4151],[11.5662,-2.3297],[11.5307,-2.8667],[11.6368,-2.8333],[11.7784,-3.0057],[11.6875,-3.171],[11.9444,-3.3032],[11.825,-3.5712],[11.9055,-3.6435],[11.6876,-3.6987],[11.4826,-3.5093],[11.2125,-3.6973],[11.114,-3.9369],[10.6354,-3.3137],[9.7251,-2.4814],[9.9773,-2.6287],[10.1375,-2.5274],[9.597,-2.3522],[9.2713,-1.8766],[9.5432,-2.0709],[9.5178,-1.9264],[9.2554,-1.8352],[8.9836,-1.2309],[9.2803,-1.6751],[9.5596,-1.6076],[9.4566,-1.4705],[9.2866,-1.5661],[9.3355,-1.2834],[9.1857,-1.4106],[9.03,-1.2991],[8.6956,-0.5866],[8.8444,-0.8002],[8.929,-0.6873],[9.0115,-0.8746],[9.3065,-0.3323],[9.3454,0.3558],[9.488,0.0983],[9.5042,0.188],[9.8122,0.0235],[9.7586,0.1256],[10.0256,0.188],[9.4981,0.291],[9.3057,0.5443],[9.5391,0.6815],[9.464,0.5989],[9.6079,0.4692],[9.5921,1.0143],[9.7451,1.0597],[9.9441,0.9249],[11.3363,0.9992],[11.3516,2.3006],[13.1638,2.2832],[13.2946,2.1611]],"ne":[[3.5964,11.6958],[2.8443,12.3992],[2.37,12.2363],[2.4573,11.9962],[2.3902,11.8965],[2.0518,12.3419],[2.245,12.4242],[2.109,12.7056],[1.9717,12.7242],[1.8437,12.6061],[1.5639,12.6321],[0.9835,13.0324],[0.9835,13.3684],[1.2687,13.3461],[1.0153,13.4657],[0.8969,13.6149],[0.5943,13.6889],[0.3692,14.0391],[0.3888,14.2516],[0.1529,14.5467],[0.2213,14.9959],[0.9493,14.9796],[1.3315,15.2836],[3.0002,15.3391],[3.0335,15.4264],[3.5071,15.354],[3.5265,15.496],[3.8461,15.6853],[4.0606,16.2983],[4.184,16.4161],[4.2286,19.1422],[5.7943,19.4498],[7.4827,20.8726],[11.9689,23.5174],[13.4823,23.1797],[14.2162,22.6163],[14.9799,22.9957],[15.172,21.9934],[15.1845,21.4912],[15.6093,20.9507],[15.5443,20.8903],[15.5702,20.7519],[15.9703,20.3363],[15.736,19.9035],[15.472,16.9167],[14.369,15.7496],[13.8339,15.0196],[13.6579,14.5487],[13.4495,14.439],[13.6073,13.7041],[13.3299,13.7094],[13.2281,13.548],[12.8423,13.4891],[12.8052,13.373],[12.5611,13.2428],[12.4727,13.0642],[12.0576,13.1168],[11.44,13.3644],[10.675,13.3751],[10.1198,13.2501],[9.5903,12.8014],[8.6793,12.9236],[8.1051,13.2997],[7.824,13.3453],[7.392,13.1125],[7.1982,13.1174],[7.0687,12.9961],[6.9371,12.9954],[6.3022,13.6638],[6.1424,13.6408],[5.5218,13.8803],[5.2276,13.7413],[4.8249,13.7707],[4.4526,13.6738],[4.1258,13.473],[4.0888,12.9962],[3.9291,12.7504],[3.6416,12.518],[3.6039,11.9056],[3.667,11.7597],[3.5964,11.6958]],"bf":[[2.3902,11.8965],[2.2874,11.6653],[2.0108,11.4269],[1.6016,11.3887],[1.4396,11.4738],[1.3304,11.2888],[1.1563,11.2855],[1.1584,11.1659],[1.0623,11.1421],[1.1175,11.0322],[0.9814,11.0801],[0.9633,10.9741],[0.4876,10.9332],[0.4866,11.0032],[-0.3017,11.1629],[-0.6348,10.908],[-0.8278,11.0058],[-2.8373,10.998],[-2.9325,10.6343],[-2.7676,10.4212],[-2.8413,10.3433],[-2.7549,10.2608],[-2.7953,9.72],[-2.6892,9.4887],[-2.7637,9.3919],[-3.2148,9.9156],[-3.3017,9.8411],[-3.755,9.9359],[-4.1503,9.8183],[-4.3712,9.583],[-4.51,9.7451],[-4.6811,9.6821],[-4.9661,9.901],[-5.1192,10.2935],[-5.4025,10.3006],[-5.5226,10.4255],[-5.4314,10.8447],[-5.5042,11.0727],[-5.3325,11.1216],[-5.2183,11.4222],[-5.2871,11.7589],[-5.4124,11.8285],[-4.7613,12.0066],[-4.4061,12.3075],[-4.4853,12.7147],[-4.2576,12.7209],[-4.2119,12.8192],[-4.327,13.1686],[-3.9777,13.4999],[-3.9716,13.3868],[-3.4495,13.1688],[-3.4485,13.2658],[-3.2486,13.2928],[-3.287,13.6978],[-3.0675,13.6068],[-2.8952,13.6516],[-2.8673,14.0005],[-2.5974,14.2223],[-2.4617,14.2809],[-2.0359,14.1812],[-1.9971,14.4707],[-1.6969,14.4961],[-0.7197,15.0785],[-0.4679,15.0799],[-0.4257,15.0026],[-0.2367,15.0656],[0.2185,14.911],[0.1588,14.4961],[0.3888,14.2516],[0.3692,14.0391],[0.5943,13.6889],[0.8969,13.6149],[1.2687,13.3461],[0.9835,13.3684],[0.9835,13.0324],[1.5639,12.6321],[2.109,12.7056],[2.245,12.4242],[2.0518,12.3419],[2.3902,11.8965]],"tg":[[-0.1661,11.135],[0.4866,11.0032],[0.4876,10.9332],[0.6753,10.9885],[0.9015,10.9927],[0.8629,10.7961],[0.7813,10.693],[0.7607,10.3822],[1.3438,9.9624],[1.3264,9.522],[1.3799,9.4625],[1.4012,9.3213],[1.5958,9.0768],[1.6076,8.5588],[1.6442,8.4935],[1.6011,8.3716],[1.6257,6.9968],[1.5319,6.9919],[1.5901,6.8673],[1.6011,6.7464],[1.566,6.6781],[1.7441,6.4257],[1.7824,6.2773],[1.6196,6.2139],[1.1854,6.1005],[1.1762,6.1515],[1.0934,6.1609],[1.0252,6.2223],[0.9835,6.3246],[0.8868,6.3304],[0.7457,6.4409],[0.7274,6.5714],[0.6322,6.624],[0.6258,6.7278],[0.5193,6.8322],[0.5463,6.9148],[0.4946,6.9731],[0.598,7.0312],[0.6457,7.3236],[0.623,7.3979],[0.5167,7.411],[0.4956,7.5043],[0.5093,7.5854],[0.6123,7.6994],[0.5738,8.2039],[0.7128,8.298],[0.6163,8.4888],[0.4435,8.6069],[0.3659,8.7743],[0.4735,8.7935],[0.5056,8.8663],[0.4313,9.0276],[0.5047,9.1872],[0.5314,9.4011],[0.4835,9.47],[0.4117,9.4922],[0.2268,9.4323],[0.2281,9.474],[0.3019,9.5075],[0.2259,9.5319],[0.2302,9.5757],[0.3777,9.5894],[0.2776,9.607],[0.261,9.6572],[0.2753,9.6856],[0.3471,9.6632],[0.3498,10.0288],[0.3981,10.0521],[0.3536,10.1155],[0.3572,10.22],[0.3966,10.2832],[0.3088,10.2971],[0.2732,10.4066],[0.1799,10.3942],[0.0294,10.5825],[-0.0881,10.6335],[0.0162,11.0626],[-0.1661,11.135]],"gh":[[-0.1661,11.135],[0.0162,11.0626],[-0.0881,10.6335],[0.3966,10.2832],[0.3471,9.6632],[0.261,9.6572],[0.3777,9.5894],[0.2302,9.5757],[0.3019,9.5075],[0.2268,9.4323],[0.4117,9.4922],[0.5314,9.4011],[0.4313,9.0276],[0.5056,8.8663],[0.4735,8.7935],[0.3659,8.7743],[0.7128,8.298],[0.5738,8.2039],[0.6123,7.6994],[0.4956,7.5043],[0.5167,7.411],[0.623,7.3979],[0.6457,7.3236],[0.598,7.0312],[0.4946,6.9731],[0.5463,6.9148],[0.5161,6.8418],[0.6258,6.7278],[0.6322,6.624],[0.7274,6.5714],[0.7457,6.4409],[0.9835,6.3246],[1.1854,6.1005],[1.048,5.9927],[0.9548,5.7911],[0.2433,5.7562],[-0.3716,5.4927],[-0.7995,5.2148],[-0.9971,5.2168],[-1.6185,5.0224],[-1.9814,4.7524],[-2.0966,4.7385],[-2.3566,4.9196],[-3.1196,5.0913],[-2.7666,5.161],[-2.7896,5.3469],[-2.7301,5.3732],[-2.7934,5.6081],[-2.9459,5.6083],[-2.9648,5.7098],[-3.0298,5.7043],[-3.0225,5.8545],[-3.2625,6.6171],[-3.2417,6.8109],[-2.9728,7.215],[-2.9415,7.5772],[-2.7906,7.9433],[-2.6106,8.04],[-2.6107,8.1649],[-2.5063,8.2093],[-2.6194,8.9236],[-2.6608,9.0149],[-2.7748,9.055],[-2.6598,9.2729],[-2.715,9.3111],[-2.6757,9.4717],[-2.7953,9.72],[-2.7339,9.8228],[-2.7989,10.1696],[-2.7549,10.2608],[-2.8413,10.3433],[-2.7676,10.4212],[-2.8596,10.4656],[-2.9325,10.6343],[-2.8373,10.998],[-0.8278,11.0058],[-0.6739,10.9835],[-0.6348,10.908],[-0.3017,11.1629],[-0.1661,11.135]],"gw":[[-13.7283,12.6734],[-13.6607,12.4587],[-13.695,12.2929],[-13.84,12.2771],[-13.9725,12.1516],[-13.7239,12.0028],[-13.7299,11.7098],[-13.8701,11.7437],[-13.8864,11.6756],[-14.0151,11.6371],[-14.2897,11.6688],[-14.5209,11.5129],[-14.7111,11.4976],[-14.9591,11.0358],[-15.0852,10.9276],[-15.0483,10.9948],[-15.114,10.9752],[-15.0994,11.0577],[-15.0009,11.2168],[-15.2301,10.9948],[-15.2711,11.0392],[-15.2066,11.2338],[-15.2335,11.1225],[-15.3582,11.1442],[-15.4261,11.2865],[-15.2681,11.433],[-15.3431,11.3784],[-15.3534,11.4633],[-15.5071,11.3579],[-15.4735,11.4945],[-15.2899,11.5428],[-15.3363,11.5838],[-15.2401,11.577],[-15.2811,11.6248],[-15.2091,11.574],[-15.1944,11.6428],[-15.1159,11.6177],[-15.1233,11.5455],[-15.0073,11.6118],[-15.0799,11.598],[-15.1103,11.6868],[-15.1719,11.6596],[-15.1308,11.708],[-15.228,11.6586],[-15.2265,11.7551],[-15.4387,11.5571],[-15.4263,11.6937],[-15.4761,11.6544],[-15.5514,11.7331],[-15.4387,11.8854],[-15.1892,11.8712],[-15.0899,11.9401],[-15.0694,11.8036],[-14.9359,11.753],[-15.0702,11.8607],[-15.0072,11.9783],[-15.2151,11.9096],[-15.4263,11.9612],[-15.8304,11.7484],[-15.7915,11.8632],[-15.7342,11.8581],[-15.7956,11.8854],[-15.8502,11.7753],[-15.8776,11.824],[-15.959,11.7348],[-15.8653,11.9722],[-15.7205,11.9674],[-15.7137,12.0158],[-15.8639,12.0295],[-16.1246,11.8854],[-16.3232,11.9953],[-16.3551,12.0873],[-16.0631,12.351],[-16.4576,12.1719],[-16.7284,12.3325],[-16.4123,12.3617],[-16.2224,12.4551],[-15.677,12.4393],[-15.1951,12.6794],[-13.7283,12.6734]],"us":[[-122.753,48.9925],[-95.1606,49.3695],[-91.5748,48.0482],[-88.3765,48.3051],[-82.5483,45.3346],[-82.1457,43.5764],[-83.1595,42.0463],[-82.6669,41.6691],[-79.0194,42.8027],[-78.7508,43.6171],[-76.8415,43.6255],[-74.9037,45.0052],[-70.8924,45.2346],[-69.2373,47.4588],[-67.8136,47.0819],[-66.9773,44.8155],[-70.2869,43.6505],[-71.0468,42.3215],[-69.9389,41.6742],[-73.9759,41.2922],[-74.4134,39.3571],[-74.9547,38.9264],[-75.5582,39.6135],[-75.0236,40.0163],[-75.5922,39.6492],[-75.0447,38.425],[-75.9491,37.125],[-75.6331,37.96],[-76.3747,38.8416],[-75.8317,39.5844],[-76.6232,39.2597],[-76.3239,38.0438],[-77.0683,38.9019],[-76.2699,37.0693],[-77.2743,37.3164],[-76.0046,36.9324],[-75.5238,35.7944],[-75.9469,36.7185],[-75.7855,36.0816],[-76.7298,36.3227],[-75.7218,35.8292],[-77.052,35.5344],[-76.3463,34.8832],[-80.855,32.5395],[-81.4924,31.3677],[-80.0382,26.8112],[-80.4025,25.1862],[-81.732,25.9205],[-83.6806,29.9213],[-85.3461,29.6784],[-85.5868,30.3261],[-88.0301,30.2238],[-88.0301,30.7377],[-90.405,30.2202],[-89.366,30.0487],[-89.4136,28.9284],[-91.8475,29.8312],[-95.0523,29.7497],[-97.7743,27.4685],[-97.366,25.8439],[-99.1064,26.423],[-101.4051,29.7784],[-103.3866,29.0288],[-106.3918,31.7459],[-111.0671,31.3336],[-114.7243,32.7128],[-117.1251,32.5317],[-118.5141,34.0222],[-120.6366,34.5629],[-122.5205,37.5314],[-121.4454,38.0039],[-123.0275,37.9945],[-124.4092,40.4438],[-124.0227,46.2308],[-123.1801,46.1791],[-124.0802,46.2717],[-124.7275,48.3877],[-122.7702,48.1411],[-122.7225,47.0893],[-122.1917,48.0263],[-122.753,48.9925]],"ca":[[-95.1606,49.3695],[-123.0905,48.9925],[-124.8138,50.9272],[-127.777,51.1672],[-126.9696,52.8294],[-128.3922,52.296],[-128.5983,54.0302],[-130.4825,54.3737],[-129.478,55.4835],[-135.0329,59.5731],[-137.4869,58.9001],[-140.9949,60.3044],[-141.0056,69.6509],[-135.149,68.6605],[-129.6739,70.2699],[-132.922,68.694],[-127.9915,70.5929],[-110.0596,68.0051],[-107.2415,66.3485],[-107.8795,68.093],[-105.6483,68.6387],[-108.8151,68.2742],[-106.2218,68.946],[-102.2189,67.6918],[-95.8967,68.2967],[-95.6435,66.6693],[-93.3749,68.6387],[-96.5319,70.1362],[-95.2252,71.9495],[-90.2708,68.2421],[-88.0226,68.8087],[-87.5162,67.1187],[-84.5282,69.0236],[-85.5632,69.8581],[-82.2534,69.6406],[-83.2938,69.5399],[-81.2733,69.0987],[-82.6506,68.4339],[-81.5029,67.001],[-84.9391,67.056],[-83.6908,66.1992],[-86.7825,66.5222],[-85.8567,66.1628],[-87.3808,65.3297],[-91.4973,65.9468],[-86.9382,65.142],[-90.2206,63.6132],[-93.7795,64.1915],[-90.6265,63.0613],[-94.8006,60.4984],[-92.882,56.9145],[-82.3309,55.1625],[-81.8828,52.1936],[-79.3343,50.7269],[-78.4082,52.2482],[-79.7656,54.6596],[-76.5151,56.4382],[-78.5665,58.6208],[-77.1892,60.0602],[-78.1549,62.2973],[-73.6707,62.4821],[-69.3697,60.9168],[-71.0351,60.0665],[-68.3616,58.7835],[-69.3697,57.7705],[-66.0534,58.3328],[-64.3806,60.2452],[-60.6876,54.9954],[-57.3504,54.5894],[-60.9203,53.7143],[-57.4662,54.2003],[-55.6848,52.1117],[-60.1349,50.213],[-66.4682,50.2685],[-82.4513,41.6719],[-82.5321,45.2935],[-84.1104,46.5265],[-95.1606,49.3695]],"mx":[[-97.1393,25.9658],[-97.891,22.5961],[-95.852,18.7168],[-94.4669,18.1442],[-92.4051,18.6711],[-91.4865,18.4364],[-91.182,18.6566],[-91.5053,18.8166],[-90.7424,19.343],[-90.1995,21.1142],[-88.1295,21.616],[-87.0433,21.6021],[-86.7416,21.1641],[-87.7412,19.6717],[-87.4127,19.5776],[-87.8446,18.1967],[-88.0363,18.875],[-88.8714,17.891],[-90.9909,17.802],[-90.992,17.2519],[-91.4425,17.238],[-90.4374,16.0989],[-91.7398,16.061],[-92.2261,14.5493],[-93.8981,16.0889],[-94.8521,16.4289],[-96.2366,15.6833],[-97.7727,15.9795],[-103.3616,18.2668],[-104.9936,19.3422],[-105.6987,20.4069],[-105.2436,20.5836],[-105.5449,20.7681],[-105.185,21.443],[-105.7325,22.5414],[-107.9952,24.6484],[-108.3279,25.1112],[-108.0035,25.0314],[-109.1161,25.5363],[-108.838,25.8046],[-109.4103,25.6461],[-109.0882,26.2904],[-110.5223,27.2904],[-110.5133,27.8732],[-111.105,27.9354],[-112.1665,28.9667],[-113.2329,31.288],[-115.0294,31.9709],[-114.6603,30.1833],[-111.5601,26.6957],[-110.6099,24.2537],[-110.2326,24.3432],[-109.468,23.5551],[-110.0275,22.9162],[-110.3074,23.541],[-112.088,24.7734],[-112.1789,25.9552],[-113.1457,26.9701],[-113.6371,26.7255],[-115.0227,27.7403],[-113.9533,27.6557],[-114.2765,27.9011],[-114.0682,28.518],[-115.6976,29.7558],[-117.1251,32.5317],[-114.7243,32.7128],[-111.0063,31.3272],[-108.2148,31.3274],[-108.2151,31.7778],[-106.429,31.7585],[-104.9343,30.6105],[-104.5166,29.6543],[-103.3866,29.0288],[-102.321,29.8939],[-101.4051,29.7784],[-99.515,27.5886],[-99.1064,26.423],[-97.1393,25.9658]],"eg":[[34.2484,31.2114],[34.8867,29.4901],[34.4102,28.32],[34.4349,27.9756],[34.2573,27.7286],[33.7634,28.024],[33.2287,28.5694],[33.1779,28.9896],[32.715,29.456],[32.5778,30.0114],[32.3377,29.5925],[32.5812,29.3691],[32.6255,28.9727],[32.8689,28.5789],[33.5544,27.8965],[33.5921,27.7912],[33.4735,27.8232],[33.5921,27.6466],[33.4963,27.6466],[33.9961,26.8966],[33.9414,26.6535],[34.5466,25.725],[35.1446,24.5002],[35.7925,23.9023],[35.4912,23.9476],[35.6211,23.1393],[35.2121,22.7863],[35.6211,23.1393],[35.8475,22.763],[36.2261,22.6354],[36.8836,21.9957],[31.4355,21.9954],[31.4233,22.227],[31.2484,21.9944],[24.9812,21.9954],[24.9812,29.1814],[24.6887,30.1828],[24.9949,30.7851],[24.849,31.3486],[25.0884,31.6135],[25.4019,31.5032],[25.9668,31.6166],[27.3206,31.381],[27.4463,31.2232],[27.8529,31.2403],[27.9269,31.099],[28.4243,31.0834],[29.0281,30.8271],[29.5033,30.9567],[30.0696,31.3335],[30.2887,31.2379],[30.1795,31.2783],[30.3622,31.5087],[30.9739,31.5867],[30.5579,31.4222],[30.7344,31.3881],[31.1192,31.4931],[30.978,31.5843],[31.0974,31.6079],[31.5568,31.4433],[31.9448,31.5209],[32.1996,31.2951],[31.8895,31.532],[31.7669,31.2847],[32.0371,31.2168],[32.1065,31.0534],[32.2835,31.1365],[32.207,31.2951],[32.6016,31.061],[32.9273,31.1554],[32.7007,31.0387],[33.0099,31.0563],[33.0643,31.1349],[32.9544,31.1069],[33.1094,31.1964],[33.1333,31.0456],[34.2003,31.3143],[34.2484,31.2114]],"mr":[[-8.6824,27.2854],[-4.8216,24.9951],[-6.5931,24.9941],[-5.6233,16.5278],[-5.3533,16.312],[-5.511,15.4942],[-9.3492,15.4956],[-9.3565,15.6979],[-9.4517,15.5883],[-9.4239,15.4405],[-10.0676,15.3647],[-10.7254,15.4331],[-10.9152,15.1029],[-11.5244,15.6435],[-11.6149,15.5412],[-11.7271,15.5412],[-11.8457,15.1791],[-11.8204,14.9],[-12.0704,14.7344],[-12.3825,14.846],[-12.4849,15.0173],[-12.8884,15.2509],[-12.828,15.2839],[-12.9643,15.5092],[-13.0838,15.4928],[-13.1097,15.5964],[-13.2378,15.6234],[-13.3159,15.9304],[-13.5017,16.0951],[-13.4849,16.155],[-13.669,16.1057],[-13.7294,16.1895],[-13.8358,16.1206],[-13.9641,16.2301],[-13.9779,16.338],[-14.4066,16.6608],[-14.9125,16.6406],[-14.9743,16.6914],[-15.1076,16.6654],[-15.1251,16.5827],[-15.464,16.5815],[-15.6722,16.4814],[-16.2738,16.5232],[-16.4588,16.1942],[-16.5423,15.8088],[-16.5293,16.3327],[-16.079,17.5448],[-16.0352,18.1414],[-16.2046,18.9827],[-16.5355,19.3808],[-16.3592,19.4158],[-16.2849,19.5293],[-16.3915,19.4641],[-16.3647,19.5393],[-16.4425,19.4103],[-16.4685,19.4557],[-16.2338,19.7936],[-16.248,19.9029],[-16.3133,19.8897],[-16.2225,20.0036],[-16.2611,20.1357],[-16.2023,20.2333],[-16.4672,20.6496],[-16.4087,20.675],[-16.5293,20.739],[-16.5423,20.5609],[-16.6727,20.6776],[-16.9176,21.1592],[-17.0159,21.0312],[-17.0569,20.7669],[-16.9683,21.3246],[-13.0152,21.3334],[-13.1523,22.82],[-13.0152,23.018],[-12.6194,23.2708],[-12.0193,23.461],[-12.0153,25.9949],[-8.6877,26],[-8.6824,27.2854]],"gq":[[9.7996,2.3417],[9.8111,2.3249],[9.8109,2.2649],[9.8237,2.249],[9.8721,2.2112],[9.9075,2.2005],[9.9707,2.1689],[9.991,2.1656],[11.3221,2.1658],[11.3297,2.1274],[11.3363,0.9992],[10.0294,1.0035],[9.9808,0.9972],[9.9718,0.9398],[9.96,0.9284],[9.9441,0.9249],[9.9086,0.9351],[9.8496,0.9892],[9.8044,0.9984],[9.7991,1.0582],[9.8198,1.0689],[9.8475,1.0725],[9.8293,1.0819],[9.7922,1.0887],[9.7551,1.1059],[9.7141,1.0971],[9.7035,1.0998],[9.6994,1.1098],[9.7038,1.1206],[9.7178,1.1333],[9.6992,1.1255],[9.6764,1.1054],[9.6421,1.0616],[9.5843,1.043],[9.5516,1.1154],[9.5248,1.1333],[9.5065,1.1376],[9.45,1.1233],[9.4204,1.1092],[9.4053,1.1059],[9.3965,1.1115],[9.3602,1.147],[9.3596,1.1649],[9.3467,1.1817],[9.374,1.1999],[9.4076,1.2838],[9.4343,1.2998],[9.4483,1.3191],[9.464,1.3598],[9.4614,1.3981],[9.4653,1.4145],[9.4928,1.4266],[9.5322,1.4623],[9.5642,1.5034],[9.5942,1.5722],[9.6148,1.588],[9.6592,1.5653],[9.6859,1.5688],[9.7445,1.6],[9.7243,1.599],[9.6834,1.5714],[9.6657,1.5756],[9.6223,1.6168],[9.6073,1.6444],[9.6175,1.6715],[9.6763,1.7297],[9.6898,1.7502],[9.7181,1.7718],[9.743,1.8157],[9.7468,1.8399],[9.7637,1.8661],[9.7991,1.9011],[9.8097,1.935],[9.8058,1.9764],[9.7644,2.0963],[9.7781,2.124],[9.7864,2.2481],[9.7835,2.2707],[9.7669,2.3105],[9.7798,2.3279],[9.7996,2.3417]],"gm":[[-16.7537,13.065],[-16.7867,13.0901],[-16.8297,13.3385],[-16.6788,13.4962],[-16.5907,13.4614],[-16.6243,13.4272],[-16.5907,13.3385],[-16.5458,13.2901],[-16.4197,13.2556],[-16.4194,13.2082],[-16.3778,13.215],[-16.4399,13.2764],[-16.3747,13.2639],[-16.258,13.3249],[-16.2301,13.3164],[-16.2276,13.256],[-16.1513,13.2764],[-16.1993,13.2591],[-16.2271,13.3337],[-16.1612,13.4287],[-15.9949,13.413],[-15.7645,13.4369],[-15.6687,13.4757],[-15.6208,13.4548],[-15.5434,13.5097],[-15.3844,13.4435],[-15.3048,13.4553],[-15.2947,13.4962],[-15.3517,13.4567],[-15.5561,13.5283],[-15.6107,13.4893],[-15.6721,13.503],[-15.7699,13.4651],[-16.1604,13.4506],[-16.273,13.3626],[-16.2891,13.4068],[-16.3021,13.3528],[-16.412,13.3385],[-16.5218,13.359],[-16.5035,13.4169],[-16.5565,13.4856],[-16.5614,13.5869],[-15.5182,13.5831],[-15.4675,13.7083],[-15.3941,13.7715],[-15.3045,13.7819],[-15.2673,13.7418],[-15.0977,13.82],[-14.8793,13.7806],[-14.8029,13.6523],[-14.7409,13.6154],[-14.6267,13.6635],[-14.5426,13.6408],[-14.4703,13.5224],[-14.3656,13.4548],[-13.9885,13.5791],[-13.9013,13.5416],[-13.8187,13.4294],[-13.851,13.3358],[-13.9006,13.3146],[-14.1168,13.2822],[-14.2017,13.2296],[-14.3688,13.2357],[-14.5922,13.3531],[-14.7316,13.359],[-14.7903,13.4173],[-14.9497,13.4631],[-15.1377,13.59],[-15.204,13.5369],[-15.2278,13.4252],[-15.2771,13.3803],[-15.3792,13.3625],[-15.5195,13.3866],[-15.8187,13.3335],[-15.8333,13.1568],[-16.6735,13.1643],[-16.7084,13.1567],[-16.7537,13.065]],"au":[[131.5359,-31.6056],[126.0076,-32.2734],[123.5266,-33.9387],[120.0022,-33.926],[117.9486,-35.1174],[115.9785,-34.8356],[115.0104,-34.245],[115.7215,-31.7726],[113.157,-26.1519],[113.8703,-26.5221],[113.5153,-25.5093],[114.2344,-26.3166],[113.3914,-24.4137],[113.947,-21.9526],[114.3374,-22.4938],[116.7909,-20.5349],[121.1283,-19.5207],[122.9162,-16.4159],[123.5708,-17.5951],[123.5702,-16.1769],[124.904,-16.4024],[124.3765,-15.5253],[125.1852,-15.5178],[124.842,-15.1412],[126.0864,-13.8894],[127.4185,-13.942],[128.2313,-14.7182],[128.0243,-15.4965],[128.5324,-14.7529],[129.7271,-15.1936],[129.3729,-14.3293],[130.5784,-12.4008],[132.7576,-12.1334],[131.9856,-11.1292],[135.2176,-12.3008],[135.9004,-11.7592],[136.0339,-12.4659],[136.5722,-11.8964],[136.9837,-12.3455],[135.8834,-13.3258],[135.4696,-14.948],[140.5232,-17.6303],[141.4312,-16.0618],[141.5906,-12.5408],[142.5462,-10.6877],[143.7827,-14.4023],[144.4863,-14.1619],[145.3457,-14.9426],[146.3333,-18.9591],[148.782,-20.2408],[149.6633,-22.4938],[150.6775,-22.358],[150.814,-23.5092],[153.2056,-25.935],[153.6306,-28.6616],[153.0595,-31.0772],[151.2307,-33.5349],[149.9854,-37.4959],[147.7576,-37.9851],[146.2241,-38.7145],[146.3942,-39.145],[144.6768,-38.3332],[144.9348,-37.8679],[143.537,-38.8583],[141.3945,-38.4016],[139.7949,-37.2687],[138.9178,-35.5768],[139.672,-36.2278],[139.363,-35.3769],[138.1074,-35.6207],[138.5506,-34.7819],[138.0781,-34.1284],[137.7495,-35.131],[136.8394,-35.2561],[137.9309,-33.6165],[137.7739,-32.5198],[135.9646,-35.0075],[135.1081,-34.596],[134.1483,-32.457],[131.5359,-31.6056]],"mg":[[44.2547,-20.3761],[44.4739,-19.9863],[44.3709,-19.774],[44.4797,-19.484],[44.2344,-19.0888],[43.924,-17.5465],[44.4307,-16.7004],[44.4447,-16.1957],[44.8683,-16.2209],[45.2664,-15.9239],[45.2979,-16.112],[45.3794,-15.9738],[45.5701,-15.943],[45.6047,-16.0519],[45.6609,-15.8038],[46.0661,-15.8619],[46.1442,-15.7037],[46.3311,-15.9772],[46.4688,-15.9567],[46.299,-15.8157],[46.3327,-15.6301],[46.9495,-15.1984],[47.0747,-15.324],[46.9583,-15.5496],[47.2321,-15.4333],[47.0559,-15.1936],[47.4448,-14.6716],[47.4277,-15.1076],[47.7971,-14.5663],[48.0003,-14.7623],[48.013,-14.6301],[47.7529,-14.6005],[47.7015,-14.4464],[47.7909,-14.2232],[47.9435,-14.2325],[47.9201,-14.086],[48.0339,-14.2624],[47.9043,-13.6032],[48.0684,-13.5173],[48.2987,-13.7983],[48.3333,-13.5463],[48.5342,-13.5208],[48.4823,-13.3604],[48.7801,-13.3839],[48.9553,-12.8117],[48.73,-12.435],[48.9458,-12.4799],[49.2781,-11.9477],[49.3731,-12.1982],[49.2297,-12.2221],[49.5394,-12.3782],[49.6548,-12.8049],[49.7442,-12.7434],[49.9421,-13.0339],[50.2097,-14.7507],[50.49,-15.2195],[50.4834,-15.4435],[50.1681,-15.9846],[49.9011,-15.4223],[49.6384,-15.5446],[49.6898,-16.0868],[49.8499,-16.2408],[49.7249,-16.754],[49.8403,-16.8276],[49.6084,-16.8966],[49.4313,-17.2839],[49.4408,-18.1542],[47.1261,-24.9343],[46.7698,-25.1535],[46.2376,-25.1982],[45.5612,-25.56],[45.1626,-25.5986],[44.0288,-24.999],[43.6729,-24.3349],[43.6248,-23.7619],[43.7612,-23.4638],[43.3612,-22.8432],[43.2229,-22.2548],[43.5075,-21.3095],[43.8128,-21.2243],[44.2547,-20.3761]],"jp":[[133.4222,34.445],[132.6563,34.1993],[132.346,34.3488],[132.0414,33.7763],[131.7415,34.0562],[130.9158,33.914],[130.9478,34.4189],[131.4143,34.424],[132.6354,35.441],[133.0913,35.6015],[133.4002,35.4543],[135.2268,35.7723],[135.319,35.447],[136.0691,35.6619],[135.9632,35.9999],[136.6897,36.7308],[136.7491,37.3579],[137.341,37.5171],[136.8609,37.0768],[137.0471,37.0988],[137.0055,36.8359],[137.3333,36.7627],[138.5506,37.3779],[138.8577,37.8278],[139.4081,38.1319],[140.0486,39.5047],[140.0276,39.8245],[139.701,39.9607],[140.0112,40.2355],[139.8604,40.6088],[140.2708,40.8124],[140.3423,41.2657],[140.6327,41.1943],[140.7031,40.8533],[140.8887,41.0108],[141.1263,40.8729],[141.2059,41.2647],[140.7658,41.145],[140.9071,41.5385],[141.4606,41.4302],[141.4374,40.6701],[141.8304,40.2512],[142.0584,39.4663],[141.4631,38.6614],[141.5254,38.2678],[141.3045,38.4079],[140.9668,38.1705],[141.0078,37.1339],[140.5641,36.2831],[140.8794,35.7205],[140.4835,35.566],[140.3776,35.1764],[139.757,34.9563],[139.781,35.3181],[140.102,35.5671],[139.7653,35.6535],[139.6851,35.1404],[139.5442,35.3079],[139.1835,35.2633],[139.1409,34.8706],[138.8353,34.5959],[138.8033,35.1219],[138.5541,35.0961],[138.2333,34.5946],[137.0161,34.5788],[137.3408,34.7265],[136.9789,34.9242],[136.9715,34.6964],[136.856,34.7398],[136.8301,35.0788],[136.5195,34.6809],[136.916,34.434],[136.8957,34.2669],[136.3367,34.184],[135.76,33.4333],[135.0589,33.88],[135.19,34.1496],[135.061,34.2673],[135.4129,34.6921],[134.4912,34.7871],[133.9363,34.4512],[133.4222,34.445]]};
function computePasses(sat, cty, numDays, startT = 0) {
  const T = orbPeriod(sat.altitude);
  const totalDays = startT / 86400 + numDays;
  const nOrbits = Math.ceil((86400 / T) * totalDays);
  
  // Adaptive ppo: satellite traverses ~2*inclination degrees of latitude per orbit.
  // For small countries, we need enough points so that passes aren't skipped.
  // Require at least 4 points within the country's latitude span per orbit.
  const latSpan = cty.latMax - cty.latMin;
  const fullLatTraverse = 2 * Math.min(sat.inclination, 180 - sat.inclination);
  const minPpoForSize = Math.ceil((fullLatTraverse / Math.max(latSpan, 0.5)) * 4);
  const ppo = Math.max(minPpoForSize, 500);
  // Cap total points at 500k for performance
  const safePpo = nOrbits * ppo > 500000 ? Math.max(500, Math.round(500000 / nOrbits)) : ppo;
  const pts = constellationTrack(sat, nOrbits, safePpo);

  const sw = effSwathKm(sat.altitude, sat.swathAngle || 10);
  const swDegLat = (sw / (2 * Math.PI * R_E)) * 360;
  const latBuf = swDegLat / 2 + 0.8;
  const lonBuf = 1.5;

  let passes = [];
  let cur = null;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const over = p.lat >= cty.latMin - latBuf && p.lat <= cty.latMax + latBuf &&
                 p.lon >= cty.lonMin - lonBuf && p.lon <= cty.lonMax + lonBuf;
    if (over) {
      if (!cur) {
        cur = [p];
      } else {
        // Split into new pass if time gap > 40% of orbital period (different orbit)
        const dt = p.t - cur[cur.length - 1].t;
        if (dt > T * 0.4) {
          if (cur.length >= 2) passes.push(cur);
          cur = [p];
        } else {
          cur.push(p);
        }
      }
    } else if (cur) {
      if (cur.length >= 2) passes.push(cur);
      cur = null;
    }
  }
  if (cur && cur.length >= 2) passes.push(cur);

  // If a start time offset is given, filter to the requested window [startT, startT + numDays*86400]
  if (startT > 0) {
    const windowEnd = startT + numDays * 86400;
    passes = passes.filter(pass => {
      const midT = pass[Math.floor(pass.length / 2)].t;
      return midT >= startT && midT < windowEnd;
    });
  }

  // Build swath polygons perpendicular to heading
  const passPolys = passes.map(pass => {
    const left = [], right = [];
    for (let i = 0; i < pass.length; i++) {
      const p = pass[i];
      let dlat, dlon;
      if (i < pass.length - 1) { dlat = pass[i+1].lat - p.lat; dlon = pass[i+1].lon - p.lon; }
      else { dlat = p.lat - pass[i-1].lat; dlon = p.lon - pass[i-1].lon; }
      const cosL = Math.max(Math.cos(p.lat * DEG), 0.05);
      const mag = Math.sqrt(dlat * dlat + (dlon * cosL) * (dlon * cosL));
      if (mag < 1e-10) { left.push(p); right.push(p); continue; }
      const perpLat = -dlon * cosL / mag * (swDegLat / 2);
      const perpLon = dlat / mag * (swDegLat / 2) / cosL;
      left.push({ lat: p.lat + perpLat, lon: p.lon + perpLon });
      right.push({ lat: p.lat - perpLat, lon: p.lon - perpLon });
    }
    return { center: pass, left, right };
  });

  // Coverage grid
  const gN = 30;
  const dLat = cty.latMax - cty.latMin, dLon = cty.lonMax - cty.lonMin;
  const cov = new Uint8Array(gN * gN);
  for (const pass of passes) {
    for (const p of pass) {
      const cosL = Math.max(Math.cos(p.lat * DEG), 0.05);
      const halfLon = swDegLat / 2 / cosL;
      for (let gi = 0; gi < gN; gi++) {
        const gLat = cty.latMin + (gi + 0.5) * dLat / gN;
        if (Math.abs(gLat - p.lat) > swDegLat / 2) continue;
        for (let gj = 0; gj < gN; gj++) {
          const gLon = cty.lonMin + (gj + 0.5) * dLon / gN;
          if (Math.abs(gLon - p.lon) > halfLon) continue;
          cov[gi * gN + gj] = 1;
        }
      }
    }
  }
  const pct = (cov.reduce((s, v) => s + v, 0) / (gN * gN) * 100).toFixed(1);
  const passMidTimes = passes.map(pass => pass[Math.floor(pass.length / 2)].t);
  // Use all buffer-zone pass points with haversine, capped at country diagonal.
  const passLengthsKm = passes.map(pass => passTrackLength(pass, cty));
  return { passPolys, covPct: pct, swDegLat, passes, passMidTimes, passLengthsKm };
}

/* Render a ground-track map to a JPEG data URL with OSM tile background.
   Replicates the full CountryTrackMap appearance: tiles → country border → swath
   bands → track lines → info badges. Uses crossOrigin="anonymous" so OSM tiles
   (which serve Access-Control-Allow-Origin: *) can be drawn to a readable canvas. */
async function renderTrackMapToDataURL(sat, cty, anchorOffset, tfDays, sunlitOnly, W = 600, H = 480) {
  const { passPolys, passes, passMidTimes, passLengthsKm } = computePasses(sat, cty, tfDays, anchorOffset);
  const midLat = (cty.latMin + cty.latMax) / 2;
  const midLon = (cty.lonMin + cty.lonMax) / 2;
  const sunlitIdxs = passes.map((_, i) => i).filter(i => isInImagingWindow(passMidTimes[i], midLat, midLon));

  // Viewport & Mercator projection (identical to CountryTrackMap)
  const dLat = cty.latMax - cty.latMin, dLon = cty.lonMax - cty.lonMin;
  const margin = Math.min(Math.max(dLat, dLon) * 0.25, 0.5);
  const vLatMin = cty.latMin - margin, vLatMax = cty.latMax + margin;
  const vLonMin = cty.lonMin - margin, vLonMax = cty.lonMax + margin;
  const vdLon = vLonMax - vLonMin;
  const latToMerc = lat => Math.log(Math.tan((45 + lat / 2) * DEG));
  const mercMin = latToMerc(vLatMin), mercMax = latToMerc(vLatMax);
  const mercSpan = mercMax - mercMin;
  const tX = lon => ((lon - vLonMin) / vdLon) * W;
  const tY = lat => ((mercMax - latToMerc(Math.max(-85, Math.min(85, lat)))) / mercSpan) * H;

  // ── OSM tile loading (crossOrigin="anonymous" so canvas stays readable) ──
  const maxSpan = Math.max(vLatMax - vLatMin, (vLonMax - vLonMin) * Math.cos(midLat * DEG));
  const zoom = Math.max(1, Math.min(12, Math.round(Math.log2(360 / maxSpan)) + 1));
  const n = Math.pow(2, zoom);
  const tileXMin = Math.floor(((vLonMin + 180) / 360) * n);
  const tileXMax = Math.floor(((vLonMax + 180) / 360) * n);
  const tileYMin = Math.floor((1 - latToMerc(vLatMax) / Math.PI) / 2 * n);
  const tileYMax = Math.floor((1 - latToMerc(vLatMin) / Math.PI) / 2 * n);
  const tW = (tileXMax - tileXMin + 1) * 256, tH = (tileYMax - tileYMin + 1) * 256;

  const tileCvs = document.createElement('canvas');
  tileCvs.width = tW; tileCvs.height = tH;
  const tileCtx = tileCvs.getContext('2d');
  tileCtx.fillStyle = '#aad3df';
  tileCtx.fillRect(0, 0, tW, tH);

  await Promise.all(
    Array.from({ length: tileXMax - tileXMin + 1 }, (_, i) => tileXMin + i).flatMap(tx =>
      Array.from({ length: tileYMax - tileYMin + 1 }, (_, j) => tileYMin + j).map(ty =>
        new Promise(res => {
          const img = new window.Image();
          img.crossOrigin = 'anonymous';
          const dx = (tx - tileXMin) * 256, dy = (ty - tileYMin) * 256;
          img.onload = () => { tileCtx.drawImage(img, dx, dy, 256, 256); res(); };
          img.onerror = () => { tileCtx.fillStyle = '#c8d8b0'; tileCtx.fillRect(dx, dy, 256, 256); res(); };
          img.src = `https://${['a','b','c'][(tx + ty) % 3]}.tile.openstreetmap.org/${zoom}/${tx}/${ty}.png`;
        })
      )
    )
  );

  // Crop tile composite to viewport → draw into final canvas
  const vpxL = ((vLonMin + 180) / 360 * n - tileXMin) * 256;
  const vpxR = ((vLonMax + 180) / 360 * n - tileXMin) * 256;
  const vpyT = ((1 - latToMerc(vLatMax) / Math.PI) / 2 * n - tileYMin) * 256;
  const vpyB = ((1 - latToMerc(vLatMin) / Math.PI) / 2 * n - tileYMin) * 256;

  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(tileCvs, vpxL, vpyT, vpxR - vpxL, vpyB - vpyT, 0, 0, W, H);

  // Light whitening overlay (matches app's rgba(255,255,255,0.1))
  ctx.fillStyle = 'rgba(255,255,255,0.1)';
  ctx.fillRect(0, 0, W, H);

  // ── Grid lines ──
  const gStep = dLat > 15 ? 5 : dLat > 5 ? 2 : dLat > 2 ? 1 : 0.5;
  ctx.save();
  ctx.strokeStyle = 'rgba(0,0,0,0.12)';
  ctx.lineWidth = 0.7;
  ctx.font = '9px monospace';
  ctx.fillStyle = 'rgba(0,0,0,0.45)';
  for (let la = Math.ceil(vLatMin / gStep) * gStep; la <= vLatMax; la += gStep) {
    ctx.beginPath(); ctx.moveTo(0, tY(la)); ctx.lineTo(W, tY(la)); ctx.stroke();
    ctx.fillText(`${la.toFixed(gStep < 1 ? 1 : 0)}°`, 4, tY(la) - 2);
  }
  for (let lo = Math.ceil(vLonMin / gStep) * gStep; lo <= vLonMax; lo += gStep) {
    ctx.beginPath(); ctx.moveTo(tX(lo), 0); ctx.lineTo(tX(lo), H); ctx.stroke();
    ctx.fillText(`${lo.toFixed(gStep < 1 ? 1 : 0)}°`, tX(lo) + 2, H - 4);
  }
  ctx.restore();

  // ── Country border ──
  const bdr = BORDERS[cty.id];
  ctx.save();
  ctx.beginPath();
  if (bdr && bdr.length > 0) {
    bdr.forEach(([lo, la], i) => { if (i === 0) ctx.moveTo(tX(lo), tY(la)); else ctx.lineTo(tX(lo), tY(la)); });
    ctx.closePath();
  } else {
    ctx.rect(tX(cty.lonMin), tY(cty.latMax), tX(cty.lonMax) - tX(cty.lonMin), tY(cty.latMin) - tY(cty.latMax));
  }
  ctx.fillStyle = 'rgba(120,175,100,0.15)';
  ctx.fill();
  ctx.setLineDash([8, 4]);
  ctx.strokeStyle = '#2a6020';
  ctx.lineWidth = 2.5;
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = '#1a5018';
  ctx.lineWidth = 2.5;
  ctx.stroke();
  ctx.restore();

  // ── Track center lines (no swath corridor — only nadir track lines) ──
  ctx.save();
  ctx.strokeStyle = 'rgba(15,70,200,0.85)';
  ctx.lineWidth = 2;
  for (let pi = 0; pi < passPolys.length; pi++) {
    const poly = passPolys[pi];
    if (poly.center.length < 2) continue;
    // Cull passes entirely outside the visible viewport
    const inView = poly.center.some(p =>
      p.lat >= vLatMin - 1 && p.lat <= vLatMax + 1 &&
      p.lon >= vLonMin - 1 && p.lon <= vLonMax + 1
    );
    if (!inView) continue;
    ctx.beginPath();
    poly.center.forEach((p, i) => { if (i === 0) ctx.moveTo(tX(p.lon), tY(p.lat)); else ctx.lineTo(tX(p.lon), tY(p.lat)); });
    ctx.stroke();
  }
  ctx.restore();

  // ── Info badge (top-left) ──
  const sw = effSwathKm(sat.altitude, sat.swathAngle || 10);
  const covKm2 = sunlitIdxs.reduce((s, i) => s + sw * (passLengthsKm[i] || 0), 0);
  const fmtA = v => v >= 1e6 ? `${(v/1e6).toFixed(1)}M` : v >= 1e3 ? `${(v/1e3).toFixed(1)}k` : v.toFixed(0);
  const lines = [
    { text: cty.name, font: 'bold 14px monospace', fill: '#ffffff' },
    { text: sat.name, font: 'bold 11px monospace', fill: sat.color || '#64FFDA' },
    { text: `${sat.altitude}km · ${sat.inclination}° · ${sw.toFixed(0)}km swath · ${sunlitOnly ? sunlitIdxs.length : passes.length}${sunlitOnly ? ' ☀' : ''} passes/${tfDays}d`, font: '9px monospace', fill: 'rgba(200,220,240,0.65)' },
    covKm2 > 0 ? { text: `☀ ${fmtA(covKm2)} km² imaged / ${fmtA(cty.area)} km²`, font: 'bold 10px monospace', fill: '#64FFDA' } : null,
  ].filter(Boolean);
  const badgeH = lines.length * 16 + 12;
  ctx.save();
  ctx.fillStyle = 'rgba(0,0,0,0.78)';
  roundRect(ctx, 8, 8, 220, badgeH, 6);
  ctx.fill();
  lines.forEach((l, i) => {
    ctx.font = l.font; ctx.fillStyle = l.fill;
    ctx.fillText(l.text, 16, 24 + i * 16);
  });
  ctx.restore();

  // ── Orbital params badge (bottom-left) ──
  const T = orbPeriod(sat.altitude);
  const cosM = Math.cos(midLat * DEG);
  const gapKm = ((360 / (86400 / T)) * (2 * Math.PI * R_E / 360) * cosM).toFixed(0);
  const params = [
    ['Orbits/d', (86400 / T).toFixed(1)],
    ['Ω̇', `${nodalPrec(sat.altitude, sat.inclination, sat.eccentricity || 0).toFixed(2)}°/d`],
    ['Gap', `~${gapKm}km`],
    ['Area', `${(cty.area/1000).toFixed(0)}k km²`],
  ];
  const pBadgeW = params.length * 70 + 16;
  ctx.save();
  ctx.fillStyle = 'rgba(0,0,0,0.68)';
  roundRect(ctx, 8, H - 42, pBadgeW, 34, 4);
  ctx.fill();
  params.forEach(([k, v], i) => {
    const px = 16 + i * 70;
    ctx.font = '8px monospace'; ctx.fillStyle = 'rgba(160,190,210,0.55)';
    ctx.fillText(k, px, H - 27);
    ctx.font = 'bold 10px monospace'; ctx.fillStyle = '#a0c8e0';
    ctx.fillText(v, px, H - 13);
  });
  ctx.restore();

  return canvas.toDataURL('image/jpeg', 0.92);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y); ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r); ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h); ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r); ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

function dataUrlToUint8Array(dataUrl) {
  const b64 = dataUrl.split(',')[1];
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return arr;
}

// OSM tile compositing on a canvas for a real map background
function useMapImage(cty, svgW, svgH) {
  const [imgSrc, setImgSrc] = useState(null);
  const canvasRef = useRef(null);

  const dLat = cty.latMax - cty.latMin, dLon = cty.lonMax - cty.lonMin;
  const margin = Math.min(Math.max(dLat, dLon) * 0.25, 0.5);
  const vLatMin = cty.latMin - margin, vLatMax = cty.latMax + margin;
  const vLonMin = cty.lonMin - margin, vLonMax = cty.lonMax + margin;

  // Calculate optimal zoom
  const maxSpan = Math.max(vLatMax - vLatMin, (vLonMax - vLonMin) * Math.cos(((cty.latMin + cty.latMax) / 2) * DEG));
  const zoom = Math.max(1, Math.min(12, Math.round(Math.log2(360 / maxSpan)) + 1));

  useEffect(() => {
    const latMerc = lat => Math.log(Math.tan((45 + lat / 2) * DEG));
    const n = Math.pow(2, zoom);
    const tileXMin = Math.floor(((vLonMin + 180) / 360) * n);
    const tileXMax = Math.floor(((vLonMax + 180) / 360) * n);
    const tileYMin = Math.floor((1 - latMerc(vLatMax) / Math.PI) / 2 * n);
    const tileYMax = Math.floor((1 - latMerc(vLatMin) / Math.PI) / 2 * n);

    const totalTilesX = tileXMax - tileXMin + 1;
    const totalTilesY = tileYMax - tileYMin + 1;
    const compositeW = totalTilesX * 256;
    const compositeH = totalTilesY * 256;

    const cvs = document.createElement("canvas");
    cvs.width = compositeW;
    cvs.height = compositeH;
    const ctx = cvs.getContext("2d");
    // Fill with water color first
    ctx.fillStyle = "#aad3df";
    ctx.fillRect(0, 0, compositeW, compositeH);

    let loaded = 0;
    const totalTiles = totalTilesX * totalTilesY;

    const tryFinalize = () => {
      loaded++;
      if (loaded >= totalTiles) {
        // Crop to our viewport
        const pxPerTileLon = 256;
        const pxPerTileY = 256;

        // Convert viewport bounds to pixel coords within the composite
        const vpxLeft = ((vLonMin + 180) / 360 * n - tileXMin) * 256;
        const vpxRight = ((vLonMax + 180) / 360 * n - tileXMin) * 256;
        const vpyTop = ((1 - latMerc(vLatMax) / Math.PI) / 2 * n - tileYMin) * 256;
        const vpyBot = ((1 - latMerc(vLatMin) / Math.PI) / 2 * n - tileYMin) * 256;

        const cropW = vpxRight - vpxLeft;
        const cropH = vpyBot - vpyTop;

        const outCvs = document.createElement("canvas");
        outCvs.width = Math.round(svgW);
        outCvs.height = Math.round(svgH);
        const outCtx = outCvs.getContext("2d");
        outCtx.drawImage(cvs, vpxLeft, vpyTop, cropW, cropH, 0, 0, svgW, svgH);
        setImgSrc(outCvs.toDataURL("image/jpeg", 0.85));
      }
    };

    for (let tx = tileXMin; tx <= tileXMax; tx++) {
      for (let ty = tileYMin; ty <= tileYMax; ty++) {
        const img = new Image();
        img.crossOrigin = "anonymous";
        const dx = (tx - tileXMin) * 256;
        const dy = (ty - tileYMin) * 256;
        img.onload = () => { ctx.drawImage(img, dx, dy, 256, 256); tryFinalize(); };
        img.onerror = () => {
          // Draw land-colored fallback
          ctx.fillStyle = "#c8d8b0";
          ctx.fillRect(dx, dy, 256, 256);
          tryFinalize();
        };
        const s = ["a","b","c"][(tx + ty) % 3];
        img.src = `https://${s}.tile.openstreetmap.org/${zoom}/${tx}/${ty}.png`;
      }
    }
  }, [cty.id, zoom, svgW, svgH]);

  return imgSrc;
}

function CountryTrackMap({ country, sat, anchorOffset = 0, width = 500, height = 440 }) {
  const [tfIdx, setTfIdx] = useState(2);
  const [sunlitOnly, setSunlitOnly] = useState(false);
  const tfDays = TRACK_TF_OPTS[tfIdx].d;

  const { passPolys, covPct, passes, passMidTimes, passLengthsKm } = useMemo(
    () => computePasses(sat, country, tfDays, anchorOffset),
    [sat.altitude, sat.inclination, sat.eccentricity, sat.raan, sat.argPerigee,
     sat.swathAngle, sat.offNadir, country.id, tfDays, anchorOffset]
  );

  const midLat = (country.latMin + country.latMax) / 2;
  const midLon = (country.lonMin + country.lonMax) / 2;
  const sunlitIdxs = passes.map((_, i) => i).filter(i => isInImagingWindow(passMidTimes[i], midLat, midLon));
  // visiblePasses used for badge count only
  const visiblePasses = sunlitOnly
    ? passes.filter((_, i) => sunlitIdxs.includes(i))
    : passes;

  // Sunlit covered area (always sunlit-only, regardless of toggle), capped at daily capacity
  const swKm = effSwathKm(sat.altitude, sat.swathAngle || 10);
  const uncappedCovKm2 = sunlitIdxs.reduce((sum, i) => sum + swKm * (passLengthsKm[i] || 0), 0);
  const dailyCap = sat.dataCapacity || 0;
  const covKm2 = dailyCap > 0 ? Math.min(uncappedCovKm2, dailyCap * tfDays) : uncappedCovKm2;
  const fmtKm2 = v => v >= 1e6 ? `${(v/1e6).toFixed(1)}M` : v >= 1e3 ? `${(v/1e3).toFixed(1)}k` : v.toFixed(0);

  const cty = country;
  const dLat = cty.latMax - cty.latMin, dLon = cty.lonMax - cty.lonMin;
  const margin = Math.min(Math.max(dLat, dLon) * 0.25, 0.5);
  const vLatMin = cty.latMin - margin, vLatMax = cty.latMax + margin;
  const vLonMin = cty.lonMin - margin, vLonMax = cty.lonMax + margin;
  const vdLon = vLonMax - vLonMin;

  const svgH = height - 48;
  const svgW = width - 8;
  const cosM = Math.cos(((cty.latMin + cty.latMax) / 2) * DEG);

  // Mercator projection
  const latToMerc = lat => Math.log(Math.tan((45 + lat / 2) * DEG));
  const mercMin = latToMerc(vLatMin), mercMax = latToMerc(vLatMax);
  const mercSpan = mercMax - mercMin;
  const tX = lon => ((lon - vLonMin) / vdLon) * svgW;
  const tY = lat => ((mercMax - latToMerc(Math.max(-85, Math.min(85, lat)))) / mercSpan) * svgH;

  // Get real map background
  const mapImg = useMapImage(cty, svgW, svgH);

  // Country border
  const bdr = BORDERS[cty.id];
  const bdrPath = bdr
    ? bdr.map(([lo, la]) => `${tX(lo).toFixed(1)},${tY(la).toFixed(1)}`).join(" ")
    : `${tX(cty.lonMin)},${tY(cty.latMax)} ${tX(cty.lonMax)},${tY(cty.latMax)} ${tX(cty.lonMax)},${tY(cty.latMin)} ${tX(cty.lonMin)},${tY(cty.latMin)}`;

  // Grid
  const gStep = dLat > 15 ? 5 : dLat > 5 ? 2 : dLat > 2 ? 1 : 0.5;
  const latLines = [], lonLines = [];
  for (let la = Math.ceil((vLatMin) / gStep) * gStep; la <= vLatMax; la += gStep) latLines.push(la);
  for (let lo = Math.ceil((vLonMin) / gStep) * gStep; lo <= vLonMax; lo += gStep) lonLines.push(lo);

  const T = orbPeriod(sat.altitude);
  const sw = effSwathKm(sat.altitude, sat.swathAngle || 10);
  const gapKm = ((360 / (86400 / T)) * (2 * Math.PI * R_E / 360) * cosM).toFixed(0);

  return (
    <div style={{
      display: "inline-flex", flexDirection: "column",
      background: "#0a1018", borderRadius: 8,
      border: "1px solid rgba(70,140,200,0.15)",
      overflow: "hidden", width
    }}>
      <div style={{ position: "relative", display: "flex", justifyContent: "center", padding: 4, minHeight: svgH }}>
        {/* Real map background image */}
        {mapImg && (
          <img src={mapImg} alt="" style={{
            position: "absolute", top: 4, left: 4,
            width: svgW, height: svgH, borderRadius: 4,
            objectFit: "fill", zIndex: 0
          }} />
        )}

        {/* SVG overlay — viewport is FIXED to country bounds, tracks clip at viewBox edge */}
        <svg width={svgW} height={svgH} viewBox={`0 0 ${svgW} ${svgH}`}
          style={{ borderRadius: 4, position: "relative", zIndex: 1 }}
          overflow="hidden">

          {/* Explicit clip region — ensures swath polygons never expand layout */}
          <defs>
            <clipPath id={`vp-${cty.id}`}>
              <rect x="0" y="0" width={svgW} height={svgH} />
            </clipPath>
          </defs>

          {/* Fallback background (hidden when map loads) */}
          {!mapImg && <>
            <rect width={svgW} height={svgH} fill="#aad3df" />
            <rect width={svgW} height={svgH} fill="#c4d4b4" opacity="0.7" />
          </>}

          {/* Semi-transparent overlay so tracks are visible over map */}
          {mapImg && <rect width={svgW} height={svgH} fill="rgba(255,255,255,0.1)" />}

          {/* Grid */}
          {latLines.map(la => (
            <g key={`la${la}`}>
              <line x1={0} y1={tY(la)} x2={svgW} y2={tY(la)} stroke="rgba(0,0,0,0.1)" strokeWidth="0.5" />
              <text x={4} y={tY(la) - 2} fontSize="8" fill="rgba(0,0,0,0.4)" fontFamily="'IBM Plex Mono',monospace" fontWeight="600">{la.toFixed(gStep < 1 ? 1 : 0)}°</text>
            </g>
          ))}
          {lonLines.map(lo => (
            <g key={`lo${lo}`}>
              <line x1={tX(lo)} y1={0} x2={tX(lo)} y2={svgH} stroke="rgba(0,0,0,0.1)" strokeWidth="0.5" />
              <text x={tX(lo)+2} y={svgH - 4} fontSize="8" fill="rgba(0,0,0,0.4)" fontFamily="'IBM Plex Mono',monospace" fontWeight="600">{lo.toFixed(gStep < 1 ? 1 : 0)}°</text>
            </g>
          ))}

          {/* Country border fill */}
          <polygon points={bdrPath} fill={mapImg ? "rgba(120,175,100,0.15)" : "rgba(120,175,100,0.45)"} stroke="#2a6020" strokeWidth="2.5" strokeDasharray="8,4" />

          {/* Track lines — clipped to viewport, sunlit filter applied per-pass */}
          <g clipPath={`url(#vp-${cty.id})`}>
            {passPolys.map((poly, pi) => {
              if (sunlitOnly && !isInImagingWindow(passMidTimes[pi], midLat, midLon)) return null;
              if (poly.center.length < 2) return null;
              // Cull passes entirely outside the visible viewport (1° slop for edge-crossing passes)
              const inView = poly.center.some(p =>
                p.lat >= vLatMin - 1 && p.lat <= vLatMax + 1 &&
                p.lon >= vLonMin - 1 && p.lon <= vLonMax + 1
              );
              if (!inView) return null;
              const d = poly.center.map((p, i) => `${i === 0 ? "M" : "L"}${tX(p.lon).toFixed(1)},${tY(p.lat).toFixed(1)}`).join(" ");
              return <path key={`tk${pi}`} d={d} fill="none" stroke="rgba(15,70,200,0.85)" strokeWidth="2" />;
            })}
          </g>

          {/* Country border top */}
          <polygon points={bdrPath} fill="none" stroke="#1a5018" strokeWidth="2.5" />
        </svg>

        {/* Info overlay */}
        <div style={{
          position: "absolute", top: 8, left: 8, zIndex: 10,
          background: "rgba(0,0,0,0.78)", borderRadius: 6, padding: "8px 12px",
          maxWidth: "60%"
        }}>
          <div style={{ color: "#fff", fontSize: 13, fontWeight: 700, fontFamily: "'IBM Plex Mono',monospace" }}>{cty.name}</div>
          <div style={{ color: sat.color, fontSize: 10, fontWeight: 600, fontFamily: "'IBM Plex Mono',monospace", marginTop: 2 }}>{sat.name}</div>
          <div style={{ color: "rgba(200,220,240,0.6)", fontSize: 8, fontFamily: "'IBM Plex Mono',monospace", marginTop: 3, lineHeight: 1.4 }}>
            {sat.altitude}km · {sat.inclination}° · {sw.toFixed(0)}km swath · {visiblePasses.length}{sunlitOnly ? ' ☀' : ''} passes/{tfDays}d
          </div>
          {covKm2 > 0 && (
            <div style={{ marginTop: 4, fontSize: 9, fontFamily: "'IBM Plex Mono',monospace", color: "#64FFDA", fontWeight: 600 }}>
              ☀ {fmtKm2(covKm2)} km² imaged / {fmtKm2(country.area)} km²
              {dailyCap > 0 && uncappedCovKm2 > dailyCap * tfDays && (
                <span style={{ color: "#FFB300", fontSize: 8, marginLeft: 4 }}>(cap {fmtKm2(dailyCap * tfDays)})</span>
              )}
            </div>
          )}
        </div>

        {/* Orbital params */}
        <div style={{
          position: "absolute", bottom: 8, left: 8, zIndex: 10,
          background: "rgba(0,0,0,0.68)", borderRadius: 4, padding: "5px 12px",
          display: "flex", gap: 16
        }}>
          {[
            ["Orbits/day", (86400 / T).toFixed(1)],
            ["Ω̇", `${nodalPrec(sat.altitude, sat.inclination, sat.eccentricity || 0).toFixed(2)}°/d`],
            ["Gap", `~${gapKm}km`],
            ["Area", `${(cty.area/1000).toFixed(0)}k km²`],
          ].map(([k, v]) => (
            <div key={k} style={{ textAlign: "center" }}>
              <div style={{ fontSize: 7, color: "rgba(160,190,210,0.5)", fontFamily: "'IBM Plex Mono',monospace" }}>{k}</div>
              <div style={{ fontSize: 10, color: "#a0c8e0", fontWeight: 600, fontFamily: "'IBM Plex Mono',monospace" }}>{v}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Timeframe buttons */}
      <div style={{
        display: "flex", gap: 4, padding: "8px 12px",
        justifyContent: "center", alignItems: "center",
        borderTop: "1px solid rgba(70,140,200,0.1)"
      }}>
        <span style={{ fontSize: 8, color: "#2a4a60", marginRight: 8, letterSpacing: 1.5, fontWeight: 600, fontFamily: "'IBM Plex Mono',monospace" }}>WINDOW</span>
        {TRACK_TF_OPTS.map((o, i) => (
          <button key={o.l} onClick={() => setTfIdx(i)} style={{
            background: tfIdx === i ? "rgba(30,110,220,0.3)" : "rgba(30,110,220,0.04)",
            border: `1px solid ${tfIdx === i ? "rgba(30,110,220,0.6)" : "rgba(30,110,220,0.1)"}`,
            color: tfIdx === i ? "#90c8f0" : "#2a5068",
            padding: "4px 11px", borderRadius: 4, cursor: "pointer",
            fontSize: 9, fontFamily: "'IBM Plex Mono',monospace",
            fontWeight: tfIdx === i ? 700 : 400, transition: "all 0.15s"
          }}>{o.l}</button>
        ))}
        <button onClick={() => setSunlitOnly(v => !v)} style={{
          marginLeft: 8, padding: "4px 10px", borderRadius: 4, cursor: "pointer",
          background: sunlitOnly ? "rgba(255,179,0,0.2)" : "rgba(70,140,200,0.04)",
          border: `1px solid ${sunlitOnly ? "rgba(255,179,0,0.55)" : "rgba(70,140,200,0.12)"}`,
          color: sunlitOnly ? "#FFB300" : "#2a5068",
          fontSize: 9, fontFamily: "'IBM Plex Mono',monospace", fontWeight: sunlitOnly ? 700 : 400,
          transition: "all 0.15s"
        }}>{sunlitOnly ? "☀ Sunlit" : "All passes"}</button>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   MAIN MAP CANVAS
   ══════════════════════════════════════════════════════════════ */
const MW = 960, MH = 480;
const lonX = lon => ((lon + 180) / 360) * MW;
const latY = lat => ((90 - lat) / 180) * MH;

function MainMap({ sats, selCtys, countries, onCtClick, anim, showSw, showGr, anchorId }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext("2d"); const dpr = window.devicePixelRatio || 1;
    c.width = MW * dpr; c.height = MH * dpr; ctx.scale(dpr, dpr);
    ctx.fillStyle = "#050a12"; ctx.fillRect(0, 0, MW, MH);
    if (showGr) {
      ctx.strokeStyle = "rgba(50,110,170,0.06)"; ctx.lineWidth = 0.5;
      for (let lo = -180; lo <= 180; lo += 30) { ctx.beginPath(); ctx.moveTo(lonX(lo), 0); ctx.lineTo(lonX(lo), MH); ctx.stroke(); }
      for (let la = -90; la <= 90; la += 30) { ctx.beginPath(); ctx.moveTo(0, latY(la)); ctx.lineTo(MW, latY(la)); ctx.stroke(); }
      ctx.strokeStyle = "rgba(50,110,170,0.14)"; ctx.lineWidth = 0.8;
      ctx.beginPath(); ctx.moveTo(0, latY(0)); ctx.lineTo(MW, latY(0)); ctx.stroke();
    }
    ctx.strokeStyle = "rgba(60,140,120,0.28)"; ctx.fillStyle = "rgba(60,140,120,0.04)"; ctx.lineWidth = 0.7;
    COASTS.forEach(path => { ctx.beginPath(); path.forEach(([lo, la], i) => { i === 0 ? ctx.moveTo(lonX(lo), latY(la)) : ctx.lineTo(lonX(lo), latY(la)); }); ctx.closePath(); ctx.fill(); ctx.stroke(); });
    // Countries
    countries.forEach(ct => {
      const sel = selCtys.includes(ct.id);
      const isAnchor = ct.id === anchorId;
      const rx = lonX(ct.lonMin), ry = latY(ct.latMax), rw = lonX(ct.lonMax) - rx, rh = latY(ct.latMin) - ry;
      if (isAnchor) {
        // Anchor country: gold highlight
        ctx.fillStyle = "rgba(255,180,0,0.15)"; ctx.fillRect(rx, ry, rw, rh);
        ctx.strokeStyle = "rgba(255,180,0,0.7)"; ctx.lineWidth = 2.5; ctx.setLineDash([6, 2]); ctx.strokeRect(rx, ry, rw, rh); ctx.setLineDash([]);
        ctx.fillStyle = "rgba(255,180,0,0.9)"; ctx.font = "bold 10px 'IBM Plex Mono', monospace"; ctx.fillText("⚓ " + ct.name, rx + 3, ry + 12);
      } else if (sel) {
        const cc = SAT_COLORS[countries.indexOf(ct) % SAT_COLORS.length];
        ctx.fillStyle = cc + "15"; ctx.fillRect(rx, ry, rw, rh);
        ctx.strokeStyle = cc + "70"; ctx.lineWidth = 1.5; ctx.setLineDash([5, 3]); ctx.strokeRect(rx, ry, rw, rh); ctx.setLineDash([]);
        ctx.fillStyle = cc + "bb"; ctx.font = "bold 9px 'IBM Plex Mono', monospace"; ctx.fillText(ct.name, rx + 3, ry + 11);
      } else {
        ctx.fillStyle = "rgba(100,160,200,0.12)"; ctx.beginPath(); ctx.arc(lonX(ct.lon), latY(ct.lat), 2.5, 0, Math.PI * 2); ctx.fill();
      }
    });
    // Ground tracks
    sats.forEach(sat => {
      const track = constellationTrack(sat, 1, 400);
      const sw = effSwathKm(sat.altitude, sat.swathAngle || 10);
      const swD = (sw / (2 * Math.PI * R_E)) * 360;
      if (showSw) { ctx.fillStyle = sat.color + "0a"; track.forEach(pt => { const x = lonX(pt.lon), y = latY(pt.lat), hh = (swD / 180) * MH / 2; ctx.fillRect(x - 1, y - hh, 2, hh * 2); }); }
      ctx.strokeStyle = sat.color; ctx.lineWidth = 1.5; ctx.shadowColor = sat.color; ctx.shadowBlur = 3; ctx.beginPath();
      let px = null;
      track.forEach((pt, i) => { const x = lonX(pt.lon), y = latY(pt.lat); if (i === 0 || (px !== null && Math.abs(x - px) > MW * 0.4)) ctx.moveTo(x, y); else ctx.lineTo(x, y); px = x; });
      ctx.stroke(); ctx.shadowBlur = 0;
      const pi = anim % track.length, pos = track[pi];
      if (pos) { ctx.beginPath(); ctx.arc(lonX(pos.lon), latY(pos.lat), 4, 0, Math.PI * 2); ctx.fillStyle = sat.color; ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.stroke(); ctx.fillStyle = sat.color; ctx.font = "bold 8px 'IBM Plex Mono', monospace"; ctx.fillText(sat.name, lonX(pos.lon) + 7, latY(pos.lat) - 5); }
    });
  }, [sats, selCtys, countries, anim, showSw, showGr, anchorId]);
  const handleClick = e => {
    const r = ref.current.getBoundingClientRect();
    const mx = (e.clientX - r.left) * (MW / r.width), my = (e.clientY - r.top) * (MH / r.height);
    const cLon = (mx / MW) * 360 - 180, cLat = 90 - (my / MH) * 180;
    for (const ct of countries) { if (cLat >= ct.latMin && cLat <= ct.latMax && cLon >= ct.lonMin && cLon <= ct.lonMax) { onCtClick(ct.id); return; } }
  };
  return <canvas ref={ref} onClick={handleClick} style={{ width: "100%", maxWidth: MW, height: "auto", aspectRatio: `${MW}/${MH}`, borderRadius: 6, cursor: "crosshair", border: "1px solid rgba(70,140,200,0.1)" }} />;
}

/* ══════════════════════════════════════════════════════════════
   TIMELINE CHART
   ══════════════════════════════════════════════════════════════ */
function TimelineChart({ results, timeframe, w = 580, h = 190 }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext("2d"); const dpr = window.devicePixelRatio || 1;
    c.width = w * dpr; c.height = h * dpr; ctx.scale(dpr, dpr);
    const pad = { t: 16, r: 14, b: 24, l: 36 }, cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;
    ctx.fillStyle = "#050a12"; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "rgba(50,110,170,0.1)"; ctx.lineWidth = 0.5;
    for (let p = 0; p <= 100; p += 25) {
      const y = pad.t + ch - (p / 100) * ch;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + cw, y); ctx.stroke();
      ctx.fillStyle = "#2a5a78"; ctx.font = "7px 'IBM Plex Mono', monospace"; ctx.textAlign = "right"; ctx.fillText(p + "%", pad.l - 4, y + 3);
    }
    ctx.strokeStyle = "rgba(100,255,218,0.2)"; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l + cw, pad.t); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#2a5a78"; ctx.font = "7px 'IBM Plex Mono', monospace"; ctx.textAlign = "center";
    const ticks = timeframe <= 7 ? 7 : timeframe <= 30 ? 6 : 8;
    for (let i = 0; i <= ticks; i++) {
      const d = (i / ticks) * timeframe, x = pad.l + (d / timeframe) * cw;
      ctx.fillText(timeframe <= 60 ? `D${Math.round(d)}` : `M${Math.round(d / 30)}`, x, h - 4);
    }
    results.forEach(({ color, data, label }) => {
      if (!data || data.length === 0) return;
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.shadowColor = color; ctx.shadowBlur = 3; ctx.beginPath();
      data.forEach((pt, i) => { const x = pad.l + Math.min(pt.day / timeframe, 1) * cw, y = pad.t + ch - (pt.pct / 100) * ch; i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
      ctx.stroke(); ctx.shadowBlur = 0;
      const last = data[data.length - 1], lx = pad.l + Math.min(last.day / timeframe, 1) * cw, ly = pad.t + ch - (last.pct / 100) * ch;
      ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
      ctx.font = "bold 7px 'IBM Plex Mono', monospace"; ctx.textAlign = "left"; ctx.fillText(`${last.pct}%`, lx + 5, ly + 3);
    });
  }, [results, timeframe, w, h]);
  return <canvas ref={ref} style={{ width: w, height: h, borderRadius: 4 }} />;
}

/* ══════════════════════════════════════════════════════════════
   MAIN APP
   ══════════════════════════════════════════════════════════════ */
export default function App() {
  const [sats, setSats] = useState([{ ...PRESETS[1], color: SAT_COLORS[0] }]); // Default: TRAP
  const [selCtys, setSelCtys] = useState(["cz"]);
  const [anchorCty, setAnchorCty] = useState(null); // anchor country id
  const [anim, setAnim] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [showSw, setShowSw] = useState(true);
  const [showGr, setShowGr] = useState(true);
  const [tab, setTab] = useState("sats");
  const [tf, setTf] = useState(16);
  const [results, setResults] = useState(null);
  const [running, setRunning] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [exporting, setExporting] = useState(false);
  const mainRef = useRef(null);
  const selCtyObjs = COUNTRIES.filter(c => selCtys.includes(c.id));
  const [showAdd, setShowAdd] = useState(false);
  const [commercialRate, setCommercialRate] = useState(15); // $/km²
  const [form, setForm] = useState({
    name: "", altitude: 500, inclination: 97.4, eccentricity: 0.001,
    raan: 0, argPerigee: 0, swathAngle: 12, offNadir: 30, dataCapacity: 500000, lifetime: 5
  });

  useEffect(() => { if (!playing) return; const iv = setInterval(() => setAnim(t => (t + 1) % 400), 80); return () => clearInterval(iv); }, [playing]);

  const toggleCty = useCallback(id => {
    setSelCtys(p => {
      const removing = p.includes(id);
      if (removing) {
        // If removing the anchor country, clear anchor
        setAnchorCty(prev => prev === id ? null : prev);
        return p.filter(x => x !== id);
      }
      return [...p, id];
    });
  }, []);

  // Pre-compute the first pass time of each sat over the anchor country (seconds from t=0).
  // Used to align CountryTrackMap windows so "1 day" means "1 day after first anchor pass".
  const anchorFirstPassTimes = useMemo(() => {
    if (!anchorCty) return {};
    const anchorObj = COUNTRIES.find(c => c.id === anchorCty);
    if (!anchorObj) return {};
    const result = {};
    for (const sat of sats) {
      const effSat = satWithAnchorRaan(sat, anchorObj);
      const { passes } = computePasses(effSat, anchorObj, 5); // 5 days: first pass is within hours, ample window
      // Use the mid-indexed point of the first pass — matches computePasses passMidTimes formula
      result[sat.id] = passes.length > 0 ? passes[0][Math.floor(passes[0].length / 2)].t : 0;
    }
    return result;
  }, [sats, anchorCty]);

  const addSat = () => {
    const s = {
      ...form, id: "sat_" + Date.now(),
      altitude: +form.altitude || 500, inclination: +form.inclination || 97, eccentricity: +form.eccentricity || 0,
      raan: +form.raan || 0, argPerigee: +form.argPerigee || 0, swathAngle: +form.swathAngle || 12,
      offNadir: +form.offNadir || 30, dataCapacity: +form.dataCapacity || 500000, lifetime: +form.lifetime || 5,
      type: "Custom", color: SAT_COLORS[sats.length % SAT_COLORS.length]
    };
    setSats(p => [...p, s]); setShowAdd(false);
    setForm({ name: "", altitude: 500, inclination: 97.4, eccentricity: 0.001, raan: 0, argPerigee: 0, swathAngle: 12, offNadir: 30, dataCapacity: 500000, lifetime: 5 });
  };

  const runAnalysis = () => {
    if (!sats.length || !selCtys.length) return;
    setRunning(true); setResults(null);
    setTimeout(() => {
      const res = [];
      const anchorObj = anchorCty ? COUNTRIES.find(c => c.id === anchorCty) : null;

      sats.forEach(sat => {
        // If anchor is set, override RAAN to optimize first pass over anchor country
        const effectiveSat = anchorObj ? satWithAnchorRaan(sat, anchorObj) : sat;

        // Find time (seconds) of anchor's first pass mid-point — day 0 for all revisit buckets.
        // Use 5 days and mid-indexed point to match anchorFirstPassTimes and computePasses.
        let anchorFirstPassT = 0;
        if (anchorObj) {
          const { passes: anchorPasses } = computePasses(effectiveSat, anchorObj, 5);
          anchorFirstPassT = anchorPasses.length > 0
            ? anchorPasses[0][Math.floor(anchorPasses[0].length / 2)].t : 0;
        }
        const anchorOffsetDays = anchorFirstPassT / 86400;

        selCtys.forEach(cId => {
          const cty = COUNTRIES.find(c => c.id === cId);
          if (!cty) return;
          const { timeline, passLines, finalPct, fullCovDay } = simulateCoverage(effectiveSat, cty, tf);
          const cloudPct = CLOUD_DATA[cty.id] || 50;
          const clearFrac = (100 - cloudPct) / 100;
          const lt = effectiveSat.lifetime || 5;
          const daysPerCov = fullCovDay || tf;
          const covsPerYear = daysPerCov > 0 ? 365.25 / daysPerCov : 0;
          const covsLifetime = covsPerYear * lt;
          const grossImgLifetime = covsLifetime * cty.area;
          const netImgLifetime = grossImgLifetime * clearFrac;
          const grossImgPerYear = covsPerYear * cty.area;
          const netImgPerYear = grossImgPerYear * clearFrac;

          // ── First imaging time for any point in the AOI ──
          const T = orbPeriod(effectiveSat.altitude);
          const sw = effSwathKm(effectiveSat.altitude, effectiveSat.swathAngle || 10);
          const swDegLat = (sw / (2 * Math.PI * R_E)) * 360;
          // Always simulate at least 30 days so sunlitPassCountByTf[30] and
          // uncappedSunlitAreaByTf[30] reflect real 30-day values regardless of
          // the selected analysis timeframe. Without this, choosing tf=7 would make
          // the 30-day columns equal to the 7-day columns.
          const nOrbits30 = Math.ceil((86400 / T) * Math.max(tf, 30));
          const latSpanCty = cty.latMax - cty.latMin;
          const fullLat = 2 * Math.min(effectiveSat.inclination, 180 - effectiveSat.inclination);
          const minPpo = Math.ceil((fullLat / Math.max(latSpanCty, 0.5)) * 4);
          const ppo = Math.max(minPpo, 500);
          const safePpo = nOrbits30 * ppo > 500000 ? Math.max(500, Math.round(500000 / nOrbits30)) : ppo;
          const trackPts = constellationTrack(effectiveSat, nOrbits30, safePpo);

          // ── Count passes per timeframe bucket ──
          const passBuckets = [1, 3, 5, 7, 14, 30];
          const passCountByTf = {};
          const latBuf = swDegLat / 2 + 0.8;
          const lonBuf = 1.5;
          
          // Detect all passes (time-gap based).
          // Collect ALL buffer-zone points per pass, then call passTrackLength() which
          // uses haversine over those points and caps at the country diagonal.
          // Detect all passes over the country buffer zone.
          // midT = middle-indexed point time — matches computePasses passMidTimes formula exactly,
          // so sunlit classification and window filtering are consistent with CountryTrackMap.
          const allPasses = [];
          let curP = null;
          const flushPass = () => {
            const midIdx = Math.floor(curP.points.length / 2);
            const midT = curP.points[midIdx].t;
            allPasses.push({ midT, trackLengthKm: passTrackLength(curP.points, cty) });
            curP = null;
          };
          for (let pi = 0; pi < trackPts.length; pi++) {
            const p = trackPts[pi];
            const overBuf = p.lat >= cty.latMin - latBuf && p.lat <= cty.latMax + latBuf &&
                            p.lon >= cty.lonMin - lonBuf && p.lon <= cty.lonMax + lonBuf;
            if (overBuf) {
              if (!curP) {
                curP = { points: [p] };
              } else {
                const dt = p.t - curP.points[curP.points.length - 1].t;
                if (dt > T * 0.4) { flushPass(); curP = { points: [p] }; }
                else curP.points.push(p);
              }
            } else if (curP) {
              flushPass();
            }
          }
          if (curP) flushPass();

          // All filtering uses midT/86400 (days) — matches computePasses window logic exactly
          const inWindow = (p, start, end) => p.midT / 86400 >= start && p.midT / 86400 <= end;

          for (const bucket of passBuckets) {
            passCountByTf[bucket] = allPasses.filter(p =>
              inWindow(p, anchorOffsetDays, anchorOffsetDays + bucket)
            ).length;
          }

          // Sunlit-only passes — midT matches computePasses passMidTimes formula
          const ctyMidLat = (cty.latMin + cty.latMax) / 2;
          const ctyMidLon = (cty.lonMin + cty.lonMax) / 2;
          const sunlitPasses = allPasses.filter(p => isInImagingWindow(p.midT, ctyMidLat, ctyMidLon));
          const sunlitPassCountByTf = {};
          for (const bucket of passBuckets) {
            sunlitPassCountByTf[bucket] = sunlitPasses.filter(p =>
              inWindow(p, anchorOffsetDays, anchorOffsetDays + bucket)
            ).length;
          }

          // Pass-day-based revisit metrics.
          // firstAllPassDay / firstSunlitPassDay are the absolute times of the very
          // first pass from simulation start — independent of the anchor window so
          // the ALL row always reflects a real first-overpass time (e.g. 48 min for SSO).
          // Mean revisit and max gap are computed over the anchor-aligned 30-day window.
          const firstAllPassDay = allPasses.length > 0 ? allPasses[0].midT / 86400 : null;
          const firstSunlitPassDay = sunlitPasses.length > 0 ? sunlitPasses[0].midT / 86400 : null;

          const allPassDays30 = allPasses
            .filter(p => inWindow(p, anchorOffsetDays, anchorOffsetDays + 30))
            .map(p => p.midT / 86400 - anchorOffsetDays).sort((a, b) => a - b);
          const meanRevisitDays = allPassDays30.length > 1
            ? (allPassDays30[allPassDays30.length - 1] - allPassDays30[0]) / (allPassDays30.length - 1)
            : null;
          const maxGapDays = allPassDays30.length > 1
            ? Math.max(...allPassDays30.slice(1).map((d, i) => d - allPassDays30[i]))
            : null;

          const sunlitPassDays30 = sunlitPasses
            .filter(p => inWindow(p, anchorOffsetDays, anchorOffsetDays + 30))
            .map(p => p.midT / 86400 - anchorOffsetDays).sort((a, b) => a - b);
          const sunlitMeanRevisitDays = sunlitPassDays30.length > 1
            ? (sunlitPassDays30[sunlitPassDays30.length - 1] - sunlitPassDays30[0]) / (sunlitPassDays30.length - 1)
            : null;
          const sunlitMaxGapDays = sunlitPassDays30.length > 1
            ? Math.max(...sunlitPassDays30.slice(1).map((d, i) => d - sunlitPassDays30[i]))
            : null;

          console.log(`[${effectiveSat.name} → ${cty.name}] RAAN: ${effectiveSat.raan.toFixed(1)}°` +
            ` | allPassDays30[0]: ${allPassDays30.length > 0 ? (allPassDays30[0]*24).toFixed(2)+'h' : 'null'}` +
            ` | sunlitPassDays30[0]: ${sunlitPassDays30.length > 0 ? (sunlitPassDays30[0]*24).toFixed(2)+'h' : 'null'}` +
            ` | firstAllPassDay: ${firstAllPassDay !== null ? (firstAllPassDay*24).toFixed(2)+'h' : 'null'}` +
            ` | firstSunlitPassDay: ${firstSunlitPassDay !== null ? (firstSunlitPassDay*24).toFixed(2)+'h' : 'null'}` +
            ` | passes 30d: ${allPassDays30.length} total / ${sunlitPassDays30.length} sunlit`);

          // ── First View & 100% Mapped (analytical, off-nadir accounting) ──
          // Track heading at equator (degrees from North)
          const inclDeg = effectiveSat.inclination;
          const headingRad_fv = (90 - Math.min(inclDeg, 180 - inclDeg)) * DEG;
          // Country dimensions
          const ctyCenterLat = (cty.latMin + cty.latMax) / 2;
          const ctyNS_km = (cty.latMax - cty.latMin) * 111.32;
          const ctyEW_km = (cty.lonMax - cty.lonMin) * 111.32 * Math.cos(ctyCenterLat * DEG);
          // Width of country perpendicular to satellite track
          const perpWidth_km = ctyNS_km * Math.abs(Math.sin(headingRad_fv)) + ctyEW_km * Math.abs(Math.cos(headingRad_fv));
          // Off-nadir corridor width
          const offNadirDeg = effectiveSat.offNadir || 0;
          let corridorKm = sw;
          if (offNadirDeg > 0) {
            const sinArg = (R_E + effectiveSat.altitude) / R_E * Math.sin(offNadirDeg * DEG);
            if (sinArg < 1) {
              const reachKm = R_E * (Math.asin(sinArg) - offNadirDeg * DEG);
              corridorKm = 2 * Math.max(0, reachKm) + sw;
            }
          }
          // First View: day when every point is within corridor of at least one pass
          let firstViewAllDay, firstViewSunlitDay;
          if (corridorKm >= perpWidth_km) {
            // One pass's corridor covers the whole country
            firstViewAllDay = firstAllPassDay;
            firstViewSunlitDay = firstSunlitPassDay;
          } else {
            // Multiple passes needed to scan country via off-nadir steering
            const stripsForView = Math.ceil(perpWidth_km / corridorKm);
            firstViewAllDay = allPassDays30.length > 0 ? 30 * stripsForView / allPassDays30.length : null;
            firstViewSunlitDay = sunlitPassDays30.length > 0 ? 30 * stripsForView / sunlitPassDays30.length : null;
          }
          // 100% Mapped (clear sky): time to image every point using optimal nadir-strip scheduling
          // Uses nadir swath (not corridor) because each imaging strip must be captured
          const stripsNeeded100 = Math.ceil(perpWidth_km / sw);
          const sunlitCnt30 = sunlitPassDays30.length;
          const allCnt30 = allPassDays30.length;
          const hundredPctMappedSunlitDays = sunlitCnt30 > 0 ? 30 * stripsNeeded100 / sunlitCnt30 : null;
          const hundredPctMappedAllDays = allCnt30 > 0 ? 30 * stripsNeeded100 / allCnt30 : null;
          const offNadirNote = offNadirDeg === 0; // flag: no off-nadir steering available

          // ── Cloud-adjusted variants ──
          // cloudFrac already computed above as (cloudPct/100); clearFrac = 1 - cloudFrac
          // First View (cloud): expected day of first CLEAR sunlit pass over any given point.
          // On average, passesNeeded = 1/(1-cloudFrac) attempts to get one clear pass.
          // firstViewCloud = firstSunlitPassDay + (passesNeeded-1) × meanSunlitRevisit
          const cloudFrac = cloudPct / 100;
          const effectiveClearFrac = Math.max(0.01, 1 - cloudFrac); // avoid div/0
          const passesNeededForView = 1 / effectiveClearFrac; // e.g. 2.22 for 55% cloud
          const firstViewSunlitCloud = (firstViewSunlitDay !== null && sunlitMeanRevisitDays !== null)
            ? firstViewSunlitDay + (passesNeededForView - 1) * sunlitMeanRevisitDays
            : null;
          const firstViewAllCloud = (firstViewAllDay !== null && meanRevisitDays !== null)
            ? firstViewAllDay + (passesNeededForView - 1) * meanRevisitDays
            : null;
          // 100% Mapped (cloud): each sunlit pass has clearFrac chance of being usable.
          // Need more passes to collect the same number of productive strips.
          const hundredPctMappedSunlitCloud = hundredPctMappedSunlitDays !== null
            ? hundredPctMappedSunlitDays / effectiveClearFrac
            : null;
          const hundredPctMappedAllCloud = hundredPctMappedAllDays !== null
            ? hundredPctMappedAllDays / effectiveClearFrac
            : null;

          // Covered area per timeframe (sunlit passes × swath × track length, capped at daily capacity)
          const coveredAreaSunlitByTf = {};
          const uncappedSunlitAreaByTf = {};
          for (const bucket of passBuckets) {
            const inBucket = sunlitPasses.filter(p =>
              inWindow(p, anchorOffsetDays, anchorOffsetDays + bucket)
            );
            const uncapped = inBucket.reduce((s, p) => s + sw * p.trackLengthKm, 0);
            uncappedSunlitAreaByTf[bucket] = uncapped;
            const cap = (effectiveSat.dataCapacity || 0) * bucket;
            coveredAreaSunlitByTf[bucket] = cap > 0 ? Math.min(uncapped, cap) : uncapped;
          }
          // Average sunlit pass track length over the 30-day window
          const sunlitIn30 = sunlitPasses.filter(p => inWindow(p, anchorOffsetDays, anchorOffsetDays + 30));
          const avgSunlitPassLengthKm = sunlitIn30.length > 0
            ? sunlitIn30.reduce((s, p) => s + p.trackLengthKm, 0) / sunlitIn30.length
            : 0;

          // Per-day sunlit coverage (uncapped) for computing max-single-day potential.
          const sunlitCoveragePerDay = new Float64Array(30);
          for (const p of sunlitIn30) {
            const di = Math.floor(p.midT / 86400 - anchorOffsetDays);
            if (di >= 0 && di < 30) sunlitCoveragePerDay[di] += sw * p.trackLengthKm;
          }
          // Diagnostic: log TRACE over Rwanda to verify track length after clipping fix
          if (effectiveSat.id === 'trace' && cty.id === 'rw') {
            console.log('[TRACE/Rwanda] sunlit passes in 30d:', sunlitIn30.length,
              '| avg track km:', avgSunlitPassLengthKm.toFixed(1),
              '| swath km:', sw.toFixed(1),
              '| km²/pass:', (avgSunlitPassLengthKm * sw).toFixed(0),
              '| uncapped daily demand km²/d:', ((uncappedSunlitAreaByTf[30] || 0) / 30).toFixed(1),
              '| trackLengths per pass:', sunlitIn30.map(p => p.trackLengthKm.toFixed(0)).join(', '));
          }

          // Grid dimensions (shared by all-pass and sunlit analysis)
          const gN = 15;
          const dLat = cty.latMax - cty.latMin, dLon = cty.lonMax - cty.lonMin;

          // Grid analysis respects the selected timeframe tf (not the full 30-day track)
          const tfEndSec = (anchorOffsetDays + tf) * 86400;
          const trackPtsTf = trackPts.filter(p => p.t <= tfEndSec);

          // Sunlit grid analysis
          const sunlitPts = trackPtsTf.filter(p => isInImagingWindow(p.t, ctyMidLat, ctyMidLon));
          const sfirstVisit = new Float64Array(gN * gN).fill(Infinity);
          const slastVisit = new Float64Array(gN * gN).fill(-1);
          const srevisitSum = new Float64Array(gN * gN);
          const srevisitCount = new Uint16Array(gN * gN);
          for (const p of sunlitPts) {
            const cosL = Math.max(Math.cos(p.lat * DEG), 0.05);
            const halfLon = swDegLat / 2 / cosL;
            const dayT = p.t / 86400;
            if (p.lat + swDegLat / 2 < cty.latMin || p.lat - swDegLat / 2 > cty.latMax) continue;
            if (p.lon + halfLon < cty.lonMin || p.lon - halfLon > cty.lonMax) continue;
            for (let gi = 0; gi < gN; gi++) {
              const gLat = cty.latMin + (gi + 0.5) * dLat / gN;
              if (Math.abs(gLat - p.lat) > swDegLat / 2) continue;
              for (let gj = 0; gj < gN; gj++) {
                const gLon = cty.lonMin + (gj + 0.5) * dLon / gN;
                if (Math.abs(gLon - p.lon) > halfLon) continue;
                const idx = gi * gN + gj;
                if (sfirstVisit[idx] === Infinity) sfirstVisit[idx] = dayT;
                if (slastVisit[idx] >= 0) {
                  const gap = dayT - slastVisit[idx];
                  if (gap > 0.02) { srevisitSum[idx] += gap; srevisitCount[idx]++; }
                }
                slastVisit[idx] = dayT;
              }
            }
          }
          let sMinFirst = Infinity;
          for (let i = 0; i < gN * gN; i++) {
            if (sfirstVisit[i] < sMinFirst) sMinFirst = sfirstVisit[i];
          }
          const sunlitFirstImgBest = sMinFirst === Infinity ? null : sMinFirst - anchorOffsetDays;
          
          // Grid-based first-visit (for 100% coverage day and first-image times)
          const firstVisit = new Float64Array(gN * gN).fill(Infinity);
          const lastVisit = new Float64Array(gN * gN).fill(-1);

          for (const p of trackPtsTf) {
            const cosL = Math.max(Math.cos(p.lat * DEG), 0.05);
            const halfLon = swDegLat / 2 / cosL;
            const dayT = p.t / 86400;
            if (p.lat + swDegLat / 2 < cty.latMin || p.lat - swDegLat / 2 > cty.latMax) continue;
            if (p.lon + halfLon < cty.lonMin || p.lon - halfLon > cty.lonMax) continue;
            for (let gi = 0; gi < gN; gi++) {
              const gLat = cty.latMin + (gi + 0.5) * dLat / gN;
              if (Math.abs(gLat - p.lat) > swDegLat / 2) continue;
              for (let gj = 0; gj < gN; gj++) {
                const gLon = cty.lonMin + (gj + 0.5) * dLon / gN;
                if (Math.abs(gLon - p.lon) > halfLon) continue;
                const idx = gi * gN + gj;
                if (firstVisit[idx] === Infinity) firstVisit[idx] = dayT;
                lastVisit[idx] = dayT;
              }
            }
          }

          let minFirstVisit = Infinity, maxFirstVisit = 0, visitedCells = 0;
          for (let i = 0; i < gN * gN; i++) {
            if (firstVisit[i] < Infinity) {
              if (firstVisit[i] < minFirstVisit) minFirstVisit = firstVisit[i];
              if (firstVisit[i] > maxFirstVisit) maxFirstVisit = firstVisit[i];
              visitedCells++;
            }
          }
          const firstImgBest = minFirstVisit === Infinity ? null : minFirstVisit - anchorOffsetDays;
          const firstImgWorst = maxFirstVisit === 0 ? null : maxFirstVisit - anchorOffsetDays;

          res.push({
            satId: sat.id, satName: sat.name, satColor: sat.color, countryId: cty.id, countryName: cty.name,
            area: cty.area, timeline, passLines, finalPct, fullCovDay,
            cloudPct, clearFrac, lifetime: lt,
            covsPerYear: finalPct >= 100 ? covsPerYear : 0,
            covsLifetime: finalPct >= 100 ? covsLifetime : 0,
            grossImgLifetime, netImgLifetime,
            grossImgPerYear, netImgPerYear,
            label: `${sat.name}${(effectiveSat.count || 1) > 1 ? " ×" + (effectiveSat.count || 1) : ""} → ${cty.name}`,
            satCount: effectiveSat.count || 1,
            isAnchor: cty.id === anchorCty,
            effectiveRaan: effectiveSat.raan,
            originalRaan: sat.raan,
            firstImgBest, firstImgWorst,
            firstAllPassDay, meanRevisitDays, maxGapDays,
            firstSunlitPassDay, sunlitMeanRevisitDays, sunlitMaxGapDays,
            firstViewAllDay, firstViewSunlitDay,
            firstViewAllCloud, firstViewSunlitCloud,
            hundredPctMappedSunlitDays, hundredPctMappedAllDays,
            hundredPctMappedSunlitCloud, hundredPctMappedAllCloud,
            offNadirNote,
            passCountByTf,
            sunlitPassCountByTf,
            coveredAreaSunlitByTf,
            uncappedSunlitAreaByTf,
            avgSunlitPassLengthKm,
            swKm: sw,
            satDataCapacity: effectiveSat.dataCapacity || 0,
            satLifetime: effectiveSat.lifetime || 5,
            sunlitCoveragePerDay: Array.from(sunlitCoveragePerDay),
          });
        });
      });
      setResults(res); setRunning(false); setTab("results");
    }, 60);
  };

  const exportPDF = useCallback(async () => {
    setExporting(true); setExportOpen(false);
    try {
      const pdf = new jsPDF({ orientation: "landscape", unit: "mm", format: "a4" });
      const pageW = pdf.internal.pageSize.getWidth();
      const pageH = pdf.internal.pageSize.getHeight();
      const margin = 10;
      const contentW = pageW - margin * 2;

      // Title page
      pdf.setFillColor(2, 6, 16);
      pdf.rect(0, 0, pageW, pageH, "F");
      pdf.setTextColor(100, 255, 218);
      pdf.setFontSize(18);
      pdf.text("ORBITAL COVERAGE ANALYSIS", pageW / 2, 40, { align: "center" });
      pdf.setTextColor(160, 208, 232);
      pdf.setFontSize(10);
      pdf.text(`Generated: ${new Date().toISOString().slice(0, 10)}`, pageW / 2, 52, { align: "center" });
      pdf.setTextColor(42, 90, 120);
      pdf.setFontSize(8);
      pdf.text(`Satellites: ${sats.map(s => s.name).join(", ")}`, pageW / 2, 62, { align: "center" });

      // Capture each named section
      const sections = mainRef.current ? Array.from(mainRef.current.querySelectorAll("[data-section]")) : [];
      for (const section of sections) {
        const canvas = await html2canvas(section, { backgroundColor: "#020610", scale: 1.5, useCORS: true, logging: false });
        const imgData = canvas.toDataURL("image/jpeg", 0.85);
        const ratio = canvas.height / canvas.width;
        const imgW = contentW;
        const imgH = Math.min(imgW * ratio, pageH - margin * 2);
        pdf.addPage();
        pdf.setFillColor(2, 6, 16);
        pdf.rect(0, 0, pageW, pageH, "F");
        pdf.addImage(imgData, "JPEG", margin, margin, imgW, imgH);
      }
      pdf.save("orbital-coverage-analysis.pdf");
    } catch (e) { console.error("PDF export error:", e); }
    setExporting(false);
  }, [sats, mainRef]);

  const exportWord = useCallback(async () => {
    setExporting(true); setExportOpen(false);
    try {
      const fmtD = (d) => {
        if (d === null || d === undefined) return "—";
        if (d < 1/24) return `${(d * 24 * 60).toFixed(0)} min`;
        if (d < 1) return `${(d * 24).toFixed(1)} hrs`;
        return `${d.toFixed(1)} days`;
      };
      const fmtN = (n) => {
        if (n >= 1e9) return (n/1e9).toFixed(1)+"B";
        if (n >= 1e6) return (n/1e6).toFixed(1)+"M";
        if (n >= 1e3) return (n/1e3).toFixed(1)+"k";
        return n.toFixed(0);
      };
      const fmtM = (n) => {
        if (n >= 1e9) return "$"+(n/1e9).toFixed(2)+"B";
        if (n >= 1e6) return "$"+(n/1e6).toFixed(2)+"M";
        if (n >= 1e3) return "$"+(n/1e3).toFixed(1)+"k";
        return "$"+n.toFixed(0);
      };

      const mkCell = (text, bold = false, center = false) =>
        new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: String(text), bold, size: 16 })], alignment: center ? AlignmentType.CENTER : AlignmentType.LEFT })] });
      const mkHdr = (cols) => new TableRow({ children: cols.map(h => mkCell(h, true, true)) });

      // ── REVISIT ANALYSIS TABLE ──
      const revisitHdr = mkHdr(["Satellite → Country", "Mode", "Mean Revisit", "Max Gap", "First Sunlit", "100% Mapped", "1d", "3d", "5d", "7d", "14d", "30d"]);
      const revisitRows = (results || []).flatMap(r => {
        const pc = r.passCountByTf || {}, spc = r.sunlitPassCountByTf || {};
        const fmtDw = (d) => {
          if (d === null || d === undefined) return "—";
          if (d < 1/24) return `${(d * 24 * 60).toFixed(0)} min`;
          if (d < 1) return `${(d * 24).toFixed(1)} hrs`;
          if (d > 365) return ">1yr";
          return `${d.toFixed(1)} days`;
        };
        return [
          new TableRow({ children: [
            mkCell(`${r.satName} → ${r.countryName}`, true),
            mkCell("ALL"),
            mkCell(fmtD(r.meanRevisitDays), false, true),
            mkCell(fmtD(r.maxGapDays), false, true),
            mkCell("—", false, true),
            mkCell("—", false, true),
            ...[1,3,5,7,14,30].map(b => mkCell(String(pc[b]||0), false, true)),
          ]}),
          new TableRow({ children: [
            mkCell(""),
            mkCell("Sunlit OPT"),
            mkCell(fmtD(r.sunlitMeanRevisitDays), false, true),
            mkCell(fmtD(r.sunlitMaxGapDays), false, true),
            mkCell(fmtDw(r.firstSunlitPassDay), false, true),
            mkCell(fmtDw(r.hundredPctMappedSunlitDays) + (r.hundredPctMappedSunlitCloud ? ` / ${fmtDw(r.hundredPctMappedSunlitCloud)} ☁` : "") + (r.offNadirNote ? " (nadir)" : ""), false, true),
            ...[1,3,5,7,14,30].map(b => mkCell(String(spc[b]||0), false, true)),
          ]}),
        ];
      });

      // ── UTILIZATION TABLE (satellite summary + per-country breakdown) ──
      const utilHdr = mkHdr(["Satellite / Country", "Avg Daily Demand", "Max Daily Potential", "Daily Capacity", "Utilization", "Headroom", "Actual km²/yr", "Lifetime km²", "Passes/30d", "km²/pass", "Demand Share"]);
      const utilRows = sats.flatMap(sat => {
        const satResults = (results || []).filter(r => r.satId === sat.id);
        if (!satResults.length) return [];
        const dailyCap = satResults[0].satDataCapacity || 0;
        const totalUncappedDaily = satResults.reduce((s, r) => s + ((r.uncappedSunlitAreaByTf[30]||0)/30), 0);
        const effectiveDaily = dailyCap > 0 ? Math.min(totalUncappedDaily, dailyCap) : totalUncappedDaily;
        const utilization = dailyCap > 0 ? (effectiveDaily / dailyCap * 100) : null;
        const headroom = dailyCap > 0 ? dailyCap - totalUncappedDaily : null;
        const actualPerYear = effectiveDaily * 365.25;
        const satLifetime = satResults[0].satLifetime || 5;
        const actualLifetime = actualPerYear * satLifetime;
        // Max daily potential: peak single-day total across all countries
        const dayTotals = new Float64Array(30);
        for (const r of satResults) {
          const cpd = r.sunlitCoveragePerDay || [];
          for (let d = 0; d < 30; d++) dayTotals[d] += cpd[d] || 0;
        }
        const maxDailyPotential = Math.max(...dayTotals);
        const isCapLimited = dailyCap > 0 && totalUncappedDaily > dailyCap;
        // Satellite summary row
        const satRow = new TableRow({ children: [
          mkCell(`${sat.name}${(satResults[0].satCount||1)>1 ? ` ×${satResults[0].satCount}` : ""} [${isCapLimited ? "CAP LIMITED" : "OPP LIMITED"}]`, true),
          mkCell(`${fmtN(totalUncappedDaily)} km²/d`, true, true),
          mkCell(maxDailyPotential > 0 ? `${fmtN(maxDailyPotential)} km²/d` : "—", true, true),
          mkCell(dailyCap > 0 ? `${fmtN(dailyCap)} km²/d` : "—", true, true),
          mkCell(utilization !== null ? `${utilization.toFixed(0)}%` : "—", true, true),
          mkCell(headroom !== null ? `${headroom >= 0 ? "+" : "−"}${fmtN(Math.abs(headroom))} km²/d` : "—", true, true),
          mkCell(`${fmtN(actualPerYear)} km²`, true, true),
          mkCell(`${fmtN(actualLifetime)} km² (${satLifetime}yr)`, true, true),
          mkCell("", false, true),
          mkCell("", false, true),
          mkCell("", false, true),
        ]});
        // Per-country detail rows
        const countryRows = satResults.map(r => {
          const grossPerPass = r.avgSunlitPassLengthKm * r.swKm;
          const dailyDemand = (r.uncappedSunlitAreaByTf[30]||0)/30;
          const share = totalUncappedDaily > 0 ? (dailyDemand/totalUncappedDaily*100) : 0;
          return new TableRow({ children: [
            mkCell(`  → ${r.countryName}`),
            mkCell(`${fmtN(dailyDemand)} km²/d`, false, true),
            mkCell("—", false, true),
            mkCell("—", false, true),
            mkCell("—", false, true),
            mkCell("—", false, true),
            mkCell("—", false, true),
            mkCell("—", false, true),
            mkCell(String(r.sunlitPassCountByTf[30]||0), false, true),
            mkCell(`${fmtN(grossPerPass)} km²`, false, true),
            mkCell(`${share.toFixed(1)}%`, false, true),
          ]});
        });
        return [satRow, ...countryRows];
      });

      // ── COMMERCIAL VALUE TABLE ──
      const commHdr = mkHdr(["Satellite", "Country", "Passes/30d", "Gross km²/yr", "Cloud%", "Net km²/yr", "Value/yr (net)", "Value/lifetime (net)"]);
      const commRows = sats.flatMap(sat => {
        const satResults = (results || []).filter(r => r.satId === sat.id);
        return satResults.map(r => {
          const passesPerYear = (r.sunlitPassCountByTf[30]||0) * (365.25/30);
          const grossAnnualKm2 = passesPerYear * r.avgSunlitPassLengthKm * r.swKm;
          const cldFrac = (r.cloudPct || 50) / 100;
          const netAnnualKm2 = grossAnnualKm2 * (1 - cldFrac);
          const annualValue = netAnnualKm2 * commercialRate;
          const lifetimeValue = annualValue * (r.satLifetime||5);
          return new TableRow({ children: [
            mkCell(sat.name, true),
            mkCell(r.countryName),
            mkCell(String(r.sunlitPassCountByTf[30]||0), false, true),
            mkCell(`${fmtN(grossAnnualKm2)} km²`, false, true),
            mkCell(`${r.cloudPct}%`, false, true),
            mkCell(`${fmtN(netAnnualKm2)} km²`, false, true),
            mkCell(fmtM(annualValue), false, true),
            mkCell(fmtM(lifetimeValue), false, true),
          ]});
        });
      });

      // ── GROUND TRACK MAP IMAGES ──
      // 2-column grid: All passes | Sunlit — one row per timeframe (6 rows × 2 cols = 12 per sat×country)
      // Each map: 600×480 canvas, displayed at 440×352 in Word (fits 2 per A4-landscape row with margins)
      const IMG_W = 440, IMG_H = 352;
      const anchorObj = anchorCty ? COUNTRIES.find(c => c.id === anchorCty) : null;
      const mapChildren = [];

      const mkImgCell = (imgData, caption) => new TableCell({
        width: { size: 50, type: WidthType.PERCENTAGE },
        children: [
          new Paragraph({
            children: imgData ? [new ImageRun({ data: imgData, transformation: { width: IMG_W, height: IMG_H }, type: "jpg" })] : [new TextRun({ text: "(no data)", size: 16 })],
            spacing: { after: 60 },
          }),
          new Paragraph({ children: [new TextRun({ text: caption, size: 14, color: "555555" })], spacing: { after: 160 } }),
        ],
      });

      for (const sat of sats) {
        const effSat = anchorObj ? satWithAnchorRaan(sat, anchorObj) : sat;
        const anchorOffset = anchorFirstPassTimes[sat.id] || 0;
        for (const cty of selCtyObjs) {
          mapChildren.push(new Paragraph({ text: `${sat.name} → ${cty.name}  ·  ${sat.altitude}km / ${sat.inclination}°`, heading: HeadingLevel.HEADING_3 }));
          // Column headers
          mapChildren.push(new Table({
            width: { size: 100, type: WidthType.PERCENTAGE },
            rows: [
              new TableRow({ children: [
                new TableCell({ width: { size: 50, type: WidthType.PERCENTAGE }, children: [new Paragraph({ children: [new TextRun({ text: "ALL PASSES", bold: true, size: 18 })] })] }),
                new TableCell({ width: { size: 50, type: WidthType.PERCENTAGE }, children: [new Paragraph({ children: [new TextRun({ text: "☀ SUNLIT ONLY", bold: true, size: 18 })] })] }),
              ]}),
              // One row per timeframe
              ...await Promise.all(TRACK_TF_OPTS.map(async ({ l, d }) => {
                const [allUrl, sunUrl] = await Promise.all([
                  renderTrackMapToDataURL(effSat, cty, anchorOffset, d, false, 600, 480),
                  renderTrackMapToDataURL(effSat, cty, anchorOffset, d, true,  600, 480),
                ]);
                const allData = allUrl ? dataUrlToUint8Array(allUrl) : null;
                const sunData = sunUrl ? dataUrlToUint8Array(sunUrl) : null;
                // Count passes for caption
                const { passes: allP, passMidTimes: allMT } = computePasses(effSat, cty, d, anchorOffset);
                const nAll = allP.length;
                const nSun = allP.filter((_, i) => isInImagingWindow(allMT[i], (cty.latMin+cty.latMax)/2, (cty.lonMin+cty.lonMax)/2)).length;
                return new TableRow({ children: [
                  mkImgCell(allData, `${cty.name} · ${sat.name} · ${d}d · All passes · ${nAll} passes`),
                  mkImgCell(sunData, `${cty.name} · ${sat.name} · ${d}d · Sunlit · ${nSun} passes`),
                ]});
              })),
            ],
          }));
          mapChildren.push(new Paragraph({ text: "", spacing: { after: 400 } }));
        }
      }

      const doc = new Document({
        sections: [{
          properties: { page: { size: { orientation: "landscape", width: 15840, height: 12240 } } },
          children: [
            new Paragraph({ text: "Orbital Coverage Analysis", heading: HeadingLevel.HEADING_1 }),
            new Paragraph({ children: [new TextRun({ text: `Generated: ${new Date().toISOString().slice(0, 10)}  |  Satellites: ${sats.map(s => s.name).join(", ")}  |  Rate: $${commercialRate}/km²`, size: 18 })], spacing: { after: 400 } }),

            new Paragraph({ text: "Revisit Analysis", heading: HeadingLevel.HEADING_2 }),
            new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, rows: [revisitHdr, ...revisitRows] }),
            new Paragraph({ text: "", spacing: { after: 400 } }),

            new Paragraph({ text: "Satellite Utilization", heading: HeadingLevel.HEADING_2 }),
            new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, rows: [utilHdr, ...utilRows] }),
            new Paragraph({ text: "", spacing: { after: 400 } }),

            new Paragraph({ text: "Commercial Imagery Value", heading: HeadingLevel.HEADING_2 }),
            new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, rows: [commHdr, ...commRows] }),
            new Paragraph({ text: "", spacing: { after: 400 } }),

            new Paragraph({ text: "Ground Track Maps", heading: HeadingLevel.HEADING_2 }),
            ...mapChildren,
          ]
        }]
      });

      const blob = await Packer.toBlob(doc);
      saveAs(blob, "orbital-coverage-analysis.docx");
    } catch (e) { console.error("Word export error:", e); }
    setExporting(false);
  }, [results, sats, tf, commercialRate, anchorCty, anchorFirstPassTimes, selCtyObjs]);

  const tfOpts = [{ l: "1 Day", v: 1 }, { l: "3 Days", v: 3 }, { l: "7 Days", v: 7 }, { l: "16 Days", v: 16 }, { l: "30 Days", v: 30 }, { l: "90 Days", v: 90 }, { l: "1 Year", v: 365 }];

  const fmt = (n) => {
    if (n >= 1e9) return (n / 1e9).toFixed(1) + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
    return n.toFixed(0);
  };
  const fmtMoney = (n) => {
    if (n >= 1e9) return "$" + (n / 1e9).toFixed(2) + "B";
    if (n >= 1e6) return "$" + (n / 1e6).toFixed(2) + "M";
    if (n >= 1e3) return "$" + (n / 1e3).toFixed(1) + "k";
    return "$" + n.toFixed(0);
  };

  return (
    <div style={{ minHeight: "100vh", background: "linear-gradient(170deg, #020610 0%, #061018 50%, #081428 100%)", color: "#a0c4d8", fontFamily: "'IBM Plex Mono','JetBrains Mono','Fira Code',monospace", fontSize: 11 }}>
      <style>{`*{box-sizing:border-box;}::-webkit-scrollbar{width:5px;}::-webkit-scrollbar-thumb{background:rgba(70,140,200,0.2);border-radius:4px;}::-webkit-scrollbar-track{background:transparent;}`}</style>

      {/* ── HEADER ── */}
      <div style={{ padding: "12px 22px", borderBottom: "1px solid rgba(70,140,200,0.1)", display: "flex", alignItems: "center", justifyContent: "space-between", background: "rgba(2,6,16,0.92)", backdropFilter: "blur(12px)", position: "sticky", top: 0, zIndex: 100 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 11 }}>
          <div style={{ width: 30, height: 30, borderRadius: "50%", background: "conic-gradient(from 0deg, #083058, #1a6aff, #083058)", display: "flex", alignItems: "center", justifyContent: "center", border: "1.5px solid rgba(70,140,200,0.3)", fontSize: 13 }}>◉</div>
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#d8ecff", letterSpacing: 3 }}>ORBITAL COVERAGE ANALYST</div>
            <div style={{ fontSize: 8, color: "#2a5068", letterSpacing: 3 }}>TRL SPACE · MAPPING · COVERAGE · ECONOMICS</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {/* Commercial rate */}
          <div style={{ display: "flex", alignItems: "center", gap: 5, background: "rgba(255,215,0,0.06)", border: "1px solid rgba(255,215,0,0.2)", borderRadius: 4, padding: "3px 8px" }}>
            <span style={{ fontSize: 8, color: "#b8a040", letterSpacing: 1 }}>$/km²</span>
            <input type="number" value={commercialRate} onChange={e => setCommercialRate(+e.target.value || 0)}
              style={{ width: 50, background: "transparent", border: "none", color: "#FFD740", fontSize: 11, fontFamily: "inherit", fontWeight: 700, textAlign: "center", outline: "none" }} />
          </div>
          <span style={{ fontSize: 9, color: "#1a3a50" }}>T+{String(anim).padStart(4, "0")}</span>
          <Btn sm on={playing} onClick={() => setPlaying(!playing)}>{playing ? "▌▌" : "▶"}</Btn>
          <Btn sm onClick={() => setAnim(0)}>↺</Btn>
          <Btn sm on style={{ background: "rgba(100,255,218,0.1)", borderColor: "rgba(100,255,218,0.3)", color: "#64FFDA" }} onClick={runAnalysis}>
            {running ? "⟳ RUNNING..." : "▶ RUN ANALYSIS"}
          </Btn>
          {results && (
            <div style={{ position: "relative" }}>
              <Btn sm on={exportOpen} onClick={() => setExportOpen(o => !o)} style={{ background: "rgba(255,215,0,0.08)", borderColor: "rgba(255,215,0,0.25)", color: "#FFD740" }}>
                {exporting ? "⟳ EXPORTING..." : "⬇ EXPORT ▾"}
              </Btn>
              {exportOpen && (
                <div style={{ position: "absolute", top: "100%", right: 0, marginTop: 4, background: "#0a1628", border: "1px solid rgba(255,215,0,0.2)", borderRadius: 4, zIndex: 100, minWidth: 130 }}>
                  <button onClick={exportPDF} style={{ display: "block", width: "100%", padding: "8px 12px", background: "transparent", border: "none", color: "#FFD740", fontSize: 10, fontFamily: "inherit", textAlign: "left", cursor: "pointer", letterSpacing: 0.5 }}>
                    ⎙ Export PDF
                  </button>
                  <button onClick={exportWord} style={{ display: "block", width: "100%", padding: "8px 12px", background: "transparent", border: "none", color: "#FFD740", fontSize: 10, fontFamily: "inherit", textAlign: "left", cursor: "pointer", letterSpacing: 0.5, borderTop: "1px solid rgba(255,215,0,0.1)" }}>
                    ⊞ Export Word
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <div style={{ display: "flex", height: "calc(100vh - 58px)" }}>
        {/* ── LEFT PANEL ── */}
        <div style={{ width: 290, minWidth: 290, borderRight: "1px solid rgba(70,140,200,0.08)", display: "flex", flexDirection: "column", background: "rgba(2,6,16,0.5)" }}>
          <div style={{ display: "flex", borderBottom: "1px solid rgba(70,140,200,0.08)" }}>
            {[["sats", "SATELLITES"], ["ctys", "COUNTRIES"], ["results", "RESULTS"]].map(([k, l]) => (
              <button key={k} onClick={() => setTab(k)} style={{
                flex: 1, padding: "9px 0", background: tab === k ? "rgba(70,140,200,0.08)" : "transparent",
                border: "none", borderBottom: tab === k ? "2px solid rgba(70,140,200,0.5)" : "2px solid transparent",
                color: tab === k ? "#90c8e8" : "#2a5068", fontSize: 9, fontFamily: "inherit", cursor: "pointer", letterSpacing: 1.5, fontWeight: 600
              }}>{l}</button>
            ))}
          </div>

          <div style={{ flex: 1, overflowY: "auto", padding: 12 }}>
            {/* ── SATELLITES TAB ── */}
            {tab === "sats" && <>
              <Lbl>CONSTELLATION ({sats.reduce((s,sat) => s + (sat.count || 1), 0)} satellites across {sats.length} orbital plane{sats.length !== 1 ? "s" : ""})</Lbl>
              {sats.map((sat, si) => {
                const cnt = sat.count || 1;
                return (
                <P key={sat.id} style={{ padding: "9px 11px", marginBottom: 5, borderLeft: `3px solid ${sat.color}` }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                      <span style={{ fontWeight: 700, color: "#d8ecff", fontSize: 11 }}>{sat.name}</span>
                      {cnt > 1 && <span style={{ fontSize: 9, color: "#FFD740", fontWeight: 700, background: "rgba(255,215,0,0.12)", padding: "1px 5px", borderRadius: 3 }}>×{cnt}</span>}
                    </div>
                    <button onClick={() => setSats(p => p.filter(s => s.id !== sat.id))} style={{ background: "none", border: "none", color: "#4a2a2a", cursor: "pointer", fontSize: 13, fontFamily: "inherit" }}>×</button>
                  </div>
                  {sat.desc && <div style={{ fontSize: 8, color: "#4a6878", marginBottom: 4, lineHeight: 1.3 }}>{sat.desc}</div>}

                  {/* Orbit type & inclination - editable inline */}
                  <div style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 5, flexWrap: "wrap" }}>
                    <span style={{ fontSize: 8, color: "#3a6080", letterSpacing: 1 }}>ORBIT:</span>
                    {[
                      { l: "SSO", inc: Math.round(Math.acos(-Math.pow(sat.altitude + R_E, 3.5) / (R_E * R_E * 1.5 * J2 * Math.sqrt(MU)) * (2 * Math.PI / (365.25 * 86400))) * RAD * 10) / 10 || 97.4 },
                      { l: "45°", inc: 45 },
                      { l: "Near-Eq", inc: 5 },
                    ].map(o => (
                      <button key={o.l} onClick={() => setSats(p => p.map(s => s.id === sat.id ? { ...s, inclination: o.inc } : s))}
                        style={{
                          padding: "2px 6px", borderRadius: 3, cursor: "pointer", fontSize: 8, fontFamily: "inherit",
                          background: Math.abs(sat.inclination - o.inc) < 2 ? "rgba(100,255,218,0.15)" : "rgba(70,140,200,0.06)",
                          border: `1px solid ${Math.abs(sat.inclination - o.inc) < 2 ? "rgba(100,255,218,0.4)" : "rgba(70,140,200,0.12)"}`,
                          color: Math.abs(sat.inclination - o.inc) < 2 ? "#64FFDA" : "#4a7090",
                          fontWeight: Math.abs(sat.inclination - o.inc) < 2 ? 700 : 400
                        }}>{o.l}</button>
                    ))}
                    <input type="number" value={sat.inclination} step="0.1"
                      onChange={e => setSats(p => p.map(s => s.id === sat.id ? { ...s, inclination: parseFloat(e.target.value) || 0 } : s))}
                      style={{ width: 48, background: "rgba(4,8,16,0.9)", border: "1px solid rgba(70,140,200,0.2)", color: "#a0d0e8", padding: "2px 4px", borderRadius: 3, fontSize: 9, fontFamily: "inherit", textAlign: "center" }} />
                    <span style={{ fontSize: 8, color: "#2a4a60" }}>°</span>
                  </div>

                  {/* Co-planar count */}
                  <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 5 }}>
                    <span style={{ fontSize: 8, color: "#3a6080", letterSpacing: 1 }}>CO-PLANAR:</span>
                    <button onClick={() => setSats(p => p.map(s => s.id === sat.id ? { ...s, count: Math.max(1, (s.count || 1) - 1) } : s))}
                      style={{ width: 18, height: 18, background: "rgba(70,140,200,0.1)", border: "1px solid rgba(70,140,200,0.2)", color: "#80b8d8", borderRadius: 3, cursor: "pointer", fontSize: 11, fontFamily: "inherit", display: "flex", alignItems: "center", justifyContent: "center", lineHeight: 1 }}>−</button>
                    <span style={{ fontSize: 12, fontWeight: 700, color: "#FFD740", minWidth: 16, textAlign: "center" }}>{cnt}</span>
                    <button onClick={() => setSats(p => p.map(s => s.id === sat.id ? { ...s, count: Math.min(12, (s.count || 1) + 1) } : s))}
                      style={{ width: 18, height: 18, background: "rgba(70,140,200,0.1)", border: "1px solid rgba(70,140,200,0.2)", color: "#80b8d8", borderRadius: 3, cursor: "pointer", fontSize: 11, fontFamily: "inherit", display: "flex", alignItems: "center", justifyContent: "center", lineHeight: 1 }}>+</button>
                    <span style={{ fontSize: 8, color: "#2a4a60" }}>in this plane</span>
                  </div>

                  {/* Daily imaging capacity */}
                  <div style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 5 }}>
                    <span style={{ fontSize: 8, color: "#3a6080", letterSpacing: 1 }}>CAP:</span>
                    <input type="number" min="0" value={sat.dataCapacity}
                      onChange={e => setSats(p => p.map(s => s.id === sat.id ? { ...s, dataCapacity: +e.target.value || 0 } : s))}
                      style={{ width: 72, background: "rgba(4,8,16,0.9)", border: "1px solid rgba(70,140,200,0.2)", color: "#a0d0e8", padding: "2px 4px", borderRadius: 3, fontSize: 9, fontFamily: "inherit", textAlign: "center" }} />
                    <span style={{ fontSize: 8, color: "#2a4a60" }}>km²/d</span>
                  </div>

                  {/* Add variant on different orbit */}
                  <button onClick={() => {
                    const newInc = sat.inclination > 50 ? 45 : 97.4;
                    setSats(p => [...p, {
                      ...sat,
                      id: sat.id.split("__")[0] + "__" + Date.now(),
                      inclination: newInc,
                      count: 1,
                      color: SAT_COLORS[(si + p.length) % SAT_COLORS.length]
                    }]);
                  }} style={{
                    width: "100%", padding: "4px", marginBottom: 5, borderRadius: 3, cursor: "pointer",
                    background: "rgba(224,64,251,0.08)", border: "1px solid rgba(224,64,251,0.2)", color: "#E040FB",
                    fontSize: 8, fontFamily: "inherit", letterSpacing: 0.5
                  }}>+ ADD ON DIFFERENT ORBIT ({sat.inclination > 50 ? "45°" : "SSO"})</button>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2px 8px", fontSize: 9 }}>
                    {sat.gsd && <div><span style={{ color: "#2a5068" }}>GSD:</span> <span style={{ color: "#FFD740", fontWeight: 600 }}>{sat.gsd}</span></div>}
                    {sat.mass && <div><span style={{ color: "#2a5068" }}>Mass:</span> <span style={{ color: "#80b8d8" }}>{sat.mass}</span></div>}
                    <div><span style={{ color: "#2a5068" }}>Alt:</span> <span style={{ color: "#80b8d8" }}>{sat.altitude}km</span></div>
                    <div><span style={{ color: "#2a5068" }}>Swath:</span> <span style={{ color: "#80b8d8" }}>{effSwathKm(sat.altitude, sat.swathAngle).toFixed(0)}km</span></div>
                    <div><span style={{ color: "#2a5068" }}>Off-nadir:</span> <span style={{ color: "#80b8d8" }}>{sat.offNadir}°</span></div>
                    <div><span style={{ color: "#2a5068" }}>Capacity:</span> <span style={{ color: "#80b8d8" }}>{fmt(sat.dataCapacity * cnt)} km²/d</span></div>
                  </div>
                </P>
              )})}

              <Lbl>TRL SPACE SATELLITES</Lbl>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 3, marginBottom: 6 }}>
                {PRESETS.filter(p => ["troll_ng","trap","trace"].includes(p.id)).map(p => (
                  <Btn key={p.id} sm on={!!sats.find(s => s.id === p.id)} onClick={() => {
                    if (!sats.find(s => s.id === p.id)) setSats(prev => [...prev, { ...p, color: SAT_COLORS[prev.length % SAT_COLORS.length] }]);
                  }} style={sats.find(s => s.id === p.id) ? {} : { borderColor: "rgba(220,200,0,0.25)", color: "#c8b830" }}>{p.name}</Btn>
                ))}
              </div>
              <Lbl>REFERENCE SATELLITES</Lbl>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 3, marginBottom: 12 }}>
                {PRESETS.filter(p => !["troll_ng","trap","trace"].includes(p.id)).map(p => (
                  <Btn key={p.id} sm on={!!sats.find(s => s.id === p.id)} onClick={() => {
                    if (!sats.find(s => s.id === p.id)) setSats(prev => [...prev, { ...p, color: SAT_COLORS[prev.length % SAT_COLORS.length] }]);
                  }}>{p.name}</Btn>
                ))}
              </div>

              <Btn onClick={() => setShowAdd(!showAdd)} style={{ width: "100%", marginBottom: 8, background: "rgba(100,255,218,0.07)", borderColor: "rgba(100,255,218,0.22)", color: "#64FFDA" }}>
                {showAdd ? "− CANCEL" : "+ CUSTOM SATELLITE"}
              </Btn>

              {showAdd && <P style={{ padding: 11 }}>
                <Lbl>DEFINE SATELLITE</Lbl>
                <Inp label="NAME" value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} placeholder="My EO Sat" />
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 7px" }}>
                  <Inp label="ALTITUDE (km)" type="number" value={form.altitude} onChange={e => setForm({ ...form, altitude: e.target.value })} />
                  <Inp label="INCLINATION (°)" type="number" value={form.inclination} onChange={e => setForm({ ...form, inclination: e.target.value })} />
                  <Inp label="ECCENTRICITY" type="number" step="0.001" value={form.eccentricity} onChange={e => setForm({ ...form, eccentricity: e.target.value })} />
                  <Inp label="RAAN (°)" type="number" value={form.raan} onChange={e => setForm({ ...form, raan: e.target.value })} />
                </div>
                <div style={{ height: 1, background: "rgba(70,140,200,0.1)", margin: "8px 0" }} />
                <Lbl>SENSOR</Lbl>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 7px" }}>
                  <Inp label="SWATH HALF-ANGLE (°)" type="number" value={form.swathAngle} onChange={e => setForm({ ...form, swathAngle: e.target.value })} />
                  <Inp label="MAX OFF-NADIR (°)" type="number" value={form.offNadir} onChange={e => setForm({ ...form, offNadir: e.target.value })} />
                </div>
                <Inp label="CAPACITY (km²/day)" type="number" value={form.dataCapacity} onChange={e => setForm({ ...form, dataCapacity: e.target.value })} />
                <div style={{ height: 1, background: "rgba(70,140,200,0.1)", margin: "8px 0" }} />
                <Lbl>MISSION</Lbl>
                <Inp label="DESIGN LIFETIME (years)" type="number" step="0.5" value={form.lifetime} onChange={e => setForm({ ...form, lifetime: e.target.value })} />
                <Btn onClick={addSat} style={{ width: "100%", marginTop: 6, background: "rgba(100,255,218,0.12)", borderColor: "rgba(100,255,218,0.35)", color: "#64FFDA" }}>CREATE</Btn>
              </P>}
            </>}

            {/* ── COUNTRIES TAB ── */}
            {tab === "ctys" && <>
              {/* Anchor country selector */}
              <div style={{ marginBottom: 12, padding: "8px 10px", background: anchorCty ? "rgba(255,170,0,0.06)" : "rgba(70,140,200,0.04)", border: `1px solid ${anchorCty ? "rgba(255,170,0,0.2)" : "rgba(70,140,200,0.1)"}`, borderRadius: 5 }}>
                <div style={{ fontSize: 8, color: "#b8a040", letterSpacing: 2, fontWeight: 700, marginBottom: 5 }}>⚓ ANCHOR CUSTOMER</div>
                {anchorCty ? (() => {
                  const ac = COUNTRIES.find(c => c.id === anchorCty);
                  return ac ? (
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <div>
                        <div style={{ fontWeight: 700, color: "#FFB300", fontSize: 11 }}>{ac.name}</div>
                        <div style={{ fontSize: 8, color: "#806020" }}>RAAN optimized for first pass over this AOI</div>
                      </div>
                      <button onClick={() => setAnchorCty(null)} style={{ background: "rgba(255,170,0,0.15)", border: "1px solid rgba(255,170,0,0.3)", color: "#FFB300", padding: "2px 8px", borderRadius: 3, cursor: "pointer", fontSize: 8, fontFamily: "inherit" }}>Clear</button>
                    </div>
                  ) : null;
                })() : (
                  <div style={{ fontSize: 9, color: "#4a6878", lineHeight: 1.4 }}>
                    No anchor set. Click ⚓ on a selected country to set it as anchor — the orbit RAAN will be optimized so the first pass crosses that country.
                  </div>
                )}
              </div>

              <Lbl>SELECTED ({selCtys.length})</Lbl>
              {selCtyObjs.map(c => {
                const cloud = CLOUD_DATA[c.id] || 50;
                const isAnchor = anchorCty === c.id;
                return (
                  <P key={c.id} style={{ padding: "7px 10px", marginBottom: 4, borderLeft: `3px solid ${isAnchor ? "#FFB300" : SAT_COLORS[COUNTRIES.indexOf(c) % SAT_COLORS.length]}`, background: isAnchor ? "rgba(255,170,0,0.06)" : undefined }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <div>
                        <div style={{ fontWeight: 600, color: isAnchor ? "#FFB300" : "#d8ecff", fontSize: 10, display: "flex", alignItems: "center", gap: 4 }}>
                          {isAnchor && <span style={{ fontSize: 10 }}>⚓</span>}
                          {c.name}
                        </div>
                        <div style={{ fontSize: 8, color: "#2a5068" }}>
                          {c.area.toLocaleString()} km² · ☁ {cloud}% cloud
                        </div>
                      </div>
                      <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                        <button onClick={(e) => { e.stopPropagation(); setAnchorCty(isAnchor ? null : c.id); }} 
                          title={isAnchor ? "Remove anchor" : "Set as anchor"}
                          style={{ background: isAnchor ? "rgba(255,170,0,0.2)" : "rgba(255,170,0,0.05)", border: `1px solid ${isAnchor ? "rgba(255,170,0,0.4)" : "rgba(255,170,0,0.15)"}`, color: isAnchor ? "#FFB300" : "#806830", padding: "2px 6px", borderRadius: 3, cursor: "pointer", fontSize: 9, fontFamily: "inherit" }}>⚓</button>
                        <button onClick={() => toggleCty(c.id)} style={{ background: "none", border: "none", color: "#4a2828", cursor: "pointer", fontSize: 12, fontFamily: "inherit" }}>×</button>
                      </div>
                    </div>
                  </P>
                );
              })}
              <div style={{ height: 1, background: "rgba(70,140,200,0.08)", margin: "10px 0" }} />
              <Lbl>ALL COUNTRIES</Lbl>
              {COUNTRIES.sort((a, b) => a.name.localeCompare(b.name)).map(c => (
                <div key={c.id} onClick={() => toggleCty(c.id)} style={{
                  padding: "5px 9px", borderRadius: 3, cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center",
                  background: selCtys.includes(c.id) ? "rgba(70,140,200,0.08)" : "transparent",
                  borderLeft: selCtys.includes(c.id) ? `2px solid ${SAT_COLORS[COUNTRIES.indexOf(c) % SAT_COLORS.length]}` : "2px solid transparent"
                }}>
                  <span style={{ fontSize: 10, color: selCtys.includes(c.id) ? "#b0d8f0" : "#3a6080" }}>{c.name}</span>
                  <div style={{ display: "flex", gap: 8, fontSize: 8, color: "#1a3a50" }}>
                    <span>☁{CLOUD_DATA[c.id] || 50}%</span>
                    <span>{fmt(c.area)}km²</span>
                  </div>
                </div>
              ))}
            </>}

            {/* ── RESULTS TAB ── */}
            {tab === "results" && <>
              <Lbl>TIMEFRAME</Lbl>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 3, marginBottom: 12 }}>
                {tfOpts.map(o => <Btn key={o.v} sm on={tf === o.v} onClick={() => setTf(o.v)}>{o.l}</Btn>)}
              </div>
              <Lbl>COMMERCIAL RATE</Lbl>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 12 }}>
                <span style={{ fontSize: 9, color: "#b8a040" }}>$</span>
                <input type="number" value={commercialRate} onChange={e => setCommercialRate(+e.target.value || 0)}
                  style={{ width: 70, background: "rgba(4,8,16,0.9)", border: "1px solid rgba(255,215,0,0.2)", color: "#FFD740", padding: "4px 6px", borderRadius: 3, fontSize: 11, fontFamily: "inherit", fontWeight: 700 }} />
                <span style={{ fontSize: 8, color: "#706830" }}>per km²</span>
              </div>
              <Btn onClick={runAnalysis} style={{ width: "100%", marginBottom: 10, padding: "7px", background: "rgba(100,255,218,0.1)", borderColor: "rgba(100,255,218,0.3)", color: "#64FFDA" }}>
                {running ? "⟳ SIMULATING..." : "▶ RUN ANALYSIS"}
              </Btn>
              {results && results.map((r, i) => (
                <P key={i} style={{ padding: "7px 10px", marginBottom: 4, borderLeft: `3px solid ${r.satColor}` }}>
                  <div style={{ fontSize: 9, fontWeight: 600, color: "#b0d8f0", marginBottom: 3 }}>{r.label}</div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2px 8px", fontSize: 9 }}>
                    <div><span style={{ color: "#2a5068" }}>100%: </span><span style={{ color: r.fullCovDay ? "#64FFDA" : "#FF8A80", fontWeight: 700 }}>{r.fullCovDay ? `Day ${r.fullCovDay.toFixed(1)}` : `>D${tf}`}</span></div>
                    <div><span style={{ color: "#2a5068" }}>Cloud: </span><span style={{ color: "#80b8d8" }}>{r.cloudPct}%</span></div>
                    <div><span style={{ color: "#2a5068" }}>Net/yr: </span><span style={{ color: "#FFD740" }}>{fmt(r.netImgPerYear)} km²</span></div>
                    <div><span style={{ color: "#2a5068" }}>Value: </span><span style={{ color: "#76FF03" }}>{fmtMoney(r.netImgLifetime * commercialRate)}</span></div>
                  </div>
                </P>
              ))}
            </>}
          </div>
        </div>

        {/* ── MAIN ── */}
        <div ref={mainRef} style={{ flex: 1, overflowY: "auto", padding: "14px 18px" }}>
          <MainMap sats={(() => { const a = anchorCty ? COUNTRIES.find(c => c.id === anchorCty) : null; return a ? sats.map(s => satWithAnchorRaan(s, a)) : sats; })()} selCtys={selCtys} countries={COUNTRIES} onCtClick={toggleCty} anim={anim} showSw={showSw} showGr={showGr} anchorId={anchorCty} />
          <div style={{ display: "flex", gap: 10, marginTop: 6, marginBottom: 14 }}>
            {[["Swath", showSw, setShowSw], ["Grid", showGr, setShowGr]].map(([l, v, s]) => (
              <div key={l} onClick={() => s(!v)} style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer", fontSize: 9 }}>
                <div style={{ width: 11, height: 11, borderRadius: 2, background: v ? "rgba(70,140,200,0.22)" : "rgba(70,140,200,0.04)", border: `1px solid ${v ? "rgba(70,140,200,0.45)" : "rgba(70,140,200,0.12)"}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 7, color: "#64B5F6" }}>{v ? "✓" : ""}</div>
                <span style={{ color: v ? "#80b0d0" : "#2a4a60" }}>{l}</span>
              </div>
            ))}
            <span style={{ fontSize: 8, color: "#1a3a50", marginLeft: "auto" }}>Click map to select countries</span>
          </div>

          {/* Sat cards */}
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
            {sats.map(sat => {
              const T = orbPeriod(sat.altitude), sw = effSwathKm(sat.altitude, sat.swathAngle);
              return (
                <P key={sat.id} style={{ flex: "1 1 200px", maxWidth: 280, padding: "10px 12px", borderLeft: `3px solid ${sat.color}` }}>
                  <div style={{ fontWeight: 700, color: "#d8ecff", fontSize: 10, marginBottom: 6 }}>{sat.name}</div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 12px" }}>
                    {[["Period", `${(T / 60).toFixed(1)} min`], ["Orbits/day", (86400 / T).toFixed(1)], ["Eff. Swath", `${sw.toFixed(0)} km`], ["Off-nadir", `${sat.offNadir}°`], ["Lifetime", `${sat.lifetime} yrs`]].map(([k, v]) => (
                      <div key={k}><div style={{ fontSize: 7, color: "#2a4a60", letterSpacing: 1 }}>{k.toUpperCase()}</div><div style={{ fontSize: 10, color: "#90c8e0", fontWeight: 600 }}>{v}</div></div>
                    ))}
                    <div>
                      <div style={{ fontSize: 7, color: "#2a4a60", letterSpacing: 1 }}>CAPACITY</div>
                      <div style={{ display: "flex", alignItems: "center", gap: 3, marginTop: 1 }}>
                        <input
                          type="number" min="0" value={sat.dataCapacity}
                          onChange={e => setSats(prev => prev.map(s => s.id === sat.id ? { ...s, dataCapacity: +e.target.value || 0 } : s))}
                          style={{ width: 58, background: "rgba(4,8,16,0.8)", border: "1px solid rgba(100,200,255,0.2)", color: "#90c8e0", padding: "1px 4px", borderRadius: 2, fontSize: 9, fontFamily: "inherit", fontWeight: 600 }}
                        />
                        <span style={{ fontSize: 7, color: "#2a4a60" }}>km²/d</span>
                      </div>
                    </div>
                  </div>
                </P>
              );
            })}
          </div>

          {/* ── RESULTS ── */}
          {results && results.length > 0 && <>
            {/* Anchor info banner */}
            {anchorCty && (() => {
              const ac = COUNTRIES.find(c => c.id === anchorCty);
              const raan0 = (results[0] || {}).effectiveRaan;
              return ac ? (
                <P style={{ padding: "10px 14px", marginBottom: 14, borderLeft: "3px solid #FFB300", background: "rgba(255,180,0,0.04)" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontSize: 14 }}>⚓</span>
                    <div>
                      <div style={{ fontSize: 10, fontWeight: 700, color: "#FFB300" }}>Anchor: {ac.name}</div>
                      <div style={{ fontSize: 8, color: "#806830", lineHeight: 1.5 }}>
                        RAAN optimized to {raan0 !== undefined ? raan0.toFixed(1) : "—"}° — first orbital pass crosses {ac.name} at T+0. All coverage timelines start from this geometry. Other AOIs are secondary beneficiaries.
                      </div>
                    </div>
                  </div>
                </P>
              ) : null;
            })()}
            {/* Timeline */}
            <P data-section="timeline" style={{ padding: 14, marginBottom: 14 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <Lbl>COVERAGE TIMELINE</Lbl>
                <div style={{ display: "flex", gap: 3 }}>{tfOpts.map(o => <Btn key={o.v} sm on={tf === o.v} onClick={() => setTf(o.v)}>{o.l}</Btn>)}</div>
              </div>
              <TimelineChart results={results.map(r => ({ color: r.satColor, data: r.timeline, label: r.label }))} timeframe={tf} />
              <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: 8 }}>
                {results.map((r, i) => <div key={i} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 8 }}><div style={{ width: 10, height: 2.5, borderRadius: 2, background: r.satColor }} /><span style={{ color: "#4a7890" }}>{r.label}</span></div>)}
              </div>
            </P>

            {/* ── REVISIT ANALYSIS ── */}
            <P data-section="revisit" style={{ padding: 14, marginBottom: 14 }}>
              <Lbl>REVISIT ANALYSIS</Lbl>
              <div style={{ fontSize: 9, color: "#3a6080", marginBottom: 10, lineHeight: 1.5 }}>
                Pass-based revisit metrics over 30 days. Mean Revisit and Max Gap computed directly from the detected pass list. 100% Coverage from grid simulation.
              </div>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
                  <thead>
                    <tr style={{ borderBottom: "2px solid rgba(70,140,200,0.15)" }}>
                      {[
                        { label: "Satellite → Country", center: false },
                        { label: "Mode", center: false },
                        { label: "Mean\nRevisit", center: true },
                        { label: "Max\nGap", center: true },
                        { label: "First\nSunlit", center: true },
                        { label: "100%\nMapped", center: true },
                        { label: "1d\npasses", center: true, sep: true },
                        { label: "3d\npasses", center: true },
                        { label: "5d\npasses", center: true },
                        { label: "7d\npasses", center: true },
                        { label: "14d\npasses", center: true },
                        { label: "30d\npasses", center: true },
                      ].map((h, i) => (
                        <th key={i} style={{
                          textAlign: h.center ? "center" : "left",
                          padding: "7px 6px",
                          color: h.sep ? "#4a8898" : i < 6 ? "#2a5a78" : "#4a8898",
                          fontWeight: 600, fontSize: 8, letterSpacing: 0.8,
                          whiteSpace: "pre-line", verticalAlign: "bottom",
                          borderLeft: h.sep ? "1px solid rgba(70,140,200,0.12)" : "none"
                        }}>{h.label}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((r, i) => {
                      const fmtD = (d) => {
                        if (d === null || d === undefined) return "—";
                        if (d < 1/24) return `${(d * 24 * 60).toFixed(0)} min`;
                        if (d < 1) return `${(d * 24).toFixed(1)} hrs`;
                        if (d > 365) return ">1yr";
                        return `${d.toFixed(1)} days`;
                      };
                      const pc = r.passCountByTf || {};
                      const spc = r.sunlitPassCountByTf || {};
                      return [
                        <tr key={`${i}-all`} style={{ borderBottom: "1px solid rgba(70,140,200,0.04)", background: r.isAnchor ? "rgba(255,180,0,0.04)" : "transparent" }}>
                          <td style={{ padding: "6px 6px", fontWeight: 600, verticalAlign: "middle" }}>
                            {r.isAnchor && <span style={{ color: "#FFB300", marginRight: 4 }}>⚓</span>}
                            <span style={{ color: r.satColor }}>{r.satName}</span>
                            <span style={{ color: "#2a4a60" }}> → </span>
                            <span style={{ color: "#80a8c0" }}>{r.countryName}</span>
                          </td>
                          <td style={{ padding: "6px 6px", fontSize: 8, color: "#4a7890", fontWeight: 600, whiteSpace: "nowrap" }}>ALL</td>
                          <td style={{ padding: "6px 6px", textAlign: "center", color: "#a0d0e8" }}>{fmtD(r.meanRevisitDays)}</td>
                          <td style={{ padding: "6px 6px", textAlign: "center", color: "#FFD740" }}>{fmtD(r.maxGapDays)}</td>
                          <td style={{ padding: "6px 6px", textAlign: "center", color: "#3a4a58" }}>—</td>
                          <td style={{ padding: "6px 6px", textAlign: "center", color: "#3a4a58" }}>—</td>
                          {[1,3,5,7,14,30].map((b, bi) => (
                            <td key={b} style={{ padding: "6px 6px", textAlign: "center", fontWeight: 600, borderLeft: bi === 0 ? "1px solid rgba(70,140,200,0.08)" : "none", color: (pc[b] || 0) === 0 ? "#3a4a58" : (pc[b] || 0) >= 5 ? "#64FFDA" : "#a0d0e8" }}>
                              {pc[b] || 0}
                            </td>
                          ))}
                        </tr>,
                        <tr key={`${i}-opt`} style={{ borderBottom: "1px solid rgba(70,140,200,0.08)", background: "rgba(255,179,0,0.03)" }}>
                          <td style={{ padding: "6px 6px", verticalAlign: "middle", color: "#3a4050", fontSize: 8 }}></td>
                          <td style={{ padding: "6px 6px", fontSize: 9, color: "#FFB300", fontWeight: 700, whiteSpace: "nowrap" }}>☀ OPT</td>
                          <td style={{ padding: "6px 6px", textAlign: "center", color: "#FFB300" }}>{fmtD(r.sunlitMeanRevisitDays)}</td>
                          <td style={{ padding: "6px 6px", textAlign: "center", color: "#cc8800" }}>{fmtD(r.sunlitMaxGapDays)}</td>
                          <td style={{ padding: "6px 6px", textAlign: "center", color: "#FFB300", fontWeight: 700 }}>{fmtD(r.firstSunlitPassDay)}</td>
                          <td style={{ padding: "6px 6px", textAlign: "center", color: "#FFB300", fontWeight: 700, lineHeight: 1.4 }}>
                            <div>{fmtD(r.hundredPctMappedSunlitDays)}</div>
                            {r.hundredPctMappedSunlitCloud !== null && <div style={{ fontSize: 8, color: "#a07820", fontWeight: 400 }}>{fmtD(r.hundredPctMappedSunlitCloud)} ☁</div>}
                            {r.offNadirNote && <div style={{ fontSize: 7, color: "#806040" }}>nadir only</div>}
                          </td>
                          {[1,3,5,7,14,30].map((b, bi) => {
                            const ca = (r.coveredAreaSunlitByTf || {})[b];
                            const caStr = ca > 0 ? (ca >= 1e6 ? `${(ca/1e6).toFixed(1)}M` : ca >= 1e3 ? `${(ca/1e3).toFixed(1)}k` : ca.toFixed(0)) : null;
                            return (
                              <td key={b} style={{ padding: "6px 6px", textAlign: "center", fontWeight: 600, borderLeft: bi === 0 ? "1px solid rgba(70,140,200,0.08)" : "none", color: (spc[b] || 0) === 0 ? "#3a4a40" : "#FFB300" }}>
                                {spc[b] || 0}
                                {caStr && <div style={{ fontSize: 7, color: "#806040", fontWeight: 400, marginTop: 1 }}>{caStr} km²</div>}
                              </td>
                            );
                          })}
                        </tr>
                      ];
                    })}
                  </tbody>
                </table>
              </div>

              {/* Aggregate stats when multiple countries */}
              {selCtyObjs.length > 1 && sats.length > 0 && (
                <div style={{ marginTop: 14, padding: "10px 12px", background: "rgba(100,255,218,0.04)", border: "1px solid rgba(100,255,218,0.12)", borderRadius: 4 }}>
                  <div style={{ fontSize: 9, color: "#64FFDA", fontWeight: 700, marginBottom: 8, letterSpacing: 1 }}>AGGREGATE ACROSS ALL AOIs</div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 10 }}>
                    {sats.map(sat => {
                      const satResults = results.filter(r => r.satId === sat.id);
                      if (satResults.length === 0) return null;
                      const bestFirst = Math.min(...satResults.map(r => r.firstAllPassDay).filter(v => v !== null && v !== undefined));
                      const avgMeanRevisit = satResults.reduce((s, r) => s + (r.meanRevisitDays || 0), 0) / satResults.length;
                      const worstMaxGap = Math.max(...satResults.map(r => r.maxGapDays || 0).filter(isFinite));
                      const allCovered = satResults.every(r => r.fullCovDay !== null);
                      const maxCovDay = Math.max(...satResults.map(r => r.fullCovDay || Infinity));
                      const fmtD = (d) => {
                        if (d === null || d === undefined || !isFinite(d)) return "—";
                        if (d < 1/24) return `${(d * 24 * 60).toFixed(0)} min`;
                        if (d < 1) return `${(d * 24).toFixed(1)} hrs`;
                        return `${d.toFixed(1)} days`;
                      };
                      return (
                        <P key={sat.id} style={{ padding: "8px 10px", borderLeft: `3px solid ${sat.color}` }}>
                          <div style={{ fontWeight: 700, color: "#d8ecff", fontSize: 10, marginBottom: 6 }}>{sat.name}</div>
                          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "3px 10px", fontSize: 9 }}>
                            <div><span style={{ color: "#2a5068" }}>First pass (best):</span> <span style={{ color: "#64FFDA", fontWeight: 700 }}>{fmtD(bestFirst)}</span></div>
                            <div><span style={{ color: "#2a5068" }}>Avg mean revisit:</span> <span style={{ color: "#a0d0e8" }}>{fmtD(avgMeanRevisit)}</span></div>
                            <div><span style={{ color: "#2a5068" }}>Worst max gap:</span> <span style={{ color: "#FFD740" }}>{fmtD(worstMaxGap)}</span></div>
                            <div style={{ gridColumn: "1/-1" }}><span style={{ color: "#2a5068" }}>All {satResults.length} AOIs covered:</span> <span style={{ color: allCovered ? "#64FFDA" : "#FF8A80", fontWeight: 700 }}>{allCovered ? `Yes (last at day ${maxCovDay.toFixed(1)})` : "No"}</span></div>
                          </div>
                        </P>
                      );
                    })}
                  </div>
                </div>
              )}
            </P>

            {/* ── COUNTRY GROUND TRACK MAPS ── */}
            <P data-section="groundtrack" style={{ padding: 14, marginBottom: 14 }}>
              <Lbl>COUNTRY GROUND TRACK ANALYSIS</Lbl>
              <div style={{ fontSize: 9, color: "#3a6080", marginBottom: 10, lineHeight: 1.5 }}>
                Zoomed maps showing full ground track lines over each target country. Use the timeframe buttons below each map to see track buildup over 1 day → 30 days. Green cells = covered area.
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 8 }}>
                {sats.map(sat => {
                  const anchorObj = anchorCty ? COUNTRIES.find(c => c.id === anchorCty) : null;
                  const effSat = anchorObj ? satWithAnchorRaan(sat, anchorObj) : sat;
                  const anchorOffset = anchorFirstPassTimes[sat.id] || 0;
                  return selCtyObjs.map(cty => (
                    <CountryTrackMap
                      key={`${sat.id}-${cty.id}-${anchorCty || ''}`}
                      country={cty}
                      sat={effSat}
                      anchorOffset={anchorOffset}
                      width={500} height={440}
                    />
                  ));
                })}
              </div>
            </P>

            {/* ── SATELLITE UTILIZATION & ECONOMICS ── */}
            <P data-section="economics" style={{ padding: 14, marginBottom: 14 }}>
              <Lbl>SATELLITE UTILIZATION &amp; ECONOMICS</Lbl>
              <div style={{ fontSize: 9, color: "#3a6080", marginBottom: 14, lineHeight: 1.5 }}>
                Per-satellite capacity analysis across all selected AOIs. Daily demand = total sunlit ground coverage the satellite could image across every country combined.
              </div>
              {sats.map(sat => {
                const satResults = results.filter(r => r.satId === sat.id);
                if (!satResults.length) return null;
                const dailyCap = satResults[0].satDataCapacity;
                const totalUncappedDaily = satResults.reduce((s, r) => s + ((r.uncappedSunlitAreaByTf[30] || 0) / 30), 0);
                const effectiveDaily = dailyCap > 0 ? Math.min(totalUncappedDaily, dailyCap) : totalUncappedDaily;
                const utilization = dailyCap > 0 ? (effectiveDaily / dailyCap * 100) : null;
                const isCapLimited = dailyCap > 0 && totalUncappedDaily > dailyCap;
                const headroom = dailyCap > 0 ? dailyCap - totalUncappedDaily : null;
                const imagerPerYear = effectiveDaily * 365.25;
                const imageryLifetime = imagerPerYear * (satResults[0].satLifetime || 5);
                // Max daily potential: highest single-day total across all countries combined
                const dayTotals = new Float64Array(30);
                for (const r of satResults) {
                  const cpd = r.sunlitCoveragePerDay || [];
                  for (let d = 0; d < 30; d++) dayTotals[d] += cpd[d] || 0;
                }
                const maxDailyPotential = Math.max(...dayTotals);
                return (
                  <div key={sat.id} style={{ marginBottom: 14, padding: "12px 14px", background: "rgba(4,8,16,0.6)", borderRadius: "0 6px 6px 0", border: `1px solid rgba(70,140,200,0.1)`, borderLeft: `3px solid ${sat.color}` }}>
                    {/* Header */}
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <span style={{ color: sat.color, fontWeight: 700, fontSize: 11 }}>{sat.name}</span>
                        {(satResults[0].satCount || 1) > 1 && <span style={{ fontSize: 9, color: "#FFD740", fontWeight: 700 }}>×{satResults[0].satCount}</span>}
                        <span style={{ fontSize: 8, color: "#2a4a60" }}>{sat.altitude}km / {sat.inclination}° / {effSwathKm(sat.altitude, sat.swathAngle).toFixed(0)}km swath</span>
                      </div>
                      <span style={{ fontSize: 9, fontWeight: 700, padding: "2px 8px", borderRadius: 3,
                        background: isCapLimited ? "rgba(255,82,82,0.12)" : "rgba(105,255,71,0.1)",
                        color: isCapLimited ? "#FF5252" : "#69FF47",
                        border: `1px solid ${isCapLimited ? "rgba(255,82,82,0.3)" : "rgba(105,255,71,0.25)"}`
                      }}>{isCapLimited ? "CAPACITY LIMITED" : "OPPORTUNITY LIMITED"}</span>
                    </div>

                    {/* A: Daily demand vs capacity */}
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 6, marginBottom: 12 }}>
                      {[
                        ["Avg Daily Demand", `${fmt(totalUncappedDaily)} km²/d`, "#FFD740"],
                        ["Max Daily Potential", maxDailyPotential > 0 ? `${fmt(maxDailyPotential)} km²/d` : "—", "#FF9800"],
                        ["Daily Capacity", dailyCap > 0 ? `${fmt(dailyCap)} km²/d` : "—", "#64FFDA"],
                        ["Utilization", utilization !== null ? `${utilization.toFixed(0)}%` : "—",
                          utilization === null ? "#4a6880" : utilization >= 95 ? "#FF5252" : utilization >= 70 ? "#FFD740" : "#69FF47"],
                        ["Headroom", headroom !== null
                          ? `${headroom >= 0 ? "+" : "−"}${fmt(Math.abs(headroom))} km²/d`
                          : "—",
                          headroom === null ? "#4a6880" : headroom >= 0 ? "#69FF47" : "#FF5252"],
                      ].map(([k, v, c]) => (
                        <div key={k} style={{ padding: "6px 8px", background: "rgba(0,0,0,0.35)", borderRadius: 4 }}>
                          <div style={{ fontSize: 7, color: "#2a4a60", letterSpacing: 1, marginBottom: 2 }}>{k.toUpperCase()}</div>
                          <div style={{ fontSize: 11, color: c, fontWeight: 700 }}>{v}</div>
                        </div>
                      ))}
                    </div>
                    {/* Actual output per year (capacity-capped) */}
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12, padding: "8px 10px", background: "rgba(100,255,218,0.05)", border: "1px solid rgba(100,255,218,0.15)", borderRadius: 4 }}>
                      <div>
                        <div style={{ fontSize: 7, color: "#3a7060", letterSpacing: 1, marginBottom: 2 }}>ACTUAL OUTPUT / YR</div>
                        <div style={{ fontSize: 16, color: "#64FFDA", fontWeight: 700 }}>{fmt(imagerPerYear)} km²</div>
                      </div>
                      <div style={{ fontSize: 8, color: "#2a5048", lineHeight: 1.5 }}>
                        = min(avg demand, capacity) × 365<br />
                        {dailyCap > 0 && isCapLimited
                          ? `Capped at ${fmt(dailyCap)} km²/d capacity`
                          : `Demand-limited at ${fmt(totalUncappedDaily)} km²/d`}
                      </div>
                    </div>

                    {/* B: Per-country breakdown */}
                    <div style={{ fontSize: 7, color: "#2a4060", letterSpacing: 1, marginBottom: 5 }}>PER-COUNTRY BREAKDOWN (SUNLIT PASSES)</div>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 9, marginBottom: 10 }}>
                      <thead>
                        <tr style={{ borderBottom: "1px solid rgba(70,140,200,0.15)" }}>
                          {["Country", "Passes/30d", "Avg track", "km²/pass", "km²/yr", "Demand share"].map((h, hi) => (
                            <th key={hi} style={{ padding: "4px 6px", textAlign: hi === 0 ? "left" : "right", color: "#2a5068", fontSize: 7, letterSpacing: 0.7, fontWeight: 600 }}>{h.toUpperCase()}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {satResults.map((r, ri) => {
                          const grossPerPass = r.avgSunlitPassLengthKm * r.swKm;
                          const annualImagery = (r.sunlitPassCountByTf[30] || 0) * (365.25 / 30) * grossPerPass;
                          const dailyDemand = (r.uncappedSunlitAreaByTf[30] || 0) / 30;
                          // Share derived from uncappedSunlitAreaByTf to match km²/yr column exactly
                          const share = totalUncappedDaily > 0 ? (dailyDemand / totalUncappedDaily * 100) : 0;
                          const shareStr = share < 0.1 ? '<0.1' : share < 10 ? share.toFixed(1) : share.toFixed(0);
                          return (
                            <tr key={ri} style={{ borderBottom: "1px solid rgba(70,140,200,0.05)" }}>
                              <td style={{ padding: "5px 6px", color: "#80a8c0", fontWeight: 600 }}>{r.countryName}</td>
                              <td style={{ padding: "5px 6px", textAlign: "right", color: "#a0d0e8" }}>{r.sunlitPassCountByTf[30] || 0}</td>
                              <td style={{ padding: "5px 6px", textAlign: "right", color: "#a0d0e8" }}>{r.avgSunlitPassLengthKm.toFixed(0)} km</td>
                              <td style={{ padding: "5px 6px", textAlign: "right", color: "#FFD740" }}>{fmt(grossPerPass)} km²</td>
                              <td style={{ padding: "5px 6px", textAlign: "right", color: "#76FF03" }}>{fmt(annualImagery)} km²</td>
                              <td style={{ padding: "5px 6px", textAlign: "right" }}>
                                <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 4 }}>
                                  <div style={{ width: 36, height: 4, background: "rgba(70,140,200,0.1)", borderRadius: 2 }}>
                                    <div style={{ width: `${Math.min(share, 100)}%`, height: "100%", background: sat.color, borderRadius: 2, opacity: 0.7 }} />
                                  </div>
                                  <span style={{ color: "#80c0d8", minWidth: 32, textAlign: "right" }}>{shareStr}%</span>
                                </div>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>

                    {/* C: Lifetime economics summary */}
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, paddingTop: 8, borderTop: "1px solid rgba(70,140,200,0.1)" }}>
                      {[
                        ["Imagery / yr", `${fmt(imagerPerYear)} km²`, "#FFD740"],
                        ["Lifetime imagery", `${fmt(imageryLifetime)} km²`, "#64FFDA"],
                        ["Peak day", maxDailyPotential > 0 ? `${fmt(maxDailyPotential)} km²` : "—", "#FF9800"],
                        ["Lifetime", `${satResults[0].satLifetime} yrs`, "#607080"],
                      ].map(([k, v, c]) => (
                        <div key={k}>
                          <div style={{ fontSize: 7, color: "#2a4a60", letterSpacing: 1 }}>{k.toUpperCase()}</div>
                          <div style={{ fontSize: 10, color: c, fontWeight: 700 }}>{v}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </P>

            {/* ── SUMMARY STATS ── */}
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
              {[
                {
                  label: "TOTAL LIFETIME VALUE (NET)",
                  value: fmtMoney(results.reduce((s, r) => s + r.netImgLifetime * commercialRate, 0)),
                  sub: `across ${results.length} combinations`,
                  color: "#76FF03"
                },
                {
                  label: "TOTAL NET IMAGERY",
                  value: fmt(results.reduce((s, r) => s + r.netImgLifetime, 0)) + " km²",
                  sub: `(${fmt(results.reduce((s, r) => s + r.grossImgLifetime, 0))} km² gross)`,
                  color: "#FFD740"
                },
                {
                  label: "FASTEST FULL COVERAGE",
                  value: (() => { const fc = results.filter(r => r.fullCovDay).sort((a, b) => a.fullCovDay - b.fullCovDay)[0]; return fc ? `Day ${fc.fullCovDay.toFixed(1)}` : "None"; })(),
                  sub: (() => { const fc = results.filter(r => r.fullCovDay).sort((a, b) => a.fullCovDay - b.fullCovDay)[0]; return fc ? fc.label : ""; })(),
                  color: "#64FFDA"
                },
                {
                  label: "AVG CLOUD PENALTY",
                  value: `${(results.reduce((s, r) => s + r.cloudPct, 0) / results.length).toFixed(0)}%`,
                  sub: `effective clear fraction: ${(100 - results.reduce((s, r) => s + r.cloudPct, 0) / results.length).toFixed(0)}%`,
                  color: "#FF80AB"
                },
              ].map(s => (
                <P key={s.label} style={{ flex: "1 1 200px", padding: "10px 12px" }}>
                  <div style={{ fontSize: 7, color: "#2a4a60", letterSpacing: 1, marginBottom: 4 }}>{s.label}</div>
                  <div style={{ fontSize: 14, color: s.color, fontWeight: 700 }}>{s.value}</div>
                  <div style={{ fontSize: 8, color: "#3a5a70", marginTop: 2 }}>{s.sub}</div>
                </P>
              ))}
            </div>

            {/* ── COMMERCIAL IMAGERY VALUE ── */}
            <P data-section="commercial" style={{ padding: 14, marginBottom: 14 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <Lbl style={{ marginBottom: 0 }}>COMMERCIAL IMAGERY VALUE</Lbl>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ fontSize: 8, color: "#3a6080", letterSpacing: 1 }}>RATE</span>
                  <input type="number" min="0" step="1" value={commercialRate}
                    onChange={e => setCommercialRate(+e.target.value || 0)}
                    style={{ width: 56, background: "rgba(4,8,16,0.9)", border: "1px solid rgba(70,140,200,0.25)", color: "#FFD740", padding: "3px 5px", borderRadius: 3, fontSize: 10, fontFamily: "inherit", textAlign: "center" }} />
                  <span style={{ fontSize: 8, color: "#2a4a60" }}>$/km²</span>
                </div>
              </div>
              {sats.map(sat => {
                const satResults = results.filter(r => r.satId === sat.id);
                if (!satResults.length) return null;
                const dailyCap = satResults[0].satDataCapacity;
                const totalUncappedDaily = satResults.reduce((s, r) => s + ((r.uncappedSunlitAreaByTf[30] || 0) / 30), 0);
                const effectiveDaily = dailyCap > 0 ? Math.min(totalUncappedDaily, dailyCap) : totalUncappedDaily;
                const imgPerMonth = effectiveDaily * 30.44;
                const imgPerYear = effectiveDaily * 365.25;
                const imgLifetime = imgPerYear * (satResults[0].satLifetime || 5);
                // Net (cloud-corrected): apply per-country cloud fraction then re-aggregate
                const totalNetDaily = satResults.reduce((s, r) => {
                  const cf = 1 - (r.cloudPct || 50) / 100;
                  return s + ((r.uncappedSunlitAreaByTf[30] || 0) / 30) * cf;
                }, 0);
                const effectiveNetDaily = dailyCap > 0 ? Math.min(totalNetDaily, dailyCap) : totalNetDaily;
                const netImgPerYear = effectiveNetDaily * 365.25;
                const netImgLifetime = netImgPerYear * (satResults[0].satLifetime || 5);
                const valuePerYear = netImgPerYear * commercialRate;
                const valueLifetime = netImgLifetime * commercialRate;
                return (
                  <div key={sat.id} style={{ marginBottom: 12, padding: "12px 14px", background: "rgba(4,8,16,0.6)", borderRadius: "0 6px 6px 0", border: "1px solid rgba(70,140,200,0.1)", borderLeft: `3px solid ${sat.color}` }}>
                    {/* Header */}
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                      <span style={{ color: sat.color, fontWeight: 700, fontSize: 11 }}>{sat.name}</span>
                      <span style={{ fontSize: 8, color: "#3a6080" }}>{sat.altitude}km · {effSwathKm(sat.altitude, sat.swathAngle).toFixed(0)}km swath · {satResults[0].satLifetime}yr lifetime</span>
                    </div>

                    {/* Imagery volume — gross (sunlit) vs net (cloud-free) */}
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 6, marginBottom: 6 }}>
                      {[
                        ["Gross imagery / mo", `${fmt(imgPerMonth)} km²`, "#a0d0e8"],
                        ["Gross imagery / yr", `${fmt(imgPerYear)} km²`, "#7090a8"],
                        ["Gross imagery / lifetime", `${fmt(imgLifetime)} km²`, "#506878"],
                      ].map(([k, v, c]) => (
                        <div key={k} style={{ padding: "6px 8px", background: "rgba(0,0,0,0.25)", borderRadius: 4 }}>
                          <div style={{ fontSize: 7, color: "#2a4a60", letterSpacing: 1, marginBottom: 2 }}>{k.toUpperCase()}</div>
                          <div style={{ fontSize: 11, color: c, fontWeight: 700 }}>{v}</div>
                        </div>
                      ))}
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 6, marginBottom: 10 }}>
                      {[
                        ["Net imagery / mo ☁", `${fmt(effectiveNetDaily * 30.44)} km²`, "#a0d0e8"],
                        ["Net imagery / yr ☁", `${fmt(netImgPerYear)} km²`, "#FFD740"],
                        ["Net imagery / lifetime ☁", `${fmt(netImgLifetime)} km²`, "#64FFDA"],
                      ].map(([k, v, c]) => (
                        <div key={k} style={{ padding: "6px 8px", background: "rgba(0,0,0,0.35)", borderRadius: 4 }}>
                          <div style={{ fontSize: 7, color: "#2a4a60", letterSpacing: 1, marginBottom: 2 }}>{k.toUpperCase()}</div>
                          <div style={{ fontSize: 11, color: c, fontWeight: 700 }}>{v}</div>
                        </div>
                      ))}
                    </div>

                    {/* Commercial value (based on net cloud-free imagery) */}
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginBottom: 10 }}>
                      {[
                        ["Commercial value / yr", `${fmtMoney(valuePerYear)}`, "#76FF03"],
                        ["Commercial value / lifetime", `${fmtMoney(valueLifetime)}`, "#69FF47"],
                      ].map(([k, v, c]) => (
                        <div key={k} style={{ padding: "7px 10px", background: "rgba(70,255,100,0.04)", border: "1px solid rgba(70,255,100,0.12)", borderRadius: 4 }}>
                          <div style={{ fontSize: 7, color: "#2a5a40", letterSpacing: 1, marginBottom: 3 }}>{k.toUpperCase()}</div>
                          <div style={{ fontSize: 13, color: c, fontWeight: 700 }}>{v}</div>
                          <div style={{ fontSize: 8, color: "#2a4a30", marginTop: 1 }}>@ ${commercialRate}/km² · net (cloud-free)</div>
                        </div>
                      ))}
                    </div>

                    {/* Per-country breakdown */}
                    <div style={{ fontSize: 7, color: "#2a4060", letterSpacing: 1, marginBottom: 5 }}>PER-COUNTRY IMAGERY (GROSS → NET)</div>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 9 }}>
                      <thead>
                        <tr style={{ borderBottom: "1px solid rgba(70,140,200,0.12)" }}>
                          {["Country", "Passes/30d", "Avg track", "Gross km²/yr", "Cloud", "Net km²/yr", "Value/yr (net)", "Share"].map((h, hi) => (
                            <th key={hi} style={{ padding: "3px 6px", textAlign: hi === 0 ? "left" : "right", color: "#2a5068", fontSize: 7, letterSpacing: 0.7, fontWeight: 600 }}>{h.toUpperCase()}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {satResults.map((r, ri) => {
                          const passesPerYear = (r.sunlitPassCountByTf[30] || 0) * (365.25 / 30);
                          const grossAnnualKm2 = passesPerYear * r.avgSunlitPassLengthKm * r.swKm;
                          const cldFrac = (r.cloudPct || 50) / 100;
                          const netAnnualKm2 = grossAnnualKm2 * (1 - cldFrac);
                          const annualValue = netAnnualKm2 * commercialRate;
                          const share = netImgPerYear > 0 ? (netAnnualKm2 / netImgPerYear * 100) : 0;
                          return (
                            <tr key={ri} style={{ borderBottom: "1px solid rgba(70,140,200,0.04)" }}>
                              <td style={{ padding: "5px 6px", color: "#80a8c0", fontWeight: 600 }}>{r.countryName}</td>
                              <td style={{ padding: "5px 6px", textAlign: "right", color: "#a0d0e8" }}>{r.sunlitPassCountByTf[30] || 0}</td>
                              <td style={{ padding: "5px 6px", textAlign: "right", color: "#a0d0e8" }}>{r.avgSunlitPassLengthKm.toFixed(0)} km</td>
                              <td style={{ padding: "5px 6px", textAlign: "right", color: "#506878" }}>{fmt(grossAnnualKm2)} km²</td>
                              <td style={{ padding: "5px 6px", textAlign: "right", color: "#7090a0" }}>{r.cloudPct}%</td>
                              <td style={{ padding: "5px 6px", textAlign: "right", color: "#FFD740" }}>{fmt(netAnnualKm2)} km²</td>
                              <td style={{ padding: "5px 6px", textAlign: "right", color: "#76FF03" }}>{fmtMoney(annualValue)}</td>
                              <td style={{ padding: "5px 6px", textAlign: "right" }}>
                                <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 4 }}>
                                  <div style={{ width: 36, height: 4, background: "rgba(70,140,200,0.1)", borderRadius: 2 }}>
                                    <div style={{ width: `${Math.min(share, 100)}%`, height: "100%", background: sat.color, borderRadius: 2, opacity: 0.7 }} />
                                  </div>
                                  <span style={{ color: "#80c0d8", minWidth: 28, textAlign: "right" }}>{share.toFixed(0)}%</span>
                                </div>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                );
              })}
            </P>
          </>}

          {(!results || results.length === 0) && !running && (
            <P style={{ padding: "36px 18px", textAlign: "center" }}>
              <div style={{ fontSize: 22, marginBottom: 8, opacity: 0.15 }}>◉</div>
              <div style={{ color: "#2a4a60", fontSize: 11, lineHeight: 1.8 }}>
                Add satellites · select countries on the map · press <span style={{ color: "#64FFDA" }}>RUN ANALYSIS</span>
              </div>
            </P>
          )}
        </div>
      </div>
    </div>
  );
}
