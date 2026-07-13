import { useState, useRef } from "react";

/* ══════════════════════════════════════════════════════════════
   CONSTELLATION STUDY
   Multi-plane constellation revisit optimizer for a single target.
   Physics is an exact copy of App.js (Kepler + J2 nodal precession,
   ppo=500 tracks, bbox-buffer pass detection, pass split at 0.4×T).
   Unlike the main analysis (per-satellite), passes here are detected
   per satellite and MERGED across the fleet, so multi-plane revisit
   is computed correctly.
   Method + reference numbers: TRAP-Taiwan-Constellation-Trade-Study.
   ══════════════════════════════════════════════════════════════ */

const DEG = Math.PI / 180, RAD = 180 / Math.PI;
const R_E = 6371, MU = 398600.4418, J2 = 1.08263e-3, OMEGA_E = 7.2921159e-5;

function orbPeriod(alt) { const a = R_E + alt; return 2 * Math.PI * Math.sqrt(a * a * a / MU); }
function nodalPrec(alt, inc, ecc = 0) {
  const a = R_E + alt, n = Math.sqrt(MU / (a * a * a)), p = a * (1 - ecc * ecc);
  return -1.5 * n * J2 * Math.pow(R_E / p, 2) * Math.cos(inc * DEG) * RAD * 86400;
}
function effSwathKm(alt, swHalf) { return 2 * alt * Math.tan(swHalf * DEG); }

function groundTrack(sat, numOrbits, ptsPerOrbit) {
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
    let lon = Math.atan2(y, x) * RAD - OMEGA_E * t * RAD;
    lon = lon % 360; if (lon > 180) lon -= 360; if (lon < -180) lon += 360;
    pts.push({ lat, lon, t });
  }
  return pts;
}
function computeOptimalRaan(sat, cty) {
  const targetLon = (cty.lonMin + cty.lonMax) / 2, targetLat = (cty.latMin + cty.latMax) / 2;
  const testPts = groundTrack({ ...sat, raan: 0 }, 1, 2000);
  let bestShift = 0, bestErr = 999;
  for (let i = 0; i < testPts.length; i++) {
    if (Math.abs(testPts[i].lat - targetLat) < 3) {
      const err = Math.abs(testPts[i].lat - targetLat);
      if (err < bestErr) { bestErr = err; bestShift = targetLon - testPts[i].lon; }
    }
  }
  return ((bestShift % 360) + 360) % 360;
}
function isInImagingWindow(tSeconds, latDeg, lonDeg, dayOfYear = 80) {
  const decl = 23.44 * Math.sin((360 / 365) * (dayOfYear - 81) * DEG);
  const cosH = -Math.tan(latDeg * DEG) * Math.tan(decl * DEG);
  if (cosH > 1) return false; if (cosH < -1) return true;
  const halfDay = Math.acos(Math.max(-1, Math.min(1, cosH))) / (15 * DEG);
  let lst = ((tSeconds / 3600) + lonDeg / 15) % 24; if (lst < 0) lst += 24;
  return lst >= 12 - halfDay + 1 && lst <= 12 + halfDay - 1;
}
/* per-satellite pass mid-points over the target (App.js computePasses core) */
function passMids(sat, cty, numDays, ppo) {
  const T = orbPeriod(sat.altitude);
  const nOrbits = Math.ceil((86400 / T) * numDays);
  const pts = groundTrack(sat, nOrbits, ppo);
  const sw = effSwathKm(sat.altitude, sat.swathAngle || 10);
  const swDegLat = (sw / (2 * Math.PI * R_E)) * 360;
  const latBuf = swDegLat / 2 + 0.8, lonBuf = 1.5;
  const mids = []; let cur = null;
  const push = () => { if (cur && cur.length >= 2) mids.push(cur[Math.floor(cur.length / 2)]); };
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const over = p.lat >= cty.latMin - latBuf && p.lat <= cty.latMax + latBuf &&
                 p.lon >= cty.lonMin - lonBuf && p.lon <= cty.lonMax + lonBuf;
    if (over) {
      if (!cur) cur = [p];
      else if (p.t - cur[cur.length - 1].t > T * 0.4) { push(); cur = [p]; }
      else cur.push(p);
    } else if (cur) { push(); cur = null; }
  }
  push();
  return mids;
}
function metricsFromMids(mids, windowDays) {
  const days = mids.map(m => m.t / 86400).filter(d => d >= 0 && d <= windowDays).sort((a, b) => a - b);
  const sun = mids.filter(m => isInImagingWindow(m.t, m.lat, m.lon))
    .map(m => m.t / 86400).filter(d => d >= 0 && d <= windowDays).sort((a, b) => a - b);
  const mean = days.length > 1 ? (days[days.length - 1] - days[0]) / (days.length - 1) : null;
  const sMean = sun.length > 1 ? (sun[sun.length - 1] - sun[0]) / (sun.length - 1) : null;
  let gap = null, sGap = null;
  if (days.length > 1) { gap = 0; for (let i = 1; i < days.length; i++) gap = Math.max(gap, days[i] - days[i - 1]); }
  if (sun.length > 1) { sGap = 0; for (let i = 1; i < sun.length; i++) sGap = Math.max(sGap, sun[i] - sun[i - 1]); }
  return { passes: days.length, sunlit: sun.length,
    meanH: mean !== null ? mean * 24 : null, sunMeanH: sMean !== null ? sMean * 24 : null,
    gapH: gap !== null ? gap * 24 : null, sunGapH: sGap !== null ? sGap * 24 : null };
}
function evalConstellation(sats, cty, days, ppo) {
  const all = [];
  for (const s of sats) { const m = passMids(s, cty, days, ppo); for (const x of m) all.push(x); }
  all.sort((a, b) => a.t - b.t);
  return { m: metricsFromMids(all, days), mids: all };
}
function mkSats(N, alt, inc, payload, baseRaan, dR, dP) {
  const sats = [];
  for (let j = 0; j < N; j++) sats.push({
    altitude: alt, inclination: inc, eccentricity: 0.0001, ...payload,
    raan: (baseRaan + j * dR) % 360, argPerigee: (j * dP) % 360,
  });
  return sats;
}
const fmtH = h => h == null ? "—" : h < 48 ? h.toFixed(1) + " h" : (h / 24).toFixed(1) + " d";
const sleep = () => new Promise(r => setTimeout(r, 0));

/* ── minimal UI primitives (match App.js styling) ── */
const Lbl = ({ children }) => <div style={{ fontSize: 9, color: "#3a6888", letterSpacing: 2, textTransform: "uppercase", marginBottom: 6, fontWeight: 600 }}>{children}</div>;
const Th = ({ children, right }) => <th style={{ padding: "5px 8px", fontSize: 8, color: "#3a6888", letterSpacing: 1, textAlign: right ? "right" : "left", borderBottom: "1px solid rgba(70,140,200,0.15)", whiteSpace: "nowrap" }}>{children}</th>;
const Td = ({ children, right, hl }) => <td style={{ padding: "5px 8px", fontSize: 10, color: hl ? "#64FFDA" : "#a0d0e8", textAlign: right ? "right" : "left", borderBottom: "1px solid rgba(70,140,200,0.06)", whiteSpace: "nowrap" }}>{children}</td>;

export default function ConstellationStudy({ countries, onClose }) {
  const [ctyId, setCtyId] = useState("tw");
  const [alt, setAlt] = useState(510);
  const [swath, setSwath] = useState(0.56);
  const [offNadir, setOffNadir] = useState(30);
  const [maxN, setMaxN] = useState(4);
  const [status, setStatus] = useState(null);
  const [prog, setProg] = useState(0);
  const [out, setOut] = useState(null);
  const runningRef = useRef(false);

  const run = async () => {
    if (runningRef.current) return;
    runningRef.current = true;
    setOut(null); setProg(0);
    try {
      const cty = countries.find(c => c.id === ctyId);
      const payload = { swathAngle: +swath || 0.56, offNadir: +offNadir || 0 };
      const A = +alt || 510;
      const phi = Math.abs((cty.latMin + cty.latMax) / 2);

      /* stage 1: single-sat inclination sweep (30 d, ppo 500) */
      const incs = [];
      for (let k = 0; k <= 6; k++) incs.push(Math.min(Math.round(phi) + k, 97));
      incs.push(Math.min(Math.round(phi) + 9, 97), Math.min(Math.round(phi) + 12, 97), 97.4);
      const uniq = [...new Set(incs)];
      const singles = [];
      for (let i = 0; i < uniq.length; i++) {
        setStatus(`Stage 1/2 — inclination sweep ${i + 1}/${uniq.length} (i=${uniq[i]}°)…`);
        setProg(0.4 * (i / uniq.length)); await sleep();
        const base = { altitude: A, inclination: uniq[i], eccentricity: 0.0001, ...payload };
        const raan = computeOptimalRaan(base, cty);
        const { m } = evalConstellation([{ ...base, raan, argPerigee: 0 }], cty, 30, 500);
        singles.push({ inc: uniq[i], ...m });
      }
      const valid = singles.filter(s => s.meanH !== null);
      if (!valid.length) { setStatus("No passes found for any inclination — check target/altitude."); runningRef.current = false; return; }
      const bestSingle = valid.reduce((b, s) => s.meanH < b.meanH ? s : b);
      const bestInc = bestSingle.inc;

      /* stage 2: plane/phasing grid search per fleet size (screen 12 d ppo 200, final 30 d ppo 500) */
      const baseSat = { altitude: A, inclination: bestInc, eccentricity: 0.0001, ...payload };
      const baseRaan = computeOptimalRaan(baseSat, cty);
      const T = orbPeriod(A);
      const fleet = [{ N: 1, dR: 0, dP: 0, ...bestSingle, mids: null }];
      let recMids = null;
      for (let N = 2; N <= Math.min(Math.max(+maxN || 4, 2), 6); N++) {
        const cands = [];
        for (let dR = 0; dR <= 180; dR += 30) for (let dP = 0; dP < 360; dP += 45) cands.push([dR, dP]);
        const tau = 86400 / N, n = 2 * Math.PI / T;
        cands.push([((-OMEGA_E * tau * RAD) % 360 + 360) % 360, ((n * tau * RAD) % 360 + 360) % 360]);
        let best = null;
        for (let ci = 0; ci < cands.length; ci++) {
          const [dR, dP] = cands[ci];
          const { m } = evalConstellation(mkSats(N, A, bestInc, payload, baseRaan, dR, dP), cty, 12, 200);
          if (m.gapH !== null && (best === null || m.gapH < best.m.gapH - 1e-9 ||
              (Math.abs(m.gapH - best.m.gapH) < 1e-9 && m.meanH < best.m.meanH))) best = { dR, dP, m };
          if (ci % 8 === 7) {
            setStatus(`Stage 2/2 — optimizing ${N}-sat constellation: ${ci + 1}/${cands.length} plane/phase configs…`);
            setProg(0.4 + 0.6 * ((N - 2 + ci / cands.length) / (Math.max(+maxN || 4, 2) - 1)));
            await sleep();
          }
        }
        const fin = evalConstellation(mkSats(N, A, bestInc, payload, baseRaan, best.dR, best.dP), cty, 30, 500);
        fleet.push({ N, dR: best.dR, dP: best.dP, inc: bestInc, ...fin.m });
        recMids = fin.mids; // keep largest-N mids; replaced below if N=3 exists
        if (N === 3) fleet.rec3mids = fin.mids;
        await sleep();
      }
      const recN = fleet.some(f => f.N === 3) ? 3 : fleet[fleet.length - 1].N;
      const rec = fleet.find(f => f.N === recN);
      const mids = fleet.rec3mids || recMids;
      setStatus(null); setProg(1);
      setOut({ cty, alt: A, bestInc, singles, fleet: fleet.filter(f => f.N), rec, mids, baseRaan });
    } catch (err) {
      setStatus("Error: " + err.message);
    }
    runningRef.current = false;
  };

  const inpStyle = { width: "100%", background: "rgba(4,8,16,0.9)", border: "1px solid rgba(70,140,200,0.16)", color: "#b0d0e8", padding: "5px 7px", borderRadius: 3, fontSize: 10, fontFamily: "inherit", boxSizing: "border-box" };
  const maxGapAll = out ? Math.max(...out.fleet.map(f => f.gapH || 0), 1) : 1;

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 300, background: "rgba(2,6,14,0.96)", overflowY: "auto", fontFamily: "'IBM Plex Mono','JetBrains Mono','Fira Code',monospace", color: "#a0c4d8", fontSize: 11 }}>
      <div style={{ maxWidth: 980, margin: "0 auto", padding: "18px 22px 60px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#d8ecff", letterSpacing: 3 }}>CONSTELLATION STUDY</div>
            <div style={{ fontSize: 8, color: "#2a5068", letterSpacing: 2 }}>MULTI-PLANE REVISIT OPTIMIZER · MERGED FLEET PASSES · 30-DAY WINDOW</div>
          </div>
          <button onClick={onClose} style={{ background: "none", border: "1px solid rgba(70,140,200,0.2)", color: "#3a6888", borderRadius: 3, padding: "4px 10px", cursor: "pointer", fontFamily: "inherit", fontSize: 10 }}>✕ CLOSE</button>
        </div>

        {/* controls */}
        <div style={{ background: "rgba(8,12,24,0.85)", border: "1px solid rgba(70,140,200,0.12)", borderRadius: 6, padding: 14, marginBottom: 14 }}>
          <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr auto", gap: 10, alignItems: "end" }}>
            <div><Lbl>Target</Lbl>
              <select value={ctyId} onChange={e => setCtyId(e.target.value)} style={inpStyle}>
                {countries.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
              </select>
            </div>
            <div><Lbl>Altitude km</Lbl><input type="number" value={alt} onChange={e => setAlt(e.target.value)} style={inpStyle} /></div>
            <div><Lbl>Swath ½° </Lbl><input type="number" step="0.01" value={swath} onChange={e => setSwath(e.target.value)} style={inpStyle} /></div>
            <div><Lbl>Off-nadir °</Lbl><input type="number" value={offNadir} onChange={e => setOffNadir(e.target.value)} style={inpStyle} /></div>
            <div><Lbl>Max sats</Lbl><input type="number" min="2" max="6" value={maxN} onChange={e => setMaxN(e.target.value)} style={inpStyle} /></div>
            <button onClick={run} style={{ background: "rgba(100,255,218,0.1)", border: "1px solid rgba(100,255,218,0.3)", color: "#64FFDA", padding: "6px 14px", borderRadius: 3, cursor: "pointer", fontSize: 10, fontFamily: "inherit", letterSpacing: 1 }}>
              {status ? "⟳ RUNNING…" : "▶ OPTIMIZE"}
            </button>
          </div>
          {status && <div style={{ marginTop: 10 }}>
            <div style={{ fontSize: 9, color: "#b8a040", marginBottom: 4 }}>{status}</div>
            <div style={{ height: 3, background: "rgba(70,140,200,0.1)", borderRadius: 2 }}>
              <div style={{ height: 3, width: `${Math.round(prog * 100)}%`, background: "#64FFDA", borderRadius: 2, transition: "width .2s" }} />
            </div>
          </div>}
        </div>

        {out && <>
          {/* headline */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10, marginBottom: 14 }}>
            {[["RECOMMENDED FLEET", `${out.rec.N}× i=${out.bestInc}°`, `${out.alt} km · ΔRAAN ${out.rec.dR}° · Δφ ${out.rec.dP}°`],
              ["MEAN REVISIT", fmtH(out.rec.meanH), `${out.rec.passes} passes / 30 d`],
              ["MAX GAP", fmtH(out.rec.gapH), "worst access gap"],
              ["SUNLIT REVISIT", fmtH(out.rec.sunMeanH), `${out.rec.sunlit} sunlit passes / 30 d`],
            ].map(([k, v, d]) => (
              <div key={k} style={{ background: "rgba(8,12,24,0.85)", border: "1px solid rgba(100,255,218,0.15)", borderRadius: 6, padding: "10px 12px" }}>
                <div style={{ fontSize: 8, color: "#3a6888", letterSpacing: 1 }}>{k}</div>
                <div style={{ fontSize: 17, fontWeight: 700, color: "#64FFDA", margin: "3px 0" }}>{v}</div>
                <div style={{ fontSize: 8, color: "#2a5068" }}>{d}</div>
              </div>
            ))}
          </div>

          {/* fleet sizing table + gap bars */}
          <div style={{ background: "rgba(8,12,24,0.85)", border: "1px solid rgba(70,140,200,0.12)", borderRadius: 6, padding: 14, marginBottom: 14 }}>
            <Lbl>Fleet sizing — optimized plane spacing & phasing at i={out.bestInc}° / {out.alt} km over {out.cty.name}</Lbl>
            <table style={{ borderCollapse: "collapse", width: "100%" }}>
              <thead><tr><Th>Sats</Th><Th>ΔRAAN</Th><Th>Δphase</Th><Th right>Passes/30d</Th><Th right>Sunlit</Th><Th right>Mean revisit</Th><Th right>Sunlit mean</Th><Th right>Max gap</Th><Th right>Sunlit gap</Th><Th>Worst gap</Th></tr></thead>
              <tbody>
                {out.fleet.map(f => (
                  <tr key={f.N} style={f.N === out.rec.N ? { background: "rgba(100,255,218,0.04)" } : null}>
                    <Td hl={f.N === out.rec.N}>{f.N}{f.N === out.rec.N ? " ◀ rec" : ""}</Td>
                    <Td>{f.N === 1 ? "—" : f.dR + "°"}</Td><Td>{f.N === 1 ? "—" : f.dP + "°"}</Td>
                    <Td right>{f.passes}</Td><Td right>{f.sunlit}</Td>
                    <Td right>{fmtH(f.meanH)}</Td><Td right>{fmtH(f.sunMeanH)}</Td>
                    <Td right hl={f.N === out.rec.N}>{fmtH(f.gapH)}</Td><Td right>{fmtH(f.sunGapH)}</Td>
                    <Td><div style={{ width: 120, height: 7, background: "rgba(70,140,200,0.08)", borderRadius: 2 }}>
                      <div style={{ width: `${(f.gapH || 0) / maxGapAll * 120}px`, height: 7, background: f.N === out.rec.N ? "#64FFDA" : "#4a90c4", borderRadius: 2 }} />
                    </div></Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* pass-time strip */}
          {out.mids && <div style={{ background: "rgba(8,12,24,0.85)", border: "1px solid rgba(70,140,200,0.12)", borderRadius: 6, padding: 14, marginBottom: 14 }}>
            <Lbl>Pass times — {out.rec.N}-sat fleet, first 10 days (gold = sunlit imaging window)</Lbl>
            <svg viewBox="0 0 900 190" style={{ width: "100%" }}>
              {[0, 6, 12, 18, 24].map(h => <g key={h}>
                <line x1={40} x2={890} y1={165 - h / 24 * 150} y2={165 - h / 24 * 150} stroke="rgba(70,140,200,0.1)" />
                <text x={34} y={168 - h / 24 * 150} fontSize={8} fill="#3a6888" textAnchor="end">{h}h</text>
              </g>)}
              {[...Array(11)].map((_, d) => <text key={d} x={40 + d * 85} y={180} fontSize={8} fill="#3a6888" textAnchor="middle">d{d}</text>)}
              {out.mids.filter(m => m.t < 10 * 86400).map((m, i) => {
                let lst = ((m.t / 3600) + m.lon / 15) % 24; if (lst < 0) lst += 24;
                const sun = isInImagingWindow(m.t, m.lat, m.lon);
                return <circle key={i} cx={40 + (m.t / 86400) * 85} cy={165 - lst / 24 * 150} r={3}
                  fill={sun ? "#FFD740" : "#3a6888"} opacity={0.9} />;
              })}
            </svg>
          </div>}

          {/* inclination sweep */}
          <div style={{ background: "rgba(8,12,24,0.85)", border: "1px solid rgba(70,140,200,0.12)", borderRadius: 6, padding: 14, marginBottom: 14 }}>
            <Lbl>Single-satellite inclination sweep (reference, {out.alt} km)</Lbl>
            <table style={{ borderCollapse: "collapse", width: "100%" }}>
              <thead><tr><Th>Inclination</Th><Th right>Passes/30d</Th><Th right>Sunlit</Th><Th right>Mean revisit</Th><Th right>Sunlit mean</Th><Th right>Max gap</Th></tr></thead>
              <tbody>
                {out.singles.map(s => (
                  <tr key={s.inc} style={s.inc === out.bestInc ? { background: "rgba(100,255,218,0.04)" } : null}>
                    <Td hl={s.inc === out.bestInc}>{s.inc}°{s.inc === 97.4 ? " (SSO)" : ""}{s.inc === out.bestInc ? " ◀ best" : ""}</Td>
                    <Td right>{s.passes}</Td><Td right>{s.sunlit}</Td>
                    <Td right>{fmtH(s.meanH)}</Td><Td right>{fmtH(s.sunMeanH)}</Td><Td right>{fmtH(s.gapH)}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ fontSize: 9, color: "#2a5068", lineHeight: 1.6 }}>
            Relative RAANs for the recommended fleet: {[...Array(out.rec.N)].map((_, j) => ((out.baseRaan + j * out.rec.dR) % 360).toFixed(0) + "°").join(" / ")} (base longitude-matched to {out.cty.name}).
            Same physics as the main analysis; fleet metrics merge passes across planes, which the per-satellite tables cannot show.
            Note: Anchor mode overrides preset RAAN per satellite and would collapse plane spacing — keep it off when simulating these planes as presets.
          </div>
        </>}
      </div>
    </div>
  );
}
