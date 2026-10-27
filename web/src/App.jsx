import React, { useCallback, useEffect, useRef, useState } from "react";

// Dev (Vite :5173) hits the backend on :8000; when served from FastAPI, same-origin.
const API = location.port === "5173" ? "http://localhost:8000" : "";

const fmtTime = (epoch) =>
  new Date((epoch || Date.now() / 1000) * 1000).toLocaleTimeString("en-GB");

// COCO class id -> coarse kind -> outline colour (mirrors config.CLASS_KIND).
const KIND = { 0: "person", 1: "vehicle", 2: "vehicle", 3: "vehicle", 5: "vehicle",
  7: "vehicle", 14: "animal", 15: "animal", 16: "animal", 17: "animal", 18: "animal",
  19: "animal", 20: "animal", 21: "animal", 22: "animal", 23: "animal" };
const KIND_COLOR = { vehicle: "#35D0E0", person: "#FFB020", animal: "#3DDC84" };

// ---- canvas outline helpers ----------------------------------------------
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  if (ctx.roundRect) { ctx.roundRect(x, y, w, h, r); return; }
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// Corner brackets => a "locked-on target" reticle around each tracked car.
function drawBrackets(ctx, x, y, w, h, len, color, lw) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.beginPath();
  const c = [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]];
  for (const [px, py, sx, sy] of c) {
    ctx.moveTo(px, py + sy * len); ctx.lineTo(px, py); ctx.lineTo(px + sx * len, py);
  }
  ctx.stroke();
  ctx.restore();
}

export default function App() {
  const [clips, setClips] = useState([]);
  const [clip, setClip] = useState("demo_crash");
  const [streaming, setStreaming] = useState(false);
  const [vlmLive, setVlmLive] = useState(false);
  const [connMode, setConnMode] = useState("idle"); // sse | polling | idle
  const [alerts, setAlerts] = useState([]);
  const [flash, setFlash] = useState(false);
  const [sms, setSms] = useState(null);
  const [speed, setSpeed] = useState(1);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const tracksRef = useRef(null);
  const crashTracks = useRef(new Set());
  const esRef = useRef(null);
  const pollRef = useRef(null);
  const seenIds = useRef(new Set());

  useEffect(() => {
    fetch(`${API}/clips`)
      .then((r) => r.json())
      .then((d) => { setClips(d.clips || []); setClip(d.default || "demo_crash"); })
      .catch(() => {});
  }, []);

  // ---- crash handling ------------------------------------------------------
  const handleCrash = useCallback((ev) => {
    if (!ev || seenIds.current.has(ev.id)) return;
    seenIds.current.add(ev.id);
    (ev.track_ids || []).forEach((t) => crashTracks.current.add(t));
    setAlerts((a) => [ev, ...a].slice(0, 8));
    setFlash(true);
    setTimeout(() => setFlash(false), 600);
    setSms({ id: ev.id, tag: ev.camera_tag });
  }, []);

  // ---- SSE with polling fallback ------------------------------------------
  const startPolling = useCallback(() => {
    if (pollRef.current) return;
    setConnMode("polling");
    pollRef.current = setInterval(async () => {
      try {
        const st = await fetch(`${API}/status`).then((r) => r.json());
        setStreaming(st.streaming);
        if (st.latest) handleCrash(st.latest);
      } catch (_) {}
    }, 500);
  }, [handleCrash]);

  const connectSSE = useCallback(() => {
    if (esRef.current) esRef.current.close();
    try {
      const es = new EventSource(`${API}/events`);
      es.addEventListener("crash", (e) => handleCrash(JSON.parse(e.data)));
      es.onopen = () => setConnMode("sse");
      es.onerror = () => { es.close(); esRef.current = null; startPolling(); };
      esRef.current = es;
    } catch (_) {
      startPolling();
    }
  }, [handleCrash, startPolling]);

  // ---- overlay: draw tracked boxes synced to video time --------------------
  useEffect(() => {
    let raf;
    const draw = () => {
      const v = videoRef.current, cv = canvasRef.current, T = tracksRef.current;
      if (v && cv && T && v.videoWidth) {
        const { frames, meta } = T;
        const fps = meta.fps || 20;
        const idx = Math.min(frames.length - 1, Math.floor(v.currentTime * fps));
        const snap = frames[idx] || {};
        const rect = v.getBoundingClientRect();
        cv.width = rect.width; cv.height = rect.height;
        // Tracks are in native video pixels; the <video> uses object-fit:contain,
        // so map through the same uniform scale + letterbox offsets it applies.
        const vw = v.videoWidth, vh = v.videoHeight;
        const scale = Math.min(rect.width / vw, rect.height / vh);
        const offX = (rect.width - vw * scale) / 2;
        const offY = (rect.height - vh * scale) / 2;
        const ctx = cv.getContext("2d");
        ctx.clearRect(0, 0, cv.width, cv.height);
        ctx.font = "600 11px ui-monospace, monospace";
        ctx.lineJoin = "round";
        const pulse = 0.5 + 0.5 * Math.sin(v.currentTime * 8); // crash-car throb
        for (const [tid, box] of Object.entries(snap)) {
          const cx = box[0], cy = box[1], w = box[2], h = box[3];
          const kind = KIND[box.length > 4 ? box[4] : 2] || "vehicle";
          const bad = crashTracks.current.has(Number(tid));
          const x = offX + (cx - w / 2) * scale, y = offY + (cy - h / 2) * scale;
          const bw = w * scale, bh = h * scale;
          const color = bad ? "#FF3B3B" : KIND_COLOR[kind];
          const r = Math.min(8, bw / 4, bh / 4);   // rounded car-outline radius

          // translucent body fill (kind-coloured) so each object reads solid
          ctx.save();
          ctx.fillStyle = color;
          ctx.globalAlpha = bad ? 0.16 + 0.20 * pulse : 0.10;
          roundRect(ctx, x, y, bw, bh, r);
          ctx.fill();
          ctx.restore();

          // outline (glowing + throbbing for crash-involved cars)
          ctx.save();
          ctx.strokeStyle = color;
          ctx.lineWidth = bad ? 2.5 : 1.5;
          ctx.shadowColor = color;
          ctx.shadowBlur = bad ? 8 + 12 * pulse : 4;
          roundRect(ctx, x, y, bw, bh, r);
          ctx.stroke();
          ctx.restore();

          // corner-bracket reticle for a tracked-target look
          drawBrackets(ctx, x, y, bw, bh, Math.min(10, bw / 3), color, bad ? 2.5 : 1.5);

          // ID chip
          const label = `#${tid}`;
          const tw = ctx.measureText(label).width;
          ctx.fillStyle = color;
          roundRect(ctx, x, y - 15, tw + 8, 13, 3);
          ctx.fill();
          ctx.fillStyle = "#0B0F17";
          ctx.fillText(label, x + 4, y - 5);
        }
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  // ---- start demo ----------------------------------------------------------
  const startDemo = useCallback(async () => {
    setAlerts([]); crashTracks.current = new Set(); seenIds.current = new Set();
    setSms(null);
    const res = await fetch(`${API}/demo/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clip }),
    }).then((r) => r.json());
    setVlmLive(res.vlm_live); setStreaming(true);
    const rate = res.playback_rate || 1;
    setSpeed(rate);
    const bust = `?v=${Date.now()}`;   // clips can be swapped under the same URL
    tracksRef.current = await fetch(`${API}${res.tracks_url}${bust}`).then((r) => r.json());
    const v = videoRef.current;
    v.src = `${API}${res.video_url}${bust}`;
    v.currentTime = 0;
    v.playbackRate = rate;
    await v.play().catch(() => {});
    connectSSE();
  }, [clip, connectSSE]);

  // live playback-speed control (also re-applied whenever the clip reloads)
  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = speed;
  }, [speed]);

  // Turnkey `make demo`: auto-start clip #1 when served from FastAPI (not Vite dev).
  const autoStarted = useRef(false);
  useEffect(() => {
    if (autoStarted.current || location.port === "5173" || !clips.length) return;
    autoStarted.current = true;
    startDemo();
  }, [clips, startDemo]);

  useEffect(() => () => {
    esRef.current?.close();
    if (pollRef.current) clearInterval(pollRef.current);
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="shield">🛡</span>
          <div>
            <h1>CrashGuard</h1>
            <p>Vision-based highway accident detection</p>
          </div>
        </div>
        <div className="controls">
          <span className={`conn ${connMode}`}>
            {connMode === "sse" ? "● SSE LIVE" : connMode === "polling" ? "● POLLING" : "○ IDLE"}
          </span>
          <span className={`vlm ${vlmLive ? "on" : "off"}`}>
            VLM {vlmLive ? "LIVE" : "CACHED"}
          </span>
          <select value={clip} onChange={(e) => setClip(e.target.value)}>
            {clips.map((c) => (
              <option key={c.name} value={c.name}>{c.name}</option>
            ))}
          </select>
          <select className="speed" value={speed} title="Playback speed"
                  onChange={(e) => setSpeed(parseFloat(e.target.value))}>
            {[0.25, 0.5, 0.75, 1].map((s) => (
              <option key={s} value={s}>{s}×</option>
            ))}
          </select>
          <button className="start" onClick={startDemo}>
            {streaming ? "Restart demo" : "Start demo"}
          </button>
        </div>
      </header>

      <main className="grid">
        <section className={`videocard ${flash ? "flash" : ""}`}>
          <div className="videowrap">
            <video ref={videoRef} muted playsInline />
            <canvas ref={canvasRef} className="overlay" />
            <div className="badge live">● LIVE DETECTION</div>
            <div className="badge cam">{clips.find((c) => c.name === clip)?.camera_tag || "CAM"}</div>
            {!streaming && <div className="idle-hint">Press “Start demo” to begin the feed</div>}
          </div>
          <div className="scanline" />
        </section>

        <aside className="side">
          <PhoneMock sms={sms} />
          <div className="feedhead">
            <h2>Alert feed</h2>
            <span>{alerts.length} event{alerts.length === 1 ? "" : "s"}</span>
          </div>
          <div className="feed">
            {alerts.length === 0 && <div className="empty">No incidents detected yet.</div>}
            {alerts.map((a) => <AlertCard key={a.id} ev={a} />)}
          </div>
        </aside>
      </main>
    </div>
  );
}

function AlertCard({ ev }) {
  const unconfirmed = ev.crash === null || ev.crash === undefined;
  return (
    <div className={`alertcard ${unconfirmed ? "unconfirmed" : ""}`}>
      <div className="bar" />
      <div className="body">
        <div className="head">
          <span className="bell">🔔</span>
          <span className="title">
            {unconfirmed ? "COLLISION — UNCONFIRMED" : "COLLISION DETECTED"}
          </span>
          <span className={`src ${ev.source}`}>{ev.source}</span>
        </div>
        <div className="meta">{fmtTime(ev.timestamp)} • {ev.camera_tag}</div>
        <div className="desc">“{ev.description}”</div>
        <div className="caption">
          — generated automatically · tracks #{(ev.track_ids || []).join(", #")}
        </div>
      </div>
    </div>
  );
}

function PhoneMock({ sms }) {
  return (
    <div className="phone">
      <div className="notch" />
      <div className="screen">
        <div className="sms-head">Highway Patrol · SMS</div>
        {sms ? (
          <div className="bubble in" key={sms.id}>
            <b>SMS SENT ✓</b>
            <span>Collision reported to Highway Patrol.</span>
            <span className="tag">{sms.tag}</span>
          </div>
        ) : (
          <div className="bubble muted">Awaiting incident…</div>
        )}
      </div>
    </div>
  );
}
