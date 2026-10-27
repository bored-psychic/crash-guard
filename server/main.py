"""
main.py — PHASE 5: FastAPI backend (the ALERT stage).

Endpoints:
  POST /demo/start {clip}  -> begin realtime replay of a cached clip
  GET  /events             -> SSE stream of confirmed crash events
  GET  /status             -> latest state (polling fallback for /events)
  GET  /clip/{name}        -> the demo video file (for the dashboard <video>)
  GET  /tracks/{name}      -> cached trajectories (for the client overlay)
  GET  /clips              -> available demo clips

On /demo/start the server drives process_stream over the cached clip in real
time; each confirmed event lands on an outbound queue, is fanned out to all SSE
subscribers, and is remembered for /status. The frame loop never awaits the VLM
(Phase 4), so a slow/absent VLM never stalls the stream — events still ship,
marked "unconfirmed".
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from config import (CACHE_DIR, CLIPS_DIR, DEFAULT_CONFIG, DEMO_CLIPS,
                    DEMO_DEFAULT_CLIP, ROOT, TRACKS_DIR, anthropic_key,
                    crash_model_url)
from pipeline.reason import gated_confirm
from pipeline.stream import process_stream, replay_source
from pipeline.track import load_tracks
from pipeline.vlm import confirm as vlm_confirm

app = FastAPI(title="CrashGuard")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


# ---------------------------------------------------------------------------
class Hub:
    """Fan-out of confirmed events to SSE subscribers + latest-state for polling."""

    def __init__(self) -> None:
        self.subscribers: set[asyncio.Queue] = set()
        self.latest: dict | None = None
        self.history: list[dict] = []
        self.streaming = False
        self.clip: str | None = None
        self.task: asyncio.Task | None = None

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)

    async def publish(self, ce_json: dict) -> None:
        self.latest = ce_json
        self.history.append(ce_json)
        for q in list(self.subscribers):
            q.put_nowait(ce_json)


hub = Hub()


def _load_precache(name: str) -> dict:
    """Pre-cached confirmations keyed by event frame_idx -> {crash, description}."""
    p = CACHE_DIR / f"{name}.json"
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text())
    return {int(k): v for k, v in raw.items()}


async def _run_demo(name: str) -> None:
    clip = DEMO_CLIPS[name]
    video = str(CLIPS_DIR / clip["video"])
    tracks = str(TRACKS_DIR / clip["tracks"])
    _frames, meta = load_tracks(tracks)
    fps = float(meta.get("fps", 20.0))
    cache = _load_precache(name)

    out_q: asyncio.Queue = asyncio.Queue()
    hub.streaming, hub.clip = True, name

    async def pump() -> None:
        while True:
            ce = await out_q.get()
            await hub.publish(ce.to_json())

    # REASON v2 opt-in: with CRASH_MODEL_URL set, the hosted crash model screens
    # every gate candidate before the VLM (pipeline/reason.py). Without it,
    # confirm_fn is the plain VLM and behavior is unchanged.
    confirm_fn = gated_confirm() if crash_model_url() else vlm_confirm

    pump_task = asyncio.create_task(pump())
    try:
        async for _f, _t, _e in process_stream(
            replay_source(video, tracks), cfg=clip.get("config", DEFAULT_CONFIG),
            camera_tag=clip.get("camera_tag"), out_queue=out_q,
            cache=cache, fps=fps, realtime=True, confirm_fn=confirm_fn,
        ):
            pass
        await asyncio.sleep(0.4)          # let final confirmations flush
    finally:
        pump_task.cancel()
        hub.streaming = False


# ---------------------------------------------------------------------------
class DemoStart(BaseModel):
    clip: str = DEMO_DEFAULT_CLIP


@app.post("/demo/start")
async def demo_start(body: DemoStart | None = None):
    name = (body or DemoStart()).clip
    if name not in DEMO_CLIPS:
        raise HTTPException(404, f"unknown clip '{name}'")
    if hub.task and not hub.task.done():
        hub.task.cancel()
    hub.history, hub.latest = [], None
    hub.task = asyncio.create_task(_run_demo(name))
    return {"ok": True, "clip": name, "camera_tag": DEMO_CLIPS[name].get("camera_tag"),
            "video_url": f"/clip/{name}", "tracks_url": f"/tracks/{name}",
            "playback_rate": DEMO_CLIPS[name].get("playback_rate", 1.0),
            "vlm_live": bool(anthropic_key())}


@app.get("/events")
async def events(request: Request):
    async def gen():
        q = hub.subscribe()
        try:
            while True:
                if await request.is_disconnected():
                    break
                ce = await q.get()
                yield {"event": "crash", "data": json.dumps(ce)}
        finally:
            hub.unsubscribe(q)

    return EventSourceResponse(gen(), ping=15)


@app.get("/status")
async def status():
    return {"streaming": hub.streaming, "clip": hub.clip,
            "latest": hub.latest, "events": hub.history,
            "vlm_live": bool(anthropic_key()),
            "model_live": bool(crash_model_url())}


# no-cache so swapping a clip's underlying file is always reflected (the URL
# stays /clip/<name> and /tracks/<name>, so a stale cache would desync overlays).
_NOCACHE = {"Cache-Control": "no-cache"}


@app.get("/clip/{name}")
async def clip_file(name: str):
    if name not in DEMO_CLIPS:
        raise HTTPException(404, "unknown clip")
    return FileResponse(str(CLIPS_DIR / DEMO_CLIPS[name]["video"]),
                        media_type="video/mp4", headers=_NOCACHE)


@app.get("/tracks/{name}")
async def clip_tracks(name: str):
    if name not in DEMO_CLIPS:
        raise HTTPException(404, "unknown clip")
    return FileResponse(str(TRACKS_DIR / DEMO_CLIPS[name]["tracks"]),
                        media_type="application/json", headers=_NOCACHE)


@app.get("/clips")
async def clips():
    return {"clips": [{"name": k, "camera_tag": v.get("camera_tag")}
                      for k, v in DEMO_CLIPS.items()],
            "default": DEMO_DEFAULT_CLIP}


# Serve the built dashboard at / when present (so `make demo` is one URL).
_DIST = ROOT / "web" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="web")
