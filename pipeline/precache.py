"""
precache.py — generate VLM confirmations for demo clips, keyed by event frame,
so the stage demo has zero live-API risk (DEMO requirement).

For each registered demo clip: run the physics detector over the cached tracks,
gather the ~2 s window around each event from the video, call the real VLM, and
write data/vlm_cache/<clip>.json = { "<frame_idx>": {crash, description} }.

Clips whose tracks were rendered by the simulator (meta.method == "simulated")
are skipped: a VLM can't meaningfully analyze schematic boxes, so their
hand-authored cache is kept. Real footage (and judge clips) use the live VLM.

CLI:  ANTHROPIC_API_KEY=... python -m pipeline.precache [clip ...]
"""

from __future__ import annotations

import asyncio
import json
import sys

import cv2

from config import (CACHE_DIR, CLIPS_DIR, DEFAULT_CONFIG, DEMO_CLIPS,
                    EVENT_WINDOW_SEC, TRACKS_DIR, VLM_N_FRAMES, anthropic_key)
from pipeline.physics import run_detector
from pipeline.stream import _buffer_frame, encode_jpeg
from pipeline.track import load_tracks
from pipeline.vlm import _sample, confirm


async def precache_clip(name: str) -> dict:
    clip = DEMO_CLIPS[name]
    video = str(CLIPS_DIR / clip["video"])
    frames, meta = load_tracks(str(TRACKS_DIR / clip["tracks"]))
    if meta.get("method") == "simulated":
        print(f"[precache] {name}: simulated clip — keeping hand-authored cache")
        return {}
    events = run_detector(frames, DEFAULT_CONFIG)
    if not events:
        print(f"[precache] {name}: no physics events")
        return {}

    fps = float(meta.get("fps", 20.0))
    win = max(VLM_N_FRAMES, int(EVENT_WINDOW_SEC * fps))
    cap = cv2.VideoCapture(video)
    out: dict[int, dict] = {}
    for ev in events:
        start = max(0, ev.frame_idx - win + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        buf = []
        for _ in range(start, ev.frame_idx + 1):
            ok, fr = cap.read()
            if not ok:
                break
            buf.append(_buffer_frame(fr))
        window = [encode_jpeg(f) for f in _sample(buf, VLM_N_FRAMES)]
        conf = await confirm(ev, window)
        out[ev.frame_idx] = {"crash": conf.crash, "description": conf.description}
        print(f"[precache] {name} @frame {ev.frame_idx}: {conf.source} "
              f"crash={conf.crash} :: {conf.description}")
    cap.release()
    return out


async def main(argv: list[str]) -> int:
    if not anthropic_key():
        print("[precache] No ANTHROPIC_API_KEY — keeping existing hand-authored "
              "caches. Add the key to .env to generate real VLM confirmations.")
        return 0
    names = argv[1:] or list(DEMO_CLIPS)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in DEMO_CLIPS:
            print(f"[precache] unknown clip {name}")
            continue
        out = await precache_clip(name)
        if out:
            with open(CACHE_DIR / f"{name}.json", "w") as f:
                json.dump({str(k): v for k, v in out.items()}, f, indent=2)
            print(f"[precache] wrote {CACHE_DIR / f'{name}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv)))
