"""
simulate.py — render a self-consistent crash-simulation clip for the demo.

Writes data/clips/demo_crash.mp4 and data/tracks/demo_crash.json whose tracks
match the rendered pixels EXACTLY. That gives the stage demo a guaranteed,
honest, end-to-end detected crash (SEE-equivalent tracks -> REASON fires ->
CONFIRM -> ALERT) with zero footage-sourcing or live-API risk. It is clearly a
simulation; the live path handles real judge footage.

CLI:  python -m pipeline.simulate
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess

import cv2
import numpy as np

from config import CLIPS_DIR, TRACKS_DIR

W, H = 960, 540
FPS = 20
N_FRAMES = 190
LANES_Y = [120.0, 230.0, 340.0, 450.0]      # 4 lanes, 110 px apart
CAR_L, CAR_W = 58.0, 28.0                    # overhead car: length(x) x width(y)


def _draw_road(img: np.ndarray) -> None:
    img[:] = (34, 33, 36)                    # dark asphalt (BGR)
    cv2.rectangle(img, (0, 70), (W, 500), (40, 39, 43), -1)     # road slab
    for yb in (70, 500):                     # solid outer edges
        cv2.line(img, (0, yb), (W, yb), (150, 150, 155), 2, cv2.LINE_AA)
    for i in range(1, len(LANES_Y)):         # dashed lane dividers
        yb = int((LANES_Y[i - 1] + LANES_Y[i]) / 2)
        for x in range(0, W, 46):
            cv2.line(img, (x, yb), (x + 26, yb), (90, 90, 96), 2, cv2.LINE_AA)


def _draw_car(img, cx, cy, color, angle=0.0, L=CAR_L, Wd=CAR_W) -> None:
    shadow = cv2.boxPoints(((cx + 3, cy + 5), (L, Wd), angle)).astype(np.int32)
    body = cv2.boxPoints(((cx, cy), (L, Wd), angle)).astype(np.int32)
    cabin = cv2.boxPoints(((cx, cy), (L * 0.42, Wd * 0.72), angle)).astype(np.int32)
    cv2.fillConvexPoly(img, shadow, (18, 18, 20), cv2.LINE_AA)
    cv2.fillConvexPoly(img, body, color, cv2.LINE_AA)
    cv2.fillConvexPoly(img, cabin, tuple(min(255, int(c * 1.3)) for c in color), cv2.LINE_AA)
    cv2.polylines(img, [body], True, (12, 12, 14), 1, cv2.LINE_AA)


def build_scene(seed: int = 5):
    """Simulate car states per frame; return (frames_tracks, colors, crash_span).
    frames_tracks[i] = {id: (cx, cy, w, h)}."""
    rng = random.Random(seed)
    RED, CYAN = (60, 60, 240), (224, 208, 53)
    palette = [(235, 235, 235), (185, 120, 55), (170, 175, 180),
               (90, 175, 110), (150, 150, 235)]
    cars = {
        1: {"x": -70.0, "y": LANES_Y[2], "vx": 6.6, "vy": 0.0, "a": 0.0,
            "mode": "striker", "color": RED},
        2: {"x": 470.0, "y": LANES_Y[2], "vx": 1.3, "vy": 0.0, "a": 0.0,
            "mode": "lead", "color": CYAN},
    }
    # background cruisers in the other lanes
    bid = 3
    for li, ly in enumerate(LANES_Y):
        if li == 2:
            continue
        for k in range(2):
            cars[bid] = {"x": rng.uniform(-200, 700), "y": ly + rng.uniform(-4, 4),
                         "vx": rng.uniform(3.0, 6.0), "vy": 0.0, "a": 0.0,
                         "mode": "cruise", "color": palette[bid % len(palette)]}
            bid += 1

    frames, crash_at = [], None
    for f in range(N_FRAMES):
        for c in cars.values():
            m = c["mode"]
            if m == "striker" and crash_at is None and cars[2]["x"] - c["x"] < 46:
                crash_at = f                                   # impact!
                c["vx"], c["vy"] = 0.5, -2.4                   # stop + veer
                cars[2]["vx"] = 3.6                            # lead lurches
            elif m == "striker" and crash_at is not None:
                c["vx"] = max(0.05, c["vx"] * 0.7)
                c["vy"] *= 0.82
                c["a"] = max(-24.0, c["a"] - 2.6)              # rotate: spun out
            elif m == "lead" and crash_at is not None:
                c["vx"] = max(0.1, c["vx"] * 0.78)
            elif m == "cruise" and c["x"] > W + 70:
                c["x"] = -70.0
            c["x"] += c["vx"]
            c["y"] += c["vy"]
        frames.append({tid: (c["x"], c["y"], CAR_L, CAR_W) for tid, c in cars.items()})

    span = (crash_at, min(crash_at + 30, N_FRAMES - 1)) if crash_at is not None else None
    colors = {tid: c["color"] for tid, c in cars.items()}
    angles = None
    return frames, colors, span


def _to_browser_h264(src: str, dst: str) -> None:
    """Transcode to baseline H.264 / yuv420p (browser-playable). Prefer ffmpeg;
    fall back to OpenCV 'avc1'; last resort keep the source and warn."""
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "baseline",
             "-level", "3.0", "-movflags", "+faststart", dst],
            check=True)
        os.remove(src)
        return
    cap = cv2.VideoCapture(src)
    w = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
    if w.isOpened():
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            w.write(fr)
        w.release(); cap.release(); os.remove(src)
        return
    cap.release(); os.replace(src, dst)
    print("[simulate] WARNING: no ffmpeg/avc1; wrote mp4v which may not play in browsers")


def render() -> tuple[str, str]:
    frames, colors, span = build_scene()
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    mp4 = str(CLIPS_DIR / "demo_crash.mp4")
    js = str(TRACKS_DIR / "demo_crash.json")

    # Render to a temp file, then transcode to browser-safe H.264 (yuv420p) so
    # the dashboard <video> can decode it. cv2's mp4v is NOT playable in Chrome.
    raw = str(CLIPS_DIR / "_demo_crash_raw.mp4")
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    ang = {tid: 0.0 for tid in colors}
    prev = {}
    for i, snap in enumerate(frames):
        img = np.zeros((H, W, 3), np.uint8)
        _draw_road(img)
        # crash flash for a few frames at impact
        if span and span[0] <= i <= span[0] + 4:
            cx, cy, _, _ = snap[1]
            overlay = img.copy()
            cv2.circle(overlay, (int(cx), int(cy)), 46, (40, 130, 255), -1, cv2.LINE_AA)
            img = cv2.addWeighted(overlay, 0.28, img, 0.72, 0)
        for tid, (cx, cy, w, h) in snap.items():
            # striker rotates after impact for a spun-out look
            if span and tid == 1 and i > span[0]:
                ang[1] = max(-26.0, ang[1] - 2.6)
            _draw_car(img, cx, cy, colors[tid], angle=ang[tid])
        # HUD caption so viewers know it's a simulated feed
        cv2.putText(img, "CrashGuard // simulated feed", (16, H - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 155), 1, cv2.LINE_AA)
        writer.write(img)
        prev = snap
    writer.release()
    _to_browser_h264(raw, mp4)

    payload = {
        "meta": {"source": "simulated", "method": "simulated", "fps": float(FPS),
                 "frame_count": len(frames), "width": W, "height": H,
                 "crash_span": list(span) if span else None},
        "frames": [{str(k): [round(x, 2), round(y, 2), w, h]
                    for k, (x, y, w, h) in snap.items()} for snap in frames],
    }
    with open(js, "w") as fp:
        json.dump(payload, fp)
    print(f"[simulate] wrote {mp4} and {js}  crash_span={span} frames={len(frames)}")
    return mp4, js


if __name__ == "__main__":
    render()
