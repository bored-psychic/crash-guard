"""
track.py — PHASE 1: video -> cached trajectory JSON (the SEE stage).

YOLOv8n detects vehicles per frame; ByteTrack (via ultralytics
model.track(persist=True)) assigns persistent track IDs across frames,
producing real trajectories. The output is cached to JSON exactly once per
clip — CTOL and every replay then consume the cached trajectories and never
re-run YOLO ("track once, tune many").

Graceful degradation (no single point of failure): if the ByteTrack path
raises for any reason, we fall back to greedy IoU (threshold 0.3)
frame-to-frame matching over YOLO detections, and record which path was used.

On-disk format:
    {"meta": {"source","method","fps","frame_count","width","height"},
     "frames": [ {"<track_id>": [cx, cy, w, h], ...}, ... ]}

In-memory (the universal interchange format):
    frames: list[dict[int, tuple[float, float, float, float]]]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

from config import (DETECT_CLASSES, DETECT_CONF, IOU_FALLBACK_THRESH, STREAM_FPS,
                    TRACKER_CFG, YOLO_IMGSZ, YOLO_MODEL)

Frames = list[dict[int, tuple[float, float, float, float]]]


# ---------------------------------------------------------------------------
def _iou(a, b) -> float:
    """IoU of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def _probe_video(path: str) -> tuple[float, int, int]:
    """(fps, width, height). Falls back to STREAM_FPS if unreadable."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or STREAM_FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (float(fps) if fps and fps > 0 else STREAM_FPS), w, h


# ---------------------------------------------------------------------------
def _track_bytetrack(model, path: str, imgsz: int = YOLO_IMGSZ) -> Frames:
    """Primary path: ByteTrack via ultralytics. Persistent IDs come from the
    tracker itself. Raises on any backend failure so the caller can fall back."""
    frames: Frames = []
    results = model.track(
        source=path, persist=True, tracker=TRACKER_CFG,
        classes=list(DETECT_CLASSES), conf=DETECT_CONF, imgsz=imgsz,
        stream=True, verbose=False,
    )
    for r in results:
        snap: dict[int, tuple] = {}
        b = r.boxes
        if b is not None and b.id is not None:
            xywh = b.xywh.cpu().numpy()          # [cx, cy, w, h]
            ids = b.id.int().cpu().numpy()
            cls = b.cls.int().cpu().numpy()
            for (cx, cy, w, h), tid, c in zip(xywh, ids, cls):
                snap[int(tid)] = (float(cx), float(cy), float(w), float(h), int(c))
        frames.append(snap)
    if not frames:
        raise RuntimeError("ByteTrack produced no frames")
    return frames


def _track_iou(model, path: str, imgsz: int = YOLO_IMGSZ) -> Frames:
    """Fallback: detection-only YOLO + greedy IoU frame-to-frame matching to
    assign persistent IDs. Dependency-free of any tracker backend."""
    frames: Frames = []
    prev: dict[int, tuple] = {}     # id -> (x1,y1,x2,y2) from previous frame
    next_id = 1
    results = model.predict(
        source=path, classes=list(DETECT_CLASSES), conf=DETECT_CONF,
        imgsz=imgsz, stream=True, verbose=False,
    )
    for r in results:
        dets = []
        b = r.boxes
        if b is not None and len(b):
            xyxy = b.xyxy.cpu().numpy()
            xywh = b.xywh.cpu().numpy()
            cls = b.cls.int().cpu().numpy()
            dets = list(zip(xyxy, xywh, cls))

        snap: dict[int, tuple] = {}
        new_prev: dict[int, tuple] = {}
        used: set[int] = set()
        for box_xyxy, (cx, cy, w, h), c in dets:
            best_id, best_iou = None, IOU_FALLBACK_THRESH
            for pid, pbox in prev.items():
                if pid in used:
                    continue
                iou = _iou(box_xyxy, pbox)
                if iou >= best_iou:
                    best_iou, best_id = iou, pid
            if best_id is None:                  # unmatched detection => new track
                best_id, next_id = next_id, next_id + 1
            used.add(best_id)
            snap[int(best_id)] = (float(cx), float(cy), float(w), float(h), int(c))
            new_prev[int(best_id)] = tuple(map(float, box_xyxy))
        prev = new_prev
        frames.append(snap)
    return frames


# ---------------------------------------------------------------------------
def track_video(path: str, out_json: str, imgsz: int = YOLO_IMGSZ) -> None:
    """Run YOLOv8n + ByteTrack over `path` and dump per-frame
    {track_id: (cx, cy, w, h)} to `out_json` with fps + frame_count metadata.
    `imgsz` is the inference resolution (raise it for small/distant vehicles).
    Falls back to greedy IoU matching if the tracking backend fails."""
    from ultralytics import YOLO  # imported here so importing the module is cheap

    if not Path(path).is_file():
        raise FileNotFoundError(f"clip not found: {path}")
    fps, width, height = _probe_video(path)
    model = YOLO(YOLO_MODEL)

    try:
        frames = _track_bytetrack(model, path, imgsz)
        method = "bytetrack"
    except Exception as e:  # tracker import/config/runtime failure -> degrade
        print(f"[track] ByteTrack path failed ({e!r}); falling back to IoU matching")
        frames = _track_iou(model, path, imgsz)
        method = "iou_fallback"

    meta = {"source": path, "method": method, "fps": fps,
            "frame_count": len(frames), "width": width, "height": height}
    payload = {
        "meta": meta,
        "frames": [{str(k): list(v) for k, v in snap.items()} for snap in frames],
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f)

    n_tracks = len({tid for snap in frames for tid in snap})
    print(f"[track] method={method} frames={len(frames)} fps={fps:.1f} "
          f"unique_tracks={n_tracks} -> {out_json}")


def load_tracks(json_path: str) -> tuple[Frames, dict]:
    """Load a cached trajectory JSON back into (frames, meta). Consumers
    (physics, CTOL, stream) use this — they never re-run YOLO."""
    with open(json_path) as f:
        payload = json.load(f)
    frames: Frames = [
        {int(k): tuple(v) for k, v in snap.items()} for snap in payload["frames"]
    ]
    return frames, payload.get("meta", {})


# ---------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        print("usage: python -m pipeline.track <in_video> <out_json> [imgsz]")
        return 2
    imgsz = int(argv[3]) if len(argv) == 4 else YOLO_IMGSZ
    track_video(argv[1], argv[2], imgsz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
