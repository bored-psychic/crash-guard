"""
ingest.py — real-dataset ingestion for CTOL (the "plug in real footage" layer).

Turns a folder of downloaded clips (HWID12, CADP, IEEE smart-city, or your own
recordings) into the cached-trajectory JSON that CTOL consumes. It writes the
SAME on-disk envelope as pipeline.track (so pipeline.track.load_tracks reads
ingest output unchanged), plus a `truth` key carrying the crash spans CTOL
needs:

    {
      "meta":  {"name", "source", "method", "fps",
                "frame_count", "width", "height"},
      "frames": [ {"7": [cx, cy, w, h], ...}, ... ],   # per-frame track dict
      "truth":  [[start_frame, end_frame], ...]         # crash spans ([] = negative)
    }

Design rules carried over from the master spec:
  * ONE SOURCE OF TRUTH — every model name / class list / threshold is imported
    from config.py; nothing is re-hardcoded here.
  * TRACK ONCE, TUNE MANY — YOLO runs once per clip here; CTOL then evaluates
    hundreds of configs against the cached JSON in seconds.
  * NO SINGLE POINT OF FAILURE — if ultralytics/ByteTrack is missing or errors,
    a pure-NumPy greedy-IoU tracker takes over (flagged in meta.method).
  * The detector is PLUGGABLE (detect_fn) so the pipeline is testable without a
    GPU or model weights.

Two trackers coexist by design: pipeline.track is the live single-clip SEE
stage; this is the offline bulk-dataset tracker (adds stride, a frame cap, and
occlusion persistence for hundreds of clips). They share every constant via
config, so there is still one source of truth for the tuning values.

Truth-labeling modes (choose per dataset):
  --truth-mode folder   HWID12-style: parent folder name is the class; names
                        matching --negative-glob get truth=[], crash classes
                        get a whole-clip span (refine with contact_sheet.py —
                        wide spans are fine for recall, refinement sharpens
                        precision).
  --truth-mode none     truth=[] everywhere (normal-traffic packs).
  --labels FILE.json    explicit spans: {"clip_name": [[s,e], ...], ...}.
                        Use `cadp` to produce this from CADP annotations.
  --vlm-label           after tracking, ask the hosted Anthropic vision model
                        (config.VLM_MODEL) to locate the impact window; fail-soft
                        to the default span (needs ANTHROPIC_API_KEY).

CLI (run from the repo root):
  python -m ctol.ingest track  IN_DIR OUT_DIR [--truth-mode folder|none]
                          [--negative-glob "*egative*,*ormal*"]
                          [--labels labels.json] [--vlm-label]
                          [--limit N] [--stride 1] [--max-frames 900]
  python -m ctol.ingest cadp   CADP_ANNOTATIONS.json OUT_labels.json
  python -m ctol.ingest apply  JSON_DIR labels.csv      # clip,start,end rows
  python -m ctol.ingest verify JSON_DIR                 # Phase-1 gate checks

Licensing reminder: HWID12/CADP/etc. are research-and-education only; blur
plates before publishing any frames from them.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import sys
from pathlib import Path

import numpy as np

from config import (IOU_FALLBACK_THRESH, TRACKER_CFG, VEHICLE_CLASSES,
                    VLM_MODEL, VLM_N_FRAMES, YOLO_MODEL)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
_CLASSES = list(VEHICLE_CLASSES)          # ultralytics wants a list


# ───────────────────────── detectors ─────────────────────────
def make_yolo_tracker(model_name: str = YOLO_MODEL):
    """Real path: ultralytics YOLO + built-in ByteTrack. Returns a
    function video_path -> (frames, 'bytetrack') or raises ImportError."""
    from ultralytics import YOLO           # may raise — caller falls back
    model = YOLO(model_name)

    def run(video_path: str, stride: int, max_frames: int):
        frames: list[dict[int, tuple]] = []
        results = model.track(
            source=video_path, stream=True, persist=True,
            classes=_CLASSES, tracker=TRACKER_CFG,
            verbose=False, vid_stride=stride,
        )
        for i, r in enumerate(results):
            if i >= max_frames:
                break
            snap: dict[int, tuple] = {}
            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.int().tolist()
                xywh = r.boxes.xywh.tolist()   # center-x, center-y, w, h
                for tid, (cx, cy, w, h) in zip(ids, xywh):
                    snap[int(tid)] = (float(cx), float(cy),
                                      float(w), float(h))
            frames.append(snap)
        return frames, "bytetrack"
    return run


class IoUTracker:
    """Greedy frame-to-frame IoU matcher — the dependency-free fallback.
    Not ByteTrack-grade through occlusion, but keeps IDs stable enough
    for trajectory physics on clean highway views."""

    def __init__(self, iou_thresh: float = IOU_FALLBACK_THRESH,
                 max_missed: int = 8):
        self.iou_thresh, self.max_missed = iou_thresh, max_missed
        self.next_id = 1
        self.tracks: dict[int, dict] = {}   # id -> {box, missed}

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = ((ax2 - ax1) * (ay2 - ay1) +
              (bx2 - bx1) * (by2 - by1) - inter)
        return inter / ua

    def update(self, boxes_xyxy: list[tuple]) -> dict[int, tuple]:
        pairs = []
        for tid, t in self.tracks.items():
            for j, b in enumerate(boxes_xyxy):
                iou = self._iou(t["box"], b)
                if iou >= self.iou_thresh:
                    pairs.append((iou, tid, j))
        pairs.sort(reverse=True)
        used_t, used_d, assign = set(), set(), {}
        for iou, tid, j in pairs:                    # greedy best-first
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid); used_d.add(j)
            assign[j] = tid
        for j, b in enumerate(boxes_xyxy):           # births
            if j not in used_d:
                assign[j] = self.next_id
                self.tracks[self.next_id] = {"box": b, "missed": 0}
                self.next_id += 1
        for j, tid in assign.items():                # refresh matched
            self.tracks[tid] = {"box": boxes_xyxy[j], "missed": 0}
        for tid in list(self.tracks):                # age & kill stale
            if tid not in assign.values():
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]
        out = {}
        for j, tid in assign.items():
            x1, y1, x2, y2 = boxes_xyxy[j]
            out[tid] = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        return out


def make_fallback_tracker(detect_fn=None):
    """Fallback path: any per-frame detector + IoUTracker.
    detect_fn(frame_bgr) -> list[(x1,y1,x2,y2)]. If detect_fn is None,
    tries ultralytics detection-only (no track mode); if that import
    fails too, raises RuntimeError (nothing left to detect with)."""
    if detect_fn is None:
        from ultralytics import YOLO
        model = YOLO(YOLO_MODEL)

        def detect_fn(frame):
            r = model.predict(frame, classes=_CLASSES, verbose=False)[0]
            return [tuple(map(float, b)) for b in r.boxes.xyxy.tolist()]

    def run(video_path: str, stride: int, max_frames: int):
        import cv2
        cap = cv2.VideoCapture(video_path)
        tracker = IoUTracker()
        frames = []
        i = 0
        while cap.isOpened() and len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                frames.append(tracker.update(detect_fn(frame)))
            i += 1
        cap.release()
        return frames, "iou-fallback"
    return run


# ───────────────────────── truth labeling ─────────────────────────
def truth_from_folder(video: Path, n_frames: int,
                      negative_globs: list[str]) -> list[list[int]]:
    cls = video.parent.name.lower()
    for g in negative_globs:
        if fnmatch.fnmatch(cls, g.lower()):
            return []
    return [[0, max(0, n_frames - 1)]]     # whole-clip span; refine later


def cadp_convert(annotations_path: str, out_path: str) -> dict:
    """CADP ships spatio-temporal annotations for its 205 annotated
    segments. Schemas in the wild vary by release, so this converter is
    deliberately tolerant: it walks the JSON and, per video key, extracts
    every frame index it can find, then emits [min, max] as the span.
    Inspect the printout — if your download's schema nests differently,
    this one function is the only thing to tweak."""
    with open(annotations_path) as f:
        data = json.load(f)

    def frame_numbers(node) -> list[int]:
        out = []
        if isinstance(node, dict):
            for k, v in node.items():
                lk = str(k).lower()
                if lk in ("frame", "frame_id", "frame_idx", "keyframe",
                          "start", "end", "start_frame", "end_frame",
                          "time_to_accident_frame"):
                    try:
                        out.append(int(float(v)))
                    except (TypeError, ValueError):
                        pass
                else:
                    out += frame_numbers(v)
        elif isinstance(node, list):
            for v in node:
                out += frame_numbers(v)
        return out

    labels = {}
    for vid_key, node in (data.items() if isinstance(data, dict) else []):
        fr = [n for n in frame_numbers(node) if n >= 0]
        if fr:
            labels[Path(str(vid_key)).stem] = [[min(fr), max(fr)]]
    with open(out_path, "w") as f:
        json.dump(labels, f, indent=1)
    print(f"cadp_convert: {len(labels)} labeled videos -> {out_path}")
    return labels


def vlm_label(video_path: str, n_frames: int,
              k: int = max(8, VLM_N_FRAMES)) -> list[list[int]] | None:
    """Optional auto-labeler: sample k frames, ask the hosted vision model
    which sampled range contains the impact. Fail-soft: returns None on any
    error and the caller keeps its default span."""
    try:
        import base64
        import os

        import anthropic
        import cv2
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        cap = cv2.VideoCapture(video_path)
        idxs = np.linspace(0, n_frames - 1, k).astype(int).tolist()
        content = []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (512, 288))
            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 70])
            content.append({"type": "text", "text": f"frame_index={fi}"})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.b64encode(buf).decode()}})
        cap.release()
        content.append({"type": "text", "text":
            'These are ordered samples from one traffic clip. If a '
            'vehicle collision occurs, reply ONLY with JSON '
            '{"start": <frame_index>, "end": <frame_index>} using the '
            'given indices (impact window). If no collision: {"none": true}.'})
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=VLM_MODEL, max_tokens=200,
            messages=[{"role": "user", "content": content}])
        txt = resp.content[0].text.strip()
        txt = txt.replace("```json", "").replace("```", "").strip()
        obj = json.loads(txt)
        if obj.get("none"):
            return []
        s, e = int(obj["start"]), int(obj["end"])
        if 0 <= s <= e < n_frames:
            return [[s, e]]
    except Exception:
        return None
    return None


# ───────────────────────── track command ─────────────────────────
def _probe(video: str) -> tuple[float, int, int]:
    """(fps, width, height) for the meta block."""
    import cv2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return float(fps), w, h


def cmd_track(args) -> None:
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(p for p in in_dir.rglob("*")
                    if p.suffix.lower() in VIDEO_EXTS)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        sys.exit(f"no videos found under {in_dir}")

    explicit = {}
    if args.labels:
        with open(args.labels) as f:
            explicit = json.load(f)

    try:
        runner = make_yolo_tracker(args.model)
        print("tracker: ultralytics ByteTrack")
    except Exception as e:                        # noqa: BLE001
        print(f"ultralytics unavailable ({e.__class__.__name__}: {e}); "
              f"using IoU fallback")
        runner = make_fallback_tracker()

    neg = [g.strip() for g in args.negative_glob.split(",") if g.strip()]
    for vi, video in enumerate(videos):
        name = video.stem
        out_path = out_dir / f"{name}.json"
        if out_path.exists() and not args.overwrite:
            print(f"[{vi+1}/{len(videos)}] {name}: exists, skip")
            continue
        try:
            frames, tracker_used = runner(str(video), args.stride,
                                          args.max_frames)
        except Exception as e:                    # noqa: BLE001
            print(f"[{vi+1}/{len(videos)}] {name}: TRACK FAILED ({e})")
            continue
        n = len(frames)

        if name in explicit:
            truth = explicit[name]
        elif args.truth_mode == "folder":
            truth = truth_from_folder(video, n, neg)
        else:
            truth = []
        if args.vlm_label and truth != []:
            got = vlm_label(str(video), n)
            if got is not None:
                truth = got

        fps, width, height = _probe(str(video))
        clip = {"meta": {"name": name, "source": str(video),
                         "method": tracker_used, "fps": fps,
                         "frame_count": n, "width": width, "height": height},
                "frames": [{str(k): list(v) for k, v in snap.items()}
                           for snap in frames],
                "truth": truth}
        with open(out_path, "w") as f:
            json.dump(clip, f)
        n_tracks = len({tid for s in frames for tid in s})
        print(f"[{vi+1}/{len(videos)}] {name}: {n} frames, "
              f"{n_tracks} tracks, truth={truth} ({tracker_used})")


# ───────────────────────── apply / verify / load ─────────────────────────
def _clip_name(clip: dict, path: Path) -> str:
    return clip.get("name") or clip.get("meta", {}).get("name") or path.stem


def cmd_apply(args) -> None:
    json_dir = Path(args.json_dir)
    spans: dict[str, list[list[int]]] = {}
    with open(args.csv) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            name, s, e = row[0].strip(), int(row[1]), int(row[2])
            spans.setdefault(name, []).append([s, e])
    for name, tr in spans.items():
        p = json_dir / f"{name}.json"
        if not p.exists():
            print(f"apply: {name}.json not found, skip")
            continue
        clip = json.loads(p.read_text())
        clip["truth"] = tr
        p.write_text(json.dumps(clip))
        print(f"apply: {name} truth={tr}")


def cmd_verify(args) -> None:
    """The Phase-1 gate from the master prompt, as a command. Reads both the
    ingest envelope and pipeline.track output (name/method fall back to meta,
    truth defaults to [])."""
    json_dir = Path(args.json_dir)
    files = sorted(json_dir.glob("*.json"))
    if not files:
        sys.exit(f"no clip JSONs in {json_dir}")
    bad = 0
    for p in files:
        clip = json.loads(p.read_text())
        meta = clip.get("meta", {})
        frames = clip["frames"]
        truth = clip.get("truth", [])
        method = clip.get("tracker") or meta.get("method") or "?"
        lifespans: dict[str, int] = {}
        for snap in frames:
            for tid in snap:
                lifespans[tid] = lifespans.get(tid, 0) + 1
        persistent = sum(1 for v in lifespans.values() if v >= 30)
        flicker = sum(1 for v in lifespans.values() if v < 5)
        total = max(1, len(lifespans))
        ok = persistent >= 1 and flicker / total <= 0.5
        bad += (not ok)
        print(f"{_clip_name(clip, p):<28} frames={len(frames):<5} "
              f"tracks={total:<4} persistent(>=30f)={persistent:<3} "
              f"flicker(<5f)={flicker:<4} truth={truth} "
              f"[{method}] {'OK' if ok else 'CHECK'}")
    print(f"\n{len(files) - bad}/{len(files)} clips pass the gate")


def load_labeled_clips(json_dir: str):
    """Import-side loader: JSON dir -> list[LabeledClip] for CTOL.
        from ctol.ingest import load_labeled_clips
        clips = load_labeled_clips('data/tracks')
        asyncio.run(ctol(clips, ...))
    """
    from ctol.evaluator import LabeledClip
    out = []
    for p in sorted(Path(json_dir).glob("*.json")):
        c = json.loads(p.read_text())
        frames = [{int(k): tuple(v) for k, v in snap.items()}
                  for snap in c["frames"]]
        truth = [tuple(t) for t in c.get("truth", [])]
        out.append(LabeledClip(name=_clip_name(c, p), frames=frames,
                               truth=truth))
    return out


# ───────────────────────── main ─────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m ctol.ingest",
                                 description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("track", help="videos -> cached-trajectory JSONs")
    t.add_argument("in_dir"); t.add_argument("out_dir")
    t.add_argument("--model", default=YOLO_MODEL)
    t.add_argument("--truth-mode", choices=["folder", "none"],
                   default="none")
    t.add_argument("--negative-glob", default="*egative*,*ormal*,*o?accident*")
    t.add_argument("--labels", default=None,
                   help="explicit spans JSON {name: [[s,e],...]}")
    t.add_argument("--vlm-label", action="store_true")
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--stride", type=int, default=1)
    t.add_argument("--max-frames", type=int, default=900)
    t.add_argument("--overwrite", action="store_true")
    t.set_defaults(fn=cmd_track)

    c = sub.add_parser("cadp", help="CADP annotations -> labels.json")
    c.add_argument("annotations"); c.add_argument("out")
    c.set_defaults(fn=lambda a: cadp_convert(a.annotations, a.out))

    a = sub.add_parser("apply", help="labels.csv (name,start,end) -> JSONs")
    a.add_argument("json_dir"); a.add_argument("csv")
    a.set_defaults(fn=cmd_apply)

    v = sub.add_parser("verify", help="Phase-1 gate checks on a JSON dir")
    v.add_argument("json_dir")
    v.set_defaults(fn=cmd_verify)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
