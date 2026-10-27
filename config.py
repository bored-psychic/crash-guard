"""
config.py — CrashGuard single source of truth.

Every threshold, model name, path, bound, and theme color lives here so no
module carries magic numbers (a hard project rule). Import from here.

Also loads a local `.env` (if present) so ANTHROPIC_API_KEY reaches both the
CLI scripts and the server without a hard python-dotenv dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# .env loader (dependency-free). Reads KEY=VALUE lines from files/.env into
# os.environ WITHOUT overwriting already-exported vars. Keeps the Anthropic
# key out of source and lets `python -m ...` and uvicorn pick it up for free.
# ---------------------------------------------------------------------------
def load_dotenv(path: str | os.PathLike | None = None) -> None:
    p = Path(path) if path else Path(__file__).resolve().parent / ".env"
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


load_dotenv()


def anthropic_key() -> str | None:
    """The hosted-API credential, or None => callers degrade gracefully."""
    return os.environ.get("ANTHROPIC_API_KEY")


def crash_model_url() -> str | None:
    """HTTPS endpoint of the hosted crash-classification model (Modal — see
    modal_app.py), or None => the REASON stage runs gate-only and callers
    degrade gracefully, exactly like a missing Anthropic key."""
    return os.environ.get("CRASH_MODEL_URL")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
CLIPS_DIR = DATA_DIR / "clips"
TRACKS_DIR = DATA_DIR / "tracks"         # cached trajectory JSON ({meta,frames,truth?})
DATASETS_DIR = DATA_DIR / "datasets"     # drop real datasets here for CTOL ingestion
LABELS_DIR = DATA_DIR / "labels"         # human truth-span inputs (labels.csv / *.json)
CACHE_DIR = DATA_DIR / "vlm_cache"       # pre-cached demo confirmations


# ---------------------------------------------------------------------------
# The tunable physics contract. frozen => hashable => memoizable, and safe to
# ship across process boundaries (CTOL evaluates many of these in parallel).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    speed_thresh: float   # px/frame: "was moving fast"          (bounds 1-20)
    stop_thresh: float    # px/frame: "is now stopped" (<speed)  (0.1-5)
    window: int           # rolling history frames per track     (6-30)
    diverge_deg: float    # abrupt heading change, degrees        (10-90)
    prox_factor: float    # contact = box-edge gap within          (1-5)
                          # (prox_factor-1)*box_size; 1=edges touch
    # px/frame^2: a single-frame deceleration this sharp, landing at a stop, is
    # an IMPACT IMPULSE — catches LOW-SPEED collisions the absolute speed gate
    # misses, while a smooth brake-to-stop (low jerk) stays silent. (0.5-10)
    impact_decel: float = 1.5

    def as_dict(self) -> dict:
        return {
            "speed_thresh": self.speed_thresh,
            "stop_thresh": self.stop_thresh,
            "window": self.window,
            "diverge_deg": self.diverge_deg,
            "prox_factor": self.prox_factor,
            "impact_decel": self.impact_decel,
        }


# Search-space bounds — physical guardrails that also sanitize LLM proposals.
BOUNDS: dict[str, tuple[float, float]] = {
    "speed_thresh": (1.0, 20.0),
    "stop_thresh": (0.1, 5.0),
    "window": (6, 30),
    "diverge_deg": (10.0, 90.0),
    "prox_factor": (1.0, 5.0),
    "impact_decel": (0.5, 10.0),
}

# Hand-picked sane default: server's live detector + a reasonable seed point.
DEFAULT_CONFIG = Config(
    speed_thresh=3.0, stop_thresh=1.0, window=12, diverge_deg=35.0, prox_factor=2.0
)

# Per-camera calibration for the aerial highway feeds: only a genuine speed
# collapse fires, so normal merging / slowing at an interchange doesn't
# false-alarm. This is the kind of per-deployment config CTOL produces.
HIGHWAY_STRICT_CONFIG = Config(
    speed_thresh=6.0, stop_thresh=0.5, window=20, diverge_deg=50.0, prox_factor=1.5,
    # dense aerial feed: only a violent single-frame stop counts as an impulse,
    # so normal hard braking behind slower traffic never false-alarms.
    impact_decel=6.0,
)

# Junction cam: a real low-speed collision — a car strikes the vehicle ahead and
# stops askew (~5.9s, verified against the burned-in CCTV clock 11:29:32). The
# scene is congested (median 0.2 px/frame), so this is tuned to catch that
# genuine speed-collapse on the approaching vehicle (track 7) while staying quiet
# through the normal stop-and-go before it; the clip is trimmed to the aftermath
# so it fires exactly once.
JUNCTION_CONFIG = Config(
    speed_thresh=3.0, stop_thresh=0.3, window=20, diverge_deg=35.0, prox_factor=1.2,
    # re-tuned after pedestrians/animals were added to tracking: the tighter
    # prox (edge-contact) + longer window keep it to one fire on the real
    # collision (~5.9s) despite the now-tracked pedestrians milling in-scene.
    impact_decel=2.0,
)

# Per-clip calibrations for the FULL real-incident clips below — each tuned so
# the detector fires exactly once over the whole clip (no trimming).
INTERSECTION_CONFIG = Config(3.0, 0.3, 12, 35.0, 1.3, 3.0)  # moto vs van ~10.7s
ROUNDABOUT_CONFIG   = Config(8.0, 0.3, 12, 35.0, 1.3, 9.0)  # roundabout exit ~14.5s
OVERTURN_CONFIG     = Config(5.0, 0.3, 20, 35.0, 1.3, 6.0)  # snowy street ~37.4s


# ---------------------------------------------------------------------------
# SEE stage — YOLOv8n + ByteTrack
# ---------------------------------------------------------------------------
YOLO_MODEL = "yolov8n.pt"          # nano: CPU-acceptable, never requires GPU
TRACKER_CFG = "bytetrack.yaml"     # ultralytics' built-in ByteTrack config
VEHICLE_CLASSES = (2, 3, 5, 7)     # COCO: car, motorcycle, bus, truck
PERSON_CLASSES = (0,)              # COCO: person (pedestrian)
RIDER_CLASSES = (1,)              # COCO: bicycle
ANIMAL_CLASSES = (14, 15, 16, 17, 18, 19, 20, 21, 22, 23)  # bird,cat,dog,horse,...
# Everything a collision can involve. NOTE: YOLO/COCO has no class for poles,
# trees, footpaths or road markings — those are background "stuff" that needs a
# semantic-segmentation model, not this object detector, and static scenery is
# not crash-relevant anyway. We box the dynamic road users that actually collide.
DETECT_CLASSES = tuple(sorted(set(
    VEHICLE_CLASSES + PERSON_CLASSES + RIDER_CLASSES + ANIMAL_CLASSES)))
# Coarse kind per COCO class id, for overlay colour-coding on the dashboard.
CLASS_KIND = {0: "person", 1: "vehicle", 2: "vehicle", 3: "vehicle", 5: "vehicle",
              7: "vehicle", **{c: "animal" for c in ANIMAL_CLASSES}}
DETECT_CONF = 0.25                 # detection confidence floor
YOLO_IMGSZ = 960                   # inference resolution; higher recovers small/
                                   # distant vehicles in high-res drone footage
IOU_FALLBACK_THRESH = 0.30         # greedy IoU matching threshold (fallback)


# ---------------------------------------------------------------------------
# REASON stage v2 — hosted crash-classification model (Lane A, HACKATHON_PLAN).
# A high-recall CPU gate nominates candidate windows; the learned model (served
# on a remote GPU — modal_app.py) decides. GPU never enters the critical path.
# Enabled by setting CRASH_MODEL_URL; without it nothing changes.
# ---------------------------------------------------------------------------
CRASH_MODEL_TIMEOUT_SEC = 3.0      # remote classify budget; late => degrade
CRASH_MODEL_N_FRAMES = 16          # clip window the classifier was trained on
CRASH_PROB_THRESH = 0.5            # below => the gate candidate is vetoed
# High-recall gate for hackathon wiring: pass as `cfg` so the cheap CPU physics
# nominates generously and the learned model does the deciding. (Demo clips
# keep their tuned per-camera configs until the model is actually deployed.)
GATE_CONFIG = Config(
    speed_thresh=2.0, stop_thresh=0.8, window=10, diverge_deg=25.0,
    prox_factor=2.5, impact_decel=1.0,
)


# ---------------------------------------------------------------------------
# CONFIRM stage — hosted VLM (Anthropic vision). NEVER a local model.
# ---------------------------------------------------------------------------
VLM_MODEL = "claude-sonnet-4-6"    # vision-capable; hosted API call only
VLM_MAX_TOKENS = 300
VLM_TIMEOUT_SEC = 10.0
VLM_N_FRAMES = 4                   # evenly-sampled frames per event window
EVENT_WINDOW_SEC = 2.0             # seconds of footage handed to the VLM
VLM_PROMPT = (
    "Did a vehicle collision occur in these frames? "
    'Reply as strict JSON: {"crash": true/false, "description": '
    '"<one line, e.g. \'Two-vehicle collision, left lane, one vehicle '
    'overturned.\'>"}'
)
VLM_UNCONFIRMED_MSG = "unconfirmed — physics-only detection"


# ---------------------------------------------------------------------------
# CTOL — optimization loop
# ---------------------------------------------------------------------------
PROPOSER_MODEL = "claude-sonnet-4-6"
PROPOSER_MAX_TOKENS = 1000
EVAL_TOLERANCE = 15                # frames of slack around a truth span
CTOL_MAX_BATCHES = 8
CTOL_BATCH_SIZE = 4
CTOL_TARGET_F1 = 0.90
CTOL_PLATEAU_EPS = 0.005
CTOL_PLATEAU_PATIENCE = 2
CTOL_TRACE_PATH = str(DATA_DIR / "ctol_trace.jsonl")     # JSONL flight recorder
CTOL_PLOT_PATH = str(DOCS_DIR / "ctol_convergence.png")  # deck deliverable


# ---------------------------------------------------------------------------
# ALERT stage — server + dashboard
# ---------------------------------------------------------------------------
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]  # Vite dev
DEFAULT_CAMERA_TAG = "CAM-07 · NH-48 Km142"
STREAM_FPS = 20.0                  # default emit/playback rate when no metadata

# Demo clip registry: name -> cached video + trajectory JSON + camera tag.
# `demo_crash` is a rendered crash simulation (video pixels match its tracks
# exactly => a guaranteed, self-consistent detected crash for the stage demo).
# `highway` is real drone footage showcasing live YOLOv8 + ByteTrack tracking.
# Real-time CC footage (Ayalon Freeway, Tel Aviv — genuine 30fps capture, not a
# time-compressed stock clip), so the highway feeds play at a natural 1.0x. The
# dashboard also exposes a live speed control for presenters who want it.
DEMO_CLIPS = {
    "demo_crash":   {"video": "demo_crash.mp4", "tracks": "demo_crash.json",
                     "camera_tag": "CAM-07 · NH-48 Km142", "playback_rate": 1.0},
    "highway":      {"video": "highway.mp4", "tracks": "highway.json",
                     "camera_tag": "CAM-31 · Ayalon Expwy", "playback_rate": 1.0,
                     "config": HIGHWAY_STRICT_CONFIG},
    "highway_wide": {"video": "highway_wide.mp4", "tracks": "highway_wide.json",
                     "camera_tag": "CAM-32 · Ayalon Expwy S", "playback_rate": 1.0,
                     "config": HIGHWAY_STRICT_CONFIG},
    # Real fixed-camera CCTV: a low-speed collision leaves a car askew and halts
    # traffic (~5.9s). Physics fires once on the approaching car's speed collapse;
    # confirmation pre-cached. Verified against the footage (CCTV clock 11:29:32).
    "junction":     {"video": "junction.mp4", "tracks": "junction.json",
                     "camera_tag": "CAM-19 · Tonghua Junction", "playback_rate": 1.0,
                     "config": JUNCTION_CONFIG},
    # Real fixed-camera expressway CCTV (public-domain automated-camera footage,
    # G56 Hangrui Expwy): a multi-vehicle rear-end pile-up. The strict highway
    # profile fires exactly once on the ~5-vehicle cluster's speed collapse
    # (peak ≈8 → ≈0.4 px/frame) while staying quiet through normal flow — the
    # same per-camera calibration that keeps the Ayalon feeds silent.
    "pileup":       {"video": "pileup.mp4", "tracks": "pileup.json",
                     "camera_tag": "CAM-56 · G56 Hangrui Expwy", "playback_rate": 1.0,
                     "config": HIGHWAY_STRICT_CONFIG},
    # Full untrimmed real clips — the whole footage plays; the detector fires
    # once within it. (No pre-cached VLM confirmation, so these ship honestly as
    # physics-only / unconfirmed unless a live key confirms them.)
    "intersection": {"video": "moto_van.mp4", "tracks": "moto_van.json",
                     "camera_tag": "CAM-22 · City Interchange", "playback_rate": 1.0,
                     "config": INTERSECTION_CONFIG},
    "roundabout":   {"video": "roundabout.mp4", "tracks": "roundabout.json",
                     "camera_tag": "CAM-14 · Roundabout Exit", "playback_rate": 1.0,
                     "config": ROUNDABOUT_CONFIG},
    "overturn":     {"video": "overturn.mp4", "tracks": "overturn.json",
                     "camera_tag": "CAM-77 · Yinchuan Rd", "playback_rate": 1.0,
                     "config": OVERTURN_CONFIG},
}
DEMO_DEFAULT_CLIP = "demo_crash"

# Deck theme — shared by the matplotlib convergence plot and the React app.
COLOR_BG = "#0B0F17"
COLOR_CARD = "#141B29"
COLOR_ALERT = "#FF3B3B"
COLOR_ACCENT = "#35D0E0"
COLOR_MUTE = "#8A97AB"
COLOR_OK = "#3DDC84"
