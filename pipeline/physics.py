"""
physics.py — CrashGuard trajectory-physics crash detector (the REASON stage).

Input: per-frame tracked vehicle states (YOLO + ByteTrack output).
Output: a list of CrashEvents.

A "crash signature" is the conjunction of:
  1. SPEED COLLAPSE  — a vehicle drops to ~zero. Two ways to trip it:
       (a) ABSOLUTE  — it was moving clearly fast, and is now stopped; or
       (b) IMPULSE   — it takes a sharp single-frame deceleration (high jerk)
                       and ends stopped. This catches LOW-SPEED impacts (a
                       creep-and-bump) that never cleared the "fast" bar, while
                       a smooth brake-to-stop — low jerk — does NOT trip it.
  2. PATH DIVERGENCE — its heading swings abruptly (spin / knock-off-course),
     OR CONTACT      — another vehicle's BOX EDGES are within prox_factor of
                       this one at collapse time (edge gap, not centroid circle).

Each signal alone false-alarms (braking at a toll = collapse; a normal turn =
heading change; dense traffic = proximity). The AND of collapse with
(divergence OR contact) is what makes it a crash. The thresholds that define
"fast", "stopped", "abrupt", and "near" are exactly what CTOL optimizes.

Data contract (the universal interchange format):
    frames: list[dict[int, tuple[float, float, float, float]]]
        frames[i][track_id] = (cx, cy, w, h)   # box center + size, pixels
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from config import Config  # the tunable threshold contract lives in config.py

__all__ = ["Config", "CrashEvent", "CrashDetector", "run_detector"]


@dataclass
class CrashEvent:
    frame_idx: int
    track_ids: tuple[int, ...]
    peak_speed: float
    final_speed: float
    heading_delta_deg: float


class CrashDetector:
    """Stateful per-track physics. Feed frames in order via update(); returns a
    CrashEvent when the signature fires, else None.

    Per-track state is a rolling deque of centers — self-trimming, O(window)
    memory per vehicle, no unbounded growth on long feeds.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pos: dict[int, deque] = defaultdict(lambda: deque(maxlen=cfg.window))
        self.size: dict[int, deque] = defaultdict(lambda: deque(maxlen=cfg.window))
        self.cooldown: dict[int, int] = defaultdict(int)  # 1 event per crash

    # ---- kinematics helpers ------------------------------------------------
    @staticmethod
    def _speeds(pts: np.ndarray) -> np.ndarray:
        """Per-step speed magnitudes (px/frame) from an (N,2) center history."""
        return np.linalg.norm(np.diff(pts, axis=0), axis=1)

    @staticmethod
    def _med3(sp: np.ndarray) -> np.ndarray:
        """3-tap median filter. A tracker glitch (ID swap / box teleport) is a
        single-frame speed spike; the median annihilates it while preserving a
        genuine speed STEP (an impact). Without this, jitter fakes huge jerks."""
        if len(sp) < 3:
            return sp
        return np.median(np.stack([sp[:-2], sp[1:-1], sp[2:]]), axis=0)

    @staticmethod
    def _heading_delta_deg(pts: np.ndarray) -> float:
        """Angle between early-window and late-window motion direction."""
        n = len(pts)
        v_early = pts[n // 2] - pts[0]
        v_late = pts[-1] - pts[n // 2]
        ne, nl = np.linalg.norm(v_early), np.linalg.norm(v_late)
        if ne < 1e-6 or nl < 1e-6:
            return 0.0
        cosang = np.clip(np.dot(v_early, v_late) / (ne * nl), -1.0, 1.0)
        return float(np.degrees(np.arccos(cosang)))

    # ---- main entry --------------------------------------------------------
    def update(self, frame_idx: int, tracks: dict[int, tuple]) -> CrashEvent | None:
        cfg = self.cfg

        for tid, box in tracks.items():
            cx, cy, w, h = box[:4]           # box may carry a 5th class id
            self.pos[tid].append((cx, cy))
            self.size[tid].append((w + h) / 2.0)
            if self.cooldown[tid] > 0:
                self.cooldown[tid] -= 1

        for tid, hist in self.pos.items():
            if len(hist) < cfg.window or self.cooldown[tid] > 0:
                continue
            pts = np.asarray(hist, dtype=np.float64)
            sp = self._speeds(pts)

            # (a) SPEED COLLAPSE ending at a stop, via EITHER path:
            #   absolute — clearly fast over the first half-window; or
            #   impulse  — a sharp single-frame deceleration (jerk), which is
            #              what a low-speed IMPACT looks like. A smooth brake has
            #              low per-frame jerk and so stays below impact_decel.
            half = len(sp) // 2
            now_stopped = sp[-2:].mean() < cfg.stop_thresh
            was_fast = sp[:half].mean() > cfg.speed_thresh
            spd = self._med3(sp)   # despiked speeds: real steps survive, jitter dies
            max_decel = float((spd[:-1] - spd[1:]).max()) if len(spd) > 1 else 0.0
            impact = max_decel > cfg.impact_decel
            if not (now_stopped and (was_fast or impact)):
                continue

            # (b) DIVERGENCE-or-CONTACT
            hd = self._heading_delta_deg(pts)
            diverged = hd > cfg.diverge_deg
            near_ids = self._nearby(tid, tracks)

            # Crash = collapse AND (divergence OR contact-proximity).
            # Divergence-only covers single-vehicle spinouts; proximity-only
            # covers straight rear-endings where heading barely changes.
            if diverged or near_ids:
                self.cooldown[tid] = cfg.window * 3
                for nid in near_ids:
                    self.cooldown[nid] = cfg.window * 3
                return CrashEvent(
                    frame_idx=frame_idx,
                    track_ids=(tid, *near_ids),
                    peak_speed=float(sp[:half].mean()),
                    final_speed=float(sp[-2:].mean()),
                    heading_delta_deg=hd,
                )
        return None

    def _nearby(self, tid: int, tracks: dict[int, tuple]) -> tuple[int, ...]:
        """Contact = the GAP BETWEEN BOX EDGES is small — not the distance
        between centers. Edge gap accounts for both vehicles' extents, so a real
        bumper-touch reads as ~0 regardless of car size or lane offset (a far
        sharper contact signal for low-speed impacts than a centroid circle).
        prox_factor is the tolerance in box-sizes: 1.0 => edges must actually
        meet; higher => looser. Overlapping boxes have gap 0."""
        if tid not in tracks:
            return ()
        cx, cy, w, h = tracks[tid][:4]
        my_size = float(np.mean(self.size[tid])) if self.size[tid] else (w + h) / 2.0
        tol = max(0.0, self.cfg.prox_factor - 1.0) * my_size
        out = []
        for oid, obox in tracks.items():
            if oid == tid:
                continue
            ox, oy, ow, oh = obox[:4]
            gap_x = abs(ox - cx) - (w + ow) / 2.0   # >0 = clear space between edges
            gap_y = abs(oy - cy) - (h + oh) / 2.0   # <=0 = overlap on that axis
            gap = np.hypot(max(0.0, gap_x), max(0.0, gap_y))
            if gap <= tol:
                out.append(oid)
        return tuple(out)


def run_detector(frames: list[dict[int, tuple]], cfg: Config) -> list[CrashEvent]:
    """Run the physics layer over a whole clip. Pure function of (frames, cfg)
    — read-only, no shared state — which is what licenses CTOL to evaluate many
    configs in parallel processes safely."""
    det = CrashDetector(cfg)
    events: list[CrashEvent] = []
    for i, tracks in enumerate(frames):
        ev = det.update(i, tracks)
        if ev is not None:
            events.append(ev)
    return events
