"""
test_physics.py — PHASE 2 verification oracle.

Four synthetic trajectories, each targeting one edge of the crash-signature
AND-gate (collapse AND (divergence OR contact)):

  1. rear-end collision  -> fires EXACTLY one event within +/-15 frames of impact
  2. smooth hard-braker  -> fires nothing (collapse, but alone + straight:
                            the divergence-OR-contact gate blocks it)
  3. lane-weaving swerver -> fires nothing (big heading change, but constant
                            speed => no collapse)
  4. congested slow cluster -> fires nothing (proximity, but never fast =>
                            no collapse)

All four must pass. Config is stated explicitly so the test is self-contained
and does not drift if config.DEFAULT_CONFIG changes (it mirrors that default).
"""

from __future__ import annotations

from config import Config
from pipeline.physics import run_detector

CFG = Config(speed_thresh=3.0, stop_thresh=1.0, window=12,
             diverge_deg=35.0, prox_factor=2.0, impact_decel=1.5)

BOX = (36.0, 20.0)   # (w, h) => avg box size 28 px => prox radius 56 px


# --- scenario builders -----------------------------------------------------
def rear_end_frames(n: int = 80, impact: int = 22, v: float = 6.0):
    """Track 1 barrels in at v px/frame and slams into stationary track 2,
    stopping dead ~8 px behind it (well inside the 56 px proximity radius).
    Heading barely changes => proximity is what makes this a crash."""
    y, bx = 100.0, 140.0
    frames = []
    for f in range(n):
        ax = v * f if f < impact else v * impact     # moving, then stopped
        frames.append({1: (ax, y, *BOX), 2: (bx, y, *BOX)})
    return frames, impact


def hard_braker_frames(n: int = 80):
    """One car, straight line, decelerates 8 -> ~0.4 px/frame. Speed DOES
    collapse, but it is alone and going straight, so the divergence-or-contact
    gate must veto it."""
    x, y, v = 0.0, 100.0, 8.0
    frames = []
    for _ in range(n):
        frames.append({1: (x, y, *BOX)})
        x += v
        v = max(0.4, v * 0.85)
    return frames


def swerver_frames(n: int = 80, vx: float = 6.0):
    """One car weaving hard side-to-side at constant speed. Large heading
    change, but it never slows => no collapse => no crash."""
    x, y = 0.0, 300.0
    frames = []
    for f in range(n):
        frames.append({1: (x, y, *BOX)})
        x += vx
        y += 4.0 if (f // 5) % 2 == 0 else -4.0
    return frames


def slow_impact_frames(n: int = 80, impact: int = 30, v: float = 2.0):
    """Track 1 CREEPS in at v=2 px/frame — BELOW speed_thresh (3.0), so the
    absolute 'was fast' gate never trips — and abruptly bumps stationary track 2,
    stopping dead. Only the impact-impulse (sharp single-frame deceleration)
    catches this low-speed collision. Proximity supplies the AND-gate."""
    y, bx = 100.0, v * impact + 8.0
    frames = []
    for f in range(n):
        ax = v * f if f < impact else v * impact
        frames.append({1: (ax, y, *BOX), 2: (bx, y, *BOX)})
    return frames, impact


def smooth_stopper_frames(n: int = 80, v: float = 2.0):
    """Two cars side by side; track 1 eases SMOOTHLY to a full stop next to
    track 2 (proximity satisfied). Low speed AND low jerk => neither the
    absolute nor the impulse collapse gate should fire. The nastiest non-crash:
    a slow stop in traffic must stay silent."""
    x, y, bx = 0.0, 100.0, 40.0
    frames = []
    for _ in range(n):
        frames.append({1: (x, y, *BOX), 2: (bx, y, *BOX)})
        x += v
        v = max(0.0, v - 0.12)      # gentle ~0.12 px/frame^2 decel, no impact jerk
    return frames


def cluster_frames(n: int = 80):
    """Four cars packed within the proximity radius, all creeping at 1 px/frame.
    Proximity is satisfied, but nobody is ever 'fast' => no collapse => no crash."""
    xs, y, v = [100.0, 130.0, 160.0, 190.0], 500.0, 1.0
    frames = []
    for f in range(n):
        snap = {i + 1: (bx + v * f, y + (i % 2) * 5.0, *BOX)
                for i, bx in enumerate(xs)}
        frames.append(snap)
    return frames


# --- tests -----------------------------------------------------------------
def test_rear_end_fires_exactly_once_near_impact():
    frames, impact = rear_end_frames()
    events = run_detector(frames, CFG)
    assert len(events) == 1, f"expected 1 event, got {len(events)}: {events}"
    assert abs(events[0].frame_idx - impact) <= 15, (
        f"event at {events[0].frame_idx}, impact at {impact}")
    assert 1 in events[0].track_ids and 2 in events[0].track_ids


def test_hard_braker_fires_nothing():
    assert run_detector(hard_braker_frames(), CFG) == []


def test_swerver_fires_nothing():
    assert run_detector(swerver_frames(), CFG) == []


def test_cluster_fires_nothing():
    assert run_detector(cluster_frames(), CFG) == []


def test_slow_impact_fires_via_impulse():
    """Low-speed collision (never 'fast') is caught by the impact-impulse gate."""
    frames, impact = slow_impact_frames()
    events = run_detector(frames, CFG)
    assert len(events) == 1, f"expected 1 event, got {len(events)}: {events}"
    assert abs(events[0].frame_idx - impact) <= 15
    assert events[0].peak_speed < CFG.speed_thresh, (
        "this must fire on impulse, not the absolute speed gate")


def test_smooth_stopper_fires_nothing():
    """A smooth low-speed stop in traffic (low jerk) must NOT false-alarm."""
    assert run_detector(smooth_stopper_frames(), CFG) == []
