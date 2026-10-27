"""
evaluator.py — the verifiable reward signal for CTOL.

A loop can only self-correct against a reward it can't fool. This module is
deliberately LLM-free: F1 over labeled clips, computed deterministically. The
optimizer proposes; this module judges; the roles never mix (Evaluator-
Optimizer pattern — generator and judge explicitly separated).

A detected event is a true positive if its frame_idx falls inside a truth span
(with EVAL_TOLERANCE frames of slack); each truth span can be claimed once
(no double-counting).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import Config, EVAL_TOLERANCE
from pipeline.physics import run_detector


@dataclass
class LabeledClip:
    name: str
    frames: list          # list[dict[track_id, (cx, cy, w, h)]]
    truth: list = field(default_factory=list)  # [(start_frame, end_frame), ...]


def evaluate(cfg: Config, clips: list[LabeledClip]) -> dict:
    """Score one config across all clips. Pure + read-only => parallel-safe."""
    tp = fp = fn = 0
    per_clip: dict[str, dict] = {}

    for clip in clips:
        events = run_detector(clip.frames, cfg)
        claimed = [False] * len(clip.truth)
        clip_tp = clip_fp = 0

        for ev in events:
            hit = False
            for i, (s, e) in enumerate(clip.truth):
                if not claimed[i] and (s - EVAL_TOLERANCE) <= ev.frame_idx <= (e + EVAL_TOLERANCE):
                    claimed[i] = True
                    hit = True
                    break
            if hit:
                clip_tp += 1
            else:
                clip_fp += 1

        clip_fn = claimed.count(False)
        tp, fp, fn = tp + clip_tp, fp + clip_fp, fn + clip_fn
        per_clip[clip.name] = {"tp": clip_tp, "fp": clip_fp, "fn": clip_fn}

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {
        "cfg": cfg,
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "per_clip": per_clip,
    }
