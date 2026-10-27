"""
logger.py — CTOL run tracing + the demo-ready convergence curve.

Two jobs:
  1. JSONL trace of every batch (configs, scores, wall-clock, speculative-hit
     flag) — the flight recorder. Non-deterministic loops without traces are
     undebuggable.
  2. plot(): the F1-vs-batch convergence curve. This image IS the "our system
     tuned itself" slide. Dark theme to match the deck.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from config import (COLOR_ACCENT, COLOR_ALERT, COLOR_BG, COLOR_CARD, COLOR_MUTE,
                    COLOR_OK, CTOL_PLOT_PATH, CTOL_TRACE_PATH)


class RunLogger:
    def __init__(self, path: str = CTOL_TRACE_PATH):
        self.path = path
        self.t0 = time.time()
        self.batches: list[dict] = []
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        open(self.path, "w").close()

    def log_batch(self, batch_idx: int, results: list[dict],
                  best_f1: float, speculative_hit: bool) -> None:
        rec = {
            "batch": batch_idx,
            "t_sec": round(time.time() - self.t0, 2),
            "speculative_hit": speculative_hit,
            "batch_best_f1": max(r["f1"] for r in results),
            "global_best_f1": best_f1,
            "results": [
                {**{k: r[k] for k in ("f1", "precision", "recall",
                                      "tp", "fp", "fn")},
                 "cfg": r["cfg"].as_dict()}
                for r in results
            ],
        }
        self.batches.append(rec)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---- the slide-worthy artifact ----------------------------------------
    def plot(self, out_path: str = CTOL_PLOT_PATH,
             target_f1: float | None = None) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [b["batch"] for b in self.batches]
        best = [b["global_best_f1"] for b in self.batches]
        batch_best = [b["batch_best_f1"] for b in self.batches]
        # every individual eval as scatter, jittered on x
        sx, sy = [], []
        for b in self.batches:
            for i, r in enumerate(b["results"]):
                sx.append(b["batch"] + (i - len(b["results"]) / 2) * 0.06)
                sy.append(r["f1"])

        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
        fig.patch.set_facecolor(COLOR_BG)
        ax.set_facecolor(COLOR_CARD)

        ax.scatter(sx, sy, s=28, color=COLOR_MUTE, alpha=0.55, zorder=2,
                   label="individual configs")
        ax.plot(xs, batch_best, color=COLOR_ACCENT, lw=1.6, ls="--", alpha=0.8,
                zorder=3, label="batch best")
        ax.plot(xs, best, color=COLOR_ALERT, lw=3.0, marker="o", ms=7, zorder=4,
                label="global best F1")
        if target_f1:
            ax.axhline(target_f1, color=COLOR_OK, lw=1.2, ls=":",
                       label=f"target ({target_f1})")

        ax.set_title("CTOL — self-optimizing crash-detector thresholds",
                     color="white", fontsize=13, pad=12, weight="bold")
        ax.set_xlabel("optimization batch", color=COLOR_MUTE)
        ax.set_ylabel("F1 on labeled clips", color=COLOR_MUTE)
        ax.tick_params(colors=COLOR_MUTE)
        for s in ax.spines.values():
            s.set_color(COLOR_MUTE)
        ax.set_ylim(0, 1.05)
        ax.grid(color="#1B2436", lw=0.7)
        ax.legend(facecolor=COLOR_CARD, edgecolor=COLOR_MUTE, labelcolor="white",
                  fontsize=9, loc="lower right")
        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, facecolor=COLOR_BG)
        plt.close(fig)
        return out_path
