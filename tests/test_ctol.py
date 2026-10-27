"""
test_ctol.py — PHASE 3 verification.

On adversarial synthetic LabeledClips (each carrying a real crash plus swerver /
hard-braker / full-stopper / congestion false-positive bait), assert that:
  * F1 improves by >= 0.2 from batch 0 to the final best,
  * the loop terminates with a stated stop reason, and
  * the convergence PNG is produced.

Offline (use_llm=False => heuristic proposer): hermetic, deterministic, no key.
"""

from __future__ import annotations

import os

from config import CTOL_BATCH_SIZE
from ctol.ctol import ctol, make_synthetic_clips


async def test_ctol_improves_and_terminates(tmp_path):
    clips = make_synthetic_clips(n_clips=6, seed=3)
    trace = tmp_path / "trace.jsonl"
    curve = tmp_path / "curve.png"

    result = await ctol(
        clips, max_batches=6, batch_size=CTOL_BATCH_SIZE, target_f1=0.90,
        use_llm=False, log_path=str(trace), plot_path=str(curve), verbose=False,
    )

    history = result["history"]
    batch0_best = max(r["f1"] for r in history[:CTOL_BATCH_SIZE])
    final_best = result["best"]["f1"]
    assert final_best - batch0_best >= 0.2, (
        f"F1 only moved {batch0_best:.3f} -> {final_best:.3f}")

    assert result["stop_reason"] in {
        "target reached", "plateau — escalate to human", "budget exhausted"}

    assert os.path.exists(result["curve"]) and os.path.getsize(result["curve"]) > 0
    assert os.path.exists(trace) and os.path.getsize(trace) > 0
