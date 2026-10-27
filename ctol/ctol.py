"""
ctol.py — the CrashGuard Threshold Optimization Loop.

Composition of loop patterns, performance-first:

  OUTER SHELL   Planner-Critique + Evaluator-Optimizer: proposer plans a BATCH
                of configs; deterministic F1 evaluator judges; proposer reads
                scores and re-plans. Batch-per-turn (not config-per-turn)
                unlocks parallelism.

  INNER ENGINE  Parallel DAG: each config eval is pure + read-only, so a whole
                batch fires concurrently over a process pool. Batch wall-clock
                ~= one eval, not batch_size evals.

  LATENCY HIDE  Speculative pre-fire: while a live LLM proposer is thinking
                (in asyncio.to_thread), a heuristic-guessed next batch evaluates
                underneath it. A wrong guess is discarded — but its completed
                evals are already cached, so the work is never wasted.

  SAFETY RAILS  hard batch budget, memoized configs (never re-eval), plateau
                detection -> escalate to human, JSONL trace of every step.

Run:  python -m ctol.ctol                       # offline, heuristic brain
      ANTHROPIC_API_KEY=... python -m ctol.ctol # LLM-brained proposer
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ProcessPoolExecutor

from config import (CTOL_BATCH_SIZE, CTOL_MAX_BATCHES, CTOL_PLATEAU_EPS,
                    CTOL_PLATEAU_PATIENCE, CTOL_PLOT_PATH, CTOL_TARGET_F1,
                    CTOL_TRACE_PATH, Config)
from ctol.evaluator import LabeledClip, evaluate
from ctol.logger import RunLogger
from ctol.optimizer import HeuristicProposer, LLMProposer, initial_batch


# ---------------------------------------------------------------------------
def _stash(cache: dict, cfg: Config, fut: Future) -> None:
    """Done-callback (runs on the pool's completion thread): cache a finished
    eval the instant its process returns — independent of whether the awaiting
    asyncio task was cancelled. This is what makes speculation lossless."""
    try:
        cache[cfg] = fut.result()
    except Exception:
        pass  # a crashed eval stays uncached; it'll be retried if re-proposed


async def eval_batch(configs: list[Config], clips: list[LabeledClip],
                     pool: ProcessPoolExecutor, cache: dict) -> list[dict]:
    """PARALLEL DAG STEP. Memoized (frozen Config => hashable => free dedupe).
    Each fresh config is submitted to the pool with a done-callback that stashes
    its result; so if this coroutine is cancelled (speculation miss), the
    underlying processes still finish and populate the cache."""
    fresh = [c for c in configs if c not in cache]
    afuts = []
    for c in fresh:
        cfut = pool.submit(evaluate, c, clips)
        cfut.add_done_callback(lambda f, cfg=c: _stash(cache, cfg, f))
        afuts.append(asyncio.wrap_future(cfut))
    if afuts:
        await asyncio.gather(*afuts)   # cancellation propagates; callbacks don't
    return [cache[c] for c in configs if c in cache]


async def ctol(clips: list[LabeledClip],
               max_batches: int = CTOL_MAX_BATCHES,
               batch_size: int = CTOL_BATCH_SIZE,
               target_f1: float = CTOL_TARGET_F1,
               plateau_eps: float = CTOL_PLATEAU_EPS,
               plateau_patience: int = CTOL_PLATEAU_PATIENCE,
               use_llm: bool = True,
               log_path: str = CTOL_TRACE_PATH,
               plot_path: str = CTOL_PLOT_PATH,
               verbose: bool = True) -> dict:
    proposer = LLMProposer() if use_llm else HeuristicProposer()
    live_llm = getattr(proposer, "live", False)
    logger = RunLogger(log_path)
    pool = ProcessPoolExecutor()
    cache: dict[Config, dict] = {}
    history: list[dict] = []
    best: dict | None = None
    prev_best_f1 = -1.0
    stalls = 0
    stop_reason = "budget exhausted"

    batch = initial_batch(batch_size)
    spec_task: asyncio.Task | None = None      # speculative pre-fired evals
    spec_batch: list[Config] | None = None

    try:
        for step in range(max_batches):
            # ---- OBSERVE: harvest speculation if it matches, else eval now
            speculative_hit = False
            if spec_task is not None and spec_batch == batch:
                results = await spec_task                 # already running/done
                speculative_hit = True
            else:
                if spec_task is not None:
                    spec_task.cancel()                    # wrong guess: discard
                results = await eval_batch(batch, clips, pool, cache)
            spec_task, spec_batch = None, None
            history.extend(results)

            # ---- CRITIQUE: rank, update champion
            top = max(results, key=lambda r: r["f1"])
            if best is None or top["f1"] > best["f1"]:
                best = top
            logger.log_batch(step, results, best["f1"], speculative_hit)
            if verbose:
                print(f"[batch {step}] batch_best={top['f1']:.3f} "
                      f"global_best={best['f1']:.3f} "
                      f"(P={best['precision']:.3f} R={best['recall']:.3f}) "
                      f"{'[spec-hit]' if speculative_hit else ''}")

            # ---- VERIFY: deterministic stop conditions
            if best["f1"] >= target_f1:
                stop_reason = "target reached"
                break
            stalls = stalls + 1 if (best["f1"] - prev_best_f1) < plateau_eps else 0
            if stalls >= plateau_patience:
                stop_reason = "plateau — escalate to human"
                break
            prev_best_f1 = best["f1"]
            if step == max_batches - 1:
                break

            # ---- PLAN: proposer generates the next batch from compact history
            tried = set(cache)
            if live_llm:
                # LLM thinking is the slow serial step -> hide eval under it.
                # A heuristic warm-guess pre-fires; if the LLM later proposes the
                # same configs, the cache serves them instantly. Even on a miss,
                # the warm evals already landed in cache — never wasted.
                warm = HeuristicProposer(seed=step).propose(history, batch_size, tried)
                spec_batch = warm
                spec_task = asyncio.ensure_future(eval_batch(warm, clips, pool, cache))
                next_batch = await asyncio.to_thread(
                    proposer.propose, history, batch_size, tried)
                if next_batch != spec_batch:
                    spec_task.cancel()
                    spec_task, spec_batch = None, None
                batch = next_batch
            else:
                batch = proposer.propose(history, batch_size, tried)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    curve = logger.plot(out_path=plot_path, target_f1=target_f1)
    if verbose:
        print(f"\nSTOP: {stop_reason}")
        print(f"BEST CONFIG: {best['cfg'].as_dict()}")
        print(f"SCORES: F1={best['f1']} P={best['precision']} R={best['recall']} "
              f"TP={best['tp']} FP={best['fp']} FN={best['fn']}")
        print(f"evals run: {len(cache)} | trace: {log_path} | curve: {curve}")
    return {"best": best, "stop_reason": stop_reason,
            "history": history, "curve": curve, "n_evals": len(cache)}


# ---------------------------------------------------------------------------
# Offline demo world: synthetic labeled clips with known crash physics, so the
# loop is runnable + verifiable before ByteTrack output exists. Swap
# make_synthetic_clips() for real (frames, truth) pairs and NOTHING else
# changes — the whole point of the pure-function evaluator.
# ---------------------------------------------------------------------------
def make_synthetic_clips(n_clips: int = 8, seed: int = 3) -> list[LabeledClip]:
    """Adversarial synthetic world. Besides a real crash, every clip is salted
    with false-positive bait targeting each physics signal:
      * SWERVER     — big heading change at speed (divergence bait)
      * CLUSTER     — slow congested pack, boxes nearly touching (proximity bait)
      * HARD-BRAKER — sharp smooth deceleration to a crawl (collapse bait)
      * STOPPER     — brakes to a FULL stop in traffic (collapse + proximity:
                      the nastiest non-crash)
    Plus tracker-like jitter on all positions. Naive threshold sets either
    false-alarm on the bait or miss the subtle crash — the loop must find the
    region that separates them.
    """
    import random
    rng = random.Random(seed)
    clips = []
    for ci in range(n_clips):
        frames: list[dict] = []
        truth: list[tuple[int, int]] = []
        n_frames = 320
        cars: dict[int, dict] = {}
        # background cruisers
        for tid in range(4):
            cars[tid] = {"x": rng.uniform(0, 300), "y": 80 + 70 * tid,
                         "v": rng.uniform(4, 8), "vy": 0.0, "mode": "cruise"}
        # scripted crash (subtle: moderate speed, modest knock-off)
        cars[100] = {"x": 0.0, "y": 380.0, "v": rng.uniform(6, 8), "vy": 0.0,
                     "mode": "crasher"}
        cars[101] = {"x": rng.uniform(150, 200), "y": 380.0, "v": 1.8,
                     "vy": 0.0, "mode": "cruise"}
        # FP bait, one of each
        cars[200] = {"x": 0.0, "y": 520.0, "v": 8.0, "vy": 0.0,
                     "mode": "braker", "t0": rng.randint(60, 120)}
        cars[201] = {"x": 0.0, "y": 590.0, "v": 7.0, "vy": 0.0,
                     "mode": "swerver", "t0": rng.randint(90, 150)}
        cars[202] = {"x": 50.0, "y": 660.0, "v": 6.5, "vy": 0.0,
                     "mode": "stopper", "t0": rng.randint(100, 160)}
        # congestion cluster: 3 slow cars packed within ~1.5 box-widths
        for k in range(3):
            cars[300 + k] = {"x": 400.0 + 45 * k, "y": 660.0 + rng.uniform(-6, 6),
                             "v": 1.2, "vy": 0.0, "mode": "cruise"}

        crashed_at = None
        for f in range(n_frames):
            snap = {}
            for tid, c in cars.items():
                m = c["mode"]
                if m == "crasher" and crashed_at is None:
                    if cars[101]["x"] - c["x"] < 20:          # impact
                        crashed_at = f
                        truth.append((f, min(f + 25, n_frames - 1)))
                        c["v"], c["vy"] = 0.4, rng.uniform(4, 7)
                        cars[101]["v"] = 0.2
                elif m == "crasher" and crashed_at is not None:
                    c["v"] = max(0.05, c["v"] * 0.75)
                    c["vy"] *= 0.8
                elif m == "braker" and f >= c["t0"]:
                    c["v"] = max(1.6, c["v"] * 0.94)          # crawl, not stop
                elif m == "swerver" and c["t0"] <= f < c["t0"] + 30:
                    c["vy"] = 3.5 if (f // 10) % 2 == 0 else -3.5  # lane weave
                elif m == "swerver":
                    c["vy"] = 0.0
                elif m == "stopper" and f >= c["t0"]:
                    c["v"] = max(0.0, c["v"] - 0.5)           # full stop
                c["x"] += c["v"] + rng.uniform(-0.5, 0.5)     # tracker jitter
                c["y"] += c["vy"] + rng.uniform(-0.7, 0.7)
                if c["x"] > 1000 and m == "cruise":
                    c["x"] = 0.0
                snap[tid] = (c["x"], c["y"], 36.0, 20.0)
            frames.append(snap)
        clips.append(LabeledClip(name=f"clip_{ci}", frames=frames, truth=truth))
    return clips


if __name__ == "__main__":
    asyncio.run(ctol(make_synthetic_clips()))
