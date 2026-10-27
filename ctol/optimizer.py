"""
optimizer.py — the "brain" of CTOL: reads score history, proposes the next
batch of configs.

Two proposers, same interface:

  * LLMProposer       — calls a hosted Claude model. The prompt engineering is
                        the interesting part: it feeds a COMPACT summary (not
                        raw history — context rot kills long runs), encodes
                        what FP vs FN means in threshold-space, and demands
                        strict JSON so the loop parses without regex heroics.

  * HeuristicProposer — a dependency-free fallback (coordinate-descent with
                        decaying jitter around the best config, FP/FN-directional
                        bias). Runs offline/keyless and is the failsafe if the
                        API flakes mid-hackathon — no single point of failure.

Both return list[Config], never repeating a config already tried (the loop
memoizes, but not re-proposing also saves LLM turns).
"""

from __future__ import annotations

import json
import random

from config import BOUNDS, Config, PROPOSER_MAX_TOKENS, PROPOSER_MODEL, anthropic_key


def _clamp(cfg_dict: dict) -> Config:
    """Coerce an arbitrary dict into a bounds-legal Config. This is also the
    sanitizer for untrusted LLM output."""
    d: dict = {}
    for k, (lo, hi) in BOUNDS.items():
        v = float(cfg_dict.get(k, (lo + hi) / 2))
        v = max(lo, min(hi, v))
        d[k] = int(round(v)) if k == "window" else round(v, 2)
    # physical sanity: "stopped" must be slower than "fast"
    if d["stop_thresh"] >= d["speed_thresh"]:
        d["stop_thresh"] = round(d["speed_thresh"] * 0.3, 2)
    return Config(**d)


def initial_batch(batch_size: int, seed: int = 7) -> list[Config]:
    """Diverse seed configs via latin-hypercube-ish spread, so the first batch
    maps the space instead of clustering."""
    rng = random.Random(seed)
    out = []
    for i in range(batch_size):
        frac = (i + 0.5) / batch_size
        out.append(_clamp({
            "speed_thresh": 1.0 + frac * 19.0 * rng.uniform(0.8, 1.2),
            "stop_thresh": 0.1 + frac * 3.0 * rng.uniform(0.8, 1.2),
            "window": 6 + frac * 20,
            "diverge_deg": 10 + frac * 70 * rng.uniform(0.8, 1.2),
            "prox_factor": 1.0 + frac * 3.5,
        }))
    return out


def summarize(history: list[dict], top_k: int = 5) -> str:
    """Compact history for the LLM: top-k configs + scores + the FP/FN picture.
    Raw history rots the context; this doesn't."""
    ranked = sorted(history, key=lambda r: r["f1"], reverse=True)[:top_k]
    lines = []
    for r in ranked:
        c = r["cfg"].as_dict()
        lines.append(
            f"F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} "
            f"(TP={r['tp']} FP={r['fp']} FN={r['fn']}) :: {json.dumps(c)}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
class HeuristicProposer:
    """Coordinate-descent + decaying jitter around the best config. Zero
    dependencies. The offline failsafe."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.step = 0

    def propose(self, history: list[dict], batch_size: int,
                tried: set) -> list[Config]:
        self.step += 1
        decay = 0.6 ** self.step                    # shrink search radius
        best = max(history, key=lambda r: r["f1"])
        base = best["cfg"].as_dict()
        fp, fn = best["fp"], best["fn"]

        out: list[Config] = []
        guard = 0
        while len(out) < batch_size and guard < 200:
            guard += 1
            cand = dict(base)
            # directional nudge: FP-heavy => tighten (raise thresholds);
            # FN-heavy => loosen. Then jitter 1-2 random dims.
            bias = 1.0 + (0.15 if fp > fn else -0.15) * decay * 2
            cand["speed_thresh"] *= bias
            cand["diverge_deg"] *= bias
            for k in self.rng.sample(list(BOUNDS), k=2):
                lo, hi = BOUNDS[k]
                span = (hi - lo) * decay * 0.5
                cand[k] = float(cand[k]) + self.rng.uniform(-span, span)
            cfg = _clamp(cand)
            if cfg not in tried and cfg not in out:
                out.append(cfg)
        return out or [best["cfg"]]


# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a hyperparameter optimizer for a highway crash \
detector. The detector fires when a tracked vehicle shows (a) speed collapse \
ending at a stop (below stop_thresh) reached EITHER by being fast — mean speed \
over the first half-window above speed_thresh px/frame — OR by an impact \
impulse — a single-frame deceleration above impact_decel px/frame^2, which \
catches low-speed collisions; AND (b) heading change above diverge_deg degrees \
OR another vehicle's box edges within (prox_factor-1) * box_size pixels \
(prox_factor=1 means the boxes must actually touch).

Threshold-space intuition you MUST use:
- Too many FALSE POSITIVES => detector too loose: raise speed_thresh, lower \
stop_thresh, raise diverge_deg, lower prox_factor, raise impact_decel, or \
lengthen window.
- Too many FALSE NEGATIVES => too strict: do the opposite. Missing SLOW/low-\
speed impacts specifically => lower impact_decel.
- window trades stability vs responsiveness: long windows smooth noise but \
miss fast crashes.

Propose diverse candidates: exploit around the best config AND include at \
least one exploratory config in an untried region.

Respond ONLY with a JSON array of config objects, no prose, no markdown \
fences. Each object has keys: speed_thresh, stop_thresh, window, diverge_deg, \
prox_factor, impact_decel. Bounds: speed_thresh 1-20, stop_thresh 0.1-5, \
window 6-30 (int), diverge_deg 10-90, prox_factor 1-5, impact_decel 0.5-10."""


class LLMProposer:
    """Claude-backed proposer. Falls back to HeuristicProposer on ANY API or
    parse failure — the loop never dies because the brain hiccuped."""

    def __init__(self, model: str = PROPOSER_MODEL):
        self.model = model
        self.fallback = HeuristicProposer()
        self._client = None
        try:
            import anthropic
            if anthropic_key():
                self._client = anthropic.Anthropic()
        except ImportError:
            pass

    @property
    def live(self) -> bool:
        return self._client is not None

    def propose(self, history: list[dict], batch_size: int,
                tried: set) -> list[Config]:
        if not self.live:
            return self.fallback.propose(history, batch_size, tried)
        try:
            best = max(history, key=lambda r: r["f1"])
            user = (
                f"Score history (best {min(5, len(history))} so far):\n"
                f"{summarize(history)}\n\n"
                f"Current best has FP={best['fp']}, FN={best['fn']}.\n"
                f"Propose exactly {batch_size} NEW configs (JSON array only)."
            )
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=PROPOSER_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
            )
            text = resp.content[0].text.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            raw = json.loads(text)
            configs: list[Config] = []
            for item in raw[:batch_size]:
                cfg = _clamp(item)
                if cfg not in tried and cfg not in configs:
                    configs.append(cfg)
            # top up with heuristic if the LLM repeated itself
            if len(configs) < batch_size:
                configs += self.fallback.propose(
                    history, batch_size - len(configs), tried | set(configs))
            return configs
        except Exception:
            return self.fallback.propose(history, batch_size, tried)
