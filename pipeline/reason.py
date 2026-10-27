"""
reason.py — REASON stage v2: gate + hosted crash-classification model (Lane A).

Replaces "hand-tuned physics decides" with "cheap CPU gate nominates, a learned
model decides":

    SEE (YOLO+ByteTrack, CPU)
      -> GATE   : the existing CrashDetector run high-recall (config.GATE_CONFIG)
                  — its only job is to nominate candidate windows cheaply.
      -> REASON : RemoteCrashModel — the ~16-frame candidate window is POSTed to
                  a hosted GPU endpoint (modal_app.py) that returns crash_prob.
                  Below CRASH_PROB_THRESH => the candidate is VETOED (never
                  reaches the queue / dashboard).
      -> CONFIRM: the VLM describes confirmed events, exactly as before.

Hard rules honored:
  * The remote call runs inside the background confirmation task (stream.py),
    NEVER in the per-frame path — same pattern as the VLM.
  * No single point of failure: endpoint missing/unreachable/slow => the
    candidate falls through to the VLM path unchanged, so today's behavior is
    the degradation mode. If the VLM is also down, the event ships as
    crash=True source="model" only when the model actually said crash.

Wire-up (server does this automatically when CRASH_MODEL_URL is set):
    process_stream(..., confirm_fn=gated_confirm())

Endpoint contract (mirrored by modal_app.py):
    POST CRASH_MODEL_URL  json={"frames": ["<b64 jpeg>", ...]}
    -> 200 json={"crash_prob": 0.0..1.0}
"""

from __future__ import annotations

import asyncio
import base64
from typing import Awaitable, Callable, Optional

from config import (CRASH_MODEL_N_FRAMES, CRASH_MODEL_TIMEOUT_SEC,
                    CRASH_PROB_THRESH, crash_model_url)
from pipeline.physics import CrashEvent
from pipeline.vlm import Confirmation, _sample, confirm as vlm_confirm

ConfirmFn = Callable[[CrashEvent, list[bytes]], Awaitable[Optional[Confirmation]]]


class RemoteCrashModel:
    """Async client for the hosted classifier. classify() returns crash_prob
    in [0, 1], or None on any failure (no URL, timeout, HTTP error, bad JSON)
    so callers can degrade — identical philosophy to vlm.confirm()."""

    def __init__(self, url: str | None = None,
                 timeout: float = CRASH_MODEL_TIMEOUT_SEC):
        self.url = url or crash_model_url()
        self.timeout = timeout

    async def classify(self, frames_jpeg: list[bytes]) -> float | None:
        if not self.url or not frames_jpeg:
            return None
        payload = {"frames": [
            base64.standard_b64encode(j).decode("ascii")
            for j in _sample(frames_jpeg, CRASH_MODEL_N_FRAMES)
        ]}
        try:
            import httpx                       # ships with the anthropic dep
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await asyncio.wait_for(
                    client.post(self.url, json=payload), timeout=self.timeout)
                resp.raise_for_status()
                prob = float(resp.json()["crash_prob"])
                return min(1.0, max(0.0, prob))
        except Exception:
            return None                        # degrade, never raise


def gated_confirm(
    confirm_fn: ConfirmFn = vlm_confirm,
    model: RemoteCrashModel | None = None,
    thresh: float = CRASH_PROB_THRESH,
) -> ConfirmFn:
    """Wrap a confirm_fn with the learned REASON decision. Same signature as
    vlm.confirm, so it drops into process_stream(confirm_fn=...) unchanged.

    model says < thresh  -> None (veto: stream.py drops the event silently)
    model says >= thresh -> confirm_fn describes it; if the VLM is down, the
                            event still ships crash=True source="model"
    model unreachable    -> fall through to confirm_fn (today's behavior)
    """
    m = model or RemoteCrashModel()

    async def _confirm(event: CrashEvent, frames_jpeg: list[bytes]):
        prob = await m.classify(frames_jpeg)
        if prob is None:                       # model down => physics-era path
            return await confirm_fn(event, frames_jpeg)
        if prob < thresh:
            return None                        # vetoed gate nomination
        conf = await confirm_fn(event, frames_jpeg)
        if conf is not None and conf.crash is None:   # VLM degraded, model sure
            return Confirmation(
                crash=True, source="model",
                description=f"crash detected by model (p={prob:.2f}) — "
                            "VLM unavailable",
                latency_s=conf.latency_s,
            )
        return conf

    return _confirm
