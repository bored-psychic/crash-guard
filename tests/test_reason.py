"""
test_reason.py — REASON v2 verification (gate + hosted model, pipeline/reason.py).

The learned model screens gate candidates before the VLM. Four behaviors:
  1. model says crash      -> VLM is consulted, its verdict/description ship
  2. model says no-crash   -> candidate VETOED: confirm returns None and
                              process_stream drops it (nothing on the queue)
  3. model unreachable     -> degrade to the plain VLM path (today's behavior)
  4. model sure + VLM down -> event still ships crash=True source="model"
"""

from __future__ import annotations

import asyncio

from config import Config
from pipeline.physics import CrashEvent
from pipeline.reason import RemoteCrashModel, gated_confirm
from pipeline.stream import process_stream
from pipeline.vlm import Confirmation

EVENT = CrashEvent(10, (1, 2), 6.0, 0.1, 5.0)
JPEG = [b"\xff\xd8fake"]


class FakeModel(RemoteCrashModel):
    def __init__(self, prob):
        super().__init__(url="http://fake")
        self._prob = prob

    async def classify(self, frames_jpeg):
        return self._prob


def _vlm(result):
    calls = []

    async def fn(event, frames_jpeg):
        calls.append(event)
        return result

    return fn, calls


async def test_model_crash_passes_to_vlm():
    vlm, calls = _vlm(Confirmation(crash=True, description="vlm says crash"))
    conf = await gated_confirm(vlm, FakeModel(0.9))(EVENT, JPEG)
    assert calls, "VLM must be consulted for model-confirmed events"
    assert conf.crash is True and conf.description == "vlm says crash"


async def test_model_veto_returns_none_and_skips_vlm():
    vlm, calls = _vlm(Confirmation(crash=True, description="never"))
    conf = await gated_confirm(vlm, FakeModel(0.1))(EVENT, JPEG)
    assert conf is None
    assert not calls, "vetoed candidates must not spend a VLM call"


async def test_model_unreachable_degrades_to_vlm():
    vlm, calls = _vlm(Confirmation(crash=True, description="vlm fallback"))
    conf = await gated_confirm(vlm, FakeModel(None))(EVENT, JPEG)
    assert calls and conf.description == "vlm fallback"


async def test_model_sure_and_vlm_down_ships_as_model():
    vlm, _ = _vlm(Confirmation(crash=None, description="unconfirmed",
                               source="unconfirmed"))
    conf = await gated_confirm(vlm, FakeModel(0.9))(EVENT, JPEG)
    assert conf.crash is True and conf.source == "model"


async def test_vetoed_event_never_reaches_queue():
    """End-to-end through process_stream: a gate firing that the model rejects
    must leave the outbound queue empty — the dashboard flashes on ANY queued
    event, so a veto must happen before the queue."""
    cfg = Config(speed_thresh=3.0, stop_thresh=1.0, window=12,
                 diverge_deg=35.0, prox_factor=2.0)
    box = (36.0, 20.0)
    src = [(None, {1: (6.0 * min(f, 22), 100.0, *box), 2: (140.0, 100.0, *box)})
           for f in range(80)]                       # rear-end: gate fires once

    vlm, calls = _vlm(Confirmation(crash=True, description="never"))
    q: asyncio.Queue = asyncio.Queue()
    fired = 0
    async for _f, _t, ev in process_stream(
            src, cfg=cfg, out_queue=q,
            confirm_fn=gated_confirm(vlm, FakeModel(0.1))):
        fired += ev is not None
    assert fired == 1, "the gate itself should still nominate"
    assert q.empty(), "vetoed event leaked to the queue (dashboard would flash)"
    assert not calls


async def test_remote_model_degrades_without_url(monkeypatch):
    monkeypatch.delenv("CRASH_MODEL_URL", raising=False)
    assert await RemoteCrashModel(url=None).classify(JPEG) is None
