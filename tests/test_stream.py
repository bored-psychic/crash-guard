"""
test_stream.py — PHASE 4 verification.

Feed a cached-style crash trajectory through process_stream with a MOCKED 3 s
VLM and assert:
  1. the frame loop's per-frame time is unaffected by confirmation latency
     (the 3 s VLM must run in the background, never inside the loop), and
  2. a Confirmation still reaches the outbound queue.

Also proves the degradation path: the real (keyless) confirm() returns an
'unconfirmed' Confirmation rather than raising.
"""

from __future__ import annotations

import asyncio
import time

from config import Config
from pipeline.physics import CrashEvent
from pipeline.stream import ConfirmedEvent, process_stream
from pipeline.vlm import Confirmation, confirm

CFG = Config(speed_thresh=3.0, stop_thresh=1.0, window=12,
             diverge_deg=35.0, prox_factor=2.0)
BOX = (36.0, 20.0)


def _rear_end(n: int = 80, impact: int = 22, v: float = 6.0):
    """Track 1 slams into stationary track 2 — fires exactly one crash event."""
    y, bx = 100.0, 140.0
    return [{1: (v * f if f < impact else v * impact, y, *BOX), 2: (bx, y, *BOX)}
            for f in range(n)]


async def test_frame_loop_unaffected_by_slow_vlm():
    source = [(None, tr) for tr in _rear_end()]   # no images; mock ignores them
    q: asyncio.Queue = asyncio.Queue()

    async def slow_vlm(event: CrashEvent, window_jpeg):
        await asyncio.sleep(3.0)                   # simulate a slow VLM
        return Confirmation(crash=True, description="mock collision", source="vlm")

    inter, events = [], 0
    last = time.perf_counter()
    async for _frame, _tracks, ev in process_stream(
            source, cfg=CFG, out_queue=q, confirm_fn=slow_vlm):
        now = time.perf_counter()
        inter.append(now - last)
        last = now
        if ev is not None:
            events += 1

    # (1) no frame ever stalled waiting on the 3 s confirmation
    assert max(inter) < 0.5, f"a frame stalled {max(inter):.2f}s — VLM leaked into the loop"
    assert events == 1, f"expected exactly one crash event, got {events}"

    # (2) the confirmation reached the outbound queue
    ce = await asyncio.wait_for(q.get(), timeout=5.0)
    assert isinstance(ce, ConfirmedEvent)
    assert ce.crash is True and ce.description == "mock collision"


async def test_confirm_degrades_without_key():
    """No key / bad input => 'unconfirmed', never an exception (no SPOF)."""
    conf = await confirm(CrashEvent(10, (1, 2), 6.0, 0.1, 5.0), frames_jpeg=[])
    assert conf.crash is None
    assert conf.source == "unconfirmed"
    assert "unconfirmed" in conf.description
