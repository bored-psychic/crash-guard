"""
vlm.py — PHASE 4: async VLM confirmation (the CONFIRM stage).

When the physics layer flags a crash candidate, we hand the ~2 s event window
to a hosted vision-language model (Anthropic vision — never a local model) to
(1) verify it's a real collision and (2) describe it in one line.

Hard rules honored here:
  * This call is async and is ALWAYS scheduled off the frame loop (see
    stream.py). It must never run inside the per-frame path.
  * No single point of failure: no key / timeout / API error / parse failure
    => degrade to Confirmation(crash=None, "unconfirmed — physics-only
    detection"). The physics event still ships, just marked unconfirmed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass

from config import (VLM_MAX_TOKENS, VLM_MODEL, VLM_N_FRAMES, VLM_PROMPT,
                    VLM_TIMEOUT_SEC, VLM_UNCONFIRMED_MSG, anthropic_key)
from pipeline.physics import CrashEvent


@dataclass
class Confirmation:
    crash: bool | None          # True / False, or None when unconfirmed
    description: str
    latency_s: float = 0.0
    source: str = "vlm"         # "vlm" | "unconfirmed" | "cache"


def _sample(frames_jpeg: list[bytes], k: int) -> list[bytes]:
    """Evenly sample k frames spanning the window (first..last inclusive)."""
    if len(frames_jpeg) <= k:
        return list(frames_jpeg)
    n = len(frames_jpeg)
    idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return [frames_jpeg[i] for i in idx]


def _image_block(jpeg: bytes) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(jpeg).decode("ascii"),
        },
    }


def _unconfirmed(t0: float) -> Confirmation:
    return Confirmation(crash=None, description=VLM_UNCONFIRMED_MSG,
                        latency_s=round(time.perf_counter() - t0, 3),
                        source="unconfirmed")


async def confirm(event: CrashEvent, frames_jpeg: list[bytes]) -> Confirmation:
    """Send up to VLM_N_FRAMES evenly-sampled JPEG frames from the event window
    to the hosted VLM; parse strict JSON {crash, description}. On no-key /
    timeout / any error => Confirmation(crash=None, "unconfirmed ...")."""
    t0 = time.perf_counter()
    if not frames_jpeg or not anthropic_key():
        return _unconfirmed(t0)
    try:
        import anthropic
        client = anthropic.AsyncAnthropic()
        content: list[dict] = [_image_block(j) for j in _sample(frames_jpeg, VLM_N_FRAMES)]
        content.append({"type": "text", "text": VLM_PROMPT})
        resp = await asyncio.wait_for(
            client.messages.create(
                model=VLM_MODEL, max_tokens=VLM_MAX_TOKENS,
                messages=[{"role": "user", "content": content}],
            ),
            timeout=VLM_TIMEOUT_SEC,
        )
        text = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return Confirmation(
            crash=bool(data["crash"]),
            description=str(data.get("description", "")).strip() or "(no description)",
            latency_s=round(time.perf_counter() - t0, 3),
            source="vlm",
        )
    except Exception:
        # timeout, network, auth, malformed JSON — all degrade identically
        return _unconfirmed(t0)
