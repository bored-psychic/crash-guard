"""
stream.py — PHASE 4: the see -> reason -> confirm wiring.

process_stream(source) is an async generator that yields (frame, tracks,
event_or_none) per frame. The moment physics flags a candidate, the buffered
~2 s window is handed to a background asyncio task for VLM confirmation and the
frame loop keeps going — it NEVER awaits the VLM. Confirmed results are pushed
onto an outbound asyncio.Queue that the server drains and forwards over SSE.

`source` yields (frame_bgr_or_None, tracks) per frame. Use replay_source() to
drive it from a cached clip (frames from the video, tracks from cached JSON —
YOLO is never re-run).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from typing import AsyncIterator, Callable, Iterable, Optional

import cv2

from config import (DEFAULT_CAMERA_TAG, DEFAULT_CONFIG, EVENT_WINDOW_SEC,
                    STREAM_FPS, VLM_N_FRAMES, Config)
from pipeline.physics import CrashDetector, CrashEvent
from pipeline.vlm import Confirmation, _sample, confirm


@dataclass
class ConfirmedEvent:
    """What lands on the outbound queue: physics event + VLM verdict + the
    presentation metadata the dashboard needs."""
    id: str
    frame_idx: int
    timestamp: float           # epoch seconds when confirmed
    camera_tag: str
    track_ids: list[int]
    crash: Optional[bool]      # True / False / None(unconfirmed)
    description: str
    source: str                # vlm | unconfirmed | cache
    peak_speed: float
    final_speed: float

    def to_json(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
def encode_jpeg(frame, max_w: int = 768, quality: int = 80) -> bytes:
    """Downscale-if-needed and JPEG-encode a BGR frame for the VLM."""
    h, w = frame.shape[:2]
    if w > max_w:
        frame = cv2.resize(frame, (max_w, int(h * max_w / w)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def _buffer_frame(frame, max_w: int = 960):
    """Keep the rolling window light (esp. for 4K sources) by downscaling."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (max_w, int(h * max_w / w))) if w > max_w else frame


async def _confirm_and_enqueue(
    event: CrashEvent, window_jpeg: list[bytes], camera_tag: str,
    confirm_fn: Callable, out_queue: Optional[asyncio.Queue],
    cache: Optional[dict],
) -> Optional[ConfirmedEvent]:
    """Runs OFF the frame loop. Uses a pre-cached confirmation if present
    (zero live-API risk for demo clips), else calls confirm_fn. A confirm_fn
    may return None to VETO the candidate (the learned REASON model rejecting
    a gate nomination — see pipeline/reason.py); vetoed events never reach the
    queue, so the dashboard never flashes for them."""
    conf: Optional[Confirmation] = None
    if cache is not None and event.frame_idx in cache:
        c = cache[event.frame_idx]
        conf = Confirmation(crash=c.get("crash"), description=c["description"],
                            source="cache")
    if conf is None:
        conf = await confirm_fn(event, window_jpeg)
        if conf is None:                   # vetoed — drop silently
            return None

    ce = ConfirmedEvent(
        id=uuid.uuid4().hex[:8], frame_idx=event.frame_idx, timestamp=time.time(),
        camera_tag=camera_tag, track_ids=list(event.track_ids),
        crash=conf.crash, description=conf.description, source=conf.source,
        peak_speed=round(event.peak_speed, 2), final_speed=round(event.final_speed, 2),
    )
    if out_queue is not None:
        await out_queue.put(ce)
    return ce


async def process_stream(
    source: Iterable[tuple],
    *,
    cfg: Config = DEFAULT_CONFIG,
    camera_tag: str = DEFAULT_CAMERA_TAG,
    out_queue: Optional[asyncio.Queue] = None,
    confirm_fn: Callable = confirm,
    fps: float = STREAM_FPS,
    realtime: bool = False,
    cache: Optional[dict] = None,
) -> AsyncIterator[tuple]:
    """Async generator yielding (frame, tracks, event_or_none). Physics events
    are handed to a background confirmation task; the frame loop never awaits
    the VLM. When `realtime`, pace playback at `fps`."""
    detector = CrashDetector(cfg)
    win = max(VLM_N_FRAMES, int(EVENT_WINDOW_SEC * fps))
    buf: deque = deque(maxlen=win)
    pending: set[asyncio.Task] = set()

    for frame_idx, (frame, tracks) in enumerate(source):
        if frame is not None:
            buf.append(_buffer_frame(frame))
        event = detector.update(frame_idx, tracks)
        if event is not None:
            # cheap: sample + encode just VLM_N_FRAMES frames in the loop, then
            # hand off. The 3 s VLM call runs in the background task, unawaited.
            sampled = _sample(list(buf), VLM_N_FRAMES) if buf else []
            window_jpeg = [encode_jpeg(f) for f in sampled]
            task = asyncio.create_task(_confirm_and_enqueue(
                event, window_jpeg, camera_tag, confirm_fn, out_queue, cache))
            pending.add(task)
            task.add_done_callback(pending.discard)

        yield (frame, tracks, event)          # <-- NEVER awaits the VLM
        await asyncio.sleep(1.0 / fps if realtime else 0)

    # let outstanding confirmations finish before the stream ends
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


# ---------------------------------------------------------------------------
def replay_source(video_path: str, tracks_json: str):
    """Drive process_stream from a cached clip: frames from the video, tracks
    from the cached JSON (YOLO is never re-run). Yields (frame_bgr, tracks)."""
    from pipeline.track import load_tracks
    frames, _meta = load_tracks(tracks_json)
    cap = cv2.VideoCapture(video_path)
    try:
        for tracks in frames:
            ok, frame = cap.read()
            yield (frame if ok else None, tracks)
    finally:
        cap.release()
