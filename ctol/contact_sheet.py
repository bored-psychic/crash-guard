"""
contact_sheet.py — fast manual truth-span labeling without a GUI.

For each video, renders one JPEG grid of every Nth frame with the frame
index burned into each cell. Eyeball the sheet, note where the impact
starts/ends, add a row to labels.csv:

    clip_name,start_frame,end_frame

then push the spans into the clip JSONs:

    python -m ctol.ingest apply data/tracks data/labels/labels.csv

Whole-clip default spans are fine for recall; tightening them with this
tool is what sharpens precision measurement (a false positive 200 frames
before the crash stops being silently absorbed by a giant span).

Usage (run from the repo root):
    python -m ctol.contact_sheet VIDEO_DIR SHEET_DIR [--every 15] [--cols 6]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from config import COLOR_BG
from ctol.ingest import VIDEO_EXTS


def _hex_to_bgr(hexstr: str) -> tuple[int, int, int]:
    h = hexstr.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)                           # OpenCV is BGR


def sheet_for(video: Path, out: Path, every: int, cols: int,
              cell_w: int = 320) -> None:
    cap = cv2.VideoCapture(str(video))
    cells = []
    i = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            h, w = frame.shape[:2]
            cell_h = int(cell_w * h / w)
            cell = cv2.resize(frame, (cell_w, cell_h))
            cv2.rectangle(cell, (0, 0), (110, 26), (0, 0, 0), -1)
            cv2.putText(cell, f"f={i}", (6, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 255), 1,
                        cv2.LINE_AA)
            cells.append(cell)
        i += 1
    cap.release()
    if not cells:
        return
    cell_h = cells[0].shape[0]
    rows = (len(cells) + cols - 1) // cols
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), np.uint8)
    grid[:] = _hex_to_bgr(COLOR_BG)            # matches the deck bg
    for k, cell in enumerate(cells):
        r, c = divmod(k, cols)
        grid[r * cell_h:(r + 1) * cell_h,
             c * cell_w:(c + 1) * cell_w] = cell
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"{video.stem}: {len(cells)} cells -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m ctol.contact_sheet")
    ap.add_argument("video_dir"); ap.add_argument("sheet_dir")
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--cols", type=int, default=6)
    args = ap.parse_args()
    vd, sd = Path(args.video_dir), Path(args.sheet_dir)
    sd.mkdir(parents=True, exist_ok=True)
    for v in sorted(p for p in vd.rglob("*")
                    if p.suffix.lower() in VIDEO_EXTS):
        sheet_for(v, sd / f"{v.stem}.jpg", args.every, args.cols)


if __name__ == "__main__":
    main()
