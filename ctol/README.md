# CTOL — the self-tuning package

The **C**rashGuard **T**hreshold **O**ptimization **L**oop tunes the crash
detector's physics thresholds (`speed_thresh`, `stop_thresh`, `window`,
`diverge_deg`, `prox_factor`) to maximize F1 over labeled clips. No hand-tuning.

| Module | Role |
|---|---|
| `evaluator.py` | deterministic F1 reward over `LabeledClip`s — the un-foolable "am I done?" oracle |
| `optimizer.py` | `LLMProposer` (Claude, strict-JSON) with `HeuristicProposer` fallback (coordinate descent) |
| `logger.py` | JSONL flight recorder + the convergence curve (`docs/ctol_convergence.png`) |
| `ctol.py` | the loop: batch-parallel, speculative pre-fire, plateau→human escalation |
| `ingest.py` | **real-dataset ingestion** — videos → cached-trajectory JSON CTOL can train on |
| `contact_sheet.py` | frame-grid JPEGs for fast manual crash-span labeling |

The loop's architecture (loop-pattern composition, the reward design, the demo
talking point) is documented in the [top-level README](../README.md#ctol). Run
the offline synthetic demo with `make ctol`.

## Ingesting real datasets (HWID12 / CADP / your own clips)

`ingest.py` turns downloaded videos into CTOL-ready JSON — YOLOv8 + ByteTrack
when ultralytics is installed, a dependency-free IoU tracker otherwise (the
tracker used is recorded in each file's `meta.method`). It writes the **same
on-disk envelope as `pipeline/track.py`** plus a `truth` key, so one schema
serves the live demo, the Phase-1 gate, and CTOL:

```json
{ "meta": {"name","source","method","fps","frame_count","width","height"},
  "frames": [ {"7": [cx,cy,w,h], "...": "..."} ],
  "truth":  [[start_frame, end_frame]] }
```

Suggested layout: drop datasets in `data/datasets/`, keep human label inputs
(`labels.csv`, CADP span JSON) in `data/labels/`, and write tracks to
`data/tracks/`.

```bash
# HWID12-style category folders (normal-traffic folders auto-labeled negative)
python -m ctol.ingest track data/datasets data/tracks --truth-mode folder
make ingest DATASET=data/datasets          # ← same thing via the Makefile

# CADP: convert its annotations to explicit spans, then track with them
python -m ctol.ingest cadp CADP_annotations.json data/labels/cadp_labels.json
python -m ctol.ingest track data/datasets data/tracks --labels data/labels/cadp_labels.json

# Phase-1 gate: persistent tracks, low flicker, per-clip stats
python -m ctol.ingest verify data/tracks    # or: make verify

# Optional precision pass: contact sheets -> labels.csv -> apply
python -m ctol.contact_sheet data/datasets data/labels/sheets --every 15
python -m ctol.ingest apply data/tracks data/labels/labels.csv

# Feed straight into CTOL
python -c "
import asyncio
from ctol.ingest import load_labeled_clips
from ctol.ctol import ctol
asyncio.run(ctol(load_labeled_clips('data/tracks'), target_f1=0.90))"
```

Optional `--vlm-label` asks the hosted vision model (`config.VLM_MODEL`) to
locate the impact window per clip (needs `ANTHROPIC_API_KEY`; fails soft to the
default span).

Tracking runs **once** per clip; CTOL then evaluates hundreds of configs against
the cached JSON in seconds. That decoupling (track once, tune many) is the
biggest practical speedup of all.

> **Licensing:** HWID12 / CADP are research-and-education only — blur plates
> before publishing any frames from them.
