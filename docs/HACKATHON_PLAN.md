# CrashGuard — Hackathon Build Plan (Lane A)

**Status:** planning / decided. The current repo is a validated prototype; this is the
plan to rebuild the REASON stage around a *learned* model for the hackathon.

**Chosen lane: A — custom model, trained free on Colab, served on serverless GPU (Modal).**
Keeps the "no GPU in the critical path" constraint (GPU is remote), reuses all existing
plumbing, and costs ~$0 for the demo / ~$5–30/mo for a 24/7 pilot.

---

## 1. Architecture (what changes vs. what stays)

```
 SEE                     REASON (NEW)              CONFIRM              ALERT
 YOLOv8n + ByteTrack     learned crash model        VLM (Claude)         FastAPI + SSE
 (CPU, keep)             on Modal GPU (remote)       (keep)               dashboard (keep)
    |                        ^                          ^                    ^
    |  cheap CPU gate        | candidate window         |                    |
    +---- nominates ---------+  (~16 frames)  ----------+--------------------+
         candidates                     classify: crash prob
```

- **KEEP:** perception (`pipeline/track.py`), server + SSE (`server/main.py`), dashboard
  (`web/`), VLM confirm (`pipeline/vlm.py`), async-off-frame-loop pattern (`pipeline/stream.py`).
- **REPLACE:** `pipeline/physics.py` (hand-tuned) → `pipeline/reason.py`:
  1. a **cheap CPU gate** (high-recall, dumb): any two tracked boxes whose edges touch
     **and** a sudden speed change → nominate a candidate window. (Reuse the edge-gap +
     speed-drop logic already written — it's fine as a *recall* filter, not the decision.)
  2. a **remote model client**: POST the buffered ~16-frame window to the Modal endpoint,
     get back `{crash: prob}`. Runs in the background task, never blocks the frame loop.
- **Contract stays identical:** the model client still emits a `CrashEvent`, so the server,
  SSE, dashboard, and VLM all work unchanged.

## 2. Model + dataset

**Model:** a small **temporal video classifier** over a rolling ~16-frame window (this is
what justifies the GPU and gives real crash understanding vs. per-frame guessing):
- Scaffolded default: **torchvision `r3d_18`** (Kinetics-400 pretrained 3D CNN) — zero
  extra deps, trains reliably on free Colab; the notebook + Modal app are built around it.
- Upgrades if time allows: **MoViNet-A2** / **X3D-S** (swap the model cell + endpoint,
  the preprocessing contract stays).
- Simpler fallback if training runs long: fine-tune **YOLOv8** as an "accident" object
  detector (per-frame) — reuses the YOLO stack, runs even on CPU.

**Datasets (fixed-camera first, since our cameras are fixed):**
- **TAD** — large CCTV accident benchmark (primary).
- **CADP** / **TU-DAT** / 2026 **ACCIDENT benchmark** — more surveillance clips.
- **CCD (Car Crash Dataset)** — clip-labeled crash/normal (good for a classifier; dashcam,
  use to bulk up positives).
- Quick option: Roboflow "car-crash" datasets for a fast YOLO fine-tune.

## 3. Training (Colab, free, ~2–4 hrs, do BEFORE hackathon)

1. New Colab notebook, free T4 GPU.
2. Pull dataset (TAD/CCD), build 16-frame clip windows (crash-centered positives + random
   negatives).
3. Fine-tune MoViNet-A2 (or YOLOv8 for the fallback). Target: clear separation on a held-out
   set; log accuracy / AP.
4. Export weights → download / push to a bucket. **Cost: $0.**

## 4. Serving (Modal, serverless GPU, pay-per-second, $30/mo free credit)

1. `pip install modal`; `modal token new`.
2. `modal_app.py`: `@app.function(gpu="T4")` loads the weights once (container warm),
   exposes a `@modal.web_endpoint` that takes JPEG frames → returns `{crash_prob}`.
3. `modal deploy` → get an HTTPS URL. Scales to zero when idle.
4. In CrashGuard, `reason.py` POSTs candidate windows to that URL (like the VLM call).
   **Degradation:** if the endpoint is unreachable → fall back to the cheap CPU gate marked
   `unconfirmed` (mirrors the existing "no single point of failure" pattern).

## 5. Cost

| | Demo | 24/7, 1 camera |
|---|---|---|
| Training (Colab) | $0 | $0 |
| Serving (Modal, gated by CPU trigger) | ~$0 (within $30 credit) | ~$5–30/mo |

## 6. Hackathon hour-by-hour (assumes model trained the night before)

1. **H0–1:** wire `reason.py` (CPU gate + Modal client) into `stream.py`, keep `CrashEvent`.
2. **H1–2:** deploy Modal endpoint, smoke-test with the `pileup` clip → confirm one on-time alert.
3. **H2–3:** tune the CPU gate's recall threshold; verify dashboard flashes at impact.
4. **H3–4:** polish demo (dropdown, outline overlay already done), rehearse.
5. **Buffer:** if Modal misbehaves live → flip to VLM-as-model or the CPU fallback (both wired).

## 7. Demo script

1. Open dashboard → pick `pileup` (overhead CCTV, real multi-car impact).
2. Play → the learned model fires the red alert at the impact + description card + "SMS SENT ✓".
3. One line: *"perception on-device, the crash model runs on serverless GPU we spin up per-event,
   and a vision LLM confirms — no GPU sits on the camera."*
4. Backup: kill the Modal URL mid-demo → alert still ships marked *unconfirmed* (resilience).

## 8. Risks & fallbacks

- **Model training overruns** → ship the YOLOv8 accident-detector fallback (CPU, no Modal).
- **Modal cold start on stage** → pre-warm with one request before demoing; or VLM-as-model.
- **Footage** → demo on `pileup` (known-good). Avoid dashcam clips (ego-motion breaks tracking).
- **Live network fails** → the `unconfirmed` degradation path keeps the demo alive.

## 9. Pre-hackathon checklist

- [x] `reason.py` scaffolded behind the existing `CrashEvent` contract (gate +
      Modal client, veto hook in `stream.py`, server auto-enables on
      `CRASH_MODEL_URL`) — unit-tested in `tests/test_reason.py`; demo verified
      unchanged without the URL
- [x] `modal_app.py` serving template written (T4, TorchScript, preprocessing
      contract matches the notebook, `smoke` entrypoint)
- [x] Colab notebook written: `training/train_crash_model.ipynb`
- [ ] Notebook **run on Colab** → `crash_r3d18.ts` exported (needs dataset + T4 session)
- [ ] Modal endpoint deployed + smoke-tested (`pip install modal`, `modal token new`,
      `modal deploy modal_app.py`)
- [ ] `pileup` fires one on-time alert end-to-end **through the live model**
- [ ] Fallback (VLM / CPU gate) verified by killing the endpoint
