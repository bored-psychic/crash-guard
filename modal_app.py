"""
modal_app.py — hosted GPU serving for the REASON-stage crash classifier (Lane A).

Serves the TorchScript model exported by training/train_crash_model.ipynb on a
serverless T4 that scales to zero. CrashGuard's frame loop never touches this —
pipeline/reason.py POSTs candidate windows from the background confirmation
task, keeping "no GPU in the critical path" true (the GPU is not even ours).

Endpoint contract (mirrored by pipeline/reason.py):
    POST <url>  json={"frames": ["<b64 jpeg>", ...]}   # ~16 frames, time order
    -> 200 json={"crash_prob": 0.0..1.0}

One-time setup:
    pip install modal && modal token new
    modal volume create crashguard-weights
    modal volume put crashguard-weights crash_r3d18.ts /crash_r3d18.ts
    modal deploy modal_app.py          # prints the HTTPS URL
    echo 'CRASH_MODEL_URL=<that url>' >> .env

Smoke test (no CrashGuard needed):
    modal run modal_app.py::smoke --video data/clips/pileup.mp4
"""

from __future__ import annotations

import base64

import modal

# Preprocessing contract — MUST match the training notebook exactly.
N_FRAMES = 16
SIDE = 112
MEAN = (0.43216, 0.394666, 0.37645)      # Kinetics-400 (torchvision r3d_18)
STD = (0.22803, 0.22145, 0.216989)

WEIGHTS_PATH = "/weights/crash_r3d18.ts"

app = modal.App("crashguard-reason")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "opencv-python-headless",
                 "numpy", "fastapi[standard]")
)
weights = modal.Volume.from_name("crashguard-weights", create_if_missing=True)


@app.cls(gpu="T4", image=image, volumes={"/weights": weights},
         scaledown_window=120)
class CrashClassifier:
    @modal.enter()
    def load(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Fail loudly with instructions rather than serve garbage.
        try:
            self.model = torch.jit.load(WEIGHTS_PATH, map_location=self.device)
        except Exception as e:
            raise RuntimeError(
                f"cannot load {WEIGHTS_PATH} — train + export with "
                "training/train_crash_model.ipynb, then "
                "`modal volume put crashguard-weights crash_r3d18.ts "
                f"/crash_r3d18.ts` ({e})"
            )
        self.model.eval()

    def _preprocess(self, jpegs: list[bytes]):
        """b64-decoded JPEGs -> (1, C, T, H, W) float tensor, notebook-identical."""
        import cv2
        import numpy as np
        import torch

        frames = []
        for j in jpegs:
            img = cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (SIDE, SIDE))
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not frames:
            raise ValueError("no decodable frames")
        while len(frames) < N_FRAMES:                 # pad short windows
            frames.append(frames[-1])
        arr = np.stack(frames[:N_FRAMES]).astype("float32") / 255.0  # T,H,W,C
        arr = (arr - MEAN) / STD
        t = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)   # 1,C,T,H,W
        return t.float().to(self.device)

    @modal.fastapi_endpoint(method="POST")
    def classify(self, body: dict) -> dict:
        import torch

        jpegs = [base64.standard_b64decode(f) for f in body.get("frames", [])]
        with torch.no_grad():
            logit = self.model(self._preprocess(jpegs))
        prob = torch.sigmoid(logit.squeeze()).item()
        return {"crash_prob": round(float(prob), 4)}


@app.local_entrypoint()
def smoke(video: str = "data/clips/pileup.mp4"):
    """Sample N_FRAMES from a local clip and hit the deployed classifier."""
    import json
    import urllib.request

    import cv2

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    jpegs = []
    for i in range(N_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * (total - 1) / (N_FRAMES - 1)))
        ok, frame = cap.read()
        if ok:
            jpegs.append(cv2.imencode(".jpg", frame)[1].tobytes())
    cap.release()

    url = CrashClassifier().classify.get_web_url()
    payload = json.dumps({"frames": [
        base64.standard_b64encode(j).decode("ascii") for j in jpegs
    ]}).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"})
    print(video, "->", urllib.request.urlopen(req).read().decode())
