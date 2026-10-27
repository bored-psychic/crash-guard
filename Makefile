HOST ?= 127.0.0.1
PORT ?= 8000
PY   ?= python3
DATASET ?= data/datasets

.PHONY: help install track sim precache test ctol ingest verify web-build server demo clean

help:
	@echo "CrashGuard targets:"
	@echo "  make install    - pip deps + npm deps"
	@echo "  make sim        - render the demo crash-simulation clip + tracks"
	@echo "  make track      - run YOLOv8+ByteTrack over the real highway clip -> cached JSON"
	@echo "  make precache   - generate VLM confirmations for demo clips (needs ANTHROPIC_API_KEY)"
	@echo "  make test       - run the pytest suite (all phases)"
	@echo "  make ctol       - run the self-tuning threshold optimization loop"
	@echo "  make ingest     - track a real dataset for CTOL (override DATASET=path)"
	@echo "  make verify     - Phase-1 gate checks over data/tracks"
	@echo "  make web-build  - build the React dashboard (web/dist)"
	@echo "  make server     - run the FastAPI server with autoreload"
	@echo "  make demo       - build web, start server, open dashboard, queue clip #1"

install:
	$(PY) -m pip install -r requirements.txt
	cd web && npm install

sim:
	$(PY) -m pipeline.simulate

track:
	$(PY) -m pipeline.track data/clips/highway.mp4 data/tracks/highway.json

precache:
	$(PY) -m pipeline.precache

test:
	$(PY) -m pytest -q

ctol:
	$(PY) -m ctol.ctol

# Real-dataset ingestion (see ctol/README.md). Folder-name truth mode by default;
# override the dataset dir with e.g. `make ingest DATASET=data/hwid12`.
ingest:
	$(PY) -m ctol.ingest track $(DATASET) data/tracks --truth-mode folder

verify:
	$(PY) -m ctol.ingest verify data/tracks

web-build:
	cd web && npm run build

server:
	$(PY) -m uvicorn server.main:app --host $(HOST) --port $(PORT) --reload

# Turnkey demo: build the dashboard, start the server, open the browser.
# The dashboard auto-queues demo clip #1 when served from the API.
demo: web-build
	@$(PY) -m uvicorn server.main:app --host $(HOST) --port $(PORT) --log-level warning & \
	SERVER=$$!; \
	until curl -sf http://$(HOST):$(PORT)/clips >/dev/null 2>&1; do sleep 0.3; done; \
	echo ""; echo "  CrashGuard dashboard -> http://$(HOST):$(PORT)"; echo ""; \
	( open http://$(HOST):$(PORT) 2>/dev/null || xdg-open http://$(HOST):$(PORT) 2>/dev/null || true ); \
	wait $$SERVER

clean:
	rm -rf web/dist .pytest_cache data/ctol_trace.jsonl
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
