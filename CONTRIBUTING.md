# Contributing

This is a hackathon prototype. Contributions and issues are welcome, and keep to the rules below.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install --no-deps -r requirements-nodeps.txt      # ultralytics, installed without its opencv-python pin
cp .env.example .env
python -m pytest -q
```

Tests that need the raw dataset (`DATASET_ROOT`), the MediaPipe face model, the hand-state checkpoint or YOLO weights **skip** automatically when those are missing.

## Rules of the codebase

- **Keep real, inferred and synthetic data separate.**
  - Real events are written only by `EventDatasetWriter(kind="real")`.
  - Synthetic data is written only to `data/simulated/` or `data/events_synthetic/`.
  - Never add synthetic rows to `data/sample/`.
- **Never invent values.**
  - Missing measurements stay `null`.
  - GPS is filled only from a real GPS log.
- **Image-space honesty.** Do not add distance, speed or TTC claims without a calibrated camera model.
- **Keep the real-time loop deterministic.** Perception → tracking → risk → alert uses no LLM. The optional LLM summary hook in `app/analytics/report.py` is post-event only and off by default.
- **Keep the risk engine explainable.** Every point it adds must be a documented factor that appears in `risk_factors` and the reason text.
- **Privacy.** Do not commit:
  - raw recordings;
  - identifiable faces;
  - readable number plates;
  - `.env` files, API keys or local absolute paths.

## Pull requests

- Add or update tests in `tests/` for behaviour changes, and run `python -m pytest -q`.
- Update the relevant document ([DATA_SCHEMA.md](DATA_SCHEMA.md), [RESULTS.md](RESULTS.md), [LIMITATIONS.md](LIMITATIONS.md)) when a change affects the data or the claims.
