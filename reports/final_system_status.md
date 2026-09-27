# Final system status: Physical-AI Driving Safety (prototype)

_Generated 2026-09-27T00:33:18+00:00 from the current project artifacts._

> **What it is:** a prototype Physical-AI safety-observation and intervention system.
>
> **What it does not claim:**
> - collision prediction;
> - real-world safety validation;
> - precise distance or vehicle speed;
> - medical drowsiness detection;
> - geographic hotspot detection;
> - synchronized driver/road events.

## A. What is implemented

Real-time loop: **perception → tracking → risk engine → alert**. The post-event analytics run afterwards, offline.

| Layer | Module | Status |
| --- | --- | --- |
| Dataset audit, hand-dataset split | `app/dataset_audit`, `app/driver/hand_dataset.py` | implemented; run on the real dataset |
| Driver perception | `app/driver` (MediaPipe face landmarks, head pose, eye closure, drowsiness score, hand-state classifier + durations) | implemented; run on real driver videos |
| Road perception | `app/road` (pretrained YOLO26n, COCO road classes, CPU) | implemented; run on real front videos |
| Tracking | `app/tracking` (class-aware greedy IoU tracker, image-space motion features) | implemented |
| Risk engine | `app/risk` (transparent rules, evidence gates, hysteresis, explained reasons) | implemented |
| Event logging | `app/events` (schema 1.0, 60 fields, evidence frames/clips, JSONL + Parquet, GPS interface) | implemented |
| Synthetic scenarios | `app/simulation` (feature-level, SYNTHETIC-only dataset) | implemented |
| Alerts / alarm | `app/alerts` (level changes → notice / warning / critical + alarm) | implemented |
| Demo runner | `app/demo`, `scripts/run_demo.py` | implemented; run on real videos |
| Post-event analytics | `app/analytics`, `scripts/generate_safety_report.py` | implemented |
| Dashboard / web UI, LLM agent, hotspot map | — | **not built** (the LLM hook is optional and off by default) |

**Test suite:** 438 passed on Python 3.13 with the real models; 409 passed + 17 skipped on Python 3.11 (tests needing YOLO/pyarrow skip)

## B. Real dataset statistics (LOCAL_REAL)

- **Events:** 66 in `data/events/`, from 16 sessions (run local_real_run_01).
  - Observation types: INFERRED 64, OBSERVED 2.
- **By type:** RISK_ESCALATED 32, RISK_DEESCALATED 18, PERSISTENT_HAZARD 9, DRIVER_STATE_CHANGE 7.
- **By camera:** front 54, driver 12.
- **By risk level:** SAFE 9, CAUTION 34, HIGH 16, CRITICAL 0.
- **By hazard type:** PEDESTRIAN_CONFLICT 28, APPROACHING_VEHICLE 19, not recorded 7, NONE 4, DRIVER_DROWSINESS 4, ANIMAL_HAZARD 2, ROAD_USER_PRESENT 2.
- **Alarm recommended by the engine:** false 43, true 16, not recorded 7.
- **Persistent hazards:** 9.
- **Driver-state changes:** 7.
- **Evidence:** 48 events with evidence frames.
- **GPS:** GPS hotspot analysis unavailable because no real GPS observations were collected.
- **Real driver + road recording together:** none exists.

## C. Synthetic validation statistics (SYNTHETIC, separate)

- **Rows:** 1441 SYNTHETIC rows in `data/simulated/`.
  - 12 scenarios; 24 instances at 10 Hz.
  - They run through the existing tracker, driver logic, risk engine and event recorder.
- **Risk-level rows:** SAFE 475, CAUTION 592, HIGH 206, CRITICAL 168.
- **Expected vs actual:** 23 of 24 instances met the expectation written before the run.
  - Mismatches: escalating_hazard__v02 (peak CAUTION not in expected HIGH/CRITICAL).
- **Scope:** this validates rule behaviour on constructed inputs only.
- **Separation:** these rows are never counted with real data.
- **SYNTHETIC_COMBINATION:** a separate software test (`data/events_synthetic/`) pairs a real driver clip with an unrelated real front clip. It is not a drive.

## D. Perception

**Road perception**
- **Model:** COCO-pretrained YOLO26n (Ultralytics) on CPU, filtered to road-relevant classes, at 640 px and a 0.35 confidence threshold.
- **Training:** not retrained or adapted to our roads or cameras.
- **Evaluation:** no ground-truth boxes exist, so detection quality is described qualitatively in `road_perception_real_data.md`.

**Driver perception**
- **Face signals:** MediaPipe face landmarks drive head pose, eye-aspect ratio, temporal eye closure and a prototype drowsiness score.
- **Hand-state classifier:** MobileNetV3-Small, trained by the team on 188 labelled images.
  - The images were re-split by class and capture format: 131 / 30 / 27.
  - Its earlier test score (1.00 on 27 images) was on the previous split and is indicative only; the classifier has not been retrained on the new split.
  - It generalises poorly to the real driver-video setup (`driver_camera_domain_comparison.md`), and one real NO_HANDS event was confirmed as a misclassification.

## E. Tracking

- **Method:** greedy class-aware IoU matching, deterministic IDs.
  - IoU ≥ 0.3 for the same class; IoU ≥ 0.6 when the label changes, and the change is recorded, not rewritten.
  - Tracks coast for up to 1 s without a match.
- **Features:** image-space only (pixels, pixels/s, box-area growth).
- **Known issues:** fragmentation on busy scenes and label flicker. In the demo clip, 148 tracks came from 613 detections.

## F. Risk engine

- **Scoring:** transparent point rules.
  - Road factors: persistence, central image band, ln-area growth, lateral motion, image size.
  - Driver factors: drowsiness, sustained hands-off, sustained head-away.
  - A combination bonus applies when a driver factor and a moving road hazard occur together.
- **Gates:** HIGH needs 2 independent evidence groups. CRITICAL needs a driver factor with a moving road hazard, or a very strong road hazard.
- **Levels:** hysteresis with thresholds 20 / 45 / 70.
- **Outputs:** raw and smoothed scores, factors, gates and a generated reason.
- **Meaning:** the score is a prototype, not a probability. The central band is not a path model, and there is no distance, speed or TTC.

## G. Event logging

- **Event types:** RISK_ESCALATED, RISK_DEESCALATED, PERSISTENT_HAZARD (≥ 2 s, 10 s cooldown) and DRIVER_STATE_CHANGE (held for 3 observations).
- **Schema:** 1.0, 60 fields; each field is tagged METADATA, OBSERVED or INFERRED.
- **Evidence:** relative evidence paths; frame or 5 s + 5 s clip evidence.
- **Storage:** JSONL + Parquet + schema.json.
- **Separation:** real and synthetic writers refuse each other's records.
- **GPS:** null / UNAVAILABLE unless a real GPS log is provided.
- **`alarm_triggered`:** stays false under schema 1.0.

## H. Alert system

`app/alerts` consumes RiskAssessment objects or recorded level events. It adds no risk logic:
- SAFE: no alert.
- CAUTION: visual/log notice.
- HIGH: "WARNING: ROAD HAZARD" (or "WARNING: DRIVER STATE" for a driver-only hazard) plus an alarm.
- CRITICAL: "CRITICAL: IMMEDIATE ATTENTION REQUIRED" plus a strong alarm.

Each alert quotes the engine's `risk_reason`. Output is concise console lines plus an optional terminal bell, and alerts are logged to `alerts.jsonl` linked to event IDs.

- **Demo `outputs/demo`:** front camera, `video_20260926_170005.mp4`.
  - 115 frames processed; 12 events.
  - Alerts: 5 escalation alerts (NOTICE 1, WARNING 4); 4 alarms.
  - Peak level: HIGH.
- **Demo `outputs/demo_driver`:** driver camera, `video_20260926_220714.mp4`.
  - 63 frames processed; 7 events.
  - Alerts: 2 escalation alerts (NOTICE 2); 0 alarms.
  - Peak level: CAUTION.

## I. Post-event analytics

`app/analytics` works offline, after events are recorded. It produces:
- LOCAL_REAL event statistics;
- descriptive repeated patterns;
- a hotspot section that reports *"GPS hotspot analysis unavailable because no real GPS observations were collected."* and non-geographic frequencies (source file, camera, stream time) instead;
- deterministic review recommendations, which are not diagnoses or fault;
- per-event entries with evidence paths;
- a deterministic summary. `generate_post_event_summary()` has an optional LLM hook that is off by default.

Synthetic data is reported only with `--include-synthetic`, in a separate section.

## J. Known limitations

- **No synchronized driver+road recording.** Real events are road-only or driver-only, so a real CRITICAL "driver + road" alert cannot occur. That path is exercised only with synthetic input.
- **Image space only.** There is no calibration. Camera ego-motion shows up as object motion and is a known source of HIGH levels on real clips.
- **Small real dataset.** The clips are short (about 1–11 s), with no GPS, no absolute clock and no ground truth for crashes, near misses, boxes or tracks. Counts describe system behaviour, not accuracy.
- **Perception errors propagate** into risk, events, alerts and analytics: detector label flicker, tracker fragmentation and hand-classifier domain shift.
- **Prototype thresholds.** Rule weights and thresholds are prototype values, not tuned or validated on real outcomes.
- **Offline alarm.** The alarm is a console / terminal-bell output in offline replay. No in-vehicle integration or latency measurement exists.
- **Drowsiness is not medical.** The drowsiness score is a behavioural eye-closure proxy, not a medical measure.

## K. Provenance and data lineage

```text
<DATASET_ROOT> (team recordings, read-only)
  frontcamera/*.mp4 ─► YOLO26n ─► IoUTracker ─► RiskEngine ─► EventRecorder ─► data/events/ (LOCAL_REAL; OBSERVED/INFERRED fields)
  drivercamera/*.mp4 ─► DriverPerceptionPipeline ─► RiskEngine ─► EventRecorder ─┘
  (driver clip + unrelated front clip, NOT recorded together) ─► data/events_synthetic/ (SYNTHETIC_COMBINATION)
data/events/events.jsonl (read-only reference) ─► app/simulation ─► data/simulated/ (SYNTHETIC; parent ids only)
one real video ─► scripts/run_demo.py ─► outputs/demo*/ (LOCAL_REAL events, alerts, evidence, timeline)
data/events/ ─► scripts/generate_safety_report.py ─► outputs/reports/safety_summary.* (LOCAL_REAL only)
```

- **Labels:** every record carries `data_source` (LOCAL_REAL / SYNTHETIC / SYNTHETIC_COMBINATION) and `observation_type`.
- **Separation:** real and synthetic records never share a file.
- **Source data:** the original dataset and videos were never modified.

## L. Exact commands to reproduce

```bat
rem setup (once)
pip install -r requirements.txt
pip install --no-deps -r requirements-nodeps.txt
rem .env: DATASET_ROOT=<path to Data>, HAND_MODEL_PATH=outputs/models/hand_state_best.pt, FACE_MODEL_PATH=models/face/face_landmarker.task

rem 1. real event dataset (road-only + driver-only; SYNTHETIC_COMBINATION test kept separate)
python scripts/record_events.py --front-source "%DATASET_ROOT%\frontcamera-20260926T193501Z-1-001\frontcamera" --driver-source "%DATASET_ROOT%\drivercamera-20260926T193339Z-1-001\drivercamera\drowsy_driver" --synthetic-combination "%DATASET_ROOT%\drivercamera-20260926T193339Z-1-001\drivercamera\drowsy_driver\video_20260926_220714.mp4" "%DATASET_ROOT%\frontcamera-20260926T193501Z-1-001\frontcamera\vehicles\video_20260926_165951.mp4"

rem 2. synthetic scenarios (seed 42)
python scripts/generate_synthetic_scenarios.py

rem 3. integrated demo: driver-only run, then front-camera run (combined report)
python scripts/run_demo.py --camera driver --video "%DATASET_ROOT%\drivercamera-20260926T193339Z-1-001\drivercamera\drowsy_driver\video_20260926_220714.mp4" --output-dir outputs/demo_driver --fps 10
python scripts/run_demo.py --video "%DATASET_ROOT%\frontcamera-20260926T193501Z-1-001\frontcamera\vehicles\video_20260926_170005.mp4" --output-dir outputs/demo --fps 10 --device auto --include-summary outputs/demo_driver/demo_summary.json

rem 4. post-event safety report (real only; add --include-synthetic for the separate synthetic section)
python scripts/generate_safety_report.py

rem 5. tests
python -m pytest -q
```

