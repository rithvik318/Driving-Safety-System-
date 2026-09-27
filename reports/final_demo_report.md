# Final integrated demo: real video to safety alert

> **No synchronized real driver+road recording exists. The driver and front cameras were recorded separately, so this demo runs one camera at a time and never pairs them.**
>
> **Inputs and outputs.**
> - Every demo input below is a real local recording (`LOCAL_REAL`).
> - Every derived value (tracks, risk, events, alerts) is `INFERRED` by the existing prototype components.
> - No synthetic rows are included: the synthetic scenario dataset (`data/simulated/`) and the SYNTHETIC_COMBINATION test (`data/events_synthetic/`) are separate and unused here.

## Pipelines

- **Front camera:** video → YOLO26n detections → IoU tracker → risk engine (road-only) → event recorder → alert layer → evidence, timeline and plot.
- **Driver camera:** video → driver perception (face landmarks, eye closure, drowsiness, hand-state classifier) → risk engine (driver-only) → event recorder → alert layer.
- **Existing components:** all are reused unmodified, with no second risk engine.
- **Alert rules:** the alert layer (`app/alerts`) only maps the engine's level changes to alerts:
  - CAUTION: visual/log notice.
  - HIGH: *WARNING: ROAD HAZARD* (or *WARNING: DRIVER STATE* for a driver-only hazard), with an alarm.
  - CRITICAL: *CRITICAL: IMMEDIATE ATTENTION REQUIRED*, with a strong alarm.
- **Explanations:** each alert is explained by the engine's own `risk_reason`. Alerts are decision support, not collision predictions, and do not guarantee safety.

## Front-camera demo: `video_20260926_170005.mp4`

- **Exact real input video:** `<DATASET_ROOT>/frontcamera-20260926T193501Z-1-001/frontcamera/vehicles/video_20260926_170005.mp4`
  - sha256 `e42f84a6b5997ac8…`
  - 1920×1080, container 54.28 fps, 624 frames
- **Duration processed:** 11.478 s of video (container duration 11.49 s).
- **Frames:** 624 decoded; 115 processed at about 10.02 fps. Wall time 11.1 s.
- **Detections:** 613 (car 256, person 172, bicycle 156, motorcycle 18, truck 11). Detector yolo26n on cpu.
- **Tracks:** 148 (person 59, bicycle 39, car 35, motorcycle 8, truck 7).
- **Risk:** frames per level SAFE 8, CAUTION 77, HIGH 30. Peak HIGH (score 62.59).
- **Alarm-recommended frames:** 30.

**Risk transitions**

| time | transition | hazard type | event |
| --- | --- | --- | --- |
| 00:00.5 | SAFE->CAUTION | PEDESTRIAN_CONFLICT | event_000001 |
| 00:01.5 | CAUTION->HIGH | PEDESTRIAN_CONFLICT | event_000002 |
| 00:02.2 | HIGH->CAUTION | APPROACHING_VEHICLE | event_000003 |
| 00:04.2 | CAUTION->HIGH | PEDESTRIAN_CONFLICT | event_000004 |
| 00:04.4 | HIGH->CAUTION | PEDESTRIAN_CONFLICT | event_000005 |
| 00:04.6 | CAUTION->HIGH | APPROACHING_VEHICLE | event_000006 |
| 00:05.3 | HIGH->CAUTION | APPROACHING_VEHICLE | event_000007 |
| 00:06.9 | CAUTION->HIGH | APPROACHING_VEHICLE | event_000008 |
| 00:08.3 | HIGH->CAUTION | APPROACHING_VEHICLE | event_000011 |
| 00:11.2 | CAUTION->SAFE | PEDESTRIAN_CONFLICT | event_000012 |

**Events generated:** 12 (RISK_ESCALATED 5, RISK_DEESCALATED 5, PERSISTENT_HAZARD 2). They are in `outputs/demo/events.jsonl` and `events.parquet`, with run id `demo_video_20260926_170005`.

**Alerts recommended:** 5 escalation alerts (NOTICE 1, WARNING 4) and 5 de-escalation notices. **4 alarm(s) raised.**

| time | level | alert | alarm | why (engine risk_reason) | event |
| --- | --- | --- | --- | --- | --- |
| 00:00.5 | CAUTION | CAUTION: hazard noted |  | Persistent person #2 (observed 0.5 s, 6 detections): moving across the central band at 0.07 frame-widths/s (image space), in the central ima | event_000001 |
| 00:01.5 | HIGH | WARNING: ROAD HAZARD | yes | Persistent person #2 (observed 1.5 s, 16 detections): image-space box area growing 21%/s, moving across the central band at 0.08 frame-width | event_000002 |
| 00:04.2 | HIGH | WARNING: ROAD HAZARD | yes | Persistent person #2 (observed 4.2 s, 43 detections): image-space box area growing 26%/s, moving across the central band at 0.06 frame-width | event_000004 |
| 00:04.6 | HIGH | WARNING: ROAD HAZARD | yes | Persistent bicycle #49 (observed 0.7 s, 8 detections): image-space box area growing 1168%/s, moving across the central band at 0.06 frame-wi | event_000006 |
| 00:06.9 | HIGH | WARNING: ROAD HAZARD | yes | Persistent car #50 (observed 2.9 s, 30 detections): image-space box area growing 94%/s, in the central image band (+2 other road hazards). | event_000008 |

**Evidence files:** 14 files for 7 events, under `outputs/demo/evidence/`. Missing: 0.

**Other outputs:** `timeline.csv`, `risk_timeline.png`, `alerts.jsonl`, `demo_summary.json`; annotated video generated: **yes**.

**Provenance:** events data_source LOCAL_REAL 12; observation_type INFERRED 12.
- Synthetic rows included: 0.
- Driver input: none (front camera only).
- Synchronised driver + road: False.
- GPS: UNAVAILABLE (no real GPS log).

## Driver-camera demo: `video_20260926_220714.mp4`

- **Exact real input video:** `<DATASET_ROOT>/drivercamera-20260926T193339Z-1-001/drivercamera/drowsy_driver/video_20260926_220714.mp4`
  - sha256 `5cbb9e3b7f6f0954…`
  - 3840×2160, container 60.0 fps, 377 frames
- **Duration processed:** 6.267 s of video (container duration 6.28 s).
- **Frames:** 377 decoded; 63 processed at about 10.05 fps. Wall time 26.5 s.
- **Driver observations:** 63.
  - Drowsiness levels: UNKNOWN 38, HIGH 12, CRITICAL 13.
  - Hand states: ONE_HAND 16, UNKNOWN 26, NO_HANDS 21.
- **Risk:** frames per level SAFE 37, CAUTION 26. Peak CAUTION (score 45.0).
- **Alarm-recommended frames:** 0.

**Risk transitions**

| time | transition | hazard type | event |
| --- | --- | --- | --- |
| 00:01.5 | SAFE->CAUTION | DRIVER_DROWSINESS | event_000001 |
| 00:02.8 | CAUTION->SAFE | NONE | event_000004 |
| 00:05.0 | SAFE->CAUTION | DRIVER_DROWSINESS | event_000005 |

**Events generated:** 7 (RISK_ESCALATED 2, DRIVER_STATE_CHANGE 4, RISK_DEESCALATED 1). They are in `outputs/demo_driver/events.jsonl` and `events.parquet`, with run id `demo_video_20260926_220714`.

**Alerts recommended:** 2 escalation alerts (NOTICE 2) and 1 de-escalation notices. **0 alarm(s) raised.**

| time | level | alert | alarm | why (engine risk_reason) | event |
| --- | --- | --- | --- | --- | --- |
| 00:01.5 | CAUTION | CAUTION: hazard noted |  | Drowsiness level HIGH, score 0.50; no road hazard with approach/conflict evidence. | event_000001 |
| 00:05.0 | CAUTION | CAUTION: hazard noted |  | Drowsiness level HIGH, score 0.73; no road hazard with approach/conflict evidence. | event_000005 |

**Evidence files:** 12 files for 6 events, under `outputs/demo_driver/evidence/`. Missing: 0.

**Other outputs:** `timeline.csv`, `risk_timeline.png`, `alerts.jsonl`, `demo_summary.json`; annotated video generated: **yes**.

**Provenance:** events data_source LOCAL_REAL 7; observation_type INFERRED 5, OBSERVED 2.
- Synthetic rows included: 0.
- Driver input: driver camera only (no road input).
- Synchronised driver + road: False.
- GPS: UNAVAILABLE (no real GPS log).

## Limitations

- **No synchronized real driver+road recording exists.** Front and driver demos are separate runs on separate videos, so no real CRITICAL "driver + road" alert can occur. The engine gates CRITICAL on that combination, or on a very strong road hazard. The combination path is exercised only with synthetic test input (tests) and the labelled SYNTHETIC_COMBINATION / synthetic scenario datasets.
- **Image-space road signals only.** There is no calibration, distance, physical speed or TTC. Motion of the camera itself shows up as object motion, so a vehicle can look like it is "approaching" because the ego car moves. This was checked on the real clips earlier, and it is a known source of HIGH levels.
- **Prototype score and thresholds.** The risk score is rule-based and not a probability. Levels are decision-support states, and alerts are not collision predictions.
- **Perception errors propagate.** Errors from the pretrained detector (e.g. label flicker) and the tracker (ID switches) flow into risk and events. The hand-state classifier generalises poorly to the demo driver setup (see `driver_camera_domain_comparison.md`).
- **Offline alarm output.** The alarm is a console / terminal-bell output in an offline replay (`--beep` enables the bell), not an in-vehicle system. Timing is video time, not measured latency.
- **No GPS.** No GPS log exists, so coordinates stay null / UNAVAILABLE.
- **`alarm_triggered` stays false.** Event schema 1.0 requires `alarm_triggered = false`, and the schema was not changed. Raised alarms are recorded in `alerts.jsonl` and linked to event ids.
- **Short clips.** The front clips are about 1–11 s long, each starting from SAFE.
- **Track fragmentation in `video_20260926_170005.mp4`:** 148 tracks from 613 detections over 115 processed frames. Tracks break on label flicker and busy scenes, and a newly started track's growth estimate can be extreme (see the `why` column above).
