# Event dataset: first recording on real local data

The dataset is event-driven: one row per meaningful change, not per frame. The records are produced by `app/events` (recorder, evidence, writer, schema) from the existing perception → tracking → risk outputs. Command:

```bash
python scripts/record_events.py \
  --front-source  "<DATASET_ROOT>/frontcamera-20260926T193501Z-1-001/frontcamera" \
  --driver-source "<DATASET_ROOT>/drivercamera-20260926T193339Z-1-001/drivercamera/drowsy_driver" \
  --synthetic-combination "<...>/drowsy_driver/video_20260926_220714.mp4" "<...>/vehicles/video_20260926_165951.mp4"
```

## Storage

| Dataset | Contents | Files |
| --- | --- | --- |
| `data/events/` | **LOCAL_REAL** only. The writer refuses any SYNTHETIC record. | `events.jsonl`, `events.parquet` (pyarrow), `schema.json`, `local_real_run_01/event_NNNNNN/` evidence |
| `data/events_synthetic/` | **SYNTHETIC / SYNTHETIC_COMBINATION** only. The writer refuses real records. | the same layout, run `synthetic_combination_run_01` |

- The schema is version 1.0: 60 fields, each with a type, nullability, provenance (METADATA / OBSERVED / INFERRED) and description. It is exported to `schema.json` and enforced on every record.
- JSON Lines is always written. Parquet is written when `pyarrow` is installed; it is now in `requirements.txt`, and without it the writer falls back to JSONL and says so.
- Every record here passed validation, and the Parquet file has 66 rows with the same columns.

## Source and collection window (REAL)

**Source:** the team's own recordings (`data_source = LOCAL_REAL`).
- **Road-only:** 14 front-camera videos, 50.1 s in total, sampled at 10 fps (515 frames).
- **Driver-only:** 3 drowsy-driver videos, 16.5 s in total, sampled at 10 fps (168 driver observations).

**Collection window:** the recordings are from 26 Sep 2026, 16:59–22:08 local time, **according to their file names**. The files carry no other clock.

**Synchronisation and GPS:**
- **Not synchronised:** the driver and front cameras were **not recorded together**, so no combined real drive exists. Road-only and driver-only sequences were recorded as separate sessions.
- **No GPS:** no GPS log exists, so every record has `gps_lat = gps_lon = null` and `gps_source = "UNAVAILABLE"`. The GPS interface (`app/sensors/gps.py`, CSV log provider) is ready but was not used.

## Real events (`data/events`, run `local_real_run_01`)

| | Count |
| --- | ---: |
| **Total events** | **66** (all LOCAL_REAL) |
| RISK_ESCALATED | 32 (road 28, driver-only 4) |
| RISK_DEESCALATED | 18 (road 17, driver-only 1) |
| PERSISTENT_HAZARD | 9 (road) |
| DRIVER_STATE_CHANGE | 7 (driver-only) |
| observation_type INFERRED / OBSERVED | 64 / 2 (the 2 OBSERVED are hand-state changes) |
| camera front / driver | 54 / 12 |
| Sessions (videos) with at least one event | 16 of 17 (`dogs_on_road/video_20260926_211200` produced none) |

**Risk-level transitions** (RISK_ESCALATED / RISK_DEESCALATED):

| Transition | Count |
| --- | ---: |
| SAFE→CAUTION | 21 |
| CAUTION→HIGH | 11 |
| HIGH→CAUTION | 9 |
| CAUTION→SAFE | 9 |

No real risk event reached CRITICAL: without synchronised driver input, CRITICAL is gated. The driver-only risk events are DRIVER_DROWSINESS at CAUTION. A single driver factor cannot pass the HIGH gate, which is the risk engine's documented behaviour.

**PERSISTENT_HAZARD events** (hazard reported continuously ≥ 2 s at CAUTION or above, one per hazard type + track per 10 s):

| Level | Count |
| --- | ---: |
| HIGH | 5 |
| CAUTION | 4 |

| Hazard type | Count |
| --- | ---: |
| APPROACHING_VEHICLE | 5 (car ×4, truck ×1) |
| PEDESTRIAN_CONFLICT | 4 |

**DRIVER_STATE_CHANGE events** (the new state must hold for 3 consecutive observations; UNKNOWN ignored; 5 s cooldown per kind and transition):

| Change | Count | Observation type |
| --- | ---: | --- |
| drowsiness HIGH→CRITICAL | 3 | INFERRED |
| drowsiness CRITICAL→HIGH | 2 | INFERRED |
| hand ONE_HAND→NO_HANDS | 1 | OBSERVED |
| hand NO_HANDS→ONE_HAND | 1 | OBSERVED |

**By hazard type (all 66):**

| Hazard type | Events |
| --- | ---: |
| PEDESTRIAN_CONFLICT | 28 |
| APPROACHING_VEHICLE | 19 |
| null (driver-state changes) | 7 |
| NONE | 4 |
| DRIVER_DROWSINESS | 4 |
| ANIMAL_HAZARD | 2 |
| ROAD_USER_PRESENT | 2 |

**Evidence:** 48 events have single-frame evidence: 96 files, 4.0 MB, as `metadata.json` plus `front_frame.jpg` or `driver_frame.jpg`, downscaled to 960 px. The 18 de-escalations have none, by configuration. Every listed file exists, and all paths are relative (`local_real_run_01/event_000012/front_frame.jpg`).

Two were checked by eye:
- `event_000012` (PERSISTENT_HAZARD, pedestrian crossing) shows the pedestrian on the crossing.
- `event_000056` (hand ONE_HAND→NO_HANDS) shows the driver's eyes closed, consistent with drowsiness CRITICAL. However, a hand is visibly on the wheel, so this **hand-state event is a classifier error**. That classifier is known to generalise poorly to this driver setup (see `driver_camera_domain_comparison.md`).

**Event cadence:** 54 road events over 50.1 s of video, and 12 driver events over 16.5 s. The source clips are short (1–11 s) and each starts from SAFE. Most clips therefore produce an initial escalation, so the rate per minute is higher than a continuous drive would give. Without deduplication the same data would have produced one row per sampled frame: 683 rows. The cooldowns and hysteresis kept repeated frames out.

## SYNTHETIC events (`data/events_synthetic`, run `synthetic_combination_run_01`)

This is a **software test only**. Real driver signals from `drowsy_driver/video_20260926_220714.mp4` were paired by elapsed time with the real front video `vehicles/video_20260926_165951.mp4`. The cameras were **not recorded together**, so this is **not a drive**. It exercises the schema's ability to hold combined driver + road events.

- 9 events, all `data_source = SYNTHETIC_COMBINATION`, `camera = front+driver`: RISK_ESCALATED 3 (SAFE→CAUTION, CAUTION→HIGH, HIGH→CRITICAL), RISK_DEESCALATED 2, PERSISTENT_HAZARD 2, DRIVER_STATE_CHANGE 2.
- Levels: CRITICAL 2 (COMBINED_DRIVER_HAZARD), HIGH 2, CAUTION 3. There are 21 evidence files: front and driver frames per event.
- Every record carries the note *"SYNTHETIC_COMBINATION: real driver and front streams that were NOT recorded together; a software test, not a drive"*.

## Event triggers (configuration `app/config/event_config.py`, `EVENT_*` env vars)

| Event | Trigger | Dedup |
| --- | --- | --- |
| RISK_ESCALATED / RISK_DEESCALATED | every change of the risk level after the engine's hysteresis | none needed: one event per real change, always recorded |
| PERSISTENT_HAZARD | the same hazard type + track reported continuously ≥ 2.0 s while the risk level is ≥ CAUTION | one per hazard type + track per 10 s |
| DRIVER_STATE_CHANGE | hand_state, driver_activity or drowsiness_level changes and holds for 3 consecutive observations (UNKNOWN ignored) | one per (kind, from, to) per 5 s |

Evidence is recorded for RISK_ESCALATED, PERSISTENT_HAZARD and DRIVER_STATE_CHANGE.
- **Frame mode (used here):** one JPEG per camera from the event frame.
- **Clip mode (implemented and tested, not used for this run):** a ring buffer keeps 5 s before the event and collects 5 s after it, then writes `front.mp4` / `driver.mp4`. The record is only released once the clip file exists. Clips cut short by the stream start or end are reported in `metadata.json`.

## Fields and their provenance

**OBSERVED** (measured directly):
- `timestamp` and `frame_index`;
- detector output for the primary object: `hazard_class`, `hazard_confidence`, `center_x/y`, `bbox_width/height/area`, `image_width/height`;
- driver per-frame signals: `driver_hand_state`, `driver_hand_confidence` (hand-classifier output, treated as observed in this project), `face_status`;
- `gps_*`, only when a real GPS source supplied it (never in this dataset).

**INFERRED** (derived by tracker or rules):
- track identity: `primary_track_id`, `hazard_track_class`, `hazard_class_changes`, `track_persistence_seconds`, `track_detection_count`;
- image-space motion: `velocity_x/y_pixels_per_second`, `area_change_rate`, `box_growth_per_second`, `lateral_toward_center_fw_per_s`;
- `hazard_persistence`;
- risk outputs: `risk_level`, `raw_risk_score`, `smoothed_risk_score`, `hazard_type`, `risk_reason`, `risk_factors`, `risk_gates`, `evidence_quality`, `alarm_recommended`;
- driver inferences: `driver_activity`, `drowsiness_level`, `drowsiness_score`;
- `event_type`, `transition`, `previous_risk_level`, `change_kind`.

**METADATA:**
- identifiers: `schema_version`, `event_id` (sequential per run, deterministic), `run_id`, `session_id`, `source_file`, `camera`;
- provenance: `data_source`, `observation_type`;
- `gps_source`;
- evidence: `evidence_kind`, `evidence_path`, `evidence_files`;
- `alarm_triggered` (always false: no alarm output exists yet);
- `notes`.

**Record-level provenance:**
- `data_source`: LOCAL_REAL | PUBLIC | SYNTHETIC | SYNTHETIC_COMBINATION.
- `observation_type`: INFERRED for every risk and hazard event. OBSERVED only for a change of the observed hand state.
- Validation rejects an OBSERVED risk event.
- `trajectory_overlap` is reserved and always null, because there is no calibrated path model. Validation rejects any value.

**SYNTHETIC in this project:** the rule-engine stress tests in `scripts/test_risk_engine.py` part A (not written to any dataset), and unsynchronised driver + road combinations (`data/events_synthetic`, SYNTHETIC_COMBINATION).

Representative records: `outputs/reports/event_dataset_sample.json` (6 LOCAL_REAL records, and 1 clearly labelled SYNTHETIC_COMBINATION record).

## Known limitations

- **No GPS:** no GPS was available. All coordinates are null, so no hotspot analysis is possible from this data yet.
- **No synchronised driver + road recording exists.** Real events are road-only or driver-only, and combined events exist only as SYNTHETIC_COMBINATION.
- **Everything derived is INFERRED from prototype components**, and errors propagate into events:
  - YOLO (e.g. the black dog at night labelled person);
  - the tracker (broken or duplicate tracks);
  - the hand classifier (e.g. the NO_HANDS event above with a hand on the wheel);
  - the rule-based risk engine, whose thresholds are prototype values and whose score is not a probability.
- **Image-space road values:** no calibration, distance, physical speed or TTC. Camera ego-motion shows up as image motion. APPROACHING_VEHICLE means relative approach in the image.
- **No ground truth:** there are no collision or near-miss labels. Event counts describe rule behaviour, not safety outcomes.
- **Short clips inflate the event rate:** each session starts at SAFE, so short clips produce proportionally more escalation events.
- **Frame evidence only in this run:** a single frame per camera. Clip mode exists but was not used, to keep the dataset small.
- **Timestamps are video stream time per session;** there is no absolute clock beyond the file names.
