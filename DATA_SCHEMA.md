# Data schema

This is a human-readable guide to the two schemas in [`data/schema/`](data/schema/). The field names, types, provenance tags and descriptions below are taken **verbatim** from the JSON schema files; this page only adds explanation. If this page and the JSON ever disagree, the JSON wins.

- [`real_event_schema.json`](data/schema/real_event_schema.json): schema version `1.0`, 60 fields. One row per real event.
- [`synthetic_schema.json`](data/schema/synthetic_schema.json): schema version `sim-1.0`, 73 fields. One row per 10 Hz step of a SYNTHETIC scenario.

Units everywhere are image-space pixels and seconds: *"No metres, physical speed, distance or TTC anywhere."*

## 1. Real event schema (1.0)

### Record-level provenance

| Field | Values |
| --- | --- |
| `data_source` | `LOCAL_REAL` (captured by the team) · `PUBLIC` (public dataset) · `SYNTHETIC` (constructed test input) · `SYNTHETIC_COMBINATION` (real streams combined although they were NOT recorded together). The public sample contains only `LOCAL_REAL`. |
| `observation_type` | `OBSERVED` (the event is a change in an observed signal) · `INFERRED` (the event is derived from inferred quantities). Every risk/hazard event is INFERRED. |

Field provenance: **METADATA** = recorder bookkeeping; **OBSERVED** = measured directly; **INFERRED** = derived from observations by the tracker or rules.

### Key fields explained

| Field | What it means in practice |
| --- | --- |
| `event_id` | `event_000001 …`, sequential and deterministic **within a run** (`run_id`, here `local_real_run_01`). The pair (`run_id`, `event_id`) is unique. |
| `timestamp` | Seconds since the start of the source video (stream time). There is no absolute clock. |
| `event_type` | `RISK_ESCALATED`, `RISK_DEESCALATED`, `PERSISTENT_HAZARD`, `DRIVER_STATE_CHANGE`. |
| `risk_level` | Level **after hysteresis**: `SAFE`, `CAUTION`, `HIGH`, `CRITICAL`. A decision-support state, not a validated classification. Null for driver-state changes. |
| `raw_risk_score` / `smoothed_risk_score` | Rule-based prototype score 0–100 for this observation, and its smoothed version. **Not a probability.** The risk engine's `risk_score` is the smoothed score. |
| `hazard_type` | The risk engine's hazard category: `NONE`, `APPROACHING_VEHICLE`, `PEDESTRIAN_CONFLICT`, `ANIMAL_HAZARD`, `ROAD_USER_PRESENT`, `DRIVER_DROWSINESS`, `HANDS_OFF_WHEEL`, `DRIVER_HEAD_AWAY`, `COMBINED_DRIVER_HAZARD`. |
| `hazard_class` vs `hazard_track_class` | The detector's label **at the event frame** (OBSERVED) vs the track's majority label (INFERRED). They can differ when labels flicker; `hazard_class_changes` counts that. |
| `primary_track_id` | Tracker id of the object that drives the hazard (identity is inferred by IoU matching). |
| `track_persistence_seconds` / `track_detection_count` | How long and how often the track was observed. |
| `hazard_persistence` | How long **this hazard** (same type + track) has been reported continuously. |
| `box_growth_per_second` | The image-space **approach proxy**: slope of ln(box area) over ~1 s. A relative "getting bigger in the image" signal, **not** distance or speed. The synthetic schema maps the requested name `approach_rate` to this field. |
| `lateral_toward_center_fw_per_s` | Horizontal motion toward or across the central image band, in frame-widths per second. |
| `trajectory_overlap` | **Reserved; always null** in real data, because there is no calibrated path model. |
| `driver_hand_state`, `driver_hand_confidence` | Hand-state classifier output (`BOTH_HANDS`/`ONE_HAND`/`NO_HANDS`/`UNKNOWN`), treated as an observed per-frame signal. |
| `driver_activity` | INFERRED over time, e.g. `HANDS_OFF_WHEEL` only after NO_HANDS is held ≥ 2 s. |
| `drowsiness_level`, `drowsiness_score` | Prototype temporal eye-closure signal (0–1) and its level. **Not medically validated.** |
| `face_status` | Whether a face was found and landmarks computed in that frame. |
| `risk_reason`, `risk_factors`, `risk_gates` | The engine's generated explanation (no LLM), the point contributions, and the level caps that applied. |
| `alarm_recommended` | The engine's recommendation (level ≥ the configured alarm level, HIGH by default). |
| `alarm_triggered` | **Always false** in schema 1.0: the event layer records no alarm output. Alarms raised by the demo's alert layer are logged separately in `alerts.jsonl`, linked by event id. |
| `gps_lat`, `gps_lon`, `gps_source` | Filled only from a real GPS log. In this dataset, always `null` and `UNAVAILABLE`, never invented. |
| `evidence_kind`, `evidence_path`, `evidence_files` | `NONE` / `FRAME` / `CLIP`, plus paths **relative to the evidence root**, e.g. `local_real_run_01/event_000012`. |
| `notes` | Provenance and limitation notes for the row (JSON list). |

### Validation rules enforced by the writer (`app/events/schema.py`)

- **Shape:** every field is present, with its declared type and nullability, and allowed values are respected.
- **Observation type:** risk and hazard events must be `INFERRED`. A hand-state change must be `OBSERVED`, and drowsiness or activity changes must be `INFERRED`.
- **Level events:** the direction of a level change must match `RISK_ESCALATED` / `RISK_DEESCALATED`.
- **GPS:** `gps_lat` and `gps_lon` are both set or both null, and `UNAVAILABLE` means no coordinates.
- **Constant fields:** `trajectory_overlap` must be null, and `alarm_triggered` must be false.
- **Evidence paths:** they must be relative, with no drive letters or `..`.
- **Writers:** the real writer refuses SYNTHETIC records, and the synthetic writer refuses real ones.

### All 60 fields (verbatim from the JSON)

| Field | Type | Nullable | Provenance | Description |
| --- | --- | --- | --- | --- |
| `schema_version` | string | no | METADATA | Event schema version. |
| `event_id` | string | no | METADATA | Sequential id within a run: event_000001, event_000002, ... (deterministic). |
| `run_id` | string | no | METADATA | Identifier of the recording/processing run that produced the event. |
| `session_id` | string | no | METADATA | Stream the event belongs to (e.g. the source video's file stem). |
| `source_file` | string | yes | METADATA | Source file name (no directory), when the stream is a file. |
| `camera` | string | no | METADATA | Which camera stream(s) the event is based on. Allowed: front, driver, front+driver. |
| `data_source` | string | no | METADATA | LOCAL_REAL (our own captures) / PUBLIC / SYNTHETIC (constructed test inputs) / SYNTHETIC_COMBINATION (real streams combined although not recorded together). Synthetic records live in a separate dataset. Allowed: LOCAL_REAL, PUBLIC, SYNTHETIC, SYNTHETIC_COMBINATION. |
| `observation_type` | string | no | METADATA | INFERRED when the event is derived from inferred quantities (all risk / hazard events); OBSERVED only for a DRIVER_STATE_CHANGE of the observed hand state. Allowed: OBSERVED, INFERRED. |
| `timestamp` | float64 | no | OBSERVED | Stream time of the event, seconds (video timestamps for files). |
| `frame_index` | int64 | yes | OBSERVED | Index of the frame that produced the event, when known. |
| `event_type` | string | no | INFERRED | Kind of event. Allowed: RISK_ESCALATED, RISK_DEESCALATED, PERSISTENT_HAZARD, DRIVER_STATE_CHANGE. |
| `transition` | string | yes | INFERRED | FROM->TO for level and driver-state changes, e.g. SAFE->CAUTION. |
| `previous_risk_level` | string | yes | INFERRED | Risk level before a level change. Allowed: SAFE, CAUTION, HIGH, CRITICAL. |
| `change_kind` | string | yes | INFERRED | For DRIVER_STATE_CHANGE: hand_state / driver_activity / drowsiness_level. Allowed: hand_state, driver_activity, drowsiness_level. |
| `gps_lat` | float64 | yes | OBSERVED | Latitude from a real GPS source; null when unavailable. Never invented. |
| `gps_lon` | float64 | yes | OBSERVED | Longitude from a real GPS source; null when unavailable. Never invented. |
| `gps_source` | string | no | METADATA | GPS source name, or UNAVAILABLE. |
| `gps_timestamp` | float64 | yes | OBSERVED | Time of the GPS fix used, seconds (same clock as timestamp). |
| `gps_accuracy_m` | float64 | yes | OBSERVED | Reported GPS accuracy in metres, when the source provides it. |
| `risk_level` | string | yes | INFERRED | Risk level after hysteresis (decision-support state, not validated). Allowed: SAFE, CAUTION, HIGH, CRITICAL. |
| `raw_risk_score` | float64 | yes | INFERRED | Rule-based prototype score 0-100 for this observation. Not a probability. |
| `smoothed_risk_score` | float64 | yes | INFERRED | Smoothed prototype score 0-100. Not a probability. |
| `hazard_type` | string | yes | INFERRED | Risk engine hazard type (NONE, APPROACHING_VEHICLE, ...). |
| `primary_track_id` | int64 | yes | INFERRED | Tracker id of the object driving the hazard (identity is inferred). |
| `image_width` | int64 | yes | OBSERVED | Front-camera frame width in pixels. |
| `image_height` | int64 | yes | OBSERVED | Front-camera frame height in pixels. |
| `hazard_class` | string | yes | OBSERVED | Detector label of the primary object at the event time. |
| `hazard_track_class` | string | yes | INFERRED | Track-level label (most frequent detector label of the track). |
| `hazard_class_changes` | int64 | yes | INFERRED | Detector label changes observed on the primary track. |
| `hazard_confidence` | float64 | yes | OBSERVED | Detector confidence of the primary object at the event time. |
| `track_persistence_seconds` | float64 | yes | INFERRED | How long the primary track has been observed. |
| `track_detection_count` | int64 | yes | INFERRED | Detections supporting the primary track. |
| `center_x` | float64 | yes | OBSERVED | Primary object box centre x, pixels. |
| `center_y` | float64 | yes | OBSERVED | Primary object box centre y, pixels. |
| `bbox_width` | float64 | yes | OBSERVED | Primary object box width, pixels. |
| `bbox_height` | float64 | yes | OBSERVED | Primary object box height, pixels. |
| `bbox_area` | float64 | yes | OBSERVED | Primary object box area, pixels². |
| `velocity_x_pixels_per_second` | float64 | yes | INFERRED | Image-space centre velocity x (not physical speed). |
| `velocity_y_pixels_per_second` | float64 | yes | INFERRED | Image-space centre velocity y (not physical speed). |
| `area_change_rate` | float64 | yes | INFERRED | Box area change, pixels² per second (tracker, frame to frame). |
| `box_growth_per_second` | float64 | yes | INFERRED | Risk-engine growth: slope of ln(box area) over ~1 s (relative approach signal, not distance). |
| `lateral_toward_center_fw_per_s` | float64 | yes | INFERRED | Horizontal motion toward/across the central image band, frame-widths per second. |
| `trajectory_overlap` | float64 | yes | INFERRED | Reserved: overlap of the object's path with the vehicle's projected path. Always null until a calibrated path model exists. |
| `hazard_persistence` | float64 | yes | INFERRED | How long this hazard (same hazard type and track) has been continuously reported, seconds. |
| `driver_hand_state` | string | yes | OBSERVED | Hand-state classifier output (BOTH_HANDS / ONE_HAND / NO_HANDS / UNKNOWN). |
| `driver_hand_confidence` | float64 | yes | OBSERVED | Hand-state classifier confidence. |
| `driver_activity` | string | yes | INFERRED | Inferred driver activity (NORMAL, HANDS_OFF_WHEEL, ...). |
| `drowsiness_level` | string | yes | INFERRED | Temporal drowsiness level (prototype, not medically validated). |
| `drowsiness_score` | float64 | yes | INFERRED | Temporal drowsiness score 0-1 (prototype). |
| `face_status` | string | yes | OBSERVED | Driver face-landmark status for the frame. |
| `risk_reason` | string | yes | INFERRED | Reason generated by the rule engine from structured factors (no LLM). |
| `risk_factors` | json | yes | INFERRED | Contributing factors [{name, points, source, detail, track_id}]. |
| `risk_gates` | json | yes | INFERRED | Level caps applied by the risk engine. |
| `evidence_quality` | float64 | yes | INFERRED | 0-1 support of the evidence behind the risk result (not a probability). |
| `alarm_recommended` | bool | yes | INFERRED | Risk engine recommendation (level >= configured alarm level). |
| `alarm_triggered` | bool | no | METADATA | Whether an alarm was actually raised. Always false: no alarm output exists yet. |
| `evidence_kind` | string | no | METADATA | NONE / FRAME (single images) / CLIP (video clips). Allowed: NONE, FRAME, CLIP. |
| `evidence_path` | string | yes | METADATA | Evidence directory relative to the event dataset root: <run_id>/<event_id>. |
| `evidence_files` | json | yes | METADATA | Evidence files relative to the event dataset root. |
| `notes` | json | yes | METADATA | Provenance and limitation notes for this record. |
## 2. Synthetic scenario schema (sim-1.0)

**Row definition:** one row = one 10 Hz step of a constructed scenario. **Nothing in a synthetic row was observed.**

**Record-level markers:**
- `data_source = observation_type = timestamp_source = gps_source = SYNTHETIC`;
- `location_source = SYNTHETIC_REFERENCE`;
- `event_id` uses the form `sim_NNNNNN`, never the real `event_NNNNNN` form.

Field provenance in the synthetic schema:

| Tag | Meaning (verbatim) |
| --- | --- |
| METADATA | ids, labels and provenance markers |
| SYNTHETIC_INPUT | constructed by the generator (time, location, boxes, labels, confidences, driver signals, design ground truth) |
| ENGINE_ON_SYNTHETIC | computed by the existing, unmodified tracker / driver temporal logic / RiskEngine / EventRecorder from the synthetic inputs |
| VALIDATION | expected qualitative behaviour vs the engine's actual output |

**Fields specific to the synthetic schema:**
- **Provenance and linkage:** `synthetic_id`, `scenario_name`, `scenario_variant`, `scenario_description`, `synthetic_generation_reason`, and `parent_real_event_id` / `parent_real_driver_event_id` / `parent_real_run_id`. The parent ids link to real events used only as value references.
- **Driver features:** `head_yaw`, `head_pitch`, `eye_closure_ratio`, `eye_closed_duration`, `distraction_duration`, `manual_state_duration`, `head_away_duration`.
- **Engine and events:** `raw_risk_level`, `risk_score` (= smoothed), `event_type` / `events_emitted` (what the existing event recorder emitted at that step).
- **Validation:** `expected_behavior`, `expected_peak_levels`, `actual_risk_level`, `instance_peak_risk_level`, `expectation_met`.

**Name mapping:** `object_id` → `primary_track_id`, and `approach_rate` → `box_growth_per_second`.

**`trajectory_overlap`** differs by dataset:
- **Synthetic data:** it is *design ground truth*, the image-space share of the constructed box inside the central band. It is not an input to the risk engine.
- **Real data:** it stays null.

### All 73 synthetic fields

| Field | Type | Provenance | Reused from event schema |
| --- | --- | --- | --- |
| `schema_version` | string | METADATA | new |
| `event_id` | string | METADATA | yes |
| `synthetic_id` | string | METADATA | new |
| `run_id` | string | METADATA | yes |
| `session_id` | string | METADATA | yes |
| `scenario_name` | string | METADATA | new |
| `scenario_variant` | int64 | METADATA | new |
| `scenario_description` | string | METADATA | new |
| `synthetic_generation_reason` | string | METADATA | new |
| `parent_real_event_id` | string | METADATA | new |
| `parent_real_driver_event_id` | string | METADATA | new |
| `parent_real_run_id` | string | METADATA | new |
| `data_source` | string | METADATA | yes |
| `observation_type` | string | METADATA | new |
| `camera` | string | METADATA | yes |
| `timestamp` | float64 | SYNTHETIC_INPUT | yes |
| `timestamp_source` | string | METADATA | new |
| `synthetic_datetime` | string | SYNTHETIC_INPUT | new |
| `frame_index` | int64 | SYNTHETIC_INPUT | yes |
| `gps_lat` | float64 | SYNTHETIC_INPUT | yes |
| `gps_lon` | float64 | SYNTHETIC_INPUT | yes |
| `gps_source` | string | METADATA | yes |
| `location_source` | string | METADATA | new |
| `driver_hand_state` | string | SYNTHETIC_INPUT | yes |
| `driver_hand_confidence` | float64 | SYNTHETIC_INPUT | yes |
| `face_status` | string | SYNTHETIC_INPUT | yes |
| `head_yaw` | float64 | SYNTHETIC_INPUT | new |
| `head_pitch` | float64 | SYNTHETIC_INPUT | new |
| `eye_closure_ratio` | float64 | ENGINE_ON_SYNTHETIC | new |
| `eye_closed_duration` | float64 | ENGINE_ON_SYNTHETIC | new |
| `drowsiness_score` | float64 | ENGINE_ON_SYNTHETIC | yes |
| `drowsiness_level` | string | ENGINE_ON_SYNTHETIC | yes |
| `driver_activity` | string | ENGINE_ON_SYNTHETIC | yes |
| `distraction_duration` | float64 | ENGINE_ON_SYNTHETIC | new |
| `manual_state_duration` | float64 | ENGINE_ON_SYNTHETIC | new |
| `head_away_duration` | float64 | ENGINE_ON_SYNTHETIC | new |
| `image_width` | int64 | SYNTHETIC_INPUT | yes |
| `image_height` | int64 | SYNTHETIC_INPUT | yes |
| `hazard_class` | string | SYNTHETIC_INPUT | yes |
| `hazard_confidence` | float64 | SYNTHETIC_INPUT | yes |
| `center_x` | float64 | SYNTHETIC_INPUT | yes |
| `center_y` | float64 | SYNTHETIC_INPUT | yes |
| `bbox_width` | float64 | SYNTHETIC_INPUT | yes |
| `bbox_height` | float64 | SYNTHETIC_INPUT | yes |
| `bbox_area` | float64 | SYNTHETIC_INPUT | yes |
| `primary_track_id` | int64 | ENGINE_ON_SYNTHETIC | yes |
| `hazard_track_class` | string | ENGINE_ON_SYNTHETIC | yes |
| `track_persistence_seconds` | float64 | ENGINE_ON_SYNTHETIC | yes |
| `box_growth_per_second` | float64 | ENGINE_ON_SYNTHETIC | yes |
| `lateral_toward_center_fw_per_s` | float64 | ENGINE_ON_SYNTHETIC | yes |
| `trajectory_overlap` | float64 | SYNTHETIC_INPUT | yes |
| `hazard_persistence` | float64 | ENGINE_ON_SYNTHETIC | yes |
| `risk_level` | string | ENGINE_ON_SYNTHETIC | yes |
| `raw_risk_level` | string | ENGINE_ON_SYNTHETIC | new |
| `raw_risk_score` | float64 | ENGINE_ON_SYNTHETIC | yes |
| `smoothed_risk_score` | float64 | ENGINE_ON_SYNTHETIC | yes |
| `risk_score` | float64 | ENGINE_ON_SYNTHETIC | new |
| `hazard_type` | string | ENGINE_ON_SYNTHETIC | yes |
| `risk_reason` | string | ENGINE_ON_SYNTHETIC | yes |
| `risk_factors` | json | ENGINE_ON_SYNTHETIC | yes |
| `risk_gates` | json | ENGINE_ON_SYNTHETIC | yes |
| `evidence_quality` | float64 | ENGINE_ON_SYNTHETIC | yes |
| `alarm_recommended` | bool | ENGINE_ON_SYNTHETIC | yes |
| `alarm_triggered` | bool | METADATA | yes |
| `event_type` | string | ENGINE_ON_SYNTHETIC | new |
| `transition` | string | ENGINE_ON_SYNTHETIC | yes |
| `events_emitted` | json | ENGINE_ON_SYNTHETIC | new |
| `expected_behavior` | string | VALIDATION | new |
| `expected_peak_levels` | json | VALIDATION | new |
| `actual_risk_level` | string | VALIDATION | new |
| `instance_peak_risk_level` | string | VALIDATION | new |
| `expectation_met` | bool | VALIDATION | new |
| `notes` | json | METADATA | yes |
