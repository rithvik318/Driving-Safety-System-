# `app/simulation`: feature-level synthetic scenarios

> **Everything produced here is SYNTHETIC. It is not an observation and does not replace real data.**

```bash
python scripts/generate_synthetic_scenarios.py            # seed 42, 2 variants x 12 scenarios, ~1,400 rows
python scripts/generate_synthetic_scenarios.py --seed 7 --variants 3
```

Output goes to `data/simulated/`:

| File | Contents |
| --- | --- |
| `scenarios.jsonl` | One row per 10 Hz step. |
| `scenarios.parquet` | Same rows (pyarrow). |
| `schema.json` | Schema `sim-1.0`: every field with its type, nullability and provenance. |
| `manifest.json` | Seed, counts, parents, provenance policy, reference ranges, expected-vs-actual results. |

The report is written to `outputs/reports/synthetic_scenario_report.md`.

## How a row is made

```text
scenarios.py   constructed ObjectStep (image-space box, label, confidence)  ─┐
               scripted DriverStep (head yaw/pitch, eye openness, hand state) │ per 10 Hz step
                                                                              ▼
generator.py   Detection ─► IoUTracker (existing)
               DriverStep ─► ScriptedDriverPipeline = existing DriverPerceptionPipeline with the camera
                             replaced (eye-closure tracking, drowsiness, durations, activity inference unchanged)
               tracks + DriverState ─► RiskEngine.evaluate (existing, default RiskConfig, unmodified)
               assessment ─► EventRecorder (existing, data_source=SYNTHETIC, no evidence)
               ─► one validated row (models.validate_sim_row)
writer.py      data/simulated/*  (refuses non-SYNTHETIC rows and any path inside data/events/)
```

- **Real parents (read-only).** The real `data/events/events.jsonl` supplies parent events. A parent's measured box area, position, confidence, growth, lateral motion or drowsiness score seeds the constructed parameters, clamped to the range each scenario tests. The link is kept in `parent_real_event_id` / `parent_real_driver_event_id`.
- **Determinism.** Each instance uses `random.Random(f"{seed}:{scenario}:{variant}")`, so the same seed gives identical rows.
- **Expectations first.** Expectations are written in `scenarios.py` before the engine runs. A mismatch is recorded in the manifest and the report; the engine is never changed to remove one.

## Provenance fields

| Field | Value |
| --- | --- |
| `data_source`, `observation_type`, `timestamp_source`, `gps_source` | `SYNTHETIC` |
| `location_source` | `SYNTHETIC_REFERENCE` |
| `event_id` | `sim_NNNNNN` (never the real `event_NNNNNN` form) |
| `synthetic_id`, `scenario_name`, `scenario_description`, `synthetic_generation_reason` | scenario metadata |
| `parent_real_event_id`, `parent_real_driver_event_id`, `parent_real_run_id` | set only when a real parent is used |

**Field provenance** in `schema.json`:
- **METADATA:** ids and labels.
- **SYNTHETIC_INPUT:** constructed by the generator.
- **ENGINE_ON_SYNTHETIC:** computed by existing code from synthetic inputs.
- **VALIDATION:** `expected_behavior` / `actual_risk_level` / `expectation_met`.

**Reused names.** Names from the real event schema are reused where they exist. Two requested names are mapped instead of duplicated:
- `object_id` → `primary_track_id`
- `approach_rate` → `box_growth_per_second`

`risk_score` is kept because `RiskAssessment.risk_score` already exists; it equals `smoothed_risk_score`.

**`trajectory_overlap`** is design ground truth here: the image-space share of the constructed box inside the central band. It is not used by the risk engine and stays null in the real dataset.

## Scenarios

`attentive_static_vehicle`, `attentive_approaching_vehicle`, `one_hand_approaching_vehicle`,
`hands_off_approaching_vehicle`, `drowsy_approaching_vehicle`, `attentive_pedestrian_crossing`,
`distracted_pedestrian_crossing`, `distracted_dog_entering_path`, `stationary_vehicle_false_positive`,
`distant_pedestrian_false_positive`, `persistent_low_risk_hazard`, `escalating_hazard`.

Each scenario has a description, a reason and an expected set of acceptable peak levels. See `scenarios.py`.

## Rules

- Never write to `data/events/`.
- Never produce video or images.
- Never put synthetic GPS or time into LOCAL_REAL records.
- Never tune the risk engine to meet an expectation.
