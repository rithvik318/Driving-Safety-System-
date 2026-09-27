"""Synthetic scenario records: configuration, schema (sim-1.0) and row validation.

A synthetic row is ONE 10 Hz time step of a constructed scenario. It is never an observation.
Field names are reused from the real event schema (app/events/schema.py) wherever an equivalent
field exists; the few names requested for the simulated dataset that have an existing equivalent
are mapped, not duplicated (REQUESTED_NAME_MAPPING).

Field provenance in this dataset (every value is synthetic; this says HOW it was produced):
  METADATA            ids, labels, provenance markers
  SYNTHETIC_INPUT     values the generator constructed: timestamps, GPS trace, detector boxes /
                      labels / confidences, per-frame driver signals, design ground truth
  ENGINE_ON_SYNTHETIC values computed by the EXISTING, unmodified project code (IoUTracker, driver
                      temporal logic, RiskEngine, EventRecorder) from the synthetic inputs
  VALIDATION          expected qualitative behaviour vs the engine's actual output
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config.settings import PROJECT_ROOT
from app.events.schema import BY_NAME as EVENT_FIELDS
from app.events.schema import RISK_LEVELS

SIM_SCHEMA_VERSION = "sim-1.0"
GENERATOR_VERSION = "1.0.0"
SYNTHETIC = "SYNTHETIC"
LOCATION_SOURCE = "SYNTHETIC_REFERENCE"
REAL_EVENT_ID = re.compile(r"^event_\d{6}$")

PROVENANCE_POLICY = (
    "Every row is SYNTHETIC: data_source=SYNTHETIC, observation_type=SYNTHETIC, timestamp_source=SYNTHETIC, "
    "gps_source=SYNTHETIC, location_source=SYNTHETIC_REFERENCE. Inputs (boxes, driver signals, time, location) are "
    "constructed by the generator; engine outputs are computed by the existing, unmodified tracker / driver / risk / "
    "event code FROM those synthetic inputs and are therefore synthetic too. Rows are written only to data/simulated/ "
    "and never to the real dataset (data/events/). parent_real_event_id only references an existing LOCAL_REAL event "
    "used as a statistical reference; it does not make the row an observation."
)
NOT_A_REPLACEMENT = ("Synthetic data does not replace real observations. It stress-tests the existing risk/event "
                     "logic with controlled feature-level scenarios; it says nothing about how often these "
                     "situations occur on real roads or whether the engine is correct on real driving.")

# Requested field name -> existing schema field carrying the same quantity (not duplicated).
REQUESTED_NAME_MAPPING = {
    "object_id": "primary_track_id",
    "approach_rate": "box_growth_per_second",
}


class SimulationError(ValueError):
    pass


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 42
    variants_per_scenario: int = 2
    cadence_hz: float = 10.0
    duration_range_seconds: tuple[float, float] = (5.0, 7.0)
    driver_preroll_seconds: float = 6.0  # synthetic driver history before t=0 (not emitted as rows)
    image_width: int = 1920  # synthetic front-camera frame (matches the 1920x1080 real clips)
    image_height: int = 1080
    # Generic Indian urban-road REFERENCE point for the synthetic GPS trace. NOT a collection location.
    base_lat: float = 28.6139
    base_lon: float = 77.2090
    location_jitter_deg: float = 0.01  # start of each scenario instance: base +- jitter
    trace_step_m_per_s: float = 8.0  # only lays out the synthetic trace; not a vehicle speed claim
    base_datetime: str = "2026-01-15T09:00:00+05:30"  # synthetic clock origin
    instance_spacing_seconds: float = 120.0
    run_id: str = "synthetic_scenarios_v1"
    real_events_path: Path = PROJECT_ROOT / "data" / "events" / "events.jsonl"

    @property
    def dt(self) -> float:
        return 1.0 / self.cadence_hz


def validate_config(c: SimulationConfig) -> None:
    lo, hi = c.duration_range_seconds
    if not (0 < lo <= hi <= 60):
        raise SimulationError("duration_range_seconds must satisfy 0 < lo <= hi <= 60")
    if c.variants_per_scenario < 1:
        raise SimulationError("variants_per_scenario must be >= 1")
    if not (1.0 <= c.cadence_hz <= 60.0):
        raise SimulationError("cadence_hz must be in [1, 60]")
    if not (abs(c.base_lat) <= 90 and abs(c.base_lon) <= 180):
        raise SimulationError("base coordinate out of range")
    if not (0 <= c.location_jitter_deg <= 0.1):
        raise SimulationError("location_jitter_deg must be in [0, 0.1] (small perturbations only)")


# ------------------------------------------------------------------------------------ schema


@dataclass(frozen=True)
class SimField:
    name: str
    type: str  # string | int64 | float64 | bool | json
    nullable: bool
    provenance: str  # METADATA | SYNTHETIC_INPUT | ENGINE_ON_SYNTHETIC | VALIDATION
    description: str
    allowed: tuple | None = None
    reused_from_event_schema: bool = False


def _reuse(name: str, provenance: str, note: str = "", nullable: bool | None = None, allowed=None) -> SimField:
    f = EVENT_FIELDS[name]
    desc = f.description + (f" SYNTHETIC dataset: {note}" if note else "")
    return SimField(name, f.type, f.nullable if nullable is None else nullable, provenance, desc,
                    allowed if allowed is not None else f.allowed, True)


def _new(name, type_, nullable, provenance, description, allowed=None) -> SimField:
    return SimField(name, type_, nullable, provenance, description, allowed)


M, IN, ENG, VAL = "METADATA", "SYNTHETIC_INPUT", "ENGINE_ON_SYNTHETIC", "VALIDATION"
SIM_FIELDS: tuple[SimField, ...] = (
    # ---- identity / provenance
    _new("schema_version", "string", False, M, "Simulated-dataset schema version.", (SIM_SCHEMA_VERSION,)),
    _reuse("event_id", M, "row id sim_000001, sim_000002, ... (never an event_NNNNNN real id)."),
    _new("synthetic_id", "string", False, M, "Stable descriptive row id: SYN-<scenario>-v<variant>-s<step>."),
    _reuse("run_id", M, "generator run id."),
    _reuse("session_id", M, "scenario instance: <scenario>__v<variant>."),
    _new("scenario_name", "string", False, M, "Scenario name (one of the 12 designed scenarios)."),
    _new("scenario_variant", "int64", False, M, "Variant index of the scenario (different seed-derived parameters)."),
    _new("scenario_description", "string", False, M, "What the scenario constructs."),
    _new("synthetic_generation_reason", "string", False, M, "Why the scenario exists (what it stress-tests)."),
    _new("parent_real_event_id", "string", True, M,
         "LOCAL_REAL road event (event_NNNNNN in parent_real_run_id) used as a statistical reference for the "
         "constructed road object; null = synthetic-only parameters."),
    _new("parent_real_driver_event_id", "string", True, M,
         "LOCAL_REAL driver-camera event used as a reference for the constructed driver signals; null if none."),
    _new("parent_real_run_id", "string", True, M, "Run id of the real dataset the parent ids belong to."),
    _reuse("data_source", M, "always SYNTHETIC.", allowed=(SYNTHETIC,)),
    _new("observation_type", "string", False, M, "Always SYNTHETIC: nothing in this row was observed.", (SYNTHETIC,)),
    _reuse("camera", M, "virtual front+driver pair (no video exists).", allowed=("front+driver",)),
    # ---- time / location (synthetic)
    _reuse("timestamp", IN, "seconds since the start of the scenario instance on a fixed synthetic cadence."),
    _new("timestamp_source", "string", False, M, "Always SYNTHETIC.", (SYNTHETIC,)),
    _new("synthetic_datetime", "string", False, IN, "ISO-8601 time on a synthetic clock (base_datetime + offsets). Not a capture time."),
    _reuse("frame_index", IN, "step index within the scenario instance (no frames exist).", nullable=False),
    _reuse("gps_lat", IN, "SYNTHETIC coordinate around a generic reference point; NOT a collection location.", nullable=False),
    _reuse("gps_lon", IN, "SYNTHETIC coordinate around a generic reference point; NOT a collection location.", nullable=False),
    _reuse("gps_source", M, "always SYNTHETIC.", allowed=(SYNTHETIC,)),
    _new("location_source", "string", False, M, "Always SYNTHETIC_REFERENCE.", (LOCATION_SOURCE,)),
    # ---- driver (inputs constructed; temporal states computed by the existing driver logic)
    _reuse("driver_hand_state", IN, "scripted hand state fed to the existing driver temporal logic."),
    _reuse("driver_hand_confidence", IN, "scripted classifier confidence."),
    _reuse("face_status", IN, "scripted face status."),
    _new("head_yaw", "float64", True, IN, "Scripted head yaw, degrees (relative head-pose signal)."),
    _new("head_pitch", "float64", True, IN, "Scripted head pitch, degrees (> 0 = down)."),
    _new("eye_closure_ratio", "float64", True, ENG, "Share of the rolling window with eyes closed (existing EyeClosureTracker on scripted eye openness)."),
    _new("eye_closed_duration", "float64", True, ENG, "Current continuous eye closure, seconds (existing EyeClosureTracker)."),
    _reuse("drowsiness_score", ENG, "existing drowsiness scoring on scripted eye signals."),
    _reuse("drowsiness_level", ENG, "existing drowsiness scoring on scripted eye signals."),
    _reuse("driver_activity", ENG, "existing activity inference (NO_HANDS >= 2 s -> HANDS_OFF_WHEEL)."),
    _new("distraction_duration", "float64", True, ENG, "Continuous INFERRED manual-distraction time, seconds (existing driver logic); 0.0 if none."),
    _new("manual_state_duration", "float64", True, ENG, "Continuous time in the current hand/manual state, seconds (existing StateDurationTracker)."),
    _new("head_away_duration", "float64", False, ENG, "Continuous time the head pose has been 'away' by the risk rules (existing HeadAwayTimer), seconds."),
    # ---- road object (constructed detections; track features computed by the existing tracker/engine)
    _reuse("image_width", IN, "virtual frame width.", nullable=False),
    _reuse("image_height", IN, "virtual frame height.", nullable=False),
    _reuse("hazard_class", IN, "detector label of the constructed detection (may flicker by design); null when the object was not detected this step."),
    _reuse("hazard_confidence", IN, "constructed detector confidence."),
    _reuse("center_x", IN, "constructed box."),
    _reuse("center_y", IN, "constructed box."),
    _reuse("bbox_width", IN, "constructed box."),
    _reuse("bbox_height", IN, "constructed box."),
    _reuse("bbox_area", IN, "constructed box."),
    _reuse("primary_track_id", ENG, "tracker id of the scenario's road object (requested name: object_id)."),
    _reuse("hazard_track_class", ENG, "track-level label from the existing IoUTracker."),
    _reuse("track_persistence_seconds", ENG, "existing IoUTracker."),
    _reuse("box_growth_per_second", ENG, "requested name: approach_rate. Existing risk-engine feature (slope of ln area); null until enough history."),
    _reuse("lateral_toward_center_fw_per_s", ENG, "existing risk-engine feature."),
    _reuse("trajectory_overlap", IN,
           "DESIGN GROUND TRUTH: fraction of the constructed box width inside the central image band "
           "(|x-0.5| <= 0.2 of the frame width). Image-space only, not a calibrated vehicle path, and NOT an "
           "input to the risk engine. (Always null in the real dataset.)"),
    _reuse("hazard_persistence", ENG, "continuous time the risk engine has reported the object as a qualifying road hazard, seconds; 0.0 when not reported."),
    # ---- risk engine output (existing, unmodified RiskEngine)
    _reuse("risk_level", ENG, "existing RiskEngine after hysteresis."),
    _new("raw_risk_level", "string", True, ENG, "Existing RiskEngine level for this step alone (after gates, before hysteresis).", RISK_LEVELS),
    _reuse("raw_risk_score", ENG),
    _reuse("smoothed_risk_score", ENG),
    _new("risk_score", "float64", True, ENG, "RiskAssessment.risk_score (= smoothed_risk_score): rule-based prototype score 0-100, not a probability."),
    _reuse("hazard_type", ENG),
    _reuse("risk_reason", ENG),
    _reuse("risk_factors", ENG),
    _reuse("risk_gates", ENG),
    _reuse("evidence_quality", ENG),
    _reuse("alarm_recommended", ENG),
    _reuse("alarm_triggered", M, "always false: no alarm output exists."),
    # ---- existing EventRecorder run on the synthetic stream
    _new("event_type", "string", True, ENG, "Event the existing EventRecorder emitted at this step (first if several); null if none.",
         EVENT_FIELDS["event_type"].allowed),
    _reuse("transition", ENG, "transition of that event."),
    _new("events_emitted", "json", False, ENG, "All events the existing EventRecorder emitted at this step: [\"TYPE FROM->TO\", ...]."),
    # ---- validation
    _new("expected_behavior", "string", False, VAL, "Expected qualitative behaviour of the scenario instance (written before running the engine)."),
    _new("expected_peak_levels", "json", False, VAL, "Acceptable peak risk levels for the scenario instance."),
    _new("actual_risk_level", "string", True, VAL, "Risk level the existing engine produced at this step (= risk_level).", RISK_LEVELS),
    _new("instance_peak_risk_level", "string", False, VAL, "Highest risk level the engine produced in this scenario instance.", RISK_LEVELS),
    _new("expectation_met", "bool", False, VAL, "Whether the instance's actual behaviour matched the expectation (same value on every row of the instance)."),
    _reuse("notes", M, "provenance notes.", nullable=False),
)
SIM_FIELD_NAMES = tuple(f.name for f in SIM_FIELDS)
SIM_BY_NAME = {f.name: f for f in SIM_FIELDS}

# The feature list requested for this dataset (resolved through REQUESTED_NAME_MAPPING).
REQUESTED_FIELDS = ("event_id", "timestamp", "gps_lat", "gps_lon", "driver_hand_state", "driver_activity", "head_yaw",
                    "head_pitch", "eye_closure_ratio", "drowsiness_score", "distraction_duration", "hazard_type",
                    "hazard_class", "object_id", "approach_rate", "trajectory_overlap", "hazard_persistence", "risk_score",
                    "risk_level", "risk_reason", "alarm_triggered", "data_source", "observation_type", "synthetic_id",
                    "parent_real_event_id", "scenario_name", "scenario_description", "synthetic_generation_reason",
                    "timestamp_source", "location_source", "gps_source", "expected_behavior", "actual_risk_level")


def resolve_field(name: str) -> str:
    return REQUESTED_NAME_MAPPING.get(name, name)


def _type_ok(v, t: str) -> bool:
    if t == "string":
        return isinstance(v, str)
    if t == "json":
        if not isinstance(v, str):
            return False
        try:
            json.loads(v)
            return True
        except ValueError:
            return False
    if t == "bool":
        return isinstance(v, bool)
    if t == "int64":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "float64":
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    return False


def validate_sim_row(row: dict) -> list[str]:
    """Problems with one synthetic row (empty list = valid)."""
    errors = []
    missing = [n for n in SIM_FIELD_NAMES if n not in row]
    extra = [n for n in row if n not in SIM_BY_NAME]
    if missing:
        errors.append(f"missing fields: {missing}")
    if extra:
        errors.append(f"unknown fields: {extra}")
    for f in SIM_FIELDS:
        if f.name not in row:
            continue
        v = row[f.name]
        if v is None:
            if not f.nullable:
                errors.append(f"{f.name} must not be null")
            continue
        if not _type_ok(v, f.type):
            errors.append(f"{f.name} must be {f.type}, got {type(v).__name__} {v!r}")
        elif f.allowed and v not in f.allowed:
            errors.append(f"{f.name}={v!r} not in {f.allowed}")
    if errors:
        return errors
    for key in ("data_source", "observation_type", "timestamp_source", "gps_source"):
        if row[key] != SYNTHETIC:
            errors.append(f"{key} must be SYNTHETIC")
    if row["location_source"] != LOCATION_SOURCE:
        errors.append("location_source must be SYNTHETIC_REFERENCE")
    if not (abs(row["gps_lat"]) <= 90 and abs(row["gps_lon"]) <= 180):
        errors.append("gps out of range")
    if REAL_EVENT_ID.match(row["event_id"]):
        errors.append("event_id must not look like a real event id (event_NNNNNN)")
    for key in ("parent_real_event_id", "parent_real_driver_event_id"):
        if row[key] is not None and not REAL_EVENT_ID.match(row[key]):
            errors.append(f"{key} must be a real event id event_NNNNNN or null")
    if (row["parent_real_event_id"] or row["parent_real_driver_event_id"]) and not row["parent_real_run_id"]:
        errors.append("parent_real_run_id required when a parent id is set")
    if row["alarm_triggered"]:
        errors.append("alarm_triggered must be false: no alarm output exists")
    if row["actual_risk_level"] != row["risk_level"]:
        errors.append("actual_risk_level must equal risk_level")
    if row["trajectory_overlap"] is not None and not 0.0 <= row["trajectory_overlap"] <= 1.0:
        errors.append("trajectory_overlap must be in [0, 1]")
    return errors


def sim_schema_document() -> dict:
    return {
        "schema_version": SIM_SCHEMA_VERSION,
        "description": "SYNTHETIC feature-level driving scenarios: one row per 10 Hz step of a constructed scenario. "
                       "Not observations. Separate from the real event dataset (data/events/).",
        "provenance_policy": PROVENANCE_POLICY,
        "statement": NOT_A_REPLACEMENT,
        "field_provenance": {
            "METADATA": "ids, labels and provenance markers",
            "SYNTHETIC_INPUT": "constructed by the generator (time, location, boxes, labels, confidences, driver signals, design ground truth)",
            "ENGINE_ON_SYNTHETIC": "computed by the existing, unmodified tracker / driver temporal logic / RiskEngine / EventRecorder from the synthetic inputs",
            "VALIDATION": "expected qualitative behaviour vs the engine's actual output",
        },
        "requested_name_mapping": REQUESTED_NAME_MAPPING,
        "units": "Image-space pixels and seconds. No metres, physical speed, distance or TTC.",
        "fields": [{"name": f.name, "type": f.type, "nullable": f.nullable, "provenance": f.provenance,
                    "reused_from_event_schema": f.reused_from_event_schema, "description": f.description,
                    **({"allowed": list(f.allowed)} if f.allowed else {})} for f in SIM_FIELDS],
    }


def arrow_sim_schema():
    import pyarrow as pa

    types = {"string": pa.string(), "json": pa.string(), "int64": pa.int64(), "float64": pa.float64(), "bool": pa.bool_()}
    return pa.schema([pa.field(f.name, types[f.type], nullable=f.nullable) for f in SIM_FIELDS],
                     metadata={"schema_version": SIM_SCHEMA_VERSION, "data_source": SYNTHETIC})


@dataclass
class InstanceResult:
    """Validation summary of one scenario instance."""

    scenario_name: str
    variant: int
    session_id: str
    rows: int
    duration_seconds: float
    expected_behavior: str
    expected_peak_levels: tuple[str, ...]
    peak_level: str
    first_level: str
    final_level: str
    level_counts: dict = field(default_factory=dict)
    alarm_recommended_steps: int = 0
    expectation_met: bool = False
    mismatch: str | None = None
    parent_real_event_id: str | None = None
    parent_real_driver_event_id: str | None = None
    events: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}
