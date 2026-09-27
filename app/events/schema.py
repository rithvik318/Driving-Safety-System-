"""Stable event-record schema (version 1.0): field names, types, nullability, provenance, meaning.

Provenance of each FIELD (the record-level provenance is data_source + observation_type):
  METADATA  bookkeeping written by the recorder (ids, source labels, file paths)
  OBSERVED  measured directly: timestamps, detector boxes/labels/confidences, driver per-frame
            signals (the hand-state classifier output is treated as an observed signal in this
            project), GPS only when a real source supplied it
  INFERRED  derived: track identity, persistence, image-space velocity, area growth, risk
            score/level/reason, driver_activity, drowsiness level

Types: string, int64, float64, bool, json (a JSON-encoded string: list/dict).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = "1.0"

EVENT_TYPES = ("RISK_ESCALATED", "RISK_DEESCALATED", "PERSISTENT_HAZARD", "DRIVER_STATE_CHANGE")
DATA_SOURCES = ("LOCAL_REAL", "PUBLIC", "SYNTHETIC", "SYNTHETIC_COMBINATION")
REAL_SOURCES = ("LOCAL_REAL", "PUBLIC")
SYNTHETIC_SOURCES = ("SYNTHETIC", "SYNTHETIC_COMBINATION")
OBSERVATION_TYPES = ("OBSERVED", "INFERRED")
RISK_LEVELS = ("SAFE", "CAUTION", "HIGH", "CRITICAL")
EVIDENCE_KINDS = ("NONE", "FRAME", "CLIP")
CAMERAS = ("front", "driver", "front+driver")


@dataclass(frozen=True)
class Field:
    name: str
    type: str
    nullable: bool
    provenance: str
    description: str
    allowed: tuple | None = None


F = Field
FIELDS: tuple[Field, ...] = (
    F("schema_version", "string", False, "METADATA", "Event schema version."),
    F("event_id", "string", False, "METADATA", "Sequential id within a run: event_000001, event_000002, ... (deterministic)."),
    F("run_id", "string", False, "METADATA", "Identifier of the recording/processing run that produced the event."),
    F("session_id", "string", False, "METADATA", "Stream the event belongs to (e.g. the source video's file stem)."),
    F("source_file", "string", True, "METADATA", "Source file name (no directory), when the stream is a file."),
    F("camera", "string", False, "METADATA", "Which camera stream(s) the event is based on.", CAMERAS),
    F("data_source", "string", False, "METADATA",
      "LOCAL_REAL (our own captures) | PUBLIC | SYNTHETIC (constructed test inputs) | SYNTHETIC_COMBINATION "
      "(real streams combined although not recorded together). Synthetic records live in a separate dataset.",
      DATA_SOURCES),
    F("observation_type", "string", False, "METADATA",
      "INFERRED when the event is derived from inferred quantities (all risk / hazard events); OBSERVED only for "
      "a DRIVER_STATE_CHANGE of the observed hand state.", OBSERVATION_TYPES),
    F("timestamp", "float64", False, "OBSERVED", "Stream time of the event, seconds (video timestamps for files)."),
    F("frame_index", "int64", True, "OBSERVED", "Index of the frame that produced the event, when known."),
    F("event_type", "string", False, "INFERRED", "Kind of event.", EVENT_TYPES),
    F("transition", "string", True, "INFERRED", "FROM->TO for level and driver-state changes, e.g. SAFE->CAUTION."),
    F("previous_risk_level", "string", True, "INFERRED", "Risk level before a level change.", RISK_LEVELS),
    F("change_kind", "string", True, "INFERRED", "For DRIVER_STATE_CHANGE: hand_state | driver_activity | drowsiness_level.",
      ("hand_state", "driver_activity", "drowsiness_level")),
    F("gps_lat", "float64", True, "OBSERVED", "Latitude from a real GPS source; null when unavailable. Never invented."),
    F("gps_lon", "float64", True, "OBSERVED", "Longitude from a real GPS source; null when unavailable. Never invented."),
    F("gps_source", "string", False, "METADATA", "GPS source name, or UNAVAILABLE."),
    F("gps_timestamp", "float64", True, "OBSERVED", "Time of the GPS fix used, seconds (same clock as timestamp)."),
    F("gps_accuracy_m", "float64", True, "OBSERVED", "Reported GPS accuracy in metres, when the source provides it."),
    F("risk_level", "string", True, "INFERRED", "Risk level after hysteresis (decision-support state, not validated).", RISK_LEVELS),
    F("raw_risk_score", "float64", True, "INFERRED", "Rule-based prototype score 0-100 for this observation. Not a probability."),
    F("smoothed_risk_score", "float64", True, "INFERRED", "Smoothed prototype score 0-100. Not a probability."),
    F("hazard_type", "string", True, "INFERRED", "Risk engine hazard type (NONE, APPROACHING_VEHICLE, ...)."),
    F("primary_track_id", "int64", True, "INFERRED", "Tracker id of the object driving the hazard (identity is inferred)."),
    F("image_width", "int64", True, "OBSERVED", "Front-camera frame width in pixels."),
    F("image_height", "int64", True, "OBSERVED", "Front-camera frame height in pixels."),
    F("hazard_class", "string", True, "OBSERVED", "Detector label of the primary object at the event time."),
    F("hazard_track_class", "string", True, "INFERRED", "Track-level label (most frequent detector label of the track)."),
    F("hazard_class_changes", "int64", True, "INFERRED", "Detector label changes observed on the primary track."),
    F("hazard_confidence", "float64", True, "OBSERVED", "Detector confidence of the primary object at the event time."),
    F("track_persistence_seconds", "float64", True, "INFERRED", "How long the primary track has been observed."),
    F("track_detection_count", "int64", True, "INFERRED", "Detections supporting the primary track."),
    F("center_x", "float64", True, "OBSERVED", "Primary object box centre x, pixels."),
    F("center_y", "float64", True, "OBSERVED", "Primary object box centre y, pixels."),
    F("bbox_width", "float64", True, "OBSERVED", "Primary object box width, pixels."),
    F("bbox_height", "float64", True, "OBSERVED", "Primary object box height, pixels."),
    F("bbox_area", "float64", True, "OBSERVED", "Primary object box area, pixels²."),
    F("velocity_x_pixels_per_second", "float64", True, "INFERRED", "Image-space centre velocity x (not physical speed)."),
    F("velocity_y_pixels_per_second", "float64", True, "INFERRED", "Image-space centre velocity y (not physical speed)."),
    F("area_change_rate", "float64", True, "INFERRED", "Box area change, pixels² per second (tracker, frame to frame)."),
    F("box_growth_per_second", "float64", True, "INFERRED", "Risk-engine growth: slope of ln(box area) over ~1 s (relative approach signal, not distance)."),
    F("lateral_toward_center_fw_per_s", "float64", True, "INFERRED", "Horizontal motion toward/across the central image band, frame-widths per second."),
    F("trajectory_overlap", "float64", True, "INFERRED", "Reserved: overlap of the object's path with the vehicle's projected path. Always null until a calibrated path model exists."),
    F("hazard_persistence", "float64", True, "INFERRED", "How long this hazard (same hazard type and track) has been continuously reported, seconds."),
    F("driver_hand_state", "string", True, "OBSERVED", "Hand-state classifier output (BOTH_HANDS / ONE_HAND / NO_HANDS / UNKNOWN)."),
    F("driver_hand_confidence", "float64", True, "OBSERVED", "Hand-state classifier confidence."),
    F("driver_activity", "string", True, "INFERRED", "Inferred driver activity (NORMAL, HANDS_OFF_WHEEL, ...)."),
    F("drowsiness_level", "string", True, "INFERRED", "Temporal drowsiness level (prototype, not medically validated)."),
    F("drowsiness_score", "float64", True, "INFERRED", "Temporal drowsiness score 0-1 (prototype)."),
    F("face_status", "string", True, "OBSERVED", "Driver face-landmark status for the frame."),
    F("risk_reason", "string", True, "INFERRED", "Reason generated by the rule engine from structured factors (no LLM)."),
    F("risk_factors", "json", True, "INFERRED", "Contributing factors [{name, points, source, detail, track_id}]."),
    F("risk_gates", "json", True, "INFERRED", "Level caps applied by the risk engine."),
    F("evidence_quality", "float64", True, "INFERRED", "0-1 support of the evidence behind the risk result (not a probability)."),
    F("alarm_recommended", "bool", True, "INFERRED", "Risk engine recommendation (level >= configured alarm level)."),
    F("alarm_triggered", "bool", False, "METADATA", "Whether an alarm was actually raised. Always false: no alarm output exists yet."),
    F("evidence_kind", "string", False, "METADATA", "NONE | FRAME (single images) | CLIP (video clips).", EVIDENCE_KINDS),
    F("evidence_path", "string", True, "METADATA", "Evidence directory relative to the event dataset root: <run_id>/<event_id>."),
    F("evidence_files", "json", True, "METADATA", "Evidence files relative to the event dataset root."),
    F("notes", "json", True, "METADATA", "Provenance and limitation notes for this record."),
)
FIELD_NAMES = tuple(f.name for f in FIELDS)
BY_NAME = {f.name: f for f in FIELDS}
EVENT_LEVEL_FIELDS = ("event_type", "risk_level", "raw_risk_score")


def _type_ok(value, type_: str) -> bool:
    if type_ == "string":
        return isinstance(value, str)
    if type_ == "json":
        if not isinstance(value, str):
            return False
        try:
            json.loads(value)
            return True
        except ValueError:
            return False
    if type_ == "bool":
        return isinstance(value, bool)
    if type_ == "int64":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_ == "float64":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    return False


def _relative_ok(p: str) -> bool:
    pp = PurePosixPath(p)
    return bool(p) and not pp.is_absolute() and ".." not in pp.parts and "\\" not in p and ":" not in p


def validate_record(rec: dict) -> list[str]:
    """Problems with one record (empty list = valid)."""
    errors = []
    missing = [n for n in FIELD_NAMES if n not in rec]
    extra = [n for n in rec if n not in BY_NAME]
    if missing:
        errors.append(f"missing fields: {missing}")
    if extra:
        errors.append(f"unknown fields: {extra}")
    for f in FIELDS:
        if f.name not in rec:
            continue
        v = rec[f.name]
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
    if rec["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if rec["event_type"] != "DRIVER_STATE_CHANGE" and rec["observation_type"] != "INFERRED":
        errors.append("risk / hazard events are derived: observation_type must be INFERRED")
    if rec["event_type"] == "DRIVER_STATE_CHANGE":
        expected = "OBSERVED" if rec["change_kind"] == "hand_state" else "INFERRED"
        if rec["observation_type"] != expected:
            errors.append(f"DRIVER_STATE_CHANGE of {rec['change_kind']} must be observation_type {expected}")
        if rec["change_kind"] is None or rec["transition"] is None:
            errors.append("DRIVER_STATE_CHANGE needs change_kind and transition")
    if rec["event_type"] in ("RISK_ESCALATED", "RISK_DEESCALATED"):
        if rec["previous_risk_level"] is None or rec["risk_level"] is None or rec["transition"] is None:
            errors.append("risk level events need previous_risk_level, risk_level and transition")
        else:
            up = RISK_LEVELS.index(rec["risk_level"]) > RISK_LEVELS.index(rec["previous_risk_level"])
            if up != (rec["event_type"] == "RISK_ESCALATED") or rec["risk_level"] == rec["previous_risk_level"]:
                errors.append("event_type does not match the direction of the level change")
    if (rec["gps_lat"] is None) != (rec["gps_lon"] is None):
        errors.append("gps_lat and gps_lon must both be set or both be null")
    if rec["gps_lat"] is None and rec["gps_source"] != "UNAVAILABLE":
        errors.append("gps_source must be UNAVAILABLE when there are no coordinates")
    if rec["gps_lat"] is not None and rec["gps_source"] == "UNAVAILABLE":
        errors.append("coordinates present but gps_source is UNAVAILABLE")
    if rec["gps_lat"] is not None and rec["gps_lon"] is not None and not (abs(rec["gps_lat"]) <= 90 and abs(rec["gps_lon"]) <= 180):
        errors.append("gps coordinates out of range")
    if rec["trajectory_overlap"] is not None:
        errors.append("trajectory_overlap is reserved and must be null (no calibrated path model)")
    if rec["alarm_triggered"]:
        errors.append("alarm_triggered must be false: no alarm output exists yet")
    files = json.loads(rec["evidence_files"]) if rec["evidence_files"] else []
    if rec["evidence_kind"] == "NONE":
        if rec["evidence_path"] is not None or files:
            errors.append("evidence_kind NONE must have no evidence_path/evidence_files")
    else:
        if rec["evidence_path"] is None or not files:
            errors.append(f"evidence_kind {rec['evidence_kind']} needs evidence_path and evidence_files")
    for p in ([rec["evidence_path"]] if rec["evidence_path"] else []) + list(files):
        if not _relative_ok(p):
            errors.append(f"evidence path must be relative (POSIX, no '..'): {p!r}")
    return errors


def schema_document() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "description": "Physical-AI-Driving-Safety event record. One row per event (not per frame).",
        "record_provenance": {
            "data_source": {"LOCAL_REAL": "captured by the team", "PUBLIC": "public dataset",
                            "SYNTHETIC": "constructed test input", "SYNTHETIC_COMBINATION":
                            "real streams combined although they were NOT recorded together"},
            "observation_type": {"OBSERVED": "event is a change in an observed signal",
                                 "INFERRED": "event derived from inferred quantities"},
            "separation": "SYNTHETIC and SYNTHETIC_COMBINATION records are written to a separate dataset root "
                          "and are rejected by the real-dataset writer.",
        },
        "field_provenance": {"METADATA": "recorder bookkeeping", "OBSERVED": "measured directly",
                             "INFERRED": "derived from observations by tracker / rules"},
        "units": "Image-space pixels and seconds. No metres, physical speed, distance or TTC anywhere.",
        "fields": [{"name": f.name, "type": f.type, "nullable": f.nullable, "provenance": f.provenance,
                    "description": f.description, **({"allowed": list(f.allowed)} if f.allowed else {})}
                   for f in FIELDS],
    }


def export_schema(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema_document(), indent=2), encoding="utf-8")
    return path


def arrow_schema():
    """pyarrow schema (json fields stored as strings). Raises ImportError without pyarrow."""
    import pyarrow as pa

    types = {"string": pa.string(), "json": pa.string(), "int64": pa.int64(), "float64": pa.float64(), "bool": pa.bool_()}
    return pa.schema([pa.field(f.name, types[f.type], nullable=f.nullable) for f in FIELDS],
                     metadata={"schema_version": SCHEMA_VERSION})
