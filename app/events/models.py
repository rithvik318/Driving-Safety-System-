"""Event record model and field builders (from RiskAssessment / TrackedObject / DriverState).

An EventRecord is a validated mapping with exactly the schema fields (app/events/schema.py).
Builders only copy values that exist upstream; nothing is invented. Missing inputs -> null.
"""

from __future__ import annotations

import json
from enum import Enum

from app.events.schema import FIELD_NAMES, SCHEMA_VERSION, validate_record


class EventType(str, Enum):
    RISK_ESCALATED = "RISK_ESCALATED"
    RISK_DEESCALATED = "RISK_DEESCALATED"
    PERSISTENT_HAZARD = "PERSISTENT_HAZARD"
    DRIVER_STATE_CHANGE = "DRIVER_STATE_CHANGE"


class DataSource(str, Enum):
    LOCAL_REAL = "LOCAL_REAL"
    PUBLIC = "PUBLIC"
    SYNTHETIC = "SYNTHETIC"
    SYNTHETIC_COMBINATION = "SYNTHETIC_COMBINATION"

    @property
    def is_synthetic(self) -> bool:
        return self in (DataSource.SYNTHETIC, DataSource.SYNTHETIC_COMBINATION)


class ObservationType(str, Enum):
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"


class EventValidationError(ValueError):
    pass


class EventRecord:
    """Immutable, schema-validated event record."""

    __slots__ = ("_data",)

    def __init__(self, values: dict):
        data = {name: values.get(name) for name in FIELD_NAMES}
        unknown = set(values) - set(FIELD_NAMES)
        if unknown:
            raise EventValidationError(f"unknown fields: {sorted(unknown)}")
        errors = validate_record(data)
        if errors:
            raise EventValidationError("; ".join(errors))
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, *_):
        raise AttributeError("EventRecord is immutable")

    def __getitem__(self, name):
        return self._data[name]

    def to_dict(self) -> dict:
        return dict(self._data)

    def with_updates(self, **changes) -> "EventRecord":
        return EventRecord({**self._data, **changes})

    def __repr__(self) -> str:
        return f"EventRecord({self.event_id} {self.event_type} t={self.timestamp})"


def _val(x):
    return getattr(x, "value", x)


def _f(x):
    return None if x is None else float(x)


def base_fields(event_id: str, run_id: str, session_id: str, source_file: str | None, camera: str,
                data_source: DataSource, observation_type: ObservationType, timestamp: float,
                frame_index: int | None, event_type: EventType, gps_fix) -> dict:
    d = {name: None for name in FIELD_NAMES}
    d.update(schema_version=SCHEMA_VERSION, event_id=event_id, run_id=run_id, session_id=session_id,
             source_file=source_file, camera=camera, data_source=_val(data_source),
             observation_type=_val(observation_type), timestamp=float(timestamp), frame_index=frame_index,
             event_type=_val(event_type), alarm_triggered=False, evidence_kind="NONE",
             gps_source="UNAVAILABLE")
    if gps_fix is not None:
        d.update(gps_lat=float(gps_fix.latitude), gps_lon=float(gps_fix.longitude), gps_source=gps_fix.source,
                 gps_timestamp=float(gps_fix.timestamp), gps_accuracy_m=_f(gps_fix.accuracy_m))
    return d


def risk_fields(assessment) -> dict:
    return dict(
        risk_level=_val(assessment.risk_level), raw_risk_score=round(float(assessment.raw_risk_score), 3),
        smoothed_risk_score=round(float(assessment.smoothed_risk_score), 3), hazard_type=_val(assessment.hazard_type),
        primary_track_id=assessment.primary_track_id, risk_reason=assessment.reason,
        risk_factors=json.dumps([f.to_dict() for f in assessment.contributing_factors]),
        risk_gates=json.dumps(list(assessment.gates)), evidence_quality=round(float(assessment.evidence_quality), 4),
        alarm_recommended=bool(assessment.alarm_recommended),
    )


def track_fields(track, hazard=None, width: int | None = None, height: int | None = None) -> dict:
    if track is None:
        return {"image_width": width or None, "image_height": height or None}
    return dict(
        image_width=width or None, image_height=height or None,
        hazard_class=track.observed_class, hazard_track_class=track.class_name,
        hazard_class_changes=int(track.class_changes), hazard_confidence=round(float(track.class_confidence), 4),
        track_persistence_seconds=round(float(track.persistence_seconds), 4), track_detection_count=int(track.detection_count),
        center_x=round(float(track.center_x), 2), center_y=round(float(track.center_y), 2),
        bbox_width=round(float(track.bbox_width), 2), bbox_height=round(float(track.bbox_height), 2),
        bbox_area=round(float(track.area), 1),
        velocity_x_pixels_per_second=None if track.velocity_x_pixels_per_second is None else round(track.velocity_x_pixels_per_second, 2),
        velocity_y_pixels_per_second=None if track.velocity_y_pixels_per_second is None else round(track.velocity_y_pixels_per_second, 2),
        area_change_rate=None if track.area_change_rate is None else round(track.area_change_rate, 1),
        box_growth_per_second=None if hazard is None or hazard.growth_per_second is None else round(hazard.growth_per_second, 4),
        lateral_toward_center_fw_per_s=None if hazard is None or hazard.lateral_toward_corridor is None
        else round(hazard.lateral_toward_corridor, 4),
    )


def driver_fields(driver) -> dict:
    if driver is None:
        return {}
    t = driver.temporal
    return dict(
        driver_hand_state=_val(driver.hand.hand_state),
        driver_hand_confidence=None if driver.hand.confidence is None else round(float(driver.hand.confidence), 4),
        driver_activity=_val(driver.driver_activity), drowsiness_level=_val(t.drowsiness_level),
        drowsiness_score=None if t.drowsiness_score is None else round(float(t.drowsiness_score), 4),
        face_status=_val(driver.observation.face_status),
    )
