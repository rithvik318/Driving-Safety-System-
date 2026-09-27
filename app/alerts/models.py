"""Safety-alert data model.

An Alert is a decision-support notification derived from the EXISTING risk engine's output
(RiskAssessment, or an event record built from one). It adds no risk logic of its own: the level,
score, hazard type, reason and alarm recommendation all come from the engine unchanged.

Alerts are not collision predictions and do not guarantee safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

LEVELS = ("SAFE", "CAUTION", "HIGH", "CRITICAL")
DISCLAIMER = ("Decision-support alert from a rule-based prototype risk score (not a probability). It is not a "
              "collision prediction and does not guarantee safety.")


class AlertSeverity(str, Enum):
    NONE = "NONE"  # SAFE: no alert
    NOTICE = "NOTICE"  # CAUTION: visual / log notification only
    WARNING = "WARNING"  # HIGH: clear warning + audible alarm
    CRITICAL = "CRITICAL"  # CRITICAL: strong warning + audible alarm

    @property
    def rank(self) -> int:
        return ["NONE", "NOTICE", "WARNING", "CRITICAL"].index(self.value)


LEVEL_TO_SEVERITY = {"SAFE": AlertSeverity.NONE, "CAUTION": AlertSeverity.NOTICE,
                     "HIGH": AlertSeverity.WARNING, "CRITICAL": AlertSeverity.CRITICAL}

CHANNELS = {
    AlertSeverity.NONE: (),
    AlertSeverity.NOTICE: ("visual", "log"),
    AlertSeverity.WARNING: ("visual", "log", "audible"),
    AlertSeverity.CRITICAL: ("visual", "log", "audible_strong"),
}

DRIVER_HAZARDS = ("DRIVER_DROWSINESS", "HANDS_OFF_WHEEL", "DRIVER_HEAD_AWAY")


def headline(severity: AlertSeverity, hazard_type: str | None) -> str | None:
    """Fixed alert wording. HIGH says ROAD HAZARD for road/combined hazards and DRIVER STATE when the only
    contributing hazard is the driver's state (a driver-only HIGH is not a road hazard)."""
    if severity is AlertSeverity.CRITICAL:
        return "CRITICAL: IMMEDIATE ATTENTION REQUIRED"
    if severity is AlertSeverity.WARNING:
        return "WARNING: DRIVER STATE" if hazard_type in DRIVER_HAZARDS else "WARNING: ROAD HAZARD"
    if severity is AlertSeverity.NOTICE:
        return "CAUTION: hazard noted"
    return None


# short, demo-friendly phrases per engine hazard type (the full explanation is always risk_reason)
HAZARD_PHRASES = {
    "NONE": "no active hazard",
    "APPROACHING_VEHICLE": "approaching vehicle / increasing image-space box",
    "PEDESTRIAN_CONFLICT": "pedestrian in or moving toward the central road region",
    "ANIMAL_HAZARD": "animal in or moving toward the central road region",
    "ROAD_USER_PRESENT": "road user present (no approach evidence)",
    "DRIVER_DROWSINESS": "driver drowsiness",
    "HANDS_OFF_WHEEL": "hands off the wheel (sustained)",
    "DRIVER_HEAD_AWAY": "driver looking away (sustained)",
    "COMBINED_DRIVER_HAZARD": "driver + road hazard combination",
}


@dataclass(frozen=True)
class Alert:
    alert_id: str
    timestamp: float
    kind: str  # "ESCALATION" | "DEESCALATION"
    risk_level: str
    previous_level: str
    severity: AlertSeverity
    headline: str | None
    phrase: str
    risk_reason: str | None  # the engine's own explanation, unchanged
    risk_score: float | None  # smoothed prototype score 0-100 (not a probability)
    hazard_type: str | None
    primary_track_id: int | None
    primary_class: str | None
    alarm: bool  # audible alarm raised by this alert (escalation into the engine's alarm level)
    channels: tuple[str, ...]
    event_ids: tuple[str, ...] = ()
    session_id: str | None = None
    data_source: str = "LOCAL_REAL"
    observation_type: str = "INFERRED"
    disclaimer: str = DISCLAIMER
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id, "timestamp": round(self.timestamp, 3), "kind": self.kind,
            "risk_level": self.risk_level, "previous_level": self.previous_level, "severity": self.severity.value,
            "headline": self.headline, "phrase": self.phrase, "risk_reason": self.risk_reason,
            "risk_score": None if self.risk_score is None else round(self.risk_score, 1), "hazard_type": self.hazard_type,
            "primary_track_id": self.primary_track_id, "primary_class": self.primary_class, "alarm": self.alarm,
            "channels": list(self.channels), "event_ids": list(self.event_ids), "session_id": self.session_id,
            "data_source": self.data_source, "observation_type": self.observation_type, "disclaimer": self.disclaimer,
        }
