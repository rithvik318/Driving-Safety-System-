"""Risk engine data models.

risk_score (0-100) is a RULE-BASED PROTOTYPE SCORE: the clamped sum of documented point
contributions (app/config/risk_config.py). It is not a probability of collision and not a
validated safety classification. Levels are decision-support states.

All road measurements come from image space (pixels, frame fractions, growth per second).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(str, Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return ["SAFE", "CAUTION", "HIGH", "CRITICAL"].index(self.value)

    @classmethod
    def from_rank(cls, rank: int) -> "RiskLevel":
        return [cls.SAFE, cls.CAUTION, cls.HIGH, cls.CRITICAL][max(0, min(3, rank))]


class HazardType(str, Enum):
    NONE = "NONE"
    APPROACHING_VEHICLE = "APPROACHING_VEHICLE"
    PEDESTRIAN_CONFLICT = "PEDESTRIAN_CONFLICT"  # image-space conflict pattern, not a predicted collision
    ANIMAL_HAZARD = "ANIMAL_HAZARD"
    ROAD_USER_PRESENT = "ROAD_USER_PRESENT"  # persistent road user WITHOUT approach/conflict evidence
    DRIVER_DROWSINESS = "DRIVER_DROWSINESS"
    HANDS_OFF_WHEEL = "HANDS_OFF_WHEEL"
    DRIVER_HEAD_AWAY = "DRIVER_HEAD_AWAY"  # head pose turned away for a sustained time
    COMBINED_DRIVER_HAZARD = "COMBINED_DRIVER_HAZARD"


@dataclass(frozen=True)
class RiskSnapshot:
    """Everything the engine needs for one decision. No images, no detector, no OpenCV.

    tracks: TrackedObject list from app.tracking (front camera).
    driver: DriverState from app.driver, or None if no driver observation is available.
    image_width / image_height: size of the front-camera frames the tracks were measured in.
    """

    timestamp: float
    tracks: tuple = ()
    driver: object | None = None
    image_width: int = 0
    image_height: int = 0


@dataclass(frozen=True)
class RiskFactor:
    """One documented contribution to the score."""

    name: str  # e.g. "hands_off_wheel", "approach", "centrality", "evidence_quality"
    points: float  # signed contribution to raw_risk_score
    source: str  # "driver" | "road" | "combined" | "quality"
    detail: str  # human-readable evidence, generated from numbers (no LLM)
    track_id: int | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "points": round(self.points, 1), "source": self.source,
                "detail": self.detail, "track_id": self.track_id}


@dataclass(frozen=True)
class TrackHazard:
    """Rule evaluation of one track (for inspection; the best one drives the result)."""

    track_id: int
    class_name: str
    kind: str  # "vehicle" | "person" | "animal"
    hazard_type: HazardType
    qualifies: bool  # has approach / conflict evidence (else presence only, capped)
    points: float  # after the evidence-quality multiplier and presence cap
    factors: tuple[RiskFactor, ...]
    evidence_groups: tuple[str, ...]  # independent evidence kinds: "motion", "position"
    growth_per_second: float | None
    lateral_toward_corridor: float | None  # frame-widths/s, > 0 = toward the image centre band
    center_x_norm: float
    bottom_norm: float
    area_fraction: float
    quality: float
    phrases: tuple[str, ...] = ()  # label-free evidence phrases used to build the reason text

    def to_dict(self) -> dict:
        r = lambda v, n=3: None if v is None else round(v, n)  # noqa: E731
        return {"track_id": self.track_id, "class_name": self.class_name, "kind": self.kind,
                "hazard_type": self.hazard_type.value, "qualifies": self.qualifies, "points": r(self.points, 1),
                "evidence_groups": list(self.evidence_groups), "growth_per_second": r(self.growth_per_second),
                "lateral_toward_corridor": r(self.lateral_toward_corridor), "center_x_norm": r(self.center_x_norm),
                "bottom_norm": r(self.bottom_norm), "area_fraction": r(self.area_fraction, 4), "quality": r(self.quality),
                "factors": [f.to_dict() for f in self.factors]}


@dataclass(frozen=True)
class RiskAssessment:
    timestamp: float
    risk_level: RiskLevel  # after hysteresis: use this for decisions
    raw_risk_level: RiskLevel  # this observation alone (after gates), before hysteresis
    raw_risk_score: float  # 0-100, this observation alone
    smoothed_risk_score: float  # 0-100, rises immediately, decays gradually
    hazard_type: HazardType
    primary_track_id: int | None
    primary_class: str | None
    contributing_factors: tuple[RiskFactor, ...]
    reason: str
    evidence_quality: float  # 0-1: how well supported the contributing evidence is (not a probability)
    evidence_notes: tuple[str, ...]
    alarm_recommended: bool
    gates: tuple[str, ...] = ()  # level caps that applied, e.g. "HIGH needs >= 2 independent evidence groups"
    hazards: tuple[TrackHazard, ...] = field(default_factory=tuple)  # all evaluated tracks, strongest first
    driver_points: float = 0.0
    road_points: float = 0.0

    @property
    def risk_score(self) -> float:
        """The score to display: the smoothed prototype score (see module docstring)."""
        return self.smoothed_risk_score

    def to_dict(self, include_hazards: bool = False) -> dict:
        d = {
            "timestamp": round(self.timestamp, 3),
            "risk_level": self.risk_level.value,
            "raw_risk_level": self.raw_risk_level.value,
            "risk_score": round(self.risk_score, 1),
            "raw_risk_score": round(self.raw_risk_score, 1),
            "smoothed_risk_score": round(self.smoothed_risk_score, 1),
            "hazard_type": self.hazard_type.value,
            "primary_track_id": self.primary_track_id,
            "primary_class": self.primary_class,
            "contributing_factors": [f.to_dict() for f in self.contributing_factors],
            "reason": self.reason,
            "evidence_quality": round(self.evidence_quality, 3),
            "evidence_notes": list(self.evidence_notes),
            "alarm_recommended": self.alarm_recommended,
            "gates": list(self.gates),
            "driver_points": round(self.driver_points, 1),
            "road_points": round(self.road_points, 1),
            "score_note": "rule-based prototype score (0-100), not a probability of collision",
        }
        if include_hazards:
            d["hazards"] = [h.to_dict() for h in self.hazards]
        return d
