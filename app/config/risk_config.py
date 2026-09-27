"""Contextual risk engine configuration. ALL values are PROTOTYPE choices, not validated.

risk_score is a bounded (0-100) RULE-BASED prototype score built from the documented point
contributions below. It is NOT a probability of collision and is not statistically calibrated.

Every field can be overridden with the environment variable RISK_<FIELD NAME IN CAPITALS>, e.g.
RISK_HIGH_MIN=50 or RISK_VEHICLE_CLASSES=car,truck,bus.

Image-space conventions: positions are normalised by frame width/height (0..1, origin top-left).
"Corridor" is an image band around the horizontal centre used as a stand-in for the area in
front of the camera; it is NOT a calibrated road model or the vehicle's real path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

from app.config.settings import PROJECT_ROOT

LEVELS = ("SAFE", "CAUTION", "HIGH", "CRITICAL")


@dataclass(frozen=True)
class RiskConfig:
    # ---------------- level thresholds on the 0-100 score
    caution_min: float = 20.0
    high_min: float = 45.0
    critical_min: float = 70.0

    # ---------------- which tracks can be hazards (COCO names from the detector)
    vehicle_classes: tuple[str, ...] = ("car", "truck", "bus", "motorcycle", "bicycle", "train")
    person_classes: tuple[str, ...] = ("person",)
    animal_classes: tuple[str, ...] = ("dog", "cat", "cow", "horse", "sheep", "elephant")

    # ---------------- track evidence gates
    min_persistence_seconds: float = 0.5  # observed span before a track can contribute
    min_detections: int = 3  # matched detections before a track can contribute
    max_track_missed_seconds: float = 0.3  # coasting longer than this: ignored this frame

    # ---------------- per-track temporal features (engine keeps a short history per track)
    feature_window_seconds: float = 1.0  # growth / lateral motion measured over this window
    min_window_seconds: float = 0.3  # ... needing at least this time span
    min_window_samples: int = 3  # ... and this many observations

    # ---------------- image regions (normalised)
    corridor_half_width: float = 0.20  # |center_x - 0.5| <= this -> "central"
    wide_half_width: float = 0.35  # <= this -> "near centre" (half centrality points)
    lower_region_min: float = 0.45  # bbox bottom (y2 / height) >= this -> lower image region

    # ---------------- motion evidence (image space)
    growth_min: float = 0.15  # log-area growth per second counted as approach (0.15 ~ +16 %/s)
    growth_full: float = 0.80  # growth giving full approach points (0.8 ~ x2.2 per second)
    lateral_min: float = 0.05  # frame-widths/s toward the corridor counted as trajectory evidence
    lateral_full: float = 0.30
    size_min: float = 0.01  # box area / frame area where image-size points start
    size_full: float = 0.12  # ... and reach their maximum
    persistence_full_seconds: float = 2.0

    # ---------------- points per road factor, by hazard kind (maximum points of each factor)
    vehicle_persistence_points: float = 10.0
    vehicle_centrality_points: float = 10.0
    vehicle_approach_points: float = 25.0
    vehicle_size_points: float = 10.0
    vehicle_trajectory_points: float = 5.0
    person_persistence_points: float = 10.0
    person_centrality_points: float = 15.0
    person_approach_points: float = 15.0
    person_size_points: float = 5.0
    person_trajectory_points: float = 20.0
    animal_persistence_points: float = 10.0
    animal_centrality_points: float = 15.0
    animal_approach_points: float = 15.0
    animal_size_points: float = 5.0
    animal_trajectory_points: float = 20.0
    presence_cap_points: float = 20.0  # a track WITHOUT hazard evidence contributes at most this
    multi_hazard_bonus_points: float = 5.0  # >= 2 qualifying road hazards at once
    road_cap_points: float = 70.0

    # ---------------- evidence quality multiplier on a track's points
    confidence_reference: float = 0.60  # mean detector confidence at/above which there is no reduction
    min_confidence_multiplier: float = 0.50
    label_change_penalty: float = 0.15  # per observed class change
    min_stability_multiplier: float = 0.50

    # ---------------- driver factors (only existing DriverState signals)
    drowsiness_high_points: float = 30.0
    drowsiness_critical_points: float = 45.0
    hands_off_wheel_points: float = 25.0  # driver_activity == HANDS_OFF_WHEEL (ONE_HAND scores 0)
    head_away_points: float = 10.0
    head_away_yaw_degrees: float = 30.0  # |head_yaw| above this ...
    head_away_pitch_degrees: float = 25.0  # ... or head_pitch (down) above this ...
    head_away_min_seconds: float = 1.0  # ... continuously for this long
    driver_cap_points: float = 60.0
    driver_max_age_seconds: float = 1.0  # older DriverState is treated as unavailable

    # ---------------- combination
    combined_bonus_points: float = 15.0  # driver factor AND qualifying road hazard
    combined_reference_points: float = 30.0  # road points giving the full combined bonus
    strong_hazard_points: float = 60.0  # road-only CRITICAL needs at least this ...

    # ---------------- hysteresis / smoothing
    escalate_caution_observations: int = 1  # consecutive observations needed to enter each level
    escalate_high_observations: int = 2
    escalate_critical_observations: int = 3
    deescalate_observations: int = 2  # consecutive lower observations to step DOWN one level
    score_decay_per_second: float = 40.0  # smoothed score falls at most this fast (rises immediately)
    reset_gap_seconds: float = 2.0  # longer gap between evaluations resets smoothing
    alarm_min_level: str = "HIGH"  # alarm_recommended when the smoothed level is at least this


def _cast(raw: str, current):
    if isinstance(current, tuple):
        return tuple(s.strip().lower() for s in raw.split(",") if s.strip())
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw.strip().upper()


def load_risk_config(env_file: Path | None = None) -> RiskConfig:
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    defaults = RiskConfig()
    kwargs = {}
    for f in fields(RiskConfig):
        raw = (os.environ.get(f"RISK_{f.name.upper()}") or "").strip().strip('"').strip("'")
        if raw:
            try:
                kwargs[f.name] = _cast(raw, getattr(defaults, f.name))
            except ValueError as exc:
                raise ValueError(f"RISK_{f.name.upper()} has an invalid value {raw!r}") from exc
    config = RiskConfig(**kwargs)
    validate_risk_config(config)
    return config


def validate_risk_config(c: RiskConfig) -> None:
    if not 0 < c.caution_min < c.high_min < c.critical_min <= 100:
        raise ValueError("need 0 < RISK_CAUTION_MIN < RISK_HIGH_MIN < RISK_CRITICAL_MIN <= 100")
    if c.alarm_min_level not in LEVELS:
        raise ValueError(f"RISK_ALARM_MIN_LEVEL must be one of {LEVELS}")
    for name in ("growth_min", "lateral_min", "size_min"):
        full = getattr(c, name.replace("_min", "_full"))
        if not 0 <= getattr(c, name) < full:
            raise ValueError(f"RISK_{name.upper()} must be >= 0 and below its _FULL value")
    for f in fields(RiskConfig):
        v = getattr(c, f.name)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0:
            raise ValueError(f"RISK_{f.name.upper()} must be >= 0")
    if min(c.escalate_caution_observations, c.escalate_high_observations, c.escalate_critical_observations,
           c.deescalate_observations, c.min_window_samples) < 1:
        raise ValueError("observation counts must be >= 1")
    for a, b in [("corridor_half_width", "wide_half_width")]:
        if getattr(c, a) > getattr(c, b):
            raise ValueError("RISK_CORRIDOR_HALF_WIDTH must be <= RISK_WIDE_HALF_WIDTH")
