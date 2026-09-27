"""Driver-perception configuration.

All values are PROTOTYPE DEFAULTS chosen to be conservative. They are not
medically or clinically validated and should be tuned on our own recordings.
Every value can be overridden with the environment variable of the same name
(upper-case), e.g. EYE_CLOSURE_THRESHOLD=0.18.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

from app.config.settings import PROJECT_ROOT, _resolve_path

# Default location for the MediaPipe Face Landmarker model (downloaded manually).
DEFAULT_FACE_MODEL = "models/face/face_landmarker.task"
DEFAULT_HAND_MODEL = "outputs/models/hand_state_best.pt"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


@dataclass(frozen=True)
class DriverConfig:
    # Face landmarks
    face_model_path: Path
    min_face_confidence: float = 0.5  # MediaPipe detection/presence/tracking confidence

    # Eye features: eye-aspect-ratio below this is treated as "closed" for that frame
    eye_closure_threshold: float = 0.20

    # Temporal eye-closure / drowsiness
    drowsiness_window_seconds: float = 60.0  # rolling window for closure ratio and events
    min_observed_seconds: float = 5.0  # observed face time needed before window stats count
    max_gap_seconds: float = 0.5  # longer gaps between frames break a closure run
    blink_max_seconds: float = 0.5  # closure shorter than this = blink
    long_closure_seconds: float = 1.0  # closure at least this long = long-closure event
    sustained_closure_max_seconds: float = 2.0  # current closure this long -> component = 1.0
    closure_ratio_max: float = 0.30  # window closure ratio this high -> component = 1.0
    long_closure_events_max: int = 3  # this many long closures in window -> component = 1.0
    closure_ratio_weight: float = 0.6  # window score = w * ratio part + (1 - w) * events part
    drowsiness_high_threshold: float = 0.5
    drowsiness_critical_threshold: float = 0.8

    # Manual-distraction duration: which hand/activity states count as distracting
    distraction_states: tuple[str, ...] = ("NO_HANDS", "PHONE", "OTHER_MANUAL_DISTRACTION")

    # Hand-state classifier (trained MobileNetV3-Small checkpoint) and its temporal inference
    hand_model_path: Path = Path(DEFAULT_HAND_MODEL)
    hand_confidence_threshold: float = 0.60  # max class probability below this -> hand_state UNKNOWN
    # NO_HANDS must be OBSERVED continuously this long before the INFERRED driver_activity becomes
    # HANDS_OFF_WHEEL. ONE_HAND is never inferred as distraction on its own.
    no_hands_inference_seconds: float = 2.0


def _cast(value: str, target_type):
    if target_type is tuple or str(target_type).startswith("tuple"):
        return tuple(s.strip().upper() for s in value.split(",") if s.strip())
    if target_type is int or target_type == "int":
        return int(value)
    return float(value)


def load_driver_config(env_file: Path | None = None) -> DriverConfig:
    """Build DriverConfig from defaults + environment. Raises ValueError on bad values."""
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

    model_raw = os.environ.get("FACE_MODEL_PATH", "").strip().strip('"').strip("'") or DEFAULT_FACE_MODEL
    hand_raw = os.environ.get("HAND_MODEL_PATH", "").strip().strip('"').strip("'") or DEFAULT_HAND_MODEL
    kwargs: dict = {
        "face_model_path": _resolve_path(model_raw, PROJECT_ROOT),
        "hand_model_path": _resolve_path(hand_raw, PROJECT_ROOT),
    }

    for f in fields(DriverConfig):
        if f.name in ("face_model_path", "hand_model_path"):
            continue
        raw = os.environ.get(f.name.upper())
        if raw is None or raw.strip() == "":
            continue
        try:
            kwargs[f.name] = _cast(raw.strip(), f.type)
        except ValueError as exc:
            raise ValueError(f"{f.name.upper()} has an invalid value {raw!r}") from exc

    config = DriverConfig(**kwargs)
    _validate(config)
    return config


def _validate(c: DriverConfig) -> None:
    positive = [
        "eye_closure_threshold", "no_hands_inference_seconds", "drowsiness_window_seconds", "max_gap_seconds", "blink_max_seconds", "long_closure_seconds",
        "sustained_closure_max_seconds", "closure_ratio_max", "long_closure_events_max",
    ]
    for name in positive:
        if getattr(c, name) <= 0:
            raise ValueError(f"{name.upper()} must be > 0")
    for name in ["min_face_confidence", "hand_confidence_threshold", "closure_ratio_weight", "drowsiness_high_threshold", "drowsiness_critical_threshold"]:
        if not 0.0 <= getattr(c, name) <= 1.0:
            raise ValueError(f"{name.upper()} must be between 0 and 1")
    if c.drowsiness_critical_threshold < c.drowsiness_high_threshold:
        raise ValueError("DROWSINESS_CRITICAL_THRESHOLD must be >= DROWSINESS_HIGH_THRESHOLD")
    if c.blink_max_seconds > c.long_closure_seconds:
        raise ValueError("BLINK_MAX_SECONDS must be <= LONG_CLOSURE_SECONDS")
    if c.min_observed_seconds < 0 or c.min_observed_seconds > c.drowsiness_window_seconds:
        raise ValueError("MIN_OBSERVED_SECONDS must be between 0 and DROWSINESS_WINDOW_SECONDS")
