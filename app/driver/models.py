"""Driver-perception data models.

Three layers, kept separate on purpose:
1. DriverFrameObservation: what one frame shows (no memory).
2. DriverTemporalState:    what the recent history shows (eye closure over time, durations).
3. DriverState:            the final record handed to the rest of the system.

Timestamps are float seconds on one clock per session: video time for recorded
files, or time.monotonic() for a live camera. They must not go backwards within
a session; call pipeline.reset() when switching videos.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class HandState(str, Enum):
    BOTH_HANDS = "BOTH_HANDS"
    ONE_HAND = "ONE_HAND"
    NO_HANDS = "NO_HANDS"
    UNKNOWN = "UNKNOWN"


class DriverActivity(str, Enum):
    """INFERRED activity. Never set from a single hand-classifier frame.

    NORMAL                    no manual distraction inferred (not a guarantee of attention)
    HANDS_OFF_WHEEL           NO_HANDS observed continuously for >= NO_HANDS_INFERENCE_SECONDS
    PHONE / OTHER_MANUAL_...  only when another observation source reports them (not the hand classifier)
    UNKNOWN                   hand state unknown, so nothing can be inferred
    """

    NORMAL = "NORMAL"
    HANDS_OFF_WHEEL = "HANDS_OFF_WHEEL"
    PHONE = "PHONE"
    OTHER_MANUAL_DISTRACTION = "OTHER_MANUAL_DISTRACTION"
    UNKNOWN = "UNKNOWN"


DISTRACTION_ACTIVITIES = frozenset({DriverActivity.HANDS_OFF_WHEEL, DriverActivity.PHONE, DriverActivity.OTHER_MANUAL_DISTRACTION})


class FaceStatus(str, Enum):
    OK = "OK"
    NO_FACE = "NO_FACE"
    INVALID_FRAME = "INVALID_FRAME"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"


class DrowsinessLevel(str, Enum):
    UNKNOWN = "UNKNOWN"  # not enough observed face time yet
    LOW = "LOW"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class HandStateResult:
    """OBSERVED output of a HandStateProvider for one frame.

    `confidence` is the classifier's top class probability (reported even when the state is
    UNKNOWN because it fell below the threshold); None when the provider has no estimate.
    `activity` is only set by a provider that actually observes an activity (e.g. a future
    phone detector); the hand classifier leaves it UNKNOWN.
    """

    hand_state: HandState = HandState.UNKNOWN
    activity: DriverActivity = DriverActivity.UNKNOWN
    confidence: float | None = None
    provider: str = "unknown"
    class_probabilities: dict | None = None


@dataclass(frozen=True)
class DriverFrameObservation:
    """Per-frame measurements. Any measurement that could not be made is None."""

    timestamp: float
    face_status: FaceStatus
    face_detected: bool
    landmarks_available: bool
    head_yaw: float | None = None  # degrees, see app/driver/head_pose.py for signs
    head_pitch: float | None = None
    head_roll: float | None = None
    left_eye_openness: float | None = None  # driver's own left eye; eye-aspect ratio
    right_eye_openness: float | None = None
    mean_eye_openness: float | None = None
    eyes_closed: bool | None = None  # None = could not be measured this frame


@dataclass(frozen=True)
class DriverTemporalState:
    """Rolling-window features. None means "not enough evidence", never "fine"."""

    eye_closure_ratio: float | None = None  # share of observed window time with eyes closed
    eye_closed_duration: float | None = None  # current continuous closure, seconds
    blink_count: int = 0  # closures shorter than BLINK_MAX_SECONDS in the window
    long_closure_count: int = 0  # closures of at least LONG_CLOSURE_SECONDS in the window
    observed_seconds: float = 0.0  # face-observed time in the window
    face_missing_duration: float = 0.0  # current continuous time without a usable face
    drowsiness_score: float | None = None  # [0, 1], prototype behavioural signal
    drowsiness_level: DrowsinessLevel = DrowsinessLevel.UNKNOWN
    manual_state: str = "UNKNOWN"  # state fed to the duration tracker (observed hand state, or a reported activity)
    manual_state_duration: float | None = None  # continuous time in the current manual state
    manual_state_transitions: int = 0  # state changes since the last reset
    longest_one_hand_duration: float = 0.0  # longest continuous ONE_HAND since reset (incl. current)
    longest_no_hands_duration: float = 0.0  # longest continuous NO_HANDS since reset (incl. current)
    distraction_duration: float | None = None  # continuous time of the INFERRED distraction; 0.0 if none; None if unknown
    timeline_reset: bool = False  # True if a backwards timestamp forced a reset


@dataclass(frozen=True)
class DriverState:
    """Final driver record consumed by the risk engine and event recorder."""

    timestamp: float
    observation: DriverFrameObservation
    temporal: DriverTemporalState
    hand: HandStateResult = field(default_factory=HandStateResult)
    # Share of per-frame signals available (face, head pose, left eye, right eye).
    # It describes data availability, NOT a calibrated probability.
    observation_quality: float = 0.0
    # INFERRED layer (kept apart from the observed hand state above)
    driver_activity: DriverActivity = DriverActivity.UNKNOWN
    activity_source: str = "none"  # "inferred_temporal" | "observed_provider" | "none"
    activity_reason: str | None = None

    def to_dict(self) -> dict:
        """Flat, JSON-friendly view with the stable field names used downstream."""
        o, t = self.observation, self.temporal
        return {
            "timestamp": self.timestamp,
            "hand_state": self.hand.hand_state.value,  # observed
            "hand_state_confidence": self.hand.confidence,
            "hand_state_provider": self.hand.provider,
            "driver_activity": self.driver_activity.value,  # inferred
            "activity_source": self.activity_source,
            "activity_reason": self.activity_reason,
            "face_detected": o.face_detected,
            "face_status": o.face_status.value,
            "head_yaw": o.head_yaw,
            "head_pitch": o.head_pitch,
            "head_roll": o.head_roll,
            "left_eye_openness": o.left_eye_openness,
            "right_eye_openness": o.right_eye_openness,
            "mean_eye_openness": o.mean_eye_openness,
            "eyes_closed": o.eyes_closed,
            "eye_closure_ratio": t.eye_closure_ratio,
            "eye_closed_duration": t.eye_closed_duration,
            "blink_count": t.blink_count,
            "long_closure_count": t.long_closure_count,
            "observed_seconds": t.observed_seconds,
            "face_missing_duration": t.face_missing_duration,
            "drowsiness_score": t.drowsiness_score,
            "drowsiness_level": t.drowsiness_level.value,
            "manual_state": t.manual_state,
            "manual_state_duration": t.manual_state_duration,
            "manual_state_transitions": t.manual_state_transitions,
            "longest_one_hand_duration": t.longest_one_hand_duration,
            "longest_no_hands_duration": t.longest_no_hands_duration,
            "distraction_duration": t.distraction_duration,
            "timeline_reset": t.timeline_reset,
            "observation_quality": self.observation_quality,
        }

    def to_nested_dict(self) -> dict:
        """Nested view that keeps the three layers apart (useful for debugging)."""
        return _jsonable(asdict(self))


def _jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
