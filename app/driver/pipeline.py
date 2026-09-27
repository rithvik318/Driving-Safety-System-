"""DriverPerceptionPipeline: frame -> DriverState.

    frame
      -> face provider           (landmarks or no-face)          app/driver/face.py
      -> head pose + eye features (per-frame observation)        head_pose.py, eyes.py
      -> temporal eye closure     (rolling window)                temporal.py
      -> drowsiness score                                         drowsiness.py
      -> hand-state provider      (plug-in, OBSERVED per frame)  hand_state.py
      -> manual-state durations   (TEMPORAL)                       state_duration.py
      -> driver activity          (INFERRED from the timeline)     infer_activity() below
      -> DriverState                                              models.py

The pipeline never loads a hand-state model itself: pass any HandStateProvider
(e.g. TrainedHandStateProvider, loaded once by the caller).

Observed vs inferred:
    hand_state (BOTH_HANDS / ONE_HAND / NO_HANDS / UNKNOWN) is what the classifier OBSERVED
    in this frame. driver_activity is INFERRED over time: NO_HANDS must be observed
    continuously for NO_HANDS_INFERENCE_SECONDS before it becomes HANDS_OFF_WHEEL.
    ONE_HAND alone never becomes a distraction, and phone use is never inferred from the
    hand classifier. PHONE / OTHER_MANUAL_DISTRACTION only come from a provider that
    reports them as observations.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from app.config.driver_config import DriverConfig
from app.driver.drowsiness import score_drowsiness
from app.driver.eyes import extract_eye_features
from app.driver.face import FaceLandmarkProvider, FaceResult
from app.driver.hand_state import HandStateProvider, UnknownHandStateProvider
from app.driver.head_pose import estimate_head_pose
from app.driver.models import (
    DriverActivity,
    DriverFrameObservation,
    DriverState,
    DISTRACTION_ACTIVITIES,
    DriverTemporalState,
    FaceStatus,
    HandState,
    HandStateResult,
)
from app.driver.state_duration import StateDurationTracker
from app.driver.temporal import EyeClosureTracker

logger = logging.getLogger(__name__)

ACTIVITY_OVERRIDES = {DriverActivity.PHONE, DriverActivity.OTHER_MANUAL_DISTRACTION}


def manual_state(hand: HandStateResult) -> str:
    """Label fed to the duration tracker: an OBSERVED distracting activity wins over the hand state."""
    if hand.activity in ACTIVITY_OVERRIDES:
        return hand.activity.value
    return hand.hand_state.value


def infer_activity(hand: HandStateResult, tracker: StateDurationTracker, config: DriverConfig) -> tuple[DriverActivity, str, str]:
    """INFERRED driver activity from the observed hand state and its duration.

    Returns (activity, source, reason). Never infers distraction from a single frame.
    """
    if hand.activity in ACTIVITY_OVERRIDES:
        return hand.activity, "observed_provider", f"{hand.provider} reported {hand.activity.value}"
    if hand.hand_state is HandState.UNKNOWN:
        return DriverActivity.UNKNOWN, "none", "hand state UNKNOWN; nothing inferred"
    duration = tracker.current_duration or 0.0
    if hand.hand_state is HandState.NO_HANDS and duration >= config.no_hands_inference_seconds:
        return (DriverActivity.HANDS_OFF_WHEEL, "inferred_temporal",
                f"NO_HANDS observed continuously for {duration:.1f} s (>= {config.no_hands_inference_seconds:g} s)")
    if hand.hand_state is HandState.NO_HANDS:
        return (DriverActivity.NORMAL, "inferred_temporal",
                f"NO_HANDS for {duration:.1f} s, below {config.no_hands_inference_seconds:g} s; no distraction inferred yet")
    return DriverActivity.NORMAL, "inferred_temporal", f"{hand.hand_state.value} observed; no manual distraction inferred"


class DriverPerceptionPipeline:
    def __init__(
        self,
        face_provider: FaceLandmarkProvider,
        config: DriverConfig,
        hand_provider: HandStateProvider | None = None,
    ):
        self.face_provider = face_provider
        self.hand_provider = hand_provider or UnknownHandStateProvider()
        self.config = config
        self.eye_tracker = EyeClosureTracker(
            window_seconds=config.drowsiness_window_seconds,
            max_gap_seconds=config.max_gap_seconds,
            blink_max_seconds=config.blink_max_seconds,
            long_closure_seconds=config.long_closure_seconds,
        )
        self.manual_tracker = StateDurationTracker(config.distraction_states)
        self._last_timestamp: float | None = None

    # -- lifecycle -------------------------------------------------------------

    def reset(self) -> None:
        """Clear all temporal state (call when switching videos or starting a new demo)."""
        self.eye_tracker.reset()
        self.manual_tracker.reset()
        self._last_timestamp = None
        for provider in (self.face_provider, self.hand_provider):
            if hasattr(provider, "reset"):
                provider.reset()

    def close(self) -> None:
        if hasattr(self.face_provider, "close"):
            self.face_provider.close()

    # -- per-frame -----------------------------------------------------------------

    def observe(self, frame: np.ndarray, timestamp: float) -> DriverFrameObservation:
        """Per-frame measurements only; no temporal state is touched."""
        try:
            face: FaceResult = self.face_provider.process(frame, timestamp)
        except Exception as exc:
            logger.warning("Face provider raised: %s", exc)
            face = FaceResult(FaceStatus.PROVIDER_ERROR, detail=str(exc))

        if not face.face_detected:
            return DriverFrameObservation(
                timestamp=timestamp, face_status=face.status, face_detected=False, landmarks_available=False
            )

        pose = estimate_head_pose(face.landmarks)
        eyes = extract_eye_features(face.landmarks)
        return DriverFrameObservation(
            timestamp=timestamp,
            face_status=FaceStatus.OK,
            face_detected=True,
            landmarks_available=True,
            head_yaw=None if pose is None else pose.yaw,
            head_pitch=None if pose is None else pose.pitch,
            head_roll=None if pose is None else pose.roll,
            left_eye_openness=eyes.left_openness,
            right_eye_openness=eyes.right_openness,
            mean_eye_openness=eyes.mean_openness,
            eyes_closed=eyes.eyes_closed(self.config.eye_closure_threshold),
        )

    def _predict_hand(self, frame: np.ndarray) -> HandStateResult:
        try:
            return self.hand_provider.predict(frame)
        except Exception as exc:
            logger.warning("Hand-state provider %s raised: %s", getattr(self.hand_provider, "name", "?"), exc)
            return HandStateResult(HandState.UNKNOWN, DriverActivity.UNKNOWN, None, getattr(self.hand_provider, "name", "unknown"))

    def _check_timestamp(self, timestamp: float | None) -> tuple[float, bool]:
        if timestamp is None:
            timestamp = time.monotonic()
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError(f"timestamp must be a finite number of seconds, got {timestamp!r}")
        timestamp = float(timestamp)
        went_back = self._last_timestamp is not None and timestamp < self._last_timestamp
        if went_back:
            logger.warning(
                "Timestamp went backwards (%.3f < %.3f); resetting temporal state. Call reset() when switching videos.",
                timestamp, self._last_timestamp,
            )
            self.reset()
        self._last_timestamp = timestamp
        return timestamp, went_back

    def process(self, frame: np.ndarray, timestamp: float | None = None) -> DriverState:
        """Run the full driver pipeline on one frame. Never raises for bad frames."""
        t, timeline_reset = self._check_timestamp(timestamp)

        obs = self.observe(frame, t)
        eye = self.eye_tracker.update(t, obs.eyes_closed)
        score, level = score_drowsiness(eye, self.config)

        hand = self._predict_hand(frame)  # OBSERVED
        state_label = manual_state(hand)
        self.manual_tracker.update(t, state_label)  # TEMPORAL
        activity, activity_source, activity_reason = infer_activity(hand, self.manual_tracker, self.config)  # INFERRED
        if activity in DISTRACTION_ACTIVITIES:
            distraction = self.manual_tracker.distraction_duration
        else:
            distraction = None if activity is DriverActivity.UNKNOWN else 0.0

        temporal = DriverTemporalState(
            eye_closure_ratio=eye.eye_closure_ratio,
            eye_closed_duration=eye.eye_closed_duration,
            blink_count=eye.blink_count,
            long_closure_count=eye.long_closure_count,
            observed_seconds=eye.observed_seconds,
            face_missing_duration=eye.face_missing_duration,
            drowsiness_score=score,
            drowsiness_level=level,
            manual_state=state_label,
            manual_state_duration=_round(self.manual_tracker.current_duration),
            manual_state_transitions=self.manual_tracker.transition_count,
            longest_one_hand_duration=_round(self.manual_tracker.longest_duration(HandState.ONE_HAND.value)),
            longest_no_hands_duration=_round(self.manual_tracker.longest_duration(HandState.NO_HANDS.value)),
            distraction_duration=_round(distraction),
            timeline_reset=timeline_reset,
        )
        available = [obs.face_detected, obs.head_yaw is not None, obs.left_eye_openness is not None, obs.right_eye_openness is not None]
        return DriverState(
            timestamp=t,
            observation=obs,
            temporal=temporal,
            hand=hand,
            observation_quality=sum(available) / len(available),
            driver_activity=activity,
            activity_source=activity_source,
            activity_reason=activity_reason,
        )


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)
