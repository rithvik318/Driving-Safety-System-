"""Drowsiness score from temporal eye-closure features.

PROTOTYPE BEHAVIOURAL-RISK SIGNAL. Not medically validated, not a diagnosis.
All thresholds come from DriverConfig and should be tuned on real recordings.

score = max(current_closure_part, window_part)
    current_closure_part = min(1, eye_closed_duration / SUSTAINED_CLOSURE_MAX_SECONDS)
    window_part          = w * min(1, eye_closure_ratio / CLOSURE_RATIO_MAX)
                         + (1 - w) * min(1, long_closure_count / LONG_CLOSURE_EVENTS_MAX)
                           (only once MIN_OBSERVED_SECONDS of face time is available)

So a single closed frame or a normal blink gives a score near 0; a sustained
closure raises the score immediately; frequent closures raise it over the window.
"""

from __future__ import annotations

from app.config.driver_config import DriverConfig
from app.driver.models import DrowsinessLevel
from app.driver.temporal import EyeClosureFeatures


def score_drowsiness(f: EyeClosureFeatures, c: DriverConfig) -> tuple[float | None, DrowsinessLevel]:
    current = 0.0
    if f.eye_closed_duration:
        current = min(1.0, f.eye_closed_duration / c.sustained_closure_max_seconds)

    window: float | None = None
    if f.observed_seconds >= c.min_observed_seconds and f.eye_closure_ratio is not None:
        ratio_part = min(1.0, f.eye_closure_ratio / c.closure_ratio_max)
        event_part = min(1.0, f.long_closure_count / c.long_closure_events_max)
        window = c.closure_ratio_weight * ratio_part + (1 - c.closure_ratio_weight) * event_part

    if window is None:
        if current == 0.0:
            return None, DrowsinessLevel.UNKNOWN  # not enough evidence either way
        score = current
    else:
        score = max(current, window)
    score = round(score, 4)

    if score >= c.drowsiness_critical_threshold:
        level = DrowsinessLevel.CRITICAL
    elif score >= c.drowsiness_high_threshold:
        level = DrowsinessLevel.HIGH
    elif window is None:
        level = DrowsinessLevel.UNKNOWN  # low score, but based on too little history
    else:
        level = DrowsinessLevel.LOW
    return score, level
