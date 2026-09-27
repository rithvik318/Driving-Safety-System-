"""Hand-state provider plug-in point.

Interface:  result = provider.predict(frame_bgr)  ->  HandStateResult

The pipeline only depends on this interface. Implementations:
- UnknownHandStateProvider: placeholder, always UNKNOWN (no model needed).
- TrainedHandStateProvider: the trained MobileNetV3-Small classifier, loaded ONCE at
  construction and reused for every frame. Requires torch/torchvision (imported lazily).

The labels describe the driver's manual-control state on the steering wheel
(both hands / one hand / no hands on the wheel). This is not a generic hand counter,
and it does not detect phone use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from app.driver.models import DriverActivity, HandState, HandStateResult


@runtime_checkable
class HandStateProvider(Protocol):
    name: str

    def predict(self, frame: np.ndarray) -> HandStateResult: ...

    def reset(self) -> None: ...


class UnknownHandStateProvider:
    """Placeholder until a hand-state model is trained. Never fabricates a prediction."""

    name = "unknown_placeholder"

    def predict(self, frame: np.ndarray) -> HandStateResult:
        return HandStateResult(
            hand_state=HandState.UNKNOWN,
            activity=DriverActivity.UNKNOWN,
            confidence=None,
            provider=self.name,
        )

    def reset(self) -> None:
        pass


class TrainedHandStateProvider:
    """Hand-state provider backed by the trained classifier (app/driver/hand_model.py).

    Per frame it returns the OBSERVED hand state. Below the confidence threshold the state is
    UNKNOWN, but the confidence (top class probability) is still reported. It never sets an
    activity: activity is inferred later by the pipeline from the timeline.
    """

    name = "mobilenet_v3_small"

    def __init__(self, classifier, color_order: str = "BGR"):
        self.classifier = classifier
        self.color_order = color_order
        self.threshold = classifier.threshold

    @classmethod
    def from_checkpoint(cls, checkpoint: Path, threshold: float = 0.60, device: str = "cpu") -> "TrainedHandStateProvider":
        from app.driver.hand_model import HandStateClassifier  # needs torch

        return cls(HandStateClassifier.from_checkpoint(Path(checkpoint), device=device, threshold=threshold))

    def predict(self, frame: np.ndarray) -> HandStateResult:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim not in (2, 3):
            return HandStateResult(HandState.UNKNOWN, DriverActivity.UNKNOWN, None, self.name)
        out = self.classifier.predict(frame, color_order=self.color_order)
        return HandStateResult(
            hand_state=HandState(out["hand_state"]),
            activity=DriverActivity.UNKNOWN,
            confidence=out["confidence"],
            provider=self.name,
            class_probabilities=out["class_probabilities"],
        )

    def reset(self) -> None:
        pass  # stateless per frame; temporal logic lives in the pipeline

