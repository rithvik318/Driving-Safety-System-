"""Front-camera road-perception data models.

Detection answers only "what is visible, where, how confidently". It carries no danger,
distance, motion or risk information; those belong to later stages (tracking, risk).

Coordinates are pixels in the frame that was given to the detector (after any rotation the
video decoder applied), origin top-left, x to the right, y down. Timestamps are float seconds
on one clock per session (video time for files), exactly like the driver pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class RoadFrameStatus(str, Enum):
    OK = "OK"  # detector ran; zero detections is still OK
    INVALID_FRAME = "INVALID_FRAME"  # frame missing / wrong shape; detector not called
    PROVIDER_ERROR = "PROVIDER_ERROR"  # detector raised; no detections reported


def _finite(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class Detection:
    """One object seen in one frame by a pretrained detector."""

    timestamp: float | None
    class_name: str
    class_id: int
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    source: str  # detector that produced it, e.g. "yolo26n"
    category: str = "other"  # coarse grouping (vehicle, person, two_wheeler, animal, ...); not a risk level
    frame_index: int | None = None

    def __post_init__(self):
        if self.timestamp is not None:
            _finite("timestamp", self.timestamp)
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("class_name must be a non-empty string")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int) or self.class_id < 0:
            raise ValueError(f"class_id must be an int >= 0, got {self.class_id!r}")
        if not 0.0 <= _finite("confidence", self.confidence) <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence!r}")
        x1, y1 = _finite("bbox_x1", self.bbox_x1), _finite("bbox_y1", self.bbox_y1)
        x2, y2 = _finite("bbox_x2", self.bbox_x2), _finite("bbox_y2", self.bbox_y2)
        if x2 < x1 or y2 < y1:
            raise ValueError(f"bbox must have x2 >= x1 and y2 >= y1, got ({x1}, {y1}, {x2}, {y2})")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty string")

    @property
    def bbox_width(self) -> float:
        return self.bbox_x2 - self.bbox_x1

    @property
    def bbox_height(self) -> float:
        return self.bbox_y2 - self.bbox_y1

    @property
    def bbox_center_x(self) -> float:
        return (self.bbox_x1 + self.bbox_x2) / 2.0

    @property
    def bbox_center_y(self) -> float:
        return (self.bbox_y1 + self.bbox_y2) / 2.0

    @property
    def bbox_area(self) -> float:
        return self.bbox_width * self.bbox_height

    def to_dict(self) -> dict:
        """Flat, JSON-friendly record with the stable field names used downstream."""
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "bbox_x1": round(self.bbox_x1, 2),
            "bbox_y1": round(self.bbox_y1, 2),
            "bbox_x2": round(self.bbox_x2, 2),
            "bbox_y2": round(self.bbox_y2, 2),
            "bbox_center_x": round(self.bbox_center_x, 2),
            "bbox_center_y": round(self.bbox_center_y, 2),
            "bbox_width": round(self.bbox_width, 2),
            "bbox_height": round(self.bbox_height, 2),
            "source": self.source,
        }


@dataclass(frozen=True)
class RoadPerceptionResult:
    """Everything the detector saw in one frame. No interpretation."""

    timestamp: float | None
    frame_index: int | None
    image_width: int | None
    image_height: int | None
    detections: tuple[Detection, ...] = field(default_factory=tuple)
    status: RoadFrameStatus = RoadFrameStatus.OK
    source: str = "unknown"  # detector name
    inference_ms: float | None = None  # time inside the detector call (pre-process + model + post-process)
    error: str | None = None

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "status": self.status.value,
            "source": self.source,
            "inference_ms": None if self.inference_ms is None else round(self.inference_ms, 2),
            "error": self.error,
            "detections": [d.to_dict() for d in self.detections],
        }
