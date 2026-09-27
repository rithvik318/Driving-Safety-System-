"""Tracked road objects: frame detections linked over time, with image-space motion features.

ALL motion quantities are IMAGE-SPACE (pixels, pixels/second, pixels²/second) in the frames the
detector saw. They are not metres, km/h, physical speed, distance or time-to-collision: there is
no camera calibration. Growing box area is only a relative "getting bigger in the image" signal.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class TrackedObject:
    """Snapshot of one track after a tracker update."""

    track_id: int
    class_name: str  # track-level label: most frequent observed class so far (ties -> earliest seen)
    observed_class: str  # class of the most recent matched detection, exactly as the detector reported it
    class_confidence: float  # detector confidence of the most recent matched detection
    mean_confidence: float
    class_history: tuple[str, ...]  # observed class per matched detection, in order
    class_changes: int  # times observed_class differed from the previous matched detection's class

    first_seen_timestamp: float
    last_seen_timestamp: float
    timestamp: float  # time of the update that produced this snapshot
    age_seconds: float  # timestamp - first_seen_timestamp
    missed_seconds: float  # timestamp - last_seen_timestamp (0 when matched in this update)
    missed_frames: int  # consecutive processed frames without a match
    persistence_seconds: float  # last_seen - first_seen: span over which the object has been observed
    detection_count: int  # matched detections supporting this track
    active: bool  # matched in this update (False = coasting through a short gap, or ended)
    ended: bool = False  # track has expired and will not be updated again

    # geometry of the most recent matched detection (pixels)
    bbox_x1: float = 0.0
    bbox_y1: float = 0.0
    bbox_x2: float = 0.0
    bbox_y2: float = 0.0
    center_x: float = 0.0
    center_y: float = 0.0
    bbox_width: float = 0.0
    bbox_height: float = 0.0
    area: float = 0.0

    # change since the previous matched detection (None until the track has two observations)
    previous_center_x: float | None = None
    previous_center_y: float | None = None
    previous_area: float | None = None
    velocity_x_pixels_per_second: float | None = None
    velocity_y_pixels_per_second: float | None = None
    speed_pixels_per_second: float | None = None
    area_change_rate: float | None = None  # pixels² per second
    relative_area_change_rate: float | None = None  # (area change / previous area) per second; > 0 = growing
    frame_index: int | None = None  # frame of the most recent matched detection

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2)

    @property
    def class_counts(self) -> dict[str, int]:
        return dict(Counter(self.class_history))

    def to_dict(self) -> dict:
        def r(v, n=2):
            return None if v is None else round(v, n)

        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "observed_class": self.observed_class,
            "class_confidence": r(self.class_confidence, 4),
            "mean_confidence": r(self.mean_confidence, 4),
            "class_history": list(self.class_history),
            "class_changes": self.class_changes,
            "first_seen_timestamp": r(self.first_seen_timestamp, 3),
            "last_seen_timestamp": r(self.last_seen_timestamp, 3),
            "timestamp": r(self.timestamp, 3),
            "age_seconds": r(self.age_seconds, 3),
            "missed_seconds": r(self.missed_seconds, 3),
            "missed_frames": self.missed_frames,
            "persistence_seconds": r(self.persistence_seconds, 3),
            "detection_count": self.detection_count,
            "active": self.active,
            "ended": self.ended,
            "bbox": [r(v) for v in self.bbox],
            "center_x": r(self.center_x),
            "center_y": r(self.center_y),
            "bbox_width": r(self.bbox_width),
            "bbox_height": r(self.bbox_height),
            "area": r(self.area, 1),
            "previous_center_x": r(self.previous_center_x),
            "previous_center_y": r(self.previous_center_y),
            "previous_area": r(self.previous_area, 1),
            "velocity_x_pixels_per_second": r(self.velocity_x_pixels_per_second, 1),
            "velocity_y_pixels_per_second": r(self.velocity_y_pixels_per_second, 1),
            "speed_pixels_per_second": r(self.speed_pixels_per_second, 1),
            "area_change_rate": r(self.area_change_rate, 1),
            "relative_area_change_rate": r(self.relative_area_change_rate, 4),
            "frame_index": self.frame_index,
        }
