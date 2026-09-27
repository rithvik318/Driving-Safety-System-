"""Eye openness from face landmarks (eye-aspect ratio, EAR).

EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|), measured in pixels.
Roughly 0.25-0.35 for an open eye and below ~0.15 for a closed one, but it
varies by person, camera angle and head pose. It says nothing about drowsiness
on its own; the temporal module decides what a closure means.

"left" and "right" are the DRIVER's own eyes (the driver's left eye appears on
the right side of a non-mirrored image).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.driver.face import FaceLandmarks

# MediaPipe Face Mesh indices, ordered p1..p6 (p1/p4 = corners, p2,p3 top, p6,p5 bottom).
DRIVER_RIGHT_EYE = [33, 160, 158, 133, 153, 144]  # image-left eye
DRIVER_LEFT_EYE = [362, 385, 387, 263, 373, 380]  # image-right eye


@dataclass(frozen=True)
class EyeFeatures:
    left_openness: float | None
    right_openness: float | None

    @property
    def mean_openness(self) -> float | None:
        values = [v for v in (self.left_openness, self.right_openness) if v is not None]
        return round(float(np.mean(values)), 4) if values else None

    def eyes_closed(self, threshold: float) -> bool | None:
        mean = self.mean_openness
        return None if mean is None else mean < threshold


def eye_aspect_ratio(points: np.ndarray) -> float | None:
    """EAR from six (x, y) pixel points in p1..p6 order; None if the eye width is ~0."""
    p1, p2, p3, p4, p5, p6 = points
    width = np.linalg.norm(p1 - p4)
    if width < 1e-6:
        return None
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * width))


def _eye(landmarks: FaceLandmarks, indices: list[int]) -> float | None:
    if landmarks.points.shape[0] <= max(indices) or not landmarks.in_frame(indices):
        return None  # eye outside the frame (partial face)
    ear = eye_aspect_ratio(landmarks.pixel_xy(indices))
    return None if ear is None else round(ear, 4)


def extract_eye_features(landmarks: FaceLandmarks) -> EyeFeatures:
    return EyeFeatures(
        left_openness=_eye(landmarks, DRIVER_LEFT_EYE),
        right_openness=_eye(landmarks, DRIVER_RIGHT_EYE),
    )
