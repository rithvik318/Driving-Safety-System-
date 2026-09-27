"""Face-landmark providers.

Interface:  result = face_provider.process(frame_bgr, timestamp)  ->  FaceResult

A provider returns either valid landmarks or a no-face / invalid-frame result.
It never raises for ordinary bad input. The rest of the pipeline only sees
FaceLandmarks, so MediaPipe can be swapped for another landmark model later.

Landmark topology: MediaPipe Face Mesh (468 points, 478 with irises).
Coordinates are normalized: x in [0, 1] left->right of the image, y in [0, 1] top->bottom.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from app.driver.models import FaceStatus

logger = logging.getLogger(__name__)

MIN_LANDMARKS = 468


@dataclass(frozen=True)
class FaceLandmarks:
    points: np.ndarray  # shape (N, 3), normalized x, y and relative depth z
    image_width: int
    image_height: int

    def pixel_xy(self, indices) -> np.ndarray:
        """Pixel coordinates (float) for the given landmark indices, shape (k, 2)."""
        pts = self.points[list(indices), :2]
        return pts * np.array([self.image_width, self.image_height], dtype=float)

    def in_frame(self, indices) -> bool:
        """True if every listed landmark lies inside the image (partial faces fail this)."""
        pts = self.points[list(indices), :2]
        return bool(np.all((pts >= 0.0) & (pts <= 1.0)))


@dataclass(frozen=True)
class FaceResult:
    status: FaceStatus
    landmarks: FaceLandmarks | None = None
    detail: str | None = None

    @property
    def face_detected(self) -> bool:
        return self.status is FaceStatus.OK and self.landmarks is not None


@runtime_checkable
class FaceLandmarkProvider(Protocol):
    def process(self, frame: np.ndarray, timestamp: float) -> FaceResult: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


def validate_frame(frame) -> tuple[np.ndarray | None, str | None]:
    """Return (BGR uint8 frame, None) or (None, reason). Accepts gray, BGR and BGRA."""
    if frame is None:
        return None, "frame is None"
    if not isinstance(frame, np.ndarray):
        return None, f"frame is {type(frame).__name__}, expected numpy array"
    if frame.size == 0 or frame.ndim not in (2, 3):
        return None, f"frame has unusable shape {frame.shape}"
    if frame.ndim == 3 and frame.shape[2] not in (1, 3, 4):
        return None, f"frame has {frame.shape[2]} channels"
    if min(frame.shape[:2]) < 16:
        return None, f"frame too small {frame.shape[:2]}"
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2 or frame.shape[2] == 1:
        frame = np.repeat(frame.reshape(frame.shape[0], frame.shape[1], 1), 3, axis=2)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]
    return np.ascontiguousarray(frame), None


class NullFaceProvider:
    """Stand-in when no landmark model is available. Always reports PROVIDER_UNAVAILABLE."""

    def process(self, frame: np.ndarray, timestamp: float) -> FaceResult:
        _, reason = validate_frame(frame)
        if reason:
            return FaceResult(FaceStatus.INVALID_FRAME, detail=reason)
        return FaceResult(FaceStatus.PROVIDER_UNAVAILABLE, detail="no face-landmark model loaded")

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


class FaceModelNotFoundError(FileNotFoundError):
    pass


class MediaPipeFaceProvider:
    """MediaPipe Tasks Face Landmarker (CPU), IMAGE mode: each frame is processed independently.

    Needs the `face_landmarker.task` model file (see FACE_MODEL_URL in app/config/driver_config.py).
    """

    def __init__(self, model_path: Path, min_face_confidence: float = 0.5):
        model_path = Path(model_path)
        if not model_path.is_file():
            from app.config.driver_config import FACE_MODEL_URL

            raise FaceModelNotFoundError(
                f"Face Landmarker model not found at {model_path}.\n"
                f"Download it (~4 MB) from:\n  {FACE_MODEL_URL}\n"
                f"and save it there, or set FACE_MODEL_PATH in .env."
            )
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_face_confidence,
            min_face_presence_confidence=min_face_confidence,
            min_tracking_confidence=min_face_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray, timestamp: float) -> FaceResult:
        bgr, reason = validate_frame(frame)
        if reason:
            return FaceResult(FaceStatus.INVALID_FRAME, detail=reason)
        try:
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(image)
        except Exception as exc:  # keep the pipeline alive; report the problem
            logger.warning("Face landmarker failed: %s", exc)
            return FaceResult(FaceStatus.PROVIDER_ERROR, detail=f"{type(exc).__name__}: {exc}")

        if not result.face_landmarks:
            return FaceResult(FaceStatus.NO_FACE)
        pts = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=float)
        if pts.shape[0] < MIN_LANDMARKS:
            return FaceResult(FaceStatus.NO_FACE, detail=f"only {pts.shape[0]} landmarks")
        h, w = bgr.shape[:2]
        return FaceResult(FaceStatus.OK, FaceLandmarks(pts, image_width=w, image_height=h))

    def reset(self) -> None:
        pass  # IMAGE mode keeps no state between frames

    def close(self) -> None:
        self._landmarker.close()
