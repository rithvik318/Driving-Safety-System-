"""Sequential video reading with trustworthy timestamps, shared by the driver and road pipelines.

Moved unchanged from scripts/test_driver_pipeline.py so the road-perception video runner can
use the same reader, timestamp rules and sampling grid. The driver script imports these names
from here, so its behaviour is unchanged.

- VideoReader: thin cv2.VideoCapture wrapper. grab() every frame (cheap, keeps timestamps exact),
  retrieve() only the frames that will be processed.
- FrameClock: container timestamp first, frame_index / fps as fallback, never decreasing.
- FrameSampler: decides which frames to process for a target sample rate (0 = every frame).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

MAX_PLAUSIBLE_FPS = 240.0


class VideoError(Exception):
    """The video cannot be opened, read or timed. The message says what to do."""


@dataclass
class VideoInfo:
    name: str
    width: int
    height: int
    fps: float | None  # None when the container reports no usable FPS
    frame_count: int | None  # container estimate; may be approximate

    @property
    def container_duration(self) -> float | None:
        if self.fps and self.frame_count:
            return self.frame_count / self.fps
        return None


class VideoReader:
    """Thin wrapper over cv2.VideoCapture (tests substitute a fake with the same methods)."""

    def __init__(self, path: Path):
        import cv2

        self._cv2 = cv2
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(path))

    def is_open(self) -> bool:
        return bool(self.cap.isOpened())

    def info(self) -> VideoInfo:
        cv2 = self._cv2
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return VideoInfo(
            name=self.path.name,
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            fps=fps if 0 < fps <= MAX_PLAUSIBLE_FPS and math.isfinite(fps) else None,
            frame_count=frames if frames > 0 else None,
        )

    def grab(self) -> bool:
        return bool(self.cap.grab())

    def retrieve(self):
        return self.cap.retrieve()

    def position_seconds(self) -> float | None:
        """Timestamp of the frame just grabbed, from the container; None if unavailable."""
        ms = self.cap.get(self._cv2.CAP_PROP_POS_MSEC)
        return ms / 1000.0 if ms is not None and math.isfinite(ms) and ms >= 0 else None

    def release(self) -> None:
        self.cap.release()


class FrameClock:
    """Assigns each frame a non-decreasing timestamp in seconds.

    Prefers the container timestamp (correct for variable-frame-rate phone videos);
    falls back to frame_index / fps. Raises VideoError if neither is available.
    """

    def __init__(self, fps: float | None):
        self.fps = fps
        self.last: float | None = None
        self.sources: Counter = Counter()

    def timestamp(self, index: int, container_t: float | None) -> float:
        usable_pts = container_t is not None and (index == 0 or container_t > 0)
        if usable_pts and (self.last is None or container_t >= self.last):
            t, source = container_t, "container"
        elif self.fps:
            t, source = index / self.fps, "index/fps"
            if self.last is not None and t < self.last:
                t, source = self.last + 1.0 / self.fps, "index/fps (adjusted)"
        else:
            raise VideoError(
                "The video reports no usable FPS and no frame timestamps, so elapsed time cannot be "
                "established. Re-run with --assume-fps N (the camera's real frame rate)."
            )
        self.sources[source] += 1
        self.last = t
        return t


class FrameSampler:
    """Sampling grid in video time: process a frame when t reaches the next grid point.

    Same rule as the driver video runner: the grid advances by 1/sample_fps; if the video
    jumps ahead (dropped frames, VFR), it re-anchors on the current frame. sample_fps = 0
    processes every frame.
    """

    def __init__(self, sample_fps: float):
        if sample_fps < 0 or not math.isfinite(sample_fps):
            raise ValueError(f"sample_fps must be >= 0, got {sample_fps!r}")
        self.interval = 1.0 / sample_fps if sample_fps else 0.0
        self.next_sample: float | None = None

    def should_process(self, t: float) -> bool:
        if self.next_sample is not None and t < self.next_sample - 1e-6:
            return False
        self.next_sample = t + self.interval if self.next_sample is None else self.next_sample + self.interval
        if self.next_sample <= t:
            self.next_sample = t + self.interval
        return True
