"""Run a road detector over images and videos.

Video follows the driver-video rules (app/sensors/video.py): every frame is grabbed so
timestamps stay exact, only frames on the sampling grid are decoded and sent to the detector.
Images are loaded with their EXIF orientation applied (read-only).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app.road.models import RoadFrameStatus, RoadPerceptionResult
from app.sensors.video import FrameClock, FrameSampler, VideoError, VideoInfo

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp"}


def read_image_bgr(path: Path) -> np.ndarray:
    """Image file -> BGR uint8 array with EXIF orientation applied. The file is only read."""
    with Image.open(path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])


@dataclass
class RoadRunStats:
    """Aggregate over the processed frames of one image or video."""

    name: str
    kind: str  # "image" | "video"
    info: VideoInfo | None = None
    sample_fps: float | None = None
    frames_read: int = 0
    frames_processed: int = 0
    unreadable_frames: int = 0
    status_counts: Counter = field(default_factory=Counter)
    class_counts: Counter = field(default_factory=Counter)
    frames_with_class: Counter = field(default_factory=Counter)
    confidences: dict = field(default_factory=lambda: defaultdict(list))
    inference_ms: list = field(default_factory=list)
    first_t: float | None = None
    last_t: float | None = None
    timestamp_sources: dict = field(default_factory=dict)
    results: list = field(default_factory=list)  # RoadPerceptionResult, if kept

    @property
    def detections(self) -> int:
        return sum(self.class_counts.values())

    def add(self, result: RoadPerceptionResult, keep: bool) -> None:
        self.frames_processed += 1
        self.status_counts[result.status.value] += 1
        if result.inference_ms is not None and result.status is RoadFrameStatus.OK:
            self.inference_ms.append(result.inference_ms)
        for d in result.detections:
            self.class_counts[d.class_name] += 1
            self.confidences[d.class_name].append(d.confidence)
        for name in {d.class_name for d in result.detections}:
            self.frames_with_class[name] += 1
        if keep:
            self.results.append(result)


def run_image(path: Path, detector, keep_results: bool = True, on_result=None) -> RoadRunStats:
    stats = RoadRunStats(name=Path(path).name, kind="image")
    frame = read_image_bgr(path)
    stats.frames_read = 1
    result = detector.detect(frame, timestamp=None, frame_index=0)
    stats.add(result, keep_results)
    if on_result:
        on_result(result, frame)
    return stats


def run_video(reader, detector, sample_fps: float = 5.0, max_seconds: float | None = None,
              assume_fps: float | None = None, keep_results: bool = True, on_result=None) -> RoadRunStats:
    """Detector over a video. Only frames on the sample_fps grid (video time) are decoded and detected.

    `reader` is an app.sensors.video.VideoReader (or anything with the same methods).
    `on_result(result, frame)` is called for every processed frame (e.g. to print or annotate).
    """
    info = reader.info()
    if assume_fps:
        info.fps = assume_fps
    stats = RoadRunStats(name=info.name, kind="video", info=info, sample_fps=sample_fps)
    clock = FrameClock(info.fps)
    sampler = FrameSampler(sample_fps)
    index = -1
    while reader.grab():
        index += 1
        t = clock.timestamp(index, reader.position_seconds())
        stats.frames_read += 1
        if stats.first_t is None:
            stats.first_t = t
        if max_seconds is not None and t - stats.first_t > max_seconds:
            break
        stats.last_t = t
        if not sampler.should_process(t):
            continue
        ok, frame = reader.retrieve()
        if not ok or frame is None:
            stats.unreadable_frames += 1
            continue
        result = detector.detect(frame, timestamp=t, frame_index=index)
        stats.add(result, keep_results)
        if on_result:
            on_result(result, frame)
    if index < 0:
        raise VideoError("No frames could be read from the video (empty, corrupt or unsupported codec).")
    stats.timestamp_sources = dict(clock.sources)
    return stats


def list_media(path: Path) -> list[Path]:
    """A file, or every image/video under a folder (sorted, recursive)."""
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES)
