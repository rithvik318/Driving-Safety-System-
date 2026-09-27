"""Front-camera road-perception configuration.

PROTOTYPE DEFAULTS. Every value can be overridden with the environment variable named in
brackets, or with the matching option of scripts/test_road_perception.py.

The detector is a PRETRAINED COCO model (Ultralytics YOLO26n by default). It is not trained
or fine-tuned on our data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from app.config.settings import PROJECT_ROOT, VALID_DEVICES, _resolve_path

DEFAULT_ROAD_MODEL = "models/road/yolo26n.pt"  # ~5.5 MB; downloaded once on first use if missing

# COCO class names kept by default, grouped into road categories. Only a "what is visible"
# grouping; it says nothing about danger.
DEFAULT_ROAD_CLASSES: dict[str, str] = {
    "person": "person",
    "bicycle": "two_wheeler",
    "motorcycle": "two_wheeler",
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "train": "vehicle",
    "dog": "animal",
    "cat": "animal",
    "cow": "animal",
    "horse": "animal",
    "sheep": "animal",
    "elephant": "animal",
    "traffic light": "traffic_control",
    "stop sign": "traffic_control",
}


@dataclass(frozen=True)
class RoadConfig:
    model_path: Path = Path(DEFAULT_ROAD_MODEL)  # [ROAD_MODEL_PATH]
    confidence_threshold: float = 0.35  # [ROAD_CONFIDENCE_THRESHOLD] detections below are dropped
    image_size: int = 640  # [ROAD_IMAGE_SIZE] YOLO inference size (long side, multiple of 32)
    sample_fps: float = 5.0  # [ROAD_SAMPLE_FPS] frames per second of video time sent to YOLO; 0 = every frame
    device: str = "auto"  # [ROAD_DEVICE, else DEVICE] auto | cpu | cuda
    classes: tuple[str, ...] = tuple(DEFAULT_ROAD_CLASSES)  # [ROAD_CLASSES] comma-separated COCO names; "all" = every class

    def category(self, class_name: str) -> str:
        return DEFAULT_ROAD_CLASSES.get(class_name, "other")


def load_road_config(env_file: Path | None = None) -> RoadConfig:
    """Build RoadConfig from defaults + environment. Raises ValueError on bad values."""
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

    def raw(name: str) -> str | None:
        value = os.environ.get(name)
        return value.strip().strip('"').strip("'") if value and value.strip() else None

    kwargs: dict = {"model_path": _resolve_path(raw("ROAD_MODEL_PATH") or DEFAULT_ROAD_MODEL, PROJECT_ROOT)}
    try:
        if raw("ROAD_CONFIDENCE_THRESHOLD"):
            kwargs["confidence_threshold"] = float(raw("ROAD_CONFIDENCE_THRESHOLD"))
        if raw("ROAD_IMAGE_SIZE"):
            kwargs["image_size"] = int(raw("ROAD_IMAGE_SIZE"))
        if raw("ROAD_SAMPLE_FPS"):
            kwargs["sample_fps"] = float(raw("ROAD_SAMPLE_FPS"))
    except ValueError as exc:
        raise ValueError(f"invalid road-perception setting: {exc}") from exc
    device = raw("ROAD_DEVICE") or raw("DEVICE")
    if device:
        kwargs["device"] = device.lower()
    if raw("ROAD_CLASSES"):
        kwargs["classes"] = parse_classes(raw("ROAD_CLASSES"))
    config = RoadConfig(**kwargs)
    validate_road_config(config)
    return config


def parse_classes(value: str) -> tuple[str, ...]:
    """'car, person,dog' -> ('car', 'person', 'dog'); 'all' -> ('all',)."""
    names = tuple(s.strip().lower() for s in value.split(",") if s.strip())
    if not names:
        raise ValueError("ROAD_CLASSES is empty")
    return names


def validate_road_config(c: RoadConfig) -> None:
    if not 0.0 <= c.confidence_threshold <= 1.0:
        raise ValueError("ROAD_CONFIDENCE_THRESHOLD must be between 0 and 1")
    if c.image_size < 32 or c.image_size % 32:
        raise ValueError("ROAD_IMAGE_SIZE must be a multiple of 32 and >= 32 (e.g. 640)")
    if c.sample_fps < 0:
        raise ValueError("ROAD_SAMPLE_FPS must be >= 0 (0 = every frame)")
    if c.device not in VALID_DEVICES:
        raise ValueError(f"ROAD_DEVICE/DEVICE must be one of {sorted(VALID_DEVICES)}, got {c.device!r}")
    if not c.classes:
        raise ValueError("at least one road class is required")
