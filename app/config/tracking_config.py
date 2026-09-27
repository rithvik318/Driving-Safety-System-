"""Road-object tracker configuration (IoU tracker). PROTOTYPE DEFAULTS.

Each value can be overridden with the environment variable in brackets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from app.config.settings import PROJECT_ROOT


@dataclass(frozen=True)
class TrackingConfig:
    # A detection joins a track of the SAME class if their boxes overlap at least this much.
    iou_threshold: float = 0.3  # [TRACK_IOU_THRESHOLD]
    # A detection of a DIFFERENT class may still join a track if the boxes overlap at least this
    # much (the same physical object whose YOLO label flickered). The detector's label is kept as
    # observed_class and recorded in class_history; it is never rewritten. > 1 disables it.
    class_switch_iou: float = 0.6  # [TRACK_CLASS_SWITCH_IOU]
    # A track that is not matched is kept (coasting, active=False) until it has been unseen for
    # longer than this, then it ends. Short gaps therefore do not break the track.
    max_missed_seconds: float = 1.0  # [TRACK_MAX_MISSED_SECONDS]
    # Optional extra limit in processed frames (0 = no frame limit, only the time limit).
    max_missed_frames: int = 0  # [TRACK_MAX_MISSED_FRAMES]


def load_tracking_config(env_file: Path | None = None) -> TrackingConfig:
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    kwargs: dict = {}
    for field_name, env, cast in [("iou_threshold", "TRACK_IOU_THRESHOLD", float),
                                  ("class_switch_iou", "TRACK_CLASS_SWITCH_IOU", float),
                                  ("max_missed_seconds", "TRACK_MAX_MISSED_SECONDS", float),
                                  ("max_missed_frames", "TRACK_MAX_MISSED_FRAMES", int)]:
        raw = (os.environ.get(env) or "").strip().strip('"').strip("'")
        if raw:
            try:
                kwargs[field_name] = cast(raw)
            except ValueError as exc:
                raise ValueError(f"{env} has an invalid value {raw!r}") from exc
    config = TrackingConfig(**kwargs)
    validate_tracking_config(config)
    return config


def validate_tracking_config(c: TrackingConfig) -> None:
    if not 0.0 < c.iou_threshold <= 1.0:
        raise ValueError("TRACK_IOU_THRESHOLD must be in (0, 1]")
    if c.class_switch_iou <= 0.0:
        raise ValueError("TRACK_CLASS_SWITCH_IOU must be > 0 (use a value > 1 to disable cross-class matching)")
    if c.class_switch_iou < c.iou_threshold:
        raise ValueError("TRACK_CLASS_SWITCH_IOU must be >= TRACK_IOU_THRESHOLD (cross-class matching must be stricter)")
    if c.max_missed_seconds < 0:
        raise ValueError("TRACK_MAX_MISSED_SECONDS must be >= 0")
    if c.max_missed_frames < 0:
        raise ValueError("TRACK_MAX_MISSED_FRAMES must be >= 0 (0 = no frame limit)")
