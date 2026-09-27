"""Pretrained YOLO road-object detector (Ultralytics). Detection only, no interpretation.

- The model is loaded ONCE in YoloRoadDetector.__init__ and reused for every frame. A warm-up call
  on a blank image (also in __init__) absorbs the one-time lazy imports inside torch/ultralytics
  (seconds on a cold start), so per-frame timings reflect steady-state inference.
- CPU works everywhere; DEVICE=auto picks CUDA only if torch reports it.
- Classes are filtered to road-relevant COCO names (app/config/road_config.py) inside the
  model call, and detections below the confidence threshold are dropped.
- Ultralytics runtime auto-install is disabled (YOLO_AUTOINSTALL=false) so it never tries to
  pip-install opencv-python next to opencv-contrib-python. Its usage analytics are switched off
  for this process only (the global Ultralytics settings file is not changed).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from app.config.road_config import DEFAULT_ROAD_CLASSES, RoadConfig
from app.road.models import Detection, RoadFrameStatus, RoadPerceptionResult

log = logging.getLogger(__name__)

INSTALL_HINT = (
    "Ultralytics is not installed. Install it WITHOUT its opencv-python dependency (it would clash "
    "with opencv-contrib-python, which MediaPipe needs):\n"
    "    pip install -r requirements.txt\n"
    "    pip install --no-deps -r requirements-nodeps.txt"
)


class RoadModelError(Exception):
    """The road detector cannot be created (missing package or weights, bad class names, bad device)."""


class RoadDetector(Protocol):
    name: str

    def detect(self, frame, timestamp: float | None = None, frame_index: int | None = None) -> RoadPerceptionResult: ...


# --------------------------------------------------------------------------- helpers


def resolve_device(name: str = "auto") -> str:
    """'auto' -> 'cuda' if torch sees a GPU else 'cpu'. 'cuda' without a GPU is an error."""
    name = (name or "auto").lower()
    if name == "cpu":
        return "cpu"
    try:
        import torch

        has_cuda = bool(torch.cuda.is_available())
    except ImportError:
        has_cuda = False
    if name == "auto":
        return "cuda" if has_cuda else "cpu"
    if name.startswith("cuda"):
        if not has_cuda:
            raise RoadModelError("DEVICE=cuda was requested but no CUDA device is available. Use auto or cpu.")
        return name
    raise RoadModelError(f"unknown device {name!r}; use auto, cpu or cuda")


def _import_ultralytics():
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # read by ultralytics at import time
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RoadModelError(INSTALL_HINT) from exc
    try:  # per-process analytics opt-out; the attribute may move between versions
        from ultralytics.utils import events as _events

        _events.events.enabled = False
    except Exception:  # pragma: no cover - best effort only
        pass
    return YOLO


def load_yolo_model(model_path: Path, allow_download: bool = True):
    """Load Ultralytics weights from model_path. If the file is missing and its name is an official
    Ultralytics asset (e.g. yolo26n.pt), download it once into that path."""
    YOLO = _import_ultralytics()
    model_path = Path(model_path)
    if not model_path.is_file():
        from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES, attempt_download_asset

        if not (allow_download and model_path.name in GITHUB_ASSETS_NAMES):
            raise RoadModelError(
                f"Road model not found: {model_path}. Use an official Ultralytics weight name "
                "(e.g. models/road/yolo26n.pt, downloaded automatically) or point ROAD_MODEL_PATH at a .pt file."
            )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading pretrained %s (one time, a few MB) to %s", model_path.name, model_path)
        try:
            attempt_download_asset(model_path)
        except Exception as exc:
            raise RoadModelError(f"Could not download {model_path.name}: {exc}") from exc
        if not model_path.is_file():
            raise RoadModelError(
                f"Download of {model_path.name} did not produce {model_path}. Check the internet connection "
                "or download it manually from https://github.com/ultralytics/assets/releases."
            )
    try:
        return YOLO(str(model_path))
    except Exception as exc:
        raise RoadModelError(f"Could not load road model {model_path.name}: {exc}") from exc


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def build_detections(xyxy, confidences, class_ids, names: dict[int, str], threshold: float,
                     allowed_ids: set[int] | None, image_size: tuple[int, int], source: str,
                     timestamp: float | None, frame_index: int | None,
                     category_of=lambda name: DEFAULT_ROAD_CLASSES.get(name, "other")) -> tuple[Detection, ...]:
    """Raw detector arrays -> Detection objects. Drops boxes below `threshold`, classes outside
    `allowed_ids` (None = keep all) and degenerate boxes; clips boxes to the image. Sorted by
    confidence, highest first."""
    w, h = image_size
    xyxy = _as_numpy(xyxy).reshape(-1, 4) if len(xyxy) else np.zeros((0, 4))
    confidences = _as_numpy(confidences).reshape(-1)
    class_ids = _as_numpy(class_ids).reshape(-1)
    out = []
    for (x1, y1, x2, y2), conf, cid in zip(xyxy, confidences, class_ids):
        conf, cid = float(conf), int(cid)
        if not np.isfinite(conf) or conf < threshold - 1e-6:  # tolerance: YOLO scores are float32
            continue
        if allowed_ids is not None and cid not in allowed_ids:
            continue
        x1, x2 = sorted((min(max(float(x1), 0.0), w), min(max(float(x2), 0.0), w)))
        y1, y2 = sorted((min(max(float(y1), 0.0), h), min(max(float(y2), 0.0), h)))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        name = names.get(cid, str(cid))
        out.append(Detection(
            timestamp=timestamp, class_name=name, class_id=cid, confidence=min(conf, 1.0),
            bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2, source=source,
            category=category_of(name), frame_index=frame_index,
        ))
    out.sort(key=lambda d: -d.confidence)
    return tuple(out)


def _frame_problem(frame) -> str | None:
    if not isinstance(frame, np.ndarray):
        return f"frame must be a numpy array, got {type(frame).__name__}"
    if frame.ndim != 3 or frame.shape[2] != 3:
        return f"frame must be HxWx3 (BGR), got shape {frame.shape}"
    if frame.shape[0] < 8 or frame.shape[1] < 8:
        return f"frame too small: {frame.shape[1]}x{frame.shape[0]}"
    if frame.dtype != np.uint8:
        return f"frame must be uint8, got {frame.dtype}"
    return None


# --------------------------------------------------------------------------- detector


class YoloRoadDetector:
    """Pretrained Ultralytics YOLO detector for front-camera frames (BGR numpy arrays)."""

    def __init__(self, model_path: Path | str = Path("models/road/yolo26n.pt"), confidence_threshold: float = 0.35,
                 image_size: int = 640, device: str = "auto", classes: tuple[str, ...] = tuple(DEFAULT_ROAD_CLASSES),
                 allow_download: bool = True, model=None, warmup: bool = True):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise RoadModelError("confidence_threshold must be between 0 and 1")
        if image_size < 32 or image_size % 32:
            raise RoadModelError("image_size must be a multiple of 32 (e.g. 640)")
        self.model_path = Path(model_path)
        self.name = self.model_path.stem  # e.g. "yolo26n"
        self.confidence_threshold = float(confidence_threshold)
        self.image_size = int(image_size)
        self.device = resolve_device(device)
        started = time.perf_counter()
        self.model = model if model is not None else load_yolo_model(self.model_path, allow_download)
        self.load_seconds = time.perf_counter() - started
        self.names: dict[int, str] = {int(k): str(v) for k, v in dict(self.model.names).items()}
        self.class_ids = self._resolve_classes(classes)
        self.warmup_seconds = 0.0
        if warmup:
            started = time.perf_counter()
            self.model.predict(np.zeros((self.image_size, self.image_size, 3), np.uint8), conf=self.confidence_threshold,
                               imgsz=self.image_size, device=self.device, verbose=False)
            self.warmup_seconds = time.perf_counter() - started
        self.calls = 0  # detector calls on real frames (warm-up excluded)
        log.info("Road detector %s loaded in %.2f s + %.2f s warm-up (device %s, conf %.2f, imgsz %d, %s classes)",
                 self.name, self.load_seconds, self.warmup_seconds, self.device, self.confidence_threshold,
                 self.image_size, "all" if self.class_ids is None else len(self.class_ids))

    @classmethod
    def from_config(cls, config: RoadConfig, **overrides) -> "YoloRoadDetector":
        kwargs = dict(model_path=config.model_path, confidence_threshold=config.confidence_threshold,
                      image_size=config.image_size, device=config.device, classes=config.classes)
        kwargs.update(overrides)
        return cls(**kwargs)

    def _resolve_classes(self, classes) -> set[int] | None:
        wanted = [c.strip().lower() for c in classes]
        if "all" in wanted:
            return None
        by_name = {v.lower(): k for k, v in self.names.items()}
        unknown = [c for c in wanted if c not in by_name]
        if unknown:
            raise RoadModelError(f"class name(s) not in the model: {', '.join(unknown)}. "
                                 f"Available: {', '.join(sorted(by_name))}")
        return {by_name[c] for c in wanted}

    @property
    def class_names(self) -> list[str]:
        ids = sorted(self.names) if self.class_ids is None else sorted(self.class_ids)
        return [self.names[i] for i in ids]

    def detect(self, frame, timestamp: float | None = None, frame_index: int | None = None) -> RoadPerceptionResult:
        problem = _frame_problem(frame)
        if problem:
            return RoadPerceptionResult(timestamp, frame_index, None, None, status=RoadFrameStatus.INVALID_FRAME,
                                        source=self.name, error=problem)
        h, w = frame.shape[:2]
        started = time.perf_counter()
        try:
            self.calls += 1
            results = self.model.predict(
                frame, conf=self.confidence_threshold, imgsz=self.image_size, device=self.device,
                classes=None if self.class_ids is None else sorted(self.class_ids), verbose=False,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            boxes = results[0].boxes if results else None
            if boxes is None or len(boxes) == 0:
                detections: tuple[Detection, ...] = ()
            else:
                detections = build_detections(boxes.xyxy, boxes.conf, boxes.cls, self.names, self.confidence_threshold,
                                              self.class_ids, (w, h), self.name, timestamp, frame_index)
        except Exception as exc:
            log.warning("Road detector failed on frame %s: %s", frame_index, exc)
            return RoadPerceptionResult(timestamp, frame_index, w, h, status=RoadFrameStatus.PROVIDER_ERROR,
                                        source=self.name, inference_ms=(time.perf_counter() - started) * 1000.0,
                                        error=f"{type(exc).__name__}: {exc}")
        return RoadPerceptionResult(timestamp, frame_index, w, h, detections, RoadFrameStatus.OK, self.name, elapsed_ms)

    def close(self) -> None:
        self.model = None
