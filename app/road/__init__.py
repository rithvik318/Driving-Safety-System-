"""Front-camera road perception: pretrained YOLO detections as structured observations.

Detection only ("what is visible"); no danger/risk interpretation here.
"""

from app.road.detector import RoadDetector, RoadModelError, YoloRoadDetector, build_detections, resolve_device
from app.road.models import Detection, RoadFrameStatus, RoadPerceptionResult

__all__ = [
    "Detection",
    "RoadDetector",
    "RoadFrameStatus",
    "RoadModelError",
    "RoadPerceptionResult",
    "YoloRoadDetector",
    "build_detections",
    "resolve_device",
]
