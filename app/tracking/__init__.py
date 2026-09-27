"""Road-object tracking: frame detections -> persistent tracks with image-space motion features.

Image-space only (pixels, pixels/second). No calibration, physical speed, distance or TTC.
"""

from app.tracking.models import TrackedObject
from app.tracking.tracker import IoUTracker, iou

__all__ = ["IoUTracker", "TrackedObject", "iou"]
