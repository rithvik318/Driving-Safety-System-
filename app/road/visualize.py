"""Development-only drawing of road detections. Not part of the data model or the pipeline."""

from __future__ import annotations

import numpy as np

from app.road.models import RoadPerceptionResult

# BGR colours per category (fallback grey)
CATEGORY_COLORS = {
    "vehicle": (60, 180, 255),
    "two_wheeler": (255, 160, 60),
    "person": (80, 220, 80),
    "animal": (220, 80, 220),
    "traffic_control": (60, 220, 220),
}


def draw_detections(frame: np.ndarray, result: RoadPerceptionResult, caption: str | None = None) -> np.ndarray:
    """Return an annotated COPY of a BGR frame: box, class name and confidence per detection."""
    import cv2

    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(0.5, min(w, h) / 900.0)
    thick = max(1, int(round(2 * scale)))
    for d in result.detections:
        color = CATEGORY_COLORS.get(d.category, (200, 200, 200))
        p1, p2 = (int(d.bbox_x1), int(d.bbox_y1)), (int(d.bbox_x2), int(d.bbox_y2))
        cv2.rectangle(out, p1, p2, color, thick)
        label = f"{d.class_name} {d.confidence:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, thick)
        y_top = max(0, p1[1] - th - base - 4)
        cv2.rectangle(out, (p1[0], y_top), (p1[0] + tw + 4, y_top + th + base + 4), color, -1)
        cv2.putText(out, label, (p1[0] + 2, y_top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 0), thick, cv2.LINE_AA)
    if caption:
        cv2.putText(out, caption, (10, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (255, 255, 255), thick + 1, cv2.LINE_AA)
    return out
