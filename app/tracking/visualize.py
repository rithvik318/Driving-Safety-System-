"""Development-only drawing of tracked objects. Not part of the tracker or its data model."""

from __future__ import annotations

import numpy as np

from app.tracking.models import TrackedObject

_PALETTE = [(60, 180, 255), (80, 220, 80), (255, 160, 60), (220, 80, 220), (60, 220, 220),
            (255, 90, 90), (170, 120, 255), (120, 255, 180), (0, 200, 255), (200, 200, 60)]


def track_color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[(track_id - 1) % len(_PALETTE)]


def track_label(t: TrackedObject) -> str:
    """'ID 7 | car | 0.82 | 2.4s' - observed class of the latest detection; '*' marks a label change."""
    flag = "*" if t.class_changes else ""
    return f"ID {t.track_id} | {t.observed_class}{flag} | {t.class_confidence:.2f} | {t.persistence_seconds:.1f}s"


def draw_tracks(frame: np.ndarray, tracks: list[TrackedObject], caption: str | None = None,
                show_coasting: bool = True) -> np.ndarray:
    """Annotated COPY of a BGR frame. Coasting tracks (not matched this frame) are drawn thin/dashed."""
    import cv2

    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(0.5, min(w, h) / 900.0)
    thick = max(1, int(round(2 * scale)))
    for t in tracks:
        if not t.active and not show_coasting:
            continue
        color = track_color(t.track_id)
        p1, p2 = (int(t.bbox_x1), int(t.bbox_y1)), (int(t.bbox_x2), int(t.bbox_y2))
        if t.active:
            cv2.rectangle(out, p1, p2, color, thick)
        else:  # coasting: corners only
            L = max(6, int(min(p2[0] - p1[0], p2[1] - p1[1]) * 0.2))
            for (x, y), (dx, dy) in [(p1, (1, 1)), ((p2[0], p1[1]), (-1, 1)), ((p1[0], p2[1]), (1, -1)), (p2, (-1, -1))]:
                cv2.line(out, (x, y), (x + dx * L, y), color, max(1, thick - 1))
                cv2.line(out, (x, y), (x, y + dy * L), color, max(1, thick - 1))
        label = track_label(t) + ("" if t.active else " (missed)")
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, thick)
        y_top = max(0, p1[1] - th - base - 4)
        cv2.rectangle(out, (p1[0], y_top), (p1[0] + tw + 4, y_top + th + base + 4), color, -1)
        cv2.putText(out, label, (p1[0] + 2, y_top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (0, 0, 0), thick,
                    cv2.LINE_AA)
        if t.active and t.velocity_x_pixels_per_second is not None:  # image-space motion over the last 0.5 s
            cx, cy = int(t.center_x), int(t.center_y)
            tip = (int(cx + 0.5 * t.velocity_x_pixels_per_second), int(cy + 0.5 * t.velocity_y_pixels_per_second))
            cv2.arrowedLine(out, (cx, cy), tip, color, thick, tipLength=0.25)
    if caption:
        cv2.putText(out, caption, (10, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (255, 255, 255), thick + 1,
                    cv2.LINE_AA)
    return out
