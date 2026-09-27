"""Simple, transparent IoU tracker for frame-level road detections.

    tracker = IoUTracker(TrackingConfig())
    tracks = tracker.update(result.detections, timestamp, frame_index)   # -> list[TrackedObject]

Per update:
  1. Tracks unseen for longer than max_missed_seconds end BEFORE matching, so nothing is
     matched across a long gap (e.g. a dropped stretch of video).
  2. Greedy association, highest IoU first (ties: lower track_id, then detection order):
       pass 1: same class (detection class == the track's latest observed class or its
               track-level class) and IoU >= iou_threshold;
       pass 2: remaining pairs of DIFFERENT class only if IoU >= class_switch_iou (the same box
               relabelled by YOLO). The new label is kept as observed_class and counted in
               class_changes; nothing is rewritten.
  3. Unmatched detections start new tracks (IDs 1, 2, 3, ... in detection order).
  4. Unmatched tracks coast (active=False) and end after max_missed_seconds or, if set,
     max_missed_frames.

Motion features compare the current matched detection with the PREVIOUS MATCHED detection of the
same track and divide by the real time between them, so irregular sampling and short gaps give
correct image-space rates. All values are pixels / pixels per second: no calibration, no
physical speed, no distance, no TTC.

Deterministic: same inputs -> same IDs and values. Timestamps must not go backwards; if they do
(new video without reset()), all tracks end and timeline_resets is incremented.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from app.config.tracking_config import TrackingConfig, validate_tracking_config
from app.tracking.models import TrackedObject

EPS = 1e-9


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes; 0 if they do not overlap."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _bbox(d) -> tuple[float, float, float, float]:
    return (float(d.bbox_x1), float(d.bbox_y1), float(d.bbox_x2), float(d.bbox_y2))


@dataclass
class _Track:
    track_id: int
    first_seen: float
    last_seen: float
    bbox: tuple[float, float, float, float]
    history: list = field(default_factory=list)  # observed class per matched detection
    confidences: list = field(default_factory=list)
    class_changes: int = 0
    missed_frames: int = 0
    frame_index: int | None = None
    prev_center: tuple[float, float] | None = None
    prev_area: float | None = None
    vx: float | None = None
    vy: float | None = None
    area_rate: float | None = None
    rel_area_rate: float | None = None
    ended: bool = False

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    @property
    def observed_class(self) -> str:
        return self.history[-1]

    @property
    def track_class(self) -> str:
        counts = Counter(self.history)
        first_index = {c: self.history.index(c) for c in counts}
        return max(counts, key=lambda c: (counts[c], -first_index[c]))

    def observe(self, det, t: float, frame_index: int | None) -> None:
        old_center, old_area, dt = self.center, self.area, t - self.last_seen
        if self.history and det.class_name != self.history[-1]:
            self.class_changes += 1
        self.bbox = _bbox(det)
        self.history.append(det.class_name)
        self.confidences.append(float(det.confidence))
        if dt > EPS:  # same-timestamp duplicates carry no motion information
            cx, cy = self.center
            self.prev_center, self.prev_area = old_center, old_area
            self.vx, self.vy = (cx - old_center[0]) / dt, (cy - old_center[1]) / dt
            self.area_rate = (self.area - old_area) / dt
            self.rel_area_rate = self.area_rate / old_area if old_area > 0 else None
        self.last_seen = t
        self.missed_frames = 0
        self.frame_index = frame_index

    def snapshot(self, t: float, active: bool) -> TrackedObject:
        cx, cy = self.center
        speed = None if self.vx is None else math.hypot(self.vx, self.vy)
        return TrackedObject(
            track_id=self.track_id, class_name=self.track_class, observed_class=self.observed_class,
            class_confidence=self.confidences[-1], mean_confidence=sum(self.confidences) / len(self.confidences),
            class_history=tuple(self.history), class_changes=self.class_changes,
            first_seen_timestamp=self.first_seen, last_seen_timestamp=self.last_seen, timestamp=t,
            age_seconds=t - self.first_seen, missed_seconds=t - self.last_seen, missed_frames=self.missed_frames,
            persistence_seconds=self.last_seen - self.first_seen, detection_count=len(self.history),
            active=active, ended=self.ended,
            bbox_x1=self.bbox[0], bbox_y1=self.bbox[1], bbox_x2=self.bbox[2], bbox_y2=self.bbox[3],
            center_x=cx, center_y=cy, bbox_width=self.bbox[2] - self.bbox[0], bbox_height=self.bbox[3] - self.bbox[1],
            area=self.area,
            previous_center_x=None if self.prev_center is None else self.prev_center[0],
            previous_center_y=None if self.prev_center is None else self.prev_center[1],
            previous_area=self.prev_area,
            velocity_x_pixels_per_second=self.vx, velocity_y_pixels_per_second=self.vy,
            speed_pixels_per_second=speed, area_change_rate=self.area_rate,
            relative_area_change_rate=self.rel_area_rate, frame_index=self.frame_index,
        )


class IoUTracker:
    """Class-aware greedy IoU tracker. Lightweight: O(tracks x detections) per frame, no model."""

    def __init__(self, config: TrackingConfig | None = None):
        self.config = config or TrackingConfig()
        validate_tracking_config(self.config)
        self.reset()

    def reset(self) -> None:
        """Forget everything; the next track gets ID 1. Call between videos."""
        self._tracks: dict[int, _Track] = {}
        self._finished: list[TrackedObject] = []
        self._next_id = 1
        self._last_t: float | None = None
        self.timeline_resets = 0
        self.ended_last_update: list[TrackedObject] = []
        self.created_last_update: list[int] = []

    # ------------------------------------------------------------------ helpers

    def _end(self, track: _Track, t: float) -> None:
        track.ended = True
        snap = track.snapshot(t, active=False)
        self._finished.append(snap)
        self.ended_last_update.append(snap)
        del self._tracks[track.track_id]

    def _match(self, detections: list, class_aware: bool, unmatched_tracks: set[int],
               unmatched_dets: set[int]) -> list[tuple[int, int]]:
        threshold = self.config.iou_threshold if class_aware else self.config.class_switch_iou
        if threshold > 1.0:
            return []
        pairs = []
        for tid in sorted(unmatched_tracks):
            tr = self._tracks[tid]
            for j in sorted(unmatched_dets):
                d = detections[j]
                same = d.class_name in (tr.observed_class, tr.track_class)
                if same != class_aware:
                    continue
                overlap = iou(tr.bbox, _bbox(d))
                if overlap >= threshold - EPS:
                    pairs.append((-overlap, tid, j))
        matches = []
        for _neg, tid, j in sorted(pairs):
            if tid in unmatched_tracks and j in unmatched_dets:
                matches.append((tid, j))
                unmatched_tracks.discard(tid)
                unmatched_dets.discard(j)
        return matches

    # ------------------------------------------------------------------ API

    def update(self, detections, timestamp: float, frame_index: int | None = None) -> list[TrackedObject]:
        """Associate this frame's detections with tracks. Returns current tracks (matched and
        coasting), sorted by track_id. Tracks that ended in this update are in ended_last_update."""
        if timestamp is None or isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) \
                or not math.isfinite(timestamp):
            raise ValueError(f"timestamp must be a finite number of seconds, got {timestamp!r}")
        t = float(timestamp)
        detections = list(detections)
        self.ended_last_update, self.created_last_update = [], []

        if self._last_t is not None and t < self._last_t - EPS:  # time went backwards: new timeline
            for tr in sorted(self._tracks.values(), key=lambda x: x.track_id):
                self._end(tr, self._last_t)
            self.timeline_resets += 1
        self._last_t = t

        # 1. tracks unseen for too long end before matching
        for tr in sorted(self._tracks.values(), key=lambda x: x.track_id):
            if t - tr.last_seen > self.config.max_missed_seconds + EPS:
                self._end(tr, t)

        # 2. association
        unmatched_tracks, unmatched_dets = set(self._tracks), set(range(len(detections)))
        matches = self._match(detections, True, unmatched_tracks, unmatched_dets)
        matches += self._match(detections, False, unmatched_tracks, unmatched_dets)
        matched_ids = set()
        for tid, j in matches:
            self._tracks[tid].observe(detections[j], t, frame_index)
            matched_ids.add(tid)

        # 3. new tracks, in detection order
        for j in sorted(unmatched_dets):
            d = detections[j]
            tr = _Track(self._next_id, t, t, _bbox(d), frame_index=frame_index)
            tr.history.append(d.class_name)
            tr.confidences.append(float(d.confidence))
            self._tracks[tr.track_id] = tr
            self.created_last_update.append(tr.track_id)
            matched_ids.add(tr.track_id)
            self._next_id += 1

        # 4. unmatched tracks coast, or end on the frame limit
        for tid in sorted(unmatched_tracks):
            tr = self._tracks[tid]
            tr.missed_frames += 1
            if self.config.max_missed_frames and tr.missed_frames > self.config.max_missed_frames:
                self._end(tr, t)

        return [tr.snapshot(t, active=tr.track_id in matched_ids) for tr in sorted(self._tracks.values(), key=lambda x: x.track_id)]

    def finish(self) -> list[TrackedObject]:
        """End all remaining tracks (e.g. at the end of a video) and return their final snapshots."""
        self.ended_last_update = []
        if self._last_t is not None:
            for tr in sorted(self._tracks.values(), key=lambda x: x.track_id):
                self._end(tr, self._last_t)
        return list(self.ended_last_update)

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    @property
    def tracks_created(self) -> int:
        return self._next_id - 1

    def all_tracks(self) -> list[TrackedObject]:
        """Final snapshot of every ended track plus the current snapshot of live ones, by track_id."""
        live = [tr.snapshot(self._last_t, active=tr.missed_frames == 0) for tr in self._tracks.values()] \
            if self._last_t is not None else []
        return sorted(self._finished + live, key=lambda s: s.track_id)
