"""Temporal eye-closure tracking over a rolling time window.

Input per frame: (timestamp, eyes_closed) where eyes_closed is True, False, or
None (could not be measured, e.g. no face). Time between two frames is credited
to the earlier frame's state, but only if the gap is at most MAX_GAP_SECONDS;
longer gaps count as unobserved time.

Missing observations are never treated as closed eyes. A closure run is broken
(and not counted as an event) if the face is lost for longer than MAX_GAP_SECONDS.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ClosureEvent:
    end_time: float
    duration: float
    kind: str  # "blink" | "long" | "intermediate"


@dataclass(frozen=True)
class EyeClosureFeatures:
    eye_closure_ratio: float | None
    eye_closed_duration: float | None
    blink_count: int
    long_closure_count: int
    observed_seconds: float
    face_missing_duration: float


class EyeClosureTracker:
    def __init__(self, window_seconds: float, max_gap_seconds: float, blink_max_seconds: float, long_closure_seconds: float):
        self.window = window_seconds
        self.max_gap = max_gap_seconds
        self.blink_max = blink_max_seconds
        self.long_min = long_closure_seconds
        self.reset()

    def reset(self) -> None:
        self._intervals: deque[tuple[float, float, bool]] = deque()  # (start, end, closed)
        self._events: deque[ClosureEvent] = deque()
        self._prev: tuple[float, bool | None] | None = None
        self._last_valid_t: float | None = None
        self._first_t: float | None = None
        self._run_start: float | None = None
        # Running totals since the last reset (not windowed). Read-only for reports;
        # they do not feed the drowsiness score.
        self.total_blinks = 0
        self.total_long_closures = 0
        self.longest_closure_seconds = 0.0

    def _end_run(self, t: float) -> None:
        if self._run_start is None:
            return
        duration = t - self._run_start
        if duration < self.blink_max:
            kind = "blink"
        elif duration >= self.long_min:
            kind = "long"
        else:
            kind = "intermediate"
        self._events.append(ClosureEvent(t, duration, kind))
        self.total_blinks += kind == "blink"
        self.total_long_closures += kind == "long"
        self.longest_closure_seconds = max(self.longest_closure_seconds, duration)
        self._run_start = None

    def update(self, t: float, eyes_closed: bool | None) -> EyeClosureFeatures:
        if self._first_t is None:
            self._first_t = t

        # Credit the elapsed time to the previous frame's state (if observed and gap is small).
        if self._prev is not None:
            prev_t, prev_closed = self._prev
            if prev_closed is not None and 0 < t - prev_t <= self.max_gap:
                self._intervals.append((prev_t, t, prev_closed))

        # A long loss of observation breaks any closure run without counting it.
        if self._last_valid_t is not None and t - self._last_valid_t > self.max_gap:
            self._run_start = None

        if eyes_closed is True:
            if self._run_start is None:
                self._run_start = t
        elif eyes_closed is False:
            self._end_run(t)

        if eyes_closed is not None:
            self._last_valid_t = t
        self._prev = (t, eyes_closed)
        return self._features(t, eyes_closed)

    def _features(self, t: float, eyes_closed: bool | None) -> EyeClosureFeatures:
        start = t - self.window
        while self._intervals and self._intervals[0][1] <= start:
            self._intervals.popleft()
        while self._events and self._events[0].end_time < start:
            self._events.popleft()

        observed = closed = 0.0
        for a, b, is_closed in self._intervals:
            span = b - max(a, start)
            observed += span
            if is_closed:
                closed += span

        if eyes_closed is True and self._run_start is not None:
            closed_duration: float | None = t - self._run_start
        elif eyes_closed is False:
            closed_duration = 0.0
        else:
            closed_duration = None  # not measurable this frame

        if eyes_closed is None:
            ref = self._last_valid_t if self._last_valid_t is not None else self._first_t
            missing = t - ref
        else:
            missing = 0.0

        return EyeClosureFeatures(
            eye_closure_ratio=round(closed / observed, 4) if observed > 0 else None,
            eye_closed_duration=None if closed_duration is None else round(closed_duration, 3),
            blink_count=sum(1 for e in self._events if e.kind == "blink"),
            long_closure_count=sum(1 for e in self._events if e.kind == "long"),
            observed_seconds=round(observed, 3),
            face_missing_duration=round(missing, 3),
        )
