"""Temporal helpers for the risk engine: per-track motion features, head-away timer, hysteresis.

All deterministic; all state is reset by RiskEngine.reset().
"""

from __future__ import annotations

import math
from collections import deque

from app.config.risk_config import LEVELS, RiskConfig
from app.risk.models import RiskLevel


def _slope(points: list[tuple[float, float]]) -> float:
    """Least-squares slope of y over t."""
    n = len(points)
    mt = sum(t for t, _ in points) / n
    my = sum(y for _, y in points) / n
    den = sum((t - mt) ** 2 for t, _ in points)
    return 0.0 if den <= 0 else sum((t - mt) * (y - my) for t, y in points) / den


class TrackFeatureHistory:
    """Short per-track history of (time, ln area, normalised center x) from matched detections.

    growth  = least-squares slope of ln(box area) over the last feature_window_seconds (1/s)
    lateral = least-squares slope of center_x / frame width over the same window (frame-widths/s)
    A slope over the window is far less noisy than the tracker's frame-to-frame rates.
    """

    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self._h: dict[int, deque] = {}

    def reset(self) -> None:
        self._h.clear()

    def update(self, tracks, width: int, height: int) -> None:
        live = set()
        for t in tracks:
            live.add(t.track_id)
            if not t.active or t.area <= 0 or width <= 0:
                continue
            h = self._h.setdefault(t.track_id, deque())
            if h and h[-1][0] >= t.last_seen_timestamp:
                continue  # already recorded this observation
            h.append((t.last_seen_timestamp, math.log(t.area), t.center_x / width))
            while h and h[-1][0] - h[0][0] > self.cfg.feature_window_seconds + 1e-9:
                h.popleft()
        for tid in list(self._h):
            if tid not in live:
                del self._h[tid]

    def features(self, track_id: int) -> tuple[float | None, float | None]:
        """(growth per second, lateral frame-widths per second), or (None, None) if too little data."""
        h = self._h.get(track_id)
        if not h or len(h) < self.cfg.min_window_samples or h[-1][0] - h[0][0] < self.cfg.min_window_seconds - 1e-9:
            return None, None
        pts = list(h)
        return _slope([(t, a) for t, a, _ in pts]), _slope([(t, x) for t, _, x in pts])


class HeadAwayTimer:
    """Continuous time the head pose has been 'away', on the driver's own clock."""

    def __init__(self):
        self.start: float | None = None
        self.last_t: float | None = None

    def reset(self) -> None:
        self.start = self.last_t = None

    def update(self, driver_t: float | None, away: bool | None) -> float:
        if driver_t is None:
            return 0.0 if self.start is None else (self.last_t - self.start)
        if self.last_t is not None and driver_t < self.last_t:
            self.reset()
        if self.last_t is not None and driver_t == self.last_t:  # same observation seen again
            return 0.0 if self.start is None else driver_t - self.start
        self.last_t = driver_t
        if not away:
            self.start = None
            return 0.0
        if self.start is None:
            self.start = driver_t
        return driver_t - self.start


class RiskHysteresis:
    """Level state machine + smoothed score.

    Escalation to a level needs that level (or higher) in escalate_<level>_observations consecutive
    raw observations; the engine then jumps to the highest level whose count is satisfied.
    De-escalation steps down ONE level after deescalate_observations consecutive lower raw
    observations (so risk decays gradually, one level at a time).
    The smoothed score rises immediately with the raw score and falls by at most
    score_decay_per_second. A gap longer than reset_gap_seconds resets everything.
    """

    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.need = {1: cfg.escalate_caution_observations, 2: cfg.escalate_high_observations,
                     3: cfg.escalate_critical_observations}
        self.reset()

    def reset(self) -> None:
        self.level = RiskLevel.SAFE
        self.at_least = {1: 0, 2: 0, 3: 0}
        self.below = 0
        self.score: float | None = None
        self.last_t: float | None = None

    def update(self, raw_level: RiskLevel, raw_score: float, t: float) -> tuple[RiskLevel, float]:
        if self.last_t is not None and (t - self.last_t > self.cfg.reset_gap_seconds or t < self.last_t):
            self.reset()
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t

        for r in (1, 2, 3):
            self.at_least[r] = self.at_least[r] + 1 if raw_level.rank >= r else 0
        target = self.level.rank
        for r in (3, 2, 1):
            if r > self.level.rank and self.at_least[r] >= self.need[r]:
                target = r
                break
        if target > self.level.rank:
            self.level, self.below = RiskLevel.from_rank(target), 0
        elif raw_level.rank < self.level.rank:
            self.below += 1
            if self.below >= self.cfg.deescalate_observations:
                self.level, self.below = RiskLevel.from_rank(self.level.rank - 1), 0
        else:
            self.below = 0

        if self.score is None or raw_score >= self.score:
            self.score = raw_score
        else:
            self.score = max(raw_score, self.score - self.cfg.score_decay_per_second * dt)
        return self.level, self.score


def alarm_level_rank(cfg: RiskConfig) -> int:
    return LEVELS.index(cfg.alarm_min_level)
