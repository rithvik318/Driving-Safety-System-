"""Duration tracking for categorical driver states (hand state / manual activity).

Works with any string labels, e.g. BOTH_HANDS, ONE_HAND, NO_HANDS, PHONE,
OTHER_MANUAL_DISTRACTION, UNKNOWN.

Example: 0 s BOTH_HANDS, 2 s ONE_HAND, 5 s NO_HANDS, 8 s BOTH_HANDS
    -> ONE_HAND lasted 3 s, NO_HANDS lasted 3 s.

UNKNOWN never counts as distraction: while the state is UNKNOWN the
distraction duration is None, and UNKNOWN ends any running distraction.

UNKNOWN is a state of its own. It never extends the state before it: a frame
reported as UNKNOWN closes the running ONE_HAND / NO_HANDS segment at that frame's
timestamp, and a later ONE_HAND / NO_HANDS starts a new segment from zero. It does
not reset anything else (totals, longest durations, transition history).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Transition:
    timestamp: float
    from_state: str | None
    to_state: str


@dataclass(frozen=True)
class Segment:
    state: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class StateDurationTracker:
    def __init__(self, distracting_states: set[str] | tuple[str, ...], history: int = 1000):
        self.distracting = {s.upper() for s in distracting_states}
        self._history = history
        self.reset()

    def reset(self) -> None:
        self.current_state: str | None = None
        self.state_start: float | None = None
        self.last_timestamp: float | None = None
        self.transitions: deque[Transition] = deque(maxlen=self._history)
        self.segments: deque[Segment] = deque(maxlen=self._history)
        self._totals: dict[str, float] = defaultdict(float)
        self._distraction_start: float | None = None
        self._longest: dict[str, float] = defaultdict(float)
        self._changes = 0  # not capped like the transition history

    def update(self, t: float, state: str) -> None:
        state = state.upper()
        if state != self.current_state:
            if self.current_state is not None and self.state_start is not None:
                seg = Segment(self.current_state, self.state_start, t)
                self.segments.append(seg)
                self._totals[seg.state] += seg.duration
                self._longest[seg.state] = max(self._longest[seg.state], seg.duration)
            if self.current_state is not None:
                self._changes += 1
            self.transitions.append(Transition(t, self.current_state, state))
            self.current_state, self.state_start = state, t

        if state in self.distracting:
            if self._distraction_start is None:
                self._distraction_start = t
        else:
            self._distraction_start = None
        self.last_timestamp = t

    @property
    def current_duration(self) -> float | None:
        if self.state_start is None or self.last_timestamp is None:
            return None
        return self.last_timestamp - self.state_start

    @property
    def distraction_duration(self) -> float | None:
        """Continuous time in any distracting state; 0.0 if not distracted; None if UNKNOWN."""
        if self.current_state is None or self.current_state == UNKNOWN:
            return None
        if self._distraction_start is None:
            return 0.0
        return self.last_timestamp - self._distraction_start

    def durations_by_state(self) -> dict[str, float]:
        """Total time per state, including the segment still in progress."""
        totals = dict(self._totals)
        if self.current_state is not None and self.current_duration is not None:
            totals[self.current_state] = totals.get(self.current_state, 0.0) + self.current_duration
        return totals

    @property
    def transition_count(self) -> int:
        """Number of state changes after the first observed state."""
        return self._changes

    def longest_duration(self, state: str) -> float:
        """Longest continuous time in `state` so far, including the segment in progress."""
        state = state.upper()
        best = self._longest.get(state, 0.0)
        if self.current_state == state and self.current_duration is not None:
            best = max(best, self.current_duration)
        return best
