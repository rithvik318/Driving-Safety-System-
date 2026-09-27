"""Event recorder: turns risk assessments and driver states into a small set of events.

    rec = EventRecorder(config, run_id="run_001", evidence_root=Path("data/events"))
    rec.start_session("video_x", source_file="video_x.mp4", data_source=DataSource.LOCAL_REAL, camera="front")
    rec.observe_frame("front", t, frame)                      # evidence buffering only (optional)
    events = rec.on_risk(assessment, tracks, driver, frame_index, (width, height))
    events += rec.on_driver(driver_state, frame_index)
    events += rec.finish_session()                             # flushes pending clips

Event triggers (event-driven, not per frame):
  RISK_ESCALATED / RISK_DEESCALATED  every change of the risk level (after the engine's hysteresis);
                                     one event per real change, never debounced away.
  PERSISTENT_HAZARD                  a qualifying road hazard (same hazard type + same track) reported
                                     continuously for >= persistent_hazard_min_seconds while the risk level
                                     is >= persistent_hazard_min_level; repeated at most once per
                                     hazard_cooldown_seconds for the same hazard type + track.
  DRIVER_STATE_CHANGE                hand_state (OBSERVED), driver_activity or drowsiness_level (INFERRED)
                                     changes that hold for driver_change_min_observations consecutive
                                     observations; UNKNOWN is ignored by default; cooldown per
                                     (kind, from, to).
No dependency on YOLO or OpenCV (OpenCV is only imported to write MP4 clips in clip mode).
Event ids are sequential per recorder (run): event_000001, event_000002, ...
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from app.config.event_config import EVENT_LEVELS, EventConfig
from app.events.evidence import EvidenceStore
from app.events.models import (DataSource, EventRecord, EventType, ObservationType, base_fields, driver_fields,
                               risk_fields, track_fields)
from app.sensors.gps import NoGPS

ROAD_HAZARDS = ("APPROACHING_VEHICLE", "PEDESTRIAN_CONFLICT", "ANIMAL_HAZARD")
DRIVER_KINDS = ("hand_state", "driver_activity", "drowsiness_level")

IMAGE_SPACE_NOTE = "road values are image-space (pixels, pixels/s); no calibration, distance, physical speed or TTC"
NOTES_BY_SOURCE = {
    "LOCAL_REAL": "real local capture; derived fields are INFERRED by tracker/risk rules",
    "PUBLIC": "public dataset; derived fields are INFERRED by tracker/risk rules",
    "SYNTHETIC": "SYNTHETIC constructed test input; not a real observation",
    "SYNTHETIC_COMBINATION": "SYNTHETIC_COMBINATION: real driver and front streams that were NOT recorded together; "
                             "a software test, not a drive",
}


@dataclass
class _DriverTrack:
    stable: str | None = None
    candidate: str | None = None
    count: int = 0


@dataclass
class _Session:
    session_id: str
    source_file: str | None
    data_source: DataSource
    camera: str
    level: str = "SAFE"
    hazard_start: dict = field(default_factory=dict)  # (hazard_type, track_id) -> start time
    hazard_last_event: dict = field(default_factory=dict)  # (hazard_type, track_id) -> time of last event
    driver: dict = field(default_factory=lambda: {k: _DriverTrack() for k in DRIVER_KINDS})
    driver_last_event: dict = field(default_factory=dict)
    last_t: float | None = None


class EventRecorder:
    def __init__(self, config: EventConfig | None = None, run_id: str = "run_001", gps=None,
                 evidence_root: Path | None = None):
        self.cfg = config or EventConfig()
        self.run_id = run_id
        self.gps = gps or NoGPS()
        self.evidence = EvidenceStore(evidence_root, self.cfg, prefix=run_id) if evidence_root is not None and self.cfg.evidence_mode != "none" else None
        self._next = 1
        self.session: _Session | None = None
        self.records: list[EventRecord] = []

    # ------------------------------------------------------------------ session

    def start_session(self, session_id: str, source_file: str | None = None,
                      data_source: DataSource = DataSource.LOCAL_REAL, camera: str = "front") -> None:
        if self.session is not None:
            self.finish_session()
        self.session = _Session(session_id, source_file, DataSource(data_source), camera)
        if self.evidence:
            self.evidence.reset()

    def finish_session(self) -> list[EventRecord]:
        out = self._flush_clips(None, force=True)
        self.session = None
        return out

    def _require_session(self) -> _Session:
        if self.session is None:
            raise RuntimeError("call start_session() first")
        return self.session

    # ------------------------------------------------------------------ inputs

    def observe_frame(self, camera: str, t: float, frame) -> list[EventRecord]:
        """Buffer a frame for evidence. Returns events whose clips completed with this frame."""
        if self.evidence is None:
            return []
        self.evidence.push(camera, t, frame)
        return self._flush_clips(t)

    def on_risk(self, assessment, tracks=(), driver=None, frame_index: int | None = None,
                image_size: tuple[int, int] | None = None) -> list[EventRecord]:
        s = self._require_session()
        t = self._check_time(assessment.timestamp)
        out: list[EventRecord] = []
        level = getattr(assessment.risk_level, "value", assessment.risk_level)
        tracks_by_id = {tr.track_id: tr for tr in tracks or ()}
        hazards_by_id = {h.track_id: h for h in assessment.hazards}
        width, height = image_size if image_size else (None, None)
        primary = tracks_by_id.get(assessment.primary_track_id)

        # 1. level transitions (always recorded)
        if level != s.level:
            up = EVENT_LEVELS.index(level) > EVENT_LEVELS.index(s.level)
            etype = EventType.RISK_ESCALATED if up else EventType.RISK_DEESCALATED
            values = self._base(t, frame_index, etype, ObservationType.INFERRED)
            values.update(risk_fields(assessment), **track_fields(primary, hazards_by_id.get(assessment.primary_track_id), width, height),
                          **driver_fields(driver), transition=f"{s.level}->{level}", previous_risk_level=s.level)
            key = (assessment.hazard_type.value, assessment.primary_track_id)
            if key in s.hazard_start:
                values["hazard_persistence"] = round(t - s.hazard_start[key], 4)
            out += self._emit(values, etype)
            s.level = level

        # 2. persistent hazards (per qualifying track, cooldown per hazard type + track)
        current = {}
        for h in assessment.hazards:
            if h.qualifies and h.hazard_type.value in ROAD_HAZARDS:
                key = (h.hazard_type.value, h.track_id)
                current[key] = s.hazard_start.get(key, t)
        s.hazard_start = current
        if EVENT_LEVELS.index(level) >= EVENT_LEVELS.index(self.cfg.persistent_hazard_min_level):
            for key in sorted(current, key=lambda k: (k[1], k[0])):
                duration = t - current[key]
                last = s.hazard_last_event.get(key)
                if duration + 1e-9 < self.cfg.persistent_hazard_min_seconds:
                    continue
                if last is not None and t - last < self.cfg.hazard_cooldown_seconds - 1e-9:
                    continue
                track = tracks_by_id.get(key[1])
                values = self._base(t, frame_index, EventType.PERSISTENT_HAZARD, ObservationType.INFERRED)
                values.update(risk_fields(assessment), **track_fields(track, hazards_by_id.get(key[1]), width, height),
                              **driver_fields(driver), hazard_persistence=round(duration, 4))
                values["primary_track_id"] = key[1]
                values["hazard_type"] = assessment.hazard_type.value if assessment.primary_track_id == key[1] else key[0]
                s.hazard_last_event[key] = t
                out += self._emit(values, EventType.PERSISTENT_HAZARD)
        return out

    def on_driver(self, driver, frame_index: int | None = None) -> list[EventRecord]:
        s = self._require_session()
        if driver is None:
            return []
        t = self._check_time(driver.timestamp)
        observed = {
            "hand_state": getattr(driver.hand.hand_state, "value", driver.hand.hand_state),
            "driver_activity": getattr(driver.driver_activity, "value", driver.driver_activity),
            "drowsiness_level": getattr(driver.temporal.drowsiness_level, "value", driver.temporal.drowsiness_level),
        }
        out = []
        for kind in DRIVER_KINDS:
            value, st = observed[kind], s.driver[kind]
            if value == "UNKNOWN" and not self.cfg.include_unknown_driver_states:
                continue  # neither a state nor a break of a streak
            if value == st.stable:
                st.candidate, st.count = None, 0
                continue
            st.count = st.count + 1 if value == st.candidate else 1
            st.candidate = value
            if st.count < self.cfg.driver_change_min_observations:
                continue
            previous, st.stable, st.candidate, st.count = st.stable, value, None, 0
            if previous is None:
                continue  # first stable state of the session: a baseline, not a change
            key = (kind, previous, value)
            last = s.driver_last_event.get(key)
            if last is not None and t - last < self.cfg.driver_change_cooldown_seconds - 1e-9:
                continue
            s.driver_last_event[key] = t
            otype = ObservationType.OBSERVED if kind == "hand_state" else ObservationType.INFERRED
            values = self._base(t, frame_index, EventType.DRIVER_STATE_CHANGE, otype)
            values.update(driver_fields(driver), change_kind=kind, transition=f"{previous}->{value}")
            out += self._emit(values, EventType.DRIVER_STATE_CHANGE)
        return out

    # ------------------------------------------------------------------ helpers

    def _check_time(self, t) -> float:
        if t is None or isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
            raise ValueError(f"timestamp must be a finite number, got {t!r}")
        return float(t)

    def _base(self, t, frame_index, etype, otype) -> dict:
        s = self.session
        values = base_fields("", self.run_id, s.session_id, s.source_file, s.camera, s.data_source, otype, t,
                             frame_index, etype, self.gps.fix_at(t))
        notes = [NOTES_BY_SOURCE[s.data_source.value]]
        if values["gps_source"] == "UNAVAILABLE":
            notes.append("GPS unavailable: coordinates null")
        if s.camera != "driver":
            notes.append(IMAGE_SPACE_NOTE)
        values["notes"] = json.dumps(notes)
        return values

    def _emit(self, values: dict, etype: EventType) -> list[EventRecord]:
        values["event_id"] = f"event_{self._next:06d}"
        self._next += 1
        wants = self.evidence is not None and etype.value in self.cfg.evidence_event_types
        if wants:
            cams = [c for c in ("front", "driver") if c in self.evidence.cameras_with_frames()]
            if self.cfg.evidence_mode == "clip" and cams:
                self.evidence.start_clip(values["event_id"], values["timestamp"], cams, values)
                return []  # released by _flush_clips once the clip exists
            kind, path, files = self.evidence.write_frames(values["event_id"], values["timestamp"], cams, values) \
                if self.cfg.evidence_mode == "frame" else ("NONE", None, [])
            if kind != "NONE":
                values.update(evidence_kind=kind, evidence_path=path, evidence_files=json.dumps(files))
        rec = EventRecord(values)
        self.records.append(rec)
        return [rec]

    def _flush_clips(self, now: float | None, force: bool = False) -> list[EventRecord]:
        if self.evidence is None or not self.evidence.pending:
            return []
        out = []
        for clip in self.evidence.ready_clips(now if now is not None else math.inf, force=force):
            _kind, _path, _files, values = self.evidence.write_clip(clip)
            rec = EventRecord(values)
            self.records.append(rec)
            out.append(rec)
        out.sort(key=lambda r: r.event_id)
        return out
