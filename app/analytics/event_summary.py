"""Loading and summarising LOCAL_REAL event records (post-event, offline).

Only real records (data_source LOCAL_REAL / PUBLIC) are accepted; SYNTHETIC and SYNTHETIC_COMBINATION rows are
counted as excluded and never enter the statistics. Malformed lines are skipped and counted. Missing optional
fields stay missing (None): nothing is invented.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REAL_SOURCES = ("LOCAL_REAL", "PUBLIC")
LEVELS = ("SAFE", "CAUTION", "HIGH", "CRITICAL")
ROAD_HAZARDS = ("APPROACHING_VEHICLE", "PEDESTRIAN_CONFLICT", "ANIMAL_HAZARD")
DRIVER_HAZARDS = ("DRIVER_DROWSINESS", "HANDS_OFF_WHEEL", "DRIVER_HEAD_AWAY")


@dataclass
class LoadResult:
    events: list = field(default_factory=list)
    sources: list = field(default_factory=list)  # files read
    missing_files: list = field(default_factory=list)
    malformed_lines: int = 0
    excluded_non_real: Counter = field(default_factory=Counter)  # data_source -> count
    duplicates: int = 0


def _num(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def load_real_events(paths) -> LoadResult:
    """Read JSONL event files; keep only real records (deduplicated by (run_id, event_id))."""
    res, seen = LoadResult(), set()
    for p in paths:
        p = Path(p)
        if not p.is_file():
            res.missing_files.append(str(p))
            continue
        res.sources.append(str(p))
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                res.malformed_lines += 1
                continue
            if not isinstance(rec, dict) or not rec.get("event_type") or not rec.get("event_id"):
                res.malformed_lines += 1
                continue
            src = rec.get("data_source")
            if src not in REAL_SOURCES:
                res.excluded_non_real[str(src)] += 1
                continue
            key = (rec.get("run_id"), rec.get("event_id"))
            if key in seen:
                res.duplicates += 1
                continue
            seen.add(key)
            res.events.append(rec)
    res.events.sort(key=lambda r: (str(r.get("run_id")), str(r.get("session_id")), _num(r.get("timestamp")) or 0.0,
                                   str(r.get("event_id"))))
    return res


def _stats(values) -> dict | None:
    vals = [v for v in (_num(x) for x in values) if v is not None]
    if not vals:
        return None
    return {"n": len(vals), "mean": round(statistics.fmean(vals), 3), "median": round(statistics.median(vals), 3),
            "max": round(max(vals), 3), "min": round(min(vals), 3)}


def elevated_episodes(events: list[dict]) -> list[dict]:
    """Per session: spans from a SAFE->X escalation until the next X->SAFE de-escalation (stream time).

    An episode still open when the session's records end is reported with end=None (the clip ended or the level
    never returned to SAFE); its duration is not invented."""
    by_session = defaultdict(list)
    for e in events:
        if e.get("event_type") in ("RISK_ESCALATED", "RISK_DEESCALATED") and _num(e.get("timestamp")) is not None:
            by_session[(e.get("run_id"), e.get("session_id"))].append(e)
    out = []
    for (run, session), evs in by_session.items():
        evs.sort(key=lambda e: e["timestamp"])
        start, peak = None, None
        for e in evs:
            level = e.get("risk_level")
            if e.get("previous_risk_level") == "SAFE" and e["event_type"] == "RISK_ESCALATED":
                start, peak = e, level
            elif start is not None:
                if level in LEVELS and LEVELS.index(level) > LEVELS.index(peak or "SAFE"):
                    peak = level
                if level == "SAFE":
                    out.append({"session_id": session, "start": start["timestamp"], "end": e["timestamp"],
                                "duration_seconds": round(e["timestamp"] - start["timestamp"], 3), "peak_level": peak,
                                "start_event": start["event_id"], "end_event": e["event_id"]})
                    start = None
        if start is not None:
            out.append({"session_id": session, "start": start["timestamp"], "end": None, "duration_seconds": None,
                        "peak_level": peak, "start_event": start["event_id"], "end_event": None})
    return out


def summarize_events(events: list[dict]) -> dict:
    """Aggregate statistics over REAL events. Counts only; no rates per km or per hour (no distance/clock data)."""
    n = len(events)
    typ = Counter(e.get("event_type") for e in events)
    level_events = [e for e in events if e.get("risk_level") is not None]
    persistent = [e for e in events if e.get("event_type") == "PERSISTENT_HAZARD"]
    driver = [e for e in events if e.get("event_type") == "DRIVER_STATE_CHANGE"]
    alarm = Counter("true" if e.get("alarm_recommended") is True else "false" if e.get("alarm_recommended") is False
                    else "not recorded" for e in events)
    episodes = elevated_episodes(events)
    closed = [ep["duration_seconds"] for ep in episodes if ep["duration_seconds"] is not None]
    return {
        "total_events": n,
        "events_by_type": dict(typ.most_common()),
        "events_by_camera": dict(Counter(e.get("camera") or "not recorded" for e in events).most_common()),
        "events_by_risk_level": {lv: sum(1 for e in level_events if e.get("risk_level") == lv) for lv in LEVELS},
        "events_without_risk_level": n - len(level_events),
        "events_by_hazard_type": dict(Counter(e.get("hazard_type") or "not recorded" for e in events).most_common()),
        "events_by_hazard_class": dict(Counter(e["hazard_class"] for e in events if e.get("hazard_class")).most_common()),
        "risk_transitions": dict(Counter(e.get("transition") for e in events
                                         if e.get("event_type") in ("RISK_ESCALATED", "RISK_DEESCALATED")
                                         and e.get("transition")).most_common()),
        "alarm_recommended": dict(alarm),
        "alarm_triggered_true": sum(1 for e in events if e.get("alarm_triggered") is True),
        "persistent_hazards": {
            "count": len(persistent),
            "by_hazard_type": dict(Counter(e.get("hazard_type") or "not recorded" for e in persistent).most_common()),
            "by_risk_level": dict(Counter(e.get("risk_level") or "not recorded" for e in persistent).most_common()),
            "hazard_persistence_seconds": _stats(e.get("hazard_persistence") for e in persistent),
        },
        "driver_state_events": {
            "count": len(driver),
            "by_change_kind": dict(Counter(e.get("change_kind") or "not recorded" for e in driver).most_common()),
            "by_transition": dict(Counter(f"{e.get('change_kind')}: {e.get('transition')}" for e in driver).most_common()),
        },
        "smoothed_risk_score": _stats(e.get("smoothed_risk_score") for e in events),
        "raw_risk_score": _stats(e.get("raw_risk_score") for e in events),
        "durations": {
            "track_persistence_seconds": _stats(e.get("track_persistence_seconds") for e in events),
            "hazard_persistence_seconds": _stats(e.get("hazard_persistence") for e in events),
            "elevated_risk_episodes": {"count": len(episodes), "closed": len(closed), "open_at_end_of_clip": len(episodes) - len(closed),
                                       "closed_duration_seconds": _stats(closed)},
            "note": "stream (video) time; an episode is SAFE->elevated until the next return to SAFE within one session",
        },
        "sessions": len({(e.get("run_id"), e.get("session_id")) for e in events}),
        "runs": sorted({str(e.get("run_id")) for e in events}),
        "data_sources": dict(Counter(e.get("data_source") for e in events)),
        "observation_types": dict(Counter(e.get("observation_type") or "not recorded" for e in events)),
        "evidence": {"events_with_evidence": sum(1 for e in events if e.get("evidence_path")),
                     "evidence_kinds": dict(Counter(e.get("evidence_kind") or "not recorded" for e in events))},
        "_episodes": episodes,
    }
