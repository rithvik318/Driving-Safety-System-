"""Descriptive repeated-pattern analysis over REAL events.

A pattern is "repeated" when its events occur at least twice in the same session, or in at least two sessions.
This describes what the prototype recorded. It does not show that a place is dangerous, that driving was
unsafe, or that the underlying detections were correct.
"""

from __future__ import annotations

from collections import Counter

PATTERN_DISCLAIMER = ("Descriptive counts of what the prototype recorded. They do not show that a location is "
                      "dangerous or that driving was unsafe, and they inherit detector/classifier errors.")


def _is_escalation_to(e, levels=("CAUTION", "HIGH", "CRITICAL")):
    return e.get("event_type") == "RISK_ESCALATED" and e.get("risk_level") in levels


PATTERNS = {
    "pedestrian_hazard": ("Pedestrian hazards", lambda e: e.get("hazard_type") == "PEDESTRIAN_CONFLICT"
                          and e.get("event_type") in ("RISK_ESCALATED", "PERSISTENT_HAZARD")),
    "approaching_vehicle_hazard": ("Approaching-vehicle hazards", lambda e: e.get("hazard_type") == "APPROACHING_VEHICLE"
                                   and e.get("event_type") in ("RISK_ESCALATED", "PERSISTENT_HAZARD")),
    "animal_hazard": ("Animal hazards", lambda e: e.get("hazard_type") == "ANIMAL_HAZARD"
                      and e.get("event_type") in ("RISK_ESCALATED", "PERSISTENT_HAZARD")),
    "driver_state_change": ("Driver-state changes", lambda e: e.get("event_type") == "DRIVER_STATE_CHANGE"),
    "risk_escalation": ("Risk escalations", _is_escalation_to),
    "high_or_critical_escalation": ("Escalations to HIGH or CRITICAL", lambda e: _is_escalation_to(e, ("HIGH", "CRITICAL"))),
}


def find_patterns(events: list[dict]) -> dict:
    out = {}
    for key, (label, pred) in PATTERNS.items():
        matched = [e for e in events if pred(e)]
        per_session = Counter(str(e.get("session_id")) for e in matched)
        repeated_within = {s: c for s, c in per_session.items() if c >= 2}
        out[key] = {
            "label": label,
            "events": len(matched),
            "sessions": len(per_session),
            "sessions_with_repeats": len(repeated_within),
            "max_in_one_session": max(per_session.values(), default=0),
            "repeated": bool(repeated_within) or len(per_session) >= 2,
            "event_ids": [e.get("event_id") for e in matched],
            "by_session": dict(per_session.most_common()),
        }
        if key == "driver_state_change":
            out[key]["by_transition"] = dict(Counter(f"{e.get('change_kind')}: {e.get('transition')}" for e in matched).most_common())
    return out


def describe_patterns(patterns: dict) -> list[str]:
    """One neutral sentence per pattern that occurred."""
    lines = []
    for p in patterns.values():
        if not p["events"]:
            lines.append(f"{p['label']}: none recorded.")
            continue
        rep = (f"repeated within {p['sessions_with_repeats']} session(s) (up to {p['max_in_one_session']} in one session)"
               if p["sessions_with_repeats"] else "no session had more than one")
        lines.append(f"{p['label']}: {p['events']} event(s) across {p['sessions']} session(s); {rep}.")
    return lines
