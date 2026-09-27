"""AlertManager: turns the existing risk engine's output into safety alerts and alarm outputs.

    manager = AlertManager(data_source="LOCAL_REAL", sinks=[ConsoleSink()])
    alerts = manager.update(assessment, events)      # live: RiskAssessment (+ events recorded at that step)
    alerts = manager.update_from_event(event_dict)   # replay: a RISK_ESCALATED / RISK_DEESCALATED event record

Rules (no risk logic here; levels, reasons and the alarm recommendation come from the engine):
  - an alert is produced only when the engine's risk level (after its hysteresis) CHANGES
  - SAFE -> nothing;  CAUTION -> visual/log notice;  HIGH -> "WARNING: ROAD HAZARD" + audible alarm;
    CRITICAL -> "CRITICAL: IMMEDIATE ATTENTION REQUIRED" + strong audible alarm
  - the audible alarm fires on ESCALATION into a level the engine marks alarm_recommended
  - de-escalations are logged (no alarm)
Sinks receive every Alert: ConsoleSink (app/alerts/console.py), TerminalBell (audible), or any callable.
"""

from __future__ import annotations

import sys
from typing import Callable, Iterable

from app.alerts.models import (CHANNELS, HAZARD_PHRASES, LEVEL_TO_SEVERITY, LEVELS, Alert, AlertSeverity, headline)


class TerminalBell:
    """Audible alarm output: the terminal bell (BEL), once for WARNING and three times for CRITICAL.
    Disabled by default so logs and tests stay quiet."""

    def __init__(self, enabled: bool = False, stream=None):
        self.enabled = enabled
        self.stream = stream or sys.stdout
        self.rings = 0

    def __call__(self, alert: Alert) -> None:
        if not alert.alarm:
            return
        n = 3 if alert.severity is AlertSeverity.CRITICAL else 1
        self.rings += n
        if self.enabled:
            self.stream.write("\a" * n)
            self.stream.flush()


class AlertManager:
    def __init__(self, data_source: str = "LOCAL_REAL", sinks: Iterable[Callable[[Alert], None]] = (),
                 session_id: str | None = None):
        self.data_source = data_source
        self.sinks = list(sinks)
        self.session_id = session_id
        self.level = "SAFE"
        self.alerts: list[Alert] = []
        self._n = 0

    def reset(self, session_id: str | None = None) -> None:
        """New stream: back to SAFE (alert ids keep counting)."""
        self.level = "SAFE"
        self.session_id = session_id

    # ------------------------------------------------------------------ inputs

    def update(self, assessment, events: Iterable = ()) -> list[Alert]:
        """Consume one RiskAssessment (and the event records produced at the same step)."""
        level = getattr(assessment.risk_level, "value", assessment.risk_level)
        return self._on_level(
            timestamp=assessment.timestamp, level=level, reason=assessment.reason, score=assessment.risk_score,
            hazard_type=getattr(assessment.hazard_type, "value", assessment.hazard_type),
            track_id=assessment.primary_track_id, track_class=assessment.primary_class,
            alarm_recommended=bool(assessment.alarm_recommended),
            event_ids=tuple(_get(e, "event_id") for e in events if _get(e, "event_id")))

    def update_from_event(self, event) -> list[Alert]:
        """Consume a recorded RISK_ESCALATED / RISK_DEESCALATED event (dict or EventRecord). Other events: []."""
        if _get(event, "event_type") not in ("RISK_ESCALATED", "RISK_DEESCALATED") or not _get(event, "risk_level"):
            return []
        return self._on_level(
            timestamp=_get(event, "timestamp"), level=_get(event, "risk_level"), reason=_get(event, "risk_reason"),
            score=_get(event, "smoothed_risk_score"), hazard_type=_get(event, "hazard_type"),
            track_id=_get(event, "primary_track_id"), track_class=_get(event, "hazard_track_class"),
            alarm_recommended=bool(_get(event, "alarm_recommended")), event_ids=(_get(event, "event_id"),),
            previous=_get(event, "previous_risk_level"))

    # ------------------------------------------------------------------ core

    def _on_level(self, timestamp, level, reason, score, hazard_type, track_id, track_class, alarm_recommended,
                  event_ids, previous=None) -> list[Alert]:
        if level not in LEVELS:
            raise ValueError(f"unknown risk level {level!r}")
        prev = previous or self.level
        if level == self.level and previous is None:
            return []
        self.level = level
        up = LEVELS.index(level) > LEVELS.index(prev)
        severity = LEVEL_TO_SEVERITY[level]
        self._n += 1
        alert = Alert(
            alert_id=f"alert_{self._n:04d}", timestamp=float(timestamp), kind="ESCALATION" if up else "DEESCALATION",
            risk_level=level, previous_level=prev, severity=severity,
            headline=headline(severity, hazard_type) if up else None,
            phrase=HAZARD_PHRASES.get(hazard_type or "NONE", hazard_type or "no active hazard") if up
            else f"risk lowered from {prev}",
            risk_reason=reason, risk_score=score, hazard_type=hazard_type, primary_track_id=track_id,
            primary_class=track_class, alarm=bool(up and alarm_recommended),
            channels=CHANNELS[severity] if up else ("log",), event_ids=tuple(event_ids), session_id=self.session_id,
            data_source=self.data_source)
        self.alerts.append(alert)
        for sink in self.sinks:
            sink(alert)
        return [alert]

    # ------------------------------------------------------------------ summary

    def summary(self) -> dict:
        esc = [a for a in self.alerts if a.kind == "ESCALATION"]
        return {
            "alerts": len(self.alerts),
            "escalations": len(esc),
            "deescalations": len(self.alerts) - len(esc),
            "by_severity": {s.value: sum(1 for a in esc if a.severity is s) for s in AlertSeverity if s is not AlertSeverity.NONE},
            "alarms": sum(a.alarm for a in self.alerts),
        }


def _get(obj, name):
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
