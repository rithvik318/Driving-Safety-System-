"""Concise, demo-friendly console output: only level transitions, alarms and recorded events.

    [00:03.2] CAUTION — pedestrian in or moving toward the central road region
              why: Persistent person #2 (observed 1.1 s, 12 detections): moving toward the central band ...
    [00:04.1] HIGH — approaching vehicle / increasing image-space box
    [00:04.1] ALARM — WARNING: ROAD HAZARD
    [00:06.0] EVENT PERSISTENT_HAZARD event_000004 → evidence demo_x/event_000004
"""

from __future__ import annotations

import sys

from app.alerts.models import Alert

DASH = "—"


def fmt_time(t: float | None) -> str:
    if t is None:
        return "--:--.-"
    t = max(0.0, float(t))
    m, s = divmod(t, 60.0)
    return f"{int(m):02d}:{s:04.1f}"


def format_alert(alert: Alert, show_reason: bool = True, max_reason: int = 150) -> list[str]:
    ts = fmt_time(alert.timestamp)
    lines = [f"[{ts}] {alert.risk_level} {DASH} {alert.phrase}"]
    if show_reason and alert.kind == "ESCALATION" and alert.risk_reason:
        reason = alert.risk_reason if len(alert.risk_reason) <= max_reason else alert.risk_reason[:max_reason - 1] + "…"
        lines.append(f"{' ' * (len(ts) + 3)}why: {reason}")
    if alert.alarm:
        lines.append(f"[{ts}] ALARM {DASH} {alert.headline}")
    return lines


def format_event(event) -> str:
    get = (lambda k: event.get(k)) if isinstance(event, dict) else (lambda k: getattr(event, k, None))
    line = f"[{fmt_time(get('timestamp'))}] EVENT {get('event_type')} {get('event_id')}"
    if get("event_type") == "DRIVER_STATE_CHANGE":
        line += f" ({get('change_kind')} {get('transition')})"
    if get("evidence_path"):
        line += f" → evidence {get('evidence_path')}"
    return line


class ConsoleSink:
    """Alert sink that prints format_alert() lines (ASCII-safe on consoles without UTF-8)."""

    def __init__(self, stream=None, show_reason: bool = True):
        self.stream = stream or sys.stdout
        self.show_reason = show_reason

    def __call__(self, alert: Alert) -> None:
        for line in format_alert(alert, self.show_reason):
            _write(self.stream, line)

    def event(self, event) -> None:
        _write(self.stream, format_event(event))


def _write(stream, line: str) -> None:
    try:
        stream.write(line + "\n")
    except UnicodeEncodeError:
        stream.write(line.replace(DASH, "-").replace("→", "->").replace("…", "...") + "\n")
    stream.flush()
