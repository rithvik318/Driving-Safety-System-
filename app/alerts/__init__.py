"""Safety alerts: decision-support notifications and alarm outputs derived from the existing risk engine.

SAFE -> no alert; CAUTION -> visual/log notice; HIGH -> "WARNING: ROAD HAZARD" + audible alarm;
CRITICAL -> "CRITICAL: IMMEDIATE ATTENTION REQUIRED" + strong audible alarm. The explanation is always the
engine's own risk_reason. No risk logic lives here; alerts are not collision predictions.
"""

from app.alerts.alarm import AlertManager, TerminalBell
from app.alerts.console import ConsoleSink, fmt_time, format_alert, format_event
from app.alerts.models import DISCLAIMER, LEVEL_TO_SEVERITY, Alert, AlertSeverity, headline

__all__ = ["Alert", "AlertManager", "AlertSeverity", "ConsoleSink", "DISCLAIMER", "LEVEL_TO_SEVERITY", "TerminalBell",
           "fmt_time", "format_alert", "format_event", "headline"]
