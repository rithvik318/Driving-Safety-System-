"""Event recorder configuration. PROTOTYPE DEFAULTS; override with EVENT_<FIELD> env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

from app.config.settings import PROJECT_ROOT, _resolve_path

EVENT_LEVELS = ("SAFE", "CAUTION", "HIGH", "CRITICAL")


@dataclass(frozen=True)
class EventConfig:
    # dataset locations (REAL and SYNTHETIC are kept in separate roots, never mixed)
    event_root: Path = Path("data/events")  # [EVENT_EVENT_ROOT] LOCAL_REAL / PUBLIC events
    synthetic_root: Path = Path("data/events_synthetic")  # [EVENT_SYNTHETIC_ROOT] SYNTHETIC / SYNTHETIC_COMBINATION

    # PERSISTENT_HAZARD
    persistent_hazard_min_seconds: float = 2.0  # track persistence needed before the event
    persistent_hazard_min_level: str = "CAUTION"  # risk level (after hysteresis) needed
    hazard_cooldown_seconds: float = 10.0  # same hazard type + same track: no repeat within this time

    # DRIVER_STATE_CHANGE
    driver_change_min_observations: int = 3  # new state must hold this many consecutive observations
    driver_change_cooldown_seconds: float = 5.0  # same kind + same from->to: no repeat within this time
    include_unknown_driver_states: bool = False  # transitions to/from UNKNOWN are not events by default

    # evidence
    evidence_mode: str = "frame"  # "frame" (one JPEG per camera) | "clip" (MP4 from a ring buffer) | "none"
    evidence_pre_seconds: float = 5.0  # clip mode: seconds kept before the event
    evidence_post_seconds: float = 5.0  # clip mode: seconds recorded after the event
    evidence_max_side: int = 960  # evidence images/clips are downscaled to this long side
    evidence_jpeg_quality: int = 85
    evidence_event_types: tuple[str, ...] = ("RISK_ESCALATED", "PERSISTENT_HAZARD", "DRIVER_STATE_CHANGE")


def load_event_config(env_file: Path | None = None) -> EventConfig:
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    defaults, kwargs = EventConfig(), {}
    for f in fields(EventConfig):
        raw = (os.environ.get(f"EVENT_{f.name.upper()}") or "").strip().strip('"').strip("'")
        if not raw:
            continue
        cur = getattr(defaults, f.name)
        try:
            if isinstance(cur, Path):
                kwargs[f.name] = Path(raw)
            elif isinstance(cur, bool):
                kwargs[f.name] = raw.lower() in ("1", "true", "yes")
            elif isinstance(cur, int):
                kwargs[f.name] = int(raw)
            elif isinstance(cur, float):
                kwargs[f.name] = float(raw)
            elif isinstance(cur, tuple):
                kwargs[f.name] = tuple(s.strip().upper() for s in raw.split(",") if s.strip())
            else:
                kwargs[f.name] = raw.strip().lower() if f.name == "evidence_mode" else raw.strip().upper()
        except ValueError as exc:
            raise ValueError(f"EVENT_{f.name.upper()} has an invalid value {raw!r}") from exc
    config = EventConfig(**kwargs)
    validate_event_config(config)
    return config


def resolved_roots(c: EventConfig) -> tuple[Path, Path]:
    return _resolve_path(str(c.event_root), PROJECT_ROOT), _resolve_path(str(c.synthetic_root), PROJECT_ROOT)


def validate_event_config(c: EventConfig) -> None:
    if c.persistent_hazard_min_level not in EVENT_LEVELS:
        raise ValueError(f"EVENT_PERSISTENT_HAZARD_MIN_LEVEL must be one of {EVENT_LEVELS}")
    if c.evidence_mode not in ("frame", "clip", "none"):
        raise ValueError("EVENT_EVIDENCE_MODE must be frame, clip or none")
    for name in ("persistent_hazard_min_seconds", "hazard_cooldown_seconds", "driver_change_cooldown_seconds",
                 "evidence_pre_seconds", "evidence_post_seconds"):
        if getattr(c, name) < 0:
            raise ValueError(f"EVENT_{name.upper()} must be >= 0")
    if c.driver_change_min_observations < 1 or c.evidence_max_side < 64 or not 1 <= c.evidence_jpeg_quality <= 100:
        raise ValueError("invalid event evidence/driver settings")
    if Path(c.event_root) == Path(c.synthetic_root):
        raise ValueError("EVENT_EVENT_ROOT and EVENT_SYNTHETIC_ROOT must differ (real and synthetic are never mixed)")
