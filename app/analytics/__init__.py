"""Post-event safety intelligence (offline, after events are recorded; not part of the real-time loop).

LOCAL_REAL event statistics, descriptive repeated-pattern analysis, hotspot analysis that refuses to fabricate
locations, deterministic review recommendations, per-event entries and an optional LLM summary hook
(generate_post_event_summary, deterministic by default). Synthetic data is only reported separately on request.
"""

from app.analytics.event_summary import elevated_episodes, load_real_events, summarize_events
from app.analytics.hotspot import GPS_UNAVAILABLE, hotspot_analysis, synthetic_gps_summary
from app.analytics.patterns import describe_patterns, find_patterns
from app.analytics.report import (build_safety_report, generate_post_event_summary, recommend, render_markdown,
                                  synthetic_validation, write_safety_report)

__all__ = ["GPS_UNAVAILABLE", "build_safety_report", "describe_patterns", "elevated_episodes", "find_patterns",
           "generate_post_event_summary", "hotspot_analysis", "load_real_events", "recommend", "render_markdown",
           "summarize_events", "synthetic_gps_summary", "synthetic_validation", "write_safety_report"]
