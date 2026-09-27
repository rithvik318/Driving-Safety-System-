"""End-to-end demo on ONE real local video (front OR driver camera), reusing the existing pipeline components.
No synchronised driver + road recording exists; the demo never pairs the two cameras."""

from app.demo.runner import (NO_SYNC_STATEMENT, SUMMARY_KEYS, TIMELINE_COLUMNS, DemoError, DemoRun, check_real_video,
                             run_driver_demo, run_front_demo)
from app.demo.report import render_final_report

__all__ = ["DemoError", "DemoRun", "NO_SYNC_STATEMENT", "SUMMARY_KEYS", "TIMELINE_COLUMNS", "check_real_video",
           "render_final_report", "run_driver_demo", "run_front_demo"]
