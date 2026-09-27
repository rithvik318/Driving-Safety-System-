"""Final integrated demo: ONE real local video -> existing pipeline -> events -> safety alerts -> evidence.

    python scripts/run_demo.py --video "D:/.../frontcamera/vehicles/video_20260926_170005.mp4" ^
        --output-dir outputs/demo --fps 10 --device auto

    python scripts/run_demo.py --camera driver --video "D:/.../drivercamera/drowsy_driver/video_20260926_220714.mp4" ^
        --output-dir outputs/demo_driver --fps 10

Front camera: YOLO -> tracker -> risk engine (road-only) -> event recorder -> alerts.
Driver camera: driver perception -> risk engine (driver-only) -> event recorder -> alerts.
The cameras were NOT recorded together: one camera per run; they are never paired here.

Outputs in --output-dir: demo_summary.json, events.jsonl, events.parquet, schema.json, evidence/, alerts.jsonl,
timeline.csv, risk_timeline.png, annotated_demo.mp4 (unless --no-video). Also (re)writes the combined report
outputs/reports/final_demo_report.md from this run plus any --include-summary files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python scripts/...`


def _rel(p: Path, root: Path) -> str:
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(p)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    from app.config import configure_logging, load_settings
    from app.config.settings import PROJECT_ROOT

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, required=True, help="REAL local video (front or driver camera)")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    p.add_argument("--fps", type=float, default=10.0, help="frames per second of video time to process")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda (front camera detector)")
    p.add_argument("--camera", choices=("front", "driver"), default="front")
    p.add_argument("--max-seconds", type=float, default=None, help="process only the first N seconds")
    p.add_argument("--no-video", action="store_true", help="skip annotated_demo.mp4")
    p.add_argument("--beep", action="store_true", help="ring the terminal bell for alarms")
    p.add_argument("--evidence-mode", choices=("frame", "clip", "none"), default="frame")
    p.add_argument("--report", type=Path, default=Path("outputs/reports/final_demo_report.md"))
    p.add_argument("--include-summary", type=Path, action="append", default=[],
                   help="another run's demo_summary.json to include in the combined report (e.g. the driver demo)")
    args = p.parse_args(argv)
    configure_logging("WARNING" if load_settings().log_level == "INFO" else load_settings().log_level)

    from app.demo import DemoError, render_final_report, run_driver_demo, run_front_demo

    out = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    print(f"REAL {args.camera}-camera demo: {args.video}")
    print("data_source=LOCAL_REAL, derived values INFERRED; no synchronized driver+road recording exists "
          f"({'road-only' if args.camera == 'front' else 'driver-only'} run)\n")
    try:
        if args.camera == "front":
            run = run_front_demo(args.video, out, args.fps, args.device, args.max_seconds, annotate=not args.no_video,
                                 beep=args.beep, evidence_mode=args.evidence_mode)
        else:
            run = run_driver_demo(args.video, out, args.fps, args.max_seconds, annotate=not args.no_video, beep=args.beep,
                                  evidence_mode=args.evidence_mode)
    except (DemoError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    summary = json.loads((out / "demo_summary.json").read_text(encoding="utf-8"))
    runs = [(_rel(out, PROJECT_ROOT), summary)]
    for extra in args.include_summary:
        extra = extra if extra.is_absolute() else PROJECT_ROOT / extra
        if extra.resolve() != (out / "demo_summary.json").resolve():
            runs.append((_rel(extra.parent, PROJECT_ROOT), json.loads(extra.read_text(encoding="utf-8"))))
    report = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_final_report(runs), encoding="utf-8")

    s = summary
    print("\n--- summary ---")
    print(f"frames processed {s['processing']['frames_processed']} ({s['input']['duration_seconds_processed']} s of video)")
    if run.camera == "front":
        print(f"detections {s['perception']['detections']}, tracks {s['perception']['tracks']}")
    print(f"risk: {s['risk']['frames_by_level']} (peak {s['risk']['peak_level']})")
    print(f"events {s['events']['total']} {s['events']['by_type']}")
    print(f"alerts {s['alerts']['escalations']} escalations, {s['alerts']['alarms']} alarm(s); evidence files {s['evidence']['files']}")
    print(f"annotated video: {'yes' if s['outputs']['annotated_demo.mp4'] else 'no'}")
    print(f"output: {out}\nreport: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
