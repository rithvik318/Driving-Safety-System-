"""Post-event safety report from LOCAL_REAL event logs (offline; after events are recorded).

    python scripts/generate_safety_report.py                       # LOCAL_REAL only (data/events/events.jsonl)
    python scripts/generate_safety_report.py --include-synthetic   # + separate SYNTHETIC SCENARIO VALIDATION section

Writes outputs/reports/safety_summary.md, safety_summary.json, event_statistics.csv and (with --system-status,
default on) outputs/reports/final_system_status.md. Synthetic data is read only with --include-synthetic and is
never combined with real counts (final_system_status.md always reports it in its own section).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python scripts/...`


def _abs(p: Path, root: Path) -> Path:
    return p if p.is_absolute() else root / p


def _rel(p: Path, root: Path) -> str:
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(p)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    from app.analytics import build_safety_report, load_real_events, synthetic_validation, write_safety_report
    from app.analytics.status import render_system_status
    from app.config.settings import PROJECT_ROOT

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", type=Path, action="append", help="LOCAL_REAL event JSONL (default data/events/events.jsonl)")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/reports"))
    p.add_argument("--include-synthetic", action="store_true", help="add a separate SYNTHETIC SCENARIO VALIDATION section")
    p.add_argument("--synthetic-dir", type=Path, default=Path("data/simulated"))
    p.add_argument("--no-system-status", action="store_true", help="do not (re)write final_system_status.md")
    p.add_argument("--demo-summary", type=Path, action="append",
                   default=None, help="demo_summary.json files for final_system_status.md (default outputs/demo*/)")
    p.add_argument("--test-results", default=None, help="test-suite result line to record in final_system_status.md")
    args = p.parse_args(argv)

    events_paths = [_abs(e, PROJECT_ROOT) for e in (args.events or [Path("data/events/events.jsonl")])]
    load = load_real_events(events_paths)
    for e in events_paths:
        if str(e) in load.sources:
            load.sources[load.sources.index(str(e))] = _rel(e, PROJECT_ROOT)
    prefix = _rel(events_paths[0].parent, PROJECT_ROOT) + "/" if len(events_paths) == 1 else ""
    syn_dir = _abs(args.synthetic_dir, PROJECT_ROOT)
    synthetic = synthetic_validation(syn_dir) if args.include_synthetic else None
    rep = build_safety_report(load, prefix, synthetic)
    out_dir = _abs(args.out_dir, PROJECT_ROOT)
    paths = write_safety_report(rep, out_dir)

    s = rep["summary"]
    print(f"LOCAL_REAL events: {s['total_events']} (excluded non-real: {dict(load.excluded_non_real) or 0}, "
          f"malformed lines: {load.malformed_lines}, missing files: {load.missing_files or 0})")
    print("by type:", s["events_by_type"])
    print("by risk level:", s["events_by_risk_level"])
    print("GPS:", rep["hotspots"]["geographic"].get("message") or f"{rep['hotspots']['geographic']['points']} real points")
    print("significant event entries:", len(rep["event_entries"]))
    if synthetic is not None:
        print(f"SYNTHETIC (separate): {synthetic.get('rows', 0)} rows")
    for k, v in paths.items():
        print(f"wrote {v}")

    if not args.no_system_status:
        demo_paths = args.demo_summary or sorted((PROJECT_ROOT / "outputs").glob("demo*/demo_summary.json"))
        demos = []
        for dp in demo_paths:
            dp = _abs(Path(dp), PROJECT_ROOT)
            if dp.is_file():
                demos.append((_rel(dp.parent, PROJECT_ROOT), json.loads(dp.read_text(encoding="utf-8"))))
        status = render_system_status(rep, synthetic or synthetic_validation(syn_dir), demos, args.test_results)
        sp = out_dir / "final_system_status.md"
        sp.write_text(status, encoding="utf-8")
        print(f"wrote {sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
