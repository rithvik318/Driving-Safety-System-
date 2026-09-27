"""Generate the SYNTHETIC feature-level scenario dataset and its report.

    python scripts/generate_synthetic_scenarios.py [--seed 42] [--variants 2] [--out data/simulated]
                                                   [--report outputs/reports/synthetic_scenario_report.md]

Writes data/simulated/{scenarios.jsonl, scenarios.parquet, schema.json, manifest.json} and the report.
Reads data/events/events.jsonl (read-only) for parent links and value ranges. Never writes to data/events/.
No video, images, models or GPS hardware are used; every row is SYNTHETIC.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python scripts/...`

from app.config.settings import PROJECT_ROOT  # noqa: E402
from app.simulation import (SimulationConfig, SimulationDatasetWriter, SyntheticScenarioGenerator,  # noqa: E402
                            build_manifest, render_report)


def _path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else PROJECT_ROOT / q


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = SimulationConfig()
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--variants", type=int, default=d.variants_per_scenario, help="instances per scenario")
    ap.add_argument("--out", default="data/simulated")
    ap.add_argument("--report", default="outputs/reports/synthetic_scenario_report.md")
    ap.add_argument("--real-events", default="data/events/events.jsonl", help="real event dataset (read-only reference)")
    ap.add_argument("--base-lat", type=float, default=d.base_lat, help="synthetic GPS reference latitude")
    ap.add_argument("--base-lon", type=float, default=d.base_lon, help="synthetic GPS reference longitude")
    args = ap.parse_args(argv)

    cfg = SimulationConfig(seed=args.seed, variants_per_scenario=args.variants, base_lat=args.base_lat,
                           base_lon=args.base_lon, real_events_path=_path(args.real_events))
    result = SyntheticScenarioGenerator(cfg).generate()
    manifest = build_manifest(result)
    out = _path(args.out)
    files = SimulationDatasetWriter(out).write(result.rows, manifest)
    manifest["files"] = files
    report = _path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(result.rows, manifest), encoding="utf-8")

    rows = result.rows
    print(f"SYNTHETIC rows: {len(rows)}  (seed {cfg.seed}, {len(result.instances)} instances)")
    for name, n in Counter(r["scenario_name"] for r in rows).items():
        print(f"  {name:36s} {n}")
    print("risk levels:", dict(Counter(r["risk_level"] for r in rows)))
    print("alarm_recommended:", dict(Counter(r["alarm_recommended"] for r in rows)))
    met = sum(i.expectation_met for i in result.instances)
    print(f"expectation met: {met}/{len(result.instances)}")
    for i in result.instances:
        if not i.expectation_met:
            print(f"  MISMATCH {i.session_id}: {i.mismatch}")
    print("files:", {k: v for k, v in files.items()})
    print("output:", out)
    print("report:", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
