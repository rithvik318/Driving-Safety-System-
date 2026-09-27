"""Simulated-dataset writer (data/simulated/) and the synthetic scenario report.

Files: scenarios.jsonl (always), scenarios.parquet (when pyarrow is installed), schema.json, manifest.json.
The writer refuses: any row that is not SYNTHETIC or fails validation, and any output directory inside the
real event dataset (data/events/). It never reads or writes the real dataset.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from app.config.settings import PROJECT_ROOT
from app.simulation.models import (SIM_FIELD_NAMES, SYNTHETIC, SimulationError, arrow_sim_schema, sim_schema_document,
                                   validate_sim_row)

REAL_DATASET_ROOTS = (PROJECT_ROOT / "data" / "events",)
LEVELS = ("SAFE", "CAUTION", "HIGH", "CRITICAL")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


class SimulationDatasetWriter:
    def __init__(self, root: Path, real_roots: tuple[Path, ...] = REAL_DATASET_ROOTS):
        self.root = Path(root)
        for real in real_roots:
            if _inside(self.root, Path(real)):
                raise SimulationError(f"refusing to write synthetic data inside the real dataset {real}")
        self.jsonl_path = self.root / "scenarios.jsonl"
        self.parquet_path = self.root / "scenarios.parquet"
        self.schema_path = self.root / "schema.json"
        self.manifest_path = self.root / "manifest.json"

    @staticmethod
    def check(rows: list[dict]) -> None:
        for row in rows:
            if row.get("data_source") != SYNTHETIC or row.get("observation_type") != SYNTHETIC:
                raise SimulationError(f"refusing non-SYNTHETIC row {row.get('synthetic_id') or row.get('event_id')!r} "
                                      f"(data_source={row.get('data_source')!r})")
            errors = validate_sim_row(row)
            if errors:
                raise SimulationError(f"invalid row {row.get('synthetic_id')!r}: {errors}")
        ids = [r["event_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise SimulationError("duplicate event_id")

    def write(self, rows: list[dict], manifest: dict) -> dict:
        """Write all files (replacing a previous generation). Returns {file: row count or note}."""
        self.check(rows)
        self.root.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps({k: row[k] for k in SIM_FIELD_NAMES}, ensure_ascii=False) + "\n")
        files = {"scenarios.jsonl": len(rows)}
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pylist([{k: r[k] for k in SIM_FIELD_NAMES} for r in rows], schema=arrow_sim_schema())
            pq.write_table(table, self.parquet_path)
            files["scenarios.parquet"] = pq.read_metadata(self.parquet_path).num_rows
        except ImportError:
            if self.parquet_path.exists():
                self.parquet_path.unlink()
            files["scenarios.parquet"] = "not written: pyarrow not installed (JSONL is complete)"
        self.schema_path.write_text(json.dumps(sim_schema_document(), indent=2), encoding="utf-8")
        files["schema.json"] = "sim-1.0"
        manifest = {**manifest, "files": files}
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        files["manifest.json"] = "written"
        return files


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------------------------------------------------------------ report


def _table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(headers))) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render_report(rows: list[dict], manifest: dict) -> str:
    n = len(rows)
    scen = Counter(r["scenario_name"] for r in rows)
    order = list(dict.fromkeys(r["scenario_name"] for r in rows))
    ts = [r["timestamp"] for r in rows]
    dts = sorted(r["synthetic_datetime"] for r in rows)
    lats = [r["gps_lat"] for r in rows]
    lons = [r["gps_lon"] for r in rows]
    lvl = Counter(r["risk_level"] for r in rows)
    raw_lvl = Counter(r["raw_risk_level"] for r in rows)
    alarm = Counter(r["alarm_recommended"] for r in rows)
    trig = Counter(r["alarm_triggered"] for r in rows)
    haz = Counter(r["hazard_type"] for r in rows)
    ev = Counter(e.split(" ")[0] for r in rows for e in json.loads(r["events_emitted"]))
    inst = manifest["instances"]
    mism = [i for i in inst if not i["expectation_met"]]
    ref = manifest["real_reference"]
    per_scen_lvl = {s: Counter(r["risk_level"] for r in rows if r["scenario_name"] == s) for s in order}
    per_scen_alarm = {s: sum(r["alarm_recommended"] for r in rows if r["scenario_name"] == s) for s in order}
    by_parent = Counter(r["parent_real_event_id"] for r in rows if r["parent_real_event_id"])
    by_dparent = Counter(r["parent_real_driver_event_id"] for r in rows if r["parent_real_driver_event_id"])

    L = []
    L.append("# Synthetic scenario dataset (feature level)\n")
    L.append("> **SYNTHETIC DATA. Not observations.** Every row has `data_source = SYNTHETIC`, `observation_type = SYNTHETIC`, "
             "`timestamp_source = SYNTHETIC`, `gps_source = SYNTHETIC`, `location_source = SYNTHETIC_REFERENCE`. "
             + manifest["statement"] + "\n")
    L.append("Generated by `scripts/generate_synthetic_scenarios.py` (generator "
             f"{manifest['generator_version']}, schema {manifest['schema_version']}, seed {manifest['random_seed']}) into "
             "`data/simulated/` (`scenarios.jsonl`, `scenarios.parquet`, `schema.json`, `manifest.json`). The real event dataset "
             "`data/events/` was only read (sha256 of `events.jsonl` recorded in the manifest) and is unchanged.\n")
    L.append("**How rows are made.**")
    L.append("- The generator constructs image-space detections and per-frame driver signals at a fixed 10 Hz cadence. No video or images exist.")
    L.append("- These inputs are fed through the **existing, unmodified** components: IoUTracker, the driver temporal logic, "
             "RiskEngine (default configuration) and EventRecorder.")
    L.append("- Engine fields are therefore computed from synthetic inputs, and are synthetic too.")
    L.append(f"- The driver logic first receives {manifest['config']['driver_preroll_seconds']:g} s of synthetic driver history "
             "(not emitted as rows), so rolling-window driver features have a baseline at t = 0.")
    L.append("- Expectations were written before the engine ran and were not adjusted to its output.\n")

    L.append("## 1. Total rows\n")
    L.append(f"**{n} rows**: {manifest['scenario_instances']} scenario instances "
             f"({manifest['variants_per_scenario']} variants × {len(order)} scenarios), 5–7 s each at {manifest['cadence_hz']:g} Hz.")
    L.append(f"JSONL rows: {manifest['files'].get('scenarios.jsonl')}; Parquet rows: {manifest['files'].get('scenarios.parquet')}.\n")

    L.append("## 2. Scenario counts\n")
    L.append(_table(["Scenario", "Rows", "Instances", "Duration(s) per instance"],
                    [[s, scen[s], sum(1 for i in inst if i["scenario_name"] == s),
                      ", ".join(f"{i['duration_seconds']:g}" for i in inst if i["scenario_name"] == s)] for s in order]))
    L.append(f"\nSmallest / largest scenario: {min(scen.values())} / {max(scen.values())} rows.\n")

    L.append("## 3. Timestamp range\n")
    L.append(f"- **`timestamp`** (seconds within each instance): {min(ts):g} to {max(ts):g} s, in fixed 0.1 s steps.")
    L.append(f"- **`synthetic_datetime`** (synthetic clock): {dts[0]} to {dts[-1]}.")
    L.append("- Instances are laid out 120 s apart on the synthetic clock.")
    L.append("- These are not capture times, and no synthetic timestamp appears in a LOCAL_REAL record.\n")

    L.append("## 4. Synthetic GPS range\n")
    L.append(f"- **Latitude:** {min(lats):.5f} to {max(lats):.5f}.")
    L.append(f"- **Longitude:** {min(lons):.5f} to {max(lons):.5f}.")
    L.append(f"- **How it's generated:** {manifest['location_policy']}")
    L.append("- Within an instance, the synthetic trace advances a few metres per second only to lay out plausible consecutive points. It is not a speed claim.\n")

    L.append("## 5. Risk-level distribution (existing RiskEngine output)\n")
    L.append(_table(["Level", "Rows (after hysteresis)", "%", "Rows (raw, before hysteresis)"],
                    [[lv, lvl.get(lv, 0), f"{100 * lvl.get(lv, 0) / n:.1f}", raw_lvl.get(lv, 0)] for lv in LEVELS]))
    L.append("\nPer scenario (rows at each level after hysteresis):\n")
    L.append(_table(["Scenario"] + list(LEVELS) + ["Peak (per instance)"],
                    [[s] + [per_scen_lvl[s].get(lv, 0) for lv in LEVELS]
                     + [" / ".join(i["peak_level"] for i in inst if i["scenario_name"] == s)] for s in order]))
    L.append("\nHazard types: " + ", ".join(f"{k} {v}" for k, v in haz.most_common()) + ".\n")
    L.append("Events the existing EventRecorder emitted on the synthetic streams: "
             + (", ".join(f"{k} {v}" for k, v in ev.most_common()) or "none") + ".\n")

    L.append("## 6. Alarm distribution\n")
    L.append(f"- **`alarm_recommended`** (engine recommendation, level ≥ HIGH): True {alarm.get(True, 0)} rows "
             f"({100 * alarm.get(True, 0) / n:.1f} %), False {alarm.get(False, 0)}.")
    L.append(f"- **`alarm_triggered`:** False in all {trig.get(False, 0)} rows. No alarm output exists, and the synthetic rows do not pretend one fired.\n")
    L.append(_table(["Scenario", "alarm_recommended rows"], [[s, per_scen_alarm[s]] for s in order]))
    L.append("")

    L.append("## 7. Parent real-event linkage\n")
    L.append(f"- **Reference pool:** `{ref['path']}`, available: {ref['available']}, {ref['events_used_as_reference_pool']} "
             f"LOCAL_REAL events, run(s) {', '.join(ref['run_ids']) or '—'}. Read-only.")
    L.append(f"- **Parents used:** {manifest['parent_real_event_count']} distinct real events ({', '.join(manifest['parent_real_event_ids'])}).")
    L.append(f"- **Linked rows:** {manifest['parent_linked_records']} rows are linked to at least one parent. "
             f"{manifest['synthetic_only_records_count']} rows are synthetic-only (the `escalating_hazard` scenario is designed without a parent).")
    L.append("- **What a parent does:** it seeds constructed parameters (box area fraction, horizontal position, detector confidence, "
             "growth, lateral speed, drowsiness score), clamped to the range each scenario is meant to test. "
             "A parent link does **not** make a row an observation.")
    L.append("- **Driver parents:** no head-away or hands-off events exist in the real data. Distraction and hands-off driver "
             "signals are therefore synthetic-only; drowsy driver signals use a real drowsiness event as a parent.\n")
    L.append(_table(["Road parent", "Rows"], sorted(by_parent.items())))
    if by_dparent:
        L.append("")
        L.append(_table(["Driver parent", "Rows"], sorted(by_dparent.items())))
    L.append("\nReal value ranges used as references (min / median / max) are in `manifest.json` → `real_reference.value_ranges`.\n")

    L.append("## 8. Expected vs actual behaviour\n")
    L.append(_table(["Instance", "Expected", "Actual peak", "First → final", "Alarm rows", "Met"],
                    [[i["session_id"], "/".join(i["expected_peak_levels"]), i["peak_level"], f"{i['first_level']} → {i['final_level']}",
                      i["alarm_recommended_steps"], "yes" if i["expectation_met"] else "**NO**"] for i in inst]))
    L.append(f"\n**{sum(i['expectation_met'] for i in inst)} of {len(inst)} instances matched the expectation.**\n")

    L.append("## 9. Mismatches\n")
    if not mism:
        L.append("None. Every instance's actual peak level was within the expected set, and the escalation instances started at SAFE.\n")
    else:
        L.append("These are recorded as findings. The risk engine was **not** changed to remove them.\n")
        for i in mism:
            L.append(f"- **`{i['session_id']}`**: {i['mismatch']}. Level counts {i['level_counts']}.")
            for note in _explain(i, rows):
                L.append(f"  - {note}")
        L.append("")

    L.append("## 10. Limitations\n")
    L.append("- **Synthetic, feature level:** there are no pixels, and the detector and hand/face models were never run. "
             "Detector noise and failure modes are only approximated (confidence jitter, label flicker, missed detections).")
    L.append("- **Designed scenarios:** each is a single road object with smooth scripted motion and a scripted driver. "
             "They are not samples of real traffic, and scenario frequencies say nothing about real-world rates.")
    L.append("- **Circular by design:** risk levels come from the same rule engine being tested. Agreement shows the rules behave "
             "as documented on these inputs, **not** that they are correct on real roads. There is no ground truth for crashes or near misses.")
    L.append(f"- **Small reference pool:** parents come from a small real dataset ({ref['events_used_as_reference_pool']} events, "
             "short clips, no GPS, no head-away or hands-off events), so the value ranges are narrow and some driver "
             "behaviours have no real reference.")
    L.append("- **`trajectory_overlap` is design ground truth:** it is image-space overlap with the central band. It is not a "
             "calibrated vehicle path and is not used by the risk engine. It stays null in the real dataset.")
    L.append("- **GPS and time are synthetic reference values** around a generic point. No map, road network or real "
             "location is implied, and there is no link between location and risk.")
    L.append("- **Default engine configuration only:** results change if `RISK_*` / `TRACK_*` / driver settings change.")
    return "\n".join(L) + "\n"


def _explain(inst: dict, rows: list[dict]) -> list[str]:
    """Factual notes from the rows of a mismatching instance (factors and gates at its peak step)."""
    rs = [r for r in rows if r["session_id"] == inst["session_id"]]
    peak = max(rs, key=lambda r: (LEVELS.index(r["risk_level"]), r["raw_risk_score"]))
    factors = ", ".join(f"{f['name']} {f['points']:+g}" for f in json.loads(peak["risk_factors"]))
    gates = json.loads(peak["risk_gates"])
    out = [f"At the peak step (t = {peak['timestamp']:g} s of {rs[-1]['timestamp']:g} s): level {peak['risk_level']} "
           f"(raw level {peak['raw_risk_level']}), raw score {peak['raw_risk_score']:.1f}, hazard {peak['hazard_type']}, "
           f"factors [{factors}]" + (f", gates {gates}" if gates else "") + "."]
    out.append(f"Engine reason: \"{peak['risk_reason']}\"")
    raw_above = [r for r in rs if LEVELS.index(r["raw_risk_level"]) > LEVELS.index(inst["peak_level"])]
    if raw_above:
        out.append(f"The raw (pre-hysteresis) level exceeded the peak on {len(raw_above)} step(s) (first at t = "
                   f"{raw_above[0]['timestamp']:g} s), but not for enough consecutive steps for the hysteresis to confirm it "
                   "before the instance ended.")
    return out
