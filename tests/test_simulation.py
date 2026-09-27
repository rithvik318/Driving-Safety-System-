"""Synthetic scenario generator (app/simulation): provenance, determinism, schema, separation from real data.

Everything generated here is SYNTHETIC. The real event dataset (data/events/) is only read.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from app.config.settings import PROJECT_ROOT
from app.simulation import (SCENARIO_NAMES, SIM_FIELD_NAMES, SimulationConfig, SimulationDatasetWriter, SimulationError,
                            SyntheticScenarioGenerator, build_manifest, load_real_reference, read_rows, render_report,
                            validate_sim_row)
from app.simulation.models import REQUESTED_FIELDS, resolve_field
from app.simulation.scenarios import BY_NAME

REAL_EVENTS_DIR = PROJECT_ROOT / "data" / "events"
REAL_EVENTS = REAL_EVENTS_DIR / "events.jsonl"


def tree_hash(root: Path) -> dict:
    if not root.exists():
        return {}
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def fixture_reference(tmp: Path) -> Path:
    """Tiny labelled FIXTURE file in the real-event format (not project data), incl. a SYNTHETIC decoy."""
    base = {"run_id": "fixture_run", "camera": "front", "image_width": 1920, "image_height": 1080,
            "box_growth_per_second": 0.5, "lateral_toward_center_fw_per_s": 0.08, "hazard_confidence": 0.8,
            "drowsiness_level": None, "drowsiness_score": None}
    rows = [
        {**base, "event_id": "event_000001", "data_source": "LOCAL_REAL", "hazard_type": "APPROACHING_VEHICLE",
         "hazard_class": "car", "bbox_area": 60000.0, "center_x": 960.0},
        {**base, "event_id": "event_000002", "data_source": "LOCAL_REAL", "hazard_type": "PEDESTRIAN_CONFLICT",
         "hazard_class": "person", "bbox_area": 40000.0, "center_x": 700.0},
        {**base, "event_id": "event_000003", "data_source": "LOCAL_REAL", "camera": "driver", "hazard_type": "DRIVER_DROWSINESS",
         "hazard_class": None, "bbox_area": None, "center_x": None, "drowsiness_level": "HIGH", "drowsiness_score": 0.7},
        {**base, "event_id": "event_000099", "data_source": "SYNTHETIC_COMBINATION", "hazard_type": "APPROACHING_VEHICLE",
         "hazard_class": "car", "bbox_area": 60000.0, "center_x": 960.0},
    ]
    path = tmp / "fixture_events.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """Default generation (project reference if present) written to a temporary data/simulated."""
    before = tree_hash(REAL_EVENTS_DIR)
    result = SyntheticScenarioGenerator(SimulationConfig()).generate()
    out = tmp_path_factory.mktemp("simulated")
    manifest = build_manifest(result)
    files = SimulationDatasetWriter(out).write(result.rows, manifest)
    return {"result": result, "out": out, "files": files, "manifest": json.loads((out / "manifest.json").read_text()),
            "before": before, "after": tree_hash(REAL_EVENTS_DIR)}


# ------------------------------------------------------------------------------------ size, determinism


def test_at_least_500_rows(generated):
    assert len(generated["result"].rows) >= 500
    assert 1000 <= len(generated["result"].rows) <= 1600  # target range ~1,000-1,500


def test_generation_is_deterministic(generated):
    again = SyntheticScenarioGenerator(SimulationConfig()).generate()
    assert again.rows == generated["result"].rows
    other = SyntheticScenarioGenerator(SimulationConfig(seed=7)).generate()
    assert [r["hazard_confidence"] for r in other.rows] != [r["hazard_confidence"] for r in again.rows]


def test_scenario_distribution_non_empty_and_balanced(generated):
    counts = Counter(r["scenario_name"] for r in generated["result"].rows)
    assert set(counts) == set(SCENARIO_NAMES) and len(counts) == 12
    assert all(n > 0 for n in counts.values())
    assert max(counts.values()) <= 2 * min(counts.values())
    assert generated["manifest"]["scenario_counts"] == dict(counts)


# ------------------------------------------------------------------------------------ provenance


def test_all_rows_have_synthetic_provenance(generated):
    for r in generated["result"].rows:
        assert r["data_source"] == "SYNTHETIC" and r["observation_type"] == "SYNTHETIC"
        assert r["timestamp_source"] == "SYNTHETIC"
        assert r["synthetic_id"].startswith("SYN-") and r["event_id"].startswith("sim_")
        assert r["scenario_name"] and r["scenario_description"] and r["synthetic_generation_reason"]
        assert r["alarm_triggered"] is False
        assert "SYNTHETIC" in json.loads(r["notes"])[0]


def test_no_local_real_rows_written(generated):
    rows = read_rows(generated["out"] / "scenarios.jsonl")
    assert rows and not any(r["data_source"] in ("LOCAL_REAL", "PUBLIC") for r in rows)
    assert {r["data_source"] for r in rows} == {"SYNTHETIC"}
    pq = pytest.importorskip("pyarrow.parquet")
    table = pq.read_table(generated["out"] / "scenarios.parquet", columns=["data_source", "observation_type"])
    assert set(table.column("data_source").to_pylist()) == {"SYNTHETIC"}
    assert set(table.column("observation_type").to_pylist()) == {"SYNTHETIC"}


def test_writer_refuses_real_rows_and_real_dataset_path(generated, tmp_path):
    row = dict(generated["result"].rows[0])
    with pytest.raises(SimulationError):
        SimulationDatasetWriter(tmp_path / "x").write([{**row, "data_source": "LOCAL_REAL"}], {})
    with pytest.raises(SimulationError):
        SimulationDatasetWriter(tmp_path / "x").write([{**row, "observation_type": "INFERRED"}], {})
    with pytest.raises(SimulationError):
        SimulationDatasetWriter(REAL_EVENTS_DIR)
    with pytest.raises(SimulationError):
        SimulationDatasetWriter(REAL_EVENTS_DIR / "simulated")
    assert not (tmp_path / "x" / "scenarios.jsonl").exists()


def test_synthetic_gps_is_explicitly_marked(generated):
    cfg = generated["result"].config
    for r in generated["result"].rows:
        assert r["gps_source"] == "SYNTHETIC" and r["location_source"] == "SYNTHETIC_REFERENCE"
        assert r["gps_lat"] is not None and r["gps_lon"] is not None
        assert abs(r["gps_lat"] - cfg.base_lat) < 0.05 and abs(r["gps_lon"] - cfg.base_lon) < 0.05
    assert "Not a collection location" in generated["manifest"]["location_policy"]


def test_validation_rejects_bad_rows(generated):
    row = dict(generated["result"].rows[0])
    assert validate_sim_row(row) == []
    assert validate_sim_row({**row, "gps_source": "UNAVAILABLE"})
    assert validate_sim_row({**row, "location_source": "GPS"})
    assert validate_sim_row({**row, "event_id": "event_000001"})  # must not look like a real id
    assert validate_sim_row({**row, "parent_real_event_id": "sim_000001"})
    assert validate_sim_row({**row, "alarm_triggered": True})
    assert validate_sim_row({**row, "actual_risk_level": "CRITICAL" if row["risk_level"] != "CRITICAL" else "SAFE"})
    assert validate_sim_row({k: v for k, v in row.items() if k != "scenario_name"})


# ------------------------------------------------------------------------------------ time and temporal patterns


def test_timestamps_ordered_at_fixed_cadence_within_each_scenario(generated):
    by = {}
    for r in generated["result"].rows:
        by.setdefault(r["session_id"], []).append(r)
    for rows in by.values():
        ts = [r["timestamp"] for r in rows]
        assert ts[0] == 0.0
        assert all(abs((b - a) - 0.1) < 1e-6 for a, b in zip(ts, ts[1:]))
        assert [r["frame_index"] for r in rows] == list(range(len(rows)))
        assert 5.0 <= len(rows) / 10 <= 7.0 + 1e-9
        dts = [r["synthetic_datetime"] for r in rows]
        assert dts == sorted(dts)


def _session(generated, name):
    return [r for r in generated["result"].rows if r["session_id"] == name]


def test_temporal_patterns(generated):
    # approaching vehicle: constructed area grows (until the documented cap)
    rows = _session(generated, "attentive_approaching_vehicle__v01")
    areas = [r["bbox_area"] for r in rows]
    assert areas[-1] > 2 * areas[0]
    # hazard persistence increases while the hazard is reported
    pers = [r["hazard_persistence"] for r in rows]
    assert any(b > a for a, b in zip(pers, pers[1:]))
    assert all(b >= a or b == 0.0 for a, b in zip(pers, pers[1:]))
    # pedestrian crossing: trajectory overlap rises from 0 toward 1
    ped = _session(generated, "distracted_pedestrian_crossing__v01")
    ov = [r["trajectory_overlap"] for r in ped]
    assert ov[0] == 0.0 and max(ov) == 1.0
    # driver-state durations increase while the state persists
    ho = _session(generated, "hands_off_approaching_vehicle__v01")
    dd = [r["distraction_duration"] for r in ho if r["driver_activity"] == "HANDS_OFF_WHEEL"]
    assert dd and all(b > a for a, b in zip(dd, dd[1:]))
    head = [r["head_away_duration"] for r in ped if r["head_away_duration"] > 0]
    assert head and all(b > a for a, b in zip(head, head[1:]))


def test_existing_driver_logic_produces_driver_states(generated):
    drowsy = [r for r in generated["result"].rows if r["scenario_name"] == "drowsy_approaching_vehicle"]
    assert any(r["drowsiness_level"] in ("HIGH", "CRITICAL") for r in drowsy)
    hands = [r for r in generated["result"].rows if r["scenario_name"] == "hands_off_approaching_vehicle"]
    # HANDS_OFF_WHEEL is inferred only after >= 2 s of NO_HANDS (existing rule)
    assert any(r["driver_activity"] == "HANDS_OFF_WHEEL" for r in hands)
    assert all(r["manual_state_duration"] >= 2.0 - 1e-6 for r in hands if r["driver_activity"] == "HANDS_OFF_WHEEL")
    one = [r for r in generated["result"].rows if r["scenario_name"] == "one_hand_approaching_vehicle"]
    assert all(r["driver_activity"] != "HANDS_OFF_WHEEL" for r in one)


# ------------------------------------------------------------------------------------ parents


def test_parent_ids_point_only_to_real_event_ids(generated):
    ref = generated["result"].reference
    rows = generated["result"].rows
    parents = {r[k] for r in rows for k in ("parent_real_event_id", "parent_real_driver_event_id") if r[k]}
    if not ref.available:
        assert not parents
        pytest.skip("no real event dataset in this checkout")
    real = {json.loads(line)["event_id"]: json.loads(line) for line in REAL_EVENTS.read_text().splitlines() if line.strip()}
    assert parents and parents <= set(real)
    assert all(real[p]["data_source"] == "LOCAL_REAL" for p in parents)
    for r in rows:
        if r["parent_real_event_id"] or r["parent_real_driver_event_id"]:
            assert r["parent_real_run_id"] == real[r["parent_real_event_id"] or r["parent_real_driver_event_id"]]["run_id"]
    assert generated["manifest"]["parent_real_event_count"] == len(parents)
    assert generated["manifest"]["synthetic_only_records_count"] == sum(
        1 for r in rows if not (r["parent_real_event_id"] or r["parent_real_driver_event_id"]))


def test_parents_from_fixture_ignore_synthetic_records(tmp_path):
    path = fixture_reference(tmp_path)
    ref = load_real_reference(path)
    assert {e["event_id"] for e in ref.events} == {"event_000001", "event_000002", "event_000003"}
    res = SyntheticScenarioGenerator(SimulationConfig(real_events_path=path, variants_per_scenario=1)).generate()
    parents = {r[k] for r in res.rows for k in ("parent_real_event_id", "parent_real_driver_event_id") if r[k]}
    assert parents and parents <= {"event_000001", "event_000002", "event_000003"}
    assert "event_000099" not in parents
    drowsy = [r for r in res.rows if r["scenario_name"] == "drowsy_approaching_vehicle"]
    assert {r["parent_real_driver_event_id"] for r in drowsy} == {"event_000003"}


def test_without_real_reference_rows_are_synthetic_only(tmp_path):
    res = SyntheticScenarioGenerator(SimulationConfig(real_events_path=tmp_path / "missing.jsonl",
                                                      variants_per_scenario=1)).generate()
    assert res.rows and all(r["parent_real_event_id"] is None and r["parent_real_driver_event_id"] is None for r in res.rows)
    assert build_manifest(res)["synthetic_only_records_count"] == len(res.rows)


# ------------------------------------------------------------------------------------ schema and files


def test_required_schema_fields_exist(generated):
    schema = json.loads((generated["out"] / "schema.json").read_text())
    names = [f["name"] for f in schema["fields"]]
    assert names == list(SIM_FIELD_NAMES)
    for requested in REQUESTED_FIELDS:
        assert resolve_field(requested) in names, requested
    assert schema["requested_name_mapping"] == {"object_id": "primary_track_id", "approach_rate": "box_growth_per_second"}
    for r in generated["result"].rows:
        assert list(r) == list(SIM_FIELD_NAMES)
    prov = {f["provenance"] for f in schema["fields"]}
    assert prov == {"METADATA", "SYNTHETIC_INPUT", "ENGINE_ON_SYNTHETIC", "VALIDATION"}


def test_reused_field_names_match_event_schema():
    from app.events.schema import BY_NAME as EVENT
    from app.simulation.models import SIM_FIELDS

    for f in SIM_FIELDS:
        if f.reused_from_event_schema:
            assert f.name in EVENT and f.type == EVENT[f.name].type


def test_jsonl_and_parquet_row_counts_match(generated):
    n = len(read_rows(generated["out"] / "scenarios.jsonl"))
    pq = pytest.importorskip("pyarrow.parquet")
    assert pq.read_metadata(generated["out"] / "scenarios.parquet").num_rows == n == len(generated["result"].rows)
    assert generated["manifest"]["files"]["scenarios.jsonl"] == n
    assert generated["manifest"]["files"]["scenarios.parquet"] == n
    assert generated["manifest"]["total_records"] == n


def test_manifest_contents(generated):
    m = generated["manifest"]
    for key in ("generation_timestamp", "generator_version", "random_seed", "total_records", "scenario_counts",
                "parent_real_event_count", "synthetic_only_records_count", "fields_generated", "provenance_policy", "statement"):
        assert key in m, key
    assert "does not replace real observations" in m["statement"]
    assert m["random_seed"] == 42 and m["fields_generated"] == list(SIM_FIELD_NAMES)


# ------------------------------------------------------------------------------------ validation vs engine


def test_expected_vs_actual_recorded(generated):
    rows = generated["result"].rows
    for r in rows:
        assert r["expected_behavior"] and json.loads(r["expected_peak_levels"])
        assert r["actual_risk_level"] == r["risk_level"]
    for inst in generated["result"].instances:
        spec = BY_NAME[inst.scenario_name]
        assert inst.expected_peak_levels == spec.expected_levels  # expectations come from the spec, not the engine
        assert inst.expectation_met == (inst.mismatch is None)
        inst_rows = [r for r in rows if r["session_id"] == inst.session_id]
        assert {r["instance_peak_risk_level"] for r in inst_rows} == {inst.peak_level}
        assert {r["expectation_met"] for r in inst_rows} == {inst.expectation_met}
    assert generated["manifest"]["validation"]["instances"] == len(generated["result"].instances)


def test_engine_outputs_are_bounded(generated):
    for r in generated["result"].rows:
        assert 0.0 <= r["raw_risk_score"] <= 100.0 and 0.0 <= r["risk_score"] <= 100.0
        assert r["risk_score"] == r["smoothed_risk_score"]
        assert r["alarm_recommended"] == (r["risk_level"] in ("HIGH", "CRITICAL"))


def test_report_sections(generated):
    text = render_report(generated["result"].rows, generated["manifest"])
    for i, title in enumerate(["Total rows", "Scenario counts", "Timestamp range", "Synthetic GPS range",
                               "Risk-level distribution", "Alarm distribution", "Parent real-event linkage",
                               "Expected vs actual behaviour", "Mismatches", "Limitations"], 1):
        assert f"## {i}. {title}" in text
    assert "SYNTHETIC DATA. Not observations." in text


# ------------------------------------------------------------------------------------ real data untouched


def test_generation_does_not_modify_real_events(generated):
    assert generated["before"] == generated["after"]


def test_script_does_not_modify_real_events(tmp_path, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_sim_cli", PROJECT_ROOT / "scripts" / "generate_synthetic_scenarios.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main = module.main
    before = tree_hash(REAL_EVENTS_DIR)
    assert main(["--variants", "1", "--out", str(tmp_path / "sim"), "--report", str(tmp_path / "r.md")]) == 0
    assert tree_hash(REAL_EVENTS_DIR) == before
    assert (tmp_path / "sim" / "scenarios.jsonl").is_file() and (tmp_path / "r.md").is_file()
    assert "SYNTHETIC rows" in capsys.readouterr().out


def test_config_validation():
    from app.simulation.models import validate_config

    with pytest.raises(SimulationError):
        validate_config(replace(SimulationConfig(), variants_per_scenario=0))
    with pytest.raises(SimulationError):
        validate_config(replace(SimulationConfig(), location_jitter_deg=1.0))
    with pytest.raises(SimulationError):
        validate_config(replace(SimulationConfig(), duration_range_seconds=(7.0, 5.0)))
