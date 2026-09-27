"""Post-event safety analytics (app/analytics): aggregation, patterns, GPS handling, recommendations,
real/synthetic separation, robustness and report generation.

Unit tests use small labelled FIXTURE records in the event-record format; the real-data tests read
data/events (read-only) when present.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from app.analytics import (GPS_UNAVAILABLE, build_safety_report, find_patterns, generate_post_event_summary,
                           hotspot_analysis, load_real_events, recommend, render_markdown, summarize_events,
                           synthetic_validation, write_safety_report)
from app.analytics.event_summary import elevated_episodes
from app.analytics.report import REC, event_entry
from app.config.settings import PROJECT_ROOT

REAL = PROJECT_ROOT / "data" / "events" / "events.jsonl"
SIM = PROJECT_ROOT / "data" / "simulated"


def ev(i, etype="RISK_ESCALATED", session="s1", t=1.0, level="CAUTION", prev="SAFE", hazard="PEDESTRIAN_CONFLICT",
       cls="person", source="LOCAL_REAL", camera="front", **kw):
    rec = {"event_id": f"event_{i:06d}", "run_id": "fixture_run", "session_id": session, "source_file": f"{session}.mp4",
           "camera": camera, "data_source": source, "observation_type": "INFERRED", "timestamp": t, "event_type": etype,
           "risk_level": level, "previous_risk_level": prev if etype.startswith("RISK_") else None,
           "transition": f"{prev}->{level}" if etype.startswith("RISK_") else None, "hazard_type": hazard,
           "hazard_class": cls, "smoothed_risk_score": 30.0, "raw_risk_score": 28.0, "alarm_recommended": level in ("HIGH", "CRITICAL"),
           "alarm_triggered": False, "gps_lat": None, "gps_lon": None, "gps_source": "UNAVAILABLE",
           "risk_reason": "fixture reason", "evidence_kind": "FRAME", "evidence_path": f"fixture_run/event_{i:06d}",
           "hazard_persistence": None, "track_persistence_seconds": 1.0, "change_kind": None}
    rec.update(kw)
    return rec


def fixture_events():
    return [
        ev(1, t=0.5), ev(2, level="HIGH", prev="CAUTION", t=1.5),
        ev(3, "PERSISTENT_HAZARD", t=2.5, level="HIGH", prev=None, hazard_persistence=2.0),
        ev(4, "RISK_DEESCALATED", t=4.0, level="SAFE", prev="HIGH", hazard="NONE", cls=None),
        ev(5, session="s2", t=0.3, hazard="APPROACHING_VEHICLE", cls="car"),
        ev(6, session="s2", t=1.2, level="HIGH", prev="CAUTION", hazard="APPROACHING_VEHICLE", cls="car"),
        ev(7, "DRIVER_STATE_CHANGE", session="d1", camera="driver", t=2.0, level=None, hazard=None, cls=None,
           change_kind="drowsiness_level", transition="HIGH->CRITICAL", observation_type="INFERRED"),
        ev(8, "DRIVER_STATE_CHANGE", session="d1", camera="driver", t=3.0, level=None, hazard=None, cls=None,
           change_kind="hand_state", transition="ONE_HAND->NO_HANDS", observation_type="OBSERVED"),
    ]


def write_jsonl(path: Path, rows, extra_lines=()):
    path.write_text("\n".join([json.dumps(r) for r in rows] + list(extra_lines)) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------ aggregation


def test_real_event_aggregation(tmp_path):
    load = load_real_events([write_jsonl(tmp_path / "e.jsonl", fixture_events())])
    s = summarize_events(load.events)
    assert s["total_events"] == 8 and s["sessions"] == 3
    assert s["events_by_type"] == {"RISK_ESCALATED": 4, "PERSISTENT_HAZARD": 1, "RISK_DEESCALATED": 1, "DRIVER_STATE_CHANGE": 2}
    assert s["events_by_camera"] == {"front": 6, "driver": 2}
    assert s["persistent_hazards"]["count"] == 1 and s["persistent_hazards"]["hazard_persistence_seconds"]["max"] == 2.0
    assert s["driver_state_events"]["count"] == 2
    assert s["alarm_recommended"]["true"] == 3
    assert s["smoothed_risk_score"]["max"] == 30.0


def test_risk_distribution(tmp_path):
    s = summarize_events(fixture_events())
    assert s["events_by_risk_level"] == {"SAFE": 1, "CAUTION": 2, "HIGH": 3, "CRITICAL": 0}
    assert s["events_without_risk_level"] == 2
    assert s["risk_transitions"] == {"SAFE->CAUTION": 2, "CAUTION->HIGH": 2, "HIGH->SAFE": 1}
    eps = elevated_episodes(fixture_events())
    closed = [e for e in eps if e["duration_seconds"] is not None]
    assert len(eps) == 2 and len(closed) == 1 and closed[0]["duration_seconds"] == 3.5 and closed[0]["peak_level"] == "HIGH"
    assert [e for e in eps if e["end"] is None][0]["duration_seconds"] is None  # open episode: no invented duration


def test_hazard_distribution_and_patterns():
    s = summarize_events(fixture_events())
    assert s["events_by_hazard_type"]["PEDESTRIAN_CONFLICT"] == 3 and s["events_by_hazard_type"]["APPROACHING_VEHICLE"] == 2
    p = find_patterns(fixture_events())
    assert p["pedestrian_hazard"]["events"] == 3 and p["pedestrian_hazard"]["sessions_with_repeats"] == 1
    assert p["approaching_vehicle_hazard"]["max_in_one_session"] == 2
    assert p["animal_hazard"]["events"] == 0 and not p["animal_hazard"]["repeated"]
    assert p["driver_state_change"]["repeated"]


# ------------------------------------------------------------------------------------ GPS


def test_missing_gps_handling():
    h = hotspot_analysis(fixture_events())
    assert h["geographic"] == {"available": False, "message": GPS_UNAVAILABLE, "points": 0}
    assert GPS_UNAVAILABLE == "GPS hotspot analysis unavailable because no real GPS observations were collected."
    assert "NOT geographic" in h["non_geographic"]["label"]
    assert h["non_geographic"]["events_by_source_file"]["s1.mp4"] == 4


def test_no_fabricated_location(tmp_path):
    rows = fixture_events()
    # synthetic coordinates in a real-looking record are not accepted as real GPS
    rows.append(ev(9, gps_lat=28.6, gps_lon=77.2, gps_source="SYNTHETIC"))
    h = hotspot_analysis(rows)
    assert not h["geographic"]["available"]
    rep = build_safety_report(load_real_events([write_jsonl(tmp_path / "e.jsonl", fixture_events())]))
    text = json.dumps(rep)
    assert '"lat"' not in text and "event_counts_by_cell" not in text
    md = render_markdown(rep)
    assert GPS_UNAVAILABLE in md
    # real GPS, when present, is only counted descriptively
    real = [ev(20, gps_lat=26.19, gps_lon=91.69, gps_source="phone_gps_log"), ev(21, gps_lat=26.19, gps_lon=91.69, gps_source="phone_gps_log")]
    g = hotspot_analysis(real)["geographic"]
    assert g["available"] and g["points"] == 2 and "not a measure of objective danger" in g["note"]


# ------------------------------------------------------------------------------------ recommendations


def test_recommendation_generation():
    e = {x["event_id"]: x for x in fixture_events()}
    assert recommend(e["event_000003"]) == ("pedestrian", REC["pedestrian"])
    assert REC["pedestrian"] == "Review pedestrian interaction and maintain additional visual attention in similar situations."
    assert recommend(e["event_000006"])[1] == "Review following/approach context and maintain additional observation of closing traffic."
    assert recommend(e["event_000007"])[1] == ("Treat prolonged eye closure as a driver-attention warning; review "
                                               "rest/alertness before continued driving.")
    assert recommend(e["event_000008"])[1] == "Review the driving segment and maintain both hands on the steering wheel when appropriate."
    assert recommend(e["event_000004"])[0] == "deescalation"
    assert recommend(ev(30, hazard="DRIVER_DROWSINESS", cls=None))[0] == "drowsiness"
    assert recommend(ev(31, "DRIVER_STATE_CHANGE", change_kind="hand_state", transition="NO_HANDS->ONE_HAND"))[0] == "hand_info"
    assert recommend(ev(32, hazard="ANIMAL_HAZARD", cls="dog"))[0] == "animal"
    assert recommend(ev(33, hazard="COMBINED_DRIVER_HAZARD"))[0] == "combined"
    for text in REC.values():  # no diagnoses, blame or collision claims
        low = text.lower()
        assert "diagnos" not in low and "fault" not in low and "collision" not in low and "you caused" not in low
    assert recommend(fixture_events()[0]) == recommend(fixture_events()[0])  # deterministic


def test_post_event_summary_hook_is_offline_by_default(monkeypatch):
    monkeypatch.delenv("SAFETY_REPORT_LLM", raising=False)
    out = generate_post_event_summary(fixture_events())
    assert out["backend"] == "deterministic" and out["text"] == out["deterministic_text"]
    assert "not collision predictions" in out["text"]
    custom = generate_post_event_summary(fixture_events(), backend=lambda prompt: "rephrased")
    assert custom["text"] == "rephrased" and custom["deterministic_text"] == out["text"]

    def broken(prompt):
        raise RuntimeError("offline")
    assert generate_post_event_summary(fixture_events(), backend=broken)["backend"] == "deterministic"
    monkeypatch.setenv("SAFETY_REPORT_LLM", "anthropic")
    monkeypatch.delenv("SAFETY_REPORT_LLM_MODEL", raising=False)
    assert generate_post_event_summary(fixture_events())["backend"] == "deterministic"  # no model/key -> offline


# ------------------------------------------------------------------------------------ separation and robustness


def test_real_synthetic_separation(tmp_path):
    rows = fixture_events() + [ev(40, source="SYNTHETIC"), ev(41, source="SYNTHETIC_COMBINATION")]
    load = load_real_events([write_jsonl(tmp_path / "e.jsonl", rows)])
    assert len(load.events) == 8 and load.excluded_non_real == {"SYNTHETIC": 1, "SYNTHETIC_COMBINATION": 1}
    assert all(e["data_source"] == "LOCAL_REAL" for e in load.events)
    rep = build_safety_report(load)
    assert rep["synthetic_scenario_validation"] == {"included": False}
    assert "SYNTHETIC SCENARIO VALIDATION" not in render_markdown(rep)
    if not (SIM / "scenarios.jsonl").is_file():
        pytest.skip("no synthetic dataset")
    syn = synthetic_validation(SIM)
    rep2 = build_safety_report(load, synthetic=syn)
    assert rep2["summary"] == rep["summary"]  # real counts unchanged by including synthetic
    md = render_markdown(rep2)
    assert md.index("## SYNTHETIC SCENARIO VALIDATION") > md.index("## 6. Post-event summary")
    paths = write_safety_report(rep2, tmp_path / "out")
    with open(paths["csv"], newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["scope"] for r in rows} == {"REAL", "SYNTHETIC"}
    real_total = [r for r in rows if r["scope"] == "REAL" and r["category"] == "total" and r["key"] == "events"][0]
    assert int(real_total["value"]) == 8


def test_empty_event_dataset(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    load = load_real_events([empty, tmp_path / "missing.jsonl"])
    assert load.events == [] and load.missing_files
    rep = build_safety_report(load)
    assert rep["summary"]["total_events"] == 0 and rep["event_entries"] == []
    assert rep["summary"]["smoothed_risk_score"] is None
    md = render_markdown(rep)
    assert "No LOCAL_REAL events found." in md and GPS_UNAVAILABLE in md
    assert "nothing to summarise" in rep["post_event_summary"]["text"]
    write_safety_report(rep, tmp_path / "out")


def test_malformed_and_missing_optional_fields(tmp_path):
    minimal = {"event_id": "event_000100", "event_type": "PERSISTENT_HAZARD", "data_source": "LOCAL_REAL"}
    weird = ev(101, smoothed_risk_score="n/a", timestamp=None, hazard_persistence=float("nan"))
    path = write_jsonl(tmp_path / "e.jsonl", [minimal, weird], extra_lines=["{not json", "[1, 2]", '{"event_id": "x"}'])
    load = load_real_events([path])
    assert load.malformed_lines == 3 and len(load.events) == 2
    rep = build_safety_report(load)
    s = rep["summary"]
    assert s["total_events"] == 2 and s["smoothed_risk_score"] is None
    entry = event_entry(minimal)
    assert entry["time"] == "not recorded" and entry["camera"] == "not recorded" and entry["evidence_path"] == "none recorded"
    assert entry["risk"] == "not recorded" and entry["reason"] == "not recorded"
    render_markdown(rep)
    # duplicates (same run + event id) are counted once
    load2 = load_real_events([write_jsonl(tmp_path / "d.jsonl", [ev(1), ev(1)])])
    assert len(load2.events) == 1 and load2.duplicates == 1


# ------------------------------------------------------------------------------------ report generation


def test_report_generation(tmp_path):
    load = load_real_events([write_jsonl(tmp_path / "e.jsonl", fixture_events())])
    rep = build_safety_report(load, evidence_prefix="data/events/")
    paths = write_safety_report(rep, tmp_path / "out")
    md = paths["md"].read_text()
    for h in ("## 1. Event summary", "## 2. Hazard patterns", "## 3. Hotspot analysis", "## 4. Post-event recommendations",
              "## 5. Event report", "## 6. Post-event summary", "## 7. Limitations"):
        assert h in md
    assert "`data/events/fixture_run/event_000003`" in md
    data = json.loads(paths["json"].read_text())
    ids = [e["event_id"] for e in data["event_entries"]]
    assert ids == ["event_000002", "event_000003", "event_000007", "event_000008", "event_000006"] or set(ids) == {
        "event_000002", "event_000003", "event_000006", "event_000007", "event_000008"}
    for e in data["event_entries"]:
        assert set(("event_id", "time", "camera", "risk", "hazard", "reason", "evidence_path", "recommendation")) <= set(e)
    with open(paths["csv"], newline="") as fh:
        assert {r["scope"] for r in csv.DictReader(fh)} == {"REAL"}


def test_real_dataset_report_and_script(tmp_path, capsys):
    if not REAL.is_file():
        pytest.skip("no real event dataset")
    before = hashlib.sha256(REAL.read_bytes()).hexdigest()
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_report_cli", PROJECT_ROOT / "scripts" / "generate_safety_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--out-dir", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "safety_summary.json").read_text())
    real = [json.loads(line) for line in REAL.read_text().splitlines() if line.strip()]
    assert data["summary"]["total_events"] == sum(r["data_source"] == "LOCAL_REAL" for r in real)
    assert data["hotspots"]["geographic"]["message"] == GPS_UNAVAILABLE
    assert data["synthetic_scenario_validation"] == {"included": False}
    status = (tmp_path / "final_system_status.md").read_text()
    for sec in "ABCDEFGHIJKL":
        assert f"## {sec}. " in status
    assert "No real driver + road recording" in status or "no synchronized" in status.lower()
    assert hashlib.sha256(REAL.read_bytes()).hexdigest() == before  # read-only
    assert "LOCAL_REAL events" in capsys.readouterr().out
