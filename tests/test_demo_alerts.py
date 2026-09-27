"""Integrated demo runner + safety-alert layer.

A, E-H run the real pipeline on a REAL local front-camera video (skipped when the dataset or the YOLO weights are
not available). B-D feed the EXISTING tracker + risk engine with constructed, SYNTHETIC test input (never written to
any dataset) to exercise each alert level; D (CRITICAL) uses synthetic input only.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from app.alerts import AlertManager, AlertSeverity, ConsoleSink, TerminalBell, format_alert, format_event
from app.alerts.models import DISCLAIMER, HAZARD_PHRASES, headline
from app.config import load_road_config, load_settings
from app.config.settings import PROJECT_ROOT
from app.driver.models import (DriverActivity, DriverFrameObservation, DriverState, DriverTemporalState, DrowsinessLevel,
                               FaceStatus, HandState, HandStateResult)
from app.risk import RiskEngine, RiskSnapshot
from app.road.models import Detection
from app.tracking import IoUTracker

W, H = 1920, 1080
REAL_EVENTS_DIR = PROJECT_ROOT / "data" / "events"


# ------------------------------------------------------------------------------------ SYNTHETIC test input


def det(cls, cx, cy, w, h, conf=0.85):
    return Detection(0.0, cls, 0, conf, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, "synthetic_test")


def driver(t, drowsy="LOW", activity="NORMAL", hand="BOTH_HANDS", dist=0.0):
    obs = DriverFrameObservation(timestamp=t, face_status=FaceStatus.OK, face_detected=True, landmarks_available=True,
                                 head_yaw=0.0, head_pitch=0.0)
    temp = DriverTemporalState(drowsiness_level=DrowsinessLevel(drowsy), drowsiness_score={"LOW": 0.1, "HIGH": 0.6, "CRITICAL": 0.9}[drowsy],
                               distraction_duration=dist)
    return DriverState(timestamp=t, observation=obs, temporal=temp, hand=HandStateResult(hand_state=HandState(hand)),
                       observation_quality=1.0, driver_activity=DriverActivity(activity))


def run_engine(frames, driver_fn=None, manager=None):
    """Existing tracker + risk engine on SYNTHETIC frames; alerts from the AlertManager."""
    tracker, engine = IoUTracker(), RiskEngine()
    manager = manager or AlertManager(data_source="SYNTHETIC")
    out = []
    for i, dets in enumerate(frames):
        t = round(i * 0.1, 6)
        tracks = tracker.update(dets, t, i)
        a = engine.evaluate(RiskSnapshot(t, tuple(tracks), driver_fn(t) if driver_fn else None, W, H))
        out.append((a, manager.update(a)))
    return out, manager


static_central_car = [[det("car", 960 + (i % 2), 700, 300, 220)] for i in range(40)]
approaching_car = [[det("car", 960, 700, 200 * (1 + 0.08 * i), 150 * (1 + 0.08 * i))] for i in range(25)]


# ------------------------------------------------------------------------------------ B / C / D alert levels


def test_B_caution_alert_is_visual_log_only():
    results, m = run_engine(static_central_car)
    levels = {a.risk_level.value for a, _ in results}
    assert "CAUTION" in levels and "HIGH" not in levels
    alerts = [al for _, als in results for al in als]
    caution = [al for al in alerts if al.risk_level == "CAUTION" and al.kind == "ESCALATION"]
    assert caution
    al = caution[0]
    assert al.severity is AlertSeverity.NOTICE and al.channels == ("visual", "log") and not al.alarm
    assert al.headline == "CAUTION: hazard noted"
    a = next(a for a, als in results if al in als)
    assert al.risk_reason == a.reason and al.risk_score == a.risk_score  # the engine's own explanation, unchanged
    assert m.summary()["alarms"] == 0


def test_C_high_alert_is_road_hazard_warning_with_alarm():
    results, m = run_engine(approaching_car)
    high = [al for _, als in results for al in als if al.risk_level == "HIGH"]
    assert high, [a.risk_level.value for a, _ in results]
    al = high[0]
    assert al.severity is AlertSeverity.WARNING and al.headline == "WARNING: ROAD HAZARD"
    assert al.alarm and "audible" in al.channels
    a = next(a for a, als in results if al in als)
    assert a.alarm_recommended and al.risk_reason == a.reason and al.hazard_type == "APPROACHING_VEHICLE"
    lines = format_alert(al)
    assert lines[0].endswith("HIGH — approaching vehicle / increasing image-space box")
    assert any(line.endswith("] ALARM \u2014 WARNING: ROAD HAZARD") for line in lines)


def test_D_critical_alert_from_synthetic_input_only():
    """SYNTHETIC test input (constructed car + DriverState): driver + road combination -> CRITICAL."""
    manager = AlertManager(data_source="SYNTHETIC")
    buf = io.StringIO()
    manager.sinks.append(ConsoleSink(buf))
    bell = TerminalBell(enabled=False)
    manager.sinks.append(bell)
    results, _ = run_engine(approaching_car, lambda t: driver(t, drowsy="CRITICAL"), manager)
    crit = [al for _, als in results for al in als if al.risk_level == "CRITICAL"]
    assert crit
    al = crit[0]
    assert al.severity is AlertSeverity.CRITICAL and al.headline == "CRITICAL: IMMEDIATE ATTENTION REQUIRED"
    assert al.alarm and "audible_strong" in al.channels and al.data_source == "SYNTHETIC"
    assert al.hazard_type == "COMBINED_DRIVER_HAZARD" and "rowsiness" in al.risk_reason
    assert "CRITICAL — driver + road hazard combination" in buf.getvalue()
    assert "ALARM — CRITICAL: IMMEDIATE ATTENTION REQUIRED" in buf.getvalue()
    assert bell.rings >= 3  # counted, not sounded (bell disabled)


def test_safe_gives_no_alert_and_deescalation_has_no_alarm():
    results, m = run_engine([[]] * 10)
    assert all(not als for _, als in results)
    m2 = AlertManager()
    esc = m2.update_from_event({"event_type": "RISK_ESCALATED", "risk_level": "HIGH", "previous_risk_level": "CAUTION",
                                "timestamp": 1.0, "risk_reason": "r", "hazard_type": "PEDESTRIAN_CONFLICT",
                                "alarm_recommended": True, "event_id": "event_000001"})
    de = m2.update_from_event({"event_type": "RISK_DEESCALATED", "risk_level": "CAUTION", "previous_risk_level": "HIGH",
                               "timestamp": 2.0, "risk_reason": "r", "alarm_recommended": False, "event_id": "event_000002"})
    assert esc[0].alarm and not de[0].alarm and de[0].kind == "DEESCALATION" and de[0].channels == ("log",)
    assert m2.update_from_event({"event_type": "PERSISTENT_HAZARD", "risk_level": "HIGH"}) == []


def test_driver_only_high_is_labelled_driver_state_not_road_hazard():
    assert headline(AlertSeverity.WARNING, "DRIVER_DROWSINESS") == "WARNING: DRIVER STATE"
    assert headline(AlertSeverity.WARNING, "PEDESTRIAN_CONFLICT") == "WARNING: ROAD HAZARD"
    assert headline(AlertSeverity.NONE, "NONE") is None


def test_alert_wording_makes_no_collision_or_safety_claim():
    texts = [headline(s, h) or "" for s in AlertSeverity for h in HAZARD_PHRASES] + list(HAZARD_PHRASES.values())
    for t in texts:
        assert "collision" not in t.lower() and "safe to" not in t.lower() and "guarantee" not in t.lower()
    assert "not a collision prediction" in DISCLAIMER and "does not guarantee safety" in DISCLAIMER


def test_alerts_replay_real_logged_events():
    """The alert layer also consumes recorded REAL events (data/events, read-only)."""
    path = REAL_EVENTS_DIR / "events.jsonl"
    if not path.is_file():
        pytest.skip("no real event dataset")
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    m = AlertManager("LOCAL_REAL")
    alerts = [al for e in events for al in m.update_from_event(e)]
    n_level = sum(e["event_type"] in ("RISK_ESCALATED", "RISK_DEESCALATED") for e in events)
    assert len(alerts) == n_level
    assert all(al.data_source == "LOCAL_REAL" for al in alerts)
    assert {al.risk_level for al in alerts if al.alarm} <= {"HIGH", "CRITICAL"}


# ------------------------------------------------------------------------------------ real front video (A, E-H)


def _real_front_video() -> Path | None:
    root = load_settings().dataset_root
    if root is None or not Path(root).is_dir():
        return None
    preferred = sorted(Path(root).rglob("frontcamera/vehicles/video_20260926_170005.mp4"))
    if preferred:
        return preferred[0]
    vids = sorted(Path(root).rglob("frontcamera/*/*.mp4"))
    return vids[0] if vids else None


def _tree_hash(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()} if root.exists() else {}


@pytest.fixture(scope="module")
def real_demo(tmp_path_factory):
    video = _real_front_video()
    if video is None:
        pytest.skip("no real front-camera video (set DATASET_ROOT)")
    if not Path(load_road_config().model_path).is_file():
        pytest.skip("YOLO weights not available locally")
    pytest.importorskip("ultralytics")
    from app.demo import run_front_demo

    before = _tree_hash(REAL_EVENTS_DIR)
    out = tmp_path_factory.mktemp("demo")
    run = run_front_demo(video, out, sample_fps=10, device="cpu", max_seconds=4.0, annotate=True, console=False)
    return {"run": run, "out": out, "video": video, "before": before, "after": _tree_hash(REAL_EVENTS_DIR),
            "summary": json.loads((out / "demo_summary.json").read_text())}


def _events(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines() if line.strip()]


def test_A_real_front_video_through_pipeline_to_events(real_demo):
    s, out = real_demo["summary"], real_demo["out"]
    assert s["camera"] == "front" and s["input"]["file_name"] == real_demo["video"].name
    assert s["processing"]["frames_processed"] >= 30  # ~4 s at 10 fps
    assert s["perception"]["detections"] > 0 and s["perception"]["tracks"] > 0
    events = _events(out)
    assert events and s["events"]["total"] == len(events)
    assert any(e["event_type"] == "RISK_ESCALATED" for e in events)
    assert s["alerts"]["escalations"] >= 1


def test_E_real_synthetic_provenance_separation(real_demo, tmp_path):
    from app.demo import DemoError, check_real_video
    from app.events import EventDatasetWriter, EventWriteError

    out = real_demo["out"]
    events = _events(out)
    assert {e["data_source"] for e in events} == {"LOCAL_REAL"}
    assert {e["observation_type"] for e in events} == {"INFERRED"}
    with open(out / "timeline.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["data_source"] for r in rows} == {"LOCAL_REAL"} and {r["observation_type"] for r in rows} == {"INFERRED"}
    assert {json.loads(line)["data_source"] for line in (out / "alerts.jsonl").read_text().splitlines()} <= {"LOCAL_REAL"}
    assert real_demo["summary"]["provenance"]["synthetic_rows_included"] == 0
    # the demo writer is the REAL writer: it refuses synthetic records
    fake = dict(events[0], data_source="SYNTHETIC", event_id="event_999999")
    with pytest.raises(EventWriteError):
        EventDatasetWriter(tmp_path / "w", kind="real").append([fake])
    # synthetic datasets are never accepted as demo input
    sim = PROJECT_ROOT / "data" / "simulated" / "scenarios.jsonl"
    if sim.is_file():
        with pytest.raises(DemoError):
            check_real_video(sim)
    # the real event dataset is untouched by the demo
    assert real_demo["before"] == real_demo["after"]


def test_F_event_evidence_linkage(real_demo):
    out = real_demo["out"]
    events = _events(out)
    ids = {e["event_id"] for e in events}
    with_evidence = [e for e in events if e["evidence_kind"] != "NONE"]
    assert with_evidence
    for e in with_evidence:
        files = json.loads(e["evidence_files"])
        assert files and all((out / "evidence" / f).is_file() for f in files)
        meta = json.loads((out / "evidence" / e["evidence_path"] / "metadata.json").read_text())
        assert meta["record"]["event_id"] == e["event_id"]
    assert real_demo["summary"]["evidence"]["missing_files"] == []
    with open(out / "timeline.csv", newline="") as fh:
        linked = {i for r in csv.DictReader(fh) if r["event_id"] for i in r["event_id"].split(";")}
    assert linked == ids
    for line in (out / "alerts.jsonl").read_text().splitlines():
        al = json.loads(line)
        assert set(al["event_ids"]) <= ids
        esc = [e for e in events if e["event_id"] in al["event_ids"] and e["event_type"] in ("RISK_ESCALATED", "RISK_DEESCALATED")]
        assert esc and abs(esc[0]["timestamp"] - al["timestamp"]) < 1e-3 and esc[0]["risk_level"] == al["risk_level"]
    assert "evidence" in format_event(with_evidence[0])


def test_G_no_synchronized_driver_claim(real_demo, tmp_path):
    from app.demo import NO_SYNC_STATEMENT, render_final_report

    s = real_demo["summary"]
    assert s["provenance"]["synchronized_driver_road"] is False
    assert s["provenance"]["driver_input"].startswith("none")
    for e in _events(real_demo["out"]):
        assert e["camera"] == "front"
        assert e["driver_hand_state"] is None and e["drowsiness_level"] is None and e["driver_activity"] is None
        assert e["hazard_type"] != "COMBINED_DRIVER_HAZARD"
    report = render_final_report([("outputs/demo", s)])
    assert NO_SYNC_STATEMENT in report and "No synchronized real driver+road recording exists" in report
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_demo_cli", PROJECT_ROOT / "scripts" / "run_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit):  # one camera per run: there is no paired mode
        mod.main(["--video", str(real_demo["video"]), "--camera", "both", "--output-dir", str(tmp_path)])


def test_H_demo_output_schema_validity(real_demo):
    from app.demo import SUMMARY_KEYS, TIMELINE_COLUMNS
    from app.events import validate_record

    out, s = real_demo["out"], real_demo["summary"]
    for name in ("demo_summary.json", "events.jsonl", "events.parquet", "schema.json", "alerts.jsonl", "timeline.csv",
                 "risk_timeline.png"):
        assert (out / name).is_file(), name
    assert (out / "evidence").is_dir()
    assert set(SUMMARY_KEYS) <= set(s)
    events = _events(out)
    for e in events:
        assert validate_record(e) == []
    pq = pytest.importorskip("pyarrow.parquet")
    assert pq.read_metadata(out / "events.parquet").num_rows == len(events)
    with open(out / "timeline.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == TIMELINE_COLUMNS
        rows = list(reader)
    assert len(rows) == s["processing"]["frames_processed"]
    required = ("timestamp", "risk_level", "risk_score", "hazard_type", "hazard_class", "primary_track_id", "risk_reason",
                "alarm_recommended", "event_id")
    assert all(c in TIMELINE_COLUMNS for c in required)
    ts = [float(r["timestamp"]) for r in rows]
    assert ts == sorted(ts) and all(r["risk_level"] in ("SAFE", "CAUTION", "HIGH", "CRITICAL") for r in rows)
    assert (out / "risk_timeline.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    if s["outputs"]["annotated_demo.mp4"]:
        assert (out / "annotated_demo.mp4").stat().st_size > 0


def test_cli_runs_on_real_video(real_demo, tmp_path, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_demo_cli2", PROJECT_ROOT / "scripts" / "run_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rc = mod.main(["--video", str(real_demo["video"]), "--output-dir", str(tmp_path / "demo"), "--fps", "5",
                   "--device", "cpu", "--max-seconds", "2", "--no-video", "--report", str(tmp_path / "report.md")])
    assert rc == 0
    text = (tmp_path / "report.md").read_text()
    assert "No synchronized real driver+road recording exists" in text and "LOCAL_REAL" in text
    assert "--- summary ---" in capsys.readouterr().out
    assert not (tmp_path / "demo" / "annotated_demo.mp4").exists()
