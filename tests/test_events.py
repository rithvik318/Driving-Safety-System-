"""Event layer: schema, provenance, triggers, dedup/cooldown, GPS, evidence, writers.

All inputs are SYNTHETIC test fixtures (constructed assessments, tracks, driver states and frames).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.config.event_config import EventConfig, load_event_config
from app.driver.models import (DriverActivity, DriverFrameObservation, DriverState, DriverTemporalState, DrowsinessLevel,
                               FaceStatus, HandState, HandStateResult)
from app.events import (DataSource, EventDatasetWriter, EventRecord, EventRecorder, EventValidationError, EventWriteError,
                        FIELDS, SCHEMA_VERSION, export_schema, read_jsonl, validate_record)
from app.events.schema import FIELD_NAMES
from app.risk import HazardType, RiskEngine, RiskLevel, RiskSnapshot
from app.risk.models import RiskAssessment, RiskFactor, TrackHazard
from app.road.models import Detection
from app.sensors.gps import GPSFix, GPSLogProvider, NoGPS, load_gps_csv
from app.tracking import IoUTracker

W, H = 1920, 1080
REQUIRED = ["event_id", "timestamp", "event_type", "gps_lat", "gps_lon", "gps_source", "risk_level", "raw_risk_score",
            "smoothed_risk_score", "hazard_type", "primary_track_id", "driver_hand_state", "driver_hand_confidence",
            "driver_activity", "drowsiness_level", "drowsiness_score", "hazard_class", "hazard_confidence",
            "track_persistence_seconds", "track_detection_count", "center_x", "center_y", "velocity_x_pixels_per_second",
            "velocity_y_pixels_per_second", "bbox_width", "bbox_height", "bbox_area", "area_change_rate",
            "trajectory_overlap", "hazard_persistence", "risk_reason", "risk_factors", "evidence_quality",
            "alarm_triggered", "data_source", "observation_type", "evidence_path"]


# --------------------------------------------------------------------------- fixtures


def hazard(track_id=1, htype=HazardType.APPROACHING_VEHICLE, qualifies=True, cls="car"):
    return TrackHazard(track_id=track_id, class_name=cls, kind="vehicle", hazard_type=htype, qualifies=qualifies,
                       points=40.0, factors=(), evidence_groups=("motion",), growth_per_second=0.5,
                       lateral_toward_corridor=0.1, center_x_norm=0.5, bottom_norm=0.8, area_fraction=0.05, quality=1.0)


def assess(t, level="SAFE", htype=HazardType.NONE, track_id=None, hazards=(), score=0.0):
    lvl = RiskLevel(level)
    return RiskAssessment(timestamp=t, risk_level=lvl, raw_risk_level=lvl, raw_risk_score=score, smoothed_risk_score=score,
                          hazard_type=htype, primary_track_id=track_id, primary_class="car" if track_id else None,
                          contributing_factors=(RiskFactor("approach", score, "road", "car #1 box growing", track_id),) if score else (),
                          reason="test reason", evidence_quality=1.0, evidence_notes=(), alarm_recommended=lvl.rank >= 2,
                          hazards=tuple(hazards))


def car_track(t, persistence=0.5):  # two detections 0.5 s apart -> one track, ID 1
    tr = IoUTracker()
    tr.update([Detection(None, "car", 2, 0.8, 900, 500, 1000, 580, "t")], t - persistence)
    return tr.update([Detection(None, "car", 2, 0.82, 890, 495, 1010, 590, "t")], t)[0]


def drv(t, hand="BOTH_HANDS", activity="NORMAL", drowsy="LOW", conf=0.9):
    obs = DriverFrameObservation(timestamp=t, face_status=FaceStatus.OK, face_detected=True, landmarks_available=True)
    return DriverState(timestamp=t, observation=obs,
                       temporal=DriverTemporalState(drowsiness_level=DrowsinessLevel(drowsy), drowsiness_score=0.1),
                       hand=HandStateResult(hand_state=HandState(hand), confidence=conf), observation_quality=1.0,
                       driver_activity=DriverActivity(activity))


def recorder(tmp_path=None, **cfg):
    rec = EventRecorder(EventConfig(**cfg), run_id="test_run", evidence_root=tmp_path)
    rec.start_session("sess", "sess.mp4", DataSource.LOCAL_REAL, "front")
    return rec


FRAME = np.full((240, 320, 3), 120, np.uint8)


# --------------------------------------------------------------------------- A-C schema + provenance


def test_A_schema_validation_of_a_real_record():
    rec = recorder()
    ev = rec.on_risk(assess(1.0, "CAUTION", HazardType.APPROACHING_VEHICLE, 1, [hazard()], 30), [car_track(1.0)])[0]
    assert validate_record(ev.to_dict()) == []
    bad = ev.to_dict() | {"raw_risk_score": "high"}
    assert any("raw_risk_score must be float64" in e for e in validate_record(bad))
    with pytest.raises(EventValidationError):
        ev.with_updates(gps_lat=12.0)  # lat without lon / source


def test_B_required_fields_present_and_ordered():
    for name in REQUIRED:
        assert name in FIELD_NAMES, name
    rec = recorder()
    ev = rec.on_risk(assess(1.0, "CAUTION"), [])[0]
    assert list(ev.to_dict()) == list(FIELD_NAMES)
    d = ev.to_dict()
    del d["event_type"]
    assert any("missing fields" in e for e in validate_record(d))
    assert any("unknown fields" in e for e in validate_record(ev.to_dict() | {"speed_kmh": 3.0}))


def test_C_provenance_labels():
    rec = recorder()
    risk_ev = rec.on_risk(assess(1.0, "CAUTION"), [])[0]
    assert risk_ev.data_source == "LOCAL_REAL" and risk_ev.observation_type == "INFERRED"
    with pytest.raises(EventValidationError, match="must be INFERRED"):
        risk_ev.with_updates(observation_type="OBSERVED")
    for src in ("PUBLIC", "SYNTHETIC", "SYNTHETIC_COMBINATION"):
        assert risk_ev.with_updates(data_source=src).data_source == src
    with pytest.raises(EventValidationError):
        risk_ev.with_updates(data_source="REALISH")
    by_name = {f.name: f.provenance for f in FIELDS}
    for inferred in ("raw_risk_score", "risk_level", "risk_reason", "velocity_x_pixels_per_second", "area_change_rate",
                     "track_persistence_seconds", "primary_track_id", "hazard_persistence"):
        assert by_name[inferred] == "INFERRED", inferred
    for observed in ("timestamp", "hazard_class", "hazard_confidence", "gps_lat", "driver_hand_state"):
        assert by_name[observed] == "OBSERVED", observed


def test_C_writers_never_mix_real_and_synthetic(tmp_path):
    rec = EventRecorder(run_id="syn")
    rec.start_session("s", None, DataSource.SYNTHETIC, "front")
    syn = rec.on_risk(assess(1.0, "CAUTION"), [])
    with pytest.raises(EventWriteError, match="cannot be written to the real dataset"):
        EventDatasetWriter(tmp_path / "real", kind="real").append(syn)
    assert EventDatasetWriter(tmp_path / "syn", kind="synthetic").append(syn) == 1
    real = recorder().on_risk(assess(1.0, "CAUTION"), [])
    with pytest.raises(EventWriteError):
        EventDatasetWriter(tmp_path / "syn2", kind="synthetic").append(real)
    combo = EventRecorder(run_id="c")
    combo.start_session("s", None, DataSource.SYNTHETIC_COMBINATION, "front+driver")
    ev = combo.on_risk(assess(1.0, "CAUTION"), [])[0]
    assert "NOT recorded together" in ev.notes


# --------------------------------------------------------------------------- D-G triggers


def test_D_escalation_creates_one_event():
    rec = recorder()
    assert rec.on_risk(assess(0.0, "SAFE"), []) == []
    ev = rec.on_risk(assess(0.1, "CAUTION", score=25), [])
    assert len(ev) == 1 and ev[0].event_type == "RISK_ESCALATED" and ev[0].transition == "SAFE->CAUTION"
    assert ev[0].previous_risk_level == "SAFE" and ev[0].risk_level == "CAUTION"
    ev2 = rec.on_risk(assess(0.2, "CRITICAL", score=80), [])  # jump two levels: one event
    assert [e.transition for e in ev2] == ["CAUTION->CRITICAL"]


def test_E_repeated_frames_do_not_duplicate():
    rec = recorder()
    events = []
    for i in range(50):
        events += rec.on_risk(assess(i * 0.1, "HIGH", score=50), [])
    assert [e.event_type for e in events] == ["RISK_ESCALATED"]


def test_E_F_persistent_hazard_once_per_cooldown():
    rec = recorder(persistent_hazard_min_seconds=2.0, hazard_cooldown_seconds=10.0)
    events = []
    for i in range(160):  # 16 s of the same qualifying hazard at 10 fps
        t = i * 0.1
        events += rec.on_risk(assess(t, "HIGH", HazardType.APPROACHING_VEHICLE, 1, [hazard()], 50), [car_track(t)])
    persistent = [e for e in events if e.event_type == "PERSISTENT_HAZARD"]
    assert [round(e.timestamp, 1) for e in persistent] == [2.0, 12.0]  # first after 2 s, then after the 10 s cooldown
    assert persistent[0].hazard_persistence == pytest.approx(2.0) and persistent[0].primary_track_id == 1
    assert persistent[0].hazard_class == "car" and persistent[0].track_detection_count == 2


def test_F_cooldown_is_per_hazard_and_track():
    rec = recorder(persistent_hazard_min_seconds=0.5, hazard_cooldown_seconds=10.0)
    events = []
    for i in range(20):
        t = i * 0.1
        hz = [hazard(1), hazard(2, HazardType.PEDESTRIAN_CONFLICT, cls="person")]
        events += rec.on_risk(assess(t, "HIGH", HazardType.APPROACHING_VEHICLE, 1, hz, 50), [])
    persistent = sorted((e.primary_track_id, e.hazard_type) for e in events if e.event_type == "PERSISTENT_HAZARD")
    assert persistent == [(1, "APPROACHING_VEHICLE"), (2, "PEDESTRIAN_CONFLICT")]


def test_persistent_hazard_needs_continuity_and_level():
    rec = recorder(persistent_hazard_min_seconds=1.0)
    events = []
    for i in range(30):  # hazard reported only on even frames: never continuous for 1 s
        t = i * 0.1
        events += rec.on_risk(assess(t, "HIGH", HazardType.APPROACHING_VEHICLE, 1, [hazard()] if i % 2 == 0 else [], 50), [])
    assert not [e for e in events if e.event_type == "PERSISTENT_HAZARD"]
    rec2 = recorder(persistent_hazard_min_seconds=0.5)
    ev2 = []
    for i in range(20):  # continuous but level stays SAFE (below CAUTION)
        ev2 += rec2.on_risk(assess(i * 0.1, "SAFE", HazardType.NONE, None, [hazard()]), [])
    assert ev2 == []


def test_G_deescalation_events():
    rec = recorder()
    seq = ["CAUTION", "HIGH", "CRITICAL", "HIGH", "CAUTION", "SAFE"]
    events = [e for i, lv in enumerate(seq) for e in rec.on_risk(assess(i * 0.5, lv, score=10), [])]
    assert [(e.event_type, e.transition) for e in events] == [
        ("RISK_ESCALATED", "SAFE->CAUTION"), ("RISK_ESCALATED", "CAUTION->HIGH"), ("RISK_ESCALATED", "HIGH->CRITICAL"),
        ("RISK_DEESCALATED", "CRITICAL->HIGH"), ("RISK_DEESCALATED", "HIGH->CAUTION"), ("RISK_DEESCALATED", "CAUTION->SAFE")]
    with pytest.raises(EventValidationError, match="direction"):
        events[0].with_updates(event_type="RISK_DEESCALATED")


# --------------------------------------------------------------------------- H driver changes


def test_H_driver_state_transition_event():
    rec = recorder(driver_change_min_observations=3)
    seq = ["BOTH_HANDS"] * 4 + ["ONE_HAND"] * 4 + ["NO_HANDS"] * 4
    events = [e for i, h in enumerate(seq) for e in rec.on_driver(drv(i * 0.1, hand=h))]
    assert [(e.change_kind, e.transition, e.observation_type) for e in events] == [
        ("hand_state", "BOTH_HANDS->ONE_HAND", "OBSERVED"), ("hand_state", "ONE_HAND->NO_HANDS", "OBSERVED")]
    assert events[0].driver_hand_state == "ONE_HAND" and events[0].driver_hand_confidence == pytest.approx(0.9)
    assert events[0].risk_level is None and events[0].data_source == "LOCAL_REAL"


def test_H_flicker_and_unknown_do_not_create_events():
    rec = recorder(driver_change_min_observations=3)
    seq = ["BOTH_HANDS"] * 4 + ["ONE_HAND", "BOTH_HANDS", "UNKNOWN", "ONE_HAND", "BOTH_HANDS"] * 4
    assert [e for i, h in enumerate(seq) for e in rec.on_driver(drv(i * 0.1, hand=h))] == []


def test_H_inferred_driver_changes_and_cooldown():
    rec = recorder(driver_change_min_observations=2, driver_change_cooldown_seconds=5.0)
    seq = [("NORMAL", "LOW")] * 3 + [("HANDS_OFF_WHEEL", "HIGH")] * 3 + [("NORMAL", "LOW")] * 3 + [("HANDS_OFF_WHEEL", "HIGH")] * 3
    events = [e for i, (a, d) in enumerate(seq) for e in rec.on_driver(drv(i * 0.1, hand="NO_HANDS", activity=a, drowsy=d))]
    kinds = [(e.change_kind, e.transition) for e in events]
    assert ("driver_activity", "NORMAL->HANDS_OFF_WHEEL") in kinds and ("drowsiness_level", "LOW->HIGH") in kinds
    assert kinds.count(("driver_activity", "NORMAL->HANDS_OFF_WHEEL")) == 1  # second one within the 5 s cooldown
    assert all(e.observation_type == "INFERRED" for e in events)


# --------------------------------------------------------------------------- I-J GPS


def test_I_missing_gps():
    ev = recorder().on_risk(assess(1.0, "CAUTION"), [])[0]
    assert ev.gps_lat is None and ev.gps_lon is None and ev.gps_source == "UNAVAILABLE"
    assert NoGPS().fix_at(5.0) is None
    assert "GPS unavailable" in ev.notes


def test_J_real_gps_values_preserved(tmp_path):
    csv = tmp_path / "gps.csv"
    csv.write_text("timestamp,lat,lon,accuracy_m\n0.0,26.1878,91.6916,4.5\n1.0,26.18785,91.69171,3.0\n")
    gps = load_gps_csv(csv, source="PHONE_GPS_LOG", max_age_seconds=0.5)
    rec = EventRecorder(run_id="g", gps=gps)
    rec.start_session("s", None, DataSource.LOCAL_REAL, "front")
    ev = rec.on_risk(assess(0.9, "CAUTION"), [])[0]
    assert (ev.gps_lat, ev.gps_lon, ev.gps_accuracy_m, ev.gps_timestamp) == (26.18785, 91.69171, 3.0, 1.0)
    assert ev.gps_source == "PHONE_GPS_LOG"
    far = rec.on_risk(assess(5.0, "HIGH"), [])[0]  # no fix within 0.5 s: nothing invented
    assert far.gps_lat is None and far.gps_source == "UNAVAILABLE"
    with pytest.raises(ValueError):
        GPSFix(95.0, 10.0, 0.0)
    assert GPSLogProvider([]).fix_at(1.0) is None


# --------------------------------------------------------------------------- K evidence


def test_K_frame_evidence_paths_relative_and_valid(tmp_path):
    rec = recorder(tmp_path)
    rec.observe_frame("front", 0.9, FRAME)
    ev = rec.on_risk(assess(1.0, "CAUTION"), [])[0]
    assert ev.evidence_kind == "FRAME" and ev.evidence_path == "test_run/event_000001"
    files = json.loads(ev.evidence_files)
    assert files == ["test_run/event_000001/metadata.json", "test_run/event_000001/front_frame.jpg"]
    for f in files:
        assert not Path(f).is_absolute() and (tmp_path / f).is_file()
    meta = json.loads((tmp_path / "test_run" / "event_000001" / "metadata.json").read_text())
    assert meta["record"]["event_id"] == "event_000001" and meta["evidence"]["kind"] == "FRAME"
    with pytest.raises(EventValidationError, match="relative"):
        ev.with_updates(evidence_path="/abs/event_000001")
    with pytest.raises(EventValidationError, match="relative"):
        ev.with_updates(evidence_files=json.dumps(["../escape.jpg"]))


def test_K_deescalation_has_no_evidence_by_default(tmp_path):
    rec = recorder(tmp_path)
    rec.observe_frame("front", 0.0, FRAME)
    rec.on_risk(assess(0.0, "CAUTION"), [])
    ev = rec.on_risk(assess(0.5, "SAFE"), [])[0]
    assert ev.event_type == "RISK_DEESCALATED" and ev.evidence_kind == "NONE" and ev.evidence_path is None


def test_K_clip_evidence_pre_and_post(tmp_path):
    pytest.importorskip("cv2")
    rec = recorder(tmp_path, evidence_mode="clip", evidence_pre_seconds=1.0, evidence_post_seconds=1.0)
    events = []
    for i in range(40):  # 4 s at 10 fps, escalation at 2.0 s
        t = i * 0.1
        events += rec.observe_frame("front", t, np.full((240, 320, 3), i * 5, np.uint8))
        events += rec.on_risk(assess(t, "CAUTION" if t >= 2.0 - 1e-9 else "SAFE"), [])
        if i == 25:
            assert events == []  # clip not complete yet: no record pretends it exists
    events += rec.finish_session()
    assert len(events) == 1
    ev = events[0]
    assert ev.evidence_kind == "CLIP"
    assert json.loads(ev.evidence_files) == ["test_run/event_000001/metadata.json", "test_run/event_000001/front.mp4"]
    meta = json.loads((tmp_path / "test_run" / "event_000001" / "metadata.json").read_text())["evidence"]["cameras"]["front"]
    assert meta["pre_seconds_covered"] == pytest.approx(1.0) and meta["post_seconds_covered"] == pytest.approx(1.0)
    assert (tmp_path / "test_run" / "event_000001" / "front.mp4").stat().st_size > 0


def test_K_clip_truncated_at_stream_end_is_reported(tmp_path):
    pytest.importorskip("cv2")
    rec = recorder(tmp_path, evidence_mode="clip", evidence_pre_seconds=5.0, evidence_post_seconds=5.0)
    for i in range(10):
        rec.observe_frame("front", i * 0.1, FRAME)
        rec.on_risk(assess(i * 0.1, "CAUTION" if i >= 5 else "SAFE"), [])
    ev = rec.finish_session()[0]
    meta = json.loads((tmp_path / ev.evidence_path / "metadata.json").read_text())["evidence"]["cameras"]["front"]
    assert meta["post_seconds_covered"] == pytest.approx(0.4) and meta["pre_seconds_covered"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- L-N writing


def _some_events():
    rec = recorder()
    out = []
    for i, lv in enumerate(["SAFE", "CAUTION", "HIGH", "CAUTION"]):
        out += rec.on_risk(assess(i * 1.0, lv, score=10 * i), [])
    return out


def test_L_jsonl_writing(tmp_path):
    w = EventDatasetWriter(tmp_path, kind="real")
    assert w.append(_some_events()) == 3
    rows = read_jsonl(tmp_path / "events.jsonl")
    assert [r["event_type"] for r in rows] == ["RISK_ESCALATED", "RISK_ESCALATED", "RISK_DEESCALATED"]
    assert all(validate_record(r) == [] for r in rows) and list(rows[0]) == list(FIELD_NAMES)
    with pytest.raises(EventWriteError, match="duplicate"):
        w.append(_some_events()[:1])
    w2 = EventDatasetWriter(tmp_path, kind="real", overwrite=True)
    assert w2.append(_some_events()) == 3 and len(read_jsonl(tmp_path / "events.jsonl")) == 3


def test_M_parquet_writing(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    w = EventDatasetWriter(tmp_path, kind="real")
    w.append(_some_events())
    out = w.finalize()
    assert out["parquet"] and Path(out["parquet"]).is_file()
    table = pq.read_table(out["parquet"])
    assert table.num_rows == 3 and table.column_names == list(FIELD_NAMES)
    assert table.schema.field("gps_lat").nullable and not table.schema.field("event_id").nullable


def test_M_parquet_fallback_note(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_pyarrow(name, *a, **k):
        if name.startswith("pyarrow"):
            raise ImportError("no pyarrow")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_pyarrow)
    w = EventDatasetWriter(tmp_path, kind="real")
    w.append(_some_events())
    out = w.finalize()
    assert out["parquet"] is None and "JSON Lines only" in out["parquet_note"] and out["records"] == 3


def test_N_schema_export(tmp_path):
    path = export_schema(tmp_path / "schema.json")
    doc = json.loads(path.read_text())
    assert doc["schema_version"] == SCHEMA_VERSION and [f["name"] for f in doc["fields"]] == list(FIELD_NAMES)
    for f in doc["fields"]:
        assert {"name", "type", "nullable", "provenance", "description"} <= set(f)
        assert f["provenance"] in ("METADATA", "OBSERVED", "INFERRED")
    assert "SYNTHETIC_COMBINATION" in doc["record_provenance"]["data_source"]


# --------------------------------------------------------------------------- O-P


def test_O_deterministic_event_ids():
    def run():
        rec = EventRecorder(run_id="r")
        out = []
        for sess in ("a", "b"):
            rec.start_session(sess, None, DataSource.LOCAL_REAL, "front")
            for i, lv in enumerate(["SAFE", "HIGH", "SAFE"]):
                out += rec.on_risk(assess(i * 1.0, lv), [])
            out += rec.finish_session()
        return [e.to_dict() for e in out]

    a, b = run(), run()
    assert a == b
    assert [e["event_id"] for e in a] == [f"event_{i:06d}" for i in range(1, 5)]  # ids continue across sessions
    assert [e["session_id"] for e in a] == ["a", "a", "b", "b"]


def test_P_empty_inputs():
    rec = recorder()
    assert rec.on_risk(assess(0.0, "SAFE"), []) == [] and rec.on_driver(None) == []
    assert rec.finish_session() == []
    w = EventDatasetWriter  # empty append is a no-op
    assert w.__name__ and rec.records == []
    with pytest.raises(RuntimeError):
        rec.on_risk(assess(1.0, "SAFE"), [])  # no session


def test_P_empty_writer(tmp_path):
    w = EventDatasetWriter(tmp_path, kind="real")
    assert w.append([]) == 0
    out = w.finalize()
    assert out["records"] == 0 and (tmp_path / "schema.json").is_file()


# --------------------------------------------------------------------------- integration + config


def test_full_chain_tracker_risk_recorder(tmp_path):
    """SYNTHETIC detections -> real tracker -> real risk engine -> recorder -> writer."""
    tracker, engine = IoUTracker(), RiskEngine()
    rec = EventRecorder(EventConfig(), run_id="chain", evidence_root=tmp_path / "ev")
    rec.start_session("synthetic_approach", None, DataSource.SYNTHETIC, "front")
    events = []
    for i in range(40):
        t = i * 0.1
        s = 1 + 0.08 * i
        dets = [Detection(t, "car", 2, 0.85, 960 - 100 * s, 700 - 75 * s, 960 + 100 * s, 700 + 75 * s, "SYNTHETIC")]
        tracks = tracker.update(dets, t, i)
        a = engine.evaluate(RiskSnapshot(t, tuple(tracks), None, W, H))
        events += rec.observe_frame("front", t, FRAME)
        events += rec.on_risk(a, tracks, None, i, (W, H))
    events += rec.finish_session()
    types = [e.event_type for e in events]
    assert types.count("RISK_ESCALATED") >= 2 and "PERSISTENT_HAZARD" in types
    esc = [e for e in events if e.event_type == "RISK_ESCALATED"][-1]
    assert esc.hazard_class == "car" and esc.primary_track_id == 1 and esc.box_growth_per_second > 0
    assert esc.trajectory_overlap is None and esc.alarm_triggered is False and esc.data_source == "SYNTHETIC"
    w = EventDatasetWriter(tmp_path / "syn", kind="synthetic")
    assert w.append(events) == len(events)


def test_event_config_env(monkeypatch):
    monkeypatch.setattr("app.config.event_config.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("EVENT_HAZARD_COOLDOWN_SECONDS", "3")
    monkeypatch.setenv("EVENT_EVIDENCE_MODE", "CLIP")
    c = load_event_config()
    assert c.hazard_cooldown_seconds == 3 and c.evidence_mode == "clip"
    monkeypatch.setenv("EVENT_SYNTHETIC_ROOT", "data/events")
    with pytest.raises(ValueError, match="never mixed"):
        load_event_config()


def test_event_record_is_immutable():
    ev = recorder().on_risk(assess(1.0, "CAUTION"), [])[0]
    with pytest.raises(AttributeError):
        ev.risk_level = "SAFE"
    assert isinstance(ev, EventRecord)


def test_record_events_script_front_only(tmp_path, monkeypatch, capsys):
    """Script end-to-end with a fake detector on a generated video (test fixture, tmp dirs only)."""
    import importlib.util
    import sys

    cv2 = pytest.importorskip("cv2")
    import app.road
    from app.road import YoloRoadDetector

    for mod in ("settings", "road_config", "tracking_config", "risk_config", "event_config"):
        monkeypatch.setattr(f"app.config.{mod}.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("EVENT_EVENT_ROOT", str(tmp_path / "events"))
    monkeypatch.setenv("EVENT_SYNTHETIC_ROOT", str(tmp_path / "events_synthetic"))
    video = tmp_path / "front" / "clip.avi"
    video.parent.mkdir()
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (640, 360))
    for _ in range(80):
        w.write(np.full((360, 640, 3), 90, np.uint8))
    w.release()

    class Boxes:
        def __init__(self, rows):
            a = np.array(rows, dtype=np.float32).reshape(-1, 6)
            self.xyxy, self.conf, self.cls = a[:, :4], a[:, 4], a[:, 5]

        def __len__(self):
            return len(self.conf)

    class GrowingCar:
        names = {2: "car"}

        def __init__(self):
            self.i = 0

        def predict(self, frame, **kw):
            s = 1 + 0.08 * self.i
            self.i += 1
            return [type("R", (), {"boxes": Boxes([[320 - 40 * s, 250 - 30 * s, 320 + 40 * s, 250 + 30 * s, 0.9, 2]])})()]

    class FakeDet(YoloRoadDetector):
        def __init__(self, *a, **k):
            super().__init__("models/road/yolo26n.pt", device="cpu", model=GrowingCar(), warmup=False, classes=("car",))

    monkeypatch.setattr(app.road, "YoloRoadDetector", FakeDet)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("rec_cli", root / "scripts" / "record_events.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    assert m.main(["--front-source", str(video.parent), "--run-id", "t1", "--summary-json", str(tmp_path / "s.json")]) == 0
    rows = read_jsonl(tmp_path / "events" / "events.jsonl")
    assert rows and all(r["data_source"] == "LOCAL_REAL" and r["camera"] == "front" for r in rows)
    assert "RISK_ESCALATED" in {r["event_type"] for r in rows}
    assert all(validate_record(r) == [] for r in rows)
    assert (tmp_path / "events" / "schema.json").is_file()
    assert not (tmp_path / "events_synthetic").exists()
    for r in rows:
        for f in json.loads(r["evidence_files"] or "[]"):
            assert (tmp_path / "events" / f).is_file()
