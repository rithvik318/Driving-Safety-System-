"""Contextual risk engine: rules, combinations, gates, hysteresis, explanations.

All inputs here are SYNTHETIC feature-level fixtures (constructed detections fed through the real
IoUTracker, and DriverState dataclasses built directly). They test the rules; they are not
observations of real driving.
"""

from __future__ import annotations

import math
import random

import pytest

from app.config.risk_config import RiskConfig, load_risk_config
from app.driver.models import (
    DriverActivity,
    DriverFrameObservation,
    DriverState,
    DriverTemporalState,
    DrowsinessLevel,
    FaceStatus,
    HandState,
    HandStateResult,
)
from app.risk import HazardType, RiskEngine, RiskLevel, RiskSnapshot
from app.risk.smoothing import RiskHysteresis, TrackFeatureHistory
from app.road.models import Detection
from app.tracking import IoUTracker

W, H = 1920, 1080


def driver(t, drowsy="LOW", activity="NORMAL", hand="BOTH_HANDS", dist=0.0, yaw=0.0, pitch=0.0,
           face="OK", quality=1.0, score=None):
    obs = DriverFrameObservation(timestamp=t, face_status=FaceStatus(face), face_detected=face == "OK",
                                 landmarks_available=face == "OK", head_yaw=yaw if face == "OK" else None,
                                 head_pitch=pitch if face == "OK" else None)
    temp = DriverTemporalState(drowsiness_level=DrowsinessLevel(drowsy),
                               drowsiness_score=score if score is not None else {"LOW": 0.1, "HIGH": 0.6, "CRITICAL": 0.9, "UNKNOWN": None}[drowsy],
                               distraction_duration=dist)
    return DriverState(timestamp=t, observation=obs, temporal=temp, hand=HandStateResult(hand_state=HandState(hand)),
                       observation_quality=quality, driver_activity=DriverActivity(activity))


def det(cls, cx, cy, w, h, conf=0.85, t=0.0):
    return Detection(t, cls, 0, conf, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, "synthetic")


# ---- SYNTHETIC scenario generators: list of per-frame detection lists at 10 fps

def approaching_vehicle(n=25):
    return [[det("car", 960, 700, 200 * (1 + 0.08 * i), 150 * (1 + 0.08 * i))] for i in range(n)]


def static_distant_vehicle(n=30, cx=1400):
    return [[det("car", cx + (i % 2), 450, 80, 60)] for i in range(n)]


def pedestrian_crossing(n=30):  # walks from x=0.25 toward the centre, feet low in the frame
    return [[det("person", 480 + 22 * i, 750, 90, 260)] for i in range(n)]


def pedestrian_static_side(n=30):
    return [[det("person", 1800, 500, 40, 110)] for _ in range(n)]


def dog_central_moving(n=30):
    return [[det("dog", 820 + 15 * i, 900, 160, 110)] for i in range(n)]


def dog_static_far(n=30):
    return [[det("dog", 150, 400, 40, 30)] for _ in range(n)]


def run(frames, driver_fn=None, cfg=None, dt=0.1):
    """Feed frames through the real tracker and the engine. Returns all assessments."""
    tracker, engine = IoUTracker(), RiskEngine(cfg or RiskConfig())
    out = []
    for i, dets in enumerate(frames):
        t = round(i * dt, 6)
        tracks = tracker.update(dets, t, i)
        d = driver_fn(t) if driver_fn else None
        out.append(engine.evaluate(RiskSnapshot(t, tuple(tracks), d, W, H)))
    return out


def peak(results):
    return max(results, key=lambda a: (a.risk_level.rank, a.raw_risk_score))


def calm(t):
    return driver(t)


# --------------------------------------------------------------------------- A-B


def test_A_no_impairment_no_hazard_is_safe():
    res = run([[] for _ in range(20)], calm)
    assert all(a.risk_level is RiskLevel.SAFE and a.raw_risk_score == 0 for a in res)
    assert res[-1].hazard_type is HazardType.NONE and not res[-1].alarm_recommended
    assert res[-1].reason == "No persistent road hazard and no driver impairment signal."


def test_B_static_distant_vehicle_not_critical():
    res = run(static_distant_vehicle(), calm)
    assert all(a.risk_level.rank <= RiskLevel.CAUTION.rank for a in res)
    last = res[-1]
    assert last.hazard_type in (HazardType.NONE, HazardType.ROAD_USER_PRESENT)
    assert all(not h.qualifies for h in last.hazards)


# --------------------------------------------------------------------------- C-E road hazards


def test_C_persistent_approaching_vehicle_elevates_risk():
    res = run(approaching_vehicle(), calm)
    last = res[-1]
    assert last.hazard_type is HazardType.APPROACHING_VEHICLE and last.primary_track_id == 1
    assert last.risk_level.rank >= RiskLevel.CAUTION.rank and last.raw_risk_score >= 40
    names = {f.name for f in last.contributing_factors}
    assert {"approach", "centrality", "persistence"} <= names
    assert res[2].risk_level is RiskLevel.SAFE  # not before the persistence gate


def test_C_approach_needs_persistence_gate():
    res = run(approaching_vehicle(4), calm)  # 0.3 s, 4 detections: below the 0.5 s gate
    assert all(a.raw_risk_score == 0 for a in res)


def test_D_pedestrian_crossing_toward_centre_elevates_risk():
    last = run(pedestrian_crossing(), calm)[-1]
    assert last.hazard_type is HazardType.PEDESTRIAN_CONFLICT
    assert last.risk_level.rank >= RiskLevel.CAUTION.rank
    assert any(f.name == "trajectory" for f in last.contributing_factors)
    assert "central image band" in last.reason


def test_D_F_pedestrian_static_far_to_side_is_not_elevated():
    res = run(pedestrian_static_side(), calm)
    assert all(a.risk_level.rank <= RiskLevel.CAUTION.rank for a in res)
    assert res[-1].raw_risk_score < RiskConfig().caution_min  # persistence only
    assert res[-1].hazard_type is HazardType.NONE


def test_E_dog_central_and_moving_elevates_risk():
    last = run(dog_central_moving(), calm)[-1]
    assert last.hazard_type is HazardType.ANIMAL_HAZARD and last.risk_level.rank >= RiskLevel.CAUTION.rank


def test_E_dog_static_far_is_not_a_hazard():
    last = run(dog_static_far(), calm)[-1]
    assert last.hazard_type is HazardType.NONE and last.risk_level is RiskLevel.SAFE


# --------------------------------------------------------------------------- F-H driver alone


def test_F_drowsiness_alone_is_driver_risk_state():
    last = run([[] for _ in range(10)], lambda t: driver(t, drowsy="HIGH"))[-1]
    assert last.hazard_type is HazardType.DRIVER_DROWSINESS and last.risk_level is RiskLevel.CAUTION
    crit = run([[] for _ in range(10)], lambda t: driver(t, drowsy="CRITICAL"))[-1]
    assert crit.hazard_type is HazardType.DRIVER_DROWSINESS
    assert crit.raw_risk_score > last.raw_risk_score
    assert crit.risk_level is RiskLevel.CAUTION  # one evidence group only: HIGH gate applies
    assert "HIGH needs at least 2 independent evidence groups" in crit.gates


def test_F_drowsiness_plus_hands_off_is_two_driver_groups():
    last = run([[] for _ in range(10)], lambda t: driver(t, drowsy="CRITICAL", activity="HANDS_OFF_WHEEL", hand="NO_HANDS", dist=3))[-1]
    assert last.risk_level is RiskLevel.HIGH and last.hazard_type is HazardType.DRIVER_DROWSINESS
    assert last.risk_level is not RiskLevel.CRITICAL  # no road hazard: CRITICAL gate


def test_G_hands_off_wheel_alone_is_driver_risk_state():
    last = run([[] for _ in range(10)], lambda t: driver(t, activity="HANDS_OFF_WHEEL", hand="NO_HANDS", dist=2.5))[-1]
    assert last.hazard_type is HazardType.HANDS_OFF_WHEEL and last.risk_level is RiskLevel.CAUTION
    assert "NO_HANDS observed continuously 2.5 s" in last.reason


def test_H_one_hand_alone_is_not_distraction():
    last = run([[] for _ in range(10)], lambda t: driver(t, hand="ONE_HAND", activity="NORMAL"))[-1]
    assert last.risk_level is RiskLevel.SAFE and last.raw_risk_score == 0
    assert any("ONE_HAND observed: not scored" in n for n in last.evidence_notes)
    # also with a no-hands observation that has NOT been inferred as hands-off yet
    short = run([[] for _ in range(5)], lambda t: driver(t, hand="NO_HANDS", activity="NORMAL"))[-1]
    assert short.raw_risk_score == 0


def test_head_away_needs_sustained_time():
    brief = run([[] for _ in range(5)], lambda t: driver(t, yaw=45))[-1]  # 0.4 s
    assert brief.raw_risk_score == 0
    long = run([[] for _ in range(15)], lambda t: driver(t, yaw=45))[-1]  # 1.4 s
    assert long.hazard_type is HazardType.DRIVER_HEAD_AWAY and long.raw_risk_score == RiskConfig().head_away_points


# --------------------------------------------------------------------------- I-J combinations


def test_I_approaching_vehicle_plus_hands_off_is_higher_than_either():
    hands = lambda t: driver(t, activity="HANDS_OFF_WHEEL", hand="NO_HANDS", dist=3)  # noqa: E731
    both = run(approaching_vehicle(), hands)
    road = run(approaching_vehicle(), calm)[-1]
    drv = run([[] for _ in range(25)], hands)[-1]
    last = both[-1]
    assert last.raw_risk_score > road.raw_risk_score and last.raw_risk_score > drv.raw_risk_score
    assert last.hazard_type is HazardType.COMBINED_DRIVER_HAZARD
    assert last.risk_level.rank >= RiskLevel.HIGH.rank and last.alarm_recommended
    assert peak(both).risk_level is RiskLevel.CRITICAL
    assert "driver_road_combination" in {f.name for f in last.contributing_factors}


def test_J_approaching_vehicle_plus_drowsiness_is_higher_than_either():
    drowsy = lambda t: driver(t, drowsy="HIGH")  # noqa: E731
    last = run(approaching_vehicle(), drowsy)[-1]
    road = run(approaching_vehicle(), calm)[-1]
    drv = run([[] for _ in range(25)], drowsy)[-1]
    assert last.raw_risk_score > max(road.raw_risk_score, drv.raw_risk_score)
    assert last.hazard_type is HazardType.COMBINED_DRIVER_HAZARD and last.risk_level.rank >= RiskLevel.HIGH.rank


def test_combined_pedestrian_and_dog_with_hands_off():
    hands = lambda t: driver(t, activity="HANDS_OFF_WHEEL", hand="NO_HANDS", dist=3)  # noqa: E731
    for frames in (pedestrian_crossing(), dog_central_moving()):
        last = run(frames, hands)[-1]
        assert last.hazard_type is HazardType.COMBINED_DRIVER_HAZARD
        assert last.risk_level.rank >= RiskLevel.HIGH.rank


def test_hands_off_with_static_side_pedestrian_is_not_combined():
    hands = lambda t: driver(t, activity="HANDS_OFF_WHEEL", hand="NO_HANDS", dist=3)  # noqa: E731
    last = run(pedestrian_static_side(), hands)[-1]
    assert last.hazard_type is HazardType.HANDS_OFF_WHEEL and last.risk_level is RiskLevel.CAUTION


# --------------------------------------------------------------------------- K-L


def test_K_weak_evidence_stays_below_critical():
    weak = [[det("car", 1500, 420, 60 * (1 + 0.02 * i), 45 * (1 + 0.02 * i), conf=0.38)] for i in range(30)]
    res = run(weak, calm)
    assert all(a.risk_level.rank < RiskLevel.CRITICAL.rank and a.raw_risk_level.rank < RiskLevel.CRITICAL.rank for a in res)
    # even an approaching vehicle alone (no driver factor) never reaches CRITICAL
    assert all(a.risk_level is not RiskLevel.CRITICAL for a in run(approaching_vehicle(40), calm))


def test_K_label_flicker_reduces_points():
    flicker = [[det("dog" if i % 2 else "cat", 820 + 15 * i, 900, 160, 110)] for i in range(30)]
    steady = run(dog_central_moving(), calm)[-1]
    shaky = run(flicker, calm)[-1]
    assert shaky.raw_risk_score < steady.raw_risk_score
    assert any(f.name == "evidence_quality" and f.points < 0 for f in shaky.contributing_factors)


def test_L_score_always_bounded():
    rng = random.Random(7)
    tracker, engine = IoUTracker(), RiskEngine()
    classes = ["car", "person", "dog", "truck", "bicycle", "traffic light", "cow"]
    for i in range(200):
        t = i * 0.1
        dets = [det(rng.choice(classes), rng.uniform(0, W), rng.uniform(0, H), rng.uniform(10, 1800), rng.uniform(10, 1000),
                    conf=rng.uniform(0.35, 1.0)) for _ in range(rng.randint(0, 6))]
        dets = [Detection(t, d.class_name, 0, d.confidence, max(0, d.bbox_x1), max(0, d.bbox_y1), min(W, d.bbox_x2),
                          min(H, d.bbox_y2), "synthetic") for d in dets if min(W, d.bbox_x2) > max(0, d.bbox_x1)
                and min(H, d.bbox_y2) > max(0, d.bbox_y1)]
        d = driver(t, drowsy=rng.choice(["LOW", "HIGH", "CRITICAL", "UNKNOWN"]),
                   activity=rng.choice(["NORMAL", "HANDS_OFF_WHEEL", "UNKNOWN"]), yaw=rng.uniform(-60, 60))
        a = engine.evaluate(RiskSnapshot(t, tuple(tracker.update(dets, t, i)), d, W, H))
        assert 0 <= a.raw_risk_score <= 100 and 0 <= a.smoothed_risk_score <= 100 and 0 <= a.evidence_quality <= 1
        assert sum(f.points for f in a.contributing_factors) == pytest.approx(a.raw_risk_score, abs=1e-6)


# --------------------------------------------------------------------------- M-N explanations


def test_M_reason_and_factors_are_consistent():
    hands = lambda t: driver(t, activity="HANDS_OFF_WHEEL", hand="NO_HANDS", dist=3.2)  # noqa: E731
    last = run(approaching_vehicle(), hands)[-1]
    assert sum(f.points for f in last.contributing_factors) == pytest.approx(last.raw_risk_score)
    r = last.reason
    assert r.startswith("Inferred HANDS_OFF_WHEEL (NO_HANDS observed continuously 3.2 s) coincides with a persistent car #1")
    assert "image-space box area growing" in r and "central image band" in r
    for f in last.contributing_factors:
        assert f.detail and f.source in ("driver", "road", "combined", "quality")
    d = last.to_dict()
    assert d["score_note"].startswith("rule-based prototype score") and d["primary_track_id"] == 1
    assert {f["name"] for f in d["contributing_factors"]} >= {"hands_off_wheel", "approach", "persistence"}


def test_N_raw_and_smoothed_scores_exposed():
    res = run(approaching_vehicle(25) + [[] for _ in range(25)], calm)
    assert res[26].raw_risk_score > 0  # coasting track (missed <= 0.3 s) still counts: short gaps tolerated
    after = res[29]  # missed 0.4 s: the track no longer counts, raw drops, smoothed decays gradually
    d = after.to_dict()
    assert "raw_risk_score" in d and "smoothed_risk_score" in d and d["risk_score"] == d["smoothed_risk_score"]
    assert after.raw_risk_score == 0 and after.smoothed_risk_score > 0
    smoothed = [a.smoothed_risk_score for a in res[29:]]
    assert smoothed == sorted(smoothed, reverse=True)  # monotone decay, at most 40 points/s
    assert all(b - a <= 40 * 0.1 + 1e-6 for a, b in zip(smoothed[1:], smoothed))
    assert res[-1].smoothed_risk_score == pytest.approx(0)


# --------------------------------------------------------------------------- O hysteresis


def test_O_single_frame_spike_does_not_escalate():
    h = RiskHysteresis(RiskConfig())
    levels = [h.update(lv, s, i * 0.1)[0] for i, (lv, s) in enumerate(
        [(RiskLevel.SAFE, 0), (RiskLevel.CRITICAL, 90), (RiskLevel.SAFE, 0), (RiskLevel.SAFE, 0), (RiskLevel.SAFE, 0)])]
    assert max(lv.rank for lv in levels) <= RiskLevel.CAUTION.rank
    assert levels[-1] is RiskLevel.SAFE


def test_O_sustained_evidence_escalates_and_decays_one_level_at_a_time():
    h = RiskHysteresis(RiskConfig())
    seq = [RiskLevel.CRITICAL] * 3 + [RiskLevel.SAFE] * 6
    levels = [h.update(lv, 80 if lv is RiskLevel.CRITICAL else 0, i * 0.1)[0].value for i, lv in enumerate(seq)]
    assert levels == ["CAUTION", "HIGH", "CRITICAL", "CRITICAL", "HIGH", "HIGH", "CAUTION", "CAUTION", "SAFE"]


def test_O_long_gap_resets_hysteresis():
    h = RiskHysteresis(RiskConfig())
    for i in range(3):
        h.update(RiskLevel.CRITICAL, 80, i * 0.1)
    assert h.update(RiskLevel.SAFE, 0, 10.0) == (RiskLevel.SAFE, 0)


def test_O_engine_level_lags_raw_level():
    hands = lambda t: driver(t, activity="HANDS_OFF_WHEEL", hand="NO_HANDS", dist=3)  # noqa: E731
    res = run(approaching_vehicle(), hands)
    first_critical_raw = next(i for i, a in enumerate(res) if a.raw_risk_level is RiskLevel.CRITICAL)
    assert res[first_critical_raw].risk_level is not RiskLevel.CRITICAL


# --------------------------------------------------------------------------- P-R robustness


def test_P_empty_track_list():
    engine = RiskEngine()
    a = engine.evaluate(RiskSnapshot(0.0, (), driver(0.0), W, H))
    assert a.risk_level is RiskLevel.SAFE and a.hazards == () and a.primary_track_id is None


def test_Q_missing_or_unknown_driver_state():
    engine = RiskEngine()
    a = engine.evaluate(RiskSnapshot(0.0, (), None, W, H))
    assert a.risk_level is RiskLevel.SAFE and a.evidence_quality <= 0.5
    assert any("no driver observation" in n for n in a.evidence_notes)
    unknown = driver(0.1, drowsy="UNKNOWN", activity="UNKNOWN", hand="UNKNOWN", face="NO_FACE", quality=0.0)
    b = engine.evaluate(RiskSnapshot(0.1, (), unknown, W, H))
    assert b.raw_risk_score == 0 and any("NO_FACE" in n for n in b.evidence_notes)
    stale = engine.evaluate(RiskSnapshot(5.0, (), driver(1.0, drowsy="CRITICAL"), W, H))
    assert stale.raw_risk_score == 0 and any("old" in n for n in stale.evidence_notes)


def test_Q_tracks_without_image_size():
    tracker = IoUTracker()
    tracks = tracker.update([det("car", 100, 100, 50, 50)], 0.0)
    a = RiskEngine().evaluate(RiskSnapshot(0.0, tuple(tracks), None, 0, 0))
    assert a.raw_risk_score == 0 and any("image size unknown" in n for n in a.evidence_notes)


@pytest.mark.parametrize("bad", [None, float("nan"), "1", True])
def test_Q_invalid_timestamp(bad):
    with pytest.raises(ValueError):
        RiskEngine().evaluate(RiskSnapshot(bad, (), None, W, H))


def test_R_deterministic():
    hands = lambda t: driver(t, activity="HANDS_OFF_WHEEL", hand="NO_HANDS", dist=3)  # noqa: E731
    frames = approaching_vehicle() + pedestrian_crossing(10)
    a = [x.to_dict(include_hazards=True) for x in run(frames, hands)]
    b = [x.to_dict(include_hazards=True) for x in run(frames, hands)]
    assert a == b


def test_backwards_time_resets_engine():
    engine, tracker = RiskEngine(), IoUTracker()
    for i, dets in enumerate(approaching_vehicle(20)):
        engine.evaluate(RiskSnapshot(i * 0.1, tuple(tracker.update(dets, i * 0.1)), None, W, H))
    a = engine.evaluate(RiskSnapshot(0.0, (), None, W, H))
    assert a.risk_level is RiskLevel.SAFE and a.smoothed_risk_score == 0


# --------------------------------------------------------------------------- features + config


def test_growth_feature_is_log_area_slope():
    cfg = RiskConfig()
    hist, tracker = TrackFeatureHistory(cfg), IoUTracker()
    for i in range(8):  # area doubles every second: ln-area slope = ln 2
        s = math.sqrt(2 ** (i * 0.1))
        tracks = tracker.update([det("car", 960, 540, 100 * s, 100 * s)], i * 0.1)
        hist.update(tracks, W, H)
    growth, lateral = hist.features(1)
    assert growth == pytest.approx(math.log(2), rel=1e-6) and lateral == pytest.approx(0, abs=1e-9)


def test_risk_config_env_and_validation(monkeypatch):
    monkeypatch.setattr("app.config.risk_config.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("RISK_HIGH_MIN", "50")
    monkeypatch.setenv("RISK_VEHICLE_CLASSES", "car, Truck")
    c = load_risk_config()
    assert c.high_min == 50 and c.vehicle_classes == ("car", "truck")
    monkeypatch.setenv("RISK_HIGH_MIN", "90")  # above critical
    with pytest.raises(ValueError):
        load_risk_config()
    monkeypatch.delenv("RISK_HIGH_MIN")
    monkeypatch.setenv("RISK_ALARM_MIN_LEVEL", "maybe")
    with pytest.raises(ValueError):
        load_risk_config()


def test_no_physical_units_in_output():
    last = run(approaching_vehicle(), calm)[-1]
    text = (last.reason + " ".join(f.detail for f in last.contributing_factors)).lower()
    for word in ("meter", "metre", "km/h", "kmph", "ttc", "time to collision", "probability"):
        assert word not in text


def test_script_part_a_synthetic_runs_without_models(capsys, monkeypatch, tmp_path):
    import importlib.util
    import json
    import sys
    from pathlib import Path

    for mod in ("settings", "road_config", "tracking_config", "risk_config"):
        monkeypatch.setattr(f"app.config.{mod}.load_dotenv", lambda *a, **k: False)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("risk_cli", root / "scripts" / "test_risk_engine.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    assert m.main(["--summary-json", str(tmp_path / "s.json")]) == 0
    out = capsys.readouterr().out
    assert "PART A - SYNTHETIC rule stress test (constructed inputs; NOT real observations)" in out
    assert "not a probability of collision" in out and "PART B" not in out
    rows = {r["scenario"]: r for r in json.loads((tmp_path / "s.json").read_text())["part_a_synthetic"]}
    assert rows["one hand on wheel, no hazard"]["final"]["risk_level"] == "SAFE"
    assert rows["approaching vehicle + hands-off-wheel"]["final"]["risk_level"] == "CRITICAL"
    assert rows["static distant vehicle"]["peak_level"] in ("SAFE", "CAUTION")
