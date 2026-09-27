"""Driver-perception tests.

All inputs are software-test fixtures (blank frames and hand-built landmark
arrays). They are not part of the hackathon dataset and are never written to it.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config.driver_config import DriverConfig, load_driver_config
from app.driver import (
    DriverPerceptionPipeline,
    DriverState,
    DrowsinessLevel,
    FaceLandmarks,
    FaceResult,
    FaceStatus,
    HandState,
    HandStateResult,
    NullFaceProvider,
    UnknownHandStateProvider,
)
from app.driver.drowsiness import score_drowsiness
from app.driver.eyes import DRIVER_LEFT_EYE, DRIVER_RIGHT_EYE, eye_aspect_ratio, extract_eye_features
from app.driver.face import FaceModelNotFoundError, MediaPipeFaceProvider, validate_frame
from app.driver.head_pose import (
    FACE_MODEL_3D,
    POSE_LANDMARKS,
    angles_to_rotation,
    camera_matrix,
    estimate_head_pose,
)
from app.driver.models import DriverActivity
from app.driver.state_duration import StateDurationTracker
from app.driver.temporal import EyeClosureFeatures, EyeClosureTracker

W, H = 640, 480
FRAME = np.zeros((H, W, 3), dtype=np.uint8)
CONFIG = DriverConfig(face_model_path=Path("unused.task"))


# --- fixture builders (test-only) ------------------------------------------------

def make_landmarks(yaw=0.0, pitch=0.0, roll=0.0, left_ear=0.30, right_ear=0.30) -> FaceLandmarks:
    """478 landmarks: pose points projected from a known head rotation, eyes with a known EAR."""
    pts = np.full((478, 3), 0.5)
    rvec, _ = cv2.Rodrigues(angles_to_rotation(yaw, pitch, roll))
    img, _ = cv2.projectPoints(FACE_MODEL_3D, rvec, np.array([0.0, 0.0, 1500.0]), camera_matrix(W, H), np.zeros(4))
    for i, idx in enumerate(POSE_LANDMARKS):
        pts[idx, :2] = img[i, 0] / [W, H]
    # Eyes: 40 px wide, anchored at the projected OUTER corners (landmarks 33 and 263 are
    # shared with the pose points), extending toward the face centre. EAR = 4h / 80 = h / 20.
    outer_left_img = img[POSE_LANDMARKS.index(33), 0]
    outer_right_img = img[POSE_LANDMARKS.index(263), 0]
    for indices, ear, cx, cy in (
        (DRIVER_RIGHT_EYE, right_ear, outer_left_img[0] + 20, outer_left_img[1]),
        (DRIVER_LEFT_EYE, left_ear, outer_right_img[0] - 20, outer_right_img[1]),
    ):
        h = ear * 20.0
        eye = [(cx - 20, cy), (cx - 7, cy - h), (cx + 7, cy - h), (cx + 20, cy), (cx + 7, cy + h), (cx - 7, cy + h)]
        for idx, (x, y) in zip(indices, eye):
            if idx not in POSE_LANDMARKS:
                pts[idx, :2] = (x / W, y / H)
    return FaceLandmarks(pts, W, H)


class ScriptedFaceProvider:
    """Returns a queued FaceResult per call (default: open-eyed frontal face)."""

    def __init__(self, results=None):
        self.results = list(results or [])
        self.resets = 0

    def process(self, frame, timestamp):
        if self.results:
            return self.results.pop(0)
        return FaceResult(FaceStatus.OK, make_landmarks())

    def reset(self):
        self.resets += 1

    def close(self):
        pass


def face(closed=False, **kw) -> FaceResult:
    ear = 0.05 if closed else 0.30
    return FaceResult(FaceStatus.OK, make_landmarks(left_ear=ear, right_ear=ear, **kw))


NO_FACE = FaceResult(FaceStatus.NO_FACE)


def run(pipeline, faces, fps=10.0, start=0.0):
    """Feed a sequence of FaceResults at a fixed frame rate; return the last DriverState."""
    pipeline.face_provider.results = list(faces)
    state = None
    for i in range(len(faces)):
        state = pipeline.process(FRAME, start + i / fps)
    return state


@pytest.fixture
def pipeline():
    return DriverPerceptionPipeline(ScriptedFaceProvider(), CONFIG)


# --- 1. DriverState construction ---------------------------------------------------

def test_driver_state_to_dict_has_stable_fields(pipeline):
    state = pipeline.process(FRAME, 0.0)
    d = state.to_dict()
    for key in ["timestamp", "hand_state", "driver_activity", "face_detected", "head_yaw", "head_pitch",
                "head_roll", "left_eye_openness", "right_eye_openness", "eye_closure_ratio",
                "eye_closed_duration", "drowsiness_score", "distraction_duration", "observation_quality"]:
        assert key in d
    nested = state.to_nested_dict()
    assert set(nested) == {"timestamp", "observation", "temporal", "hand", "observation_quality",
                           "driver_activity", "activity_source", "activity_reason"}  # observed + inferred layers
    assert nested["hand"]["hand_state"] == "UNKNOWN"


# --- 2. Hand-state placeholder + plug-in ---------------------------------------------

def test_unknown_hand_provider_never_fabricates():
    result = UnknownHandStateProvider().predict(FRAME)
    assert result.hand_state is HandState.UNKNOWN
    assert result.activity is DriverActivity.UNKNOWN
    assert result.confidence is None


def test_pipeline_output_unknown_hand_state_and_no_distraction(pipeline):
    d = run(pipeline, [face()] * 20).to_dict()
    assert d["hand_state"] == "UNKNOWN"
    assert d["distraction_duration"] is None  # unknown is not "not distracted"


def test_any_provider_plugs_in_without_pipeline_changes():
    class ScriptedHands:
        name = "scripted"

        def __init__(self, labels):
            self.labels = list(labels)

        def predict(self, frame):
            return HandStateResult(HandState(self.labels.pop(0)), DriverActivity.NORMAL, 0.9, self.name)

        def reset(self):
            pass

    labels = ["BOTH_HANDS"] * 2 + ["NO_HANDS"] * 3
    p = DriverPerceptionPipeline(ScriptedFaceProvider(), CONFIG, hand_provider=ScriptedHands(labels))
    state = run(p, [face()] * 5, fps=1.0)
    assert state.hand.hand_state is HandState.NO_HANDS
    assert state.temporal.distraction_duration == pytest.approx(2.0)


def test_failing_hand_provider_degrades_to_unknown():
    class Broken:
        name = "broken"

        def predict(self, frame):
            raise RuntimeError("model file missing")

    p = DriverPerceptionPipeline(ScriptedFaceProvider(), CONFIG, hand_provider=Broken())
    assert p.process(FRAME, 0.0).hand.hand_state is HandState.UNKNOWN


# --- 3. No-face / invalid input -------------------------------------------------------

@pytest.mark.parametrize("status", [FaceStatus.NO_FACE, FaceStatus.PROVIDER_UNAVAILABLE])
def test_no_face_gives_none_measurements(pipeline, status):
    s = run(pipeline, [FaceResult(status)])
    o = s.observation
    assert not o.face_detected
    assert (o.head_yaw, o.head_pitch, o.head_roll) == (None, None, None)
    assert (o.left_eye_openness, o.right_eye_openness, o.eyes_closed) == (None, None, None)
    assert s.observation_quality == 0.0


def test_no_face_is_not_drowsy_or_distracted(pipeline):
    s = run(pipeline, [face()] * 60 + [NO_FACE] * 100)  # 6 s open, then 10 s no face
    assert s.temporal.eye_closed_duration is None
    assert s.temporal.face_missing_duration == pytest.approx(10.0, abs=0.11)
    assert s.temporal.drowsiness_level in (DrowsinessLevel.LOW, DrowsinessLevel.UNKNOWN)
    assert (s.temporal.drowsiness_score or 0.0) < CONFIG.drowsiness_high_threshold
    assert s.temporal.distraction_duration is None


@pytest.mark.parametrize("bad", [None, "not a frame", np.zeros((0, 0, 3), np.uint8), np.zeros((4, 4, 3), np.uint8), np.zeros((50, 50, 5), np.uint8)])
def test_invalid_frames_do_not_crash(bad):
    frame, reason = validate_frame(bad)
    assert frame is None and reason
    p = DriverPerceptionPipeline(NullFaceProvider(), CONFIG)
    s = p.process(bad, 0.0)
    assert s.observation.face_status is FaceStatus.INVALID_FRAME and not s.observation.face_detected


def test_grayscale_and_bgra_frames_accepted():
    assert validate_frame(np.zeros((40, 40), np.uint8))[0].shape == (40, 40, 3)
    assert validate_frame(np.zeros((40, 40, 4), np.uint8))[0].shape == (40, 40, 3)


def test_face_provider_exception_is_contained():
    class Exploding:
        def process(self, frame, t):
            raise RuntimeError("boom")

    s = DriverPerceptionPipeline(Exploding(), CONFIG).process(FRAME, 0.0)
    assert s.observation.face_status is FaceStatus.PROVIDER_ERROR


def test_partial_face_eye_out_of_frame_gives_none_for_that_eye():
    lm = make_landmarks()
    pts = lm.points.copy()
    pts[DRIVER_LEFT_EYE, 0] = 1.2  # driver's left eye pushed outside the image
    feats = extract_eye_features(FaceLandmarks(pts, W, H))
    assert feats.left_openness is None and feats.right_openness is not None
    assert feats.mean_openness == feats.right_openness


def test_mediapipe_provider_missing_model_explains(tmp_path):
    with pytest.raises(FaceModelNotFoundError, match="face_landmarker.task"):
        MediaPipeFaceProvider(tmp_path / "missing.task")


# --- 4. Eye features ------------------------------------------------------------------

def test_eye_aspect_ratio_formula():
    pts = np.array([[0, 0], [7, -3], [13, -3], [20, 0], [13, 3], [7, 3]], float)
    assert eye_aspect_ratio(pts) == pytest.approx((6 + 6) / 40)
    assert eye_aspect_ratio(np.zeros((6, 2))) is None


def test_eye_features_left_right_and_threshold():
    feats = extract_eye_features(make_landmarks(left_ear=0.32, right_ear=0.10))
    assert feats.left_openness == pytest.approx(0.32, abs=1e-3)
    assert feats.right_openness == pytest.approx(0.10, abs=1e-3)
    assert feats.mean_openness == pytest.approx(0.21, abs=1e-3)
    assert feats.eyes_closed(0.25) is True and feats.eyes_closed(0.20) is False


# --- head pose (supports the observation layer) ----------------------------------------

@pytest.mark.parametrize("yaw,pitch,roll", [(0, 0, 0), (25, 0, 0), (-30, 0, 0), (0, 15, 0), (0, -15, 0), (0, 0, 12), (20, -10, 8)])
def test_head_pose_recovers_known_rotation(yaw, pitch, roll):
    pose = estimate_head_pose(make_landmarks(yaw, pitch, roll))
    assert pose.yaw == pytest.approx(yaw, abs=1.0)
    assert pose.pitch == pytest.approx(pitch, abs=1.0)
    assert pose.roll == pytest.approx(roll, abs=1.0)


def test_head_pose_sign_convention_matches_image_geometry():
    def nose_offset(lm):
        eyes_mid = (lm.pixel_xy([33])[0] + lm.pixel_xy([263])[0]) / 2
        return lm.pixel_xy([1])[0] - eyes_mid

    assert nose_offset(make_landmarks(yaw=20))[0] > 0  # yaw > 0: nose toward image right
    assert nose_offset(make_landmarks(pitch=20))[1] > nose_offset(make_landmarks())[1]  # pitch > 0: nose drops
    lm = make_landmarks(roll=15)
    assert lm.pixel_xy([263])[0][1] > lm.pixel_xy([33])[0][1]  # roll > 0: clockwise


def test_head_pose_none_when_pose_points_outside_frame():
    lm = make_landmarks()
    pts = lm.points.copy()
    pts[152, 1] = 1.5  # chin below the image
    assert estimate_head_pose(FaceLandmarks(pts, W, H)) is None


# --- 5. Temporal eye closure -------------------------------------------------------------

def tracker():
    return EyeClosureTracker(window_seconds=60, max_gap_seconds=0.5, blink_max_seconds=0.5, long_closure_seconds=1.0)


def feed(tr, pattern, fps=10.0, start=0.0):
    f = None
    for i, closed in enumerate(pattern):
        f = tr.update(start + i / fps, closed)
    return f


def test_blink_vs_sustained_closure():
    tr = tracker()
    f = feed(tr, [False] * 10 + [True] * 2 + [False] * 10)  # 0.2 s closure
    assert (f.blink_count, f.long_closure_count) == (1, 0)
    f = feed(tr, [True] * 15 + [False], start=10.0)  # 1.5 s closure
    assert f.long_closure_count == 1


def test_closed_duration_and_ratio():
    f = feed(tracker(), [False] * 50 + [True] * 21)  # 5 s open then 2 s closed
    assert f.eye_closed_duration == pytest.approx(2.0, abs=0.01)
    assert f.eye_closure_ratio == pytest.approx(2.0 / 7.0, abs=0.02)
    assert f.observed_seconds == pytest.approx(7.0, abs=0.01)


def test_missing_observations_are_not_closed_time():
    tr = tracker()
    f = feed(tr, [False] * 20 + [None] * 30 + [False] * 20)
    assert f.eye_closure_ratio == 0.0
    assert f.observed_seconds == pytest.approx(3.9, abs=0.2)


def test_long_face_loss_breaks_closure_run_without_event():
    tr = tracker()
    f = feed(tr, [True] * 5 + [None] * 20 + [True] * 3 + [False])
    assert f.long_closure_count == 0  # the 0.5 s closure before the loss is not merged


def test_window_forgets_old_events():
    tr = tracker()
    feed(tr, [True] * 15 + [False] * 5)  # long closure near t=1.5 s
    f = feed(tr, [False] * 10, start=70.0)  # 70 s later: outside the 60 s window
    assert f.long_closure_count == 0


# --- 6. Drowsiness scoring -----------------------------------------------------------------

def features(**kw) -> EyeClosureFeatures:
    base = dict(eye_closure_ratio=0.0, eye_closed_duration=0.0, blink_count=0, long_closure_count=0,
                observed_seconds=30.0, face_missing_duration=0.0)
    base.update(kw)
    return EyeClosureFeatures(**base)


def test_single_closed_frame_is_not_drowsy():
    score, level = score_drowsiness(features(eye_closed_duration=0.1, eye_closure_ratio=0.01), CONFIG)
    assert score < CONFIG.drowsiness_high_threshold and level is DrowsinessLevel.LOW


def test_sustained_closure_is_critical():
    score, level = score_drowsiness(features(eye_closed_duration=2.5), CONFIG)
    assert score == 1.0 and level is DrowsinessLevel.CRITICAL


def test_frequent_long_closures_raise_score():
    score, level = score_drowsiness(features(eye_closure_ratio=0.15, long_closure_count=3), CONFIG)
    assert score == pytest.approx(0.6 * (0.15 / 0.3) + 0.4 * 1.0, abs=1e-3)  # 0.7
    assert level is DrowsinessLevel.HIGH


def test_too_little_history_is_unknown_not_low():
    score, level = score_drowsiness(features(observed_seconds=1.0), CONFIG)
    assert score is None and level is DrowsinessLevel.UNKNOWN


def test_thresholds_are_configurable():
    strict = replace(CONFIG, drowsiness_high_threshold=0.1, drowsiness_critical_threshold=0.2)
    _, level = score_drowsiness(features(eye_closure_ratio=0.06), strict)
    assert level is DrowsinessLevel.HIGH


def test_pipeline_end_to_end_drowsy_sequence(pipeline):
    s = run(pipeline, [face()] * 60 + [face(closed=True)] * 25)  # 6 s open, 2.4 s closed
    assert s.temporal.drowsiness_level is DrowsinessLevel.CRITICAL
    assert s.temporal.eye_closed_duration == pytest.approx(2.4, abs=0.11)


def test_pipeline_normal_blinking_stays_low(pipeline):
    seq = ([face()] * 38 + [face(closed=True)] * 2) * 4 + [face()]  # a 0.2 s blink every 4 s
    s = run(pipeline, seq)
    assert s.temporal.blink_count == 4
    assert s.temporal.drowsiness_level is DrowsinessLevel.LOW


# --- 7. Distraction-duration tracking ---------------------------------------------------------

def test_state_duration_example_from_spec():
    tr = StateDurationTracker(CONFIG.distraction_states)
    for t, s in [(0, "BOTH_HANDS"), (2, "ONE_HAND"), (5, "NO_HANDS"), (8, "BOTH_HANDS")]:
        tr.update(t, s)
    totals = tr.durations_by_state()
    assert totals["ONE_HAND"] == 3 and totals["NO_HANDS"] == 3
    assert [x.to_state for x in tr.transitions] == ["BOTH_HANDS", "ONE_HAND", "NO_HANDS", "BOTH_HANDS"]
    assert tr.distraction_duration == 0.0


def test_distraction_run_spans_distracting_states():
    tr = StateDurationTracker(CONFIG.distraction_states)
    for t, s in [(0, "BOTH_HANDS"), (1, "NO_HANDS"), (3, "PHONE"), (6, "PHONE")]:
        tr.update(t, s)
    assert tr.distraction_duration == 5.0  # NO_HANDS -> PHONE is one continuous distraction
    assert tr.current_duration == 3.0


def test_unknown_ends_distraction_and_reports_none():
    tr = StateDurationTracker(CONFIG.distraction_states)
    tr.update(0, "NO_HANDS")
    tr.update(2, "UNKNOWN")
    assert tr.distraction_duration is None
    tr.update(3, "NO_HANDS")
    assert tr.distraction_duration == 0.0  # restarted, not carried over


def test_activity_overrides_hand_state_for_distraction():
    from app.driver.pipeline import manual_state

    assert manual_state(HandStateResult(HandState.BOTH_HANDS, DriverActivity.PHONE)) == "PHONE"
    assert manual_state(HandStateResult(HandState.ONE_HAND, DriverActivity.NORMAL)) == "ONE_HAND"


# --- 8. Reset ----------------------------------------------------------------------------------

def test_reset_clears_all_temporal_state(pipeline):
    run(pipeline, [face()] * 60 + [face(closed=True)] * 25)
    pipeline.reset()
    assert pipeline.face_provider.resets == 1
    assert pipeline.manual_tracker.current_state is None
    s = pipeline.process(FRAME, 0.0)  # timestamps may restart after reset
    t = s.temporal
    assert (t.eye_closure_ratio, t.long_closure_count, t.drowsiness_score, t.timeline_reset) == (None, 0, None, False)


# --- 9. Timestamps ----------------------------------------------------------------------------------

def test_backwards_timestamp_auto_resets_and_flags(pipeline):
    run(pipeline, [face()] * 20, start=100.0)
    s = pipeline.process(FRAME, 5.0)
    assert s.temporal.timeline_reset is True
    assert s.temporal.observed_seconds == 0.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, "12", True])
def test_invalid_timestamps_rejected(pipeline, bad):
    with pytest.raises(ValueError):
        pipeline.process(FRAME, bad)


def test_missing_timestamp_uses_monotonic_clock(pipeline):
    a = pipeline.process(FRAME).timestamp
    b = pipeline.process(FRAME).timestamp
    assert b >= a


def test_equal_timestamps_are_accepted(pipeline):
    pipeline.process(FRAME, 1.0)
    assert pipeline.process(FRAME, 1.0).temporal.timeline_reset is False


# --- 10. Pipeline output structure ---------------------------------------------------------------------

def test_pipeline_output_structure(pipeline):
    s = run(pipeline, [face(yaw=15)] * 3)
    assert isinstance(s, DriverState)
    o = s.observation
    assert o.face_detected and o.landmarks_available
    assert o.head_yaw == pytest.approx(15, abs=1.0)
    assert o.eyes_closed is False
    assert s.observation_quality == 1.0
    assert s.hand.provider == "unknown_placeholder"


# --- configuration -------------------------------------------------------------------------------------

def test_driver_config_env_overrides(monkeypatch):
    monkeypatch.setattr("app.config.driver_config.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("EYE_CLOSURE_THRESHOLD", "0.18")
    monkeypatch.setenv("LONG_CLOSURE_EVENTS_MAX", "4")
    monkeypatch.setenv("DISTRACTION_STATES", "no_hands, phone")
    monkeypatch.setenv("FACE_MODEL_PATH", "models/face/custom.task")
    c = load_driver_config()
    assert c.eye_closure_threshold == 0.18 and c.long_closure_events_max == 4
    assert c.distraction_states == ("NO_HANDS", "PHONE")
    assert c.face_model_path.name == "custom.task" and c.face_model_path.is_absolute()


@pytest.mark.parametrize("key,value", [("DROWSINESS_HIGH_THRESHOLD", "0.9"), ("EYE_CLOSURE_THRESHOLD", "abc"), ("BLINK_MAX_SECONDS", "5")])
def test_driver_config_rejects_bad_values(monkeypatch, key, value):
    monkeypatch.setattr("app.config.driver_config.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        load_driver_config()
