"""Hand-state classifier integration into DriverPerceptionPipeline.

Most tests use a scripted fake classifier, so no torch or trained model is needed. Tests
marked `real_model` load outputs/models/hand_state_best.pt (or HAND_MODEL_PATH) and are
skipped when it or torch is not available. Fixture images and videos are software-test
data, not part of the dataset.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.config.driver_config import DriverConfig
from app.driver import (
    DriverActivity,
    DriverPerceptionPipeline,
    FaceResult,
    FaceStatus,
    HandState,
    HandStateResult,
    TrainedHandStateProvider,
    UnknownHandStateProvider,
)
from app.driver.state_duration import StateDurationTracker
from test_driver_perception import make_landmarks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = DriverConfig(face_model_path=Path("unused.task"))  # no_hands_inference_seconds = 2.0
FRAME = np.zeros((480, 640, 3), np.uint8)


class FakeClassifier:
    """Stands in for HandStateClassifier: returns scripted predictions in order and counts calls."""

    def __init__(self, script, threshold=0.60):
        self.script = list(script)
        self.threshold = threshold
        self.calls = 0

    def predict(self, frame, color_order="BGR"):
        self.calls += 1
        state, conf = self.script.pop(0) if self.script else ("BOTH_HANDS", 0.95)
        probs = {"BOTH_HANDS": 0.0, "ONE_HAND": 0.0, "NO_HANDS": 0.0}
        top = state if state != "UNKNOWN" else "ONE_HAND"
        probs[top] = conf
        return {"hand_state": state, "confidence": conf, "class_probabilities": probs}


class OpenFaceProvider:
    def process(self, frame, timestamp):
        return FaceResult(FaceStatus.OK, make_landmarks())

    def reset(self):
        pass

    def close(self):
        pass


def pipeline_with(script, config=CONFIG):
    clf = FakeClassifier(script, config.hand_confidence_threshold)
    return DriverPerceptionPipeline(OpenFaceProvider(), config, hand_provider=TrainedHandStateProvider(clf)), clf


def run(pipeline, n, fps=10.0, start=0.0):
    return [pipeline.process(FRAME, start + i / fps) for i in range(n)]


# --- C, D: confidence and provider are returned ------------------------------------------------

def test_provider_returns_state_confidence_provider_and_no_activity():
    provider = TrainedHandStateProvider(FakeClassifier([("NO_HANDS", 0.91)]))
    r = provider.predict(FRAME)
    assert (r.hand_state, r.confidence, r.provider) == (HandState.NO_HANDS, 0.91, "mobilenet_v3_small")
    assert r.activity is DriverActivity.UNKNOWN  # the hand classifier never reports an activity
    assert r.class_probabilities["NO_HANDS"] == 0.91


@pytest.mark.parametrize("state", ["BOTH_HANDS", "ONE_HAND", "NO_HANDS", "UNKNOWN"])
def test_all_four_states_reach_driver_state(state):
    p, _ = pipeline_with([(state, 0.55 if state == "UNKNOWN" else 0.9)])
    d = p.process(FRAME, 0.0).to_dict()
    assert d["hand_state"] == state
    assert d["hand_state_confidence"] == (0.55 if state == "UNKNOWN" else 0.9)
    assert d["hand_state_provider"] == "mobilenet_v3_small"
    for key in ["driver_activity", "manual_state_duration", "distraction_duration", "activity_source"]:
        assert key in d


def test_invalid_frame_gives_unknown_without_calling_model():
    clf = FakeClassifier([])
    r = TrainedHandStateProvider(clf).predict(None)
    assert r.hand_state is HandState.UNKNOWN and r.provider == "mobilenet_v3_small" and clf.calls == 0


# --- E: identical consecutive states accumulate duration -----------------------------------------

def test_consecutive_identical_states_accumulate():
    p, _ = pipeline_with([("ONE_HAND", 0.9)] * 31)
    states = run(p, 31)  # 0.0 .. 3.0 s
    assert [s.temporal.manual_state_duration for s in states[::10]] == [0.0, 1.0, 2.0, 3.0]
    assert states[-1].temporal.longest_one_hand_duration == 3.0
    assert states[-1].temporal.manual_state_transitions == 0


# --- F: a transition resets the relevant duration -----------------------------------------------

def test_transition_resets_current_duration_keeps_longest():
    p, _ = pipeline_with([("ONE_HAND", 0.9)] * 21 + [("BOTH_HANDS", 0.9)] * 5)
    s = run(p, 26)[-1]  # ONE_HAND 0.0-2.0, BOTH_HANDS from 2.1
    assert s.hand.hand_state is HandState.BOTH_HANDS
    assert s.temporal.manual_state_duration == pytest.approx(0.4)
    assert s.temporal.longest_one_hand_duration == pytest.approx(2.1)  # closed at the first BOTH_HANDS frame
    assert s.temporal.manual_state_transitions == 1


# --- G: UNKNOWN never extends a previous ONE_HAND / NO_HANDS state ---------------------------------

def test_unknown_does_not_extend_previous_no_hands():
    script = [("NO_HANDS", 0.9)] * 15 + [("UNKNOWN", 0.45)] * 20 + [("NO_HANDS", 0.9)] * 5
    p, _ = pipeline_with(script)
    states = run(p, 40)
    during_unknown = states[20]
    assert during_unknown.hand.hand_state is HandState.UNKNOWN
    assert during_unknown.driver_activity is DriverActivity.UNKNOWN
    assert during_unknown.temporal.distraction_duration is None
    back = states[-1]  # NO_HANDS resumed at t=3.5, now t=3.9
    assert back.temporal.manual_state_duration == pytest.approx(0.4)  # restarted, not 3.9
    assert back.temporal.longest_no_hands_duration == pytest.approx(1.5)  # 0.0 -> first UNKNOWN frame at 1.5
    assert back.driver_activity is DriverActivity.NORMAL  # 0.4 s < 2.0 s: no distraction inferred


def test_unknown_does_not_reset_face_or_drowsiness_timeline():
    p, _ = pipeline_with([("ONE_HAND", 0.9)] * 30 + [("UNKNOWN", 0.4)] * 30)
    s = run(p, 60)[-1]
    assert s.temporal.timeline_reset is False
    assert s.temporal.observed_seconds == pytest.approx(5.9, abs=0.01)  # eye timeline continuous across UNKNOWN
    assert s.temporal.longest_one_hand_duration == pytest.approx(3.0)  # history kept


def test_tracker_level_unknown_semantics():
    tr = StateDurationTracker(("NO_HANDS",))
    for t, st in [(0, "NO_HANDS"), (1, "NO_HANDS"), (2, "UNKNOWN"), (5, "UNKNOWN"), (6, "NO_HANDS")]:
        tr.update(t, st)
    assert tr.current_duration == 0 and tr.longest_duration("NO_HANDS") == 2
    assert tr.durations_by_state()["UNKNOWN"] == 4
    assert tr.transition_count == 2


# --- observed vs inferred ----------------------------------------------------------------------------

def test_one_hand_is_never_inferred_as_distraction():
    p, _ = pipeline_with([("ONE_HAND", 0.95)] * 200)
    s = run(p, 200)[-1]  # 19.9 s of ONE_HAND
    assert s.driver_activity is DriverActivity.NORMAL and s.temporal.distraction_duration == 0.0


def test_no_hands_needs_sustained_observation_before_inference():
    p, _ = pipeline_with([("NO_HANDS", 0.95)] * 31)
    states = run(p, 31)
    first = states[0]
    assert first.hand.hand_state is HandState.NO_HANDS  # observed immediately
    assert first.driver_activity is DriverActivity.NORMAL  # but nothing inferred from one frame
    assert first.temporal.distraction_duration == 0.0
    assert states[19].driver_activity is DriverActivity.NORMAL  # 1.9 s
    at_2s = states[20]
    assert at_2s.driver_activity is DriverActivity.HANDS_OFF_WHEEL
    assert at_2s.activity_source == "inferred_temporal" and "NO_HANDS observed continuously" in at_2s.activity_reason
    assert states[-1].temporal.distraction_duration == pytest.approx(3.0)


def test_inference_threshold_is_configurable():
    p, _ = pipeline_with([("NO_HANDS", 0.95)] * 6, replace(CONFIG, no_hands_inference_seconds=0.5))
    assert run(p, 6)[-1].driver_activity is DriverActivity.HANDS_OFF_WHEEL


def test_phone_is_never_inferred_from_hand_state():
    p, _ = pipeline_with([("NO_HANDS", 0.99)] * 100)
    assert all(s.driver_activity is not DriverActivity.PHONE for s in run(p, 100))


def test_observed_activity_from_another_provider_is_kept_separate():
    class PhoneDetector:
        name = "phone_detector"

        def predict(self, frame):
            return HandStateResult(HandState.ONE_HAND, DriverActivity.PHONE, 0.8, self.name)

        def reset(self):
            pass

    p = DriverPerceptionPipeline(OpenFaceProvider(), CONFIG, hand_provider=PhoneDetector())
    s = p.process(FRAME, 0.0)
    assert s.driver_activity is DriverActivity.PHONE and s.activity_source == "observed_provider"
    assert s.hand.hand_state is HandState.ONE_HAND


def test_placeholder_provider_still_works():
    s = DriverPerceptionPipeline(OpenFaceProvider(), CONFIG, hand_provider=UnknownHandStateProvider()).process(FRAME, 0.0)
    assert s.hand.hand_state is HandState.UNKNOWN and s.driver_activity is DriverActivity.UNKNOWN


def test_face_head_eye_signals_unchanged_by_hand_model():
    p, _ = pipeline_with([("NO_HANDS", 0.9)])
    s = p.process(FRAME, 0.0)
    assert s.observation.face_detected and s.observation.head_yaw is not None and s.observation.left_eye_openness is not None


def test_reset_clears_hand_timeline():
    p, _ = pipeline_with([("NO_HANDS", 0.9)] * 40)
    run(p, 30)
    p.reset()
    s = p.process(FRAME, 0.0)
    assert s.temporal.manual_state_duration == 0.0 and s.temporal.longest_no_hands_duration == 0.0


# --- performance: classifier runs only on sampled frames -----------------------------------------------

def _smoke_script():
    spec = importlib.util.spec_from_file_location("driver_smoke_hand", PROJECT_ROOT / "scripts" / "test_driver_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = _smoke_script()


def _write_video(path: Path, seconds=3.0, fps=30.0):
    import cv2

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for _ in range(round(seconds * fps)):
        w.write(np.full((48, 64, 3), 200, np.uint8))
    w.release()
    return path


def test_classifier_runs_at_sampling_rate_not_camera_fps(tmp_path):
    video = _write_video(tmp_path / "v.avi")
    p, clf = pipeline_with([("ONE_HAND", 0.9)] * 1000)
    reader = smoke.VideoReader(video)
    try:
        r = smoke.run_video(reader, p, sample_fps=10)
    finally:
        reader.release()
    assert r.frames_read == 90
    assert clf.calls == r.frames_processed == pytest.approx(30, abs=1)


def test_video_summary_and_timeline(tmp_path, capsys):
    video = _write_video(tmp_path / "v.avi", seconds=4.0)
    clf = FakeClassifier([("BOTH_HANDS", 0.9)] * 10 + [("NO_HANDS", 0.9)] * 30)
    code = smoke.video_smoke_test(video, CONFIG, 10, None, None, True, provider=OpenFaceProvider(),
                                  hand_provider=TrainedHandStateProvider(clf), timeline_interval=1.0)
    out = capsys.readouterr().out
    assert code == 0
    assert "TIMELINE" in out and "hand_state" in out and "head_pitch" in out
    rows = [ln for ln in out.splitlines() if ln.strip()[:1].isdigit() and "|" in ln]
    assert 4 <= len(rows) <= 10  # compact: ~1 per second + state changes, not one per frame
    assert "HAND STATE / MANUAL CONTROL" in out and "Inferred distraction episodes  1" in out
    assert "Longest continuous NO_HANDS" in out


def test_build_hand_provider_missing_checkpoint(tmp_path):
    cfg = replace(CONFIG, hand_model_path=tmp_path / "none.pt")
    provider, note, error = smoke.build_hand_provider(cfg)
    assert isinstance(provider, UnknownHandStateProvider) and "not found" in note and error is None
    provider, note, error = smoke.build_hand_provider(cfg, explicit=True)
    assert provider is None and "not found" in error


# --- A, B: real trained checkpoint (skipped if unavailable) ------------------------------------------------

REAL_CKPT = Path(os.environ.get("HAND_MODEL_PATH") or PROJECT_ROOT / "outputs" / "models" / "hand_state_best.pt")
real_model = pytest.mark.skipif(
    not REAL_CKPT.is_file() or importlib.util.find_spec("torch") is None,
    reason=f"trained checkpoint {REAL_CKPT} or torch not available",
)


@pytest.fixture(scope="module")
def real_provider():
    return TrainedHandStateProvider.from_checkpoint(REAL_CKPT, threshold=0.60, device="cpu")


@real_model
def test_real_checkpoint_loads_once(real_provider):
    meta = real_provider.classifier.meta
    assert meta["arch"] == "mobilenet_v3_small" and meta["class_names"] == ["both_hands_on_steering", "one_hand_on_steering", "no_hands"]
    assert real_provider.classifier.device == "cpu"


@real_model
def test_real_checkpoint_predicts_a_valid_state(real_provider):
    frame = np.random.default_rng(0).integers(0, 255, (1920, 1080, 3), dtype=np.uint8)
    r = real_provider.predict(frame)
    assert r.hand_state.value in {"BOTH_HANDS", "ONE_HAND", "NO_HANDS", "UNKNOWN"}
    assert 0.0 <= r.confidence <= 1.0 and r.provider == "mobilenet_v3_small"
    assert sum(r.class_probabilities.values()) == pytest.approx(1.0, abs=1e-3)


def _real_test_images():
    root = os.environ.get("DATASET_ROOT")
    manifest = PROJECT_ROOT / "data" / "processed" / "hand_test.json"
    if not root or not manifest.is_file():
        return []
    records = json.loads(manifest.read_text(encoding="utf-8"))
    return [(Path(root) / r["path"], r["hand_state"]) for r in records if (Path(root) / r["path"]).is_file()]


@real_model
@pytest.mark.skipif(not _real_test_images(), reason="DATASET_ROOT or data/processed/hand_test.json not available")
def test_real_labelled_images_through_the_pipeline(real_provider):
    """Integration sanity check on real labelled images from hand_test.json.

    Not a held-out evaluation: after a re-split, the current checkpoint may have trained on some of
    these images. Held-out metrics come from scripts/evaluate_hand_model.py after (re)training.
    """
    from app.driver.hand_dataset import load_training_image

    images = _real_test_images()
    p = DriverPerceptionPipeline(OpenFaceProvider(), CONFIG, hand_provider=real_provider)
    correct = 0
    for i, (path, expected) in enumerate(images):
        rgb = np.asarray(load_training_image(path))
        s = p.process(rgb[:, :, ::-1].copy(), float(i))  # BGR, like an OpenCV frame
        assert s.hand.hand_state.value in {"BOTH_HANDS", "ONE_HAND", "NO_HANDS", "UNKNOWN"}
        correct += s.hand.hand_state.value == expected
    assert correct / len(images) >= 0.8  # sanity check only; the report has the real metrics
