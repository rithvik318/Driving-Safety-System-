"""Tests for the --video smoke test in scripts/test_driver_pipeline.py.

Videos here are tiny software-test fixtures written to a temp dir: frame brightness
encodes what a fake face provider should report (bright = eyes open, dim = eyes
closed, black = no face). They are not dataset samples. No face model is needed.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config.driver_config import DriverConfig
from app.driver import DriverPerceptionPipeline, DrowsinessLevel, FaceResult, FaceStatus
from test_driver_perception import make_landmarks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = DriverConfig(face_model_path=Path("unused.task"))
OPEN, CLOSED, NOFACE = 220, 110, 0


def load_script():
    spec = importlib.util.spec_from_file_location("driver_smoke", PROJECT_ROOT / "scripts" / "test_driver_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses in the script need the module registered
    spec.loader.exec_module(module)
    return module


smoke = load_script()


class BrightnessFaceProvider:
    """Fake landmark provider: reads the frame's mean brightness instead of finding a face."""

    def process(self, frame, timestamp):
        mean = float(frame.mean())
        if mean < 40:
            return FaceResult(FaceStatus.NO_FACE)
        ear = 0.30 if mean > 165 else 0.05
        return FaceResult(FaceStatus.OK, make_landmarks(left_ear=ear, right_ear=ear))

    def reset(self):
        pass

    def close(self):
        pass


def write_video(path: Path, segments, fps=30.0, size=(64, 48)) -> Path:
    """segments: list of (brightness, seconds)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    for value, seconds in segments:
        for _ in range(round(seconds * fps)):
            writer.write(np.full((size[1], size[0], 3), value, np.uint8))
    writer.release()
    return path


def pipeline(config=CONFIG):
    return DriverPerceptionPipeline(BrightnessFaceProvider(), config)


def run(path, config=CONFIG, **kw):
    reader = smoke.VideoReader(path)
    try:
        return smoke.run_video(reader, pipeline(config), **kw)
    finally:
        reader.release()


# --- temporal behaviour through a real decoded video ---------------------------------------

def test_sustained_closure_video_reaches_critical(tmp_path):
    video = write_video(tmp_path / "drowsy.avi", [(OPEN, 6.0), (CLOSED, 2.5)])
    r = run(video, sample_fps=10)
    s = r.final_state.temporal
    assert r.info.fps == 30 and r.info.frame_count == 255 and r.frames_read == 255
    assert r.frames_processed == pytest.approx(85, abs=2)
    assert r.face_pct == 100.0
    assert s.drowsiness_level is DrowsinessLevel.CRITICAL
    assert r.max_eye_closed_duration == pytest.approx(2.4, abs=0.15)
    assert r.timestamp_sources.get("container", 0) == 255  # real container timestamps used
    assert r.timeline_resets == 0


def test_sampling_rate_does_not_change_elapsed_time(tmp_path):
    video = write_video(tmp_path / "v.avi", [(OPEN, 6.0), (CLOSED, 2.0)])
    every = run(video, sample_fps=0).final_state.temporal
    sampled = run(video, sample_fps=10).final_state.temporal
    assert every.observed_seconds == pytest.approx(sampled.observed_seconds, abs=0.15)
    assert every.eye_closed_duration == pytest.approx(sampled.eye_closed_duration, abs=0.15)
    assert every.eye_closure_ratio == pytest.approx(sampled.eye_closure_ratio, abs=0.02)


def test_whole_video_totals_outlive_the_window(tmp_path):
    blink_cycle = [(OPEN, 1.8), (CLOSED, 0.2)]
    long_cycle = [(OPEN, 2.0), (CLOSED, 1.2)]
    video = write_video(tmp_path / "blinks.avi", blink_cycle * 6 + long_cycle * 2 + [(OPEN, 1.0)])
    short_window = replace(CONFIG, drowsiness_window_seconds=5.0, min_observed_seconds=2.0)
    r = run(video, config=short_window, sample_fps=10)
    assert r.total_blinks == 6 and r.total_long_closures == 2
    assert r.final_state.temporal.blink_count < 6  # the 5 s window has forgotten the early blinks
    assert r.max_eye_closed_duration == pytest.approx(1.2, abs=0.15)


def test_no_face_for_entire_video(tmp_path):
    video = write_video(tmp_path / "empty.avi", [(NOFACE, 3.0)])
    r = run(video, sample_fps=10)
    s = r.final_state.temporal
    assert r.face_pct == 0.0 and r.face_status == {"NO_FACE": r.frames_processed}
    assert s.drowsiness_score is None and s.drowsiness_level is DrowsinessLevel.UNKNOWN
    assert r.total_blinks == 0
    assert "No face was detected in any processed frame" in smoke.format_summary(r)


def test_hand_state_stays_unknown_and_distraction_unavailable(tmp_path):
    r = run(write_video(tmp_path / "v.avi", [(OPEN, 1.0)]), sample_fps=10)
    assert r.final_state.hand.hand_state.value == "UNKNOWN"
    assert r.max_distraction_duration is None
    assert "(hand state UNKNOWN)" in smoke.format_summary(r)


def test_max_seconds_limits_processing(tmp_path):
    r = run(write_video(tmp_path / "v.avi", [(OPEN, 4.0)]), sample_fps=10, max_seconds=1.0)
    assert r.last_t <= 1.0 + 1e-6 and not r.stopped_early


def test_summary_contains_every_requested_field(tmp_path):
    text = smoke.format_summary(run(write_video(tmp_path / "v.avi", [(OPEN, 1.0)]), sample_fps=10))
    for label in ["Video", "Resolution", "FPS", "Total frames", "Processed frames", "Duration", "Face detected",
                  "Hand state", "Driver activity", "Head yaw / pitch / roll", "Eye openness", "Eyes closed",
                  "Eye closure ratio", "Blink count", "Long closure count", "Observed seconds", "Drowsiness score",
                  "Drowsiness level", "Manual state", "Manual state duration", "Distraction duration",
                  "Timeline reset", "Observation quality", "Frames with face", "Longest continuous eye closure",
                  "Total blinks", "Total long closures", "Max drowsiness score", "Max distraction duration"]:
        assert label in text, label


# --- timing edge cases (fake reader) -----------------------------------------------------------

class FakeReader:
    """Mimics VideoReader. `pts` = container timestamps (None = unavailable); `bad` = undecodable indices."""

    def __init__(self, n=20, fps=None, pts=None, bad=()):
        self.n, self._fps, self.pts, self.bad, self.i = n, fps, pts, set(bad), -1

    def info(self):
        return smoke.VideoInfo("fake.mp4", 64, 48, self._fps, self.n)

    def grab(self):
        self.i += 1
        return self.i < self.n

    def position_seconds(self):
        return None if self.pts is None else self.pts[self.i]

    def retrieve(self):
        if self.i in self.bad:
            return False, None
        return True, np.full((48, 64, 3), OPEN, np.uint8)


def test_zero_fps_without_timestamps_fails_clearly():
    with pytest.raises(smoke.VideoError, match="--assume-fps"):
        smoke.run_video(FakeReader(fps=None, pts=None), pipeline(), sample_fps=0)


def test_zero_fps_with_assume_fps_works():
    r = smoke.run_video(FakeReader(n=20, fps=None, pts=None), pipeline(), sample_fps=0, assume_fps=10.0)
    assert r.frames_processed == 20 and r.last_t == pytest.approx(1.9)
    assert r.timestamp_sources == {"index/fps": 20}


def test_zero_fps_with_container_timestamps_works():
    pts = [i * 0.05 for i in range(20)]  # 20 fps timestamps, no FPS header
    r = smoke.run_video(FakeReader(n=20, fps=None, pts=pts), pipeline(), sample_fps=0)
    assert r.last_t == pytest.approx(0.95) and r.timestamp_sources == {"container": 20}


def test_backwards_container_timestamps_never_reset_the_pipeline():
    pts = [0.0, 0.1, 0.2, 0.15, 0.3, 0.4]  # one glitch
    r = smoke.run_video(FakeReader(n=6, fps=10.0, pts=pts), pipeline(), sample_fps=0)
    assert r.timeline_resets == 0 and r.frames_processed == 6


def test_unreadable_frames_are_counted_and_skipped():
    r = smoke.run_video(FakeReader(n=10, fps=10.0, bad={3, 4}), pipeline(), sample_fps=0)
    assert (r.frames_read, r.frames_processed, r.unreadable_frames) == (10, 8, 2)
    assert "could not be decoded" in smoke.format_summary(r)


def test_stream_ending_early_is_reported():
    reader = FakeReader(n=5, fps=10.0)
    reader.info = lambda: smoke.VideoInfo("fake.mp4", 64, 48, 10.0, 100)  # header claims 100 frames
    r = smoke.run_video(reader, pipeline(), sample_fps=0)
    assert r.stopped_early and "Reading stopped after 5" in smoke.format_summary(r)


def test_empty_video_raises():
    with pytest.raises(smoke.VideoError, match="No frames"):
        smoke.run_video(FakeReader(n=0, fps=30.0), pipeline())


def test_too_sparse_sampling_rejected():
    with pytest.raises(smoke.VideoError, match="too sparse"):
        smoke.run_video(FakeReader(fps=30.0), pipeline(), sample_fps=1.0)


# --- CLI ---------------------------------------------------------------------------------------

@pytest.fixture
def no_dotenv(monkeypatch):
    monkeypatch.setattr("app.config.settings.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr("app.config.driver_config.load_dotenv", lambda *a, **k: False)


def test_cli_invalid_path(no_dotenv, tmp_path, capsys):
    assert smoke.main(["--video", str(tmp_path / "missing.mp4")]) == 2
    assert "video not found" in capsys.readouterr().err


def test_cli_corrupt_video(no_dotenv, tmp_path, capsys):
    bad = tmp_path / "corrupt.mp4"
    bad.write_bytes(b"this is not a video" * 100)
    assert smoke.main(["--video", str(bad)]) == 2
    assert "could not open video" in capsys.readouterr().err


def test_cli_missing_face_model(no_dotenv, tmp_path, capsys):
    video = write_video(tmp_path / "v.avi", [(OPEN, 0.5)])
    assert smoke.main(["--video", str(video), "--model", str(tmp_path / "none.task")]) == 2
    assert "face_landmarker.task" in capsys.readouterr().err


def test_cli_rejects_bad_options(no_dotenv, capsys):
    assert smoke.main(["--video", "x.mp4", "--sample-fps", "-1"]) == 2
    assert smoke.main(["--video", "x.mp4", "--max-seconds", "0"]) == 2
    with pytest.raises(SystemExit):
        smoke.main(["--video", "x.mp4", "--image", "y.jpg"])  # mutually exclusive


def test_video_smoke_test_end_to_end_with_injected_provider(no_dotenv, tmp_path, capsys):
    video = write_video(tmp_path / "drive.avi", [(OPEN, 6.0), (CLOSED, 2.5)])
    code = smoke.video_smoke_test(video, CONFIG, sample_fps=10, max_seconds=None, assume_fps=None,
                                  quiet=False, provider=BrightnessFaceProvider())
    captured = capsys.readouterr()
    assert code == 0
    assert "[100%]" in captured.err  # progress lines, not per-frame JSON
    assert captured.out.count("\n") < 80
    assert "Drowsiness level" in captured.out and "CRITICAL" in captured.out


def test_existing_modes_still_work(no_dotenv, capsys):
    assert smoke.main([]) == 0
    assert "Structural smoke test passed" in capsys.readouterr().out
