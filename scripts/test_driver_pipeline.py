"""Driver-pipeline smoke test.

    python scripts/test_driver_pipeline.py --image path/to/driver.jpg
        MediaPipe Face Landmarker + the full driver pipeline on one image.

    python scripts/test_driver_pipeline.py --video "path/to/driver_video.mp4"
        Runs the SAME DriverPerceptionPipeline over a real video, frame by frame, keeping
        its temporal state, using the video's own timestamps. Prints progress and a
        temporal summary. Options: --sample-fps (default 10; 0 = every frame),
        --max-seconds, --assume-fps (only if the file reports no FPS/timestamps), --quiet.

    python scripts/test_driver_pipeline.py
        Structural check only: runs the pipeline with NO face model on blank frames to
        show the output shape. It performs no perception; the values are not results.

--image and --video need models/face/face_landmarker.task (or FACE_MODEL_PATH / --model).
They also use the trained hand-state classifier (outputs/models/hand_state_best.pt, or
HAND_MODEL_PATH / --hand-model), loaded ONCE and run only on the sampled frames. If the default
checkpoint is missing, the hand state stays UNKNOWN and the run says so. --no-hand-model skips it.

Video timeline (stdout): one row per --timeline-interval seconds (default 1) plus a row whenever
the hand state or the inferred activity changes. --every-frame prints every processed frame.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python scripts/...`

from app.sensors.video import (  # noqa: E402  (shared with road perception; names kept for callers/tests)
    MAX_PLAUSIBLE_FPS,
    FrameClock,
    VideoError,
    VideoInfo,
    VideoReader,
)

DISTRACTION_ACTIVITY_NAMES = {"HANDS_OFF_WHEEL", "PHONE", "OTHER_MANUAL_DISTRACTION"}
HAND_STATE_NAMES = ("BOTH_HANDS", "ONE_HAND", "NO_HANDS", "UNKNOWN")


# --------------------------------------------------------------------------- image / structural


def load_image(path: Path):
    import cv2
    import numpy as np

    try:
        data = np.fromfile(str(path), dtype=np.uint8)  # works with non-ASCII Windows paths
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def structural_smoke_test(config) -> int:
    import numpy as np

    from app.driver import DriverPerceptionPipeline, NullFaceProvider

    print("STRUCTURAL SMOKE TEST: no face model is used and no perception is performed.")
    print("The values below only show the DriverState shape; they are not measurements.\n")
    pipeline = DriverPerceptionPipeline(NullFaceProvider(), config)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    state = None
    for i in range(3):
        state = pipeline.process(blank, i * 0.1)
    print(json.dumps(state.to_dict(), indent=2))
    pipeline.reset()
    print("\nreset() OK. Structural smoke test passed.")
    return 0


def build_face_provider(config):
    """Return (provider, None) or (None, error message)."""
    from app.driver import FaceModelNotFoundError, MediaPipeFaceProvider

    try:
        return MediaPipeFaceProvider(config.face_model_path, config.min_face_confidence), None
    except FaceModelNotFoundError as exc:
        return None, str(exc)
    except ImportError as exc:
        return None, f"mediapipe is not installed ({exc}). Run: pip install -r requirements.txt"


def build_hand_provider(config, explicit: bool = False, disabled: bool = False):
    """Return (provider, note, error). The model is loaded once here and reused for every frame.

    Missing default checkpoint -> UNKNOWN placeholder with a note; missing EXPLICIT checkpoint -> error.
    """
    from app.driver import UnknownHandStateProvider

    if disabled:
        return UnknownHandStateProvider(), "hand-state model disabled (--no-hand-model): hand_state stays UNKNOWN", None
    path = Path(config.hand_model_path)
    if not path.is_file():
        msg = f"hand-state checkpoint not found: {path}"
        if explicit:
            return None, None, msg
        return UnknownHandStateProvider(), msg + " -> hand_state stays UNKNOWN (train with scripts/train_hand_model.py)", None
    try:
        from app.driver import TrainedHandStateProvider

        provider = TrainedHandStateProvider.from_checkpoint(path, threshold=config.hand_confidence_threshold, device="cpu")
    except ImportError as exc:
        msg = f"torch/torchvision not installed ({exc}); run: pip install -r requirements.txt"
        return (None, None, msg) if explicit else (UnknownHandStateProvider(), msg + " -> hand_state stays UNKNOWN", None)
    except Exception as exc:  # incompatible/corrupt checkpoint
        return None, None, f"could not load hand-state checkpoint {path.name}: {exc}"
    return provider, f"hand-state model {path.name} (epoch {provider.classifier.meta.get('epoch')}, CPU, threshold {provider.threshold:.2f})", None


def image_smoke_test(image_path: Path, config, hand_provider=None) -> int:
    from app.driver import DriverPerceptionPipeline

    frame = load_image(image_path)
    if frame is None:
        print(f"ERROR: could not read image: {image_path}", file=sys.stderr)
        return 2
    provider, error = build_face_provider(config)
    if error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    pipeline = DriverPerceptionPipeline(provider, config, hand_provider=hand_provider)
    try:
        state = pipeline.process(frame, 0.0)
    finally:
        pipeline.close()
    print(f"Image: {image_path.name} ({frame.shape[1]}x{frame.shape[0]})")
    print("Single frame: temporal fields (closure ratio, drowsiness) need a video and stay empty/UNKNOWN here.\n")
    print(json.dumps(state.to_dict(), indent=2))
    return 0


# --------------------------------------------------------------------------- video


@dataclass
class VideoRunResult:
    info: VideoInfo
    sample_fps: float
    frames_read: int = 0
    frames_processed: int = 0
    unreadable_frames: int = 0
    stopped_early: bool = False
    first_t: float | None = None
    last_t: float | None = None
    face_frames: int = 0
    face_status: Counter = field(default_factory=Counter)
    max_eye_closed_duration: float = 0.0
    max_drowsiness_score: float | None = None
    max_distraction_duration: float | None = None
    timeline_resets: int = 0
    total_blinks: int = 0
    total_long_closures: int = 0
    timestamp_sources: dict = field(default_factory=dict)
    final_state: object = None  # DriverState
    hand_provider: str = "unknown"
    hand_counts: Counter = field(default_factory=Counter)  # OBSERVED hand_state per processed frame
    activity_counts: Counter = field(default_factory=Counter)  # INFERRED driver_activity per processed frame
    distraction_episodes: int = 0  # times the inferred activity entered a distraction state
    hand_confidences: list = field(default_factory=list)

    @property
    def face_pct(self) -> float:
        return 100.0 * self.face_frames / self.frames_processed if self.frames_processed else 0.0

    @property
    def timeline_duration(self) -> float | None:
        if self.first_t is None or self.last_t is None:
            return None
        step = 1.0 / self.info.fps if self.info.fps else 0.0
        return self.last_t - self.first_t + step


def _max(a, b):
    if b is None:
        return a
    return b if a is None else max(a, b)


def run_video(reader, pipeline, sample_fps: float = 10.0, max_seconds: float | None = None,
              assume_fps: float | None = None, progress=None, on_state=None) -> VideoRunResult:
    """Feed a video through the existing DriverPerceptionPipeline, keeping its temporal state.

    Every frame is grabbed so timestamps stay exact; only sampled frames are decoded and
    processed. All temporal logic (closure, drowsiness, durations) is the pipeline's own.
    """
    info = reader.info()
    if assume_fps:
        info.fps = assume_fps
    max_gap = pipeline.config.max_gap_seconds
    if sample_fps and 1.0 / sample_fps > max_gap:
        raise VideoError(
            f"--sample-fps {sample_fps:g} is too sparse: frames would be {1 / sample_fps:.2f} s apart, more than "
            f"MAX_GAP_SECONDS={max_gap:g}, so no eye-closure time could be observed. Use a higher --sample-fps."
        )

    result = VideoRunResult(info=info, sample_fps=sample_fps)
    clock = FrameClock(info.fps)
    interval = 1.0 / sample_fps if sample_fps else 0.0
    next_sample: float | None = None
    report_every = max(1, (info.frame_count or 0) // 10) if info.frame_count else None
    next_report_t = 10.0
    previous_activity = None
    pipeline.reset()

    index = -1
    while reader.grab():
        index += 1
        t = clock.timestamp(index, reader.position_seconds())
        result.frames_read += 1
        if result.first_t is None:
            result.first_t = t
        if max_seconds is not None and t - result.first_t > max_seconds:
            break
        result.last_t = t

        if next_sample is not None and t < next_sample - 1e-6:
            continue
        # Advance the sampling grid; if the video jumped ahead, re-anchor on this frame.
        next_sample = t + interval if next_sample is None else next_sample + interval
        if next_sample <= t:
            next_sample = t + interval

        ok, frame = reader.retrieve()
        if not ok or frame is None:
            result.unreadable_frames += 1
            continue

        state = pipeline.process(frame, t)
        result.frames_processed += 1
        result.final_state = state
        o, tmp = state.observation, state.temporal
        result.face_status[o.face_status.value] += 1
        result.face_frames += o.face_detected
        result.timeline_resets += tmp.timeline_reset
        if tmp.eye_closed_duration:
            result.max_eye_closed_duration = max(result.max_eye_closed_duration, tmp.eye_closed_duration)
        result.max_drowsiness_score = _max(result.max_drowsiness_score, tmp.drowsiness_score)
        result.max_distraction_duration = _max(result.max_distraction_duration, tmp.distraction_duration)
        result.hand_provider = state.hand.provider
        result.hand_counts[state.hand.hand_state.value] += 1
        if state.hand.confidence is not None:
            result.hand_confidences.append(state.hand.confidence)
        result.activity_counts[state.driver_activity.value] += 1
        if state.driver_activity.value in DISTRACTION_ACTIVITY_NAMES and previous_activity not in DISTRACTION_ACTIVITY_NAMES:
            result.distraction_episodes += 1
        previous_activity = state.driver_activity.value
        if on_state:
            on_state(state)

        if progress:
            due = (report_every and result.frames_read % report_every == 0) or (not report_every and t >= next_report_t)
            if due:
                next_report_t = t + 10.0
                progress(result, t)

    if index < 0:
        raise VideoError("No frames could be read from the video (empty, corrupt or unsupported codec).")
    if info.frame_count and max_seconds is None and result.frames_read < 0.98 * info.frame_count:
        result.stopped_early = True

    tracker = pipeline.eye_tracker
    result.total_blinks = tracker.total_blinks
    result.total_long_closures = tracker.total_long_closures
    result.max_eye_closed_duration = max(result.max_eye_closed_duration, tracker.longest_closure_seconds)
    result.timestamp_sources = dict(clock.sources)
    return result


def _fmt(value, unit: str = "", digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{round(value, digits) + 0.0:g}{unit}"  # + 0.0 turns -0 into 0
    return f"{value}{unit}"


def format_summary(r: VideoRunResult) -> str:
    info = r.info
    s = r.final_state.to_dict() if r.final_state is not None else {}
    fps = f"{info.fps:g}" if info.fps else "not reported"
    rows: list[tuple[str, str]] = [
        ("Video", info.name),
        ("Resolution", f"{info.width}x{info.height}"),
        ("FPS", fps),
        ("Total frames (container)", _fmt(info.frame_count)),
        ("Frames read", str(r.frames_read)),
        ("Processed frames", f"{r.frames_processed} (sampling {'every frame' if not r.sample_fps else f'{r.sample_fps:g} fps'})"),
        ("Unreadable frames", str(r.unreadable_frames)),
        ("Duration (timestamps)", _fmt(r.timeline_duration, " s", 2)),
        ("Duration (container)", _fmt(info.container_duration, " s", 2)),
        ("Timestamp source", ", ".join(f"{k}: {v}" for k, v in r.timestamp_sources.items()) or "n/a"),
        ("Face detected", f"{r.face_pct:.1f}% of processed frames {dict(r.face_status)}"),
    ]
    final = [
        ("Hand state (observed)", f"{s.get('hand_state')} (confidence {_fmt(s.get('hand_state_confidence'), '', 2)}, "
                                  f"provider {s.get('hand_state_provider')})"),
        ("Driver activity (inferred)", f"{s.get('driver_activity')} [{s.get('activity_source')}]"),
        ("Activity reason", _fmt(s.get("activity_reason"))),
        ("Head yaw / pitch / roll", " / ".join(_fmt(s.get(k), "°", 1) for k in ("head_yaw", "head_pitch", "head_roll"))),
        ("Eye openness L / R", f"{_fmt(s.get('left_eye_openness'))} / {_fmt(s.get('right_eye_openness'))}"),
        ("Eyes closed", _fmt(s.get("eyes_closed"))),
        ("Eye closure ratio (window)", _fmt(s.get("eye_closure_ratio"))),
        ("Blink count (window)", _fmt(s.get("blink_count"))),
        ("Long closure count (window)", _fmt(s.get("long_closure_count"))),
        ("Observed seconds (window)", _fmt(s.get("observed_seconds"), " s", 2)),
        ("Drowsiness score", _fmt(s.get("drowsiness_score"))),
        ("Drowsiness level", _fmt(s.get("drowsiness_level"))),
        ("Manual state", _fmt(s.get("manual_state"))),
        ("Manual state duration", _fmt(s.get("manual_state_duration"), " s", 2)),
        ("Manual state transitions", _fmt(s.get("manual_state_transitions"))),
        ("Distraction duration", _fmt(s.get("distraction_duration"), " s", 2)),
        ("Timeline reset", _fmt(s.get("timeline_reset"))),
        ("Observation quality", _fmt(s.get("observation_quality"), "", 2)),
    ]
    stats = [
        ("Frames with face", f"{r.face_pct:.1f}%"),
        ("Longest continuous eye closure", _fmt(r.max_eye_closed_duration, " s", 2)),
        ("Total blinks", str(r.total_blinks)),
        ("Total long closures", str(r.total_long_closures)),
        ("Max drowsiness score", _fmt(r.max_drowsiness_score)),
        ("Max distraction duration", _fmt(r.max_distraction_duration, " s", 2) + ("" if r.max_distraction_duration is not None else " (hand state UNKNOWN)")),
        ("Timeline resets", str(r.timeline_resets)),
    ]
    n = r.frames_processed or 1
    hand = [
        ("Provider", r.hand_provider),
        *[(f"{name} frames", f"{r.hand_counts.get(name, 0)} ({100 * r.hand_counts.get(name, 0) / n:.1f}%)") for name in HAND_STATE_NAMES],
        ("Mean / min confidence", (f"{sum(r.hand_confidences) / len(r.hand_confidences):.2f} / {min(r.hand_confidences):.2f}"
                                   if r.hand_confidences else "n/a")),
        ("Longest continuous ONE_HAND", _fmt(s.get("longest_one_hand_duration"), " s", 2)),
        ("Longest continuous NO_HANDS", _fmt(s.get("longest_no_hands_duration"), " s", 2)),
        ("Hand-state transitions", _fmt(s.get("manual_state_transitions"))),
        ("Inferred activity frames", ", ".join(f"{k} {v}" for k, v in r.activity_counts.most_common()) or "n/a"),
        ("Inferred distraction episodes", str(r.distraction_episodes)),
    ]

    def block(title, items):
        width = max(len(k) for k, _ in items)
        return [title] + [f"  {k:<{width}}  {v}" for k, v in items]

    lines = ["", *block("VIDEO", rows), "", *block("FINAL DRIVER STATE (last processed frame)", final), "",
             *block("TEMPORAL STATISTICS (whole video)", stats), "",
             *block("HAND STATE / MANUAL CONTROL (whole video)", hand)]
    notes = []
    if r.frames_processed and r.face_frames == 0:
        notes.append("No face was detected in any processed frame: eye and drowsiness metrics are unavailable. "
                     "Check camera angle, lighting and that the driver's face is in view.")
    if r.stopped_early:
        notes.append(f"Reading stopped after {r.frames_read} of ~{info.frame_count} frames (unreadable or corrupt data).")
    if r.unreadable_frames:
        notes.append(f"{r.unreadable_frames} sampled frame(s) could not be decoded and were skipped.")
    if info.fps and r.sample_fps and info.fps < r.sample_fps:
        notes.append(f"Video FPS ({info.fps:g}) is below --sample-fps; every frame was processed.")
    notes.append("Drowsiness values are a prototype behavioural signal (not medically validated).")
    if r.hand_provider == "unknown_placeholder":
        notes.append("No hand-state model was used: hand_state is UNKNOWN throughout.")
    else:
        notes.append("hand_state is OBSERVED per frame by a classifier trained on 188 images (27-image test set): "
                     "indicative only. driver_activity is INFERRED over time; ONE_HAND is never treated as distraction "
                     "and phone use is not inferred from hand state.")
    lines += ["", "NOTES"] + [f"  - {n}" for n in notes]
    return "\n".join(lines)


class TimelinePrinter:
    """Compact per-video timeline: a row every `interval` seconds and on any hand/activity change."""

    HEADER = (f"{'t (s)':>7} | {'hand_state':<10} | {'conf':>4} | {'manual_dur':>10} | {'activity':<15} | "
              f"{'drowsiness':<15} | {'head_yaw':>8} | {'head_pitch':>10}")

    def __init__(self, interval: float = 1.0, every_frame: bool = False, out=None):
        self.interval, self.every_frame = interval, every_frame
        self.out = out or sys.stdout
        self.last_t: float | None = None
        self.last_key = None
        self.rows = 0

    @staticmethod
    def _num(value, fmt: str) -> str:
        return format(value, fmt) if value is not None else "n/a"

    def __call__(self, state) -> None:
        key = (state.hand.hand_state.value, state.driver_activity.value)
        due = self.every_frame or self.last_t is None or key != self.last_key or (
            self.interval > 0 and state.timestamp - self.last_t >= self.interval - 1e-9)
        if not due:
            return
        if self.rows == 0:
            print("\nTIMELINE (sampled frames; rows every "
                  f"{'frame' if self.every_frame else f'{self.interval:g} s'} and on state changes)", file=self.out)
            print(self.HEADER, file=self.out)
        t, o = state.temporal, state.observation
        drowsy = f"{t.drowsiness_level.value} {self._num(t.drowsiness_score, '.2f')}"
        print(f"{state.timestamp:7.2f} | {state.hand.hand_state.value:<10} | {self._num(state.hand.confidence, '.2f'):>4} | "
              f"{self._num(t.manual_state_duration, '.1f') + ' s':>10} | {state.driver_activity.value:<15} | {drowsy:<15} | "
              f"{self._num(o.head_yaw, '+.1f'):>8} | {self._num(o.head_pitch, '+.1f'):>10}", file=self.out)
        self.last_t, self.last_key = state.timestamp, key
        self.rows += 1


def video_smoke_test(video_path: Path, config, sample_fps: float, max_seconds: float | None,
                     assume_fps: float | None, quiet: bool, provider=None, hand_provider=None,
                     timeline_interval: float = 1.0, every_frame: bool = False) -> int:
    from app.driver import DriverPerceptionPipeline

    if not video_path.is_file():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        return 2
    reader = VideoReader(video_path)
    try:
        if not reader.is_open():
            print(f"ERROR: could not open video (unsupported codec or corrupt file): {video_path.name}", file=sys.stderr)
            return 2
        if provider is None:
            provider, error = build_face_provider(config)
            if error:
                print(f"ERROR: {error}", file=sys.stderr)
                return 2
        if hand_provider is None:
            hand_provider, note, error = build_hand_provider(config)
            if error:
                print(f"ERROR: {error}", file=sys.stderr)
                return 2
            if note and not quiet:
                print(f"Hand state: {note}", file=sys.stderr)
        pipeline = DriverPerceptionPipeline(provider, config, hand_provider=hand_provider)
        timeline = TimelinePrinter(timeline_interval, every_frame) if (timeline_interval > 0 or every_frame) else None

        def progress(r: VideoRunResult, t: float) -> None:
            if quiet:
                return
            pct = f"{100 * r.frames_read / r.info.frame_count:3.0f}%" if r.info.frame_count else "  ? "
            level = r.final_state.temporal.drowsiness_level.value if r.final_state else "-"
            print(f"[{pct}] t={t:7.1f}s  processed={r.frames_processed:5d}  face={r.face_pct:5.1f}%  drowsiness={level}",
                  file=sys.stderr, flush=True)

        info = reader.info()
        if not quiet:
            print(f"Processing {info.name} ({info.width}x{info.height}, fps={info.fps or assume_fps or 'unknown'}) ...",
                  file=sys.stderr)
        try:
            result = run_video(reader, pipeline, sample_fps, max_seconds, assume_fps, progress, on_state=timeline)
            progress(result, result.last_t or 0.0)  # final line
        except VideoError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        finally:
            pipeline.close()
    finally:
        reader.release()

    print(format_summary(result))
    return 0


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles may not encode every character
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="Path to one driver-camera image.")
    source.add_argument("--video", type=Path, help="Path to a driver-camera video.")
    parser.add_argument("--model", type=Path, help="Face Landmarker .task file (overrides FACE_MODEL_PATH).")
    parser.add_argument("--sample-fps", type=float, default=10.0,
                        help="Frames per second of video time to process (default 10; 0 = every frame).")
    parser.add_argument("--max-seconds", type=float, help="Only process the first N seconds of the video.")
    parser.add_argument("--assume-fps", type=float,
                        help="Frame rate to use if the video reports no FPS and no timestamps.")
    parser.add_argument("--quiet", action="store_true", help="Hide progress lines.")
    parser.add_argument("--hand-model", type=Path, help="Hand-state checkpoint (overrides HAND_MODEL_PATH).")
    parser.add_argument("--hand-threshold", type=float, help="Confidence below this -> UNKNOWN (default 0.60).")
    parser.add_argument("--no-hand-model", action="store_true", help="Do not load the hand-state classifier.")
    parser.add_argument("--timeline-interval", type=float, default=1.0,
                        help="Seconds of video between timeline rows (default 1; 0 = no timeline unless --every-frame).")
    parser.add_argument("--every-frame", action="store_true", help="Print a timeline row for every processed frame.")
    args = parser.parse_args(argv)

    if args.sample_fps < 0 or (args.assume_fps is not None and not 0 < args.assume_fps <= MAX_PLAUSIBLE_FPS):
        print("ERROR: --sample-fps must be >= 0 and --assume-fps must be between 0 and 240.", file=sys.stderr)
        return 2
    if args.max_seconds is not None and args.max_seconds <= 0:
        print("ERROR: --max-seconds must be > 0.", file=sys.stderr)
        return 2

    from dataclasses import replace

    from app.config import configure_logging, load_driver_config, load_settings

    try:
        configure_logging(load_settings().log_level)
        config = load_driver_config()
    except ValueError as exc:
        print(f"ERROR: invalid configuration: {exc}", file=sys.stderr)
        return 2
    if args.model:
        config = replace(config, face_model_path=args.model)
    if args.hand_model:
        config = replace(config, hand_model_path=args.hand_model)
    if args.hand_threshold is not None:
        if not 0 <= args.hand_threshold <= 1:
            print("ERROR: --hand-threshold must be between 0 and 1", file=sys.stderr)
            return 2
        config = replace(config, hand_confidence_threshold=args.hand_threshold)
    if args.timeline_interval < 0:
        print("ERROR: --timeline-interval must be >= 0", file=sys.stderr)
        return 2

    if args.video is not None or args.image is not None:
        if args.video is not None and not args.video.is_file():
            print(f"ERROR: video not found: {args.video}", file=sys.stderr)
            return 2
        hand_provider, note, error = build_hand_provider(config, explicit=args.hand_model is not None,
                                                         disabled=args.no_hand_model)
        if error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        if note and not args.quiet:
            print(f"Hand state: {note}", file=sys.stderr)
        if args.video is not None:
            return video_smoke_test(args.video, config, args.sample_fps, args.max_seconds, args.assume_fps, args.quiet,
                                    hand_provider=hand_provider, timeline_interval=args.timeline_interval,
                                    every_frame=args.every_frame)
        return image_smoke_test(args.image, config, hand_provider=hand_provider)
    return structural_smoke_test(config)


if __name__ == "__main__":
    sys.exit(main())
