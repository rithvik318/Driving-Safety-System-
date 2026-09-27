"""Road-object tracking smoke test: pretrained YOLO -> IoU tracker on real front-camera videos. Read-only.

Usage (from the project root):
    python scripts/test_road_tracking.py --source "D:\\...\\frontcamera\\vehicles"
    python scripts/test_road_tracking.py --source clip.mp4 --save-annotated --jsonl outputs/tracks.jsonl

For every video: frames are read sequentially, only frames on the --sample-fps grid go to YOLO
(loaded once), and each frame's detections go to tracker.update(detections, timestamp). A new
tracker is used per video (IDs restart at 1).

Timeline (compact, not every detection): one row per track when it is created, when its observed
class changes, every --timeline-interval seconds while it is alive, and when it ends:
    t | id | class | conf | cx | cy | vx | vy | area | persist | note

All motion values are IMAGE-SPACE: pixels, pixels/second, pixels². Not metres, not km/h, not
distance, not TTC. Folder names are printed as human context, never as ground truth.
Images are skipped (tracking needs video).

Sampling: default 10 fps for tracking (ROAD_SAMPLE_FPS=5 stays the detection-only default).
On our real videos, a pedestrian walking toward the camera nearly doubled in box area between
5 fps samples, IoU fell below the threshold and the track got a new ID; at 10 fps it kept one ID.
YOLO26n at ~65 ms/frame on CPU still keeps up with 10 fps.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python scripts/...`

DEFAULT_TRACKING_SAMPLE_FPS = 10.0  # see docstring: 5 fps broke IDs of fast-growing close objects on our videos
GROWTH_RATIO = 1.2  # "increasing bbox area": last/first area >= this over >= MIN_DETS_FOR_TREND detections
MIN_DETS_FOR_TREND = 3
DUPLICATE_IOU = 0.7  # two ACTIVE tracks overlapping this much in one frame: likely one object, two YOLO boxes


def _f(v, fmt="{:7.0f}"):
    return "      -" if v is None else fmt.format(v)


def timeline_row(t, note: str = "") -> str:
    return (f"  {t.timestamp:6.2f}s | {t.track_id:>3} | {t.observed_class:<13} | {t.class_confidence:.2f} | "
            f"{t.center_x:6.0f} | {t.center_y:6.0f} | {_f(t.velocity_x_pixels_per_second)} | "
            f"{_f(t.velocity_y_pixels_per_second)} | {t.area:9.0f} | {t.persistence_seconds:5.2f}s"
            + (f" | {note}" if note else ""))


TIMELINE_HEADER = (f"  {'t':>7} | {'id':>3} | {'class':<13} | conf | {'cx':>6} | {'cy':>6} | {'vx px/s':>7} | "
                   f"{'vy px/s':>7} | {'area px²':>9} | persist | note")


@dataclass
class VideoTrackStats:
    name: str
    folder: str
    width: int = 0
    height: int = 0
    duration: float = 0.0
    frames_read: int = 0
    frames_processed: int = 0
    detections: int = 0
    yolo_ms: list = field(default_factory=list)
    tracker_ms: list = field(default_factory=list)
    tracks: list = field(default_factory=list)  # final TrackedObject per track
    areas: dict = field(default_factory=lambda: defaultdict(list))  # track_id -> [(t, area)] when matched
    max_speed: dict = field(default_factory=dict)  # track_id -> max px/s seen
    rows: int = 0
    duplicate_pairs: dict = field(default_factory=dict)  # (id_a, id_b) -> frames where they overlapped


class TrackTimeline:
    """Decides which track rows to print (created / class change / every interval / ended)."""

    def __init__(self, interval: float, every_frame: bool = False, out=sys.stdout):
        self.interval, self.every_frame, self.out = interval, every_frame, out
        self.last_printed: dict[int, float] = {}
        self.last_class: dict[int, str] = {}
        self.rows = 0

    def emit(self, t, note=""):
        print(timeline_row(t, note), file=self.out)
        self.rows += 1
        self.last_printed[t.track_id] = t.timestamp

    def frame(self, tracks, created: set[int], ended) -> None:
        for t in tracks:
            if not t.active:
                continue
            note = ""
            if t.track_id in created:
                note = "NEW"
            elif self.last_class.get(t.track_id) not in (None, t.observed_class):
                note = f"class {self.last_class[t.track_id]} -> {t.observed_class}"
            due = self.every_frame or (self.interval > 0 and t.timestamp - self.last_printed.get(t.track_id, -1e9) >= self.interval - 1e-6)
            if note or due:
                self.emit(t, note)
            self.last_class[t.track_id] = t.observed_class
        for t in ended:
            self.emit(t, f"END ({t.detection_count} det, {len(set(t.class_history))} class(es))")


class Annotator:
    """Dev-only annotated video of the processed frames with track IDs."""

    def __init__(self, out_dir: Path, name: str, fps: float, max_side: int = 1280):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / f"{Path(name).stem}_tracks.mp4"
        self.fps, self.max_side, self.writer = max(fps, 1.0), max_side, None

    def __call__(self, frame, tracks, caption):
        import cv2

        from app.tracking.visualize import draw_tracks

        img = draw_tracks(frame, tracks, caption)
        h, w = img.shape[:2]
        s = self.max_side / max(h, w)
        if s < 1:
            img = cv2.resize(img, (int(w * s) // 2 * 2, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA)
        if self.writer is None:
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (img.shape[1], img.shape[0]))
        self.writer.write(img)

    def close(self):
        if self.writer is not None:
            self.writer.release()


def track_one_video(path: Path, detector, tracker, args, jsonl=None) -> VideoTrackStats | None:
    from app.road.runner import run_video
    from app.sensors.video import VideoReader
    from app.tracking import iou

    stats = VideoTrackStats(name=path.name, folder=path.parent.name)
    reader = VideoReader(path)
    if not reader.is_open():
        print(f"  ERROR: cannot open {path.name}")
        return None
    tracker.reset()
    timeline = TrackTimeline(args.timeline_interval, args.every_frame)
    annot = Annotator(args.save_annotated, path.name, args.sample_fps or 10.0) if args.save_annotated else None
    print(TIMELINE_HEADER)

    def on_result(result, frame):
        stats.detections += len(result.detections)
        if result.inference_ms is not None:
            stats.yolo_ms.append(result.inference_ms)
        started = time.perf_counter()
        detections = result.detections if result.status.value == "OK" else ()
        tracks = tracker.update(detections, result.timestamp, result.frame_index)
        stats.tracker_ms.append((time.perf_counter() - started) * 1000.0)
        live = [t for t in tracks if t.active]
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if iou(a.bbox, b.bbox) >= DUPLICATE_IOU:
                    key = (a.track_id, b.track_id, a.observed_class, b.observed_class)
                    stats.duplicate_pairs[key[:2]] = stats.duplicate_pairs.get(key[:2], 0) + 1
        for t in tracks:
            if t.active:
                stats.areas[t.track_id].append((t.timestamp, t.area))
                if t.speed_pixels_per_second is not None:
                    stats.max_speed[t.track_id] = max(stats.max_speed.get(t.track_id, 0.0), t.speed_pixels_per_second)
        timeline.frame(tracks, set(tracker.created_last_update), tracker.ended_last_update)
        if annot:
            annot(frame, tracks, f"{path.name}  t={result.timestamp:.2f}s  tracks={len(tracks)}")
        if jsonl:
            jsonl.write(json.dumps({"file": path.name, "folder": stats.folder, "timestamp": result.timestamp,
                                    "frame_index": result.frame_index, "status": result.status.value,
                                    "tracks": [t.to_dict() for t in tracks]}) + "\n")

    try:
        run = run_video(reader, detector, args.sample_fps, args.max_seconds, keep_results=False, on_result=on_result)
    finally:
        reader.release()
        if annot:
            annot.close()
    final = tracker.finish()
    timeline.frame([], set(), final)
    stats.tracks = tracker.all_tracks()
    stats.rows = timeline.rows
    stats.width, stats.height = run.info.width, run.info.height
    stats.frames_read, stats.frames_processed = run.frames_read, run.frames_processed
    stats.duration = (run.last_t - run.first_t) if run.last_t is not None else 0.0
    if annot:
        print(f"  annotated: {annot.path}")
    return stats


def growing(stats: VideoTrackStats, track) -> bool:
    series = stats.areas.get(track.track_id, [])
    return len(series) >= MIN_DETS_FOR_TREND and series[0][1] > 0 and series[-1][1] / series[0][1] >= GROWTH_RATIO


def video_summary(s: VideoTrackStats) -> str:
    by_class = Counter(t.class_name for t in s.tracks)
    changed = [t for t in s.tracks if t.class_changes]
    return (f"  => {len(s.tracks)} tracks {dict(by_class.most_common())} | {s.detections} detections in "
            f"{s.frames_processed} frames | label-change tracks {len(changed)} | "
            f"single-detection tracks {sum(1 for t in s.tracks if t.detection_count == 1)} | {s.rows} timeline rows")


def overall_summary(all_stats: list[VideoTrackStats], detector, config) -> str:
    tracks = [(s, t) for s in all_stats for t in s.tracks]
    lines = ["", "=" * 86, "TRACKING SUMMARY (image-space only: pixels, pixels/s; no distance, speed or TTC)", "=" * 86,
             f"Detector : {detector.name} on {detector.device}, conf >= {detector.confidence_threshold:.2f}, imgsz {detector.image_size}",
             f"Tracker  : IoU >= {config.iou_threshold}, label-switch IoU >= {config.class_switch_iou}, "
             f"max missed {config.max_missed_seconds:g} s" + (f" / {config.max_missed_frames} frames" if config.max_missed_frames else ""),
             f"Videos   : {len(all_stats)} | frames read {sum(s.frames_read for s in all_stats)}, "
             f"sent to YOLO {sum(s.frames_processed for s in all_stats)} | detections {sum(s.detections for s in all_stats)}"]
    if not tracks:
        return "\n".join(lines + ["No tracks."])
    persist = [t.persistence_seconds for _, t in tracks]
    multi = [t for _, t in tracks if t.detection_count >= 2]
    lines.append(f"Tracks created        : {len(tracks)} ({len(multi)} with >= 2 detections, "
                 f"{len(tracks) - len(multi)} single-detection)")
    lines.append("Tracks by class       : " + ", ".join(f"{c} {n}" for c, n in Counter(t.class_name for _, t in tracks).most_common())
                 + "   (track label = most frequent observed class)")
    s_long, t_long = max(tracks, key=lambda x: (x[1].persistence_seconds, x[1].detection_count))
    lines.append(f"Longest-lived track   : {s_long.name} ID {t_long.track_id} {t_long.class_name}, "
                 f"{t_long.persistence_seconds:.2f} s, {t_long.detection_count} detections")
    lines.append(f"Persistence           : mean {statistics.fmean(persist):.2f} s over all tracks; "
                 f"mean {statistics.fmean([t.persistence_seconds for t in multi]):.2f} s over tracks with >= 2 detections"
                 if multi else f"Persistence           : mean {statistics.fmean(persist):.2f} s")
    speeds = [(s, tid, v) for s in all_stats for tid, v in s.max_speed.items()]
    if speeds:
        s_fast, tid_fast, v_fast = max(speeds, key=lambda x: x[2])
        tr_fast = next(t for t in s_fast.tracks if t.track_id == tid_fast)
        lines.append(f"Max image-space speed : {v_fast:.0f} px/s = {v_fast / s_fast.width:.2f} frame-widths/s "
                     f"({s_fast.name} ID {tid_fast} {tr_fast.class_name}, frame {s_fast.width}x{s_fast.height})")
    grow = [(s, t) for s, t in tracks if growing(s, t)]
    lines.append(f"Growing bbox area     : {len(grow)} track(s) with last/first area >= {GROWTH_RATIO} over >= "
                 f"{MIN_DETS_FOR_TREND} detections" + ("" if not grow else ": " + ", ".join(
                     f"{s.name} ID {t.track_id} {t.class_name} x{s.areas[t.track_id][-1][1] / s.areas[t.track_id][0][1]:.1f}"
                     for s, t in grow[:8]) + (" ..." if len(grow) > 8 else "")))
    changes = Counter(min(t.class_changes, 3) for _, t in tracks)
    lines.append("Class changes / track : " + ", ".join(f"{'3+' if k == 3 else k}: {v}" for k, v in sorted(changes.items())))
    for s, t in [(s, t) for s, t in tracks if t.class_changes][:10]:
        lines.append(f"    {s.name} ID {t.track_id}: {t.class_changes} change(s), history {dict(Counter(t.class_history))}")
    dup = [(s, k, n) for s in all_stats for k, n in s.duplicate_pairs.items()]
    lines.append(f"Overlapping tracks    : {len(dup)} pair(s) of active tracks with IoU >= {DUPLICATE_IOU} in the same frame "
                 "(either one object given two YOLO boxes of different classes, or truly overlapping objects such as a rider on a bicycle)" + ("" if not dup else ": " + ", ".join(
                     f"{s.name} IDs {k[0]}+{k[1]} ({n} frame(s))" for s, k, n in dup[:6]) + (" ..." if len(dup) > 6 else "")))
    yolo = [m for s in all_stats for m in s.yolo_ms]
    trk = [m for s in all_stats for m in s.tracker_ms]
    lines.append(f"Time per frame        : YOLO mean {statistics.fmean(yolo):.1f} ms, tracker mean {statistics.fmean(trk):.3f} ms "
                 f"(max {max(trk):.3f} ms)")
    lines.append("Folder names are human context only; they are not ground truth.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    from app.config import configure_logging, load_road_config, load_settings, load_tracking_config
    from app.config.tracking_config import TrackingConfig, validate_tracking_config

    try:
        settings, rcfg, tcfg = load_settings(), load_road_config(), load_tracking_config()
    except ValueError as exc:
        print(f"ERROR: invalid configuration: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", type=Path, required=True, help="Video or folder (repeatable).")
    p.add_argument("--sample-fps", type=float, default=DEFAULT_TRACKING_SAMPLE_FPS,
                   help="Frames/s of video time sent to YOLO (default 10 for tracking; 0 = all).")
    p.add_argument("--max-seconds", type=float, help="Only the first N seconds of each video.")
    p.add_argument("--conf", type=float, default=rcfg.confidence_threshold)
    p.add_argument("--imgsz", type=int, default=rcfg.image_size)
    p.add_argument("--device", default=rcfg.device)
    p.add_argument("--iou", type=float, default=tcfg.iou_threshold, help="Same-class association IoU.")
    p.add_argument("--class-switch-iou", type=float, default=tcfg.class_switch_iou, help="Cross-class IoU (> 1 disables).")
    p.add_argument("--max-missed-seconds", type=float, default=tcfg.max_missed_seconds)
    p.add_argument("--max-missed-frames", type=int, default=tcfg.max_missed_frames)
    p.add_argument("--timeline-interval", type=float, default=1.0, help="Seconds between rows per live track (0 = only events).")
    p.add_argument("--every-frame", action="store_true", help="Print every active track on every processed frame.")
    p.add_argument("--save-annotated", nargs="?", const=settings.output_dir / "road_debug", type=Path,
                   help="Dev only: annotated videos with track IDs (default outputs/road_debug).")
    p.add_argument("--jsonl", type=Path, help="Write per-frame track snapshots as JSON lines.")
    args = p.parse_args(argv)

    tracking = TrackingConfig(args.iou, args.class_switch_iou, args.max_missed_seconds, args.max_missed_frames)
    try:
        validate_tracking_config(tracking)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    from app.road import RoadModelError, YoloRoadDetector
    from app.road.runner import VIDEO_SUFFIXES, list_media
    from app.sensors.video import VideoError
    from app.tracking import IoUTracker

    videos, skipped = [], 0
    for src in args.source:
        if not src.exists():
            print(f"ERROR: not found: {src}", file=sys.stderr)
            return 2
        for m in list_media(src):
            (videos.append(m) if m.suffix.lower() in VIDEO_SUFFIXES else None)
            skipped += m.suffix.lower() not in VIDEO_SUFFIXES
    if not videos:
        print("ERROR: no videos found in --source (tracking needs video).", file=sys.stderr)
        return 2
    try:
        detector = YoloRoadDetector(Path(rcfg.model_path), args.conf, args.imgsz, args.device, rcfg.classes)
    except (RoadModelError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    tracker = IoUTracker(tracking)
    print(f"Detector {detector.name} on {detector.device} (load {detector.load_seconds:.2f} s + warm-up "
          f"{detector.warmup_seconds:.2f} s, once) | {len(videos)} video(s)" + (f", {skipped} image(s) skipped" if skipped else ""))

    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.jsonl.open("w", encoding="utf-8") if args.jsonl else None
    all_stats = []
    try:
        for path in videos:
            print(f"\n--- {path.name}  folder: {path.parent.name} (human context, not ground truth)")
            try:
                s = track_one_video(path, detector, tracker, args, jsonl)
            except (VideoError, OSError, ValueError) as exc:
                print(f"  ERROR: {exc}")
                continue
            if s:
                print(video_summary(s))
                all_stats.append(s)
    finally:
        if jsonl:
            jsonl.close()
    print(overall_summary(all_stats, detector, tracking))
    if args.jsonl:
        print(f"Track snapshots (JSON lines): {args.jsonl}")
    print("Source files were only read; nothing was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
