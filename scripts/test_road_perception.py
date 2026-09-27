"""Front-camera road-perception smoke test: pretrained YOLO on real images/videos. Read-only.

Usage (from the project root):
    python scripts/test_road_perception.py --source "D:\\...\\frontcamera\\vehicles"
    python scripts/test_road_perception.py --source path/to/image.jpg --source path/to/video.mp4
    python scripts/test_road_perception.py --source "...\\frontcamera" --save-annotated

--source accepts files or folders (images and videos found recursively). For each file it prints
    timestamp | class | confidence | bbox
rows (capped by --max-rows per file) and a summary: images / video frames processed, detections,
class counts, confidence ranges and mean inference time.

Detection only: it reports what is visible. Folder names (vehicles, pedestrains, ...) are shown
as human-provided context, NOT as ground-truth labels, and nothing here scores danger or risk.

The model (default models/road/yolo26n.pt, ~5.5 MB) is downloaded once on first use if missing,
loaded ONCE, and reused for every frame. Videos are read sequentially; only frames on the
--sample-fps grid (video time) are sent to YOLO.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python scripts/...`


def _fmt_ts(t) -> str:
    return "   image" if t is None else f"{t:7.2f}s"


def format_row(d) -> str:
    return (f"  {_fmt_ts(d.timestamp)} | {d.class_name:<13} | {d.confidence:.2f} | "
            f"({d.bbox_x1:6.0f},{d.bbox_y1:6.0f})-({d.bbox_x2:6.0f},{d.bbox_y2:6.0f})  {d.bbox_width:.0f}x{d.bbox_height:.0f}")


class RowPrinter:
    """Prints detection rows for one file, up to max_rows (None = all)."""

    def __init__(self, max_rows: int | None, out=sys.stdout):
        self.max_rows, self.out = max_rows, out
        self.printed = self.skipped = 0

    def __call__(self, result, frame=None) -> None:
        if result.status.value != "OK":
            print(f"  {_fmt_ts(result.timestamp)} | {result.status.value}: {result.error}", file=self.out)
            return
        for d in result.detections:
            if self.max_rows is None or self.printed < self.max_rows:
                print(format_row(d), file=self.out)
                self.printed += 1
            else:
                self.skipped += 1

    def finish(self) -> None:
        if self.skipped:
            print(f"  ... {self.skipped} more detection row(s) not shown (use --max-rows 0 for all)", file=self.out)


class Annotator:
    """Dev-only: writes annotated images, or an annotated video of the processed frames."""

    def __init__(self, out_dir: Path, media: Path, kind: str, fps: float, max_side: int = 1280):
        self.out_dir, self.media, self.kind, self.fps, self.max_side = out_dir, media, kind, fps, max_side
        self.writer = None
        self.path: Path | None = None
        out_dir.mkdir(parents=True, exist_ok=True)

    def _shrink(self, img):
        import cv2

        h, w = img.shape[:2]
        s = self.max_side / max(h, w)
        return cv2.resize(img, (int(w * s) // 2 * 2, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA) if s < 1 else img

    def __call__(self, result, frame) -> None:
        import cv2

        from app.road.visualize import draw_detections

        caption = f"{self.media.name}" + ("" if result.timestamp is None else f"  t={result.timestamp:.2f}s")
        img = self._shrink(draw_detections(frame, result, caption))
        if self.kind == "image":
            self.path = self.out_dir / f"{self.media.stem}_annotated.jpg"
            cv2.imwrite(str(self.path), img)
            return
        if self.writer is None:
            self.path = self.out_dir / f"{self.media.stem}_annotated.mp4"
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), max(self.fps, 1.0),
                                          (img.shape[1], img.shape[0]))
        self.writer.write(img)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()


def summarize(all_stats, detector, wall_seconds: float) -> str:
    images = [s for s in all_stats if s.kind == "image"]
    videos = [s for s in all_stats if s.kind == "video"]
    classes: Counter = Counter()
    frames_with: Counter = Counter()
    confs = defaultdict(list)
    times = []
    statuses: Counter = Counter()
    for s in all_stats:
        classes.update(s.class_counts)
        frames_with.update(s.frames_with_class)
        for k, v in s.confidences.items():
            confs[k] += v
        times += s.inference_ms
        statuses.update(s.status_counts)
    processed = sum(s.frames_processed for s in all_stats)
    empty = sum(1 for s in all_stats for r in s.results if r.status.value == "OK" and not r.detections)
    lines = [
        "",
        "=" * 78,
        "ROAD PERCEPTION SUMMARY (detections only; no risk interpretation)",
        "=" * 78,
        f"Model                : {detector.name} ({detector.model_path.name}), device {detector.device}, "
        f"conf >= {detector.confidence_threshold:.2f}, imgsz {detector.image_size}",
        f"Model load / warm-up : {detector.load_seconds:.2f} s / {detector.warmup_seconds:.2f} s (once)",
        f"Files                : {len(all_stats)} ({len(images)} image(s), {len(videos)} video(s))",
        f"Images processed     : {sum(s.frames_processed for s in images)}",
        f"Video frames         : {sum(s.frames_read for s in videos)} read, {sum(s.frames_processed for s in videos)} "
        f"sent to YOLO" + (f" (sample {videos[0].sample_fps:g} fps)" if videos else ""),
        f"Frame status         : {dict(statuses)}",
        f"Frames with no detections: {empty} of {processed}",
        f"Detections           : {sum(classes.values())}",
    ]
    if classes:
        lines.append("Class counts         : (detections / frames containing it / confidence min-mean-max)")
        for name, n in classes.most_common():
            c = confs[name]
            lines.append(f"  {name:<14} {n:>5} / {frames_with[name]:>4} frames / "
                         f"{min(c):.2f}-{statistics.fmean(c):.2f}-{max(c):.2f}")
    if times:
        st = sorted(times)
        p95 = st[min(len(st) - 1, int(round(0.95 * (len(st) - 1))))]
        lines.append(f"Inference time       : mean {statistics.fmean(times):.1f} ms, median {statistics.median(times):.1f} ms, "
                     f"p95 {p95:.1f} ms, max {max(times):.1f} ms over {len(times)} frames "
                     f"(~{1000.0 / statistics.fmean(times):.1f} frames/s on this device)")
    lines.append(f"Wall time            : {wall_seconds:.1f} s (includes video decoding and file reading)")
    lines.append("Folder names are human context only; they are not used as ground-truth labels.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    from app.config import configure_logging, load_road_config, load_settings
    from app.config.road_config import parse_classes

    try:
        settings = load_settings()
        config = load_road_config()
    except ValueError as exc:
        print(f"ERROR: invalid configuration: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", type=Path, required=True,
                        help="Image, video or folder (repeatable). Folders are searched recursively.")
    parser.add_argument("--model", type=Path, default=config.model_path, help="YOLO weights (default from ROAD_MODEL_PATH).")
    parser.add_argument("--conf", type=float, default=config.confidence_threshold, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=config.image_size, help="Inference image size (multiple of 32).")
    parser.add_argument("--sample-fps", type=float, default=config.sample_fps, help="Video frames per second sent to YOLO (0 = all).")
    parser.add_argument("--max-seconds", type=float, help="Only the first N seconds of each video.")
    parser.add_argument("--assume-fps", type=float, help="Only for videos that report neither FPS nor timestamps.")
    parser.add_argument("--device", default=config.device, help="auto | cpu | cuda (default from ROAD_DEVICE/DEVICE).")
    parser.add_argument("--classes", help='Comma-separated COCO names, or "all" (default: road-relevant set).')
    parser.add_argument("--max-rows", type=int, default=25, help="Detection rows printed per file (0 = all).")
    parser.add_argument("--save-annotated", nargs="?", const=settings.output_dir / "road_debug", type=Path,
                        help="Dev only: save annotated images/videos (default folder outputs/road_debug).")
    parser.add_argument("--jsonl", type=Path, help="Write every RoadPerceptionResult as one JSON line.")
    args = parser.parse_args(argv)

    if not 0 <= args.conf <= 1 or args.imgsz < 32 or args.imgsz % 32 or args.sample_fps < 0 or args.max_rows < 0:
        print("ERROR: need 0 <= --conf <= 1, --imgsz a multiple of 32, --sample-fps >= 0, --max-rows >= 0.", file=sys.stderr)
        return 2

    from app.road import RoadModelError, YoloRoadDetector
    from app.road.runner import IMAGE_SUFFIXES, list_media, run_image, run_video
    from app.sensors.video import VideoError, VideoReader

    media = []
    for src in args.source:
        if not src.exists():
            print(f"ERROR: not found: {src}", file=sys.stderr)
            return 2
        media += list_media(src)
    if not media:
        print("ERROR: no images or videos found in --source.", file=sys.stderr)
        return 2

    try:
        classes = parse_classes(args.classes) if args.classes else config.classes
        detector = YoloRoadDetector(args.model, args.conf, args.imgsz, args.device, classes)
    except (RoadModelError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Road detector: {detector.name} on {detector.device} | load {detector.load_seconds:.2f} s + warm-up "
          f"{detector.warmup_seconds:.2f} s (once) | classes: {', '.join(detector.class_names)}")
    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.jsonl.open("w", encoding="utf-8") if args.jsonl else None
    all_stats = []
    started = time.perf_counter()
    try:
        for path in media:
            folder = path.parent.name or None
            kind = "image" if path.suffix.lower() in IMAGE_SUFFIXES else "video"
            print(f"\n--- {path.name}  [{kind}]" + (f"  folder: {folder} (human context, not ground truth)" if folder else ""))
            print(f"  {'time':>8} | {'class':<13} | conf | bbox (x1,y1)-(x2,y2) px  w x h")
            rows = RowPrinter(args.max_rows or None)
            annot = Annotator(args.save_annotated, path, kind, args.sample_fps or 10.0) if args.save_annotated else None

            def on_result(result, frame, rows=rows, annot=annot):
                rows(result)
                if annot:
                    annot(result, frame)
                if jsonl:
                    jsonl.write(json.dumps({"file": path.name, "folder": folder, **result.to_dict()}) + "\n")

            try:
                if kind == "image":
                    stats = run_image(path, detector, on_result=on_result)
                else:
                    reader = VideoReader(path)
                    if not reader.is_open():
                        print(f"  ERROR: cannot open video {path.name}")
                        continue
                    try:
                        stats = run_video(reader, detector, args.sample_fps, args.max_seconds, args.assume_fps, on_result=on_result)
                    finally:
                        reader.release()
            except (VideoError, OSError, ValueError) as exc:
                print(f"  ERROR: {exc}")
                continue
            finally:
                if annot:
                    annot.close()
            rows.finish()
            info = stats.info
            extra = ""
            if info:
                dur = f"{stats.last_t - stats.first_t:.2f}" if stats.last_t is not None else "?"
                extra = (f" | video {info.width}x{info.height} @ {f'{info.fps:.1f}' if info.fps else '?'} fps, {dur} s, "
                         f"{stats.frames_read} read / {stats.frames_processed} detected")
            mean_ms = f"{sum(stats.inference_ms) / len(stats.inference_ms):.0f} ms" if stats.inference_ms else "n/a"
            print(f"  => {stats.detections} detections {dict(stats.class_counts.most_common())} | mean {mean_ms}{extra}")
            if annot and annot.path:
                print(f"  annotated: {annot.path}")
            all_stats.append(stats)
    finally:
        if jsonl:
            jsonl.close()

    print(summarize(all_stats, detector, time.perf_counter() - started))
    if args.jsonl:
        print(f"Results (JSON lines): {args.jsonl}")
    print("Source files were only read; nothing was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
