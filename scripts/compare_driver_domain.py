"""Visual comparison of hand-state training images vs. frames from a real driver video. Read-only.

Usage (from the project root):
    python scripts/compare_driver_domain.py --root "D:\\path\\to\\Data"
    python scripts/compare_driver_domain.py --root "D:\\path\\to\\Data" --video "D:\\path\\to\\video.mp4"

By default the video is video_20260926_220714.mp4, searched for under the dataset root
(it lives in drivercamera/drowsy_driver/).

Writes:
    outputs/plots/driver_camera_domain_comparison.jpg
    outputs/reports/driver_camera_domain_comparison.md   (section 3, "Visual observations",
                                                          is kept when the report is regenerated)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python scripts/...`
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    from app.config import configure_logging, load_settings

    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"ERROR: invalid configuration: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level)

    from app.dataset_audit.domain_compare import DEFAULT_VIDEO_NAME

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help="Dataset root folder (overrides DATASET_ROOT).")
    parser.add_argument("--video", type=Path, help=f"Driver video (default: {DEFAULT_VIDEO_NAME} under --root).")
    parser.add_argument("--per-class", type=int, default=3, help="Training images per class (default 3).")
    parser.add_argument("--frames", type=int, default=3, help="Video frames, evenly spaced (default 3).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the representative-sample clustering.")
    parser.add_argument("--output", type=Path, default=settings.output_dir / "plots" / "driver_camera_domain_comparison.jpg")
    parser.add_argument("--report", type=Path, default=settings.output_dir / "reports" / "driver_camera_domain_comparison.md")
    parser.add_argument("--no-report", action="store_true", help="Only write the comparison image.")
    args = parser.parse_args(argv)

    if args.per_class < 1 or args.frames < 1:
        print("ERROR: --per-class and --frames must be >= 1.", file=sys.stderr)
        return 2

    from app.dataset_audit import DatasetRootError, validate_root
    from app.dataset_audit.domain_compare import DomainCompareError, find_video, run_comparison
    from app.dataset_audit.hand_preview import PreviewError

    # evenly spaced interior fractions, e.g. 3 -> 0.2, 0.5, 0.8
    fractions = (0.5,) if args.frames == 1 else tuple(0.2 + 0.6 * i / (args.frames - 1) for i in range(args.frames))
    try:
        root = validate_root(args.root or settings.dataset_root)
        video = args.video if args.video else find_video(root)
        if not video.is_file():
            raise DomainCompareError(f"Video not found: {video}")
        result = run_comparison(root, video, args.output, None if args.no_report else args.report,
                                args.per_class, fractions, args.seed, PROJECT_ROOT)
    except (DatasetRootError, PreviewError, DomainCompareError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Driver-camera domain comparison\n")
    print(f"  {'Source type':<15} {'Label':<24} {'Time':>8} {'Original':>11} {'Shown':>11} {'Luma':>6}  File")
    for s in result["samples"]:
        t = f"{s['timestamp_s']:.2f}s" if s["timestamp_s"] is not None else "-"
        print(f"  {s['source_type']:<15} {(s['label'] or 'none'):<24} {t:>8} "
              f"{s['stored_width']}x{s['stored_height']:<6} {s['display_width']}x{s['display_height']:<6} "
              f"{s['mean_luma']:>6}  {Path(s['source']).name}")
    v = result["video"]
    print(f"\nVideo: {v['stored_width']}x{v['stored_height']}, {v['fps']} fps, {v['frame_count']} frames, "
          f"{v['duration_s']} s, rotation metadata {v['rotation_meta']}")
    print(f"\nComparison image: {result['image']}")
    if "report" in result:
        print(f"Report:           {result['report']}")
    print("Sources were only read; nothing in the dataset or the video was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
