"""Contact-sheet preview of the hand-state classes for visual label checking. Read-only.

Usage (from the project root):
    python scripts/preview_hand_dataset.py --root "D:\\path\\to\\Data" --samples-per-class 12

Writes:
    outputs/plots/hand_dataset_preview.jpg
    outputs/reports/hand_preview_manifest.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python scripts/...`


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles may not encode every filename
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    from app.config import configure_logging, load_settings

    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"ERROR: invalid configuration: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help="Dataset root folder (overrides DATASET_ROOT).")
    parser.add_argument("--samples-per-class", type=int, default=12, help="Images shown per class (default 12).")
    parser.add_argument("--columns", type=int, default=6, help="Tiles per row (default 6).")
    parser.add_argument("--tile-width", type=int, default=420, help="Tile width in pixels (default 420).")
    parser.add_argument("--tile-height", type=int, default=640, help="Tile height in pixels (default 640).")
    parser.add_argument("--output", type=Path, default=settings.output_dir / "plots" / "hand_dataset_preview.jpg")
    parser.add_argument("--manifest", type=Path, default=settings.output_dir / "reports" / "hand_preview_manifest.json")
    args = parser.parse_args(argv)

    if args.samples_per_class < 1 or args.columns < 1 or min(args.tile_width, args.tile_height) < 64:
        print("ERROR: --samples-per-class and --columns must be >= 1; tile sizes >= 64 px.", file=sys.stderr)
        return 2

    from app.dataset_audit import DatasetRootError, validate_root
    from app.dataset_audit.hand_preview import PreviewError, run_preview

    try:
        root = validate_root(args.root or settings.dataset_root)
        manifest = run_preview(root, args.samples_per_class, args.columns, args.tile_width, args.tile_height,
                               args.output, args.manifest)
    except (DatasetRootError, PreviewError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Hand-state preview (layout: {manifest['layout']})\n")
    print(f"  {'Class':<26} {'Total':>6} {'Readable':>9} {'Selected':>9}")
    for name, c in manifest["classes"].items():
        if c["missing"]:
            print(f"  {name:<26} {'FOLDER MISSING':>26}")
        else:
            print(f"  {name:<26} {c['total_images']:>6} {c['readable_images']:>9} {c['selected']:>9}")
    unreadable = [u for c in manifest["classes"].values() for u in c["unreadable"]]
    print(f"\nUnreadable files: {len(unreadable)}")
    for u in unreadable:
        print(f"  - {u['source_path']}: {u['error']}")
    print(f"Exact-duplicate groups: {len(manifest['duplicate_groups'])}")
    for g in manifest["duplicate_groups"]:
        print(f"  - group {g['group']}: " + ", ".join(g["paths"]))
    shown_dups = [s["source_path"] for s in manifest["samples"] if s["duplicate_group"] is not None]
    if shown_dups:
        print(f"  (shown in the sheet: {', '.join(shown_dups)})")
    print(f"\nContact sheet: {args.output}\nManifest:      {args.manifest}")
    print("Source images were only read; nothing in the dataset was modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
