"""Prepare the hand-state train/val/test split as manifests. Read-only on the dataset.

Usage (from the project root):
    python scripts/prepare_hand_dataset.py --root "D:\\path\\to\\Data" --seed 42
    python scripts/prepare_hand_dataset.py --root "..." --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
    python scripts/prepare_hand_dataset.py --root "..." --split-method class   # the original class-only split

Default split method: class_and_capture_format (stratified by class and by the capture format
read from each image header). If split manifests already exist in --out-dir, the report
records how many images moved to a different split.

Writes:
    data/processed/hand_train.json, hand_val.json, hand_test.json
    data/processed/hand_dataset_manifest.json
    outputs/reports/hand_dataset_split.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python scripts/...`


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

    from app.driver.hand_dataset import DEFAULT_RATIOS, DEFAULT_SEED, DEFAULT_SPLIT_METHOD, SPLIT_METHODS

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help="Dataset root folder (overrides DATASET_ROOT).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default {DEFAULT_SEED}).")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS[0])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS[1])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS[2])
    parser.add_argument("--split-method", choices=sorted(SPLIT_METHODS), default=DEFAULT_SPLIT_METHOD,
                        help=f"Stratification (default {DEFAULT_SPLIT_METHOD}).")
    parser.add_argument("--out-dir", type=Path, default=settings.data_dir / "processed", help="Where the JSON manifests go.")
    parser.add_argument("--report", type=Path, default=settings.output_dir / "reports" / "hand_dataset_split.md")
    args = parser.parse_args(argv)

    from app.dataset_audit import DatasetRootError, validate_root
    from app.driver.hand_dataset import CLASS_NAMES, SPLITS, HandDatasetError, prepare_hand_dataset, write_outputs

    try:
        root = validate_root(args.root or settings.dataset_root)
        plan = prepare_hand_dataset(root, args.seed, (args.train_ratio, args.val_ratio, args.test_ratio), args.split_method)
        manifest = write_outputs(plan, args.out_dir, args.report)
    except (DatasetRootError, HandDatasetError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    pc = manifest["per_class_counts"]
    print(f"Hand-state split  (seed {manifest['seed']}, ratios {manifest['split_ratios']}, method {manifest['split_method']})\n")
    print(f"  {'Class':<24} {'Source':>6} {'Usable':>6} {'Train':>6} {'Val':>5} {'Test':>5}")
    for name in CLASS_NAMES:
        c = pc[name]
        print(f"  {name:<24} {c['source_images']:>6} {c['usable']:>6} {c['train']:>6} {c['val']:>5} {c['test']:>5}")
    print(f"  {'TOTAL':<24} {manifest['total_source_images']:>6} {manifest['total_usable_images']:>6} "
          f"{manifest['train_count']:>6} {manifest['val_count']:>5} {manifest['test_count']:>5}")
    print("\n  Capture formats (from image headers):")
    print(f"  {'Class':<24} {'Capture format':<34} {'Total':>5} {'Train':>6} {'Val':>5} {'Test':>5}")
    for name, formats in manifest["capture_format_counts"].items():
        for fmt, c in formats.items():
            print(f"  {name:<24} {fmt:<34} {c['total']:>5} {c['train']:>6} {c['val']:>5} {c['test']:>5}")
    comp = manifest.get("previous_split_comparison")
    if comp:
        print(f"\nCompared with the previous split: {comp['images_moved']} of {comp['images_compared']} images moved "
              f"({', '.join(f'{k} {v}' for k, v in comp['moves'].items()) or 'none'}).")
        if comp["test_images_previously_in_train"] or comp["val_images_previously_in_train"]:
            print(f"  WARNING: {comp['test_images_previously_in_train']} new test and {comp['val_images_previously_in_train']} "
                  "new validation images were in the previous TRAIN split. A checkpoint trained on the previous split "
                  "cannot be evaluated on this split; retrain first.")
    print(f"\nUnreadable images: {len(manifest['unreadable_images'])}")
    for u in manifest["unreadable_images"]:
        print(f"  - {u['path']}: {u['error']}")
    print(f"Exact-duplicate groups: {len(manifest['duplicate_files'])}")
    for g in manifest["duplicate_files"]:
        print(f"  - {' | '.join(g['paths'])}  -> {g['resolution']}")
    if manifest["ignored_non_image_files"]:
        print(f"Ignored non-image files in class folders: {len(manifest['ignored_non_image_files'])}")
    print("\nValidation: every image in exactly one split; no hash across splits; all classes in every split; sources unchanged.")
    print(f"\nWrote: {', '.join(str(args.out_dir / f'hand_{s}.json') for s in SPLITS)}")
    print(f"       {args.out_dir / 'hand_dataset_manifest.json'}\n       {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
