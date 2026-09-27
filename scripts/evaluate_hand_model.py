"""Evaluate a trained hand-state checkpoint on the untouched test split.

Usage (from the project root):
    python scripts/evaluate_hand_model.py --root "D:\\path\\to\\Data"
    python scripts/evaluate_hand_model.py --root "..." --checkpoint outputs/models/hand_state_last.pt --threshold 0.6

Writes:
    <output-dir>/reports/hand_test_metrics.json
    <output-dir>/reports/hand_test_report.md
    <output-dir>/plots/hand_confusion_matrix.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python scripts/...`


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    missing = [m for m in ("torch", "torchvision", "matplotlib") if importlib.util.find_spec(m) is None]
    if missing:
        print(f"ERROR: {', '.join(missing)} not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2

    from app.config import configure_logging, load_settings
    from app.driver.hand_model import DEFAULT_THRESHOLD

    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"ERROR: invalid configuration: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="Dataset root (overrides DATASET_ROOT).")
    p.add_argument("--manifest-dir", type=Path, default=settings.data_dir / "processed")
    p.add_argument("--output-dir", type=Path, default=settings.output_dir)
    p.add_argument("--checkpoint", type=Path, help="Default: <output-dir>/models/hand_state_best.pt")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Confidence below this -> UNKNOWN (analysis only).")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    checkpoint = args.checkpoint or args.output_dir / "models" / "hand_state_best.pt"
    if not 0 <= args.threshold <= 1:
        print("ERROR: --threshold must be between 0 and 1", file=sys.stderr)
        return 2

    from app.dataset_audit import DatasetRootError, validate_root
    from app.driver.hand_model import (
        HandModelError,
        HandStateClassifier,
        evaluate_split,
        load_split,
        plot_confusion_matrix,
        render_test_report,
    )

    try:
        root = validate_root(args.root or settings.dataset_root)
        records = load_split(args.manifest_dir / f"hand_{args.split}.json")
        clf = HandStateClassifier.from_checkpoint(checkpoint, device=args.device, threshold=args.threshold)
        metrics = evaluate_split(clf, records, root, args.threshold)
    except (DatasetRootError, HandModelError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    ds_manifest = args.manifest_dir / "hand_dataset_manifest.json"
    total_usable = None
    if ds_manifest.is_file():
        total_usable = json.loads(ds_manifest.read_text(encoding="utf-8")).get("total_usable_images")
    if total_usable is None:
        total_usable = sum(len(load_split(args.manifest_dir / f"hand_{s}.json")) for s in ("train", "val", "test"))

    metrics.update(
        evaluated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        split=args.split,
        checkpoint=checkpoint.name,
        checkpoint_epoch=clf.meta.get("epoch"),
        device=clf.device,
        dataset_total_usable_images=total_usable,
        note=f"Small dataset: {total_usable} unique usable images; {args.split} set has {metrics['num_samples']} images. "
             "Not a production accuracy claim.",
    )
    reports, plots = args.output_dir / "reports", args.output_dir / "plots"
    reports.mkdir(parents=True, exist_ok=True)
    prefix = "hand_test" if args.split == "test" else "hand_val"
    (reports / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusion_matrix(metrics["confusion_matrix"], plots / f"hand_{'' if args.split == 'test' else 'val_'}confusion_matrix.png",
                          f"Hand-state confusion matrix ({args.split}, n={metrics['num_samples']})")
    (reports / f"{prefix}_report.md").write_text(
        render_test_report(metrics, clf.meta, {"total_usable": total_usable}, args.split), encoding="utf-8")

    print(f"{args.split} set: {metrics['num_samples']} images (dataset: {total_usable} usable) · checkpoint {checkpoint.name} "
          f"(epoch {clf.meta.get('epoch')})")
    print(f"  accuracy {metrics['accuracy']:.4f} · macro P {metrics['macro_precision']:.4f} · "
          f"macro R {metrics['macro_recall']:.4f} · macro F1 {metrics['macro_f1']:.4f}")
    for name, m in metrics["per_class"].items():
        print(f"  {name:<24} n={m['support']:<3} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}")
    ta = metrics["threshold_analysis"]
    print(f"  threshold {ta['threshold']:.2f}: {ta['reported_unknown']} reported UNKNOWN (coverage {ta['coverage']:.1%})")
    print(f"Wrote {reports / f'{prefix}_metrics.json'}, {reports / f'{prefix}_report.md'} and the confusion-matrix PNG.")
    print("Small test set: treat these numbers as indicative only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
