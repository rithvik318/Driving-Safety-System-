"""Train the hand-state classifier (MobileNetV3-Small, ImageNet-pretrained) on the prepared split.

Usage (from the project root):
    python scripts/train_hand_model.py --root "D:\\path\\to\\Data"
    python scripts/train_hand_model.py --root "..." --epochs 25 --batch-size 16 --lr 1e-3 \\
        --weight-decay 1e-4 --seed 42 --device auto --output-dir outputs

Reads data/processed/hand_train.json and hand_val.json (never the test split).
Writes:
    <output-dir>/models/hand_state_best.pt   (best validation macro-F1)
    <output-dir>/models/hand_state_last.pt   (last epoch)
    <output-dir>/reports/hand_training_history.json
The first run downloads the ImageNet weights once (~10 MB) into the torch cache.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python scripts/...`


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    missing = [m for m in ("torch", "torchvision") if importlib.util.find_spec(m) is None]
    if missing:
        print(f"ERROR: {', '.join(missing)} not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2

    from app.config import configure_logging, load_settings
    from app.driver.hand_model import TrainConfig

    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"ERROR: invalid configuration: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level)
    d = TrainConfig()

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="Dataset root (overrides DATASET_ROOT). Manifest paths are relative to it.")
    p.add_argument("--manifest-dir", type=Path, default=settings.data_dir / "processed")
    p.add_argument("--output-dir", type=Path, default=settings.output_dir, help="Base for models/ and reports/ (default outputs).")
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr, help="Head learning rate; backbone uses lr x --backbone-lr-factor.")
    p.add_argument("--backbone-lr-factor", type=float, default=d.backbone_lr_factor)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device, help="auto (CUDA if available, else CPU), cpu, or cuda.")
    p.add_argument("--hflip-prob", type=float, default=d.hflip_prob)
    p.add_argument("--patience", type=int, default=d.patience, help="Early-stopping patience in epochs (0 = off).")
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--no-pretrained", action="store_true", help="Random init (for offline smoke tests only).")
    args = p.parse_args(argv)

    from app.dataset_audit import DatasetRootError, validate_root
    from app.driver.hand_model import HandModelError, train_hand_model

    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, backbone_lr_factor=args.backbone_lr_factor,
        weight_decay=args.weight_decay, seed=args.seed, device=args.device, hflip_prob=args.hflip_prob,
        patience=args.patience, num_workers=args.num_workers, pretrained=not args.no_pretrained,
    )
    try:
        root = validate_root(args.root or settings.dataset_root)
        history = train_hand_model(root, args.manifest_dir, args.output_dir, cfg)
    except (DatasetRootError, HandModelError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"\nBest epoch {history['best_epoch']} · validation macro-F1 {history['best_val_macro_f1']:.4f}")
    print(f"Checkpoints: {args.output_dir / 'models' / 'hand_state_best.pt'}, {args.output_dir / 'models' / 'hand_state_last.pt'}")
    print(f"History:     {args.output_dir / 'reports' / 'hand_training_history.json'}")
    if not cfg.pretrained:
        print("NOTE: trained WITHOUT pretrained weights (--no-pretrained). Use only as a smoke test.")
    print("Next: python scripts/evaluate_hand_model.py --root <same root>   (runs on the untouched test split)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
