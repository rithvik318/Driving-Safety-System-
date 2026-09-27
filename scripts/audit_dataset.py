"""Audit the external local dataset. Read-only: never modifies, copies or deletes data.

Usage (from the project root):
    python scripts/audit_dataset.py --root "G:/My Drive/dataset"
    python scripts/audit_dataset.py            # uses DATASET_ROOT from .env

Writes:
    data/processed/dataset_manifest.json
    outputs/reports/dataset_audit.md
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python scripts/...`

REQUIRED_MODULES = {"PIL": "Pillow", "cv2": "opencv-contrib-python", "dotenv": "python-dotenv"}


def parse_args(argv: list[str] | None, settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help="Dataset root folder (overrides DATASET_ROOT).")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=settings.data_dir / "processed" / "dataset_manifest.json",
        help="Output path for the JSON manifest.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=settings.output_dir / "reports" / "dataset_audit.md",
        help="Output path for the Markdown report.",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide per-file progress.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles may not encode every filename
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    missing = [pkg for mod, pkg in REQUIRED_MODULES.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"ERROR: missing packages: {', '.join(missing)}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2

    from app.config import configure_logging, load_settings

    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"ERROR: invalid configuration: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level)
    args = parse_args(argv, settings)

    from app.dataset_audit import DatasetRootError, run_audit, validate_root, write_manifest, write_report

    try:
        root = validate_root(args.root or settings.dataset_root)
    except DatasetRootError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    def progress(i: int, total: int, rel: str) -> None:
        if not args.quiet:
            print(f"[{i}/{total}] {rel}", file=sys.stderr)

    logging.getLogger(__name__).info("Auditing dataset folder: %s", root.name)
    manifest = run_audit(root, progress=progress)
    write_manifest(manifest, args.manifest)
    write_report(manifest, args.report)

    t = manifest["totals"]
    print(
        f"\nAudit complete: {t['files']} files, {t['images']} images, {t['videos']} videos, "
        f"{t['unsupported']} unsupported, {t['corrupt']} corrupt, {t['empty']} empty, "
        f"{len(manifest['duplicate_groups'])} duplicate group(s)."
    )
    print(f"Manifest: {args.manifest}\nReport:   {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
