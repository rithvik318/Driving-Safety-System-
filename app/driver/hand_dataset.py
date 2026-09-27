"""Hand-state dataset preparation: discovery, duplicate handling, stratified split, manifests.

Rules:
- Exactly three classes, found inside the driver-camera folder:
  both_hands_on_steering (0), one_hand_on_steering (1), no_hands (2).
- Only images are used. Videos, unsupported files, drowsy_driver and front-camera data are ignored.
- Each image is an independent labelled observation; the split is done at IMAGE level.
  Nothing is inferred from filenames.
- Exact duplicates (same SHA-256): all source files stay untouched, but only one copy
  (shortest filename, then alphabetical) enters the modeling manifests. Identical files
  under DIFFERENT labels are a label conflict: every copy is excluded and reported.
- Deterministic stratified split with a fixed seed. Split sizes use the largest-remainder
  method per stratum so proportions stay as close as possible to the ratios.
- Split methods (SPLIT_METHODS):
    "class"                     strata = class. The original method.
    "class_and_capture_format"  strata = (class, capture format). Default.
  Capture format = stored pixel size + EXIF orientation tag, read from the image header.
  It is a file property, not a group ID: it does not identify source videos, recording
  sessions or people, and images from the same recording still land in different splits.
  It only makes sure each capture format is represented in every split in proportion.
- Read-only: sources are only read. No copies, resizing, re-encoding or augmentation.
  EXIF orientation is applied only when an image is loaded for training (load_training_image).
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from app.dataset_audit.probe import sha256_file
from app.dataset_audit.scan import IMAGE_EXTENSIONS, find_camera_dirs

CLASS_NAMES: tuple[str, ...] = ("both_hands_on_steering", "one_hand_on_steering", "no_hands")
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
CLASS_TO_HAND_STATE = {
    "both_hands_on_steering": "BOTH_HANDS",
    "one_hand_on_steering": "ONE_HAND",
    "no_hands": "NO_HANDS",
}
SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
DEFAULT_SEED = 42
SPLIT_METHODS = {
    "class": "stratified by class only; seeded shuffle within each class",
    "class_and_capture_format": (
        "stratified by (class, capture format); seeded shuffle within each stratum; "
        "largest-remainder sizes per stratum"
    ),
}
DEFAULT_SPLIT_METHOD = "class_and_capture_format"
CAPTURE_FORMAT_DEFINITION = (
    "stored width x height in pixels plus the EXIF orientation tag, read from the image header "
    "(not from the filename). A file property only: it does not identify source videos, sessions or people."
)
LIMITATIONS = [
    "True setup/recording-group separation is impossible: the source videos and group IDs are not available, "
    "and groups are deliberately NOT inferred from filenames.",
    "The split unit is the image. Many images are near-consecutive video frames that look almost identical, so "
    "near-duplicates of test images are very likely in train. Test metrics are therefore optimistic.",
    "Capture-format stratification only balances how each capture format is represented across splits; it does "
    "not separate recordings, people or sessions.",
    "In this dataset some capture formats occur in only one class, so scene/format can predict the label for those "
    "images. A stratified split cannot detect a model that relies on that shortcut.",
    "This split does not address the camera-domain shift: the intended demo video uses a different physical setup "
    "from every training setup (see outputs/reports/driver_camera_domain_comparison.md).",
    "With 27-30 validation/test images, per-class metrics have wide uncertainty.",
]


class HandDatasetError(Exception):
    """The dataset cannot be prepared (missing classes, bad ratios, too few images, failed validation)."""


@dataclass
class Candidate:
    path: str  # relative to the dataset root, POSIX style
    label: str
    size_bytes: int
    sha256: str | None = None
    error: str | None = None  # set if the image could not be read
    capture_format: str | None = None  # stored size + EXIF orientation, from the image header


@dataclass
class HandDatasetPlan:
    root_name: str
    seed: int
    ratios: dict[str, float]
    candidates: list[Candidate]
    excluded_non_images: list[str]
    unreadable: list[dict]
    duplicate_groups: list[dict]
    excluded_duplicates: list[dict]
    splits: dict[str, list[dict]] = field(default_factory=dict)
    split_method: str = DEFAULT_SPLIT_METHOD

    @property
    def records(self) -> list[dict]:
        return [r for s in SPLITS for r in self.splits.get(s, [])]


# --------------------------------------------------------------------------- discovery


def validate_ratios(train: float, val: float, test: float) -> dict[str, float]:
    ratios = {"train": train, "val": val, "test": test}
    for name, value in ratios.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 or value >= 1:
            raise HandDatasetError(f"{name} ratio must be between 0 and 1 (exclusive), got {value!r}")
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise HandDatasetError(f"split ratios must sum to 1.0, got {sum(ratios.values()):.6f} ({ratios})")
    return ratios


def discover_hand_images(root: Path) -> tuple[list[Candidate], list[str]]:
    """Find the three class folders under the driver-camera folder(s).

    Returns (image candidates sorted by path, non-image files ignored inside class folders).
    Raises HandDatasetError if the driver-camera folder or any class folder is missing.
    """
    camera_dirs = find_camera_dirs(root).get("driver_cam", [])
    if not camera_dirs:
        raise HandDatasetError(
            "No driver-camera folder found (looked for 'drivercamera' or 'driver_cam' up to 4 levels below the root)."
        )
    found: dict[str, list[Path]] = defaultdict(list)
    for cam in camera_dirs:
        for name in CLASS_NAMES:
            folder = root / cam / name
            if folder.is_dir():
                found[name].append(folder)
    missing = [n for n in CLASS_NAMES if n not in found]
    if missing:
        raise HandDatasetError(
            "Missing hand-state class folder(s): " + ", ".join(missing)
            + f" (expected inside {', '.join(camera_dirs)})"
        )

    candidates, ignored = [], []
    for name in CLASS_NAMES:
        for folder in found[name]:
            for p in sorted(folder.rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(root).as_posix()
                if p.name.startswith(".") or p.name.lower() in {"desktop.ini", "thumbs.db"}:
                    continue
                if p.suffix.lower() not in IMAGE_EXTENSIONS:
                    ignored.append(rel)
                    continue
                candidates.append(Candidate(path=rel, label=name, size_bytes=p.stat().st_size))
    candidates.sort(key=lambda c: (CLASS_TO_INDEX[c.label], c.path.casefold(), c.path))
    return candidates, sorted(ignored)


def check_image(path: Path) -> str | None:
    """None if the image decodes, else an error message. Read-only."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def read_capture_format(path: Path) -> str:
    """'<stored W>x<stored H>|exif_orientation=<tag or none>' from the image header. Read-only."""
    with Image.open(path) as im:
        w, h = im.size
        orientation = im.getexif().get(0x0112)
    return f"{w}x{h}|exif_orientation={orientation if orientation is not None else 'none'}"


def load_training_image(path: Path) -> Image.Image:
    """Load an image for training: EXIF orientation applied in memory, RGB. The file is never rewritten."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


# --------------------------------------------------------------------------- duplicates


def resolve_duplicates(candidates: list[Candidate]) -> tuple[list[Candidate], list[dict], list[dict]]:
    """Return (kept candidates, duplicate groups, excluded entries)."""
    by_hash: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_hash[c.sha256].append(c)

    kept, groups, excluded = [], [], []
    for digest, members in by_hash.items():
        if len(members) == 1:
            kept.append(members[0])
            continue
        labels = sorted({m.label for m in members})
        ordered = sorted(members, key=lambda m: (len(m.path.rsplit("/", 1)[-1]), m.path))
        group = {"sha256": digest, "paths": [m.path for m in ordered], "labels": labels}
        if len(labels) > 1:
            group["resolution"] = "label conflict: all copies excluded"
            excluded += [{"path": m.path, "label": m.label, "reason": "identical file under different labels"} for m in ordered]
        else:
            group["resolution"] = f"kept {ordered[0].path}"
            kept.append(ordered[0])
            excluded += [{"path": m.path, "label": m.label, "reason": f"exact duplicate of {ordered[0].path}"} for m in ordered[1:]]
        groups.append(group)
    kept.sort(key=lambda c: (CLASS_TO_INDEX[c.label], c.path.casefold(), c.path))
    return kept, sorted(groups, key=lambda g: g["paths"][0]), excluded


# --------------------------------------------------------------------------- split


def split_sizes(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation of n items; every split gets >= 1 item when n >= 3."""
    exact = {s: n * ratios[s] for s in SPLITS}
    sizes = {s: math.floor(exact[s]) for s in SPLITS}
    remainder = n - sum(sizes.values())
    for s in sorted(SPLITS, key=lambda s: (-(exact[s] - sizes[s]), SPLITS.index(s)))[:remainder]:
        sizes[s] += 1
    if n >= len(SPLITS):
        for s in SPLITS:
            if sizes[s] == 0:
                donor = max(SPLITS, key=lambda d: sizes[d])
                sizes[donor] -= 1
                sizes[s] += 1
    return sizes


def _strata(items: list[Candidate], label: str, seed: int, method: str) -> list[tuple[str, list[Candidate]]]:
    """[(RNG key, items)] for one class. 'class' keeps the original single stratum and RNG key."""
    if method == "class":
        return [(f"{seed}:{label}", items)]
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for c in items:
        groups[c.capture_format or "unknown"].append(c)
    return [(f"{seed}:{label}:{fmt}", groups[fmt]) for fmt in sorted(groups)]


def stratified_split(kept: list[Candidate], ratios: dict[str, float], seed: int,
                     method: str = DEFAULT_SPLIT_METHOD) -> dict[str, list[dict]]:
    """Shuffle each stratum with a seeded RNG (independent of file-system order) and cut by split_sizes."""
    if method not in SPLIT_METHODS:
        raise HandDatasetError(f"unknown split method {method!r}; choose one of: {', '.join(SPLIT_METHODS)}")
    by_class: dict[str, list[Candidate]] = defaultdict(list)
    for c in kept:
        by_class[c.label].append(c)
    splits: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for label in CLASS_NAMES:
        items = sorted(by_class[label], key=lambda c: c.path)
        if len(items) < len(SPLITS):
            raise HandDatasetError(f"class {label} has only {len(items)} usable image(s); need at least 3 for train/val/test")
        for rng_key, stratum in _strata(items, label, seed, method):
            stratum = sorted(stratum, key=lambda c: c.path)
            random.Random(rng_key).shuffle(stratum)
            sizes = split_sizes(len(stratum), ratios)
            start = 0
            for s in SPLITS:
                for c in stratum[start : start + sizes[s]]:
                    splits[s].append({
                        "path": c.path,
                        "label": c.label,
                        "class_index": CLASS_TO_INDEX[c.label],
                        "split": s,
                        "hand_state": CLASS_TO_HAND_STATE[c.label],
                        "sha256": c.sha256,
                        "capture_format": c.capture_format,
                    })
                start += sizes[s]
    for s in SPLITS:
        splits[s].sort(key=lambda r: (r["class_index"], r["path"]))
    return splits


def validate_plan(plan: HandDatasetPlan, kept: list[Candidate]) -> None:
    """Raise HandDatasetError if any split invariant is violated."""
    problems = []
    paths = [r["path"] for r in plan.records]
    counts = Counter(paths)
    repeated = [p for p, n in counts.items() if n > 1]
    if repeated:
        problems.append(f"images in more than one split: {repeated[:5]}")
    missing = {c.path for c in kept} - set(paths)
    if missing:
        problems.append(f"usable images missing from all splits: {sorted(missing)[:5]}")
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for r in plan.records:
        hash_splits[r["sha256"]].add(r["split"])
    crossing = [h for h, s in hash_splits.items() if len(s) > 1]
    if crossing:
        problems.append(f"{len(crossing)} identical file hash(es) occur in different splits")
    for s in SPLITS:
        present = {r["label"] for r in plan.splits[s]}
        absent = [c for c in CLASS_NAMES if c not in present]
        if absent:
            problems.append(f"split {s} has no images of: {', '.join(absent)}")
    if problems:
        raise HandDatasetError("split validation failed: " + "; ".join(problems))


# --------------------------------------------------------------------------- orchestration


def _fingerprint(root: Path, candidates: list[Candidate]) -> dict[str, tuple[int, int]]:
    out = {}
    for c in candidates:
        st = (root / c.path).stat()
        out[c.path] = (st.st_size, st.st_mtime_ns)
    return out


def prepare_hand_dataset(root: Path, seed: int = DEFAULT_SEED, ratios: tuple[float, float, float] = DEFAULT_RATIOS,
                         split_method: str = DEFAULT_SPLIT_METHOD) -> HandDatasetPlan:
    """Discover, check, de-duplicate, split and validate. Reads the dataset only; writes nothing."""
    ratio_map = validate_ratios(*ratios)
    if split_method not in SPLIT_METHODS:
        raise HandDatasetError(f"unknown split method {split_method!r}; choose one of: {', '.join(SPLIT_METHODS)}")
    candidates, ignored = discover_hand_images(root)
    before = _fingerprint(root, candidates)

    unreadable = []
    readable = []
    for c in candidates:
        c.error = check_image(root / c.path)
        if c.error:
            unreadable.append({"path": c.path, "label": c.label, "error": c.error.replace(str(root / c.path), c.path)})
            continue
        c.sha256 = sha256_file(root / c.path)
        c.capture_format = read_capture_format(root / c.path)
        readable.append(c)

    kept, dup_groups, dup_excluded = resolve_duplicates(readable)
    plan = HandDatasetPlan(
        root_name=root.name,
        seed=seed,
        ratios=ratio_map,
        candidates=candidates,
        excluded_non_images=ignored,
        unreadable=unreadable,
        duplicate_groups=dup_groups,
        excluded_duplicates=dup_excluded,
        split_method=split_method,
    )
    plan.splits = stratified_split(kept, ratio_map, seed, split_method)
    validate_plan(plan, kept)

    if _fingerprint(root, candidates) != before:
        raise HandDatasetError("source files changed while preparing the split (another program is writing to the dataset?)")
    return plan


def per_class_counts(plan: HandDatasetPlan) -> dict[str, dict[str, int]]:
    table = {}
    for label in CLASS_NAMES:
        row = {
            "source_images": sum(1 for c in plan.candidates if c.label == label),
            "unreadable": sum(1 for u in plan.unreadable if u["label"] == label),
            "excluded_duplicates": sum(1 for d in plan.excluded_duplicates if d["label"] == label),
        }
        for s in SPLITS:
            row[s] = sum(1 for r in plan.splits[s] if r["label"] == label)
        row["usable"] = row["train"] + row["val"] + row["test"]
        table[label] = row
    return table


def capture_format_counts(plan: HandDatasetPlan) -> dict[str, dict[str, dict[str, int]]]:
    """{class: {capture format: {train, val, test, total}}} over the usable images."""
    table: dict[str, dict[str, dict[str, int]]] = {}
    for r in plan.records:
        row = table.setdefault(r["label"], {}).setdefault(r["capture_format"] or "unknown", {s: 0 for s in SPLITS})
        row[r["split"]] += 1
    for label in table:
        for row in table[label].values():
            row["total"] = sum(row[s] for s in SPLITS)
    return {label: dict(sorted(table[label].items())) for label in CLASS_NAMES if label in table}


def compare_splits(old: dict[str, list[dict]], new: dict[str, list[dict]]) -> dict:
    """How many images changed split between two sets of split records (matched by path)."""
    old_of = {r["path"]: s for s in SPLITS for r in old.get(s, [])}
    new_of = {r["path"]: s for s in SPLITS for r in new.get(s, [])}
    common = old_of.keys() & new_of.keys()
    moved = sorted(p for p in common if old_of[p] != new_of[p])
    moves: dict[str, int] = Counter(f"{old_of[p]}->{new_of[p]}" for p in moved)
    return {
        "images_compared": len(common),
        "images_moved": len(moved),
        "images_unchanged": len(common) - len(moved),
        "moves": dict(sorted(moves.items())),
        "only_in_previous": len(old_of.keys() - new_of.keys()),
        "only_in_new": len(new_of.keys() - old_of.keys()),
        "test_images_previously_in_train": sum(1 for p in common if new_of[p] == "test" and old_of[p] == "train"),
        "val_images_previously_in_train": sum(1 for p in common if new_of[p] == "val" and old_of[p] == "train"),
    }


def build_manifest(plan: HandDatasetPlan, previous_splits: dict[str, list[dict]] | None = None) -> dict:
    per_class = per_class_counts(plan)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_root": plan.root_name,
        "path_note": "Record paths are relative to DATASET_ROOT; only the root folder name is stored here.",
        "data_source": "real_local",
        "split_unit": "image (each image treated as an independent labelled observation)",
        "class_mapping": {name: CLASS_TO_INDEX[name] for name in CLASS_NAMES},
        "hand_state_mapping": CLASS_TO_HAND_STATE,
        "seed": plan.seed,
        "split_ratios": plan.ratios,
        "split_method": plan.split_method,
        "split_method_description": SPLIT_METHODS[plan.split_method],
        "capture_format_definition": CAPTURE_FORMAT_DEFINITION,
        "duplicate_rule": "exact SHA-256 duplicates: one copy kept (shortest filename, then alphabetical); "
                          "identical files under different labels are all excluded",
        "total_source_images": len(plan.candidates),
        "total_usable_images": len(plan.records),
        "unreadable_images": plan.unreadable,
        "ignored_non_image_files": plan.excluded_non_images,
        "duplicate_files": plan.duplicate_groups,
        "excluded_duplicates": plan.excluded_duplicates,
        "train_count": len(plan.splits["train"]),
        "val_count": len(plan.splits["val"]),
        "test_count": len(plan.splits["test"]),
        "per_class_counts": per_class,
        "per_split_class_counts": {s: {c: per_class[c][s] for c in CLASS_NAMES} for s in SPLITS},
        "capture_format_counts": capture_format_counts(plan),
        "augmentation": "none",
        "exif_handling": "Source files are not rewritten; apply EXIF orientation when loading (load_training_image).",
        "split_files": {s: f"hand_{s}.json" for s in SPLITS},
        "limitations": LIMITATIONS,
    }
    if previous_splits is not None:
        manifest["previous_split_comparison"] = compare_splits(previous_splits, plan.splits)
    return manifest


def render_report(manifest: dict) -> str:
    pc = manifest["per_class_counts"]
    r = manifest["split_ratios"]
    lines = [
        "# Hand-state dataset split",
        "",
        f"- Dataset folder: `{manifest['dataset_root']}` · created {manifest['created_at']} (UTC) · provenance `real_local`",
        f"- Seed: {manifest['seed']} · ratios train {r['train']:.2f} / val {r['val']:.2f} / test {r['test']:.2f}",
        f"- Split method: `{manifest['split_method']}`: {manifest['split_method_description']}",
        f"- Capture format = {manifest['capture_format_definition']}",
        "- Split unit: image. Each image is treated as an independent labelled observation; nothing is inferred from filenames.",
        "- No augmentation, resizing or copying. Source files are untouched; EXIF orientation is applied only at training-load time.",
        "",
        "| Class | Index | Total usable | Train | Validation | Test |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in CLASS_NAMES:
        c = pc[name]
        lines.append(f"| `{name}` | {manifest['class_mapping'][name]} | {c['usable']} | {c['train']} | {c['val']} | {c['test']} |")
    total = manifest["total_usable_images"]
    lines += [
        f"| **Total** | | **{total}** | **{manifest['train_count']}** | **{manifest['val_count']}** | **{manifest['test_count']}** |",
        "",
        "## Source accounting",
        "",
        "| Class | Source images | Unreadable | Excluded duplicates | Usable |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in CLASS_NAMES:
        c = pc[name]
        lines.append(f"| `{name}` | {c['source_images']} | {c['unreadable']} | {c['excluded_duplicates']} | {c['usable']} |")
    lines += ["", "## Exact duplicates (SHA-256)", ""]
    if manifest["duplicate_files"]:
        for g in manifest["duplicate_files"]:
            lines.append(f"- {', '.join(f'`{p}`' for p in g['paths'])} → {g['resolution']}")
        lines.append("")
        lines.append("All source files are kept on disk; excluded copies are only left out of the split manifests.")
    else:
        lines.append("None found.")
    lines += ["", "## Unreadable images", ""]
    lines += [f"- `{u['path']}`: {u['error']}" for u in manifest["unreadable_images"]] or ["None."]
    if manifest["ignored_non_image_files"]:
        lines += ["", "## Ignored non-image files in class folders", ""]
        lines += [f"- `{p}`" for p in manifest["ignored_non_image_files"]]
    lines += ["", "## Capture formats per split", "",
              "| Class | Capture format | Total | Train | Validation | Test |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, formats in manifest["capture_format_counts"].items():
        for fmt, c in formats.items():
            lines.append(f"| `{name}` | `{fmt}` | {c['total']} | {c['train']} | {c['val']} | {c['test']} |")
    comp = manifest.get("previous_split_comparison")
    if comp:
        lines += ["", "## Change from the previous split", "",
                  f"- Images compared: {comp['images_compared']} · moved to a different split: {comp['images_moved']} · "
                  f"unchanged: {comp['images_unchanged']}",
                  f"- Moves: {', '.join(f'{k}: {v}' for k, v in comp['moves'].items()) or 'none'}",
                  f"- New test images that were in the previous train split: {comp['test_images_previously_in_train']}; "
                  f"new validation images previously in train: {comp['val_images_previously_in_train']}"]
        if comp["test_images_previously_in_train"] or comp["val_images_previously_in_train"]:
            lines.append("- A checkpoint trained on the previous split has seen some of the new validation/test images. "
                         "Its metrics on the new split are NOT a valid held-out evaluation; retrain before evaluating.")
    lines += [
        "",
        "## Checks passed",
        "",
        "- Every usable image appears in exactly one split.",
        "- No file hash appears in more than one split.",
        "- All three classes appear in train, validation and test.",
        "- Source files were unchanged (size and modification time) before and after preparation.",
        "",
        "## Limitations",
        "",
        *[f"- {x}" for x in manifest["limitations"]],
        "- Only byte-identical duplicates are detected; visually near-identical images are not controlled across splits.",
        "",
    ]
    return "\n".join(lines)


def read_previous_splits(processed_dir: Path) -> dict[str, list[dict]] | None:
    """Existing hand_{split}.json records, or None if any file is missing or unreadable."""
    out = {}
    for s in SPLITS:
        f = processed_dir / f"hand_{s}.json"
        try:
            out[s] = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return out


def write_outputs(plan: HandDatasetPlan, processed_dir: Path, report_path: Path) -> dict:
    manifest = build_manifest(plan, read_previous_splits(processed_dir))
    processed_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        (processed_dir / f"hand_{s}.json").write_text(json.dumps(plan.splits[s], indent=2, ensure_ascii=False), encoding="utf-8")
    (processed_dir / "hand_dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(manifest), encoding="utf-8")
    return manifest
