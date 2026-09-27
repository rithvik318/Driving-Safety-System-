"""Tests for hand-state dataset preparation (app/driver/hand_dataset.py).

Images are tiny software-test fixtures in temp dirs; no real dataset is used.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

from app.driver.hand_dataset import (
    CLASS_NAMES,
    DEFAULT_SPLIT_METHOD,
    SPLITS,
    HandDatasetError,
    compare_splits,
    discover_hand_images,
    load_training_image,
    prepare_hand_dataset,
    read_capture_format,
    split_sizes,
    validate_ratios,
    write_outputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIVER = "drivercamera-20260926T193339Z-1-001/drivercamera"
REAL_COUNTS = {"both_hands_on_steering": 30, "one_hand_on_steering": 97, "no_hands": 61}


def make_dataset(tmp_path: Path, counts=None, with_duplicate=True) -> Path:
    """Unique images per class (+ one exact duplicate in no_hands), plus data that must be ignored."""
    counts = counts or {"both_hands_on_steering": 10, "one_hand_on_steering": 20, "no_hands": 12}
    root = tmp_path / "Data"
    k = 0
    for cls, n in counts.items():
        folder = root / DRIVER / cls
        folder.mkdir(parents=True)
        for i in range(n):
            k += 1
            Image.new("RGB", (24, 40), (k % 256, (k * 7) % 256, (k * 13) % 256)).save(folder / f"frame_{i:04d}.jpg")
    if with_duplicate:
        shutil.copy(root / DRIVER / "no_hands" / "frame_0001.jpg", root / DRIVER / "no_hands" / "Copy of frame_0001.jpg")
    (root / DRIVER / "drowsy_driver").mkdir()
    Image.new("RGB", (8, 8)).save(root / DRIVER / "drowsy_driver" / "still.jpg")  # must be ignored
    (root / DRIVER / "one_hand_on_steering" / "clip.mp4").write_bytes(b"\x00" * 16)  # non-image in class folder
    front = root / "frontcamera-x" / "frontcamera" / "roads"
    front.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(front / "r.jpg")
    return root


def fingerprint(root: Path) -> dict:
    return {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


# --- class discovery ---------------------------------------------------------------------------------

def test_discovers_exactly_three_classes_and_ignores_the_rest(tmp_path):
    root = make_dataset(tmp_path)
    candidates, ignored = discover_hand_images(root)
    assert {c.label for c in candidates} == set(CLASS_NAMES)
    assert len(candidates) == 10 + 20 + 13  # includes the duplicate copy
    assert not any("drowsy_driver" in c.path or "frontcamera" in c.path for c in candidates)
    assert ignored == [f"{DRIVER}/one_hand_on_steering/clip.mp4"]


@pytest.mark.parametrize("missing", CLASS_NAMES)
def test_missing_class_raises(tmp_path, missing):
    root = make_dataset(tmp_path)
    shutil.rmtree(root / DRIVER / missing)
    with pytest.raises(HandDatasetError, match=f"Missing hand-state class folder.*{missing}"):
        prepare_hand_dataset(root)


def test_no_driver_camera_folder_raises(tmp_path):
    (tmp_path / "Data" / "other").mkdir(parents=True)
    with pytest.raises(HandDatasetError, match="No driver-camera folder"):
        prepare_hand_dataset(tmp_path / "Data")


def test_too_few_images_in_a_class_raises(tmp_path):
    root = make_dataset(tmp_path, {"both_hands_on_steering": 2, "one_hand_on_steering": 9, "no_hands": 9}, with_duplicate=False)
    with pytest.raises(HandDatasetError, match="only 2 usable"):
        prepare_hand_dataset(root)


# --- ratios -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ratios", [(0.7, 0.2, 0.2), (0.5, 0.25, 0.2), (1.0, 0.0, 0.0), (0.8, -0.1, 0.3), (0.7, 0.15, float("nan"))])
def test_invalid_split_ratios_rejected(ratios):
    with pytest.raises(HandDatasetError):
        validate_ratios(*ratios)


def test_valid_ratios_accepted():
    assert validate_ratios(0.8, 0.1, 0.1) == {"train": 0.8, "val": 0.1, "test": 0.1}


# --- split sizes / stratification -------------------------------------------------------------------------

def test_split_sizes_for_the_real_class_counts():
    r = {"train": 0.70, "val": 0.15, "test": 0.15}
    assert split_sizes(30, r) == {"train": 21, "val": 5, "test": 4}
    assert split_sizes(97, r) == {"train": 68, "val": 15, "test": 14}
    assert split_sizes(61, r) == {"train": 43, "val": 9, "test": 9}
    assert split_sizes(3, r) == {"train": 1, "val": 1, "test": 1}


def test_stratification_matches_real_counts(tmp_path):
    root = make_dataset(tmp_path, dict(REAL_COUNTS), with_duplicate=True)  # no_hands: 61 unique + 1 copy = 62 files
    plan = prepare_hand_dataset(root, seed=42)
    per = {s: {c: sum(1 for r in plan.splits[s] if r["label"] == c) for c in CLASS_NAMES} for s in SPLITS}
    assert per["train"] == {"both_hands_on_steering": 21, "one_hand_on_steering": 68, "no_hands": 43}
    assert per["val"] == {"both_hands_on_steering": 5, "one_hand_on_steering": 15, "no_hands": 9}
    assert per["test"] == {"both_hands_on_steering": 4, "one_hand_on_steering": 14, "no_hands": 9}
    assert (len(plan.splits["train"]), len(plan.splits["val"]), len(plan.splits["test"])) == (132, 29, 27)
    for label, n in REAL_COUNTS.items():  # each split's class share stays close to the overall share
        overall = n / 188
        for s in SPLITS:
            share = per[s][label] / len(plan.splits[s])
            assert abs(share - overall) < 0.06


# --- determinism ------------------------------------------------------------------------------------------------

def test_split_is_deterministic_for_a_seed(tmp_path):
    root = make_dataset(tmp_path)
    a = prepare_hand_dataset(root, seed=42).splits
    b = prepare_hand_dataset(root, seed=42).splits
    assert a == b


def test_different_seed_changes_membership_not_counts(tmp_path):
    root = make_dataset(tmp_path)
    a = prepare_hand_dataset(root, seed=42).splits
    b = prepare_hand_dataset(root, seed=7).splits
    assert {s: len(a[s]) for s in SPLITS} == {s: len(b[s]) for s in SPLITS}
    assert {r["path"] for r in a["test"]} != {r["path"] for r in b["test"]}


# --- duplicates ----------------------------------------------------------------------------------------------------

def test_exact_duplicate_kept_once_sources_untouched(tmp_path):
    root = make_dataset(tmp_path)
    before = fingerprint(root)
    plan = prepare_hand_dataset(root)
    paths = [r["path"] for r in plan.records]
    kept, copy = f"{DRIVER}/no_hands/frame_0001.jpg", f"{DRIVER}/no_hands/Copy of frame_0001.jpg"
    assert kept in paths and copy not in paths
    assert plan.excluded_duplicates == [{"path": copy, "label": "no_hands", "reason": f"exact duplicate of {kept}"}]
    assert (root / copy).is_file()  # still on disk
    assert fingerprint(root) == before


def test_identical_file_under_two_labels_is_excluded_entirely(tmp_path):
    root = make_dataset(tmp_path, with_duplicate=False)
    shutil.copy(root / DRIVER / "no_hands" / "frame_0000.jpg", root / DRIVER / "both_hands_on_steering" / "conflict.jpg")
    plan = prepare_hand_dataset(root)
    paths = {r["path"] for r in plan.records}
    assert f"{DRIVER}/no_hands/frame_0000.jpg" not in paths and f"{DRIVER}/both_hands_on_steering/conflict.jpg" not in paths
    assert plan.duplicate_groups[0]["resolution"].startswith("label conflict")


def test_no_hash_occurs_in_two_splits_and_each_image_once(tmp_path):
    plan = prepare_hand_dataset(make_dataset(tmp_path))
    hashes = {}
    for r in plan.records:
        assert r["sha256"] not in hashes or hashes[r["sha256"]] == r["split"]
        hashes[r["sha256"]] = r["split"]
    paths = [r["path"] for r in plan.records]
    assert len(paths) == len(set(paths)) == 10 + 20 + 12
    assert all({r["label"] for r in plan.splits[s]} == set(CLASS_NAMES) for s in SPLITS)


# --- corrupt images --------------------------------------------------------------------------------------------------

def test_corrupt_image_reported_and_excluded(tmp_path):
    root = make_dataset(tmp_path)
    (root / DRIVER / "no_hands" / "broken.jpg").write_bytes(b"not a jpeg at all")
    plan = prepare_hand_dataset(root)
    assert [u["path"] for u in plan.unreadable] == [f"{DRIVER}/no_hands/broken.jpg"]
    assert str(root) not in plan.unreadable[0]["error"]
    assert f"{DRIVER}/no_hands/broken.jpg" not in {r["path"] for r in plan.records}


# --- manifests + report -----------------------------------------------------------------------------------------------

def test_manifest_generation(tmp_path):
    root = make_dataset(tmp_path)
    out, report = tmp_path / "processed", tmp_path / "reports" / "hand_dataset_split.md"
    manifest = write_outputs(prepare_hand_dataset(root, seed=42), out, report)

    for s in SPLITS:
        records = json.loads((out / f"hand_{s}.json").read_text(encoding="utf-8"))
        assert records and all(r["split"] == s for r in records)
        assert set(records[0]) >= {"path", "label", "class_index", "split"}
        assert all(r["class_index"] == CLASS_NAMES.index(r["label"]) for r in records)

    m = json.loads((out / "hand_dataset_manifest.json").read_text(encoding="utf-8"))
    for key in ["dataset_root", "class_mapping", "seed", "split_ratios", "total_source_images", "duplicate_files",
                "excluded_duplicates", "train_count", "val_count", "test_count", "per_class_counts"]:
        assert key in m, key
    assert m["class_mapping"] == {"both_hands_on_steering": 0, "one_hand_on_steering": 1, "no_hands": 2}
    assert m["total_source_images"] == 43 and m["total_usable_images"] == 42
    assert m["train_count"] + m["val_count"] + m["test_count"] == 42
    assert m["dataset_root"] == "Data" and str(root) not in json.dumps(m)
    assert m == manifest

    text = report.read_text(encoding="utf-8")
    assert "| Class | Index | Total usable | Train | Validation | Test |" in text
    assert "Copy of frame_0001.jpg" in text


def test_training_loader_applies_exif_without_rewriting(tmp_path):
    src = tmp_path / "rot.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (40, 24), (1, 2, 3)).save(src, exif=exif)
    before = src.read_bytes()
    img = load_training_image(src)
    assert img.size == (24, 40) and img.mode == "RGB"
    assert src.read_bytes() == before


# --- CLI -----------------------------------------------------------------------------------------------------------------

def _cli():
    spec = importlib.util.spec_from_file_location("prepare_cli", PROJECT_ROOT / "scripts" / "prepare_hand_dataset.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def no_dotenv(monkeypatch):
    monkeypatch.setattr("app.config.settings.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("DATASET_ROOT", raising=False)


def test_cli_end_to_end(no_dotenv, tmp_path, capsys):
    root = make_dataset(tmp_path)
    out, report = tmp_path / "processed", tmp_path / "split.md"
    code = _cli().main(["--root", str(root), "--seed", "42", "--out-dir", str(out), "--report", str(report)])
    text = capsys.readouterr().out
    assert code == 0 and (out / "hand_train.json").is_file() and report.is_file()
    assert "Exact-duplicate groups: 1" in text and "TOTAL" in text


def test_cli_rejects_bad_ratios_and_missing_root(no_dotenv, tmp_path, capsys):
    root = make_dataset(tmp_path)
    assert _cli().main(["--root", str(root), "--train-ratio", "0.9", "--out-dir", str(tmp_path / "o")]) == 2
    assert "sum to 1.0" in capsys.readouterr().err
    assert _cli().main(["--root", str(tmp_path / "nope")]) == 2
    assert not (tmp_path / "o").exists()  # nothing written on failure


# --- capture-format stratification ---------------------------------------------------------------------------------------

OFFICE, CAR_A, CAR_B = "1080x1920|exif_orientation=8", "720x1280|exif_orientation=none", "848x480|exif_orientation=8"
# Mirrors the real dataset's formats and counts (unique images), read from headers, not filenames.
REAL_FORMATS = {
    "both_hands_on_steering": {OFFICE: 30},
    "one_hand_on_steering": {OFFICE: 56, CAR_A: 41},
    "no_hands": {OFFICE: 43, CAR_B: 18},
}
FMT_SPEC = {OFFICE: ((24, 40), 8), CAR_A: ((18, 32), None), CAR_B: ((32, 18), 8)}  # tiny stand-ins


def make_format_dataset(tmp_path: Path, spec=None) -> Path:
    """Tiny images whose header size/EXIF identify a capture format; filenames carry no format hint."""
    spec = spec or REAL_FORMATS
    root = tmp_path / "Data"
    k = 0
    for cls, formats in spec.items():
        folder = root / DRIVER / cls
        folder.mkdir(parents=True, exist_ok=True)
        for fmt, n in formats.items():
            size, orientation = FMT_SPEC[fmt]
            for _ in range(n):
                k += 1
                im = Image.new("RGB", size, (k % 256, (k * 7) % 256, (k * 13) % 256 or 1))
                kwargs = {}
                if orientation:
                    exif = Image.Exif()
                    exif[0x0112] = orientation
                    kwargs["exif"] = exif
                im.save(folder / f"img_{k:04d}.jpg", **kwargs)
    return root


def fmt_counts(plan, split):
    out = {}
    for r in plan.splits[split]:
        out.setdefault(r["label"], {}).setdefault(r["capture_format"], 0)
        out[r["label"]][r["capture_format"]] += 1
    return out


def _normalise(fmt: str) -> str:
    """Map the tiny stand-in sizes back to the real format names for readable assertions."""
    for real, (size, o) in FMT_SPEC.items():
        if fmt == f"{size[0]}x{size[1]}|exif_orientation={o if o else 'none'}":
            return real
    return fmt


def test_capture_format_read_from_header_not_filename(tmp_path):
    a = tmp_path / "office_car_whatever.jpg"
    exif = Image.Exif()
    exif[0x0112] = 8
    Image.new("RGB", (30, 20)).save(a, exif=exif)
    b = tmp_path / "b.jpg"
    Image.new("RGB", (30, 20)).save(b)
    assert read_capture_format(a) == "30x20|exif_orientation=8"
    assert read_capture_format(b) == "30x20|exif_orientation=none"


def test_default_method_is_class_and_capture_format():
    assert DEFAULT_SPLIT_METHOD == "class_and_capture_format"


def test_real_shaped_split_counts_and_format_balance(tmp_path):
    plan = prepare_hand_dataset(make_format_dataset(tmp_path), seed=42)
    per = {s: {c: sum(1 for r in plan.splits[s] if r["label"] == c) for c in CLASS_NAMES} for s in SPLITS}
    assert per["train"] == {"both_hands_on_steering": 21, "one_hand_on_steering": 68, "no_hands": 42}
    assert per["val"] == {"both_hands_on_steering": 5, "one_hand_on_steering": 15, "no_hands": 10}
    assert per["test"] == {"both_hands_on_steering": 4, "one_hand_on_steering": 14, "no_hands": 9}
    assert sum(len(plan.splits[s]) for s in SPLITS) == 188
    # every capture format of every class is present in every split, roughly 70/15/15
    for s in SPLITS:
        counts = {c: {_normalise(f): n for f, n in v.items()} for c, v in fmt_counts(plan, s).items()}
        for cls, formats in REAL_FORMATS.items():
            for fmt, total in formats.items():
                n = counts[cls][fmt]
                expected = total * {"train": 0.70, "val": 0.15, "test": 0.15}[s]
                assert abs(n - expected) <= 1, (s, cls, fmt, n, expected)
    test = {c: {_normalise(f): n for f, n in v.items()} for c, v in fmt_counts(plan, "test").items()}
    assert test["one_hand_on_steering"][CAR_A] == 6 and test["no_hands"][CAR_B] == 3


def test_class_only_method_reproduces_original_split(tmp_path):
    """--split-method class must give exactly the old algorithm (same RNG keys, same membership)."""
    import random
    root = make_format_dataset(tmp_path)
    plan = prepare_hand_dataset(root, seed=42, split_method="class")
    for label in CLASS_NAMES:
        items = sorted(r["path"] for s in SPLITS for r in plan.splits[s] if r["label"] == label)
        random.Random(f"42:{label}").shuffle(items)
        sizes = split_sizes(len(items), plan.ratios)
        expected_test = set(items[sizes["train"] + sizes["val"]:])
        assert {r["path"] for r in plan.splits["test"] if r["label"] == label} == expected_test


def test_unknown_split_method_rejected(tmp_path):
    with pytest.raises(HandDatasetError, match="unknown split method"):
        prepare_hand_dataset(make_dataset(tmp_path), split_method="by_filename")


@pytest.mark.parametrize("method", ["class", "class_and_capture_format"])
def test_split_invariants_for_both_methods(tmp_path, method):
    root = make_format_dataset(tmp_path)
    shutil.copy(next((root / DRIVER / "no_hands").glob("*.jpg")), root / DRIVER / "no_hands" / "Copy of dup.jpg")
    a = prepare_hand_dataset(root, seed=42, split_method=method)
    b = prepare_hand_dataset(root, seed=42, split_method=method)
    assert a.splits == b.splits  # deterministic
    paths = [r["path"] for r in a.records]
    assert len(paths) == len(set(paths)) == 188  # every usable image exactly once, no overlap
    split_sets = [{r["path"] for r in a.splits[s]} for s in SPLITS]
    assert not (split_sets[0] & split_sets[1]) and not (split_sets[0] & split_sets[2]) and not (split_sets[1] & split_sets[2])
    assert all({r["label"] for r in a.splits[s]} == set(CLASS_NAMES) for s in SPLITS)
    assert [d["path"] for d in a.excluded_duplicates] == [f"{DRIVER}/no_hands/Copy of dup.jpg"]
    assert f"{DRIVER}/no_hands/Copy of dup.jpg" not in paths
    assert len({r["sha256"] for r in a.records}) == 188


def test_seed_changes_membership_but_not_stratum_counts(tmp_path):
    root = make_format_dataset(tmp_path)
    a, b = prepare_hand_dataset(root, seed=42), prepare_hand_dataset(root, seed=7)
    assert {s: fmt_counts(a, s) for s in SPLITS} == {s: fmt_counts(b, s) for s in SPLITS}
    assert {r["path"] for r in a.splits["test"]} != {r["path"] for r in b.splits["test"]}


def test_manifest_records_method_formats_limitations_and_change(tmp_path):
    root = make_format_dataset(tmp_path)
    out, report = tmp_path / "processed", tmp_path / "split.md"
    old = write_outputs(prepare_hand_dataset(root, seed=42, split_method="class"), out, report)
    assert "previous_split_comparison" not in old  # nothing to compare against on the first run
    m = write_outputs(prepare_hand_dataset(root, seed=42), out, report)
    assert m["split_method"] == "class_and_capture_format" and m["seed"] == 42
    assert m["total_source_images"] == 188 and m["total_usable_images"] == 188
    assert m["per_split_class_counts"]["test"] == {"both_hands_on_steering": 4, "one_hand_on_steering": 14, "no_hands": 9}
    assert set(m["capture_format_counts"]) == set(CLASS_NAMES)
    assert any("impossible" in x and "filenames" in x for x in m["limitations"])
    assert any("domain shift" in x for x in m["limitations"])
    comp = m["previous_split_comparison"]
    assert comp["images_compared"] == 188 and comp["images_moved"] > 0
    records = json.loads((out / "hand_test.json").read_text(encoding="utf-8"))
    assert all("capture_format" in r for r in records)
    text = report.read_text(encoding="utf-8")
    for needle in ["class_and_capture_format", "Capture formats per split", "Change from the previous split",
                   "## Limitations", "NOT a valid held-out evaluation"]:
        assert needle in text, needle


def test_compare_splits_counts_moves():
    old = {"train": [{"path": "a"}, {"path": "b"}], "val": [{"path": "c"}], "test": [{"path": "d"}]}
    new = {"train": [{"path": "a"}, {"path": "d"}], "val": [{"path": "c"}], "test": [{"path": "b"}]}
    c = compare_splits(old, new)
    assert c["images_moved"] == 2 and c["moves"] == {"test->train": 1, "train->test": 1}
    assert c["test_images_previously_in_train"] == 1
