"""Tests for the collected-dataset layout and the hand-state contact-sheet preview.

All images are tiny software-test fixtures created in temp dirs; no real dataset is used.
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

from app.dataset_audit import render_report, run_audit
from app.dataset_audit.hand_preview import (
    PreviewError,
    discover_hand_classes,
    load_for_display,
    run_preview,
    select_samples,
)
from app.dataset_audit.scan import discover_structure, scan_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIVER = "drivercamera-20260926T193339Z-1-001/drivercamera"
FRONT = "frontcamera-20260926T193501Z-1-001/frontcamera"
COUNTS = {"both_hands_on_steering": 5, "one_hand_on_steering": 9, "no_hands": 4}


def jpg(path: Path, size=(60, 100), color=(10, 20, 30), orientation=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color)
    if orientation:
        exif = Image.Exif()
        exif[0x0112] = orientation
        im.save(path, format="JPEG", exif=exif)
    else:
        im.save(path, format="JPEG")
    return path


@pytest.fixture
def collected(tmp_path) -> Path:
    """Mirrors the real layout: wrapper folders, camera folders, real class names."""
    root = tmp_path / "Data"
    for ci, (cls, n) in enumerate(COUNTS.items()):
        for i in range(n):
            jpg(root / DRIVER / cls / f"frame_{i:04d}.jpg", color=(ci * 60, i * 20, 7))
    shutil.copy(root / DRIVER / "no_hands" / "frame_0001.jpg", root / DRIVER / "no_hands" / "Copy of frame_0001.jpg")
    (root / DRIVER / "drowsy_driver").mkdir(parents=True)
    (root / DRIVER / "drowsy_driver" / "clip.mp4").write_bytes(b"\x00" * 10)
    for cls in ["dogs_on_road", "pedestrains", "roads", "vehicles"]:
        jpg(root / FRONT / cls / "a.jpg", size=(100, 60), color=(len(cls) * 9, 1, 1))
    return root


def digest(root: Path) -> dict:
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


# --- Task 1: audit understands the collected layout ------------------------------------------

def test_structure_detects_collected_layout(collected):
    s = discover_structure(collected)
    assert s["layout"] == "collected_2026_09"
    assert s["camera_dirs"] == {"driver_cam": [DRIVER], "front_cam": [FRONT]}
    assert s["found"]["driver_cam"] == sorted(["both_hands_on_steering", "one_hand_on_steering", "no_hands", "drowsy_driver"])
    assert s["missing_expected_folders"] == [] and s["unexpected_folders"] == []


def test_scan_assigns_real_classes_through_wrapper_folders(collected):
    e = {x.rel_path: x for x in scan_dataset(collected)}[f"{DRIVER}/one_hand_on_steering/frame_0003.jpg"]
    assert (e.camera, e.class_name, e.subfolder, e.camera_dir) == ("driver_cam", "one_hand_on_steering", None, DRIVER)


def test_audit_counts_real_classes_and_no_false_warnings(collected):
    m = run_audit(collected)
    d, f = m["driver_camera"], m["front_camera"]
    assert list(d) == ["both_hands_on_steering", "one_hand_on_steering", "no_hands", "drowsy_driver"]
    assert [d[c]["images"] for c in COUNTS] == [5, 9, 5]  # no_hands includes the copy
    assert d["drowsy_driver"]["videos"] == 1
    assert list(f) == ["dogs_on_road", "pedestrains", "roads", "vehicles"]
    assert all(f[c]["images"] == 1 for c in f)
    assert m["other_folders"] == {}
    report = render_report(m)
    assert "not in expected structure" not in report
    assert "Expected folders missing" not in report
    assert "collected_2026_09" in report
    assert "frame-style names" not in report  # no grouping is inferred from filenames


def test_original_layout_still_recognised(tmp_path):
    root = tmp_path / "ds"
    for cls in ["2hands", "1hand", "no_hands", "drowsiness"]:
        jpg(root / "driver_cam" / cls / "a.jpg")
    s = discover_structure(root)
    assert s["layout"] == "original_plan" and s["missing_expected_folders"] == ["front_cam/"]


# --- class discovery -----------------------------------------------------------------------------

def test_discover_hand_classes_in_order_excludes_drowsy(collected):
    layout, classes = discover_hand_classes(collected)
    assert layout == "collected_2026_09"
    assert [c.name for c in classes] == ["both_hands_on_steering", "one_hand_on_steering", "no_hands"]
    assert [len(c.files) for c in classes] == [5, 9, 5]
    assert all(f.endswith(".jpg") for c in classes for f in c.files)


def test_discover_hand_classes_legacy_names(tmp_path):
    root = tmp_path / "ds"
    for cls in ["2hands", "1hand", "no_hands"]:
        jpg(root / "driver_cam" / cls / "a.jpg")
    _, classes = discover_hand_classes(root)
    assert [c.name for c in classes] == ["2hands", "1hand", "no_hands"]


# --- missing folders ---------------------------------------------------------------------------------

def test_missing_hand_folder_is_reported_not_fatal(collected, tmp_path):
    shutil.rmtree(collected / DRIVER / "one_hand_on_steering")
    m = run_preview(collected, 3, out_image=tmp_path / "p.jpg", out_manifest=tmp_path / "m.json")
    assert m["missing_classes"] == ["one_hand_on_steering"]
    assert m["classes"]["one_hand_on_steering"]["selected"] == 0
    assert (tmp_path / "p.jpg").is_file()


def test_no_driver_camera_folder_raises(tmp_path):
    jpg(tmp_path / "Data" / "somewhere" / "x.jpg")
    with pytest.raises(PreviewError, match="No driver-camera folder"):
        discover_hand_classes(tmp_path / "Data")


def test_driver_folder_without_hand_classes_raises(tmp_path):
    (tmp_path / "Data" / "drivercamera" / "drowsy_driver").mkdir(parents=True)
    with pytest.raises(PreviewError, match="none of the hand-state folders"):
        discover_hand_classes(tmp_path / "Data")


# --- deterministic selection ----------------------------------------------------------------------------

def test_selection_is_deterministic_and_spans_the_folder():
    items = [f"f{i:03d}" for i in range(97)]
    a, b = select_samples(items, 12), select_samples(items, 12)
    assert a == b and len(a) == 12
    assert a[0] == (0, "f000") and a[-1] == (96, "f096")
    positions = [p for p, _ in a]
    assert positions == sorted(set(positions))


@pytest.mark.parametrize("n,total,expected", [(12, 5, 5), (1, 10, 1), (0, 10, 0), (12, 0, 0), (12, 12, 12)])
def test_selection_edge_cases(n, total, expected):
    assert len(select_samples(list(range(total)), n)) == expected


def test_preview_selection_repeatable_across_runs(collected, tmp_path):
    first = run_preview(collected, 3, out_manifest=tmp_path / "a.json")
    second = run_preview(collected, 3, out_manifest=tmp_path / "b.json")
    assert [s["source_path"] for s in first["samples"]] == [s["source_path"] for s in second["samples"]]


# --- EXIF-safe loading --------------------------------------------------------------------------------------

def test_exif_orientation_applied_for_display_only(tmp_path):
    src = jpg(tmp_path / "rot.jpg", size=(160, 90), orientation=6)  # stored landscape, displays portrait
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    img, meta = load_for_display(src)
    assert img.size == (90, 160)
    assert (meta["original_width"], meta["original_height"]) == (160, 90)
    assert (meta["display_width"], meta["display_height"]) == (90, 160)
    assert meta["exif_orientation"] == 6
    assert hashlib.sha256(src.read_bytes()).hexdigest() == before


def test_image_without_exif_unchanged(tmp_path):
    img, meta = load_for_display(jpg(tmp_path / "plain.jpg", size=(60, 100)))
    assert img.size == (60, 100) and meta["exif_orientation"] is None


# --- manifest + outputs ------------------------------------------------------------------------------------

def test_manifest_contents_and_dataset_untouched(collected, tmp_path):
    jpg(collected / DRIVER / "both_hands_on_steering" / "frame_0002.jpg", size=(160, 90), orientation=6)
    (collected / DRIVER / "no_hands" / "broken.jpg").write_bytes(b"not an image")
    before = digest(collected)

    out_img, out_json = tmp_path / "plots" / "prev.jpg", tmp_path / "reports" / "m.json"
    m = run_preview(collected, 4, columns=2, tile_w=120, tile_h=180, out_image=out_img, out_manifest=out_json)

    assert digest(collected) == before  # nothing in the dataset changed, nothing added
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["dataset_root"] == "Data" and str(collected) not in out_json.read_text(encoding="utf-8")
    c = loaded["classes"]
    assert (c["both_hands_on_steering"]["total_images"], c["both_hands_on_steering"]["selected"]) == (5, 4)
    assert (c["no_hands"]["total_images"], c["no_hands"]["readable_images"]) == (6, 5)
    assert [u["source_path"].rsplit("/", 1)[-1] for u in c["no_hands"]["unreadable"]] == ["broken.jpg"]
    assert loaded["duplicate_groups"][0]["paths"][0].endswith("Copy of frame_0001.jpg")

    sample = loaded["samples"][0]
    for key in ["source_path", "class", "filename", "original_width", "original_height", "sample_index",
                "class_sample_index", "position_in_class", "exif_orientation"]:
        assert key in sample
    assert [s["sample_index"] for s in loaded["samples"]] == list(range(len(loaded["samples"])))
    # 5 files, 4 samples -> positions 0, 1, 3, 4: first and last always included
    assert [s["position_in_class"] for s in loaded["samples"] if s["class"] == "both_hands_on_steering"] == [0, 1, 3, 4]

    with Image.open(out_img) as sheet:
        assert sheet.format == "JPEG"
        assert sheet.width == 12 + 2 * (120 + 12)  # pad + columns * (tile + pad)


def test_rotated_sample_recorded_with_both_dimensions(collected, tmp_path):
    jpg(collected / DRIVER / "both_hands_on_steering" / "frame_0000.jpg", size=(160, 90), orientation=6)
    m = run_preview(collected, 12)
    s = next(x for x in m["samples"] if x["source_path"].endswith("both_hands_on_steering/frame_0000.jpg"))
    assert (s["original_width"], s["original_height"], s["display_width"], s["display_height"]) == (160, 90, 90, 160)


# --- CLI ----------------------------------------------------------------------------------------------------

def _cli():
    spec = importlib.util.spec_from_file_location("preview_cli", PROJECT_ROOT / "scripts" / "preview_hand_dataset.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def no_dotenv(monkeypatch):
    monkeypatch.setattr("app.config.settings.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("DATASET_ROOT", raising=False)


def test_cli_writes_outputs(no_dotenv, collected, tmp_path, capsys):
    out_img, out_json = tmp_path / "p.jpg", tmp_path / "m.json"
    code = _cli().main(["--root", str(collected), "--samples-per-class", "3", "--output", str(out_img), "--manifest", str(out_json)])
    out = capsys.readouterr().out
    assert code == 0 and out_img.is_file() and out_json.is_file()
    assert "one_hand_on_steering" in out and "Exact-duplicate groups: 1" in out


def test_cli_bad_root_and_options(no_dotenv, tmp_path, capsys):
    assert _cli().main(["--root", str(tmp_path / "nope")]) == 2
    assert "does not exist" in capsys.readouterr().err
    assert _cli().main(["--root", str(tmp_path), "--samples-per-class", "0"]) == 2
    assert _cli().main(["--root", str(tmp_path)]) == 2  # exists but has no driver camera folder
    assert "No driver-camera folder" in capsys.readouterr().err
