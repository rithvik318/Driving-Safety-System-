"""Tests for the dataset audit. Every test builds its own miniature dataset in a temp dir."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
Image = pytest.importorskip("PIL.Image")
np = pytest.importorskip("numpy")

from app.dataset_audit import DatasetRootError, render_report, run_audit, validate_root, write_manifest  # noqa: E402
from app.dataset_audit.scan import classify_kind, discover_structure, scan_dataset  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_image(path: Path, size=(64, 48), color=(200, 30, 30), fmt=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format=fmt)
    return path


def make_video(path: Path, frames=10, fps=5.0, size=(32, 24)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*("MJPG" if path.suffix == ".avi" else "mp4v"))
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    assert writer.isOpened(), "OpenCV cannot write test video"
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), i * 20 % 255, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def mini_dataset(tmp_path) -> Path:
    """Known contents, so every count below is exact."""
    root = tmp_path / "dataset"
    d, f = root / "driver_cam", root / "front_cam"
    make_image(d / "2hands" / "a.jpg")
    make_image(d / "2hands" / "b.png", size=(48, 64), color=(0, 90, 0))
    make_image(d / "1hand" / "c.jpg", color=(10, 10, 200))
    shutil.copy(d / "2hands" / "a.jpg", d / "1hand" / "a_copy.jpg")  # cross-label duplicate
    make_image(d / "no_hands" / "d.webp", color=(5, 5, 5), fmt="WEBP")
    (d / "no_hands" / "broken.jpg").write_bytes(b"not really a jpeg")  # corrupt
    (d / "no_hands" / "empty.jpg").write_bytes(b"")  # empty
    make_video(d / "drowsiness" / "videos" / "v1.mp4", frames=10, fps=5.0)
    make_video(d / "drowsiness" / "videos" / "v2.avi", frames=20, fps=10.0)
    make_image(f / "roads" / "r1.jpg", size=(160, 90), color=(90, 90, 90))
    make_image(f / "pedestrian" / "p1.jpg", size=(160, 90), color=(1, 2, 3))
    (f / "pedestrian" / "notes.txt").write_text("hello")  # unsupported
    (f / "pedestrian" / "desktop.ini").write_text("[x]")  # ignored system file
    make_video(f / "vehicles" / "car.mp4", frames=15, fps=5.0)
    (f / "dogs").mkdir(parents=True)  # present but empty
    make_image(root / "stray.jpg", color=(3, 3, 3))  # outside any class folder
    make_image(root / "extra_cam" / "misc" / "x.jpg", color=(4, 4, 4))  # unexpected folder
    return root


# --- path handling ---------------------------------------------------------

@pytest.mark.parametrize("bad", [None, "", "   "])
def test_validate_root_rejects_missing_value(bad):
    with pytest.raises(DatasetRootError, match="--root"):
        validate_root(bad)


def test_validate_root_rejects_nonexistent_and_file(tmp_path):
    with pytest.raises(DatasetRootError, match="does not exist"):
        validate_root(tmp_path / "nope")
    file = tmp_path / "file.txt"
    file.write_text("x")
    with pytest.raises(DatasetRootError, match="not a directory"):
        validate_root(file)


def test_validate_root_strips_quotes(mini_dataset):
    assert validate_root(f'"{mini_dataset}"') == mini_dataset.resolve()


# --- discovery and classification -----------------------------------------

@pytest.mark.parametrize(
    "name,kind",
    [("a.JPG", "image"), ("a.jpeg", "image"), ("a.png", "image"), ("a.webp", "image"),
     ("v.MP4", "video"), ("v.mov", "video"), ("v.avi", "video"), ("v.mkv", "video"),
     ("p.heic", "unsupported"), ("n.txt", "unsupported"), ("desktop.ini", "ignored"), (".DS_Store", "ignored")],
)
def test_classify_kind(name, kind):
    assert classify_kind(Path(name)) == kind


def test_discover_structure(mini_dataset):
    s = discover_structure(mini_dataset)
    assert s["found"]["driver_cam"] == ["1hand", "2hands", "drowsiness", "no_hands"]
    assert s["found"]["front_cam"] == ["dogs", "pedestrian", "roads", "vehicles"]
    assert s["missing_expected_folders"] == []
    assert s["unexpected_folders"] == ["extra_cam/"]


def test_missing_expected_folder_reported(tmp_path):
    root = tmp_path / "ds"
    make_image(root / "driver_cam" / "2hands" / "a.jpg")
    s = discover_structure(root)
    assert "driver_cam/1hand/" in s["missing_expected_folders"]
    assert "front_cam/" in s["missing_expected_folders"]


def test_scan_labels_from_folders(mini_dataset):
    by_path = {e.rel_path: e for e in scan_dataset(mini_dataset)}
    v1 = by_path["driver_cam/drowsiness/videos/v1.mp4"]
    assert (v1.camera, v1.class_name, v1.subfolder, v1.kind) == ("driver_cam", "drowsiness", "videos", "video")
    stray = by_path["stray.jpg"]
    assert stray.camera is None and stray.class_name is None


# --- full audit: exact counts -----------------------------------------------

def test_exact_counts(mini_dataset):
    m = run_audit(mini_dataset)
    d, f = m["driver_camera"], m["front_camera"]

    assert (d["2hands"]["images"], d["2hands"]["readable_images"]) == (2, 2)
    assert d["1hand"]["images"] == 2
    assert (d["no_hands"]["files"], d["no_hands"]["images"], d["no_hands"]["readable_images"]) == (3, 3, 1)
    assert (d["no_hands"]["corrupt"], d["no_hands"]["empty"]) == (1, 1)
    assert (d["drowsiness"]["videos"], d["drowsiness"]["readable_videos"]) == (2, 2)

    assert f["roads"]["images"] == 1
    assert (f["pedestrian"]["files"], f["pedestrian"]["unsupported"], f["pedestrian"]["ignored_system_files"]) == (3, 1, 1)
    assert f["vehicles"]["videos"] == 1
    assert f["dogs"]["files"] == 0

    t = m["totals"]
    assert (t["files"], t["images"], t["videos"], t["unsupported"]) == (16, 11, 3, 1)
    assert m["images"]["total"] == 11 and m["images"]["readable"] == 9
    assert "extra_cam" in m["other_folders"]
    assert m["structure"]["files_outside_class_folders"] == ["stray.jpg"]


def test_image_and_video_metadata(mini_dataset):
    m = run_audit(mini_dataset)
    img = m["images"]
    assert img["min_dimensions"] == "48x64" or img["min_dimensions"] == "64x48"
    assert img["max_dimensions"] == "160x90"
    assert img["orientation"]["portrait"] == 1
    assert img["formats"]["WEBP"] == 1

    drowsy = m["videos"]["per_class"]["driver_cam/drowsiness"]
    assert drowsy["total"] == 2 and drowsy["readable"] == 2
    assert drowsy["duration_s"]["min"] == pytest.approx(2.0, abs=0.2)
    assert drowsy["total_duration_s"] == pytest.approx(4.0, abs=0.4)
    assert drowsy["common_resolutions"][0]["resolution"] == "32x24"


def test_corrupt_and_empty_files_listed(mini_dataset):
    m = run_audit(mini_dataset)
    assert [c["path"] for c in m["corrupt_files"]] == ["driver_cam/no_hands/broken.jpg"]
    assert m["empty_files"] == ["driver_cam/no_hands/empty.jpg"]


def test_duplicate_detection(mini_dataset):
    m = run_audit(mini_dataset)
    assert m["duplicate_file_count"] == 1
    (group,) = m["duplicate_groups"]
    assert group["paths"] == ["driver_cam/1hand/a_copy.jpg", "driver_cam/2hands/a.jpg"]
    assert group["cross_label"] is True
    assert (mini_dataset / "driver_cam" / "1hand" / "a_copy.jpg").exists()  # never deleted


def test_audit_is_read_only(mini_dataset):
    before = {p: p.stat().st_mtime_ns for p in mini_dataset.rglob("*") if p.is_file()}
    run_audit(mini_dataset)
    after = {p: p.stat().st_mtime_ns for p in mini_dataset.rglob("*") if p.is_file()}
    assert before == after


# --- manifest and report ------------------------------------------------------

def test_manifest_has_required_keys_and_relative_paths(mini_dataset, tmp_path):
    m = run_audit(mini_dataset)
    out = tmp_path / "manifest.json"
    write_manifest(m, out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    for key in ["dataset_root", "audit_timestamp", "driver_camera", "front_camera", "images",
                "videos", "corrupt_files", "duplicate_groups"]:
        assert key in loaded
    assert set(loaded["driver_camera"]) >= {"2hands", "1hand", "no_hands", "drowsiness"}
    assert set(loaded["front_camera"]) >= {"roads", "pedestrian", "dogs", "vehicles"}
    assert loaded["dataset_root"] == "dataset"
    assert str(mini_dataset) not in out.read_text(encoding="utf-8")  # no absolute paths


def test_report_sections(mini_dataset):
    text = render_report(run_audit(mini_dataset))
    for n in range(1, 11):
        assert f"## {n}." in text
    assert "different labels" in text
    assert "Readable files are not the same as training-ready data" in text
    assert "imbalanced" not in text.lower()


# --- CLI -------------------------------------------------------------------------

def _load_cli():
    spec = importlib.util.spec_from_file_location("audit_cli", PROJECT_ROOT / "scripts" / "audit_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_invalid_root_exits_cleanly(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("DATASET_ROOT", raising=False)
    monkeypatch.setattr("app.config.settings.load_dotenv", lambda *a, **k: False)
    code = _load_cli().main(["--root", str(tmp_path / "missing"), "--quiet"])
    assert code == 2
    assert "does not exist" in capsys.readouterr().err


def test_cli_no_root_explains_how(capsys, monkeypatch):
    monkeypatch.delenv("DATASET_ROOT", raising=False)
    monkeypatch.setattr("app.config.settings.load_dotenv", lambda *a, **k: False)
    assert _load_cli().main(["--quiet"]) == 2
    err = capsys.readouterr().err
    assert "--root" in err and "DATASET_ROOT" in err


def test_cli_writes_outputs(mini_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.load_dotenv", lambda *a, **k: False)
    manifest, report = tmp_path / "out" / "m.json", tmp_path / "out" / "r.md"
    code = _load_cli().main(["--root", str(mini_dataset), "--manifest", str(manifest), "--report", str(report), "--quiet"])
    assert code == 0
    assert json.loads(manifest.read_text(encoding="utf-8"))["totals"]["files"] == 16
    assert report.read_text(encoding="utf-8").startswith("# Dataset audit")
