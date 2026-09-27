"""Tests for the training-vs-video domain comparison. Tiny generated fixtures only."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.dataset_audit.domain_compare import (
    OBSERVATIONS_HEADING,
    REAL_VIDEO,
    TRAINING,
    DomainCompareError,
    aspect_label,
    existing_observations,
    extract_video_frames,
    find_video,
    kmeans_representatives,
    run_comparison,
    select_training_samples,
)

cv2 = pytest.importorskip("cv2")

DRIVER = "drivercamera-x/drivercamera"


def jpg(path: Path, size, color, orientation=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color)
    if orientation:
        exif = Image.Exif()
        exif[0x0112] = orientation
        im.save(path, format="JPEG", exif=exif)
    else:
        im.save(path, format="JPEG")


def write_video(path: Path, n=30, fps=10.0, size=(64, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), i * 8, np.uint8))  # brightness encodes the frame index
    w.release()


@pytest.fixture
def dataset(tmp_path) -> Path:
    """Each class has two visually distinct looks (dark / bright) plus near-identical repeats."""
    root = tmp_path / "Data"
    for ci, cls in enumerate(["both_hands_on_steering", "one_hand_on_steering", "no_hands"]):
        for i in range(4):
            jpg(root / DRIVER / cls / f"dark_{i}.jpg", (40, 60), (10 + i, 10, 10 + ci), orientation=8)
            jpg(root / DRIVER / cls / f"bright_{i}.jpg", (40, 60), (240 - i, 240, 240 - ci), orientation=8)
    shutil.copy(root / DRIVER / "no_hands" / "dark_0.jpg", root / DRIVER / "no_hands" / "Copy of dark_0.jpg")
    write_video(root / DRIVER / "drowsy_driver" / "video_20260926_220714.mp4")
    return root


def digest(root: Path) -> dict:
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


def test_kmeans_picks_one_per_distinct_group_and_is_deterministic():
    rng = np.random.default_rng(0)
    groups = [np.zeros(8), np.ones(8), np.full(8, 0.5)]
    vecs = np.array([g + rng.normal(0, 0.01, 8) for g in groups for _ in range(5)])
    reps = kmeans_representatives(vecs, 3, seed=1)
    assert sorted(i // 5 for i, _ in reps) == [0, 1, 2]
    assert [s for _, s in reps] == [5, 5, 5]
    assert reps == kmeans_representatives(vecs, 3, seed=1)


def test_kmeans_small_inputs():
    assert kmeans_representatives(np.zeros((2, 4)), 3) == [(0, 1), (1, 1)]
    assert kmeans_representatives(np.zeros((0, 4)), 3) == []


def test_training_selection_skips_duplicates_and_covers_both_looks(dataset):
    samples, info = select_training_samples(dataset, per_class=2)
    assert info["no_hands"]["duplicates_excluded"] == 1
    for cls in ["both_hands_on_steering", "one_hand_on_steering", "no_hands"]:
        names = sorted(Path(s.source).name.split("_")[0] for s in samples if s.label == cls)
        assert names == ["bright", "dark"]
    s = samples[0]
    assert s.source_type == TRAINING
    assert (s.stored_width, s.stored_height) == (40, 60)
    assert (s.display_width, s.display_height) == (60, 40)  # EXIF 8 applied
    assert "EXIF 8" in s.orientation


def test_video_frames_at_distinct_timestamps(dataset):
    video = find_video(dataset)
    frames, info = extract_video_frames(video, (0.2, 0.5, 0.8))
    idx = [i for _, _, i in frames]
    assert idx == sorted(set(idx)) and len(idx) == 3
    assert idx == [round(f * 29) for f in (0.2, 0.5, 0.8)]
    ts = [t for _, t, _ in frames]
    assert ts == sorted(ts) and ts[0] < ts[-1]
    assert info["frame_count"] == 30 and info["stored_width"] == 64
    means = [np.asarray(img).mean() for img, _, _ in frames]
    assert means == sorted(means)  # frames really come from different points in the video


def test_missing_video_raises(tmp_path):
    with pytest.raises(DomainCompareError, match="not found"):
        find_video(tmp_path)


def test_aspect_labels():
    assert aspect_label(1920, 1080) == "16:9 landscape"
    assert aspect_label(720, 1280) == "9:16 portrait"
    assert aspect_label(480, 848).endswith("portrait")


def test_run_writes_image_and_report_without_touching_sources(dataset, tmp_path):
    before = digest(dataset)
    out_img, out_md = tmp_path / "out" / "cmp.jpg", tmp_path / "out" / "cmp.md"
    result = run_comparison(dataset, find_video(dataset), out_img, out_md, per_class=2)
    assert digest(dataset) == before
    assert Image.open(out_img).size[0] > 0
    types = [s["source_type"] for s in result["samples"]]
    assert types.count(TRAINING) == 6 and types.count(REAL_VIDEO) == 3
    text = out_md.read_text()
    for needle in ["TRAINING_IMAGE", "REAL_VIDEO", "none (unlabelled)", "Timestamp", "Original (stored)",
                   "EXIF 8", OBSERVATIONS_HEADING]:
        assert needle in text
    assert existing_observations(out_md) is None  # placeholder is not treated as written notes


def test_observations_section_survives_regeneration(dataset, tmp_path):
    out_img, out_md = tmp_path / "cmp.jpg", tmp_path / "cmp.md"
    run_comparison(dataset, find_video(dataset), out_img, out_md, per_class=1)
    text = out_md.read_text()
    notes = f"{OBSERVATIONS_HEADING}\n\n- Camera is mounted higher in the video.\n"
    out_md.write_text(text[: text.find(OBSERVATIONS_HEADING)] + notes)
    run_comparison(dataset, find_video(dataset), out_img, out_md, per_class=1)
    assert "Camera is mounted higher in the video." in out_md.read_text()
