"""Run the full audit and build the machine-readable manifest."""

from __future__ import annotations

import logging
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Callable

from app.dataset_audit.probe import find_exact_duplicates, probe_image, probe_video
from app.dataset_audit.scan import CAMERA_ALIASES, discover_structure, label_of, scan_dataset

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = 2  # 2: camera folders found at any depth; `layout`, `camera_dirs` added


def _tool_versions() -> dict:
    versions = {"python": platform.python_version()}
    try:
        import PIL

        versions["pillow"] = PIL.__version__
    except ImportError:
        versions["pillow"] = None
    try:
        import cv2

        versions["opencv"] = cv2.__version__
    except ImportError:
        versions["opencv"] = None
    return versions


def _num_stats(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"min": min(values), "max": max(values), "mean": round(mean(values), 3)}


def _top(counter: Counter, n: int = 5, key: str = "value") -> list[dict]:
    return [{key: k, "count": c} for k, c in counter.most_common(n)]


def _aspect_label(w: int, h: int) -> str:
    ratio = w / h
    named = {"1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4, "16:9": 16 / 9, "9:16": 9 / 16, "3:2": 1.5, "2:3": 2 / 3}
    for label, value in named.items():
        if abs(ratio - value) < 0.02:
            return label
    return f"other ({ratio:.2f})"


def _orientation(w: int, h: int) -> str:
    return "square" if w == h else ("landscape" if w > h else "portrait")


def _folder_counts(records: list[dict]) -> dict:
    kinds = Counter(r["kind"] for r in records)
    return {
        "files": len(records),
        "images": kinds["image"],
        "videos": kinds["video"],
        "unsupported": kinds["unsupported"],
        "ignored_system_files": kinds["ignored"],
        "empty": sum(1 for r in records if r["size_bytes"] == 0),
        "corrupt": sum(1 for r in records if r.get("readable") is False and r["size_bytes"] != 0),
        "readable_images": sum(1 for r in records if r["kind"] == "image" and r.get("readable")),
        "readable_videos": sum(1 for r in records if r["kind"] == "video" and r.get("readable")),
    }


def _image_stats(images: list[dict]) -> dict:
    ok = [r for r in images if r.get("readable")]
    dims = [(r["width"], r["height"]) for r in ok]
    stats = {
        "total": len(images),
        "readable": len(ok),
        "formats": dict(Counter(r.get("format") for r in ok)),
        "extensions": dict(Counter(r["extension"] for r in images)),
    }
    if dims:
        by_area = sorted(dims, key=lambda d: d[0] * d[1])
        stats.update(
            min_dimensions=f"{by_area[0][0]}x{by_area[0][1]}",
            max_dimensions=f"{by_area[-1][0]}x{by_area[-1][1]}",
            width=_num_stats([w for w, _ in dims]),
            height=_num_stats([h for _, h in dims]),
            distinct_dimensions=len(set(dims)),
            common_dimensions=_top(Counter(f"{w}x{h}" for w, h in dims), key="dimensions"),
            aspect_ratios=dict(Counter(_aspect_label(w, h) for w, h in dims)),
            orientation=dict(Counter(_orientation(w, h) for w, h in dims)),
            exif_rotated=sum(1 for r in ok if r.get("exif_orientation") not in (None, 1)),
            file_size_bytes=_num_stats([r["size_bytes"] for r in ok]),
        )
    return stats


def _video_stats(videos: list[dict]) -> dict:
    ok = [r for r in videos if r.get("readable")]
    durations = [r["duration_s"] for r in ok if r.get("duration_s")]
    stats = {"total": len(videos), "readable": len(ok)}
    if ok:
        stats.update(
            total_duration_s=round(sum(durations), 3) if durations else None,
            duration_s=_num_stats(durations),
            videos_without_duration=len(ok) - len(durations),
            common_resolutions=_top(Counter(f"{r['width']}x{r['height']}" for r in ok), key="resolution"),
            common_fps=_top(Counter(r.get("fps") for r in ok), key="fps"),
            codecs=dict(Counter(r.get("codec") for r in ok)),
            rotation_meta=dict(Counter(str(r.get("rotation_meta")) for r in ok)),
            frame_count=_num_stats([r["frame_count"] for r in ok if r.get("frame_count")]),
            file_size_bytes=_num_stats([r["size_bytes"] for r in ok]),
        )
    return stats


def run_audit(root: Path, progress: Callable[[int, int, str], None] | None = None) -> dict:
    """Audit `root` (already validated) and return the manifest dict. Read-only."""
    structure = discover_structure(root)
    entries = scan_dataset(root)
    logger.info("Found %d files under %s", len(entries), root.name)

    records: list[dict] = []
    for i, entry in enumerate(entries, 1):
        record = entry.to_dict()
        path = root / entry.rel_path
        if entry.size_bytes == 0:
            record.update(readable=False, error="empty file")
        elif entry.kind == "image":
            record.update(probe_image(path))
        elif entry.kind == "video":
            record.update(probe_video(path))
        if record.get("error"):  # decoder messages can embed the absolute path
            record["error"] = record["error"].replace(str(path), entry.rel_path)
        records.append(record)
        if progress:
            progress(i, len(entries), entry.rel_path)

    duplicates = find_exact_duplicates(root, entries)

    # Per-folder counts, keyed exactly as the folders are named on disk.
    camera_counts: dict[str, dict[str, dict]] = {}
    for camera, classes in structure["found"].items():
        ordered = [c for c in structure["expected"].get(camera, []) if c in classes]
        ordered += [c for c in classes if c not in ordered]
        camera_counts[camera] = {
            cls: _folder_counts([r for r in records if r["camera"] == camera and r["class_name"] == cls])
            for cls in ordered
        }
    for camera, classes in structure["expected"].items():  # missing folders appear as zeros
        for cls in classes:
            camera_counts.setdefault(camera, {}).setdefault(cls, _folder_counts([]))

    images = [r for r in records if r["kind"] == "image"]
    videos = [r for r in records if r["kind"] == "video"]
    video_classes = sorted({(r["camera"], r["class_name"]) for r in videos}, key=str)

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_root": root.name,
        "path_note": "Only the dataset folder name is stored. All paths are relative to DATASET_ROOT.",
        "audit_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": "real_local",
        "tool_versions": _tool_versions(),
        "structure": {
            **structure,
            "files_outside_class_folders": [r["rel_path"] for r in records if r["class_name"] is None and r["kind"] != "ignored"],
            "nested_subfolders": sorted({f"{r['camera']}/{r['class_name']}/{r['subfolder']}" for r in records if r["subfolder"]}),
        },
        "totals": _folder_counts(records),
        "driver_camera": camera_counts.get("driver_cam", {}),
        "front_camera": camera_counts.get("front_cam", {}),
        "other_folders": {k: v for k, v in camera_counts.items() if k not in CAMERA_ALIASES},
        "images": {
            **_image_stats(images),
            "per_class": {
                label_of(cam, cls): _image_stats([r for r in images if r["camera"] == cam and r["class_name"] == cls])
                for cam, cls in sorted({(r["camera"], r["class_name"]) for r in images}, key=str)
            },
        },
        "videos": {
            **_video_stats(videos),
            "per_class": {
                label_of(cam, cls): _video_stats([r for r in videos if r["camera"] == cam and r["class_name"] == cls])
                for cam, cls in video_classes
            },
        },
        "unsupported_files": [{"path": r["rel_path"], "extension": r["extension"]} for r in records if r["kind"] == "unsupported"],
        "ignored_system_files": [r["rel_path"] for r in records if r["kind"] == "ignored"],
        "empty_files": [r["rel_path"] for r in records if r["size_bytes"] == 0],
        "corrupt_files": [
            {"path": r["rel_path"], "kind": r["kind"], "error": r.get("error")}
            for r in records
            if r.get("readable") is False and r["size_bytes"] != 0
        ],
        "duplicate_groups": duplicates,
        "duplicate_file_count": sum(len(g["paths"]) - 1 for g in duplicates),
        "files": records,
    }
