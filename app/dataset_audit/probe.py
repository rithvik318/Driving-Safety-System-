"""Read-only checks on individual files: decode images, open videos, hash content.

Files are opened read-only. Nothing is written, resized or re-encoded.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from app.dataset_audit.scan import label_of

EXIF_ORIENTATION_TAG = 0x0112


def probe_image(path: Path) -> dict:
    """Decode an image fully and return its metadata, or an error."""
    from PIL import Image

    result = {"readable": False, "error": None}
    try:
        with Image.open(path) as im:
            im.verify()  # structural check; must be the first call after open()
        with Image.open(path) as im:
            result.update(
                width=im.width,
                height=im.height,
                format=im.format,
                mode=im.mode,
                exif_orientation=im.getexif().get(EXIF_ORIENTATION_TAG),
            )
            im.load()  # full decode: catches truncated files verify() misses
        result["readable"] = True
    except Exception as exc:  # any decode failure means "unreadable"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def probe_video(path: Path) -> dict:
    """Open a video container, read its metadata and decode the first frame."""
    import cv2

    result = {"readable": False, "error": None}
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            result["error"] = "container could not be opened"
            return result
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ") or None
        rotation = None
        if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
            rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
        result.update(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            fps=round(fps, 3) if fps > 0 else None,
            frame_count=frames if frames > 0 else None,
            duration_s=round(frames / fps, 3) if fps > 0 and frames > 0 else None,
            codec=codec,
            rotation_meta=rotation,
        )
        ok, _ = cap.read()
        if not ok:
            result["error"] = "container opened but no frame could be decoded"
            return result
        result["readable"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cap.release()
    return result


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_exact_duplicates(root: Path, entries: list) -> list[dict]:
    """Group byte-identical files (SHA-256). Only files sharing a size are hashed."""
    by_size: dict[int, list] = defaultdict(list)
    for e in entries:
        if e.kind in ("image", "video", "unsupported") and e.size_bytes > 0:
            by_size[e.size_bytes].append(e)

    groups = []
    for size, same_size in by_size.items():
        if len(same_size) < 2:
            continue
        by_hash: dict[str, list] = defaultdict(list)
        for e in same_size:
            try:
                by_hash[sha256_file(root / e.rel_path)].append(e)
            except OSError:
                continue  # unreadable files are reported by the probe step
        for digest, members in by_hash.items():
            if len(members) > 1:
                labels = sorted({label_of(m.camera, m.class_name) for m in members})
                groups.append(
                    {
                        "sha256": digest,
                        "size_bytes": size,
                        "paths": sorted(m.rel_path for m in members),
                        "labels": labels,
                        "cross_label": len(labels) > 1,
                    }
                )
    return sorted(groups, key=lambda g: g["paths"][0])
