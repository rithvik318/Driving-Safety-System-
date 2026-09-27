"""Walk an external dataset root and classify every file.

The folder name is treated as the label. Nothing is renamed, moved or fixed
here: unexpected folders and files are reported, not corrected.

Camera folders are recognised by name at any depth, so wrapper folders (for
example Google Drive export folders like `drivercamera-20260926T193339Z-1-001/`)
are handled. Two folder layouts are known:

    original_plan:     driver_cam/{2hands,1hand,no_hands,drowsiness}
                       front_cam/{roads,pedestrian,dogs,vehicles}
    collected_2026_09: .../drivercamera/{both_hands_on_steering,one_hand_on_steering,no_hands,drowsy_driver}
                       .../frontcamera/{dogs_on_road,pedestrains,roads,vehicles}

("pedestrains" is the folder's real spelling and is kept as-is.)
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

# Canonical camera key -> folder names that mean that camera (matched case-insensitively).
CAMERA_ALIASES: dict[str, tuple[str, ...]] = {
    "driver_cam": ("driver_cam", "drivercamera"),
    "front_cam": ("front_cam", "frontcamera"),
}

KNOWN_LAYOUTS: dict[str, dict[str, list[str]]] = {
    "original_plan": {
        "driver_cam": ["2hands", "1hand", "no_hands", "drowsiness"],
        "front_cam": ["roads", "pedestrian", "dogs", "vehicles"],
    },
    "collected_2026_09": {
        "driver_cam": ["both_hands_on_steering", "one_hand_on_steering", "no_hands", "drowsy_driver"],
        "front_cam": ["dogs_on_road", "pedestrains", "roads", "vehicles"],
    },
}
DEFAULT_LAYOUT = "original_plan"
EXPECTED_STRUCTURE = KNOWN_LAYOUTS[DEFAULT_LAYOUT]  # kept for backwards compatibility

# What each known class folder is for (driver camera).
CLASS_ROLES: dict[str, str] = {
    "2hands": "hand_state",
    "1hand": "hand_state",
    "no_hands": "hand_state",
    "both_hands_on_steering": "hand_state",
    "one_hand_on_steering": "hand_state",
    "drowsiness": "drowsiness",
    "drowsy_driver": "drowsiness",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
# OS / sync-client files that are not data. Reported as "ignored", never deleted.
SYSTEM_FILENAMES = {"desktop.ini", "thumbs.db", ".ds_store"}
MAX_CAMERA_SEARCH_DEPTH = 4  # how deep below the root camera folders are looked for


class DatasetRootError(Exception):
    """The dataset root is missing, not a directory, or not readable."""


@dataclass
class FileEntry:
    rel_path: str  # POSIX-style, relative to the dataset root
    camera: str | None  # canonical camera (driver_cam / front_cam) or an unrecognised top-level folder
    class_name: str | None  # folder directly inside the camera folder; None if the file is not inside one
    subfolder: str | None  # anything between the class folder and the file (e.g. "videos")
    kind: str  # image | video | unsupported | ignored
    extension: str
    size_bytes: int
    camera_dir: str | None = None  # relative path of the camera folder on disk

    def to_dict(self) -> dict:
        return asdict(self)


def camera_key(folder_name: str) -> str | None:
    """Canonical camera key for a folder name, or None if it is not a camera folder."""
    name = folder_name.lower()
    for key, aliases in CAMERA_ALIASES.items():
        if name in aliases:
            return key
    return None


def label_of(camera: str | None, class_name: str | None) -> str:
    """Human-readable label for a file's location, e.g. 'driver_cam/no_hands'."""
    if camera is None:
        return "(dataset root)"
    return f"{camera}/{class_name}" if class_name else f"{camera}/(no class folder)"


def validate_root(root: str | Path | None) -> Path:
    """Return the resolved root or raise DatasetRootError with a helpful message."""
    how = (
        "Provide it with --root \"PATH_TO_DATASET\" or set DATASET_ROOT in your .env file "
        "(see .env.example)."
    )
    if root is None or str(root).strip() == "":
        raise DatasetRootError(f"No dataset root given. {how}")
    path = Path(str(root).strip().strip('"').strip("'")).expanduser()
    if not path.exists():
        raise DatasetRootError(f"Dataset root does not exist: {path}\n{how}")
    if not path.is_dir():
        raise DatasetRootError(f"Dataset root is not a directory: {path}\n{how}")
    try:
        next(path.iterdir(), None)
    except OSError as exc:
        raise DatasetRootError(f"Dataset root cannot be read: {path} ({exc})") from exc
    return path.resolve()


def classify_kind(path: Path) -> str:
    name = path.name.lower()
    if name in SYSTEM_FILENAMES or name.startswith("."):
        return "ignored"
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return "unsupported"


def locate(parts: tuple[str, ...]) -> tuple[str | None, str | None, str | None, str | None]:
    """(camera, class_name, subfolder, camera_dir) for a relative file path split into parts."""
    folders = parts[:-1]
    for k, folder in enumerate(folders[: MAX_CAMERA_SEARCH_DEPTH]):
        key = camera_key(folder)
        if key:
            class_name = parts[k + 1] if len(parts) > k + 2 else None
            subfolder = "/".join(parts[k + 2 : -1]) or None
            return key, class_name, subfolder, "/".join(parts[: k + 1])
    # No camera folder: fall back to top-level folder / second-level folder.
    camera = parts[0] if len(parts) >= 2 else None
    class_name = parts[1] if len(parts) >= 3 else None
    subfolder = "/".join(parts[2:-1]) if len(parts) >= 4 else None
    return camera, class_name, subfolder, camera


def scan_dataset(root: Path) -> list[FileEntry]:
    """Recursively list every file under root (sorted, deterministic)."""
    entries: list[FileEntry] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        camera, class_name, subfolder, camera_dir = locate(rel.parts)
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        entries.append(
            FileEntry(
                rel_path=rel.as_posix(),
                camera=camera,
                class_name=class_name,
                subfolder=subfolder,
                kind=classify_kind(path),
                extension=path.suffix.lower(),
                size_bytes=size,
                camera_dir=camera_dir,
            )
        )
    return entries


def _visible_dirs(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_dir() and not p.name.startswith("."))


def find_camera_dirs(root: Path) -> dict[str, list[str]]:
    """Canonical camera key -> relative paths of matching folders (searched to MAX_CAMERA_SEARCH_DEPTH)."""
    found: dict[str, list[str]] = {}
    for dirpath, dirnames, _ in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        if depth >= MAX_CAMERA_SEARCH_DEPTH:
            dirnames[:] = []
            continue
        for d in list(dirnames):
            key = camera_key(d)
            if key:
                found.setdefault(key, []).append((rel / d).as_posix())
                dirnames.remove(d)  # classes live inside; don't search for cameras within a camera
    return found


def detect_layout(found: dict[str, list[str]]) -> str:
    """Known layout whose class names best match the folders found (ties -> DEFAULT_LAYOUT)."""
    best, best_score = DEFAULT_LAYOUT, 0
    for name, layout in KNOWN_LAYOUTS.items():
        score = sum(len(set(found.get(cam, [])) & set(classes)) for cam, classes in layout.items())
        if score > best_score:
            best, best_score = name, score
    return best


def discover_structure(root: Path) -> dict:
    """Find camera folders and their class folders; compare with the best-matching known layout."""
    camera_dirs = find_camera_dirs(root)
    found: dict[str, list[str]] = {}
    for key, dirs in camera_dirs.items():
        classes: set[str] = set()
        for d in dirs:
            classes.update(c.name for c in _visible_dirs(root / d))
        found[key] = sorted(classes)

    # Top-level folders that neither are nor contain a camera folder are reported as-is.
    containers = {d.split("/")[0] for dirs in camera_dirs.values() for d in dirs}
    for top in _visible_dirs(root):
        if top.name not in containers and camera_key(top.name) is None:
            found[top.name] = [c.name for c in _visible_dirs(top)]

    layout_name = detect_layout(found)
    expected = KNOWN_LAYOUTS[layout_name]

    missing = []
    for camera, classes in expected.items():
        if camera not in found:
            missing.append(camera + "/")
            continue
        missing.extend(f"{camera}/{c}/" for c in classes if c not in found[camera])

    unexpected = []
    for camera, classes in found.items():
        if camera not in expected:
            unexpected.append(camera + "/")
            continue
        unexpected.extend(f"{camera}/{c}/" for c in classes if c not in expected[camera])

    return {
        "layout": layout_name,
        "expected": expected,
        "camera_dirs": camera_dirs,
        "found": found,
        "missing_expected_folders": missing,
        "unexpected_folders": unexpected,
    }
