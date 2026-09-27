"""Contact-sheet preview of the hand-state classes, for visual label checking.

Read-only: source images are opened, never written. EXIF orientation is applied
to an in-memory copy for display only.

Selection is deterministic: within each class, readable images are sorted by
relative path (case-insensitive) and N are taken at evenly spaced positions from
first to last, so the sample spans the whole folder instead of its first N files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from app.dataset_audit.probe import sha256_file
from app.dataset_audit.scan import CLASS_ROLES, IMAGE_EXTENSIONS, KNOWN_LAYOUTS, discover_structure

EXIF_ORIENTATION_TAG = 0x0112
BG = (24, 26, 30)
TILE_BG = (44, 47, 54)
TEXT = (235, 235, 235)
MUTED = (160, 165, 175)
HEADER_COLORS = [(38, 92, 160), (150, 95, 30), (120, 45, 110), (40, 120, 90)]
ALERT = (200, 60, 60)


class PreviewError(Exception):
    """No usable hand-state folders were found. The message says what is missing."""


@dataclass
class HandClass:
    name: str
    camera_dir: str | None
    files: list[str] = field(default_factory=list)  # relative to dataset root, sorted
    unreadable: list[dict] = field(default_factory=list)
    missing: bool = False

    @property
    def readable(self) -> list[str]:
        bad = {u["source_path"] for u in self.unreadable}
        return [f for f in self.files if f not in bad]


def hand_class_names(layout: str) -> list[str]:
    return [c for c in KNOWN_LAYOUTS[layout]["driver_cam"] if CLASS_ROLES.get(c) == "hand_state"]


def discover_hand_classes(root: Path) -> tuple[str, list[HandClass]]:
    """Return (layout, classes in layout order). Raises PreviewError if no driver-camera folder exists."""
    structure = discover_structure(root)
    camera_dirs = structure["camera_dirs"].get("driver_cam", [])
    if not camera_dirs:
        raise PreviewError(
            "No driver-camera folder found (looked for folders named "
            "'drivercamera' or 'driver_cam' up to 4 levels below the root)."
        )
    layout = structure["layout"]
    classes = []
    for name in hand_class_names(layout):
        hc = HandClass(name=name, camera_dir=None)
        for cam_dir in camera_dirs:
            folder = root / cam_dir / name
            if folder.is_dir():
                hc.camera_dir = cam_dir
                hc.files += [
                    p.relative_to(root).as_posix()
                    for p in folder.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
                ]
        hc.missing = hc.camera_dir is None
        hc.files.sort(key=lambda s: (s.casefold(), s))
        classes.append(hc)
    if all(hc.missing for hc in classes):
        raise PreviewError(
            "Driver-camera folder found, but none of the hand-state folders exist: " + ", ".join(hand_class_names(layout))
        )
    return layout, classes


def select_samples(items: list, n: int) -> list[tuple[int, object]]:
    """Deterministic, evenly spaced (position, item) pairs including first and last. All items if len <= n."""
    total = len(items)
    if n <= 0 or total == 0:
        return []
    if total <= n:
        return list(enumerate(items))
    if n == 1:
        return [(0, items[0])]
    positions = sorted({round(i * (total - 1) / (n - 1)) for i in range(n)})
    return [(p, items[p]) for p in positions]


def check_readable(path: Path) -> str | None:
    """None if the image decodes structurally, else an error string."""
    try:
        with Image.open(path) as im:
            im.verify()
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def load_for_display(path: Path, max_size: tuple[int, int] | None = None) -> tuple[Image.Image, dict]:
    """Open an image, apply EXIF orientation to an in-memory copy, return (RGB image, metadata).

    The source file is only read. metadata: original (stored) width/height, EXIF orientation,
    and display width/height after orientation.
    """
    with Image.open(path) as im:
        original = im.size
        orientation = im.getexif().get(EXIF_ORIENTATION_TAG)
        if max_size and im.format == "JPEG":
            side = max(max_size)
            im.draft("RGB", (side, side))  # faster JPEG decode at reduced scale (>= requested)
        shown = ImageOps.exif_transpose(im).convert("RGB")
    if orientation in (5, 6, 7, 8):
        display = (original[1], original[0])
    else:
        display = original
    meta = {
        "original_width": original[0],
        "original_height": original[1],
        "exif_orientation": orientation,
        "display_width": display[0],
        "display_height": display[1],
    }
    return shown, meta


def _font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    while text and draw.textlength(text + "…", font=font) > max_width:
        text = text[:-1]
    return text + "…"


def find_duplicates(root: Path, rel_paths: list[str]) -> list[list[str]]:
    by_hash: dict[str, list[str]] = {}
    for rel in rel_paths:
        try:
            by_hash.setdefault(sha256_file(root / rel), []).append(rel)
        except OSError:
            continue
    return sorted(sorted(g) for g in by_hash.values() if len(g) > 1)


def render_contact_sheet(root: Path, classes: list[HandClass], selections: dict[str, list], duplicates_of: dict[str, int],
                         columns: int, tile_w: int, tile_h: int, title: str) -> tuple[Image.Image, list[dict]]:
    pad, caption_h, header_h, title_h = 12, 46, 56, 70
    f_title, f_head, f_cap, f_small = _font(30), _font(26), _font(17), _font(14)
    rows_per_class = {hc.name: max(1, -(-len(selections.get(hc.name, [])) // columns)) for hc in classes}
    width = pad + columns * (tile_w + pad)
    height = title_h + sum(header_h + rows_per_class[hc.name] * (tile_h + caption_h + pad) + pad for hc in classes)
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 18), _fit_text(draw, title, f_title, width - 2 * pad), font=f_title, fill=TEXT)

    samples: list[dict] = []
    y = title_h
    for ci, hc in enumerate(classes):
        color = ALERT if hc.missing else HEADER_COLORS[ci % len(HEADER_COLORS)]
        draw.rectangle([0, y, width, y + header_h - 8], fill=color)
        if hc.missing:
            head = f"{hc.name}  —  FOLDER MISSING"
        else:
            chosen = selections.get(hc.name, [])
            head = (f"{hc.name}  —  {len(hc.files)} images, {len(hc.readable)} readable, {len(chosen)} shown "
                    f"(evenly spaced over sorted filenames)")
        draw.text((pad, y + 10), _fit_text(draw, head, f_head, width - 2 * pad), font=f_head, fill=TEXT)
        y += header_h
        for k, (position, rel) in enumerate(selections.get(hc.name, [])):
            col, row = k % columns, k // columns
            x0 = pad + col * (tile_w + pad)
            y0 = y + row * (tile_h + caption_h + pad)
            draw.rectangle([x0, y0, x0 + tile_w, y0 + tile_h], fill=TILE_BG)
            record = {
                "sample_index": len(samples),
                "class_sample_index": k,
                "class": hc.name,
                "filename": rel.rsplit("/", 1)[-1],
                "source_path": rel,
                "position_in_class": position,
                "class_total_readable": len(hc.readable),
                "duplicate_group": duplicates_of.get(rel),
            }
            try:
                img, meta = load_for_display(root / rel, (tile_w, tile_h))
                img.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)  # keeps aspect ratio
                sheet.paste(img, (x0 + (tile_w - img.width) // 2, y0 + (tile_h - img.height) // 2))
                record.update(meta)
                dims = f"{meta['original_width']}x{meta['original_height']}"
                if meta["exif_orientation"] not in (None, 1):
                    dims += f"  (EXIF {meta['exif_orientation']}, shown rotated)"
            except Exception as exc:
                record.update(load_error=f"{type(exc).__name__}: {exc}")
                draw.text((x0 + 10, y0 + tile_h // 2), "UNREADABLE", font=f_head, fill=ALERT)
                dims = "unreadable"
            name = f"#{k + 1:02d}  {record['filename']}"
            if record["duplicate_group"] is not None:
                name += f"  [dup group {record['duplicate_group']}]"
            draw.text((x0, y0 + tile_h + 4), _fit_text(draw, name, f_cap, tile_w), font=f_cap, fill=TEXT)
            draw.text((x0, y0 + tile_h + 25), _fit_text(draw, dims, f_small, tile_w), font=f_small, fill=MUTED)
            samples.append(record)
        y += rows_per_class[hc.name] * (tile_h + caption_h + pad) + pad
    return sheet, samples


def run_preview(root: Path, samples_per_class: int = 12, columns: int = 6, tile_w: int = 420, tile_h: int = 640,
                out_image: Path | None = None, out_manifest: Path | None = None) -> dict:
    """Build the contact sheet + manifest. Returns the manifest dict. Never writes inside `root`."""
    layout, classes = discover_hand_classes(root)

    for hc in classes:
        for rel in hc.files:
            error = check_readable(root / rel)
            if error:
                hc.unreadable.append({"source_path": rel, "error": error.replace(str(root / rel), rel)})

    all_files = [f for hc in classes for f in hc.files]
    duplicate_groups = find_duplicates(root, all_files)
    duplicates_of = {path: i + 1 for i, group in enumerate(duplicate_groups) for path in group}

    selections = {hc.name: select_samples(hc.readable, samples_per_class) for hc in classes if not hc.missing}
    title = (f"Hand-state label preview · {root.name} · layout {layout} · "
             f"{datetime.now().strftime('%Y-%m-%d %H:%M')} · EXIF orientation applied for display only")
    sheet, samples = render_contact_sheet(root, classes, selections, duplicates_of, columns, tile_w, tile_h, title)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_root": root.name,
        "path_note": "source_path is relative to the dataset root; only the root folder name is stored.",
        "layout": layout,
        "selection": {
            "method": "evenly spaced positions over readable images sorted by relative path (case-insensitive)",
            "samples_per_class": samples_per_class,
        },
        "classes": {
            hc.name: {
                "camera_dir": hc.camera_dir,
                "missing": hc.missing,
                "total_images": len(hc.files),
                "readable_images": len(hc.readable),
                "selected": len(selections.get(hc.name, [])),
                "unreadable": hc.unreadable,
            }
            for hc in classes
        },
        "missing_classes": [hc.name for hc in classes if hc.missing],
        "duplicate_groups": [{"group": i + 1, "paths": g} for i, g in enumerate(duplicate_groups)],
        "samples": samples,
        "output_image": out_image.name if out_image else None,
    }

    if out_image:
        out_image.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out_image, format="JPEG", quality=90)
    if out_manifest:
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
