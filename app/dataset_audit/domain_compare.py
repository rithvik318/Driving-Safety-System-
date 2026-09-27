"""Visual domain comparison: hand-state training images vs. frames of a real driver video.

Read-only. Source images and the video are opened, never written. EXIF orientation is
applied to in-memory copies (via hand_preview.load_for_display); video frames use
OpenCV's automatic orientation handling.

Training samples per class are chosen by content, not by filename:
  1. exact byte duplicates (SHA-256) are dropped, keeping the first path in sort order;
  2. every remaining image becomes a small grayscale thumbnail;
  3. seeded k-means (k = samples per class) groups the thumbnails;
  4. from each cluster, the image closest to the cluster centre is taken.
So each sample stands for a group of similar-looking images and no two come from the
same group. Video frames are taken at evenly spaced fractions of the video duration.

Measured per sample: stored and displayed resolution, EXIF orientation / rotation
metadata, aspect ratio, mean and standard deviation of luma (0-255). These are the
only automatic "facts"; visual observations are written by a person (section 3 of the
report is kept when the report is regenerated).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from app.dataset_audit.hand_preview import (
    BG,
    HEADER_COLORS,
    MUTED,
    TEXT,
    TILE_BG,
    _fit_text,
    _font,
    discover_hand_classes,
    find_duplicates,
    load_for_display,
)

TRAINING = "TRAINING_IMAGE"
REAL_VIDEO = "REAL_VIDEO"
DEFAULT_VIDEO_NAME = "video_20260926_220714.mp4"
DEFAULT_FRACTIONS = (0.2, 0.5, 0.8)
THUMB = (32, 32)
OBSERVATIONS_HEADING = "## 3. Visual observations"

CLASS_TO_STATE = {
    "both_hands_on_steering": "BOTH_HANDS",
    "one_hand_on_steering": "ONE_HAND",
    "no_hands": "NO_HANDS",
}


class DomainCompareError(Exception):
    """Missing video / dataset folder etc. The message says what is missing."""


@dataclass
class Sample:
    source_type: str  # TRAINING_IMAGE | REAL_VIDEO
    label: str | None  # dataset folder label, or None for the unlabelled video
    source: str  # path relative to the dataset root
    stored_width: int
    stored_height: int
    display_width: int
    display_height: int
    orientation: str  # e.g. "EXIF 8 (rotate 90 CW applied)" / "none" / "rotation meta 0"
    mean_luma: float
    std_luma: float
    timestamp_s: float | None = None
    frame_index: int | None = None
    cluster_size: int | None = None  # training: how many images this sample stands for
    class_total: int | None = None

    @property
    def aspect(self) -> str:
        return aspect_label(self.display_width, self.display_height)


# ---------------------------------------------------------------- measurements

def aspect_label(w: int, h: int) -> str:
    """'16:9 landscape', '9:16 portrait', or a ratio like '1.77:1'."""
    if w <= 0 or h <= 0:
        return "n/a"
    shape = "landscape" if w > h else "portrait" if h > w else "square"
    known = {(16, 9): "16:9", (9, 16): "9:16", (4, 3): "4:3", (3, 4): "3:4", (1, 1): "1:1"}
    for (a, b), name in known.items():
        if abs(w / h - a / b) < 0.02:
            return f"{name} {shape}"
    return f"{w / h:.2f}:1 {shape}"


def luma_stats(img: Image.Image) -> tuple[float, float]:
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    return round(float(arr.mean()), 1), round(float(arr.std()), 1)


EXIF_NAMES = {
    1: "EXIF 1 (none)", 2: "EXIF 2 (mirror)", 3: "EXIF 3 (rotate 180 applied)", 4: "EXIF 4 (flip)",
    5: "EXIF 5 (transpose applied)", 6: "EXIF 6 (rotate 90 CW applied)", 7: "EXIF 7 (transverse applied)",
    8: "EXIF 8 (rotate 90 CCW applied)",
}


def thumbnail_vector(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L").resize(THUMB, Image.BILINEAR), dtype=np.float32).ravel() / 255.0


# ---------------------------------------------------------------- selection

def kmeans_representatives(vectors: np.ndarray, k: int, seed: int = 42, iters: int = 50) -> list[tuple[int, int]]:
    """Return [(index of the item closest to each cluster centre, cluster size)], largest cluster first.

    Deterministic for a given seed. Uses k-means++ initialisation. If there are <= k items,
    every item is returned with size 1.
    """
    n = len(vectors)
    if n == 0 or k <= 0:
        return []
    if n <= k:
        return [(i, 1) for i in range(n)]
    rng = np.random.default_rng(seed)
    centres = [vectors[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([((vectors - c) ** 2).sum(1) for c in centres], axis=0)
        if d2.sum() == 0:  # all remaining points identical to a centre
            centres.append(vectors[rng.integers(n)])
        else:
            centres.append(vectors[rng.choice(n, p=d2 / d2.sum())])
    centres = np.array(centres)
    assign = np.zeros(n, dtype=int)
    for _ in range(iters):
        dist = ((vectors[:, None, :] - centres[None, :, :]) ** 2).sum(2)
        new = dist.argmin(1)
        if _ and (new == assign).all():
            break
        assign = new
        for j in range(k):
            if (assign == j).any():
                centres[j] = vectors[assign == j].mean(0)
    reps = []
    for j in range(k):
        members = np.flatnonzero(assign == j)
        if len(members) == 0:
            continue
        d = ((vectors[members] - centres[j]) ** 2).sum(1)
        reps.append((int(members[d.argmin()]), int(len(members))))
    reps.sort(key=lambda r: (-r[1], r[0]))
    return reps


def select_training_samples(root: Path, per_class: int = 3, seed: int = 42) -> tuple[list[Sample], dict]:
    """Pick representative training images per hand-state class. Returns (samples, selection info)."""
    _layout, classes = discover_hand_classes(root)
    samples: list[Sample] = []
    info: dict = {}
    for hc in classes:
        if hc.missing or not hc.files:
            info[hc.name] = {"total": 0, "duplicates_excluded": 0, "unreadable": 0}
            continue
        dup_groups = find_duplicates(root, hc.files)
        dropped = {p for g in dup_groups for p in g[1:]}
        candidates = [f for f in hc.files if f not in dropped]
        loaded, vectors, bad = [], [], 0
        for rel in candidates:
            try:
                img, meta = load_for_display(root / rel, max_size=(256, 256))
            except Exception:
                bad += 1
                continue
            loaded.append((rel, meta))
            vectors.append(thumbnail_vector(img))
        reps = kmeans_representatives(np.array(vectors), per_class, seed=seed)
        for idx, size in reps:
            rel, meta = loaded[idx]
            full, _ = load_for_display(root / rel)
            mean, std = luma_stats(full)
            orient = meta["exif_orientation"]
            samples.append(Sample(
                source_type=TRAINING, label=hc.name, source=rel,
                stored_width=meta["original_width"], stored_height=meta["original_height"],
                display_width=full.width, display_height=full.height,
                orientation=EXIF_NAMES.get(orient, "no EXIF orientation"),
                mean_luma=mean, std_luma=std, cluster_size=size, class_total=len(loaded),
            ))
        info[hc.name] = {"total": len(hc.files), "duplicates_excluded": len(dropped), "unreadable": bad,
                         "clustered": len(loaded)}
    return samples, info


# ---------------------------------------------------------------- video

def find_video(root: Path, name: str = DEFAULT_VIDEO_NAME) -> Path:
    matches = sorted(p for p in root.rglob(name) if p.is_file())
    if not matches:
        raise DomainCompareError(f"Video '{name}' not found under the dataset root. Pass --video PATH.")
    return matches[0]


def extract_video_frames(video: Path, fractions=DEFAULT_FRACTIONS) -> tuple[list[tuple[Image.Image, float, int]], dict]:
    """Decode the video once, keep frames nearest to each fraction of the duration.

    Returns ([(RGB PIL image, timestamp s, frame index)], video info). Frames come out with
    OpenCV's automatic orientation (CAP_PROP_ORIENTATION_AUTO) applied.
    """
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise DomainCompareError(f"Could not open video: {video.name}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stored = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0) if hasattr(cv2, "CAP_PROP_ORIENTATION_META") else None
        if count <= 0:
            raise DomainCompareError(f"Video reports no frames: {video.name}")
        targets = sorted({min(count - 1, max(0, round(f * (count - 1)))) for f in fractions})
        wanted = set(targets)
        out = []
        idx = 0
        while wanted:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                ts = ts_ms / 1000.0 if ts_ms and ts_ms > 0 else (idx / fps if fps else 0.0)
                out.append((Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), round(ts, 2), idx))
                wanted.discard(idx)
            idx += 1
    finally:
        cap.release()
    if not out:
        raise DomainCompareError(f"No frames could be decoded from {video.name}")
    info = {"fps": round(fps, 3), "frame_count": count, "duration_s": round(count / fps, 2) if fps else None,
            "stored_width": stored[0], "stored_height": stored[1], "rotation_meta": rotation}
    return out, info


def video_samples(video: Path, root: Path | None, fractions=DEFAULT_FRACTIONS) -> tuple[list[Sample], list[Image.Image], dict]:
    frames, info = extract_video_frames(video, fractions)
    try:
        rel = video.relative_to(root).as_posix() if root else video.name
    except ValueError:
        rel = video.name
    samples, images = [], []
    for img, ts, idx in frames:
        mean, std = luma_stats(img)
        samples.append(Sample(
            source_type=REAL_VIDEO, label=None, source=rel,
            stored_width=info["stored_width"], stored_height=info["stored_height"],
            display_width=img.width, display_height=img.height,
            orientation=f"rotation meta {info['rotation_meta']}" if info["rotation_meta"] is not None else "unknown",
            mean_luma=mean, std_luma=std, timestamp_s=ts, frame_index=idx,
        ))
        images.append(img)
    return samples, images, info


# ---------------------------------------------------------------- rendering

def render_comparison(rows: list[tuple[str, list[tuple[Sample, Image.Image]]]], out_path: Path,
                      tile_w: int = 480, tile_h: int = 360, title: str = "") -> Path:
    """One row per group, images letterboxed (aspect kept) with a caption under each."""
    cols = max(len(r[1]) for r in rows)
    pad, cap_h, head_h, title_h = 14, 92, 34, 56
    width = pad + cols * (tile_w + pad)
    height = title_h + sum(head_h + tile_h + cap_h + pad for _ in rows) + pad
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)
    f_title, f_head, f_cap, f_small = _font(24), _font(19), _font(15), _font(13)
    draw.text((pad, 14), title or "Driver-camera domain comparison", fill=TEXT, font=f_title)
    y = title_h
    for r, (heading, items) in enumerate(rows):
        color = HEADER_COLORS[r % len(HEADER_COLORS)]
        draw.rectangle([pad, y, width - pad, y + head_h - 6], fill=color)
        draw.text((pad + 10, y + 5), heading, fill=TEXT, font=f_head)
        y += head_h
        for c, (s, img) in enumerate(items):
            x = pad + c * (tile_w + pad)
            draw.rectangle([x, y, x + tile_w, y + tile_h], fill=TILE_BG)
            shown = ImageOps.contain(img, (tile_w, tile_h))
            sheet.paste(shown, (x + (tile_w - shown.width) // 2, y + (tile_h - shown.height) // 2))
            lines = caption_lines(s)
            ty = y + tile_h + 4
            for i, line in enumerate(lines):
                font = f_cap if i < 2 else f_small
                draw.text((x + 2, ty), _fit_text(draw, line, font, tile_w - 4), fill=TEXT if i < 2 else MUTED, font=font)
                ty += 20 if i < 2 else 17
        y += tile_h + cap_h + pad
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)
    return out_path


def caption_lines(s: Sample) -> list[str]:
    if s.source_type == REAL_VIDEO:
        first = f"{REAL_VIDEO} | label: none (unlabelled)"
        second = f"t = {s.timestamp_s:.2f} s (frame {s.frame_index})"
    else:
        first = f"{TRAINING} | label: {s.label}"
        second = f"represents {s.cluster_size} of {s.class_total} images in class"
    res = f"original {s.stored_width}x{s.stored_height}"
    if (s.display_width, s.display_height) != (s.stored_width, s.stored_height):
        res += f" -> shown {s.display_width}x{s.display_height}"
    return [first, second, f"{res} | {s.orientation}", Path(s.source).name]


# ---------------------------------------------------------------- report

def _group_summary(samples: list[Sample]) -> dict:
    if not samples:
        return {}
    return {
        "n": len(samples),
        "display_resolutions": sorted({f"{s.display_width}x{s.display_height}" for s in samples}),
        "aspects": sorted({s.aspect for s in samples}),
        "orientations": sorted({s.orientation for s in samples}),
        "mean_luma_range": [min(s.mean_luma for s in samples), max(s.mean_luma for s in samples)],
        "std_luma_range": [min(s.std_luma for s in samples), max(s.std_luma for s in samples)],
    }


def render_report(samples: list[Sample], selection: dict, video_info: dict, image_rel: str,
                  observations: str | None = None) -> str:
    lines = [
        "# Driver-camera domain comparison",
        "",
        "Purpose: show how the hand-state **training images** and the **real driver video** differ visually "
        "(camera viewpoint, framing, resolution, lighting). This report does not evaluate the classifier.",
        "",
        f"Comparison sheet: `{image_rel}`",
        "",
        "Sources were only read. EXIF orientation was applied to in-memory copies; video frames use OpenCV's "
        "automatic orientation handling.",
        "",
        "## 1. Samples",
        "",
        "Training samples: per class, exact duplicates removed, then seeded k-means (k = 3) on 32x32 grayscale "
        "thumbnails; the image nearest each cluster centre is shown. \"Represents\" = cluster size. "
        "Video frames: evenly spaced at 20 %, 50 % and 80 % of the duration.",
        "",
        "| # | Source type | Label | Timestamp | Original (stored) | Shown | Aspect | Orientation | Mean luma | Luma std | File |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, s in enumerate(samples, 1):
        ts = f"{s.timestamp_s:.2f} s (frame {s.frame_index})" if s.timestamp_s is not None else "-"
        label = s.label or "none (unlabelled)"
        if s.cluster_size:
            label += f" (represents {s.cluster_size}/{s.class_total})"
        lines.append(
            f"| {i} | {s.source_type} | {label} | {ts} | {s.stored_width}x{s.stored_height} | "
            f"{s.display_width}x{s.display_height} | {s.aspect} | {s.orientation} | {s.mean_luma} | {s.std_luma} | "
            f"`{Path(s.source).name}` |"
        )
    lines += ["", "Class folders:", ""]
    for name, inf in selection.items():
        lines.append(f"- `{name}`: {inf.get('total', 0)} files, {inf.get('duplicates_excluded', 0)} exact duplicate(s) "
                     f"excluded from selection, {inf.get('unreadable', 0)} unreadable")
    if video_info:
        lines += ["", f"Video: {video_info['stored_width']}x{video_info['stored_height']}, {video_info['fps']} fps, "
                      f"{video_info['frame_count']} frames, {video_info['duration_s']} s, rotation metadata "
                      f"{video_info['rotation_meta']}."]
    lines += ["", "## 2. Measured summary", "",
              "| Group | Shown resolution(s) | Aspect(s) | Orientation handling | Mean luma range | Luma std range |",
              "| --- | --- | --- | --- | --- | --- |"]
    groups = [(CLASS_TO_STATE.get(n, n), [s for s in samples if s.label == n]) for n in selection]
    groups.append((REAL_VIDEO, [s for s in samples if s.source_type == REAL_VIDEO]))
    for name, group in groups:
        g = _group_summary(group)
        if g:
            lines.append(f"| {name} | {', '.join(g['display_resolutions'])} | {', '.join(g['aspects'])} | "
                         f"{'; '.join(g['orientations'])} | {g['mean_luma_range'][0]}-{g['mean_luma_range'][1]} | "
                         f"{g['std_luma_range'][0]}-{g['std_luma_range'][1]} |")
    lines += ["", "Luma = 0.299R + 0.587G + 0.114B on the full-resolution image (0-255). Mean luma is overall "
                  "brightness; std is a rough global-contrast measure.", ""]
    lines.append(observations.rstrip() + "\n" if observations else
                 f"{OBSERVATIONS_HEADING}\n\n_Not written yet. Add visual observations here; this section is kept "
                 "when the report is regenerated._\n")
    return "\n".join(lines)


def existing_observations(report_path: Path) -> str | None:
    """Return section 3 of an existing report (heading included), or None."""
    if not report_path.is_file():
        return None
    text = report_path.read_text(encoding="utf-8")
    pos = text.find(OBSERVATIONS_HEADING)
    if pos < 0:
        return None
    section = text[pos:]
    return None if "_Not written yet." in section else section


def run_comparison(root: Path, video: Path, out_image: Path, out_report: Path | None,
                   per_class: int = 3, fractions=DEFAULT_FRACTIONS, seed: int = 42, project_root: Path | None = None) -> dict:
    train, selection = select_training_samples(root, per_class, seed)
    if not train:
        raise DomainCompareError("No readable hand-state training images found.")
    vid_samples, vid_images, vinfo = video_samples(video, root, fractions)

    by_label: dict[str, list[tuple[Sample, Image.Image]]] = {}
    for s in train:
        img, _ = load_for_display(root / s.source)
        by_label.setdefault(s.label, []).append((s, img))
    rows = [(f"{TRAINING}  -  {CLASS_TO_STATE.get(lbl, lbl)}  ({lbl})", items) for lbl, items in by_label.items()]
    rows.append((f"{REAL_VIDEO}  -  {Path(vid_samples[0].source).name}  (unlabelled)", list(zip(vid_samples, vid_images))))
    render_comparison(rows, out_image, title="Hand-state training images vs. real driver video")

    all_samples = train + vid_samples
    result = {"image": str(out_image), "samples": [asdict(s) for s in all_samples], "selection": selection, "video": vinfo}
    if out_report:
        rel_img = out_image
        if project_root:
            try:
                rel_img = out_image.resolve().relative_to(project_root.resolve())
            except ValueError:
                pass
        text = render_report(all_samples, selection, vinfo, Path(rel_img).as_posix(), existing_observations(out_report))
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(text, encoding="utf-8")
        result["report"] = str(out_report)
    return result


def to_json(result: dict) -> str:
    return json.dumps(result, indent=2)
