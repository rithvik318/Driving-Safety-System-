"""Write the manifest (JSON) and the human-readable audit report (Markdown)."""

from __future__ import annotations

import json
from pathlib import Path

from app.dataset_audit.scan import CLASS_ROLES

HAND_STATE_CLASSES = [c for c, role in CLASS_ROLES.items() if role == "hand_state"]
DROWSINESS_CLASSES = [c for c, role in CLASS_ROLES.items() if role == "drowsiness"]


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _fmt_seconds(value) -> str:
    if value is None:
        return "n/a"
    minutes, seconds = divmod(float(value), 60)
    return f"{value:.1f} s ({int(minutes)}m {seconds:04.1f}s)"


def _stat(stats: dict | None, key: str = "mean") -> str:
    return "n/a" if not stats else str(stats[key])


def _top_str(items: list[dict], key: str, n: int = 3) -> str:
    return ", ".join(f"{item[key]} ({item['count']})" for item in items[:n])


def _counts_table(title: str, counts: dict[str, dict]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Folder | Files | Images | Videos | Unsupported | Ignored | Empty | Corrupt |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cls, c in counts.items():
        lines.append(
            f"| `{cls}` | {c['files']} | {c['images']} | {c['videos']} | {c['unsupported']} "
            f"| {c['ignored_system_files']} | {c['empty']} | {c['corrupt']} |"
        )
    return lines + [""]


def _distribution(title: str, counts: dict[str, dict], unit_for) -> list[str]:
    rows = [(cls, unit_for(cls, c)) for cls, c in counts.items()]
    total = sum(n for _, (n, _) in rows) or 1
    lines = [f"**{title}**", "", "| Class | Count | Unit | Share of camera total |", "| --- | ---: | --- | ---: |"]
    for cls, (n, unit) in rows:
        lines.append(f"| `{cls}` | {n} | {unit} | {100 * n / total:.1f}% |")
    return lines + [""]


def _driver_unit(cls: str, c: dict):
    if cls in DROWSINESS_CLASSES:
        return c["readable_videos"], "readable videos"
    return c["readable_images"], "readable images"


def _front_unit(cls: str, c: dict):
    return c["readable_images"] + c["readable_videos"], f"readable items ({c['readable_images']} img, {c['readable_videos']} vid)"


def collect_issues(m: dict) -> list[str]:
    """Factual issues derived from the manifest. No model advice."""
    issues = []
    s = m["structure"]
    if s["missing_expected_folders"]:
        issues.append("Expected folders missing: " + ", ".join(f"`{p}`" for p in s["missing_expected_folders"]))
    if s["unexpected_folders"]:
        issues.append(
            "Folders not in the expected structure (not renamed or merged by this audit): "
            + ", ".join(f"`{p}`" for p in s["unexpected_folders"])
        )
    if s["files_outside_class_folders"]:
        issues.append(f"{len(s['files_outside_class_folders'])} file(s) sit outside any class folder, so they have no label.")
    if m["unsupported_files"]:
        exts = sorted({f["extension"] or "(none)" for f in m["unsupported_files"]})
        issues.append(f"{len(m['unsupported_files'])} unsupported file(s); extensions: {', '.join(exts)}.")
    if m["empty_files"]:
        issues.append(f"{len(m['empty_files'])} empty (0-byte) file(s).")
    if m["corrupt_files"]:
        issues.append(f"{len(m['corrupt_files'])} corrupt/unreadable file(s); see section 6.")
    if m["duplicate_groups"]:
        cross = [g for g in m["duplicate_groups"] if g["cross_label"]]
        issues.append(
            f"{len(m['duplicate_groups'])} exact-duplicate group(s) ({m['duplicate_file_count']} redundant file(s))."
            + (f" {len(cross)} group(s) span different labels (the same file carries conflicting labels)." if cross else "")
        )
    for cls in HAND_STATE_CLASSES:
        c = m["driver_camera"].get(cls)
        if c and c["videos"]:
            issues.append(f"`driver_cam/{cls}` contains {c['videos']} video(s) in an image-classification folder.")
    for cls in DROWSINESS_CLASSES:
        drowsy = m["driver_camera"].get(cls)
        if drowsy and drowsy["images"]:
            issues.append(f"`driver_cam/{cls}` contains {drowsy['images']} image(s); drowsiness is planned as video/temporal data.")

    img = m["images"]
    if img.get("distinct_dimensions", 0) > 1:
        issues.append(f"Images come in {img['distinct_dimensions']} different dimensions (see section 4).")
    if len(img.get("orientation", {})) > 1:
        issues.append(f"Mixed image orientation: {img['orientation']}.")
    if img.get("exif_rotated"):
        issues.append(
            f"{img['exif_rotated']} image(s) have an EXIF rotation tag; stored pixel orientation differs from how they display."
        )
    vids = m["videos"]
    if vids.get("readable") and len(vids.get("common_resolutions", [])) > 1:
        issues.append("Videos come in more than one resolution (see section 5).")
    if vids.get("readable") and any(k not in ("0", "None") for k in vids.get("rotation_meta", {})):
        issues.append("Some videos carry rotation metadata; frame width/height may not match how they play.")
    if vids.get("videos_without_duration"):
        issues.append(f"{vids['videos_without_duration']} readable video(s) report no duration (missing FPS or frame count).")
    issues.append("Labels were not verified: this audit treats the folder name as the label and does not inspect content.")
    return issues


def render_report(m: dict) -> str:
    t = m["totals"]
    img, vid = m["images"], m["videos"]
    L: list[str] = [
        "# Dataset audit",
        "",
        f"- Dataset folder: `{m['dataset_root']}` (paths below are relative to it)",
        f"- Audited: {m['audit_timestamp']} (UTC)",
        f"- Provenance: `{m['data_source']}`",
        f"- Tools: " + ", ".join(f"{k} {v}" for k, v in m["tool_versions"].items()),
        "",
        "> Readable files are not the same as training-ready data. This report describes what is on disk; it does not assess label quality.",
        "",
        "## 1. Dataset overview",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Files | {t['files']} |",
        f"| Images (readable) | {t['images']} ({t['readable_images']}) |",
        f"| Videos (readable) | {t['videos']} ({t['readable_videos']}) |",
        f"| Unsupported | {t['unsupported']} |",
        f"| Ignored system files | {t['ignored_system_files']} |",
        f"| Empty | {t['empty']} |",
        f"| Corrupt / unreadable | {t['corrupt']} |",
        f"| Exact-duplicate groups | {len(m['duplicate_groups'])} |",
        "",
        f"Folder layout recognised: `{m['structure'].get('layout', 'original_plan')}`",
        "",
        "Camera folders found: " + (", ".join(
            f"{cam} → " + ", ".join(f"`{d}/`" for d in dirs)
            for cam, dirs in m["structure"].get("camera_dirs", {}).items()
        ) or "none"),
        "",
        "## 2. Driver-camera counts",
        "",
        *_counts_table("driver_cam", m["driver_camera"]),
        "## 3. Front-camera counts",
        "",
        *_counts_table("front_cam", m["front_camera"]),
    ]
    for camera, counts in m["other_folders"].items():
        L += _counts_table(f"{camera} (not in expected structure)", counts)

    L += ["## 4. Image statistics", ""]
    if img.get("readable"):
        L += [
            f"- Total images: {img['total']} ({img['readable']} readable)",
            f"- Formats: {img['formats']} · extensions: {img['extensions']}",
            f"- Smallest: {img['min_dimensions']} · largest: {img['max_dimensions']} (by pixel area)",
            f"- Mean size: {_stat(img['width'])} × {_stat(img['height'])} px · distinct dimensions: {img['distinct_dimensions']}",
            f"- Common dimensions: {_top_str(img['common_dimensions'], 'dimensions', 5)}",
            f"- Aspect ratios: {img['aspect_ratios']}",
            f"- Orientation: {img['orientation']} · EXIF-rotated: {img['exif_rotated']}",
            f"- File size (bytes): min {_stat(img['file_size_bytes'], 'min')}, mean {_stat(img['file_size_bytes'])}, max {_stat(img['file_size_bytes'], 'max')}",
            "",
            "| Class | Readable | Common dimensions | Orientation |",
            "| --- | ---: | --- | --- |",
        ]
        for cls, st in img["per_class"].items():
            common = _top_str(st.get("common_dimensions", []), "dimensions")
            L.append(f"| `{cls}` | {st['readable']} | {common or 'n/a'} | {st.get('orientation', {})} |")
        L.append("")
    else:
        L += ["No readable images found.", ""]

    L += ["## 5. Video statistics", ""]
    if vid.get("readable"):
        L += [
            "| Class | Videos (readable) | Total duration | Min | Max | Mean | Common resolution | Common FPS | Codecs |",
            "| --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
        for cls, st in {**vid["per_class"], "ALL": vid}.items():
            if not st.get("readable"):
                L.append(f"| `{cls}` | {st['total']} (0) | n/a | | | | | | |")
                continue
            d = st.get("duration_s")
            L.append(
                f"| `{cls}` | {st['total']} ({st['readable']}) | {_fmt_seconds(st.get('total_duration_s'))} "
                f"| {_stat(d, 'min')} s | {_stat(d, 'max')} s | {_stat(d)} s "
                f"| {_top_str(st['common_resolutions'], 'resolution')} "
                f"| {_top_str(st['common_fps'], 'fps')} | {st['codecs']} |"
            )
        L += [
            "",
            "Duration = frame count ÷ FPS as reported by the container. Phone recordings are often variable-frame-rate, so treat durations as approximate.",
            "",
        ]
    else:
        L += ["No readable videos found.", ""]

    L += ["## 6. Corrupt / unreadable files", ""]
    L += [f"- `{c['path']}` ({c['kind']}): {c['error']}" for c in m["corrupt_files"]] or ["None found."]
    if m["empty_files"]:
        L += ["", "Empty (0-byte) files:"] + [f"- `{p}`" for p in m["empty_files"]]
    if m["unsupported_files"]:
        L += ["", "Unsupported files:"] + [f"- `{f['path']}`" for f in m["unsupported_files"]]
    L.append("")

    L += ["## 7. Exact-duplicate groups (SHA-256)", ""]
    if m["duplicate_groups"]:
        L.append(f"{len(m['duplicate_groups'])} group(s), {m['duplicate_file_count']} redundant file(s). Nothing was deleted.")
        L.append("")
        for i, g in enumerate(m["duplicate_groups"], 1):
            flag = " **(different labels)**" if g["cross_label"] else ""
            L.append(f"{i}. {g['size_bytes']} bytes{flag}: " + ", ".join(f"`{p}`" for p in g["paths"]))
    else:
        L.append("None found. (Only byte-identical files are detected; resized or re-encoded copies are not.)")
    L.append("")

    L += ["## 8. Class distribution", ""]
    L += _distribution("Driver camera", m["driver_camera"], _driver_unit)
    L += _distribution("Front camera", m["front_camera"], _front_unit)
    L += ["Counts only. No balance judgement is made here.", ""]

    L += [
        "## 9. Data-format observations",
        "",
        f"- Image extensions: {img.get('extensions', {})}",
        f"- Video extensions: {_ext_counts(m, 'video')}",
        f"- Nested subfolders inside class folders: " + (", ".join(f"`{p}`" for p in m["structure"]["nested_subfolders"]) or "none"),
        f"- Ignored system files: {len(m['ignored_system_files'])}",
        "",
        "## 10. Potential issues requiring attention",
        "",
    ]
    L += [f"- {issue}" for issue in collect_issues(m)]
    L.append("")
    return "\n".join(L)


def _ext_counts(m: dict, kind: str) -> dict:
    counts: dict[str, int] = {}
    for r in m["files"]:
        if r["kind"] == kind:
            counts[r["extension"]] = counts.get(r["extension"], 0) + 1
    return counts


def write_report(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(manifest), encoding="utf-8")
