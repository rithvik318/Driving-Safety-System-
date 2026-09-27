"""Evidence for events: single frames or pre/post clips from a ring buffer.

Layout (paths stored in records are RELATIVE to the event dataset root):
    <event_root>/<run_id>/event_000001/metadata.json
    <event_root>/<run_id>/event_000001/front_frame.jpg | driver_frame.jpg   (evidence_mode "frame")
    <event_root>/<run_id>/event_000001/front.mp4       | driver.mp4         (evidence_mode "clip")
(event ids restart per run, so the run id keeps evidence of different runs apart)

Frame mode keeps only the latest frame per camera (no copies). Clip mode keeps a ring buffer of
downscaled frames covering evidence_pre_seconds, then collects frames until
evidence_post_seconds after the event; the clip is written when complete or when the stream ends
(then shorter, and metadata.json says so). Clips contain only the frames that were processed
(the sampled frames), at their real timestamps. Frames are BGR uint8 arrays (camera convention);
images are written with Pillow, clips with OpenCV (imported only in clip mode).
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from app.config.event_config import EventConfig


def _to_rgb_small(frame: np.ndarray, max_side: int) -> np.ndarray:
    img = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    return np.asarray(img)


@dataclass
class PendingClip:
    event_id: str
    event_time: float
    cameras: tuple[str, ...]
    frames: dict = field(default_factory=dict)  # camera -> [(t, rgb)]
    record: dict | None = None  # record values waiting for the evidence to exist


class EvidenceStore:
    """Buffers frames per camera and writes evidence directories."""

    def __init__(self, root: Path, config: EventConfig, prefix: str = ""):
        self.root = Path(root)
        self.prefix = prefix.strip("/")
        self.cfg = config
        self.mode = config.evidence_mode
        self._latest: dict[str, tuple[float, np.ndarray]] = {}
        self._buffers: dict[str, deque] = {}
        self.pending: list[PendingClip] = []

    def _rel(self, event_id: str, name: str | None = None) -> str:
        parts = [p for p in (self.prefix, event_id, name) if p]
        return "/".join(parts)

    def _dir(self, event_id: str) -> Path:
        return self.root / self._rel(event_id)

    def reset(self) -> None:
        self._latest.clear()
        self._buffers.clear()

    # ------------------------------------------------------------------ buffering

    def push(self, camera: str, t: float, frame: np.ndarray) -> None:
        if self.mode == "none" or frame is None:
            return
        self._latest[camera] = (t, frame)
        if self.mode != "clip":
            return
        small = _to_rgb_small(frame, self.cfg.evidence_max_side)
        buf = self._buffers.setdefault(camera, deque())
        buf.append((t, small))
        while buf and t - buf[0][0] > self.cfg.evidence_pre_seconds + 1e-9:
            buf.popleft()
        for clip in self.pending:
            if camera in clip.cameras and t > clip.event_time and t <= clip.event_time + self.cfg.evidence_post_seconds + 1e-9:
                clip.frames.setdefault(camera, []).append((t, small))

    def cameras_with_frames(self) -> list[str]:
        return sorted(self._latest)

    # ------------------------------------------------------------------ writing

    def _write_metadata(self, event_dir: Path, record: dict, evidence: dict) -> None:
        (event_dir / "metadata.json").write_text(json.dumps({"record": record, "evidence": evidence}, indent=2),
                                                 encoding="utf-8")

    def write_frames(self, event_id: str, t: float, cameras: list[str], record: dict) -> tuple[str, str, list[str]]:
        """Frame mode: one JPEG per camera (the latest frame at or before the event)."""
        event_dir = self._dir(event_id)
        event_dir.mkdir(parents=True, exist_ok=True)
        files, info = [], {}
        for cam in cameras:
            if cam not in self._latest:
                continue
            ft, frame = self._latest[cam]
            name = f"{cam}_frame.jpg"
            Image.fromarray(_to_rgb_small(frame, self.cfg.evidence_max_side)).save(
                event_dir / name, quality=self.cfg.evidence_jpeg_quality)
            files.append(self._rel(event_id, name))
            info[cam] = {"file": name, "frame_timestamp": round(ft, 4)}
        if not files:
            return "NONE", None, []
        rel = [self._rel(event_id, "metadata.json")] + files
        record = {**record, "evidence_kind": "FRAME", "evidence_path": self._rel(event_id), "evidence_files": json.dumps(rel)}
        self._write_metadata(event_dir, record, {"kind": "FRAME", "cameras": info,
                                                 "note": "single frame per camera; no clip recorded (evidence_mode=frame)"})
        return "FRAME", self._rel(event_id), rel

    def start_clip(self, event_id: str, t: float, cameras: list[str], record: dict) -> PendingClip:
        clip = PendingClip(event_id, t, tuple(c for c in cameras if c in self._buffers), record=record)
        for cam in clip.cameras:
            clip.frames[cam] = [(ft, img) for ft, img in self._buffers[cam] if ft <= t + 1e-9]
        self.pending.append(clip)
        return clip

    def ready_clips(self, now: float, force: bool = False) -> list[PendingClip]:
        done = [c for c in self.pending if force or now >= c.event_time + self.cfg.evidence_post_seconds - 1e-9]
        self.pending = [c for c in self.pending if c not in done]
        return done

    def write_clip(self, clip: PendingClip) -> tuple[str, str | None, list[str], dict]:
        """Write the clip files. Returns (kind, path, files, record-with-evidence)."""
        import cv2

        event_dir = self._dir(clip.event_id)
        files, info = [], {}
        for cam in clip.cameras:
            frames = sorted(clip.frames.get(cam, []), key=lambda x: x[0])
            if len(frames) < 2:
                continue
            event_dir.mkdir(parents=True, exist_ok=True)
            span = frames[-1][0] - frames[0][0]
            fps = max(1.0, (len(frames) - 1) / span) if span > 0 else 10.0
            h, w = frames[0][1].shape[:2]
            name = f"{cam}.mp4"
            writer = cv2.VideoWriter(str(event_dir / name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w - w % 2, h - h % 2))
            for _, img in frames:
                if img.shape[:2] != (h, w):
                    img = np.asarray(Image.fromarray(img).resize((w, h)))
                writer.write(np.ascontiguousarray(img[: h - h % 2, : w - w % 2, ::-1]))
            writer.release()
            files.append(self._rel(clip.event_id, name))
            info[cam] = {"file": name, "frames": len(frames), "first_timestamp": round(frames[0][0], 4),
                         "last_timestamp": round(frames[-1][0], 4),
                         "pre_seconds_covered": round(clip.event_time - frames[0][0], 3),
                         "post_seconds_covered": round(frames[-1][0] - clip.event_time, 3), "fps_written": round(fps, 3)}
        record = dict(clip.record)
        if not files:
            return "NONE", None, [], record
        rel = [self._rel(clip.event_id, "metadata.json")] + files
        record.update(evidence_kind="CLIP", evidence_path=self._rel(clip.event_id), evidence_files=json.dumps(rel))
        self._write_metadata(event_dir, record, {
            "kind": "CLIP", "cameras": info, "target_pre_seconds": self.cfg.evidence_pre_seconds,
            "target_post_seconds": self.cfg.evidence_post_seconds,
            "note": "clips contain the processed (sampled) frames only; shorter than the target when the stream "
                    "started or ended inside the window"})
        return "CLIP", self._rel(clip.event_id), rel, record
