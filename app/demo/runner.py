"""End-to-end demo runner on ONE REAL local video, reusing the existing components only.

front camera:  VideoReader -> YoloRoadDetector -> IoUTracker -> RiskEngine -> EventRecorder -> AlertManager
driver camera: VideoReader -> DriverPerceptionPipeline (face + hand model) -> RiskEngine -> EventRecorder -> AlertManager

The two cameras were NOT recorded together; each mode runs on its own video. There is no synchronised
driver + road mode here (that exists only as the separately labelled SYNTHETIC_COMBINATION test).

Output directory (all LOCAL_REAL / INFERRED; never synthetic rows):
    events.jsonl, events.parquet, schema.json     event dataset (EventDatasetWriter kind="real")
    evidence/<run_id>/event_NNNNNN/...            evidence frames (paths in events are relative to evidence/)
    alerts.jsonl                                  alerts raised (linked to event ids)
    timeline.csv                                  one row per processed frame
    risk_timeline.png                             score/level plot with events and alarms
    annotated_demo.mp4                            overlay video (optional)
    demo_summary.json                             counts, provenance, files
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.alerts import AlertManager, ConsoleSink, TerminalBell
from app.alerts.models import DISCLAIMER, LEVEL_TO_SEVERITY
from app.config.settings import PROJECT_ROOT

TIMELINE_COLUMNS = ("timestamp", "frame_index", "risk_level", "risk_score", "raw_risk_score", "raw_risk_level",
                    "hazard_type", "hazard_class", "primary_track_id", "risk_reason", "alarm_recommended", "event_id",
                    "event_types", "alert_id", "alert_severity", "alarm", "detections", "tracks", "data_source",
                    "observation_type")
SUMMARY_KEYS = ("demo_version", "generated_at", "camera", "input", "processing", "perception", "risk", "events",
                "alerts", "evidence", "provenance", "outputs")
SYNTHETIC_DIRS = ("data/simulated", "data/events_synthetic")
NO_SYNC_STATEMENT = ("No synchronized real driver+road recording exists. The driver and front cameras were recorded "
                     "separately, so this demo runs one camera at a time and never pairs them.")
DEMO_VERSION = "1.0.0"


class DemoError(ValueError):
    pass


def check_real_video(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise DemoError(f"video not found: {path}")
    resolved = path.resolve()
    for d in SYNTHETIC_DIRS:
        root = (PROJECT_ROOT / d).resolve()
        if resolved == root or root in resolved.parents:
            raise DemoError(f"{path} is inside {d}: the demo only runs on real local recordings")
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------ overlay


LEVEL_BGR = {"SAFE": (12, 163, 12), "CAUTION": (25, 178, 250), "HIGH": (90, 131, 236), "CRITICAL": (59, 59, 208)}


def _overlay(frame, tracks, assessment, alert_text: str | None, t: float, max_side: int = 960):
    import cv2

    from app.tracking.visualize import draw_tracks

    out = draw_tracks(frame, list(tracks)) if tracks is not None else frame.copy()
    h, w = out.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        out = cv2.resize(out, (int(w * s) // 2 * 2, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA)
    h, w = out.shape[:2]
    level = assessment.risk_level.value
    bar = int(max(36, h * 0.07))
    cv2.rectangle(out, (0, 0), (w, bar), LEVEL_BGR[level], -1)
    text = f"{level}  score {assessment.risk_score:.0f}  t={t:.1f}s" + (f"  |  {alert_text}" if alert_text else "")
    scale = bar / 55
    cv2.putText(out, text, (10, int(bar * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, "LOCAL_REAL video | risk: rule-based prototype score, not a collision prediction", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class _VideoOut:
    def __init__(self, path: Path, fps: float):
        self.path, self.fps, self.writer, self.frames = path, fps, None, 0

    def write(self, img) -> None:
        import cv2

        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self.size = (w, h)
        if (img.shape[1], img.shape[0]) != self.size:
            img = cv2.resize(img, self.size)
        self.writer.write(img)
        self.frames += 1

    def close(self) -> bool:
        if self.writer is not None:
            self.writer.release()
        return self.writer is not None and self.path.is_file() and self.path.stat().st_size > 0


# ------------------------------------------------------------------------------------ run


@dataclasses.dataclass
class DemoRun:
    camera: str
    video: Path
    out_dir: Path
    run_id: str
    timeline: list = dataclasses.field(default_factory=list)
    events: list = dataclasses.field(default_factory=list)
    alerts: AlertManager | None = None
    stats: dict = dataclasses.field(default_factory=dict)
    video_info: object | None = None
    annotated: bool = False
    storage: dict = dataclasses.field(default_factory=dict)


def _step(run: DemoRun, recorder, alerts: AlertManager, console: ConsoleSink | None, a, tracks, driver, frame_index, size,
          frame, detections: int):
    evs = recorder.observe_frame(run.camera, a.timestamp, frame)
    evs += recorder.on_risk(a, tracks, driver, frame_index, size)
    if driver is not None:
        evs += recorder.on_driver(driver, frame_index)
    new_alerts = alerts.update(a, [e for e in evs if e.event_type in ("RISK_ESCALATED", "RISK_DEESCALATED")])
    if console is not None:
        for e in evs:
            if e.event_type in ("PERSISTENT_HAZARD", "DRIVER_STATE_CHANGE"):
                console.event(e)
    run.events += evs
    by_id = {tr.track_id: tr for tr in tracks or ()}
    prim = by_id.get(a.primary_track_id)
    al = new_alerts[0] if new_alerts else None
    run.timeline.append({
        "timestamp": round(a.timestamp, 3), "frame_index": frame_index, "risk_level": a.risk_level.value,
        "risk_score": round(a.risk_score, 2), "raw_risk_score": round(a.raw_risk_score, 2),
        "raw_risk_level": a.raw_risk_level.value, "hazard_type": a.hazard_type.value,
        "hazard_class": prim.observed_class if prim is not None else None, "primary_track_id": a.primary_track_id,
        "risk_reason": a.reason, "alarm_recommended": bool(a.alarm_recommended),
        "event_id": ";".join(e.event_id for e in evs) or None, "event_types": ";".join(e.event_type for e in evs) or None,
        "alert_id": al.alert_id if al else None, "alert_severity": al.severity.value if al else None,
        "alarm": bool(al.alarm) if al else False, "detections": detections, "tracks": len(tracks or ()),
        "data_source": "LOCAL_REAL", "observation_type": "INFERRED",
    })
    return al


def run_front_demo(video: Path, out_dir: Path, sample_fps: float = 10.0, device: str = "auto", max_seconds: float | None = None,
                   annotate: bool = True, console: bool = True, beep: bool = False, detector=None, evidence_mode: str = "frame") -> DemoRun:
    from app.config import load_event_config, load_risk_config, load_road_config, load_tracking_config
    from app.events import DataSource, EventRecorder
    from app.risk import RiskEngine, RiskSnapshot
    from app.road.runner import run_video
    from app.sensors.video import VideoReader
    from app.tracking import IoUTracker

    video = check_real_video(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rcfg = load_road_config()
    if detector is None:
        from app.road import YoloRoadDetector

        detector = YoloRoadDetector(Path(rcfg.model_path), rcfg.confidence_threshold, rcfg.image_size, device, rcfg.classes)
    ecfg = dataclasses.replace(load_event_config(), evidence_mode=evidence_mode)
    run = DemoRun("front", video, out_dir, f"demo_{video.stem}")
    sink = ConsoleSink() if console else None
    bell = TerminalBell(enabled=beep)
    run.alerts = AlertManager("LOCAL_REAL", [s for s in (sink, bell) if s is not None], session_id=video.stem)
    recorder = EventRecorder(ecfg, run_id=run.run_id, evidence_root=out_dir / "evidence")
    recorder.start_session(video.stem, video.name, DataSource.LOCAL_REAL, "front")
    tracker, engine = IoUTracker(load_tracking_config()), RiskEngine(load_risk_config())
    vout = _VideoOut(out_dir / "annotated_demo.mp4", sample_fps) if annotate else None
    det_classes, track_ids, track_classes = Counter(), set(), {}
    banner = {"text": None, "until": -1.0}

    def on_result(result, frame):
        t = result.timestamp
        dets = result.detections if result.status.value == "OK" else ()
        det_classes.update(d.class_name for d in dets)
        tracks = tracker.update(dets, t, result.frame_index)
        for tr in tracks:
            track_ids.add(tr.track_id)
            track_classes[tr.track_id] = tr.class_name
        h, w = frame.shape[:2]
        a = engine.evaluate(RiskSnapshot(t, tuple(tracks), None, w, h))
        al = _step(run, recorder, run.alerts, sink, a, tracks, None, result.frame_index, (w, h), frame, len(dets))
        if al is not None and al.kind == "ESCALATION":
            banner.update(text=al.headline, until=t + 2.0)
        if vout is not None:
            vout.write(_overlay(frame, tracks, a, banner["text"] if t <= banner["until"] else None, t))

    reader = VideoReader(video)
    started = time.perf_counter()
    try:
        run.video_info = reader.info()
        stats = run_video(reader, detector, sample_fps, max_seconds=max_seconds, keep_results=False, on_result=on_result)
    finally:
        reader.release()
    run.events += recorder.finish_session()
    run.annotated = vout.close() if vout is not None else False
    run.stats = {
        "frames_read": stats.frames_read, "frames_processed": len(run.timeline), "unreadable_frames": stats.unreadable_frames,
        "first_t": stats.first_t, "last_t": stats.last_t, "timestamp_sources": stats.timestamp_sources,
        "detections": sum(det_classes.values()), "detections_by_class": dict(det_classes.most_common()),
        "tracks": len(track_ids), "tracks_by_class": dict(Counter(track_classes.values()).most_common()),
        "detector": getattr(detector, "name", "detector"), "device": getattr(detector, "device", device),
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    _finish(run)
    return run


def run_driver_demo(video: Path, out_dir: Path, sample_fps: float = 10.0, max_seconds: float | None = None,
                    annotate: bool = True, console: bool = True, beep: bool = False, evidence_mode: str = "frame") -> DemoRun:
    import cv2

    from app.config import load_driver_config, load_event_config, load_risk_config
    from app.driver import DriverPerceptionPipeline, MediaPipeFaceProvider, TrainedHandStateProvider, UnknownHandStateProvider
    from app.events import DataSource, EventRecorder
    from app.risk import RiskEngine, RiskSnapshot
    from app.sensors.video import FrameClock, FrameSampler, VideoReader

    video = check_real_video(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_driver_config()
    face = MediaPipeFaceProvider(cfg.face_model_path, cfg.min_face_confidence)
    hand_loaded = Path(cfg.hand_model_path).is_file()
    hand = (TrainedHandStateProvider.from_checkpoint(cfg.hand_model_path, threshold=cfg.hand_confidence_threshold)
            if hand_loaded else UnknownHandStateProvider())
    pipeline = DriverPerceptionPipeline(face, cfg, hand_provider=hand)
    run = DemoRun("driver", video, out_dir, f"demo_{video.stem}")
    sink = ConsoleSink() if console else None
    run.alerts = AlertManager("LOCAL_REAL", [s for s in (sink, TerminalBell(enabled=beep)) if s is not None], session_id=video.stem)
    recorder = EventRecorder(dataclasses.replace(load_event_config(), evidence_mode=evidence_mode), run_id=run.run_id,
                             evidence_root=out_dir / "evidence")
    recorder.start_session(video.stem, video.name, DataSource.LOCAL_REAL, "driver")
    engine = RiskEngine(load_risk_config())
    vout = _VideoOut(out_dir / "annotated_demo.mp4", sample_fps) if annotate else None
    reader = VideoReader(video)
    run.video_info = reader.info()
    clock, sampler = FrameClock(run.video_info.fps), FrameSampler(sample_fps)
    started, i, first_t, last_t, frames_read = time.perf_counter(), -1, None, None, 0
    drowsy, hands = Counter(), Counter()
    try:
        while reader.grab():
            i += 1
            frames_read += 1
            t = clock.timestamp(i, reader.position_seconds())
            first_t = t if first_t is None else first_t
            if max_seconds is not None and t - first_t > max_seconds:
                break
            last_t = t
            if not sampler.should_process(t):
                continue
            ok, frame = reader.retrieve()
            if not ok or frame is None:
                continue
            state = pipeline.process(frame, t)
            drowsy[state.temporal.drowsiness_level.value] += 1
            hands[state.hand.hand_state.value] += 1
            a = engine.evaluate(RiskSnapshot(t, (), state, 0, 0))
            _step(run, recorder, run.alerts, sink, a, (), state, i, None, frame, 0)
            if vout is not None:
                img = _overlay(frame, None, a, None, t)
                cv2.putText(img, f"drowsiness {state.temporal.drowsiness_level.value}  hand {state.hand.hand_state.value}",
                            (10, img.shape[0] - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                vout.write(img)
    finally:
        reader.release()
        pipeline.close()
    run.events += recorder.finish_session()
    run.annotated = vout.close() if vout is not None else False
    run.stats = {"frames_read": frames_read, "frames_processed": len(run.timeline), "first_t": first_t, "last_t": last_t,
                 "driver_observations": len(run.timeline), "drowsiness_levels": dict(drowsy), "hand_states": dict(hands),
                 "hand_model_loaded": hand_loaded, "detections": 0, "tracks": 0,
                 "wall_seconds": round(time.perf_counter() - started, 1)}
    _finish(run)
    return run


# ------------------------------------------------------------------------------------ outputs


def _finish(run: DemoRun) -> None:
    from app.events import EventDatasetWriter

    run.events.sort(key=lambda e: e.event_id)
    writer = EventDatasetWriter(run.out_dir, kind="real", overwrite=True)  # refuses any synthetic record
    writer.append(run.events)
    run.storage = writer.finalize()
    with open(run.out_dir / "alerts.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for al in run.alerts.alerts:
            fh.write(json.dumps(al.to_dict()) + "\n")
    with open(run.out_dir / "timeline.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TIMELINE_COLUMNS)
        w.writeheader()
        for row in run.timeline:
            w.writerow({k: ("" if row[k] is None else row[k]) for k in TIMELINE_COLUMNS})
    plot_risk_timeline(run.timeline, run.events, run.out_dir / "risk_timeline.png", f"{run.video.name} ({run.camera} camera)")
    summary = build_summary(run)
    (run.out_dir / "demo_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def plot_risk_timeline(timeline: list[dict], events, path: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#1f1f1e", "#8a8a86", "#e6e6e3"
    status = {"SAFE": "#0ca30c", "CAUTION": "#fab219", "HIGH": "#ec835a", "CRITICAL": "#d03b3b"}
    ts = [r["timestamp"] for r in timeline]
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=120)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    if ts:
        dt = (ts[-1] - ts[0]) / max(1, len(ts) - 1)
        start = 0
        for i in range(1, len(timeline) + 1):  # level bands (after hysteresis)
            if i == len(timeline) or timeline[i]["risk_level"] != timeline[start]["risk_level"]:
                lv = timeline[start]["risk_level"]
                ax.axvspan(ts[start] - dt / 2, ts[i - 1] + dt / 2, ymin=0, ymax=0.045, color=status[lv], lw=0)
                start = i
        ax.plot(ts, [r["raw_risk_score"] for r in timeline], color=muted, lw=1, label="raw score (this frame)")
        ax.plot(ts, [r["risk_score"] for r in timeline], color=ink, lw=2, label="risk score (smoothed)")
    for y, lv in ((20, "CAUTION"), (45, "HIGH"), (70, "CRITICAL")):
        ax.axhline(y, color=status[lv], lw=1, ls=(0, (4, 3)))
        ax.text(1.005, y, f"{lv} ≥{y}", transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=ink)
    for e in events:
        if e.event_type in ("RISK_ESCALATED", "PERSISTENT_HAZARD"):
            ax.axvline(e.timestamp, color=grid if e.event_type == "RISK_ESCALATED" else "#c9c9c5", lw=1, zorder=0)
    alarms = [r for r in timeline if r["alarm"]]
    if alarms:
        ax.scatter([r["timestamp"] for r in alarms], [min(98, r["risk_score"] + 5) for r in alarms], marker="v", s=70,
                   color=status["CRITICAL"], edgecolor=surface, linewidth=2, zorder=5, label="alarm raised")
    ax.set_ylim(0, 100)
    ax.set_xlabel("video time (s)", color=ink)
    ax.set_ylabel("risk score (0-100, not a probability)", color=ink)
    ax.set_title(f"{title} · LOCAL_REAL input, INFERRED risk", color=ink, fontsize=11, loc="left")
    ax.grid(axis="y", color=grid, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted)
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=3)
    fig.text(0.01, 0.01, "bottom strip = risk level after hysteresis; vertical lines = recorded events", fontsize=7, color=muted)
    fig.tight_layout(rect=(0, 0.03, 0.97, 1))
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return path


def _dataset_relative(path: Path) -> str | None:
    """Path relative to DATASET_ROOT (machine-independent) when the video lives under it."""
    from app.config import load_settings

    try:
        root = load_settings().dataset_root
    except ValueError:
        return None
    if root is None:
        return None
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return None


def build_summary(run: DemoRun) -> dict:
    ev = run.events
    tl = run.timeline
    info = run.video_info
    transitions = [{"timestamp": e.timestamp, "transition": e.transition, "event_id": e.event_id, "hazard_type": e.hazard_type}
                   for e in ev if e.event_type in ("RISK_ESCALATED", "RISK_DEESCALATED")]
    evidence_files = [p for e in ev for p in (json.loads(e.evidence_files) if e.evidence_files else [])]
    missing = [p for p in evidence_files if not (run.out_dir / "evidence" / p).is_file()]
    duration = None
    if run.stats.get("first_t") is not None and run.stats.get("last_t") is not None:
        duration = round(run.stats["last_t"] - run.stats["first_t"], 3)
    return {
        "demo_version": DEMO_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "camera": run.camera,
        "input": {"video": str(run.video), "dataset_relative_path": _dataset_relative(run.video),
                  "file_name": run.video.name, "sha256": _sha256(run.video),
                  "width": getattr(info, "width", None), "height": getattr(info, "height", None),
                  "container_fps": getattr(info, "fps", None), "container_frame_count": getattr(info, "frame_count", None),
                  "duration_seconds_processed": duration, "container_duration_seconds": getattr(info, "container_duration", None)},
        "processing": {"sample_fps": round(len(tl) / duration, 2) if duration else None, **run.stats},
        "perception": {k: run.stats.get(k) for k in ("detections", "detections_by_class", "tracks", "tracks_by_class",
                                                        "driver_observations", "drowsiness_levels", "hand_states") if k in run.stats},
        "risk": {"frames_by_level": dict(Counter(r["risk_level"] for r in tl)), "transitions": transitions,
                 "peak_level": max((r["risk_level"] for r in tl), key=["SAFE", "CAUTION", "HIGH", "CRITICAL"].index, default=None),
                 "peak_score": max((r["risk_score"] for r in tl), default=None),
                 "frames_alarm_recommended": sum(r["alarm_recommended"] for r in tl),
                 "note": "risk_score is a rule-based prototype score (0-100), not a probability; image-space road signals only"},
        "events": {"total": len(ev), "by_type": dict(Counter(e.event_type for e in ev)),
                   "by_risk_level": dict(Counter(e.risk_level or "null" for e in ev)),
                   "by_hazard_type": dict(Counter(e.hazard_type or "null" for e in ev)), "run_id": run.run_id,
                   "storage": run.storage},
        "alerts": {**run.alerts.summary(), "severity_mapping": {k: v.value for k, v in LEVEL_TO_SEVERITY.items()},
                   "disclaimer": DISCLAIMER, "list": [a.to_dict() for a in run.alerts.alerts]},
        "evidence": {"events_with_evidence": sum(1 for e in ev if e.evidence_kind != "NONE"), "files": len(evidence_files),
                     "missing_files": missing, "root": "evidence/", "paths_relative_to": "evidence/"},
        "provenance": {"data_source": "LOCAL_REAL", "observation_type_of_derived_values": "INFERRED",
                       "event_data_sources": dict(Counter(e.data_source for e in ev)),
                       "event_observation_types": dict(Counter(e.observation_type for e in ev)),
                       "synthetic_rows_included": 0, "synchronized_driver_road": False,
                       "driver_input": "none (front camera only)" if run.camera == "front" else "driver camera only (no road input)",
                       "statement": NO_SYNC_STATEMENT, "gps": "UNAVAILABLE (no real GPS log)",
                       "alarm_triggered_field": "event records keep alarm_triggered=false (event schema 1.0); alarms raised by "
                                                "the alert layer are logged in alerts.jsonl and linked by event_id"},
        "outputs": {"events.jsonl": True, "events.parquet": (run.out_dir / "events.parquet").is_file(), "schema.json": True,
                    "alerts.jsonl": True, "timeline.csv": True, "risk_timeline.png": (run.out_dir / "risk_timeline.png").is_file(),
                    "evidence/": True, "annotated_demo.mp4": run.annotated},
    }
