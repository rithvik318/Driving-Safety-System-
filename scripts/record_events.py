"""Record events from real streams into the event dataset. Read-only on the source videos.

Three separate modes (run any combination):

  --front-source <video|folder>   REAL road-only: YOLO26n -> IoUTracker -> RiskEngine (no driver input)
                                  -> EventRecorder.  data_source=LOCAL_REAL, camera=front.
  --driver-source <video|folder>  REAL driver-only: DriverPerceptionPipeline -> RiskEngine (no road input)
                                  + driver-state changes -> EventRecorder.  data_source=LOCAL_REAL, camera=driver.
  --synthetic-combination DRIVER_VIDEO FRONT_VIDEO
                                  SOFTWARE TEST ONLY: real driver signals paired with a real front video by
                                  elapsed time. The cameras were NOT recorded together, so these events are
                                  data_source=SYNTHETIC_COMBINATION and go to the SEPARATE synthetic dataset.

Outputs:
  data/events/events.jsonl, events.parquet (if pyarrow), schema.json, <run_id>/event_NNNNNN/...  (REAL)
  data/events_synthetic/...                                                                     (SYNTHETIC)

GPS: --gps-csv FILE (timestamp,lat,lon[,accuracy_m] on the stream clock) if a real log exists;
otherwise GPS is UNAVAILABLE and coordinates stay null. Nothing is invented.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python scripts/...`


def _videos(sources) -> list[Path]:
    from app.road.runner import VIDEO_SUFFIXES, list_media

    out = []
    for s in sources or []:
        if not Path(s).exists():
            raise FileNotFoundError(f"not found: {s}")
        out += [m for m in list_media(Path(s)) if m.suffix.lower() in VIDEO_SUFFIXES]
    return out


def run_front(videos, recorder, detector, tracking_cfg, risk_cfg, sample_fps, data_source, driver_states=None):
    from app.risk import RiskEngine, RiskSnapshot
    from app.road.runner import run_video
    from app.sensors.video import VideoReader
    from app.tracking import IoUTracker

    produced = []
    for path in videos:
        camera = "front+driver" if driver_states is not None else "front"
        session = f"{path.stem}" if driver_states is None else f"{path.stem}+{driver_states[0]}"
        recorder.start_session(session, path.name, data_source, camera)
        tracker, engine = IoUTracker(tracking_cfg), RiskEngine(risk_cfg)
        states = driver_states[1] if driver_states is not None else None

        def on_result(result, frame, tracker=tracker, engine=engine, states=states):
            t = result.timestamp
            tracks = tracker.update(result.detections if result.status.value == "OK" else (), t, result.frame_index)
            driver = _driver_at(states, t) if states is not None else None
            h, w = frame.shape[:2]
            a = engine.evaluate(RiskSnapshot(t, tuple(tracks), driver, w, h))
            produced.extend(recorder.observe_frame("front", t, frame))
            if driver is not None and driver_frames is not None:
                df = driver_frames.get(round(driver.timestamp, 3))
                if df is not None:
                    produced.extend(recorder.observe_frame("driver", t, df))
            produced.extend(recorder.on_risk(a, tracks, driver, result.frame_index, (w, h)))
            if driver is not None:
                produced.extend(recorder.on_driver(driver, result.frame_index))

        driver_frames = driver_states[2] if driver_states is not None else None
        reader = VideoReader(path)
        try:
            run_video(reader, detector, sample_fps, keep_results=False, on_result=on_result)
        finally:
            reader.release()
        produced.extend(recorder.finish_session())
        print(f"  {path.parent.name}/{path.name}: {sum(1 for e in produced if e.session_id == session)} event(s)")
    return produced


def run_driver_pipeline(path: Path, sample_fps: float, keep_frames: bool = False):
    """Real DriverStates (and optionally their frames, downscaled) from a driver video."""
    import numpy as np
    from PIL import Image

    from app.config import load_driver_config
    from app.driver import DriverPerceptionPipeline, MediaPipeFaceProvider, TrainedHandStateProvider, UnknownHandStateProvider
    from app.sensors.video import FrameClock, FrameSampler, VideoReader

    cfg = load_driver_config()
    face = MediaPipeFaceProvider(cfg.face_model_path, cfg.min_face_confidence)
    hand = (TrainedHandStateProvider.from_checkpoint(cfg.hand_model_path, threshold=cfg.hand_confidence_threshold)
            if Path(cfg.hand_model_path).is_file() else UnknownHandStateProvider())
    pipeline = DriverPerceptionPipeline(face, cfg, hand_provider=hand)
    reader = VideoReader(path)
    clock, sampler, out, frames, i = FrameClock(reader.info().fps), FrameSampler(sample_fps), [], {}, -1
    try:
        while reader.grab():
            i += 1
            t = clock.timestamp(i, reader.position_seconds())
            if not sampler.should_process(t):
                continue
            ok, frame = reader.retrieve()
            if ok and frame is not None:
                state = pipeline.process(frame, t)
                out.append((state, i, frame if not keep_frames else None))
                if keep_frames:
                    img = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))
                    img.thumbnail((960, 960))
                    frames[round(t, 3)] = np.asarray(img)[:, :, ::-1].copy()
    finally:
        reader.release()
        pipeline.close()
    return out, frames


def run_driver(videos, recorder, risk_cfg, sample_fps, data_source):
    from app.risk import RiskEngine, RiskSnapshot

    produced = []
    for path in videos:
        states, _ = run_driver_pipeline(path, sample_fps)
        recorder.start_session(path.stem, path.name, data_source, "driver")
        engine = RiskEngine(risk_cfg)
        for state, idx, frame in states:
            produced.extend(recorder.observe_frame("driver", state.timestamp, frame))
            a = engine.evaluate(RiskSnapshot(state.timestamp, (), state, 0, 0))
            produced.extend(recorder.on_risk(a, (), state, idx, None))
            produced.extend(recorder.on_driver(state, idx))
        produced.extend(recorder.finish_session())
        n = sum(1 for e in produced if e.session_id == path.stem)
        levels = Counter(s.temporal.drowsiness_level.value for s, _, _ in states)
        hands = Counter(s.hand.hand_state.value for s, _, _ in states)
        print(f"  {path.parent.name}/{path.name}: {len(states)} driver observations, drowsiness {dict(levels)}, "
              f"hand {dict(hands)} -> {n} event(s)")
    return produced


def _driver_at(states, t):
    best = None
    for s in states:
        if s.timestamp <= t + 1e-9:
            best = s
        else:
            break
    if best is None or t > states[-1].timestamp + 0.2:
        return None
    return dataclasses.replace(best, timestamp=t)  # SYNTHETIC_COMBINATION pairing only


def summarize(records) -> dict:
    return {
        "events": len(records),
        "by_type": dict(Counter(r.event_type for r in records)),
        "by_risk_level": dict(Counter(r.risk_level or "null (driver-state change)" for r in records)),
        "by_data_source": dict(Counter(r.data_source for r in records)),
        "by_observation_type": dict(Counter(r.observation_type for r in records)),
        "by_camera": dict(Counter(r.camera for r in records)),
        "by_hazard_type": dict(Counter(r.hazard_type or "null" for r in records)),
        "by_evidence_kind": dict(Counter(r.evidence_kind for r in records)),
        "evidence_files": sum(len(json.loads(r.evidence_files)) if r.evidence_files else 0 for r in records),
        "sessions": len({r.session_id for r in records}),
    }


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    from app.config import (configure_logging, load_event_config, load_risk_config, load_road_config, load_settings,
                            load_tracking_config)
    from app.config.event_config import resolved_roots

    try:
        settings, rcfg, tcfg, kcfg, ecfg = (load_settings(), load_road_config(), load_tracking_config(),
                                            load_risk_config(), load_event_config())
    except ValueError as exc:
        print(f"ERROR: invalid configuration: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--front-source", action="append", type=Path, help="REAL front-camera video/folder (road-only).")
    p.add_argument("--driver-source", action="append", type=Path, help="REAL driver-camera video/folder (driver-only).")
    p.add_argument("--synthetic-combination", nargs=2, type=Path, metavar=("DRIVER_VIDEO", "FRONT_VIDEO"),
                   help="Software test: pair a driver video with a front video (SYNTHETIC_COMBINATION, separate dataset).")
    p.add_argument("--run-id", default="local_real_run_01")
    p.add_argument("--synthetic-run-id", default="synthetic_combination_run_01")
    p.add_argument("--sample-fps", type=float, default=10.0)
    p.add_argument("--gps-csv", type=Path, help="Real GPS log (timestamp,lat,lon[,accuracy_m]) on the stream clock.")
    p.add_argument("--append", action="store_true", help="Append to existing datasets instead of replacing them.")
    p.add_argument("--summary-json", type=Path)
    args = p.parse_args(argv)
    if not (args.front_source or args.driver_source or args.synthetic_combination):
        print("ERROR: give --front-source, --driver-source and/or --synthetic-combination", file=sys.stderr)
        return 2

    from app.events import DataSource, EventDatasetWriter, EventRecorder
    from app.sensors.gps import NoGPS, load_gps_csv

    real_root, syn_root = resolved_roots(ecfg)
    gps = load_gps_csv(args.gps_csv) if args.gps_csv else NoGPS()
    summary, started = {"gps": getattr(gps, "name", "UNAVAILABLE")}, time.perf_counter()
    try:
        front = _videos(args.front_source)
        driver = _videos(args.driver_source)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    detector = None
    if front or args.synthetic_combination:
        from app.road import YoloRoadDetector

        detector = YoloRoadDetector(Path(rcfg.model_path), rcfg.confidence_threshold, rcfg.image_size, rcfg.device, rcfg.classes)

    if front or driver:
        writer = EventDatasetWriter(real_root, kind="real", overwrite=not args.append)
        rec = EventRecorder(ecfg, run_id=args.run_id, gps=gps, evidence_root=real_root)
        real = []
        if front:
            print(f"\nREAL road-only: {len(front)} front-camera video(s) -> data_source=LOCAL_REAL")
            real += run_front(front, rec, detector, tcfg, kcfg, args.sample_fps, DataSource.LOCAL_REAL)
        if driver:
            print(f"\nREAL driver-only: {len(driver)} driver-camera video(s) -> data_source=LOCAL_REAL")
            real += run_driver(driver, rec, kcfg, args.sample_fps, DataSource.LOCAL_REAL)
        real.sort(key=lambda r: r.event_id)
        writer.append(real)
        summary["real"] = summarize(real) | {"storage": writer.finalize(), "run_id": args.run_id,
                                             "front_videos": [v.name for v in front], "driver_videos": [v.name for v in driver]}
        print("\nREAL events:", json.dumps(summary["real"], indent=2))

    if args.synthetic_combination:
        dpath, fpath = args.synthetic_combination
        print(f"\nSYNTHETIC_COMBINATION software test: {dpath.name} (driver) + {fpath.name} (front) - NOT recorded together")
        states, frames = run_driver_pipeline(dpath, args.sample_fps, keep_frames=True)
        swriter = EventDatasetWriter(syn_root, kind="synthetic", overwrite=not args.append)
        srec = EventRecorder(ecfg, run_id=args.synthetic_run_id, gps=NoGPS(), evidence_root=syn_root)
        syn = run_front([fpath], srec, detector, tcfg, kcfg, args.sample_fps, DataSource.SYNTHETIC_COMBINATION,
                        driver_states=(dpath.stem, [s for s, _, _ in states], frames))
        syn.sort(key=lambda r: r.event_id)
        swriter.append(syn)
        summary["synthetic_combination"] = summarize(syn) | {"storage": swriter.finalize(), "run_id": args.synthetic_run_id,
                                                             "driver_video": dpath.name, "front_video": fpath.name}
        print("\nSYNTHETIC_COMBINATION events:", json.dumps(summary["synthetic_combination"], indent=2))

    summary["wall_seconds"] = round(time.perf_counter() - started, 1)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\nSource videos were only read. Real and synthetic events are stored in separate datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
