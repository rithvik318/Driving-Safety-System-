"""Risk-engine integration test. Three clearly separated parts:

  PART A  SYNTHETIC rule stress test (always runs, no models). Constructed detections go through
          the real IoUTracker; DriverState objects are built directly. These are NOT observations.
  PART B  REAL front-camera videos: YOLO26n -> IoUTracker (10 fps) -> RiskEngine with NO driver
          input (road-only), for every video under --source.
  PART C  (optional, --driver-video) REAL driver-camera signals from the DriverPerceptionPipeline,
          PAIRED WITH the front-camera videos by elapsed time. The two cameras were NOT recorded
          together, so this only exercises the combination rules on real signals; it is not a
          record of what happened on any drive.

Usage (from the project root):
    python scripts/test_risk_engine.py                                  # part A only
    python scripts/test_risk_engine.py --source "<...>/frontcamera"     # parts A + B
    python scripts/test_risk_engine.py --source "<...>/frontcamera" --driver-video "<...>/drowsy_driver/video.mp4"

Risk levels are decision-support states from prototype rules; counts are not collision labels or
ground-truth safety labels. risk_score is a rule-based prototype score, not a probability.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python scripts/...`

LEVEL_NAMES = ("SAFE", "CAUTION", "HIGH", "CRITICAL")
W, H = 1920, 1080


# =========================================================================== PART A: SYNTHETIC


def _driver(t, drowsy="LOW", activity="NORMAL", hand="BOTH_HANDS", dist=0.0, yaw=0.0):
    from app.driver.models import (DriverActivity, DriverFrameObservation, DriverState, DriverTemporalState,
                                   DrowsinessLevel, FaceStatus, HandState, HandStateResult)

    obs = DriverFrameObservation(timestamp=t, face_status=FaceStatus.OK, face_detected=True, landmarks_available=True,
                                 head_yaw=yaw, head_pitch=0.0)
    temp = DriverTemporalState(drowsiness_level=DrowsinessLevel(drowsy),
                               drowsiness_score={"LOW": 0.1, "HIGH": 0.6, "CRITICAL": 0.9}[drowsy], distraction_duration=dist)
    return DriverState(timestamp=t, observation=obs, temporal=temp, hand=HandStateResult(hand_state=HandState(hand)),
                       observation_quality=1.0, driver_activity=DriverActivity(activity))


def _det(cls, cx, cy, w, h, conf=0.85):
    from app.road.models import Detection

    return Detection(None, cls, 0, conf, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, "SYNTHETIC")


def synthetic_scenarios():
    """(name, frames at 10 fps, driver function or None). SYNTHETIC feature-level fixtures."""
    calm = lambda t: _driver(t)  # noqa: E731
    hands = lambda t: _driver(t, activity="HANDS_OFF_WHEEL", hand="NO_HANDS", dist=2.0 + t)  # noqa: E731
    drowsy = lambda t: _driver(t, drowsy="HIGH")  # noqa: E731
    one_hand = lambda t: _driver(t, hand="ONE_HAND")  # noqa: E731
    approach = [[_det("car", 960, 700, 200 * (1 + 0.08 * i), 150 * (1 + 0.08 * i))] for i in range(30)]
    static_car = [[_det("car", 1400 + (i % 2), 450, 80, 60)] for i in range(30)]
    ped_cross = [[_det("person", 480 + 22 * i, 750, 90, 260)] for i in range(30)]
    ped_side = [[_det("person", 1800, 500, 40, 110)] for _ in range(30)]
    dog = [[_det("dog", 820 + 15 * i, 900, 160, 110)] for i in range(30)]
    empty = [[] for _ in range(30)]
    return [
        ("no hazard, attentive driver", empty, calm),
        ("static distant vehicle", static_car, calm),
        ("approaching vehicle, attentive driver", approach, calm),
        ("pedestrian moving toward centre band", ped_cross, calm),
        ("pedestrian static far to the side", ped_side, calm),
        ("dog central and moving", dog, calm),
        ("drowsiness HIGH, no hazard", empty, drowsy),
        ("hands-off-wheel, no hazard", empty, hands),
        ("one hand on wheel, no hazard", empty, one_hand),
        ("approaching vehicle + hands-off-wheel", approach, hands),
        ("approaching vehicle + drowsiness HIGH", approach, drowsy),
        ("pedestrian toward centre + hands-off-wheel", ped_cross, hands),
        ("dog central + hands-off-wheel", dog, hands),
        ("static side pedestrian + hands-off-wheel", ped_side, hands),
    ]


def run_synthetic(risk_cfg, tracking_cfg) -> list[dict]:
    from app.risk import RiskEngine, RiskSnapshot
    from app.tracking import IoUTracker

    rows = []
    print("\n" + "=" * 100 + "\nPART A - SYNTHETIC rule stress test (constructed inputs; NOT real observations)\n" + "=" * 100)
    print(f"  {'scenario':<44} {'final':<9} {'peak':<9} {'raw':>5} {'smooth':>6}  hazard_type")
    for name, frames, drv in synthetic_scenarios():
        tracker, engine = IoUTracker(tracking_cfg), RiskEngine(risk_cfg)
        results = []
        for i, dets in enumerate(frames):
            t = round(i * 0.1, 6)
            results.append(engine.evaluate(RiskSnapshot(t, tuple(tracker.update(dets, t, i)), drv(t) if drv else None, W, H)))
        last = results[-1]
        peak = max(results, key=lambda a: (a.risk_level.rank, a.raw_risk_score))
        print(f"  {name:<44} {last.risk_level.value:<9} {peak.risk_level.value:<9} {last.raw_risk_score:5.0f} "
              f"{last.smoothed_risk_score:6.0f}  {last.hazard_type.value}")
        rows.append({"scenario": name, "final": last.to_dict(), "peak_level": peak.risk_level.value})
    example = next(r for r in rows if r["scenario"] == "approaching vehicle + hands-off-wheel")["final"]
    print("\n  Example (SYNTHETIC) - approaching vehicle + hands-off-wheel, last frame:")
    print(format_assessment(example, indent="    "))
    return rows


def format_assessment(d: dict, indent: str = "  ") -> str:
    lines = [f"{indent}timestamp={d['timestamp']}  risk_level={d['risk_level']} (raw {d['raw_risk_level']})  "
             f"raw_score={d['raw_risk_score']:.0f}  smoothed_score={d['smoothed_risk_score']:.0f}  "
             f"track_id={d['primary_track_id']}  hazard_type={d['hazard_type']}  alarm_recommended={d['alarm_recommended']}",
             f"{indent}factors:"]
    lines += [f"{indent}  - {f['name']}: {f['points']:+.1f}  ({f['detail']})" for f in d["contributing_factors"]]
    lines.append(f"{indent}reason: {d['reason']}")
    lines.append(f"{indent}evidence_quality={d['evidence_quality']:.2f}" + (f"  gates: {'; '.join(d['gates'])}" if d["gates"] else ""))
    return "\n".join(lines)


# =========================================================================== PART B/C: REAL


def run_driver_video(path: Path, sample_fps: float) -> list:
    """Real DriverState sequence from the driver pipeline (face model + trained hand model, loaded once)."""
    from app.config import load_driver_config
    from app.driver import DriverPerceptionPipeline, MediaPipeFaceProvider, TrainedHandStateProvider, UnknownHandStateProvider
    from app.sensors.video import FrameClock, FrameSampler, VideoReader

    cfg = load_driver_config()
    face = MediaPipeFaceProvider(cfg.face_model_path, cfg.min_face_confidence)
    hand = (TrainedHandStateProvider.from_checkpoint(cfg.hand_model_path, threshold=cfg.hand_confidence_threshold)
            if Path(cfg.hand_model_path).is_file() else UnknownHandStateProvider())
    pipeline = DriverPerceptionPipeline(face, cfg, hand_provider=hand)
    reader = VideoReader(path)
    info = reader.info()
    clock, sampler, states, i = FrameClock(info.fps), FrameSampler(sample_fps), [], -1
    try:
        while reader.grab():
            i += 1
            t = clock.timestamp(i, reader.position_seconds())
            if not sampler.should_process(t):
                continue
            ok, frame = reader.retrieve()
            if ok and frame is not None:
                states.append(pipeline.process(frame, t))
    finally:
        reader.release()
        pipeline.close()
    return states


def driver_at(states: list, t: float):
    """Latest driver state with timestamp <= t, re-stamped to t (PAIRING ONLY), or None past the end."""
    best = None
    for s in states:
        if s.timestamp <= t + 1e-9:
            best = s
        else:
            break
    if best is None or t > states[-1].timestamp + 0.2:
        return None
    return dataclasses.replace(best, timestamp=t)


def run_real(videos, detector, tracking_cfg, risk_cfg, sample_fps, driver_states=None, label="B", jsonl=None) -> dict:
    from app.risk import RiskEngine, RiskSnapshot
    from app.road.runner import run_video
    from app.sensors.video import VideoReader
    from app.tracking import IoUTracker

    title = ("PART B - REAL front-camera videos, road-only (no driver input)" if driver_states is None else
             "PART C - REAL driver signals PAIRED with REAL front-camera videos (NOT synchronised recordings)")
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)
    agg = {"levels": Counter(), "raw_levels": Counter(), "hazards": Counter(), "factors": Counter(), "scores": [],
           "observations": 0, "examples": [], "per_video": [], "alarms": 0, "gates": Counter()}
    for path in videos:
        tracker, engine = IoUTracker(tracking_cfg), RiskEngine(risk_cfg)
        vid = {"video": path.name, "folder": path.parent.name, "levels": Counter(), "max_raw": 0.0, "max_level": "SAFE"}
        reader = VideoReader(path)
        info = reader.info()
        state = {"prev": None}

        def on_result(result, frame, vid=vid, tracker=tracker, engine=engine, info=info, state=state):
            tracks = tracker.update(result.detections if result.status.value == "OK" else (), result.timestamp, result.frame_index)
            d = driver_at(driver_states, result.timestamp) if driver_states else None
            h, w = frame.shape[:2]
            a = engine.evaluate(RiskSnapshot(result.timestamp, tuple(tracks), d, w, h))
            dd = a.to_dict()
            agg["observations"] += 1
            agg["levels"][a.risk_level.value] += 1
            agg["raw_levels"][a.raw_risk_level.value] += 1
            agg["hazards"][a.hazard_type.value] += 1
            agg["scores"].append((a.raw_risk_score, a.smoothed_risk_score))
            agg["alarms"] += a.alarm_recommended
            for g in a.gates:
                agg["gates"][g] += 1
            for f in a.contributing_factors:
                agg["factors"][f.name] += 1
            vid["levels"][a.risk_level.value] += 1
            vid["max_raw"] = max(vid["max_raw"], a.raw_risk_score)
            if LEVEL_NAMES.index(a.risk_level.value) > LEVEL_NAMES.index(vid["max_level"]):
                vid["max_level"] = a.risk_level.value
            if state["prev"] != a.risk_level.value:  # print level changes only (compact)
                print(f"  {path.name} t={a.timestamp:5.2f}s  {state['prev'] or '-':>8} -> {a.risk_level.value:<8} "
                      f"raw {a.raw_risk_score:4.0f} smooth {a.smoothed_risk_score:4.0f}  {a.hazard_type.value:<24} {a.reason}")
                state["prev"] = a.risk_level.value
            agg["examples"].append({"file": path.name, "folder": path.parent.name, **dd})
            if jsonl:
                jsonl.write(json.dumps({"part": label, "file": path.name, "folder": path.parent.name, **dd}) + "\n")

        try:
            run_video(reader, detector, sample_fps, keep_results=False, on_result=on_result)
        finally:
            reader.release()
        agg["per_video"].append(vid)
    return agg


def summarize_real(agg: dict) -> str:
    n = agg["observations"]
    if not n:
        return "  no observations"
    raw = [r for r, _ in agg["scores"]]
    sm = [s for _, s in agg["scores"]]
    pct = lambda c: f"{c} ({100 * c / n:.0f}%)"  # noqa: E731
    lines = [f"  Observations (sampled frames): {n}",
             "  Risk level (after hysteresis): " + ", ".join(f"{lv} {pct(agg['levels'][lv])}" for lv in LEVEL_NAMES),
             "  Raw level (before hysteresis): " + ", ".join(f"{lv} {pct(agg['raw_levels'][lv])}" for lv in LEVEL_NAMES),
             f"  raw_risk_score: min {min(raw):.0f}, median {statistics.median(raw):.0f}, mean {statistics.fmean(raw):.1f}, max {max(raw):.0f}",
             f"  smoothed_risk_score: min {min(sm):.0f}, median {statistics.median(sm):.0f}, mean {statistics.fmean(sm):.1f}, max {max(sm):.0f}",
             f"  alarm_recommended: {agg['alarms']} observation(s)",
             "  Hazard types: " + ", ".join(f"{k} {v}" for k, v in agg["hazards"].most_common()),
             "  Factors seen (observations): " + ", ".join(f"{k} {v}" for k, v in agg["factors"].most_common()),
             "  Gates applied: " + (", ".join(f"'{k}' {v}" for k, v in agg["gates"].most_common()) or "none"),
             "  Per video (max level / max raw score / level counts):"]
    for v in agg["per_video"]:
        lines.append(f"    {v['folder']:<13} {v['video']:<28} {v['max_level']:<9} {v['max_raw']:5.0f}  "
                     + ", ".join(f"{k} {c}" for k, c in sorted(v["levels"].items(), key=lambda x: LEVEL_NAMES.index(x[0]))))
    lines.append("  These are decision-support outputs of prototype rules, NOT collision or ground-truth safety labels.")
    return "\n".join(lines)


def top_examples(agg: dict, k: int = 3) -> list[dict]:
    seen, out = set(), []
    for e in sorted(agg["examples"], key=lambda e: (-e["raw_risk_score"], e["file"], e["timestamp"])):
        key = (e["file"], e["primary_track_id"], e["hazard_type"])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
        if len(out) == k:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    from app.config import configure_logging, load_risk_config, load_road_config, load_settings, load_tracking_config

    try:
        settings, rcfg, tcfg, kcfg = load_settings(), load_road_config(), load_tracking_config(), load_risk_config()
    except ValueError as exc:
        print(f"ERROR: invalid configuration: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", type=Path, help="Front-camera video or folder (repeatable) for part B.")
    p.add_argument("--driver-video", type=Path, help="Real driver-camera video for part C (paired, not synchronised).")
    p.add_argument("--sample-fps", type=float, default=10.0, help="Frames/s sent to YOLO and tracker (default 10).")
    p.add_argument("--driver-sample-fps", type=float, default=10.0)
    p.add_argument("--jsonl", type=Path, help="Write every assessment (parts B/C) as JSON lines.")
    p.add_argument("--summary-json", type=Path, help="Write the aggregated results as JSON.")
    args = p.parse_args(argv)

    summary = {"part_a_synthetic": run_synthetic(kcfg, tcfg), "risk_config": dataclasses.asdict(kcfg)}
    if args.source:
        from app.road import RoadModelError, YoloRoadDetector
        from app.road.runner import VIDEO_SUFFIXES, list_media

        videos = []
        for src in args.source:
            if not src.exists():
                print(f"ERROR: not found: {src}", file=sys.stderr)
                return 2
            videos += [m for m in list_media(src) if m.suffix.lower() in VIDEO_SUFFIXES]
        if not videos:
            print("ERROR: no videos in --source", file=sys.stderr)
            return 2
        try:
            detector = YoloRoadDetector(Path(rcfg.model_path), rcfg.confidence_threshold, rcfg.image_size, rcfg.device, rcfg.classes)
        except RoadModelError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if args.jsonl:
            args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl = args.jsonl.open("w", encoding="utf-8") if args.jsonl else None
        try:
            agg_b = run_real(videos, detector, tcfg, kcfg, args.sample_fps, None, "B", jsonl)
            print("\nPART B summary (REAL front camera, road-only):\n" + summarize_real(agg_b))
            print("\n  Highest-scoring examples (REAL, road-only):")
            for e in top_examples(agg_b):
                print(f"  [{e['folder']}/{e['file']}]\n" + format_assessment(e, "    "))
            summary["part_b_real_road_only"] = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in agg_b.items()
                                                if k not in ("examples",)} | {"top_examples": top_examples(agg_b, 5)}
            if args.driver_video:
                if not args.driver_video.is_file():
                    print(f"ERROR: driver video not found: {args.driver_video}", file=sys.stderr)
                    return 2
                states = run_driver_video(args.driver_video, args.driver_sample_fps)
                levels = Counter(s.temporal.drowsiness_level.value for s in states)
                acts = Counter(s.driver_activity.value for s in states)
                print(f"\n  Driver video {args.driver_video.name}: {len(states)} DriverStates over "
                      f"{states[-1].timestamp:.1f} s; drowsiness levels {dict(levels)}; driver_activity {dict(acts)}")
                agg_c = run_real(videos, detector, tcfg, kcfg, args.sample_fps, states, "C", jsonl)
                print("\nPART C summary (REAL signals, PAIRED, NOT synchronised):\n" + summarize_real(agg_c))
                print("\n  Highest-scoring examples (PAIRED, not synchronised):")
                for e in top_examples(agg_c):
                    print(f"  [{e['folder']}/{e['file']}]\n" + format_assessment(e, "    "))
                summary["part_c_paired_not_synchronised"] = {
                    k: (dict(v) if isinstance(v, Counter) else v) for k, v in agg_c.items() if k not in ("examples",)
                } | {"top_examples": top_examples(agg_c, 5), "driver_video": args.driver_video.name,
                     "driver_drowsiness_levels": dict(levels), "driver_activity": dict(acts)}
        finally:
            if jsonl:
                jsonl.close()
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\nrisk_score is a rule-based prototype score (0-100), not a probability of collision. "
          "Levels are decision-support states, not validated safety classifications.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
