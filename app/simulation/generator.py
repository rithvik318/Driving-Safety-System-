"""Feature-level synthetic scenario generator.

For every scenario instance (scenario x variant) at a fixed synthetic cadence (10 Hz):

    constructed detection (image space)  -> EXISTING IoUTracker
    scripted driver signals (per frame)  -> EXISTING driver temporal logic (DriverPerceptionPipeline.process
                                            with a scripted observation/hand source instead of a camera)
    tracks + DriverState                 -> EXISTING RiskEngine.evaluate (unmodified, default RiskConfig)
    assessment + DriverState             -> EXISTING EventRecorder (data_source=SYNTHETIC, no evidence)

and one row per step. Nothing here changes the engine; expectations are compared with its output and
mismatches are reported, never corrected. No video, no images, no real GPS. Deterministic for a seed.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config.driver_config import DriverConfig
from app.config.event_config import EventConfig
from app.config.risk_config import RiskConfig
from app.config.tracking_config import TrackingConfig
from app.driver.models import (DriverActivity, DriverFrameObservation, FaceStatus, HandState, HandStateResult)
from app.driver.pipeline import DriverPerceptionPipeline
from app.events.models import DataSource
from app.events.recorder import EventRecorder
from app.events.schema import REAL_SOURCES
from app.risk import RiskEngine, RiskSnapshot
from app.risk import rules
from app.risk.smoothing import HeadAwayTimer
from app.road.models import Detection
from app.tracking import IoUTracker
from app.simulation.models import (GENERATOR_VERSION, LOCATION_SOURCE, NOT_A_REPLACEMENT, PROVENANCE_POLICY,
                                   REQUESTED_NAME_MAPPING, SIM_FIELD_NAMES, SIM_SCHEMA_VERSION, SYNTHETIC,
                                   InstanceResult, SimulationConfig, validate_config, validate_sim_row)
from app.simulation.scenarios import SCENARIOS, ObjectStep, Reference, ScenarioSpec

LEVELS = ("SAFE", "CAUTION", "HIGH", "CRITICAL")
CENTRAL_BAND = (0.3, 0.7)  # |x - 0.5| <= 0.2, the risk engine's default corridor_half_width
ROW_NOTES = json.dumps([
    "SYNTHETIC scenario step: constructed inputs, not an observation",
    "engine fields computed by the existing tracker / driver logic / RiskEngine / EventRecorder from synthetic inputs",
    "road values are image-space (pixels, pixels/s); no distance, physical speed or TTC",
    "timestamp, synthetic_datetime and GPS are synthetic reference values, not a capture time or collection location",
])


# ------------------------------------------------------------------------------------ real reference


@dataclass
class RealReference:
    """Read-only view of the real event dataset, used only for parent links and value ranges."""

    path: Path
    available: bool
    sha256: str | None
    events: list = field(default_factory=list)  # LOCAL_REAL / PUBLIC records only
    run_ids: tuple = ()
    candidates: dict = field(default_factory=dict)  # kind -> [event dict]

    @property
    def ids(self) -> set[str]:
        return {e["event_id"] for e in self.events}

    def run_of(self, event_id: str) -> str:
        return next(e["run_id"] for e in self.events if e["event_id"] == event_id)


VEHICLE_LABELS = ("car", "truck", "bus")


def _norm(e: dict) -> dict:
    w, h = e.get("image_width"), e.get("image_height")
    ok = bool(w and h and e.get("bbox_area") is not None and e.get("center_x") is not None)
    return {"area_frac": e["bbox_area"] / (w * h) if ok else None, "cx_norm": e["center_x"] / w if ok else None}


def load_real_reference(path: Path) -> RealReference:
    path = Path(path)
    if not path.is_file():
        return RealReference(path, False, None)
    raw = path.read_bytes()  # read-only
    events = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    events = [e for e in events if e.get("data_source") in REAL_SOURCES]
    ref = RealReference(path, True, hashlib.sha256(raw).hexdigest(), events, tuple(sorted({e["run_id"] for e in events})))
    if len(ref.run_ids) > 1:  # parent ids are only unique within a run: keep the first run for linkage
        ref.events = [e for e in events if e["run_id"] == ref.run_ids[0]]
    road = [e for e in ref.events if e["camera"] == "front" and _norm(e)["area_frac"] is not None and e.get("hazard_class")]
    cand = {
        "vehicle": [e for e in road if e["hazard_type"] == "APPROACHING_VEHICLE" and e["hazard_class"] in VEHICLE_LABELS],
        "vehicle_presence": [e for e in road if e["hazard_type"] == "ROAD_USER_PRESENT" and e["hazard_class"] in VEHICLE_LABELS],
        "person": [e for e in road if e["hazard_class"] == "person" and _norm(e)["area_frac"] >= 0.005],
        "person_distant": [e for e in road if e["hazard_class"] == "person" and _norm(e)["area_frac"] < 0.005],
        "animal": [e for e in road if e["hazard_type"] == "ANIMAL_HAZARD"],
        "drowsy": [e for e in ref.events if e["camera"] == "driver" and e.get("drowsiness_level") in ("HIGH", "CRITICAL")
                   and e.get("drowsiness_score") is not None],
    }
    if not cand["vehicle_presence"]:
        cand["vehicle_presence"] = cand["vehicle"]
    ref.candidates = {k: sorted(v, key=lambda e: e["event_id"]) for k, v in cand.items()}
    return ref


def reference_ranges(ref: RealReference) -> dict:
    """min / median / max of the real values that seed the constructed parameters (per parent kind)."""
    def stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return {"n": len(vals), "min": round(min(vals), 4), "median": round(statistics.median(vals), 4), "max": round(max(vals), 4)}

    out = {}
    for kind, evs in ref.candidates.items():
        if kind == "drowsy":
            out[kind] = {"drowsiness_score": stats([e["drowsiness_score"] for e in evs])}
            continue
        out[kind] = {
            "area_fraction": stats([_norm(e)["area_frac"] for e in evs]),
            "center_x_norm": stats([_norm(e)["cx_norm"] for e in evs]),
            "hazard_confidence": stats([e.get("hazard_confidence") for e in evs]),
            "box_growth_per_second": stats([e.get("box_growth_per_second") for e in evs]),
            "lateral_toward_center_fw_per_s": stats([e.get("lateral_toward_center_fw_per_s") for e in evs]),
        }
    return out


def to_reference(e: dict | None) -> Reference:
    if e is None:
        return Reference()
    n = _norm(e)
    return Reference(e["event_id"], e.get("hazard_class"), n["area_frac"], n["cx_norm"], e.get("hazard_confidence"),
                     e.get("box_growth_per_second"), e.get("lateral_toward_center_fw_per_s"), e.get("drowsiness_score"))


# ------------------------------------------------------------------------------------ driver input


class _NoFaceProvider:
    """Placeholder: the scripted pipeline never calls a face model."""

    def process(self, frame, timestamp):  # pragma: no cover - never called
        raise RuntimeError("scripted pipeline has no face provider")


class ScriptedHandProvider:
    name = "synthetic_script"

    def __init__(self):
        self.next = HandStateResult()

    def predict(self, frame) -> HandStateResult:
        return self.next

    def reset(self) -> None:
        self.next = HandStateResult()


class ScriptedDriverPipeline(DriverPerceptionPipeline):
    """The existing DriverPerceptionPipeline with the camera replaced by scripted per-frame signals.

    Only observe() (face -> head pose / eye openness) and the hand provider are substituted; eye-closure
    tracking, drowsiness scoring, state durations and activity inference run unchanged."""

    def __init__(self, config: DriverConfig):
        super().__init__(_NoFaceProvider(), config, ScriptedHandProvider())
        self._step = None

    def observe(self, frame, timestamp: float) -> DriverFrameObservation:
        s = self._step
        left, right = s.eye_openness + 0.005, s.eye_openness - 0.005
        mean = (left + right) / 2
        return DriverFrameObservation(timestamp=timestamp, face_status=FaceStatus.OK, face_detected=True, landmarks_available=True,
                                      head_yaw=s.head_yaw, head_pitch=s.head_pitch, head_roll=0.0, left_eye_openness=left,
                                      right_eye_openness=right, mean_eye_openness=mean,
                                      eyes_closed=mean < self.config.eye_closure_threshold)

    def feed(self, step, timestamp: float):
        self._step = step
        self.hand_provider.next = HandStateResult(HandState(step.hand_state), DriverActivity.UNKNOWN, step.hand_confidence,
                                                  ScriptedHandProvider.name)
        return self.process(None, timestamp)


# ------------------------------------------------------------------------------------ helpers


def _box(o: ObjectStep, w: int, h: int) -> tuple[float, float, float, float]:
    area = o.area_frac * w * h
    bw = math.sqrt(area * o.aspect)
    bh = area / bw
    cx = o.cx_norm * w
    y2 = min(o.bottom_norm * h, h - 1.0)
    return cx - bw / 2, y2 - bh, cx + bw / 2, y2


def trajectory_overlap(o: ObjectStep | None) -> float | None:
    """Design ground truth: share of the constructed box width inside the central image band."""
    if o is None:
        return None
    half = math.sqrt(o.area_frac * o.aspect * (9 / 16)) / 2  # half box width / frame width (16:9 frame)
    x1, x2 = o.cx_norm - half, o.cx_norm + half
    inside = max(0.0, min(x2, CENTRAL_BAND[1]) - max(x1, CENTRAL_BAND[0]))
    return round(min(1.0, inside / (x2 - x1)), 4) if x2 > x1 else 0.0


def _r(v, n=4):
    return None if v is None else round(float(v), n)


def _gps_trace(rng: random.Random, cfg: SimulationConfig):
    lat0 = cfg.base_lat + rng.uniform(-cfg.location_jitter_deg, cfg.location_jitter_deg)
    lon0 = cfg.base_lon + rng.uniform(-cfg.location_jitter_deg, cfg.location_jitter_deg)
    heading = math.radians(rng.uniform(0.0, 360.0))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))

    def at(t: float) -> tuple[float, float]:
        d = cfg.trace_step_m_per_s * t
        return round(lat0 + d * math.cos(heading) / m_per_deg_lat, 7), round(lon0 + d * math.sin(heading) / m_per_deg_lon, 7)
    return at


def _object_track(tracks):
    live = [tr for tr in tracks if not tr.ended]
    return max(live, key=lambda tr: (tr.detection_count, -tr.track_id)) if live else None


def evaluate_expectation(spec: ScenarioSpec, levels: list[str]) -> tuple[bool, str | None]:
    peak = max(levels, key=LEVELS.index)
    problems = []
    if peak not in spec.expected_levels:
        problems.append(f"peak {peak} not in expected {'/'.join(spec.expected_levels)}")
    if spec.must_start_safe:
        if levels[0] != "SAFE":
            problems.append(f"started at {levels[0]}, expected SAFE")
        elif levels.index(peak) == 0:
            problems.append("peak reached at the first step (no escalation)")
    return (not problems), ("; ".join(problems) or None)


# ------------------------------------------------------------------------------------ generator


@dataclass
class GenerationResult:
    rows: list
    instances: list  # InstanceResult
    reference: RealReference
    config: SimulationConfig


class SyntheticScenarioGenerator:
    def __init__(self, config: SimulationConfig | None = None, scenarios: tuple[ScenarioSpec, ...] = SCENARIOS):
        self.cfg = config or SimulationConfig()
        validate_config(self.cfg)
        self.scenarios = scenarios
        # default (unmodified) configurations of the existing components: deterministic, independent of .env
        self.risk_cfg = RiskConfig()
        self.tracking_cfg = TrackingConfig()
        self.driver_cfg = DriverConfig(face_model_path=Path("unused-synthetic"))
        self.event_cfg = EventConfig(evidence_mode="none")

    def generate(self) -> GenerationResult:
        cfg = self.cfg
        ref = load_real_reference(cfg.real_events_path)
        rows, instances = [], []
        idx = 0
        for spec in self.scenarios:
            for variant in range(1, cfg.variants_per_scenario + 1):
                inst_rows, inst = self._instance(spec, variant, idx, ref)
                rows += inst_rows
                instances.append(inst)
                idx += 1
        for i, row in enumerate(rows, 1):
            row["event_id"] = f"sim_{i:06d}"
            errors = validate_sim_row(row)
            if errors:
                raise AssertionError(f"generated row {row['synthetic_id']} invalid: {errors}")
        return GenerationResult(rows, instances, ref, cfg)

    def _instance(self, spec: ScenarioSpec, variant: int, idx: int, ref: RealReference):
        cfg, dt = self.cfg, self.cfg.dt
        rng = random.Random(f"{cfg.seed}:{spec.name}:{variant}")  # str seeds are deterministic across runs
        road_parent = rng.choice(ref.candidates[spec.road_parent]) if spec.road_parent and ref.candidates.get(spec.road_parent) else None
        drv_parent = rng.choice(ref.candidates[spec.driver_parent]) if spec.driver_parent and ref.candidates.get(spec.driver_parent) else None
        duration = round(rng.uniform(*cfg.duration_range_seconds), 1)
        n = int(round(duration * cfg.cadence_hz))
        n_pre = int(round(cfg.driver_preroll_seconds * cfg.cadence_hz))
        objects = spec.road(rng, to_reference(road_parent), n, dt)
        driver_script = spec.driver(rng, to_reference(drv_parent), n_pre, n, dt)
        gps_at = _gps_trace(rng, cfg)
        t0 = datetime.fromisoformat(cfg.base_datetime) + timedelta(seconds=idx * cfg.instance_spacing_seconds)
        W, H = cfg.image_width, cfg.image_height
        session = f"{spec.name}__v{variant:02d}"
        parent_run = None
        if road_parent or drv_parent:
            parent_run = (road_parent or drv_parent)["run_id"]

        pipeline = ScriptedDriverPipeline(self.driver_cfg)
        for j in range(n_pre):  # synthetic driver history before t=0 (not emitted)
            pipeline.feed(driver_script[j], round((j - n_pre) * dt, 6))
        tracker = IoUTracker(self.tracking_cfg)
        engine = RiskEngine(self.risk_cfg)
        head_timer = HeadAwayTimer()
        recorder = EventRecorder(self.event_cfg, run_id=cfg.run_id)
        recorder.start_session(session, None, DataSource.SYNTHETIC, "front+driver")

        rows, levels, event_counts = [], [], Counter()
        hz_key, hz_start = None, None
        for i in range(n):
            t = round(i * dt, 6)
            o = objects[i]
            dets = []
            if o is not None:
                x1, y1, x2, y2 = _box(o, W, H)
                dets = [Detection(t, o.class_name, 0, o.confidence, x1, y1, x2, y2, "synthetic_scenario", frame_index=i)]
            tracks = tracker.update(dets, t, i)
            ds = pipeline.feed(driver_script[n_pre + i], t)
            a = engine.evaluate(RiskSnapshot(t, tuple(tracks), ds, W, H))
            head_s = head_timer.update(ds.timestamp, rules.head_is_away(ds, self.risk_cfg))
            evs = recorder.on_risk(a, tracks, ds, i, (W, H)) + recorder.on_driver(ds, i)
            for e in evs:
                event_counts[e.event_type] += 1

            tr = _object_track(tracks)
            hz = next((h for h in a.hazards if tr is not None and h.track_id == tr.track_id), None)
            if hz is not None and hz.qualifies:
                key = (hz.hazard_type.value, hz.track_id)
                if key != hz_key:
                    hz_key, hz_start = key, t
                persistence = round(t - hz_start, 4)
            else:
                hz_key, hz_start, persistence = None, None, 0.0
            lat, lon = gps_at(t)
            box = _box(o, W, H) if o is not None else None
            dsd = ds.to_dict()
            level = a.risk_level.value
            levels.append(level)
            rows.append({
                "schema_version": SIM_SCHEMA_VERSION, "event_id": "", "synthetic_id": f"SYN-{spec.name}-v{variant:02d}-s{i:03d}",
                "run_id": cfg.run_id, "session_id": session, "scenario_name": spec.name, "scenario_variant": variant,
                "scenario_description": spec.description, "synthetic_generation_reason": spec.reason,
                "parent_real_event_id": road_parent["event_id"] if road_parent else None,
                "parent_real_driver_event_id": drv_parent["event_id"] if drv_parent else None,
                "parent_real_run_id": parent_run, "data_source": SYNTHETIC, "observation_type": SYNTHETIC, "camera": "front+driver",
                "timestamp": t, "timestamp_source": SYNTHETIC,
                "synthetic_datetime": (t0 + timedelta(seconds=t)).isoformat(timespec="milliseconds"),
                "frame_index": i, "gps_lat": lat, "gps_lon": lon, "gps_source": SYNTHETIC, "location_source": LOCATION_SOURCE,
                "driver_hand_state": dsd["hand_state"], "driver_hand_confidence": _r(dsd["hand_state_confidence"]),
                "face_status": dsd["face_status"], "head_yaw": _r(dsd["head_yaw"], 2), "head_pitch": _r(dsd["head_pitch"], 2),
                "eye_closure_ratio": _r(dsd["eye_closure_ratio"]), "eye_closed_duration": _r(dsd["eye_closed_duration"]),
                "drowsiness_score": _r(dsd["drowsiness_score"]), "drowsiness_level": dsd["drowsiness_level"],
                "driver_activity": dsd["driver_activity"], "distraction_duration": _r(dsd["distraction_duration"]),
                "manual_state_duration": _r(dsd["manual_state_duration"]), "head_away_duration": round(head_s, 4),
                "image_width": W, "image_height": H,
                "hazard_class": o.class_name if o else None, "hazard_confidence": o.confidence if o else None,
                "center_x": _r((box[0] + box[2]) / 2, 2) if box else None, "center_y": _r((box[1] + box[3]) / 2, 2) if box else None,
                "bbox_width": _r(box[2] - box[0], 2) if box else None, "bbox_height": _r(box[3] - box[1], 2) if box else None,
                "bbox_area": _r((box[2] - box[0]) * (box[3] - box[1]), 1) if box else None,
                "primary_track_id": tr.track_id if tr else None, "hazard_track_class": tr.class_name if tr else None,
                "track_persistence_seconds": _r(tr.persistence_seconds) if tr else None,
                "box_growth_per_second": _r(hz.growth_per_second) if hz else None,
                "lateral_toward_center_fw_per_s": _r(hz.lateral_toward_corridor) if hz else None,
                "trajectory_overlap": trajectory_overlap(o), "hazard_persistence": persistence,
                "risk_level": level, "raw_risk_level": a.raw_risk_level.value, "raw_risk_score": round(a.raw_risk_score, 3),
                "smoothed_risk_score": round(a.smoothed_risk_score, 3), "risk_score": round(a.risk_score, 3),
                "hazard_type": a.hazard_type.value, "risk_reason": a.reason,
                "risk_factors": json.dumps([f.to_dict() for f in a.contributing_factors]), "risk_gates": json.dumps(list(a.gates)),
                "evidence_quality": round(a.evidence_quality, 4), "alarm_recommended": bool(a.alarm_recommended),
                "alarm_triggered": False,
                "event_type": evs[0].event_type if evs else None, "transition": evs[0].transition if evs else None,
                "events_emitted": json.dumps([f"{e.event_type} {e.transition or ''}".strip() for e in evs]),
                "expected_behavior": spec.expected_text, "expected_peak_levels": json.dumps(list(spec.expected_levels)),
                "actual_risk_level": level, "instance_peak_risk_level": "", "expectation_met": False, "notes": ROW_NOTES,
            })
        recorder.finish_session()
        peak = max(levels, key=LEVELS.index)
        met, mismatch = evaluate_expectation(spec, levels)
        for row in rows:
            row["instance_peak_risk_level"] = peak
            row["expectation_met"] = met
        assert list(rows[0]) == list(SIM_FIELD_NAMES), "row field order must follow the schema"
        inst = InstanceResult(spec.name, variant, session, n, duration, spec.expected_text, spec.expected_levels, peak,
                              levels[0], levels[-1], dict(Counter(levels)), sum(r["alarm_recommended"] for r in rows), met,
                              mismatch, road_parent["event_id"] if road_parent else None,
                              drv_parent["event_id"] if drv_parent else None, dict(event_counts))
        return rows, inst


# ------------------------------------------------------------------------------------ manifest


def build_manifest(result: GenerationResult, files: dict | None = None, generated_at: str | None = None) -> dict:
    rows, cfg, ref = result.rows, result.config, result.reference
    parents = sorted({r[k] for r in rows for k in ("parent_real_event_id", "parent_real_driver_event_id") if r[k]})
    linked = sum(1 for r in rows if r["parent_real_event_id"] or r["parent_real_driver_event_id"])
    cfg_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    if isinstance(cfg.real_events_path, Path):
        try:
            cfg_dict["real_events_path"] = cfg.real_events_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            cfg_dict["real_events_path"] = cfg.real_events_path.name
    return {
        "dataset": "data/simulated",
        "data_source": SYNTHETIC,
        "generation_timestamp": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator_version": GENERATOR_VERSION,
        "schema_version": SIM_SCHEMA_VERSION,
        "random_seed": cfg.seed,
        "total_records": len(rows),
        "scenario_counts": dict(Counter(r["scenario_name"] for r in rows)),
        "scenario_instances": len(result.instances),
        "variants_per_scenario": cfg.variants_per_scenario,
        "cadence_hz": cfg.cadence_hz,
        "parent_real_event_count": len(parents),
        "parent_real_event_ids": parents,
        "parent_linked_records": linked,
        "synthetic_only_records_count": len(rows) - linked,
        "fields_generated": list(SIM_FIELD_NAMES),
        "requested_name_mapping": REQUESTED_NAME_MAPPING,
        "provenance_policy": PROVENANCE_POLICY,
        "statement": NOT_A_REPLACEMENT,
        "location_policy": (f"Synthetic GPS around a generic Indian urban-road reference point ({cfg.base_lat}, {cfg.base_lon}) "
                            f"with +-{cfg.location_jitter_deg} deg start perturbation per instance; gps_source=SYNTHETIC, "
                            "location_source=SYNTHETIC_REFERENCE. Not a collection location. The real dataset keeps GPS null."),
        "time_policy": "timestamp = seconds within the scenario instance at a fixed synthetic cadence; synthetic_datetime on a "
                       "synthetic clock from base_datetime. timestamp_source=SYNTHETIC. Never used in LOCAL_REAL records.",
        "real_reference": {"path": cfg_dict["real_events_path"], "available": ref.available, "sha256": ref.sha256,
                           "events_used_as_reference_pool": len(ref.events), "run_ids": list(ref.run_ids),
                           "access": "read-only", "value_ranges": reference_ranges(ref)},
        "engine_configuration": "default RiskConfig / TrackingConfig / DriverConfig / EventConfig(evidence_mode=none); the "
                                "engine was not modified or tuned for this dataset",
        "config": cfg_dict,
        "validation": {
            "instances": len(result.instances),
            "expectation_met": sum(i.expectation_met for i in result.instances),
            "mismatches": [i.to_dict() for i in result.instances if not i.expectation_met],
        },
        "instances": [i.to_dict() for i in result.instances],
        "files": files or {},
    }
