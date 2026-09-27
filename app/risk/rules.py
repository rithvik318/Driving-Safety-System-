"""Transparent risk rules. Pure functions: no state, no images, no ML.

Road rules (per tracked object, image space only):
  gate      persistence >= min_persistence_seconds, detections >= min_detections, not coasting long
  factors   persistence  - how long the object has been observed
            centrality   - horizontal position in a central image band (a stand-in for "ahead",
                           not a calibrated path), full points only in the lower image region
            approach     - growth of ln(box area) per second over a short window
            trajectory   - horizontal motion toward (or across) the central band, frame-widths/s
            image_size   - box area as a fraction of the frame (bigger in the image, NOT a distance)
  qualifies vehicle: approach evidence
            person:  in / near the central band AND (trajectory toward or across it, or approach)
            animal:  central band or lower region AND moving (lateral motion or approach)
  A track that does not qualify is "presence only" and contributes at most presence_cap_points.
  Evidence quality (detector confidence, label changes) scales the track's points.

Driver rules (existing DriverState signals only):
  drowsiness_level HIGH / CRITICAL, driver_activity HANDS_OFF_WHEEL (inferred upstream from
  sustained NO_HANDS), head turned away for a sustained time. ONE_HAND scores nothing.
"""

from __future__ import annotations

import math

from app.config.risk_config import RiskConfig
from app.risk.models import HazardType, RiskFactor, RiskLevel, TrackHazard

KIND_TO_HAZARD = {"vehicle": HazardType.APPROACHING_VEHICLE, "person": HazardType.PEDESTRIAN_CONFLICT,
                  "animal": HazardType.ANIMAL_HAZARD}


def ramp(x: float, lo: float, hi: float) -> float:
    """0 at lo, 1 at hi, linear between, clamped."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def track_kind(class_name: str, cfg: RiskConfig) -> str | None:
    name = (class_name or "").lower()
    if name in cfg.vehicle_classes:
        return "vehicle"
    if name in cfg.person_classes:
        return "person"
    if name in cfg.animal_classes:
        return "animal"
    return None


def lateral_toward_corridor(center_x_norm: float, lateral_rate: float | None, cfg: RiskConfig) -> float | None:
    """Horizontal motion that brings the object toward the central band (frame-widths/s, > 0).

    Outside the band: the velocity component toward the image centre. Inside the band: the
    absolute horizontal speed (moving across the area ahead). None if motion is unknown."""
    if lateral_rate is None:
        return None
    offset = center_x_norm - 0.5
    if abs(offset) <= cfg.corridor_half_width:
        return abs(lateral_rate)
    return -lateral_rate if offset > 0 else lateral_rate


def track_quality(track, cfg: RiskConfig) -> tuple[float, list[str]]:
    """Multiplier in [min, 1] from detector confidence and label stability, with explanations."""
    notes = []
    conf_mult = max(cfg.min_confidence_multiplier, min(1.0, track.mean_confidence / cfg.confidence_reference))
    if conf_mult < 1.0:
        notes.append(f"mean detector confidence {track.mean_confidence:.2f} < {cfg.confidence_reference:.2f}")
    stability = max(cfg.min_stability_multiplier, 1.0 - cfg.label_change_penalty * track.class_changes)
    if track.class_changes:
        labels = "/".join(sorted(set(track.class_history)))
        notes.append(f"{track.class_changes} detector label change(s) ({labels})")
    return conf_mult * stability, notes


def _weights(kind: str, cfg: RiskConfig) -> dict[str, float]:
    return {k: getattr(cfg, f"{kind}_{k}_points") for k in ("persistence", "centrality", "approach", "size", "trajectory")}


def evaluate_track(track, growth: float | None, lateral_rate: float | None, width: int, height: int,
                   cfg: RiskConfig) -> TrackHazard | None:
    """Apply the road rules to one TrackedObject. None if the class is not a hazard class or the
    track does not pass the evidence gates."""
    kind = track_kind(track.class_name, cfg)
    if kind is None or width <= 0 or height <= 0:
        return None
    if track.persistence_seconds < cfg.min_persistence_seconds - 1e-9 or track.detection_count < cfg.min_detections:
        return None
    if track.missed_seconds > cfg.max_track_missed_seconds + 1e-9:
        return None

    w = _weights(kind, cfg)
    tid, label = track.track_id, f"{track.class_name} #{track.track_id}"
    cxn = track.center_x / width
    bottom = track.bbox_y2 / height
    area_frac = track.area / float(width * height)
    toward = lateral_toward_corridor(cxn, lateral_rate, cfg)
    factors: list[RiskFactor] = []
    groups: list[str] = []
    phrases: list[str] = []

    p = w["persistence"] * ramp(track.persistence_seconds, 0.0, cfg.persistence_full_seconds)
    factors.append(RiskFactor("persistence", p, "road",
                              f"{label} observed for {track.persistence_seconds:.1f} s ({track.detection_count} detections)", tid))

    offset = abs(cxn - 0.5)
    central = offset <= cfg.corridor_half_width
    near = offset <= cfg.wide_half_width
    lower = bottom >= cfg.lower_region_min
    position = 1.0 if central else 0.5 if near else 0.0
    if position:
        c = w["centrality"] * position * (1.0 if lower else 0.5)
        where = "in the central image band" if central else "near the central image band"
        where += ", lower image region" if lower else ", upper image region"
        factors.append(RiskFactor("centrality", c, "road", f"{label} {where} (x={cxn:.2f}, bottom={bottom:.2f} of frame)", tid))
        phrases.append(where.split(",")[0])
        if central and lower:
            groups.append("position")

    approaching = growth is not None and growth >= cfg.growth_min
    if approaching:
        a = w["approach"] * (0.4 + 0.6 * ramp(growth, cfg.growth_min, cfg.growth_full))
        grow_txt = f"image-space box area growing {100 * (math.exp(growth) - 1):.0f}%/s"
        factors.append(RiskFactor("approach", a, "road", f"{label} {grow_txt}", tid))
        phrases.insert(0, grow_txt)
    moving_in = toward is not None and toward >= cfg.lateral_min
    if moving_in:
        tr = w["trajectory"] * (0.4 + 0.6 * ramp(toward, cfg.lateral_min, cfg.lateral_full))
        what = "moving across the central band" if central else "moving toward the central band"
        factors.append(RiskFactor("trajectory", tr, "road", f"{label} {what} at {toward:.2f} frame-widths/s", tid))
        phrases.insert(1 if approaching else 0, f"{what} at {toward:.2f} frame-widths/s (image space)")
    if approaching or moving_in:
        groups.append("motion")

    s = w["size"] * ramp(area_frac, cfg.size_min, cfg.size_full)
    if s > 0:
        factors.append(RiskFactor("image_size", s, "road", f"{label} covers {100 * area_frac:.1f}% of the frame", tid))

    lateral_moving = lateral_rate is not None and abs(lateral_rate) >= cfg.lateral_min
    if kind == "vehicle":
        qualifies = approaching
    elif kind == "person":
        qualifies = near and (moving_in or approaching)
    else:
        qualifies = (near or lower) and (approaching or lateral_moving)

    total = sum(f.points for f in factors)
    quality, qnotes = track_quality(track, cfg)
    if quality < 1.0 and total > 0:
        factors.append(RiskFactor("evidence_quality", total * quality - total, "quality",
                                  f"{label} points x{quality:.2f}: " + "; ".join(qnotes), tid))
        total *= quality
    if not qualifies and total > cfg.presence_cap_points:
        factors.append(RiskFactor("presence_cap", cfg.presence_cap_points - total, "road",
                                  f"{label} has no approach/conflict evidence: capped at {cfg.presence_cap_points:g} points", tid))
        total = cfg.presence_cap_points

    return TrackHazard(
        track_id=tid, class_name=track.class_name, kind=kind,
        hazard_type=KIND_TO_HAZARD[kind] if qualifies else HazardType.ROAD_USER_PRESENT,
        qualifies=qualifies, points=total, factors=tuple(factors),
        evidence_groups=tuple(groups) if qualifies else (), growth_per_second=growth,
        lateral_toward_corridor=toward, center_x_norm=cxn, bottom_norm=bottom, area_fraction=area_frac, quality=quality,
        phrases=(f"observed {track.persistence_seconds:.1f} s, {track.detection_count} detections", *phrases),
    )


def driver_factors(driver, timestamp: float, head_away_seconds: float, cfg: RiskConfig) -> tuple[list[RiskFactor], list[str]]:
    """Driver contributions from an existing DriverState (or None). Returns (factors, notes)."""
    if driver is None:
        return [], ["no driver observation: driver factors not evaluated"]
    age = timestamp - driver.timestamp
    if age > cfg.driver_max_age_seconds:
        return [], [f"driver observation is {age:.1f} s old (> {cfg.driver_max_age_seconds:g} s): driver factors not evaluated"]
    factors, notes = [], []
    t, o = driver.temporal, driver.observation
    level = getattr(t.drowsiness_level, "value", str(t.drowsiness_level))
    score = f", score {t.drowsiness_score:.2f}" if t.drowsiness_score is not None else ""
    if level == "CRITICAL":
        factors.append(RiskFactor("drowsiness", cfg.drowsiness_critical_points, "driver", f"drowsiness level CRITICAL{score}"))
    elif level == "HIGH":
        factors.append(RiskFactor("drowsiness", cfg.drowsiness_high_points, "driver", f"drowsiness level HIGH{score}"))
    elif level == "UNKNOWN":
        notes.append("drowsiness UNKNOWN (not enough observed face time)")

    activity = getattr(driver.driver_activity, "value", str(driver.driver_activity))
    if activity == "HANDS_OFF_WHEEL":
        dur = t.distraction_duration
        dur_txt = f"NO_HANDS observed continuously {dur:.1f} s" if dur is not None else "sustained NO_HANDS"
        factors.append(RiskFactor("hands_off_wheel", cfg.hands_off_wheel_points, "driver",
                                  f"inferred HANDS_OFF_WHEEL ({dur_txt})"))
    hand = getattr(driver.hand.hand_state, "value", str(driver.hand.hand_state))
    if hand == "ONE_HAND":
        notes.append("ONE_HAND observed: not scored (one hand on the wheel is not treated as distraction)")

    if head_away_seconds >= cfg.head_away_min_seconds:
        yaw = f"yaw {o.head_yaw:+.0f}°" if o.head_yaw is not None else "yaw n/a"
        pitch = f"pitch {o.head_pitch:+.0f}°" if o.head_pitch is not None else "pitch n/a"
        factors.append(RiskFactor("head_away", cfg.head_away_points, "driver",
                                  f"head turned away for {head_away_seconds:.1f} s ({yaw}, {pitch}; relative head-pose signal)"))

    status = getattr(o.face_status, "value", str(o.face_status))
    if status != "OK":
        notes.append(f"face status {status}: eye/head evidence unavailable this frame")
    return factors, notes


def head_is_away(driver, cfg: RiskConfig) -> bool:
    o = driver.observation
    yaw_away = o.head_yaw is not None and abs(o.head_yaw) > cfg.head_away_yaw_degrees
    pitch_away = o.head_pitch is not None and o.head_pitch > cfg.head_away_pitch_degrees
    return bool(yaw_away or pitch_away)


def level_for_score(score: float, cfg: RiskConfig) -> RiskLevel:
    if score >= cfg.critical_min:
        return RiskLevel.CRITICAL
    if score >= cfg.high_min:
        return RiskLevel.HIGH
    if score >= cfg.caution_min:
        return RiskLevel.CAUTION
    return RiskLevel.SAFE


def apply_gates(level: RiskLevel, driver_names: list[str], primary: TrackHazard | None, road_points: float,
                cfg: RiskConfig) -> tuple[RiskLevel, list[str]]:
    """Level caps that make HIGH/CRITICAL require the kind of evidence their definitions state.

    HIGH     needs >= 2 independent evidence groups (each driver factor, road motion, road position).
    CRITICAL needs (a driver factor AND a qualifying road hazard with motion evidence) OR
             (road points >= strong_hazard_points with both road motion and position evidence).
    """
    gates = []
    road_groups = list(primary.evidence_groups) if primary is not None and primary.qualifies else []
    groups = set(driver_names) | {f"road_{g}" for g in road_groups}
    if level.rank >= RiskLevel.CRITICAL.rank:
        combined = bool(driver_names) and "motion" in road_groups
        strong_road = road_points >= cfg.strong_hazard_points and {"motion", "position"} <= set(road_groups)
        if not (combined or strong_road):
            level = RiskLevel.HIGH
            gates.append("CRITICAL needs a driver factor with a moving road hazard, or a very strong road hazard")
    if level.rank >= RiskLevel.HIGH.rank and len(groups) < 2:
        level = RiskLevel.CAUTION
        gates.append("HIGH needs at least 2 independent evidence groups")
    return level, gates


# ------------------------------------------------------------------------------ reason text


def _road_phrase(primary: TrackHazard) -> str:
    seen, *bits = primary.phrases or ("",)
    lead = f"persistent {primary.class_name} #{primary.track_id} ({seen})"
    return lead + (": " + ", ".join(bits) if bits else "")


def build_reason(hazard_type: HazardType, driver_list: list[RiskFactor], primary: TrackHazard | None,
                 other_hazards: int) -> str:
    """One sentence generated from the structured factors (templated, no LLM)."""
    driver_txt = " and ".join(f.detail for f in driver_list)
    driver_txt = driver_txt[:1].upper() + driver_txt[1:] if driver_txt else ""
    more = f" (+{other_hazards} other road hazard{'s' if other_hazards > 1 else ''})" if other_hazards else ""
    if hazard_type is HazardType.COMBINED_DRIVER_HAZARD and primary is not None:
        return f"{driver_txt} coincides with a {_road_phrase(primary)}{more}."
    if primary is not None and primary.qualifies:
        text = _road_phrase(primary)
        return text[:1].upper() + text[1:] + more + "."
    if driver_list:
        tail = ""
        if primary is not None:
            tail = f"; {primary.class_name} #{primary.track_id} visible without approach/conflict evidence"
        return f"{driver_txt}; no road hazard with approach/conflict evidence{tail}."
    if primary is not None:
        return (f"Persistent {primary.class_name} #{primary.track_id} is visible but shows no approach or conflict "
                "evidence in image space.")
    return "No persistent road hazard and no driver impairment signal."
