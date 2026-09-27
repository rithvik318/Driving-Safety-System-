"""The 12 designed scenarios: constructed road-object and driver-signal scripts + expectations.

Each scenario produces, per 10 Hz step:
  road   ObjectStep (a constructed detection in image space) or None (not detected this step)
  driver DriverStep (scripted per-frame head pose, eye openness and hand state)
The generator feeds them through the EXISTING tracker, driver temporal logic and risk engine.

Where a real LOCAL_REAL event is available as a parent, its measured values (box area fraction,
horizontal position, detector confidence, growth, lateral motion, drowsiness score) seed the
constructed parameters, clamped to the ranges the scenario is meant to test. Expectations are
written here BEFORE the engine runs and are never adjusted to its output.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

EYE_OPEN = 0.30  # eye-aspect ratio of an open eye (the driver config closes eyes below 0.20)
EYE_CLOSED = 0.10


@dataclass(frozen=True)
class ObjectStep:
    class_name: str
    cx_norm: float  # box centre x / frame width
    bottom_norm: float  # box bottom y / frame height
    area_frac: float  # box area / frame area
    aspect: float  # box width / height
    confidence: float


@dataclass(frozen=True)
class DriverStep:
    head_yaw: float
    head_pitch: float
    eye_openness: float
    hand_state: str  # BOTH_HANDS | ONE_HAND | NO_HANDS | UNKNOWN
    hand_confidence: float


@dataclass(frozen=True)
class Reference:
    """Values taken from a real parent event (normalised); None when there is no parent."""

    event_id: str | None = None
    class_name: str | None = None
    area_frac: float | None = None
    cx_norm: float | None = None
    confidence: float | None = None
    growth: float | None = None
    lateral: float | None = None
    drowsiness_score: float | None = None


RoadFn = Callable[[random.Random, Reference, int, float], list]
DriverFn = Callable[[random.Random, Reference, int, int, float], list]


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    reason: str
    road: RoadFn
    driver: DriverFn
    expected_levels: tuple[str, ...]  # acceptable PEAK risk levels of an instance
    expected_text: str
    road_parent: str | None  # "vehicle" | "person" | "animal" | "vehicle_presence" | "person_distant" | None
    driver_parent: str | None = None  # "drowsy" | None
    must_start_safe: bool = False  # escalation scenarios: first level SAFE, peak reached later


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _pick(ref_value: float | None, lo: float, hi: float, rng: random.Random) -> float:
    """Reference value clamped into [lo, hi]; uniform in [lo, hi] without a reference. Small jitter either way."""
    base = rng.uniform(lo, hi) if ref_value is None else clamp(ref_value, lo, hi)
    return clamp(base * rng.uniform(0.95, 1.05), lo, hi)


# ------------------------------------------------------------------------------------ driver scripts


def _face(rng: random.Random, n: int, dt: float, yaw0=0.0, pitch0=2.0) -> tuple[list, list, list]:
    """Attentive face: small head motion, normal blinks (0.1-0.25 s every 2.5-5 s)."""
    yaw, pitch, eye = [], [], []
    phase = rng.uniform(0, 2 * math.pi)
    next_blink, blink_end = rng.uniform(1.0, 4.0), -1.0
    for i in range(n):
        t = i * dt
        yaw.append(yaw0 + 4.0 * math.sin(0.6 * t + phase) + rng.gauss(0, 1.5))
        pitch.append(pitch0 + rng.gauss(0, 1.5))
        if t >= next_blink:
            blink_end = t + rng.uniform(0.1, 0.25)
            next_blink = t + rng.uniform(2.5, 5.0)
        eye.append(EYE_CLOSED if t < blink_end else EYE_OPEN + rng.gauss(0, 0.02))
    return yaw, pitch, eye


def _steps(yaw, pitch, eye, hands, confs) -> list[DriverStep]:
    return [DriverStep(round(y, 2), round(p, 2), round(e, 4), h, round(c, 3)) for y, p, e, h, c in zip(yaw, pitch, eye, hands, confs)]


def attentive_driver(hand: str = "BOTH_HANDS") -> DriverFn:
    def fn(rng, ref, n_pre, n, dt):
        yaw, pitch, eye = _face(rng, n_pre + n, dt)
        return _steps(yaw, pitch, eye, [hand] * (n_pre + n), [rng.uniform(0.75, 0.95) for _ in range(n_pre + n)])
    return fn


def hands_off_driver(rng, ref, n_pre, n, dt):
    """BOTH_HANDS during the pre-roll, NO_HANDS from 0-0.5 s into the scenario (the existing logic infers
    HANDS_OFF_WHEEL only after NO_HANDS has lasted 2 s)."""
    yaw, pitch, eye = _face(rng, n_pre + n, dt)
    start = n_pre + int(round(rng.uniform(0.0, 0.5) / dt))
    hands = ["BOTH_HANDS" if i < start else "NO_HANDS" for i in range(n_pre + n)]
    return _steps(yaw, pitch, eye, hands, [rng.uniform(0.75, 0.95) for _ in range(n_pre + n)])


def distracted_driver(rng, ref, n_pre, n, dt):
    """Attentive, then the head turns away (|yaw| 40-55 deg, 0.3 s turn) from 0.3-1.2 s into the scenario and
    stays away. Hands stay on the wheel: this is visual distraction, not manual."""
    yaw, pitch, eye = _face(rng, n_pre + n, dt)
    start = n_pre + int(round(rng.uniform(0.3, 1.2) / dt))
    target = rng.choice((-1, 1)) * rng.uniform(40.0, 55.0)
    turn = max(1, int(round(0.3 / dt)))
    for i in range(start, n_pre + n):
        k = min(1.0, (i - start + 1) / turn)
        yaw[i] = yaw[i] * (1 - k) + (target + rng.gauss(0, 2.0)) * k
    return _steps(yaw, pitch, eye, ["BOTH_HANDS"] * (n_pre + n), [rng.uniform(0.75, 0.95) for _ in range(n_pre + n)])


def drowsy_driver(rng, ref, n_pre, n, dt):
    """Repeated long eye closures (1.0-1.6 s closed, then ~1.5-3.3 s open) through the pre-roll and scenario, with a
    slight downward head pitch (5-16 deg, below the head-away threshold). A real drowsiness parent with a higher
    score shortens the open phases."""
    total = n_pre + n
    yaw, pitch, eye = _face(rng, total, dt, pitch0=rng.uniform(5.0, 12.0))
    severity = clamp(ref.drowsiness_score, 0.5, 0.9) if ref.drowsiness_score is not None else rng.uniform(0.5, 0.9)
    open_lo, open_hi = 3.0 - 1.5 * severity, 4.0 - 1.5 * severity
    t, closed = rng.uniform(0.0, 1.0), False
    boundary = t
    for i in range(total):
        now = i * dt
        while now >= boundary:
            closed = not closed
            boundary += rng.uniform(1.0, 1.6) if closed else rng.uniform(open_lo, open_hi)
        if closed:
            eye[i] = EYE_CLOSED + rng.uniform(-0.02, 0.03)
            pitch[i] = clamp(pitch[i] + 4.0, 0.0, 20.0)
    return _steps(yaw, pitch, eye, ["BOTH_HANDS"] * total, [rng.uniform(0.75, 0.95) for _ in range(total)])


# ------------------------------------------------------------------------------------ road scripts


def _conf(rng, base, lo=0.3, hi=0.97, sd=0.03):
    return round(clamp(base + rng.gauss(0, sd), lo, hi), 4)


def static_vehicle(rng, ref, n, dt):
    """Car ahead in the central band, lower image region, constant image size (stopped traffic ahead)."""
    cx = _pick(ref.cx_norm, 0.42, 0.58, rng)
    area = _pick(ref.area_frac, 0.02, 0.08, rng)
    conf = _pick(ref.confidence, 0.6, 0.95, rng)
    bottom = rng.uniform(0.62, 0.72)
    return [ObjectStep("car", cx + rng.gauss(0, 0.002), bottom + rng.gauss(0, 0.002), area * math.exp(rng.gauss(0, 0.01)),
                       1.4, _conf(rng, conf)) for _ in range(n)]


def approaching_vehicle(rng, ref, n, dt):
    """Car ahead in the central band whose image area grows at a constant log rate (relative approach in the
    image); its bottom edge moves down as it grows. The area stops growing at 40 % of the frame (the car has
    stopped close ahead), so late steps of fast variants test the decay after an approach ends."""
    cx = _pick(ref.cx_norm, 0.42, 0.58, rng)
    a0 = _pick(ref.area_frac, 0.01, 0.03, rng)
    g = _pick(ref.growth, 0.25, 0.5, rng)
    conf = _pick(ref.confidence, 0.6, 0.95, rng)
    out = []
    for i in range(n):
        a = min(0.40, a0 * math.exp(g * i * dt))
        out.append(ObjectStep("car", cx + rng.gauss(0, 0.002), min(0.97, 0.5 + 0.8 * math.sqrt(a)), a * math.exp(rng.gauss(0, 0.01)),
                              1.4, _conf(rng, conf)))
    return out


def _crossing(cls, aspect, bottom_rng, area_rng, speed_rng, start_rng):
    def fn(rng, ref, n, dt):
        side = rng.choice((-1, 1))  # -1: enters from the left, +1: from the right
        x = 0.5 + side * rng.uniform(*start_rng)
        v = _pick(abs(ref.lateral) if ref.lateral is not None else None, *speed_rng, rng)
        area = _pick(ref.area_frac, *area_rng, rng)
        conf = _pick(ref.confidence, 0.55, 0.92, rng)
        bottom = rng.uniform(*bottom_rng)
        out = []
        for _ in range(n):
            out.append(ObjectStep(cls, clamp(x + rng.gauss(0, 0.002), 0.02, 0.98), bottom + rng.gauss(0, 0.003),
                                  area * math.exp(rng.gauss(0, 0.015)), aspect, _conf(rng, conf)))
            x -= side * v * dt  # toward and then across the centre
        return out
    return fn


pedestrian_crossing = _crossing("person", 0.35, (0.66, 0.78), (0.01, 0.05), (0.06, 0.12), (0.26, 0.34))
dog_entering_path = _crossing("dog", 1.5, (0.78, 0.88), (0.01, 0.03), (0.10, 0.15), (0.30, 0.36))


def stationary_vehicle_false_positive(rng, ref, n, dt):
    """Parked car at the road side, outside the central band. The ego camera passes it, so its image area grows
    (0.2-0.35 ln/s) while it drifts AWAY from the centre; confidence is middling and the label sometimes flickers
    car -> truck. Tests that image growth alone (the ego-motion effect seen in the real clips) does not alarm."""
    side = rng.choice((-1, 1))
    x = 0.5 + side * rng.uniform(0.32, 0.36)
    a0 = _pick(ref.area_frac, 0.01, 0.03, rng)
    g = rng.uniform(0.2, 0.35)
    conf = _pick(ref.confidence, 0.45, 0.7, rng)
    out = []
    for i in range(n):
        a = min(0.2, a0 * math.exp(g * i * dt))
        cls = "truck" if rng.random() < 0.08 else "car"
        out.append(ObjectStep(cls, clamp(x + side * 0.03 * i * dt + rng.gauss(0, 0.002), 0.02, 0.98),
                              min(0.95, 0.5 + 0.6 * math.sqrt(a)), a, 1.4, _conf(rng, conf, sd=0.06)))
    return out


def distant_pedestrian_false_positive(rng, ref, n, dt):
    """Small, low-confidence person far away at the side (upper image region), jittering, missed in ~10 % of
    the steps. Tests that a distant, uncertain detection is not escalated."""
    side = rng.choice((-1, 1))
    x = 0.5 + side * rng.uniform(0.33, 0.4)
    area = _pick(ref.area_frac, 0.0015, 0.004, rng)
    conf = _pick(ref.confidence, 0.35, 0.6, rng)
    bottom = rng.uniform(0.38, 0.44)
    return [None if rng.random() < 0.10 else
            ObjectStep("person", x + rng.gauss(0, 0.004), bottom + rng.gauss(0, 0.004), area * math.exp(rng.gauss(0, 0.05)),
                       0.4, _conf(rng, conf, sd=0.06)) for _ in range(n)]


def persistent_low_risk(rng, ref, n, dt):
    """Pedestrian on the footpath at the side, visible for the whole scenario, walking parallel to the road and
    slightly away from the centre. Persistent presence without conflict evidence."""
    side = rng.choice((-1, 1))
    x = 0.5 + side * rng.uniform(0.27, 0.3)
    area = _pick(ref.area_frac, 0.01, 0.02, rng)
    conf = _pick(ref.confidence, 0.7, 0.9, rng)
    bottom = rng.uniform(0.6, 0.68)
    out = []
    for i in range(n):
        out.append(ObjectStep("person", clamp(x + side * 0.01 * i * dt + rng.gauss(0, 0.002), 0.02, 0.98),
                              bottom + rng.gauss(0, 0.003), area * math.exp(rng.gauss(0, 0.015)), 0.35, _conf(rng, conf)))
    return out


def escalating_hazard(rng, ref, n, dt):
    """Car far ahead in the central band: steady for 1.5-2.5 s, then its image growth rises linearly from 0 to
    0.8-1.0 ln/s by the end of the scenario (a vehicle ahead braking progressively harder, in image terms)."""
    cx = rng.uniform(0.45, 0.55)
    a = rng.uniform(0.008, 0.015)
    t1 = rng.uniform(1.5, 2.5)
    g_end = rng.uniform(0.8, 1.0)
    t_end = n * dt
    out = []
    for i in range(n):
        t = i * dt
        g = 0.0 if t < t1 else g_end * (t - t1) / max(dt, t_end - t1)
        a = min(0.35, a * math.exp(g * dt))
        out.append(ObjectStep("car", cx + rng.gauss(0, 0.002), min(0.97, 0.5 + 0.8 * math.sqrt(a)), a * math.exp(rng.gauss(0, 0.008)),
                              1.4, _conf(rng, 0.85)))
    return out


# ------------------------------------------------------------------------------------ the 12 scenarios

HIGHISH = ("HIGH", "CRITICAL")
ROAD_ONLY = ("CAUTION", "HIGH")
NO_ALARM = ("SAFE", "CAUTION")

SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("attentive_static_vehicle",
                 "Attentive driver (both hands, eyes on road) behind a car that stays at a constant image size in the central band.",
                 "Baseline: a present but non-approaching vehicle with an attentive driver should not raise HIGH risk.",
                 static_vehicle, attentive_driver(), NO_ALARM,
                 "SAFE or CAUTION (presence-only road user is capped; no driver factor)", "vehicle_presence"),
    ScenarioSpec("attentive_approaching_vehicle",
                 "Attentive driver; car in the central band growing in the image at a constant log rate.",
                 "Road-only approach: tests the approach/centrality rules and the 2-evidence-group HIGH gate without driver input.",
                 approaching_vehicle, attentive_driver(), ROAD_ONLY,
                 "CAUTION or HIGH (road-only; CRITICAL only for a very strong road hazard)", "vehicle"),
    ScenarioSpec("one_hand_approaching_vehicle",
                 "Driver with ONE hand on the wheel; same approaching-vehicle pattern.",
                 "Checks the documented rule that ONE_HAND alone is not scored as distraction.",
                 approaching_vehicle, attentive_driver("ONE_HAND"), ROAD_ONLY,
                 "Same as the attentive approach: CAUTION or HIGH (ONE_HAND is not a driver factor)", "vehicle"),
    ScenarioSpec("hands_off_approaching_vehicle",
                 "Driver removes both hands at the start (NO_HANDS); same approaching-vehicle pattern.",
                 "Driver factor (inferred HANDS_OFF_WHEEL after 2 s of NO_HANDS) combined with a moving road hazard.",
                 approaching_vehicle, hands_off_driver, HIGHISH,
                 "HIGH or CRITICAL once HANDS_OFF_WHEEL is inferred while the car approaches", "vehicle"),
    ScenarioSpec("drowsy_approaching_vehicle",
                 "Drowsy driver (repeated 1.0-1.6 s eye closures every few seconds, head slightly down); same approaching-vehicle pattern.",
                 "Driver drowsiness combined with a moving road hazard (the combined CRITICAL path).",
                 approaching_vehicle, drowsy_driver, HIGHISH,
                 "HIGH or CRITICAL (drowsiness HIGH/CRITICAL + approaching vehicle)", "vehicle", driver_parent="drowsy"),
    ScenarioSpec("attentive_pedestrian_crossing",
                 "Attentive driver; pedestrian walks from the side toward and across the central band, low in the image.",
                 "Road-only pedestrian conflict via lateral trajectory (no approach growth).",
                 pedestrian_crossing, attentive_driver(), ROAD_ONLY,
                 "CAUTION or HIGH (road-only pedestrian conflict)", "person"),
    ScenarioSpec("distracted_pedestrian_crossing",
                 "Driver looks away (|yaw| 40-55 deg, sustained); pedestrian crosses toward the central band.",
                 "Visual distraction (head away >= 1 s) combined with a pedestrian conflict.",
                 pedestrian_crossing, distracted_driver, HIGHISH,
                 "HIGH or CRITICAL (head away + pedestrian conflict)", "person"),
    ScenarioSpec("distracted_dog_entering_path",
                 "Driver looks away; a dog runs in from the side into the central band, low in the image.",
                 "Animal hazard (lateral motion into the central band) combined with visual distraction.",
                 dog_entering_path, distracted_driver, HIGHISH,
                 "HIGH or CRITICAL (head away + animal entering the central band)", "animal"),
    ScenarioSpec("stationary_vehicle_false_positive",
                 "Attentive driver passes a parked car at the road side: the car grows in the image but moves away from the centre; label flickers car/truck.",
                 "False-positive check: image growth caused by ego motion, outside the central band, with uncertain labels.",
                 stationary_vehicle_false_positive, attentive_driver(), NO_ALARM,
                 "SAFE or CAUTION (no position evidence, so the HIGH gate should hold)", "vehicle_presence"),
    ScenarioSpec("distant_pedestrian_false_positive",
                 "Attentive driver; a small, low-confidence person far away at the side, flickering in and out.",
                 "False-positive check: distant, uncertain, non-conflicting detection.",
                 distant_pedestrian_false_positive, attentive_driver(), NO_ALARM,
                 "SAFE or CAUTION (never HIGH)", "person_distant"),
    ScenarioSpec("persistent_low_risk_hazard",
                 "Attentive driver; pedestrian visible the whole time on the footpath, walking parallel to the road.",
                 "Persistence without conflict: hazard persistence grows but should not escalate beyond the presence cap.",
                 persistent_low_risk, attentive_driver(), NO_ALARM,
                 "SAFE or CAUTION (presence-only, capped)", "person"),
    ScenarioSpec("escalating_hazard",
                 "Attentive driver; a car ahead is steady for ~2 s, then grows in the image ever faster.",
                 "Temporal escalation: checks that the level rises step by step (hysteresis) from SAFE as evidence builds.",
                 escalating_hazard, attentive_driver(), HIGHISH,
                 "Starts SAFE, then rises to HIGH (or CRITICAL) as the approach strengthens", None, must_start_safe=True),
)
SCENARIO_NAMES = tuple(s.name for s in SCENARIOS)
BY_NAME = {s.name: s for s in SCENARIOS}
