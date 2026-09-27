"""Contextual risk engine: DriverState + tracked road objects -> explained risk level.

    engine = RiskEngine(RiskConfig())
    assessment = engine.evaluate(RiskSnapshot(timestamp, tracks, driver_state, width, height))

Per evaluation:
  1. per-track motion features over a short window (smoothing.TrackFeatureHistory)
  2. road rules per track (rules.evaluate_track); primary hazard = strongest qualifying track
     (else the strongest presence-only track); + multi-hazard bonus; capped at road_cap_points
  3. driver rules (rules.driver_factors); capped at driver_cap_points
  4. combined bonus when a driver factor coincides with a qualifying road hazard
  5. raw_risk_score = clamp(sum of all factor points, 0, 100). The factor list always sums
     exactly to raw_risk_score (caps and clamping appear as negative adjustment factors).
  6. raw level from thresholds, then evidence gates (rules.apply_gates)
  7. hysteresis -> risk_level and smoothed_risk_score (smoothing.RiskHysteresis)

risk_score is a rule-based prototype score (0-100), NOT a probability of collision; levels are
decision-support states, not validated safety classifications. No YOLO/OpenCV dependency.
"""

from __future__ import annotations

import math

from app.config.risk_config import RiskConfig, validate_risk_config
from app.risk import rules
from app.risk.models import HazardType, RiskAssessment, RiskFactor, RiskSnapshot, TrackHazard
from app.risk.smoothing import HeadAwayTimer, RiskHysteresis, TrackFeatureHistory, alarm_level_rank

DRIVER_HAZARD = {"drowsiness": HazardType.DRIVER_DROWSINESS, "hands_off_wheel": HazardType.HANDS_OFF_WHEEL,
                 "head_away": HazardType.DRIVER_HEAD_AWAY}


class RiskEngine:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        validate_risk_config(self.config)
        self.history = TrackFeatureHistory(self.config)
        self.head_timer = HeadAwayTimer()
        self.hysteresis = RiskHysteresis(self.config)
        self._last_t: float | None = None

    def reset(self) -> None:
        """Forget all temporal state (call between videos / sessions)."""
        self.history.reset()
        self.head_timer.reset()
        self.hysteresis.reset()
        self._last_t = None

    def evaluate(self, snapshot: RiskSnapshot) -> RiskAssessment:
        cfg = self.config
        t = snapshot.timestamp
        if t is None or isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
            raise ValueError(f"timestamp must be a finite number of seconds, got {t!r}")
        if self._last_t is not None and t < self._last_t:
            self.reset()  # new timeline
        self._last_t = t
        tracks = list(snapshot.tracks or ())
        width, height = int(snapshot.image_width or 0), int(snapshot.image_height or 0)
        notes: list[str] = []
        if tracks and (width <= 0 or height <= 0):
            notes.append("image size unknown: road factors not evaluated")

        # 1-2. road
        self.history.update(tracks, width, height)
        hazards: list[TrackHazard] = []
        for tr in tracks:
            growth, lateral = self.history.features(tr.track_id)
            h = rules.evaluate_track(tr, growth, lateral, width, height, cfg)
            if h is not None:
                hazards.append(h)
        hazards.sort(key=lambda h: (not h.qualifies, -h.points, h.track_id))
        qualifying = [h for h in hazards if h.qualifies]
        primary = hazards[0] if hazards else None
        factors: list[RiskFactor] = []
        road_points = 0.0
        if primary is not None:
            factors += list(primary.factors)
            road_points = primary.points
            if len(qualifying) >= 2:
                factors.append(RiskFactor("multiple_hazards", cfg.multi_hazard_bonus_points, "road",
                                          f"{len(qualifying)} road hazards with approach/conflict evidence at once"))
                road_points += cfg.multi_hazard_bonus_points
            if road_points > cfg.road_cap_points:
                factors.append(RiskFactor("road_cap", cfg.road_cap_points - road_points, "road",
                                          f"road contribution capped at {cfg.road_cap_points:g}"))
                road_points = cfg.road_cap_points

        # 3. driver
        driver = snapshot.driver
        head_away_s = 0.0
        if driver is not None:
            head_away_s = self.head_timer.update(driver.timestamp, rules.head_is_away(driver, cfg))
        d_factors, d_notes = rules.driver_factors(driver, t, head_away_s, cfg)
        notes += d_notes
        driver_points = sum(f.points for f in d_factors)
        factors = d_factors + factors
        if driver_points > cfg.driver_cap_points:
            factors.append(RiskFactor("driver_cap", cfg.driver_cap_points - driver_points, "driver",
                                      f"driver contribution capped at {cfg.driver_cap_points:g}"))
            driver_points = cfg.driver_cap_points

        # 4. combination
        combined = bool(d_factors) and primary is not None and primary.qualifies
        if combined:
            bonus = cfg.combined_bonus_points * min(1.0, road_points / cfg.combined_reference_points)
            factors.append(RiskFactor("driver_road_combination", bonus, "combined",
                                      "driver factor coincides with a road hazard with approach/conflict evidence"))

        # 5. score
        total = sum(f.points for f in factors)
        raw_score = max(0.0, min(100.0, total))
        if raw_score != total:
            factors.append(RiskFactor("score_clamp", raw_score - total, "quality", "score clamped to 0-100"))

        # 6. level + gates
        raw_level = rules.level_for_score(raw_score, cfg)
        raw_level, gates = rules.apply_gates(raw_level, [f.name for f in d_factors], primary, road_points, cfg)

        # 7. hysteresis
        level, smoothed = self.hysteresis.update(raw_level, raw_score, t)

        # hazard type + primary track
        if combined:
            hazard_type = HazardType.COMBINED_DRIVER_HAZARD
        elif primary is not None and primary.qualifies:
            hazard_type = primary.hazard_type
        elif d_factors:
            hazard_type = DRIVER_HAZARD[max(d_factors, key=lambda f: f.points).name]
        elif primary is not None and primary.points >= cfg.caution_min:
            hazard_type = HazardType.ROAD_USER_PRESENT
        else:
            hazard_type = HazardType.NONE
        show_track = primary is not None and (primary.qualifies or hazard_type is HazardType.ROAD_USER_PRESENT)

        # evidence quality: weakest supporting evidence; halved when the driver branch is missing
        parts = []
        if d_factors:
            parts.append(float(driver.observation_quality))
        if show_track:
            parts.append(primary.quality)
        evidence = min(parts) if parts else 1.0
        if driver is None or any("not evaluated" in n for n in d_notes):
            evidence = min(evidence, 0.5)
        if primary is not None and primary.quality < 1.0:
            notes.append(f"track #{primary.track_id} evidence multiplier {primary.quality:.2f}")
        notes.append("image-space road signals only: no calibration, distance, physical speed or TTC; "
                     "camera motion is included in image motion")

        reason = rules.build_reason(hazard_type, d_factors, primary if show_track else None,
                                    max(0, len(qualifying) - (1 if primary is not None and primary.qualifies else 0)))
        return RiskAssessment(
            timestamp=float(t), risk_level=level, raw_risk_level=raw_level, raw_risk_score=raw_score,
            smoothed_risk_score=smoothed, hazard_type=hazard_type,
            primary_track_id=primary.track_id if show_track else None,
            primary_class=primary.class_name if show_track else None,
            contributing_factors=tuple(factors), reason=reason, evidence_quality=evidence,
            evidence_notes=tuple(notes), alarm_recommended=level.rank >= alarm_level_rank(cfg), gates=tuple(gates),
            hazards=tuple(hazards), driver_points=driver_points, road_points=road_points,
        )
