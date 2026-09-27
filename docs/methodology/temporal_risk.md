# Temporal features and the rule-based risk engine

These are the default values in `app/config/risk_config.py`. Every value can be overridden with `RISK_<FIELD>`. They are prototype choices, not tuned on real outcomes.

## Temporal features

| Feature | How it is computed | Meaning |
| --- | --- | --- |
| Track persistence | observed span of an IoU track; a track contributes only after ≥ 0.5 s and ≥ 3 detections | the object is really there, not a one-frame flicker |
| Growth (approach proxy) | slope of ln(box area) over ~1 s; counted from 0.15/s (≈ +16 %/s), full points at 0.80/s | "getting bigger in the image" — **not** distance or speed |
| Central-band position | box centre within ±0.20 of the frame width around the centre, full points in the lower image region | a stand-in for "ahead", **not** a calibrated path |
| Lateral motion | horizontal motion toward/across the band, counted from 0.05 frame-widths/s | crossing or entering the area ahead |
| Eye closure | rolling window of eye-aspect-ratio closures (blinks vs long closures, current closure duration) → drowsiness score 0–1 | prototype drowsiness signal, not medical |
| Hands-off | NO_HANDS observed continuously ≥ 2 s → inferred `HANDS_OFF_WHEEL` | one frame never counts |
| Head away | yaw > 30° or pitch > 25° for ≥ 1 s | sustained looking away |

## Scoring, gates and hysteresis

- **Road points per track:** persistence, centrality, approach, image size and lateral trajectory. The weights depend on the kind of object (vehicle / person / animal).
  - A track without approach or conflict evidence is capped at **20 points** ("road user present").
- **Driver points:** drowsiness HIGH 30 / CRITICAL 45, hands-off 25, head-away 10.
  - ONE_HAND scores 0.
  - When a driver factor coincides with a moving road hazard, a bonus of up to 15 is added.
- **Score:** 0–100. The level thresholds are CAUTION ≥ 20, HIGH ≥ 45 and CRITICAL ≥ 70.
- **Gates:**
  - HIGH needs **2 independent evidence groups**: a driver factor, road motion, or road position.
  - CRITICAL needs a driver factor with a moving road hazard, **or** road points ≥ 60 with both motion and position evidence.
- **Hysteresis:** entering HIGH needs 2 consecutive observations and entering CRITICAL needs 3. The level steps down one level after 2 lower observations. The smoothed score falls at most 40 points/s.
- **Alarm:** recommended at level ≥ HIGH.
- **Explanation:** every point is a named factor, and `risk_reason` is generated from those factors, with no LLM.

## Why no real CRITICAL

Real driver and front recordings are not synchronized, so no real event can combine a driver factor with a road hazard. No real road-only hazard reached the "very strong" CRITICAL gate. The combined path is validated only on labelled synthetic input (e.g. `drowsy_approaching_vehicle`, `hands_off_approaching_vehicle`).
