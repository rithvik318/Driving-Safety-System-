# Contextual risk engine: rules, synthetic stress test and first real-data run

**What this is.** A transparent, deterministic, rule-based engine: `RiskEngine.evaluate(RiskSnapshot(timestamp, tracks, driver_state, width, height))` returns an explained assessment. `risk_score` (0–100) is a **rule-based prototype score**: the clamped sum of documented point contributions. It is **not a probability of collision**. SAFE / CAUTION / HIGH / CRITICAL are **decision-support states**, not validated safety classifications. No ML, no LLM, no YOLO or OpenCV dependency in `app/risk`.

Every assessment carries:
- `timestamp`, `risk_level` (after hysteresis) and `raw_risk_level`;
- `raw_risk_score` and `smoothed_risk_score` (`risk_score` = smoothed);
- `hazard_type`, `primary_track_id`, `contributing_factors` (name, signed points, source, evidence text), and a `reason` generated from those factors;
- `evidence_quality` (0–1, not a probability), `evidence_notes`, any `gates` applied, and `alarm_recommended` (only a flag; no alarm is raised in this step).

The factor points always sum exactly to `raw_risk_score`; caps and clamping appear as negative factors.

## Rule configuration (prototype values, `app/config/risk_config.py`, `RISK_*` env vars)

**Levels on the score:** SAFE < 20 ≤ CAUTION < 45 ≤ HIGH < 70 ≤ CRITICAL.

**Evidence gates:**
- **HIGH** needs ≥ 2 independent evidence groups. The groups are each driver factor, road *motion* (approach or trajectory) and road *position* (central band + lower image region).
- **CRITICAL** needs one of:
  - a driver factor **and** a road hazard with motion evidence;
  - road points ≥ 60 with both motion and position evidence.

**Road track gates:** persistence ≥ 0.5 s, ≥ 3 detections, not coasting > 0.3 s.

**Motion features:** least-squares slope over the last 1.0 s (≥ 3 samples, ≥ 0.3 s) of:
- ln(box area) → "growth" per second (approach evidence ≥ 0.15/s ≈ +16 %/s; full at 0.8/s);
- center x / frame width → lateral frame-widths/s (trajectory evidence ≥ 0.05; full at 0.30).

**Regions (image space):**
- central band: |x − 0.5| ≤ 0.20 of the frame width;
- near-centre band: ≤ 0.35;
- lower region: box bottom ≥ 0.45 of the frame height.

The central band is a stand-in for "ahead of the camera", **not** a calibrated road model or the vehicle's real path.

**Maximum points per road factor:**

| Factor | Vehicle | Person | Animal |
| --- | ---: | ---: | ---: |
| persistence (full at 2 s) | 10 | 10 | 10 |
| centrality (half if near-centre or upper region) | 10 | 15 | 15 |
| approach (box growth) | 25 | 15 | 15 |
| trajectory (toward / across central band) | 5 | 20 | 20 |
| image_size (area fraction 1 %→12 %) | 10 | 5 | 5 |

**When a track counts as a hazard (qualifies):**
- vehicle: needs approach evidence;
- person: in / near the central band **and** trajectory or approach;
- animal: near-centre or lower region **and** moving (lateral or growth).

A track that does not qualify is presence-only and capped at 20 points. Evidence quality scales a track's points: mean detector confidence below 0.60 (floor ×0.5) and −15 % per detector label change (floor ×0.5). Other road terms: +5 when ≥ 2 road hazards qualify at once, and a road cap of 70.

**Driver factors** (existing `DriverState` signals only; driver cap 60):

| Signal | Points |
| --- | ---: |
| drowsiness_level HIGH | 30 |
| drowsiness_level CRITICAL | 45 |
| driver_activity HANDS_OFF_WHEEL (inferred upstream from ≥ 2 s continuous NO_HANDS) | 25 |
| head turned away (\|yaw\| > 30° or pitch > 25° continuously ≥ 1 s) | 10 |
| ONE_HAND | 0 (recorded in the evidence notes) |

A driver observation older than 1 s is ignored.

**Combination:** +15 × min(1, road points / 30) when a driver factor coincides with a qualifying road hazard.

**Hysteresis:**
- entering CAUTION / HIGH / CRITICAL needs 1 / 2 / 3 consecutive raw observations at that level or above;
- the level steps down one level after 2 consecutive lower observations;
- the smoothed score rises immediately and falls at most 40 points/s;
- a gap of more than 2 s resets the smoothing.

## Part A: SYNTHETIC rule stress test (constructed inputs; NOT observations)

Constructed detections were fed through the real `IoUTracker` at 10 fps (3 s each), with `DriverState` objects built directly.

| Scenario (SYNTHETIC) | Final level | Raw score | Hazard type |
| --- | --- | ---: | --- |
| no hazard, attentive driver | SAFE | 0 | NONE |
| static distant vehicle | SAFE | 12 | NONE |
| approaching vehicle, attentive driver | HIGH | 49 | APPROACHING_VEHICLE |
| pedestrian moving toward centre band | CAUTION | 36 | PEDESTRIAN_CONFLICT |
| pedestrian static far to the side | SAFE | 10 | NONE |
| dog central and moving | CAUTION | 34 | ANIMAL_HAZARD |
| drowsiness HIGH, no hazard | CAUTION | 30 | DRIVER_DROWSINESS |
| hands-off-wheel, no hazard | CAUTION | 25 | HANDS_OFF_WHEEL |
| one hand on wheel, no hazard | SAFE | 0 | NONE |
| approaching vehicle + hands-off-wheel | CRITICAL | 89 | COMBINED_DRIVER_HAZARD |
| approaching vehicle + drowsiness HIGH | CRITICAL | 94 | COMBINED_DRIVER_HAZARD |
| pedestrian toward centre + hands-off-wheel | CRITICAL | 76 | COMBINED_DRIVER_HAZARD |
| dog central + hands-off-wheel | CRITICAL | 74 | COMBINED_DRIVER_HAZARD |
| static side pedestrian + hands-off-wheel | CAUTION | 35 | HANDS_OFF_WHEEL |

Example output (SYNTHETIC, approaching vehicle + hands-off-wheel, t = 2.9 s):

```text
risk_level=CRITICAL  raw_score=89  smoothed_score=89  track_id=1  hazard_type=COMBINED_DRIVER_HAZARD
factors:
- hands_off_wheel: +25.0  (inferred HANDS_OFF_WHEEL (NO_HANDS observed continuously 4.9 s))
- persistence: +10.0  (car #1 observed for 2.9 s (30 detections))
- centrality: +10.0  (car #1 in the central image band, lower image region (x=0.50, bottom=0.88 of frame))
- approach: +19.2  (car #1 image-space box area growing 73%/s)
- image_size: +10.0  (car #1 covers 15.9% of the frame)
- driver_road_combination: +15.0
reason: Inferred HANDS_OFF_WHEEL (NO_HANDS observed continuously 4.9 s) coincides with a persistent car #1
(observed 2.9 s, 30 detections): image-space box area growing 73%/s, in the central image band.
```

## Part B: REAL front-camera videos, road-only (no driver input)

This covered 14 real front-camera videos (the same set as the tracking report): YOLO26n → IoUTracker at 10 fps → RiskEngine, **with no driver observation**. The front videos have no simultaneous driver recording. Road-only results therefore cap `evidence_quality` at 0.5. The folder names are collection context, not ground truth. **The level counts below are outputs of prototype rules, not collision labels or ground-truth safety labels.**

| | Observations |
| --- | ---: |
| Sampled frames evaluated | 515 |
| SAFE (after hysteresis) | 184 (36 %) |
| CAUTION | 221 (43 %) |
| HIGH | 110 (21 %) |
| CRITICAL | 0 (0 %) |
| Raw level before hysteresis | SAFE 194, CAUTION 209, HIGH 112, CRITICAL 0 |
| raw_risk_score | min 0, median 33, mean 28.4, max 65 |
| smoothed_risk_score | min 0, median 35, mean 29.7, max 65 |
| HIGH-gate caps applied | 42 observations |

- **Hazard types:** APPROACHING_VEHICLE 172, NONE 171, PEDESTRIAN_CONFLICT 159, ANIMAL_HAZARD 7, ROAD_USER_PRESENT 6.
- **Factors triggered** (observations): persistence 367, centrality 346, approach 312, image_size 312, multiple_hazards 230, trajectory 163, evidence_quality 118, presence_cap 6.

| Folder (context) | Video | Max level | Max raw score |
| --- | --- | --- | ---: |
| dogs_on_road | video_20260926_211127.mp4 | CAUTION | 29 |
| dogs_on_road | video_20260926_211145.mp4 | CAUTION | 33 |
| dogs_on_road | video_20260926_211200.mp4 | SAFE | 11 |
| pedestrains | video_20260926_210423.mp4 | HIGH | 52 |
| pedestrains | video_20260926_210536.mp4 | HIGH | 65 |
| pedestrains | video_20260926_211603.mp4 | CAUTION | 38 |
| roads | video_20260926_165942.mp4 | CAUTION | 24 |
| roads | video_20260926_210938.mp4 | CAUTION | 25 |
| vehicles | video_20260926_165951.mp4 | HIGH | 62 |
| vehicles | video_20260926_170005.mp4 | HIGH | 63 |
| vehicles | video_20260926_172658.mp4 | HIGH | 49 |
| vehicles | video_20260926_173058.mp4 | HIGH | 62 |
| vehicles | video_20260926_211717.mp4 | HIGH | 65 |
| vehicles | video_20260926_211739.mp4 | CAUTION | 20 |

Highest-scoring real example (road-only), `pedestrains/video_20260926_210536.mp4` t = 3.2 s, **HIGH**, raw 65:

- persistence +10 (person #1 observed 3.2 s, 33 detections);
- centrality +15 (x = 0.67, bottom = 0.89);
- approach +15 (box area growing 143 %/s);
- trajectory +20 (moving across the central band at 0.37 frame-widths/s);
- image_size +5 (24 % of the frame).

**Visual check of the top HIGH frames** (`outputs/plots/risk_engine_examples.jpg`):
- In three of the four the box grows mainly because **the camera is closing in**: a parked red car passed at close range (`170005`), a parked car directly ahead (`165951`), and a person walking just ahead of the camera (`211717`).
- In one (`173058`) a cyclist is riding toward the camera.

Image-space growth cannot distinguish "we are closing in" from "it is coming at us". APPROACHING_VEHICLE therefore means *relative approach in image space*, not that the vehicle itself is moving. This ego-motion effect is the main reason 21 % of road-only observations reached HIGH on ordinary campus footage. No road-only observation reached CRITICAL.

Example level timeline (`vehicles/video_20260926_165951.mp4`, road-only):

```text
t=0.00s  -> SAFE      raw  0  No persistent road hazard and no driver impairment signal.
t=0.52s  -> CAUTION   raw 34  PEDESTRIAN_CONFLICT  person #2 ... box area growing 53%/s, moving across the central band
t=1.42s  -> HIGH      raw 48  APPROACHING_VEHICLE  car #1 ... box area growing 86%/s, in the central image band
t=3.22s  -> CAUTION   raw 55  car #1 now only NEAR the central band: position evidence lost, HIGH gate applies
```

## Part C: REAL driver signals PAIRED with REAL front-camera videos (NOT synchronised)

The real drowsy-driver video `drowsy_driver/video_20260926_220714.mp4` was run through the DriverPerceptionPipeline: MediaPipe plus the trained hand model, 63 DriverStates over 6.2 s.
- drowsiness: UNKNOWN 38, HIGH 12, CRITICAL 13;
- driver_activity: NORMAL 37, UNKNOWN 26 (no HANDS_OFF_WHEEL was inferred).

The states were paired with each front video by elapsed time from 0 s; after 6.2 s there is no driver input. **The two cameras were not recorded together**, so this only exercises the combination rules on real signals. It is not a record of any real drive.

| | Observations |
| --- | ---: |
| SAFE / CAUTION / HIGH / CRITICAL (after hysteresis) | 152 / 185 / 97 / 81 |
| Raw level before hysteresis | SAFE 161, CAUTION 180, HIGH 79, CRITICAL 95 |
| raw_risk_score | min 0, median 39, mean 40.1, max 100 (clamped) |
| COMBINED_DRIVER_HAZARD | 104 observations |
| DRIVER_DROWSINESS alone | 28 observations |

Example (PAIRED, not synchronised), `vehicles/video_20260926_165951.mp4` t = 1.88 s, **CRITICAL**, raw 100:

- drowsiness HIGH +30 (score 0.65);
- car #1: persistence +9.4, centrality +10, approach +24.8 (growing 121 %/s), image_size +7.6;
- multiple_hazards +5, driver_road_combination +15, score_clamp −1.8.

The reason text reads: *"Drowsiness level HIGH, score 0.65 coincides with a persistent car #1 (observed 1.9 s, 19 detections): image-space box area growing 121%/s, in the central image band (+2 other road hazards)."*

Hysteresis on real data: the raw level was CRITICAL on 95 observations, but the level after hysteresis was CRITICAL on 81, because entering CRITICAL needs 3 consecutive observations. Levels then stepped down one at a time while the smoothed score decayed; for example, on the same video the score fell 99 → 92 → 76 after the paired drowsiness level fell back from HIGH, and the car drifted out of the central band.

## Limitations

- **No camera calibration.** No real distance estimation, no physical speed, and no validated time-to-collision. All road signals are image-space (pixels, frame fractions, growth per second).
- **Camera motion affects image-space motion.** A moving camera makes stationary objects ahead grow and shift. On our footage most HIGH road-only frames were caused by the camera closing in.
- **The central band is not a path model.** Being "central" is not the vehicle's real path, and no collision is predicted.
- **YOLO errors propagate:** missed and mislabelled objects, e.g. the black dog at night labelled person.
- **Tracker errors propagate:** broken IDs, duplicate parallel tracks for one object, and growth measured on jittery boxes.
- **Prototype thresholds and weights.** They were set by reasoning, not fitted or validated. Changing them changes the level counts.
- **The risk score is not a probability.** Levels are decision-support states, not validated safety classifications.
- **No ground truth:** no collision or near-miss labels. The counts in parts B and C describe rule behaviour, not accuracy.
- **Pairing is artificial.** The driver and front cameras in this dataset were not recorded simultaneously, so part C is pairing only.
- **Driver signal limits:** the hand-state classifier generalises poorly to the demo driver setup (see the domain comparison), so HANDS_OFF_WHEEL was never inferred on the real driver video. PHONE / OTHER_MANUAL_DISTRACTION are not scored until an observing provider exists.
