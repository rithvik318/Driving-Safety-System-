# Post-event safety summary (LOCAL_REAL)

> Prototype Physical-AI safety-observation system. Risk levels are rule-based decision-support states (score not a probability); road signals are image-space only. No collision prediction, no real-world safety validation, no distance/speed measurement, no medical drowsiness assessment, no geographic hotspots, no synchronized driver+road recording.

**Input:** `data/events/events.jsonl`.
- Excluded non-real records: none.
- Malformed lines: 0.

## 1. Event summary

- **Total events:** 66 (runs local_real_run_01; 16 sessions).
  - Data sources: LOCAL_REAL 66.
  - Observation types: INFERRED 64, OBSERVED 2.
- **By type:** RISK_ESCALATED 32, RISK_DEESCALATED 18, PERSISTENT_HAZARD 9, DRIVER_STATE_CHANGE 7.
- **By camera:** front 54, driver 12.
- **By risk level:** SAFE 9, CAUTION 34, HIGH 16, CRITICAL 0.
  - Events without a risk level (driver-state changes): 7.
- **By hazard type:** PEDESTRIAN_CONFLICT 28, APPROACHING_VEHICLE 19, not recorded 7, NONE 4, DRIVER_DROWSINESS 4, ANIMAL_HAZARD 2, ROAD_USER_PRESENT 2.
- **Risk transitions:** SAFE->CAUTION 21, CAUTION->HIGH 11, HIGH->CAUTION 9, CAUTION->SAFE 9.
- **Alarm recommended (engine):** false 43, true 16, not recorded 7.
  - `alarm_triggered` true: 0. The event schema keeps it false; alarms from the demo alert layer are logged separately.
- **Persistent hazards:** 9.
  - By hazard type: APPROACHING_VEHICLE 5, PEDESTRIAN_CONFLICT 4.
  - By level: HIGH 5, CAUTION 4.
  - Hazard persistence (s): n 9, mean 2.046, median 2.016, max 2.099, min 2.0.
- **Driver-state events:** 7. drowsiness_level: HIGH->CRITICAL 3, drowsiness_level: CRITICAL->HIGH 2, hand_state: ONE_HAND->NO_HANDS 1, hand_state: NO_HANDS->ONE_HAND 1.
- **Risk score** (smoothed, 0–100, not a probability): n 59, mean 38.634, median 37.63, max 60.0, min 18.859.
  - Raw: n 59, mean 36.321, median 34.928, max 60.0, min 0.0.
- **Durations** (stream time):
  - Track persistence at the event: n 51, mean 1.741, median 1.416, max 4.415, min 0.5.
  - Elevated-risk episodes: 21, of which 9 returned to SAFE (durations n 9, mean 2.099, median 0.8, max 10.695, min 0.2) and 12 were still open when the clip ended.

## 2. Hazard patterns (descriptive)

_Descriptive counts of what the prototype recorded. They do not show that a location is dangerous or that driving was unsafe, and they inherit detector/classifier errors._

- Pedestrian hazards: 21 event(s) across 12 session(s); repeated within 7 session(s) (up to 3 in one session).
- Approaching-vehicle hazards: 12 event(s) across 5 session(s); repeated within 5 session(s) (up to 4 in one session).
- Animal hazards: 2 event(s) across 2 session(s); no session had more than one.
- Driver-state changes: 7 event(s) across 3 session(s); repeated within 2 session(s) (up to 4 in one session).
- Risk escalations: 32 event(s) across 16 session(s); repeated within 12 session(s) (up to 5 in one session).
- Escalations to HIGH or CRITICAL: 11 event(s) across 7 session(s); repeated within 2 session(s) (up to 4 in one session).

## 3. Hotspot analysis

**GPS hotspot analysis unavailable because no real GPS observations were collected.**

**NOT geographic hotspots: event frequencies by recording and by stream time.**

| source file | events |
| --- | --- |
| video_20260926_170005.mp4 | 12 |
| video_20260926_173058.mp4 | 7 |
| video_20260926_220714.mp4 | 7 |
| video_20260926_165951.mp4 | 5 |
| video_20260926_172658.mp4 | 4 |
| video_20260926_211127.mp4 | 4 |
| video_20260926_211717.mp4 | 4 |
| video_20260926_210423.mp4 | 3 |
| video_20260926_210536.mp4 | 3 |
| video_20260926_211145.mp4 | 3 |
| video_20260926_211603.mp4 | 3 |
| video_20260926_211739.mp4 | 3 |
| video_20260926_220747.mp4 | 3 |
| video_20260926_210938.mp4 | 2 |
| video_20260926_220840.mp4 | 2 |
| video_20260926_165942.mp4 | 1 |

By camera/source: front 54, driver 12.

Events by stream-time bin (2 s; stream time = seconds since the start of each clip (no absolute clock); clips differ in length):

| stream time | events by type |
| --- | --- |
| 0-2 s | RISK_ESCALATED 24, RISK_DEESCALATED 6 |
| 2-4 s | PERSISTENT_HAZARD 7, RISK_DEESCALATED 7, RISK_ESCALATED 4, DRIVER_STATE_CHANGE 4 |
| 4-6 s | RISK_ESCALATED 3, RISK_DEESCALATED 3, DRIVER_STATE_CHANGE 3 |
| 6-8 s | RISK_ESCALATED 1, PERSISTENT_HAZARD 2 |
| 8-10 s | RISK_DEESCALATED 1 |
| 10-12 s | RISK_DEESCALATED 1 |

## 4. Post-event recommendations

_Safety-oriented review suggestions derived from recorded events; not medical diagnoses and not claims of fault._

| recommendation | events | type |
| --- | --- | --- |
| Review pedestrian interaction and maintain additional visual attention in similar situations. | 21 | review |
| Review following/approach context and maintain additional observation of closing traffic. | 12 | review |
| Treat prolonged eye closure as a driver-attention warning; review rest/alertness before continued driving. | 9 | review |
| Review the animal's position relative to the road and maintain additional attention and readiness to slow in similar situations. | 2 | review |
| Review the driving segment and maintain both hands on the steering wheel when appropriate. | 1 | review |
| Informational: risk level decreased; no additional action. | 18 | informational |
| Informational: a road user was present without approach/conflict evidence; no specific action beyond normal observation. | 2 | informational |
| Informational: hand-position change recorded; no action implied on its own. | 1 | informational |

## 5. Event report (significant events)

Significant events: escalations to HIGH or CRITICAL, persistent hazards and driver-state changes. Missing values are shown as *not recorded*.

**event_000021** (RISK_ESCALATED, INFERRED):
- **Time:** 1.42 s into video_20260926_165951.mp4
- **Camera:** front
- **Risk:** HIGH (score 48.0), CAUTION->HIGH
- **Hazard:** APPROACHING_VEHICLE (car)
- **Reason:** Persistent car #1 (observed 1.4 s, 15 detections): image-space box area growing 86%/s, in the central image band (+3 other road hazards).
- **Evidence:** `data/events/local_real_run_01/event_000021`
- **Recommended review/action:** Review following/approach context and maintain additional observation of closing traffic.

**event_000022** (PERSISTENT_HAZARD, INFERRED):
- **Time:** 2.62 s into video_20260926_165951.mp4
- **Camera:** front
- **Risk:** HIGH (score 60.0)
- **Hazard:** APPROACHING_VEHICLE (car)
- **Reason (engine's overall assessment at this time):** Persistent car #1 (observed 2.6 s, 27 detections): image-space box area growing 125%/s, in the central image band (+3 other road hazards).
- **Evidence:** `data/events/local_real_run_01/event_000022`
- **Recommended review/action:** Review following/approach context and maintain additional observation of closing traffic.

**event_000024** (PERSISTENT_HAZARD, INFERRED):
- **Time:** 3.32 s into video_20260926_165951.mp4
- **Camera:** front
- **Risk:** CAUTION (score 55.0)
- **Hazard:** PEDESTRIAN_CONFLICT (person)
- **Reason (engine's overall assessment at this time):** Persistent car #1 (observed 3.3 s, 34 detections): image-space box area growing 194%/s, near the central image band (+2 other road hazards).
- **Evidence:** `data/events/local_real_run_01/event_000024`
- **Recommended review/action:** Review pedestrian interaction and maintain additional visual attention in similar situations.

**event_000026** (RISK_ESCALATED, INFERRED):
- **Time:** 1.52 s into video_20260926_170005.mp4
- **Camera:** front
- **Risk:** HIGH (score 48.6), CAUTION->HIGH
- **Hazard:** PEDESTRIAN_CONFLICT (person)
- **Reason:** Persistent person #2 (observed 1.5 s, 16 detections): image-space box area growing 21%/s, moving across the central band at 0.08 frame-widths/s (image space), in the central image band (+3 other road hazards).
- **Evidence:** `data/events/local_real_run_01/event_000026`
- **Recommended review/action:** Review pedestrian interaction and maintain additional visual attention in similar situations.

**event_000028** (RISK_ESCALATED, INFERRED):
- **Time:** 4.21 s into video_20260926_170005.mp4
- **Camera:** front
- **Risk:** HIGH (score 50.4), CAUTION->HIGH
- **Hazard:** PEDESTRIAN_CONFLICT (person)
- **Reason:** Persistent person #2 (observed 4.2 s, 43 detections): image-space box area growing 26%/s, moving across the central band at 0.06 frame-widths/s (image space), in the central image band (+1 other road hazard).
- **Evidence:** `data/events/local_real_run_01/event_000028`
- **Recommended review/action:** Review pedestrian interaction and maintain additional visual attention in similar situations.

**event_000030** (RISK_ESCALATED, INFERRED):
- **Time:** 4.63 s into video_20260926_170005.mp4
- **Camera:** front
- **Risk:** HIGH (score 51.8), CAUTION->HIGH
- **Hazard:** APPROACHING_VEHICLE (bicycle)
- **Reason:** Persistent bicycle #49 (observed 0.7 s, 8 detections): image-space box area growing 1168%/s, moving across the central band at 0.06 frame-widths/s (image space), in the central image band (+1 other road hazard).
- **Evidence:** `data/events/local_real_run_01/event_000030`
- **Recommended review/action:** Review following/approach context and maintain additional observation of closing traffic.

**event_000032** (RISK_ESCALATED, INFERRED):
- **Time:** 6.91 s into video_20260926_170005.mp4
- **Camera:** front
- **Risk:** HIGH (score 52.0), CAUTION->HIGH
- **Hazard:** APPROACHING_VEHICLE (car)
- **Reason:** Persistent car #50 (observed 2.9 s, 30 detections): image-space box area growing 94%/s, in the central image band (+2 other road hazards).
- **Evidence:** `data/events/local_real_run_01/event_000032`
- **Recommended review/action:** Review following/approach context and maintain additional observation of closing traffic.

**event_000033** (PERSISTENT_HAZARD, INFERRED):
- **Time:** 7.31 s into video_20260926_170005.mp4
- **Camera:** front
- **Risk:** HIGH (score 57.7)
- **Hazard:** APPROACHING_VEHICLE (car)
- **Reason (engine's overall assessment at this time):** Persistent car #50 (observed 3.3 s, 34 detections): image-space box area growing 122%/s, in the central image band (+1 other road hazard).
- **Evidence:** `data/events/local_real_run_01/event_000033`
- **Recommended review/action:** Review following/approach context and maintain additional observation of closing traffic.

**event_000034** (PERSISTENT_HAZARD, INFERRED):
- **Time:** 7.81 s into video_20260926_170005.mp4
- **Camera:** front
- **Risk:** HIGH (score 60.0)
- **Hazard:** APPROACHING_VEHICLE (car)
- **Reason (engine's overall assessment at this time):** Persistent car #50 (observed 3.8 s, 39 detections): image-space box area growing 172%/s, in the central image band (+2 other road hazards).
- **Evidence:** `data/events/local_real_run_01/event_000034`
- **Recommended review/action:** Review following/approach context and maintain additional observation of closing traffic.

**event_000038** (RISK_ESCALATED, INFERRED):
- **Time:** 2.71 s into video_20260926_172658.mp4
- **Camera:** front
- **Risk:** HIGH (score 47.9), CAUTION->HIGH
- **Hazard:** PEDESTRIAN_CONFLICT (person)
- **Reason:** Persistent person #24 (observed 0.7 s, 8 detections): image-space box area growing 107%/s, moving across the central band at 0.10 frame-widths/s (image space), in the central image band (+3 other road hazards).
- **Evidence:** `data/events/local_real_run_01/event_000038`
- **Recommended review/action:** Review pedestrian interaction and maintain additional visual attention in similar situations.

**event_000040** (PERSISTENT_HAZARD, INFERRED):
- **Time:** 3.83 s into video_20260926_172658.mp4
- **Camera:** front
- **Risk:** CAUTION (score 39.3)
- **Hazard:** APPROACHING_VEHICLE (truck)
- **Reason (engine's overall assessment at this time):** Persistent person #24 (observed 1.8 s, 19 detections): image-space box area growing 214%/s, in the central image band (+2 other road hazards).
- **Evidence:** `data/events/local_real_run_01/event_000040`
- **Recommended review/action:** Review following/approach context and maintain additional observation of closing traffic.

**event_000042** (RISK_ESCALATED, INFERRED):
- **Time:** 1.82 s into video_20260926_173058.mp4
- **Camera:** front
- **Risk:** HIGH (score 52.7), CAUTION->HIGH
- **Hazard:** PEDESTRIAN_CONFLICT (person)
- **Reason:** Persistent person #1 (observed 1.8 s, 19 detections): image-space box area growing 77%/s, moving across the central band at 0.11 frame-widths/s (image space), in the central image band (+1 other road hazard).
- **Evidence:** `data/events/local_real_run_01/event_000042`
- **Recommended review/action:** Review pedestrian interaction and maintain additional visual attention in similar situations.

**event_000044** (PERSISTENT_HAZARD, INFERRED):
- **Time:** 2.72 s into video_20260926_173058.mp4
- **Camera:** front
- **Risk:** CAUTION (score 39.9)
- **Hazard:** PEDESTRIAN_CONFLICT (person)
- **Reason (engine's overall assessment at this time):** Persistent person #1 (observed 2.7 s, 28 detections): image-space box area growing 61%/s, in the central image band.
- **Evidence:** `data/events/local_real_run_01/event_000044`
- **Recommended review/action:** Review pedestrian interaction and maintain additional visual attention in similar situations.

**event_000045** (RISK_ESCALATED, INFERRED):
- **Time:** 2.92 s into video_20260926_173058.mp4
- **Camera:** front
- **Risk:** HIGH (score 49.8), CAUTION->HIGH
- **Hazard:** APPROACHING_VEHICLE (bicycle)
- **Reason:** Persistent bicycle #3 (observed 2.8 s, 24 detections): image-space box area growing 249%/s, in the central image band (+1 other road hazard).
- **Evidence:** `data/events/local_real_run_01/event_000045`
- **Recommended review/action:** Review following/approach context and maintain additional observation of closing traffic.

**event_000009** (RISK_ESCALATED, INFERRED):
- **Time:** 0.80 s into video_20260926_210423.mp4
- **Camera:** front
- **Risk:** HIGH (score 49.6), CAUTION->HIGH
- **Hazard:** PEDESTRIAN_CONFLICT (person)
- **Reason:** Persistent person #2 (observed 0.8 s, 9 detections): image-space box area growing 113%/s, moving across the central band at 0.06 frame-widths/s (image space), in the central image band (+1 other road hazard).
- **Evidence:** `data/events/local_real_run_01/event_000009`
- **Recommended review/action:** Review pedestrian interaction and maintain additional visual attention in similar situations.

**event_000012** (PERSISTENT_HAZARD, INFERRED):
- **Time:** 2.70 s into video_20260926_210536.mp4
- **Camera:** front
- **Risk:** CAUTION (score 47.1)
- **Hazard:** PEDESTRIAN_CONFLICT (person)
- **Reason (engine's overall assessment at this time):** Persistent person #1 (observed 2.7 s, 28 detections): image-space box area growing 111%/s, in the central image band.
- **Evidence:** `data/events/local_real_run_01/event_000012`
- **Recommended review/action:** Review pedestrian interaction and maintain additional visual attention in similar situations.

**event_000013** (RISK_ESCALATED, INFERRED):
- **Time:** 2.90 s into video_20260926_210536.mp4
- **Camera:** front
- **Risk:** HIGH (score 53.6), CAUTION->HIGH
- **Hazard:** PEDESTRIAN_CONFLICT (person)
- **Reason:** Persistent person #1 (observed 2.9 s, 30 detections): image-space box area growing 91%/s, moving across the central band at 0.11 frame-widths/s (image space), in the central image band.
- **Evidence:** `data/events/local_real_run_01/event_000013`
- **Recommended review/action:** Review pedestrian interaction and maintain additional visual attention in similar situations.

**event_000049** (RISK_ESCALATED, INFERRED):
- **Time:** 1.40 s into video_20260926_211717.mp4
- **Camera:** front
- **Risk:** HIGH (score 53.9), CAUTION->HIGH
- **Hazard:** PEDESTRIAN_CONFLICT (person)
- **Reason:** Persistent person #1 (observed 1.4 s, 15 detections): image-space box area growing 180%/s, moving across the central band at 0.10 frame-widths/s (image space), in the central image band (+3 other road hazards).
- **Evidence:** `data/events/local_real_run_01/event_000049`
- **Recommended review/action:** Review pedestrian interaction and maintain additional visual attention in similar situations.

**event_000050** (PERSISTENT_HAZARD, INFERRED):
- **Time:** 2.50 s into video_20260926_211717.mp4
- **Camera:** front
- **Risk:** HIGH (score 47.5)
- **Hazard:** PEDESTRIAN_CONFLICT (person)
- **Reason (engine's overall assessment at this time):** Persistent person #1 (observed 2.5 s, 26 detections): image-space box area growing 125%/s, in the central image band (+3 other road hazards).
- **Evidence:** `data/events/local_real_run_01/event_000050`
- **Recommended review/action:** Review pedestrian interaction and maintain additional visual attention in similar situations.

**event_000051** (PERSISTENT_HAZARD, INFERRED):
- **Time:** 2.50 s into video_20260926_211717.mp4
- **Camera:** front
- **Risk:** HIGH (score 47.5)
- **Hazard:** APPROACHING_VEHICLE (car)
- **Reason (engine's overall assessment at this time):** Persistent person #1 (observed 2.5 s, 26 detections): image-space box area growing 125%/s, in the central image band (+3 other road hazards).
- **Evidence:** `data/events/local_real_run_01/event_000051`
- **Recommended review/action:** Review following/approach context and maintain additional observation of closing traffic.

**event_000056** (DRIVER_STATE_CHANGE, OBSERVED):
- **Time:** 2.20 s into video_20260926_220714.mp4
- **Camera:** driver
- **Risk:** not recorded
- **Hazard:** hand_state: ONE_HAND->NO_HANDS
- **Reason:** hand_state changed ONE_HAND->NO_HANDS (held for the configured observations)
- **Evidence:** `data/events/local_real_run_01/event_000056`
- **Recommended review/action:** Review the driving segment and maintain both hands on the steering wheel when appropriate.

**event_000057** (DRIVER_STATE_CHANGE, INFERRED):
- **Time:** 2.30 s into video_20260926_220714.mp4
- **Camera:** driver
- **Risk:** not recorded
- **Hazard:** drowsiness_level: HIGH->CRITICAL
- **Reason:** drowsiness_level changed HIGH->CRITICAL (held for the configured observations)
- **Evidence:** `data/events/local_real_run_01/event_000057`
- **Recommended review/action:** Treat prolonged eye closure as a driver-attention warning; review rest/alertness before continued driving.

**event_000060** (DRIVER_STATE_CHANGE, INFERRED):
- **Time:** 5.20 s into video_20260926_220714.mp4
- **Camera:** driver
- **Risk:** not recorded
- **Hazard:** drowsiness_level: CRITICAL->HIGH
- **Reason:** drowsiness_level changed CRITICAL->HIGH (held for the configured observations)
- **Evidence:** `data/events/local_real_run_01/event_000060`
- **Recommended review/action:** Treat prolonged eye closure as a driver-attention warning; review rest/alertness before continued driving.

**event_000061** (DRIVER_STATE_CHANGE, OBSERVED):
- **Time:** 5.50 s into video_20260926_220714.mp4
- **Camera:** driver
- **Risk:** not recorded
- **Hazard:** hand_state: NO_HANDS->ONE_HAND
- **Reason:** hand_state changed NO_HANDS->ONE_HAND (held for the configured observations)
- **Evidence:** `data/events/local_real_run_01/event_000061`
- **Recommended review/action:** Informational: hand-position change recorded; no action implied on its own.

**event_000063** (DRIVER_STATE_CHANGE, INFERRED):
- **Time:** 3.20 s into video_20260926_220747.mp4
- **Camera:** driver
- **Risk:** not recorded
- **Hazard:** drowsiness_level: HIGH->CRITICAL
- **Reason:** drowsiness_level changed HIGH->CRITICAL (held for the configured observations)
- **Evidence:** `data/events/local_real_run_01/event_000063`
- **Recommended review/action:** Treat prolonged eye closure as a driver-attention warning; review rest/alertness before continued driving.

**event_000064** (DRIVER_STATE_CHANGE, INFERRED):
- **Time:** 5.20 s into video_20260926_220747.mp4
- **Camera:** driver
- **Risk:** not recorded
- **Hazard:** drowsiness_level: CRITICAL->HIGH
- **Reason:** drowsiness_level changed CRITICAL->HIGH (held for the configured observations)
- **Evidence:** `data/events/local_real_run_01/event_000064`
- **Recommended review/action:** Treat prolonged eye closure as a driver-attention warning; review rest/alertness before continued driving.

**event_000066** (DRIVER_STATE_CHANGE, INFERRED):
- **Time:** 2.30 s into video_20260926_220840.mp4
- **Camera:** driver
- **Risk:** not recorded
- **Hazard:** drowsiness_level: HIGH->CRITICAL
- **Reason:** drowsiness_level changed HIGH->CRITICAL (held for the configured observations)
- **Evidence:** `data/events/local_real_run_01/event_000066`
- **Recommended review/action:** Treat prolonged eye closure as a driver-attention warning; review rest/alertness before continued driving.

## 6. Post-event summary

66 real events were recorded across 16 session(s) (front 54, driver 12 camera). Risk levels on level-carrying events: SAFE 9, CAUTION 34, HIGH 16, CRITICAL 0. The most frequent recorded pattern was pedestrian hazards (21 events in 12 session(s)). Main review points: (1) Review pedestrian interaction and maintain additional visual attention in similar situations. (2) Review following/approach context and maintain additional observation of closing traffic. (3) Treat prolonged eye closure as a driver-attention warning; review rest/alertness before continued driving. These are descriptive observations from a prototype; they are not collision predictions, safety validations, medical assessments or findings of fault.

_(summary backend: deterministic)_

## 7. Limitations of this analysis

- **Scope:** descriptive statistics of what the prototype recorded on short, separate real clips (road-only or driver-only). They are not safety outcomes; there is no collision or near-miss ground truth.
- **Inherited errors:** events inherit errors from the detector, the tracker, the hand-state classifier and the rule thresholds.
  - The hand classifier generalises poorly to the driver-camera setup.
  - A real NO_HANDS event was confirmed as a misclassification (see `event_dataset.md`), so hands-off recommendations need checking against the evidence frame.
- **Ego-motion:** road risk is image-space. Camera ego-motion can look like an approaching object.
- **No location:** there is no GPS and no absolute clock, so there is no geographic or time-of-day analysis.
- **Drowsiness is a proxy:** drowsiness levels are a prototype eye-closure signal, not a medical assessment.

