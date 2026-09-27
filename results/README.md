# Result artifacts

| File | What it shows | Provenance |
| --- | --- | --- |
| `images/pedestrian_timeline_t0_0.5s.jpg`, `…t1_1.5s.jpg`, `…t2_2.9s.jpg` | Frames of one real pedestrian clip (`video_20260926_210536`) at 0.5 s / 1.5 s / 2.9 s. The box marks the tracked pedestrian | LOCAL_REAL frames; box drawn for illustration |
| `images/evidence_event_000012_pedestrian.jpg` | The evidence frame the recorder saved for `event_000012` (PERSISTENT_HAZARD, CAUTION, score 47.1) | LOCAL_REAL, unaltered |
| `images/evidence_event_000022_vehicle_plate_blurred.jpg` | Evidence frame for `event_000022` (PERSISTENT_HAZARD, approaching vehicle, HIGH) | LOCAL_REAL, **number plate blurred** (derived copy) |
| `plots/demo_front_risk_timeline.png` | Real rule-based risk score and level over the 11.5 s front-camera demo clip, with events and alarms | computed from LOCAL_REAL video |
| `plots/demo_driver_risk_timeline.png` | The same for the separate driver-camera demo clip | computed from LOCAL_REAL video |
| `plots/tracking_pedestrian_and_dog.jpg` | Tracker output on a pedestrian crossing and a black dog at night (label flicker visible) | LOCAL_REAL frames + tracker annotations |
| `plots/hand_confusion_matrix.png` | Hand-state classifier test confusion matrix (27 images) | test split of the team's images |

The annotated demo videos are not published (see [DEMO.md](../DEMO.md)).
