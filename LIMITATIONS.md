# Known limitations

This is an early-stage prototype. These are engineering limitations, stated so reviewers can judge the results correctly.

## Data

- **Small real data volume.** There are 240 raw files, ≈131.7 s of video and 66 real events. The front clips are 1–11 s long, and each starts from SAFE, so short clips over-represent escalations.
- **No synchronized driver + front recording.** Real events are road-only or driver-only. The combined driver + road hazard path (the only real route to CRITICAL besides a very strong road hazard) is exercised only with labelled synthetic input.
- **No real GPS.** Every real event has null coordinates and `gps_source = UNAVAILABLE`. There is no geographic hotspot analysis. Synthetic GPS exists only in the synthetic dataset and is labelled as such.
- **No ground truth.** There are no collision, near-miss, bounding-box or track labels. Event counts describe rule behaviour, not safety outcomes or accuracy.
- **Undocumented origin of some hand-state frames.** Some image file names indicate frames extracted from source videos (e.g. `Normal_Driving-vid_7-frame_…`). Those source videos and their capture details are not part of the dataset.

## Driver perception

- **Hand-state dataset domain differences.** The three classes come from different capture setups (resolution/orientation differ per class), so the classifier may partly learn the setup.
- **No person/session-grouped split.** Source-video or person groups could not be established reliably, so the split is image-level. Near-duplicate consecutive frames can appear on both sides of it.
- **Test metrics are indicative only.** The 1.0 test accuracy is on 27 images and is not evidence of generalization.
- **Domain shift in the real driver demo.** The drowsy-driver recordings use a different, staged setup, and the classifier's output there was unreliable. One real event (`event_000056`, hand ONE_HAND→NO_HANDS) records an **incorrect NO_HANDS state**: a hand is visible on the wheel.
- **Prototype drowsiness signal.** The drowsiness score is a behavioural eye-closure proxy (rolling closure ratio, long closures). It is **not medically validated** and is not a diagnosis.

## Road perception and tracking

- **Image-space only.** There is no camera calibration, so no physical distance, speed or TTC is estimated anywhere. "Approach" means the box grows in the image (slope of ln(area)), and the "central band" is a stand-in for "ahead", not a path model.
- **Camera motion looks like approach.** When the recording camera moves toward a parked object, the object grows in the image. Several real HIGH levels are influenced by this ego-motion.
- **Detector errors propagate.** Pretrained YOLO26n (not fine-tuned) mislabels some objects (e.g. a black dog at night as "person") and misses small, distant or glare-affected objects. Label flicker and misses flow into tracks, risk and events.
- **Some false detections occur.** Examples include duplicate labels on close vehicles (car and bus) and a very young track with an extreme growth estimate that triggered a HIGH warning in the demo.
- **Track fragmentation.** Busy scenes and label changes split objects into many short tracks (148 tracks from 613 detections in the 11.5 s demo clip).

## Risk engine and alerts

- **Rule-based prototype thresholds.** The engine is deterministic and configurable, but its weights and thresholds are prototype choices, not tuned or validated on real outcomes. The score is **not a calibrated probability** and not a collision prediction.
- **Offline alarm.** The alert layer is a console / terminal-bell output in offline replay. There is no in-vehicle integration and no latency measurement. Event schema 1.0 keeps `alarm_triggered = false`; raised alarms are logged separately.
- **Synthetic validation is circular.** Synthetic scenarios test that the rules behave as designed on constructed inputs. They do not show the rules are correct on real roads.

## Privacy

- **Withheld from the public repository:**
  - raw recordings;
  - driver-camera frames (identifiable team-member face);
  - road frames with readable number plates or bystanders;
  - the annotated demo videos.
- **Published images:** only a few road evidence frames are included, plus one derived frame with its plate blurred.
