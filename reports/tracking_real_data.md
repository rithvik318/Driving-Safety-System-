# Road-object tracking: first run on real front-camera videos

Pipeline: YOLO26n (pretrained COCO, unchanged; conf ≥ 0.35, imgsz 640, CPU) → `IoUTracker.update(detections, timestamp)`.

Tracker settings:
- same-class match at IoU ≥ 0.3;
- a different-class detection may join a track only at IoU ≥ 0.6, recorded as a label change and never rewritten;
- unmatched tracks coast for up to 1.0 s before they end;
- one new tracker per video.

**All motion values are image-space** (pixels, pixels/s, pixels²/s). They are not metres, km/h, distance or time-to-collision.

Command (read-only on the dataset):

```bash
python scripts/test_road_tracking.py --source "<DATASET_ROOT>/frontcamera-20260926T193501Z-1-001/frontcamera" --save-annotated
```

## Data used

This run used 14 of the 26 front-camera videos: all 3 dogs_on_road, 3 pedestrains, 2 roads and 6 vehicles, including the two longest (11.5 s and 9.7 s). The videos are either 1080p landscape (42–60 fps) or 4K portrait (30 fps). The folder names are how the videos were collected; they are **not** ground truth. There are no ground-truth tracks, so the numbers below describe tracker behaviour, not accuracy.

| | 10 fps (default for tracking) | 5 fps (comparison) |
| --- | ---: | ---: |
| Video frames read | 2262 | 2262 |
| Frames sent to YOLO | 515 | 260 |
| Detections | 1616 | 818 |
| Tracks created | 312 | 269 |
| Tracks with ≥ 2 detections | 182 (58 %) | 125 (46 %) |
| Single-detection tracks | 130 (42 %) | 144 (54 %) |
| Mean persistence (tracks with ≥ 2 detections) | 0.92 s | 1.02 s |
| Longest-lived track | 5.61 s (57 det) | 5.01 s (26 det) |
| YOLO / tracker time per frame | 64 ms / 0.24 ms | 65 ms / 0.24 ms |

**Why 10 fps for tracking.** In `pedestrains/video_20260926_210536.mp4`, at 5 fps, a pedestrian walking toward the camera nearly doubled in box area between two samples (1.09 M → 2.01 M px²). The IoU fell below 0.3 and the track restarted as a new ID (1 → 13). At 10 fps the same person keeps ID 1 for the whole 3.3 s (34 detections). YOLO at about 64 ms per frame still keeps up with 10 fps on CPU. The detection-only default (`ROAD_SAMPLE_FPS=5`) is unchanged.

## Per video (10 fps)

| Folder (context) | Video | Duration s | Frames | Tracks | ≥ 2 det | Longest track |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dogs_on_road | video_20260926_211127.mp4 | 2.4 | 25 | 11 | 4 | ID 2 person 2.0 s (18 det) |
| dogs_on_road | video_20260926_211145.mp4 | 1.5 | 16 | 4 | 3 | ID 1 dog 0.8 s (9 det) |
| dogs_on_road | video_20260926_211200.mp4 | 1.5 | 16 | 4 | 2 | ID 2 person 0.9 s (10 det) |
| pedestrains | video_20260926_210423.mp4 | 1.8 | 19 | 4 | 4 | ID 1 person 1.8 s (19 det) |
| pedestrains | video_20260926_210536.mp4 | 3.3 | 34 | 12 | 6 | ID 1 person 3.3 s (34 det) |
| pedestrains | video_20260926_211603.mp4 | 1.7 | 18 | 17 | 13 | ID 1 person 1.5 s (15 det) |
| roads | video_20260926_165942.mp4 | 1.8 | 19 | 20 | 12 | ID 1 person 1.8 s (5 det) |
| roads | video_20260926_210938.mp4 | 2.3 | 24 | 7 | 4 | ID 2 person 0.9 s (3 det) |
| vehicles | video_20260926_165951.mp4 | 3.5 | 36 | 25 | 15 | ID 1 car 3.5 s (36 det) |
| vehicles | video_20260926_170005.mp4 | 11.4 | 115 | 148 | 73 | ID 2 person 5.6 s (57 det) |
| vehicles | video_20260926_172658.mp4 | 3.8 | 39 | 31 | 26 | ID 1 truck 3.7 s (31 det) |
| vehicles | video_20260926_173058.mp4 | 9.7 | 98 | 8 | 4 | ID 1 person 4.1 s (42 det) |
| vehicles | video_20260926_211717.mp4 | 4.3 | 44 | 14 | 10 | ID 1 person 4.3 s (44 det) |
| vehicles | video_20260926_211739.mp4 | 1.1 | 12 | 7 | 6 | ID 1 car 1.1 s (12 det) |

## Track statistics (10 fps, all videos)

- **Tracks by class** (track label = most frequent observed class): person 156, car 67, bicycle 53, motorcycle 19, truck 9, dog 4, traffic light 3, cat 1.
- **Maximum image-space speed:** 2831 px/s, which is 1.31 frame-widths/s (`210423` ID 1, a person close to the camera, 2160 px wide frame). Pixel speed depends on resolution and distance and is not comparable across videos or to real speed.
- **Growing bbox area:** 61 tracks have last/first area ≥ 1.2 over ≥ 3 detections. Examples: the parked car approached in `165951` (ID 1, ×13.5 over 3.5 s) and the pedestrian in `210536` (ID 1, ×13.9 over 3.3 s). This is a relative "getting bigger in the image" signal only.
- **Frame-to-frame area is noisy.** For the `210536` pedestrian it went 340k → 233k → 445k px² within 0.4 s, because YOLO boxes jitter and the person's pose changes. Raw `area_change_rate` values should be smoothed before any later use.

## Class-switch observations (10 fps)

- 13 of 312 tracks changed observed class at least once: 6 once, 4 twice, 3 three or more times. Labels were recorded, never rewritten.
- **Night black dog** (`211127`): the dog's track ID 1 history is person 1, cat 3, dog 2, horse 1.
  - At 0.4 s YOLO also gave a second, overlapping box of another class on the same animal. That box started ID 2, which then continued as person 16 / horse 1 / cat 1.
  - So one physical dog is represented by two tracks, and the longer one is labelled "person". The dark-object misclassification seen in detection therefore persists through tracking.
- **Two-wheelers** are the most common switch: bicycle ↔ motorcycle in `211603` (7 bicycle / 1 motorcycle), `172658` (9 / 1), `211717` (6 motorcycle / 2 bicycle, 4 changes) and three tracks in `170005`. A motorcycle track in `165951` was briefly labelled car once (18 / 1).
- **Other one-off confusions:** car ↔ truck and bus ↔ truck (`170005`, `211739`), and car ↔ person and motorcycle ↔ person in the busy scene `170005`. The cause of the person confusions was not verified visually.
- **Overlapping active tracks** (IoU ≥ 0.7 in the same frame): 17 pairs. Some are the duplicate-box case above; others are genuinely overlapping objects, such as a cyclist (person) and their bicycle.

## Visual verification

The annotated videos (development only) show `ID n | class | conf | persistence` per box, with an image-space motion arrow. Sample frames are in `outputs/plots/tracking_samples.jpg`. Rows, top to bottom:

1. **`165951`:** the approached car (ID 1) and the person with a bicycle (ID 2) keep their IDs for the full 3.5 s.
2. **`170005`:** the cyclist ahead keeps ID 2 for 5.6 s, but the bicycle under them is detected intermittently and gets new IDs. Parked bicycles and cars along the roadside create many short tracks.
3. **`210536`:** the pedestrian crossing toward the camera keeps ID 1 at 10 fps.
4. **`211127`:** the black dog at night, split into ID 1 (cat*, coasting) and ID 2 (person*).

## Limitations

- **Image space only:** pixel velocity is not physical velocity, and pixel-area growth is only a relative approach signal. The camera also moves (handheld / vehicle), so all image motion includes ego-motion.
- **No calibration:** there is no camera calibration, no true distance estimation and no true TTC.
- **Detector errors carry into tracks:** YOLO errors propagate into tracking. Missed detections break tracks (42 % of tracks are single-detection, mostly small, distant or partly hidden objects near the 0.35 threshold). Duplicate boxes of different classes on one object create parallel tracks.
- **Night / dark objects:** misclassification of dark objects at night persists across frames (the dog tracked as "person").
- **Simple association:** greedy IoU with no motion prediction or appearance model, so fast image motion between samples can break IDs. 10 fps mitigates this on these videos but does not remove it.
- **No accuracy measure:** there are no ground-truth tracks, and the videos are short (1.1–11.4 s).
- **Not built yet:** risk scoring, TTC, alerts, event logging and driver–road fusion are not part of this step.
