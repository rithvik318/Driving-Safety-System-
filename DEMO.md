# Demo

## Status

| Item | Status |
| --- | --- |
| Integrated demo runner | `scripts/run_demo.py`: implemented and run on real videos |
| Front-camera demo run | `vehicles/video_20260926_170005.mp4` (LOCAL_REAL, 11.5 s). Outputs summarised below |
| Driver-camera demo run | `drowsy_driver/video_20260926_220714.mp4` (LOCAL_REAL, 6.3 s, staged setup). Separate run, **not synchronized** with any road video |
| Annotated demo videos | Generated locally (`annotated_demo.mp4`) but **not published**: they show readable number plates, bystanders and a team member's face. Publishing a blurred version is a team decision |
| Recorded demo link | **Not yet available.** No URL exists; add it here when the recording is published |

## Intended 3-minute flow

| Time | Segment | What is shown (real outputs) |
| --- | --- | --- |
| 0:00–0:20 | Physical problem / driving scene | A real front-camera clip; "seeing objects is not enough, the situation changes" |
| 0:20–1:00 | Driver + road perception | YOLO26n detections on the road clip; the driver clip with face landmarks, eye closure and hand state (separate recording) |
| 1:00–1:40 | Temporal tracking and contextual risk | Track IDs persisting; the risk timeline ([plot](results/plots/demo_front_risk_timeline.png)) rising from SAFE to CAUTION to HIGH; the pedestrian sequence in [README §7](README.md#7-how-temporal-reasoning-changes-the-system) |
| 1:40–2:20 | Alert / intervention | Console alerts: `HIGH — …` + `ALARM — WARNING: ROAD HAZARD`, each explained by the engine's `risk_reason` |
| 2:20–2:50 | Structured event appears | The event row (JSONL/Parquet) next to its evidence frame and `metadata.json` |
| 2:50–3:00 | Dataset / analytics / takeaway | 66 real events, provenance labels, post-event safety summary; "physical observation → decision → traceable data" |

## Real demo results

**Front camera** (`video_20260926_170005.mp4`, 10 fps):

| Metric | Value |
| --- | --- |
| Frames processed | 115 (of 624 decoded) |
| Detections / tracks | 613 / 148 |
| Frames per level | SAFE 8 · CAUTION 77 · HIGH 30 |
| Events | 12: 5 escalations, 5 de-escalations, 2 persistent hazards |
| Alerts | 1 CAUTION notice; 4 "WARNING: ROAD HAZARD" with alarm |
| Evidence | 14 files for 7 events |

**Console excerpt:**

```text
[00:00.5] CAUTION — pedestrian in or moving toward the central road region
[00:01.5] HIGH — pedestrian in or moving toward the central road region
[00:01.5] ALARM — WARNING: ROAD HAZARD
[00:04.6] HIGH — approaching vehicle / increasing image-space box
[00:06.9] HIGH — approaching vehicle / increasing image-space box
[00:07.3] EVENT PERSISTENT_HAZARD event_000009 → evidence demo_video_20260926_170005/event_000009
[00:11.2] SAFE — risk lowered from CAUTION
```

**Driver camera** (`video_20260926_220714.mp4`, 10 fps):

| Metric | Value |
| --- | --- |
| Driver observations | 63 |
| Events | 7: 2 CAUTION escalations (drowsiness), 1 de-escalation, 4 driver-state changes |
| Alarms | 0 (a single driver factor cannot reach HIGH by design) |
| Known issue | one hand-state change there is a known classifier error |

**Risk timeline plots:** [front](results/plots/demo_front_risk_timeline.png), [driver](results/plots/demo_driver_risk_timeline.png). Full report: [reports/final_demo_report.md](reports/final_demo_report.md).

## Reproduce

Requires the raw videos, which are not public:

```bash
python scripts/run_demo.py --camera driver --video "<DATASET_ROOT>/drivercamera-…/drivercamera/drowsy_driver/video_20260926_220714.mp4" --output-dir outputs/demo_driver --fps 10
python scripts/run_demo.py --video "<DATASET_ROOT>/frontcamera-…/frontcamera/vehicles/video_20260926_170005.mp4" --output-dir outputs/demo --fps 10 --device auto --include-summary outputs/demo_driver/demo_summary.json
```

Add `--beep` to ring the terminal bell on alarms, and `--no-video` to skip the annotated video.
