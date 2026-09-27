# Road perception: first run on real front-camera data

Model: **YOLO26n** (Ultralytics 8.4.163, COCO-pretrained, not fine-tuned), CPU, confidence ≥ 0.35, image size 640, videos sampled at 5 fps (video time). Road-relevant COCO classes only: person, bicycle, car, motorcycle, bus, train, truck, traffic light, stop sign, cat, dog, horse, sheep, cow, elephant.

Command (read-only on the dataset):

```bash
python scripts/test_road_perception.py --source "<DATASET_ROOT>/frontcamera-20260926T193501Z-1-001/frontcamera" --save-annotated
```

This run covered a representative subset: all 15 images and 10 of the 26 videos (every dogs_on_road video; 2–3 per other folder). Run in the development container (Linux, torch limited to 1 CPU thread), not yet on the team laptop.

## Summary

| | |
| --- | --- |
| Files | 25 (15 images, 10 videos) |
| Video frames | 820 read, 114 sent to YOLO |
| Frames processed | 129, all status OK |
| Frames with no detection | 13 |
| Detections | 365 |
| Model load / warm-up | 0.2 s / 1.4 s, once |
| Inference per frame (CPU, 1 thread) | mean 69 ms, median 68 ms, p95 87 ms, max 102 ms (~14 frames/s) |

| Class | Detections | Frames containing it | Confidence min–mean–max |
| --- | ---: | ---: | --- |
| person | 226 | 106 | 0.35–0.64–0.91 |
| car | 53 | 34 | 0.36–0.65–0.93 |
| bicycle | 28 | 23 | 0.36–0.52–0.66 |
| motorcycle | 27 | 18 | 0.35–0.53–0.74 |
| truck | 17 | 17 | 0.37–0.52–0.69 |
| dog | 10 | 10 | 0.36–0.69–0.92 |
| traffic light | 2 | 2 | 0.45–0.52 |
| cat | 1 | 1 | 0.65 |
| bus | 1 | 1 | 0.48 |

By folder. Folder names are human context, **not** ground truth; these are counts, not accuracy:

| Folder | Files | Frames | Frames with nothing | Detections |
| --- | ---: | ---: | ---: | --- |
| dogs_on_road | 3 videos | 29 | 3 | person 21, dog 10, car 3, traffic light 2, cat 1 |
| pedestrains | 5 images + 2 videos | 24 | 0 | person 79, bicycle 12 |
| roads | 7 images + 2 videos | 29 | 10 | person 29, car 5 |
| vehicles | 3 images + 3 videos | 47 | 0 | person 97, car 45, motorcycle 27, truck 17, bicycle 16, bus 1 |

Sample annotated frames: `outputs/plots/road_perception_samples.jpg`.

## Failure modes seen on inspection (not a measured error rate)

- **Black dog at night is labelled "person".** In `dogs_on_road/video_20260926_211127.mp4` the dog is the main object in all 13 sampled frames. The dog's box is labelled person in 10 of them; in one of those the same box also gets dog at 0.64. It is labelled cat in 1 frame and not detected in 1 frame. The dog in `video_20260926_211145.mp4`, on a lit red surface, is detected as dog at 0.58–0.92. Dark animals at night are unreliable, and the label can flip between frames.
- **Two-wheeler under headlight glare.** In `vehicles/IMG_20260926_172431.jpg` the scooter rider is detected as person (0.41), but the scooter itself is missed.
- **Close vehicle filling the frame.** In `vehicles/video_20260926_211739.mp4` the rear of a car at very close range gets full-width boxes labelled both car (0.79) and bus (0.48). Passengers visible through the rear window are detected as persons.
- **Small or distant objects are missed.** A distant cyclist and a roadside bin in `roads/IMG_20260926_172652.jpg` are not detected at 640 px inference on a 4608 px wide image.
- **Motion blur.** Several still images have strong blur. Empty-road images (`roads/`) correctly produced no detections, so there was no obvious false alarm on the road surface itself.
- **Label flicker across frames** for the same object (person / dog / cat) is expected without tracking. A tracker with class voting would be the next step.

## Limitations

- COCO-pretrained YOLO26n, not adapted to Indian roads, night scenes or our camera; the numbers above are counts, not accuracy. No ground-truth boxes exist.
- Timing is from a 1-thread container. The laptop (Intel Iris Xe CPU, multi-core) should be similar or faster; confirm with the same command.
- Detection only: no tracking, distance, time-to-collision (TTC), risk or alarms in this layer.
