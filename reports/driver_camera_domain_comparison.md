# Driver-camera domain comparison

Purpose: show how the hand-state **training images** and the **real driver video** differ visually (camera viewpoint, framing, resolution, lighting). This report does not evaluate the classifier.

Comparison sheet: `outputs/plots/driver_camera_domain_comparison.jpg`

Sources were only read. EXIF orientation was applied to in-memory copies; video frames use OpenCV's automatic orientation handling.

## 1. Samples

Training samples: per class, exact duplicates removed, then seeded k-means (k = 3) on 32x32 grayscale thumbnails; the image nearest each cluster centre is shown. "Represents" = cluster size. Video frames: evenly spaced at 20 %, 50 % and 80 % of the duration.

| # | Source type | Label | Timestamp | Original (stored) | Shown | Aspect | Orientation | Mean luma | Luma std | File |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | TRAINING_IMAGE | both_hands_on_steering (represents 15/30) | - | 1080x1920 | 1920x1080 | 16:9 landscape | EXIF 8 (rotate 90 CCW applied) | 141.6 | 76.3 | `Copy of Normal_Driving-vid_7-frame_0526.jpg` |
| 2 | TRAINING_IMAGE | both_hands_on_steering (represents 10/30) | - | 1080x1920 | 1920x1080 | 16:9 landscape | EXIF 8 (rotate 90 CCW applied) | 147.5 | 76.6 | `Copy of Passenegr_Interaction-vid_5-frame_0912.jpg` |
| 3 | TRAINING_IMAGE | both_hands_on_steering (represents 5/30) | - | 1080x1920 | 1920x1080 | 16:9 landscape | EXIF 8 (rotate 90 CCW applied) | 146.4 | 74.9 | `Copy of Normal_Driving-vid_7-frame_0539.jpg` |
| 4 | TRAINING_IMAGE | one_hand_on_steering (represents 41/97) | - | 720x1280 | 720x1280 | 9:16 portrait | no EXIF orientation | 110.9 | 59.2 | `Copy of jumon-vid_0-frame_0816.jpg` |
| 5 | TRAINING_IMAGE | one_hand_on_steering (represents 36/97) | - | 1080x1920 | 1920x1080 | 16:9 landscape | EXIF 8 (rotate 90 CCW applied) | 145.5 | 76.4 | `Copy of Using_Electronics_Items-vid_7-frame_0555.jpg` |
| 6 | TRAINING_IMAGE | one_hand_on_steering (represents 20/97) | - | 1080x1920 | 1920x1080 | 16:9 landscape | EXIF 8 (rotate 90 CCW applied) | 140.7 | 75.5 | `Copy of Using_Electronics_Items-vid_7-frame_1151.jpg` |
| 7 | TRAINING_IMAGE | no_hands (represents 26/61) | - | 1080x1920 | 1920x1080 | 16:9 landscape | EXIF 8 (rotate 90 CCW applied) | 160.8 | 74.4 | `Copy of Using_Electronics_Items-vid_4-frame_1031.jpg` |
| 8 | TRAINING_IMAGE | no_hands (represents 18/61) | - | 848x480 | 480x848 | 9:16 portrait | EXIF 8 (rotate 90 CCW applied) | 122.0 | 51.0 | `Copy of frame_0240.jpg` |
| 9 | TRAINING_IMAGE | no_hands (represents 17/61) | - | 1080x1920 | 1920x1080 | 16:9 landscape | EXIF 8 (rotate 90 CCW applied) | 133.7 | 69.6 | `Copy of Using_Electronics_Items-vid_4-frame_1422.jpg` |
| 10 | REAL_VIDEO | none (unlabelled) | 1.25 s (frame 75) | 3840x2160 | 3840x2160 | 16:9 landscape | rotation meta 0 | 136.2 | 51.6 | `video_20260926_220714.mp4` |
| 11 | REAL_VIDEO | none (unlabelled) | 3.13 s (frame 188) | 3840x2160 | 3840x2160 | 16:9 landscape | rotation meta 0 | 135.9 | 51.1 | `video_20260926_220714.mp4` |
| 12 | REAL_VIDEO | none (unlabelled) | 5.02 s (frame 301) | 3840x2160 | 3840x2160 | 16:9 landscape | rotation meta 0 | 132.1 | 54.4 | `video_20260926_220714.mp4` |

Class folders:

- `both_hands_on_steering`: 30 files, 0 exact duplicate(s) excluded from selection, 0 unreadable
- `one_hand_on_steering`: 97 files, 0 exact duplicate(s) excluded from selection, 0 unreadable
- `no_hands`: 62 files, 1 exact duplicate(s) excluded from selection, 0 unreadable

Video: 3840x2160, 59.998 fps, 377 frames, 6.28 s, rotation metadata 0.

## 2. Measured summary

| Group | Shown resolution(s) | Aspect(s) | Orientation handling | Mean luma range | Luma std range |
| --- | --- | --- | --- | --- | --- |
| BOTH_HANDS | 1920x1080 | 16:9 landscape | EXIF 8 (rotate 90 CCW applied) | 141.6-147.5 | 74.9-76.6 |
| ONE_HAND | 1920x1080, 720x1280 | 16:9 landscape, 9:16 portrait | EXIF 8 (rotate 90 CCW applied); no EXIF orientation | 110.9-145.5 | 59.2-76.4 |
| NO_HANDS | 1920x1080, 480x848 | 16:9 landscape, 9:16 portrait | EXIF 8 (rotate 90 CCW applied) | 122.0-160.8 | 51.0-74.4 |
| REAL_VIDEO | 3840x2160 | 16:9 landscape | rotation meta 0 | 132.1-136.2 | 51.1-54.4 |

Luma = 0.299R + 0.587G + 0.114B on the full-resolution image (0-255). Mean luma is overall brightness; std is a rough global-contrast measure.

## 3. Visual observations

Written after inspecting the comparison sheet. The notes describe only what is visible in the 12 samples shown, plus the full-class resolution counts from the audit. They make no claim about classifier behaviour.

**Setups in the training data.** The sampled images come from three visibly different setups:

- **(A) Office / lab rig:** a gaming wheel on a stand, recorded at 1920x1080. This is all 30 BOTH_HANDS, 56 of 97 ONE_HAND and 43 of 62 NO_HANDS images.
- **(B) Inside a real car, portrait:** 720x1280 images. These appear only in ONE_HAND (41 images).
- **(C) Inside a real car, portrait:** 480x848 images. These appear only in NO_HANDS (19 images).

The real video matches none of these setups.

| Aspect | Training images | Real video |
| --- | --- | --- |
| Camera viewpoint | (A) Frontal, camera roughly at chest height and about level, a few metres back. (B) and (C) Frontal from the dashboard, close to the driver, in portrait. | Frontal and close, mounted low near the wheel and angled upward: the ceiling, a ceiling fan and tube lights are in frame. |
| Steering-wheel visibility | (A) The whole wheel rim, hub and gear lever are visible in the lower centre. (B) The wheel is not clearly visible in the sample. (C) The upper arc of a real car wheel fills the lower foreground. | A round black disc with raised lettering is used as the wheel. It looks like a gym weight plate. Only its upper half is in frame; the bottom is cut off by the frame edge. |
| Hand location and scale | (A) Hands sit on the rim in the lower-middle and are small relative to the frame. Hands off the wheel are visible at face height (phone, bottle). | The hands are at the bottom edge of the frame. The driver's right hand (image right) is visible on the disc and looks larger relative to the frame than in (A). The other hand is not visible in the three sampled frames. |
| Driver body visibility | (A) Head to lap: both arms, torso, seat and part of the legs. (B) and (C) Head to chest. | Head to upper chest; the lap and lower torso are below the frame. |
| Background / environment | (A) An office with white walls, cabinets, desks, a monitor and a whiteboard; other people are sometimes at the frame edges. (B) and (C) Car interiors with windows. | A gym: exercise machines and a weight rack, a cream wall, blue curtains, a ceiling fan and lights. |
| Image orientation | Stored rotated: 1080x1920 or 848x480 with EXIF orientation 8, or 720x1280 portrait with no EXIF. All are upright once EXIF is applied (checked visually). | Stored 3840x2160 with rotation metadata 0; upright as decoded. |
| Resolution / aspect ratio | 1920x1080 (16:9 landscape), 720x1280 and 480x848 (9:16 portrait). | 3840x2160 (16:9 landscape): twice the linear resolution of the largest training images. The classifier pads every input to a square and resizes it to 224 px, so the absolute resolution largely disappears. The landscape/portrait split and the framing still matter. |
| Lighting | (A) Bright, even and neutral-white, with mean luma 134–161. (B) and (C) Car interiors with window light, mean luma 111–122. | Indoor artificial light with a warm/yellow cast, lights visible in frame, mean luma 132–136. Global contrast is lower: luma std 51–54, against 74–77 for (A). |

**Obvious domain differences between the video and the closest training setup (A):**

- different location and background;
- the camera is lower, closer and angled upward;
- the framing is tighter, so the lap and the lower half of the wheel are cut off;
- the "wheel" is a different object (a disc, not a gaming wheel with a hub and shifter);
- fewer hands are visible in frame;
- a different person and clothing;
- warmer, lower-contrast lighting;
- 4K rather than 1080p.

Within the training data, the three setups are not spread evenly across the classes: the portrait car setups appear in only one class each.
