# Dataset audit

- Dataset folder: `Data` (paths below are relative to it)
- Audited: 2026-09-26T19:53:14+00:00 (UTC)
- Provenance: `real_local`
- Tools: python 3.13.4, pillow 12.3.0, opencv 5.0.0

> Readable files are not the same as training-ready data. This report describes what is on disk; it does not assess label quality.

## 1. Dataset overview

| Metric | Count |
| --- | ---: |
| Files | 240 |
| Images (readable) | 204 (204) |
| Videos (readable) | 36 (36) |
| Unsupported | 0 |
| Ignored system files | 0 |
| Empty | 0 |
| Corrupt / unreadable | 0 |
| Exact-duplicate groups | 1 |

Folder layout recognised: `collected_2026_09`

Camera folders found: driver_cam → `drivercamera-20260926T193339Z-1-001/drivercamera/`, front_cam → `frontcamera-20260926T193501Z-1-001/frontcamera/`

## 2. Driver-camera counts

### driver_cam

| Folder | Files | Images | Videos | Unsupported | Ignored | Empty | Corrupt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `both_hands_on_steering` | 31 | 30 | 1 | 0 | 0 | 0 | 0 |
| `one_hand_on_steering` | 97 | 97 | 0 | 0 | 0 | 0 | 0 |
| `no_hands` | 62 | 62 | 0 | 0 | 0 | 0 | 0 |
| `drowsy_driver` | 9 | 0 | 9 | 0 | 0 | 0 | 0 |

## 3. Front-camera counts

### front_cam

| Folder | Files | Images | Videos | Unsupported | Ignored | Empty | Corrupt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dogs_on_road` | 3 | 0 | 3 | 0 | 0 | 0 | 0 |
| `pedestrains` | 11 | 5 | 6 | 0 | 0 | 0 | 0 |
| `roads` | 10 | 7 | 3 | 0 | 0 | 0 | 0 |
| `vehicles` | 17 | 3 | 14 | 0 | 0 | 0 | 0 |

## 4. Image statistics

- Total images: 204 (204 readable)
- Formats: {'JPEG': 204} · extensions: {'.jpg': 204}
- Smallest: 848x480 · largest: 4608x2076 (by pixel area)
- Mean size: 1234.157 × 1678.627 px · distinct dimensions: 5
- Common dimensions: 1080x1920 (129), 720x1280 (41), 848x480 (19), 4608x2076 (14), 2304x4096 (1)
- Aspect ratios: {'9:16': 171, '16:9': 19, 'other (2.22)': 14}
- Orientation: {'portrait': 171, 'landscape': 33} · EXIF-rotated: 148
- File size (bytes): min 87792, mean 501484.034, max 3373128

| Class | Readable | Common dimensions | Orientation |
| --- | ---: | --- | --- |
| `driver_cam/both_hands_on_steering` | 30 | 1080x1920 (30) | {'portrait': 30} |
| `driver_cam/no_hands` | 62 | 1080x1920 (43), 848x480 (19) | {'landscape': 19, 'portrait': 43} |
| `driver_cam/one_hand_on_steering` | 97 | 1080x1920 (56), 720x1280 (41) | {'portrait': 97} |
| `front_cam/pedestrains` | 5 | 4608x2076 (4), 2304x4096 (1) | {'landscape': 4, 'portrait': 1} |
| `front_cam/roads` | 7 | 4608x2076 (7) | {'landscape': 7} |
| `front_cam/vehicles` | 3 | 4608x2076 (3) | {'landscape': 3} |

## 5. Video statistics

| Class | Videos (readable) | Total duration | Min | Max | Mean | Common resolution | Common FPS | Codecs |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| `driver_cam/both_hands_on_steering` | 1 (1) | 2.3 s (0m 02.3s) | 2.267 s | 2.267 s | 2.267 s | 3840x2160 (1) | 60.0 (1) | {'hevc': 1} |
| `driver_cam/drowsy_driver` | 9 (9) | 45.0 s (0m 45.0s) | 4.233 s | 6.284 s | 5.0 s | 3840x2160 (9) | 59.999 (4), 59.997 (2), 59.998 (2) | {'hevc': 9} |
| `front_cam/dogs_on_road` | 3 (3) | 5.6 s (0m 05.6s) | 1.567 s | 2.434 s | 1.867 s | 2160x3840 (3) | 29.998 (2), 29.997 (1) | {'hevc': 3} |
| `front_cam/pedestrains` | 6 (6) | 13.3 s (0m 13.3s) | 1.7 s | 3.4 s | 2.211 s | 2160x3840 (6) | 29.998 (3), 29.997 (3) | {'hevc': 6} |
| `front_cam/roads` | 3 (3) | 9.8 s (0m 09.8s) | 1.849 s | 5.564 s | 3.26 s | 1920x1080 (2), 2160x3840 (1) | 43.26 (1), 59.667 (1), 29.997 (1) | {'h264': 2, 'hevc': 1} |
| `front_cam/vehicles` | 14 (14) | 55.8 s (0m 55.8s) | 1.167 s | 11.495 s | 3.985 s | 2160x3840 (9), 1920x1080 (5) | 29.998 (4), 29.997 (3), 42.437 (1) | {'h264': 5, 'hevc': 9} |
| `ALL` | 36 (36) | 131.7 s (2m 11.7s) | 1.167 s | 11.495 s | 3.658 s | 2160x3840 (19), 3840x2160 (10), 1920x1080 (7) | 29.998 (9), 29.997 (8), 59.999 (4) | {'hevc': 29, 'h264': 7} |

Duration = frame count ÷ FPS as reported by the container. Phone recordings are often variable-frame-rate, so treat durations as approximate.

## 6. Corrupt / unreadable files

None found.

## 7. Exact-duplicate groups (SHA-256)

1 group(s), 1 redundant file(s). Nothing was deleted.

1. 88110 bytes: `drivercamera-20260926T193339Z-1-001/drivercamera/no_hands/Copy of frame_0231.jpg`, `drivercamera-20260926T193339Z-1-001/drivercamera/no_hands/frame_0231.jpg`

## 8. Class distribution

**Driver camera**

| Class | Count | Unit | Share of camera total |
| --- | ---: | --- | ---: |
| `both_hands_on_steering` | 30 | readable images | 15.2% |
| `one_hand_on_steering` | 97 | readable images | 49.0% |
| `no_hands` | 62 | readable images | 31.3% |
| `drowsy_driver` | 9 | readable videos | 4.5% |

**Front camera**

| Class | Count | Unit | Share of camera total |
| --- | ---: | --- | ---: |
| `dogs_on_road` | 3 | readable items (0 img, 3 vid) | 7.3% |
| `pedestrains` | 11 | readable items (5 img, 6 vid) | 26.8% |
| `roads` | 10 | readable items (7 img, 3 vid) | 24.4% |
| `vehicles` | 17 | readable items (3 img, 14 vid) | 41.5% |

Counts only. No balance judgement is made here.

## 9. Data-format observations

- Image extensions: {'.jpg': 204}
- Video extensions: {'.mp4': 36}
- Nested subfolders inside class folders: none
- Ignored system files: 0

## 10. Potential issues requiring attention

- 1 exact-duplicate group(s) (1 redundant file(s)).
- `driver_cam/both_hands_on_steering` contains 1 video(s) in an image-classification folder.
- 189 of 189 hand-state images have frame-style names (e.g. `frame_0231.jpg`), which suggests frames extracted from videos. Neighbouring frames are near-duplicates, so any future train/validation split must keep frames from the same clip together.
- Images come in 5 different dimensions (see section 4).
- Mixed image orientation: {'portrait': 171, 'landscape': 33}.
- 148 image(s) have an EXIF rotation tag; stored pixel orientation differs from how they display.
- Videos come in more than one resolution (see section 5).
- Some videos carry rotation metadata; frame width/height may not match how they play.
- Hand-state folders carry no person/session/recording identifiers. Near-identical frames of the same person or clip cannot be told apart from the folder structure.
- Labels were not verified: this audit treats the folder name as the label and does not inspect content.
