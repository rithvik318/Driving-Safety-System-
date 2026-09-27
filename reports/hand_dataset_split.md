# Hand-state dataset split

- Dataset folder: `Data` · created 2026-09-26T21:34:15+00:00 (UTC) · provenance `real_local`
- Seed: 42 · ratios train 0.70 / val 0.15 / test 0.15
- Split method: `class_and_capture_format`: stratified by (class, capture format); seeded shuffle within each stratum; largest-remainder sizes per stratum
- Capture format = stored width x height in pixels plus the EXIF orientation tag, read from the image header (not from the filename). A file property only: it does not identify source videos, sessions or people.
- Split unit: image. Each image is treated as an independent labelled observation; nothing is inferred from filenames.
- No augmentation, resizing or copying. Source files are untouched; EXIF orientation is applied only at training-load time.

| Class | Index | Total usable | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| `both_hands_on_steering` | 0 | 30 | 21 | 5 | 4 |
| `one_hand_on_steering` | 1 | 97 | 68 | 15 | 14 |
| `no_hands` | 2 | 61 | 42 | 10 | 9 |
| **Total** | | **188** | **131** | **30** | **27** |

## Source accounting

| Class | Source images | Unreadable | Excluded duplicates | Usable |
| --- | ---: | ---: | ---: | ---: |
| `both_hands_on_steering` | 30 | 0 | 0 | 30 |
| `one_hand_on_steering` | 97 | 0 | 0 | 97 |
| `no_hands` | 62 | 0 | 1 | 61 |

## Exact duplicates (SHA-256)

- `drivercamera-20260926T193339Z-1-001/drivercamera/no_hands/frame_0231.jpg`, `drivercamera-20260926T193339Z-1-001/drivercamera/no_hands/Copy of frame_0231.jpg` → kept drivercamera-20260926T193339Z-1-001/drivercamera/no_hands/frame_0231.jpg

All source files are kept on disk; excluded copies are only left out of the split manifests.

## Unreadable images

None.

## Capture formats per split

| Class | Capture format | Total | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: | ---: |
| `both_hands_on_steering` | `1080x1920|exif_orientation=8` | 30 | 21 | 5 | 4 |
| `one_hand_on_steering` | `1080x1920|exif_orientation=8` | 56 | 39 | 9 | 8 |
| `one_hand_on_steering` | `720x1280|exif_orientation=none` | 41 | 29 | 6 | 6 |
| `no_hands` | `1080x1920|exif_orientation=8` | 43 | 30 | 7 | 6 |
| `no_hands` | `848x480|exif_orientation=8` | 18 | 12 | 3 | 3 |

## Change from the previous split

- Images compared: 188 · moved to a different split: 93 · unchanged: 95
- Moves: test->train: 22, test->val: 1, train->test: 18, train->val: 26, val->test: 5, val->train: 21
- New test images that were in the previous train split: 18; new validation images previously in train: 26
- A checkpoint trained on the previous split has seen some of the new validation/test images. Its metrics on the new split are NOT a valid held-out evaluation; retrain before evaluating.

## Checks passed

- Every usable image appears in exactly one split.
- No file hash appears in more than one split.
- All three classes appear in train, validation and test.
- Source files were unchanged (size and modification time) before and after preparation.

## Limitations

- True setup/recording-group separation is impossible: the source videos and group IDs are not available, and groups are deliberately NOT inferred from filenames.
- The split unit is the image. Many images are near-consecutive video frames that look almost identical, so near-duplicates of test images are very likely in train. Test metrics are therefore optimistic.
- Capture-format stratification only balances how each capture format is represented across splits; it does not separate recordings, people or sessions.
- In this dataset some capture formats occur in only one class, so scene/format can predict the label for those images. A stratified split cannot detect a model that relies on that shortcut.
- This split does not address the camera-domain shift: the intended demo video uses a different physical setup from every training setup (see outputs/reports/driver_camera_domain_comparison.md).
- With 27-30 validation/test images, per-class metrics have wide uncertainty.
- Only byte-identical duplicates are detected; visually near-identical images are not controlled across splits.
