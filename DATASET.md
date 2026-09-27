# Dataset plan

No public datasets have been downloaded and no synthetic data has been generated. The local dataset lives outside the repository (currently on Google Drive) and has not yet been audited. This file records the plan and the rules.

## Provenance categories

Every record will carry a `data_source` label.

| Label | Meaning | Example | Stored under |
| --- | --- | --- | --- |
| `real_local` | Observed by us with our own cameras/phones during the hackathon | Driver clip, front-camera sequence, GPS fix | `data/raw/local/` |
| `public` | Taken from a published dataset | BDD100K frame, RDD2022 image | `data/raw/public/` |
| `inferred` | Computed by our pipeline from observations | approach_rate, drowsiness_score, risk_level | `data/processed/`, `data/events/` |
| `synthetic` | Generated to extend real observations (rare combinations, threshold tests) | Feature-level "drowsy + dog entering path" case | `data/raw/synthetic/` |

Rules:

- Real local observations are the core evidence; public data is auxiliary.
- Synthetic data extends real observations and never replaces them. It is never reported as real collection.
- Inferred features are never presented as raw observations.

## Planned public datasets (small, relevant subsets only)

| Dataset | Use | Source |
| --- | --- | --- |
| BDD100K | Primary road/temporal data: detection, tracking, driving sequences | https://bdd-data.berkeley.edu/ · https://github.com/bdd100k/bdd100k |
| RDD2022 | Road damage / potholes | https://github.com/sekilab/RoadDamageDetector |
| KITTI Tracking (optional) | Extra tracking validation only if needed | https://www.cvlibs.net/datasets/kitti/eval_tracking.php |
| nuScenes | Not required for the first prototype | — |

## Current local dataset (REAL_LOCAL)

Stored outside the repo and pointed to by `DATASET_ROOT` (or `--root`). It is never copied into the repository, and no tool here renames, moves, deletes or modifies it.

Actual layout (Google Drive export, audited 26 Sep 2026; recognised by the audit as layout `collected_2026_09`):

```text
Data/
├── drivercamera-20260926T193339Z-1-001/drivercamera/
│   ├── both_hands_on_steering/   30 images
│   ├── one_hand_on_steering/     97 images
│   ├── no_hands/                 62 images (includes one exact duplicate: "Copy of frame_0231.jpg")
│   └── drowsy_driver/            10 videos, 47.3 s total (3840x2160 HEVC, ~60 fps)
└── frontcamera-20260926T193501Z-1-001/frontcamera/
    ├── dogs_on_road/  pedestrains/  roads/  vehicles/   15 images + 26 videos, 84.4 s total
```

Totals: 240 files, 204 readable JPEG images, 36 readable MP4 videos, 0 corrupt, 0 unsupported, 1 exact-duplicate pair. 148 images carry EXIF rotation; images are mixed portrait/landscape in 5 resolutions. `pedestrains` is the folder's real spelling and is kept as-is.

The audit also recognises the originally planned layout (`driver_cam/{2hands,1hand,no_hands,drowsiness}`, `front_cam/{roads,pedestrian,dogs,vehicles}`). Camera folders (`drivercamera` / `driver_cam`, `frontcamera` / `front_cam`) are found up to 4 levels deep, so wrapper folders are handled.

| Part | Provenance | Interpretation |
| --- | --- | --- |
| `both_hands_on_steering`, `one_hand_on_steering`, `no_hands` | `real_local` | Candidate supervised image-classification data (hand state → BOTH_HANDS / ONE_HAND / NO_HANDS). Folder name = label; **labels not yet visually verified** (see the preview below). Each image is a standalone labelled observation. The source videos are not part of the dataset, and no grouping is inferred from filenames. |
| `drowsy_driver` | `real_local` | **Temporal** data (videos). Not an image class; used through face/eye features over time. |
| `dogs_on_road`, `pedestrains`, `roads`, `vehicles` | `real_local` | **Local validation/calibration** data for the pretrained road-perception pipeline. Not automatically a four-class classifier. |

Also planned per the spec: GPS and timestamps from our own drives. Only consented footage is used. Faces of bystanders and licence plates are not published.

Current use status:

- **Hand-state classifier:** trained once on the original class-only split (`outputs/models/hand_state_best.pt`). The split has since been rebuilt (class + capture format, below); the model has **not** been retrained on it, so its old test metrics do not apply to the new split.
- **Drowsiness:** TEMPORAL INFRASTRUCTURE IMPLEMENTED; ACTUAL LOCAL DROWSINESS VIDEO VALIDATION PENDING. No accuracy is claimed and the thresholds are prototype defaults.
- Software test fixtures (blank frames, hand-built landmark arrays in `tests/`) exist only to test code. They are not data and are never added to the dataset.

### Visual label check (hand state)

```bash
python scripts/preview_hand_dataset.py --root "<DATASET_ROOT>" --samples-per-class 12
```

Writes `outputs/plots/hand_dataset_preview.jpg`: a contact sheet with one section per hand-state class, showing the folder name, file counts, and each sample's filename and stored dimensions. It also writes `outputs/reports/hand_preview_manifest.json`, which records for every shown sample its source path relative to the root, class, filename, original and display dimensions, EXIF orientation, sample index and position in the class. The manifest also lists unreadable files and exact duplicates.

- Samples are evenly spaced over each class's files sorted by name, so the same command always shows the same images, and they span the whole folder.
- EXIF orientation is applied to an in-memory copy for display only.

### Hand-state split (modeling manifests)

```bash
python scripts/prepare_hand_dataset.py --root "<DATASET_ROOT>" --seed 42
```

Builds a deterministic image-level split (70 / 15 / 15, seed 42) as JSON manifests. It does not copy images: records point to the source files by path relative to the dataset root.

**Split method (default `class_and_capture_format`).** Images are stratified by class **and** by capture format: the stored pixel size plus the EXIF orientation tag, read from each image header, never from the filename. The dataset has three capture formats:

- `1080x1920` with EXIF 8, found in all three classes;
- `720x1280` with no EXIF, one_hand only;
- `848x480` with EXIF 8, no_hands only.

A visual check of every image showed each format corresponds to one physical setup here: the office rig and two different cars. Stratifying by format makes each format appear in train, validation and test in proportion. The previous class-only split had only 2 of the 41 `720x1280` images and 2 of the 18 `848x480` images in test.

Capture format is **not** a recording-group ID. It does not separate sessions, people or source videos. True group separation is impossible because the source videos are unavailable, and groups are not inferred from filenames. `--split-method class` reproduces the original class-only split exactly.

- Classes and indices: `both_hands_on_steering` = 0, `one_hand_on_steering` = 1, `no_hands` = 2.
- Each image is an independent labelled observation. The source videos are not part of the dataset, and no grouping is inferred from filenames.
- Exact duplicates (SHA-256): both files stay on disk; only one (`no_hands/frame_0231.jpg`) enters the manifests. `Copy of frame_0231.jpg` is excluded. Identical files under different labels would be excluded entirely as a label conflict.
- Checks enforced on every run:
  - each usable image is in exactly one split;
  - no file hash appears in two splits;
  - all three classes appear in every split;
  - source files are unchanged.
- No augmentation. EXIF orientation is applied only when an image is loaded for training (`app.driver.hand_dataset.load_training_image`).

| Class | Source images | Usable | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| `both_hands_on_steering` | 30 | 30 | 21 | 5 | 4 |
| `one_hand_on_steering` | 97 | 97 | 68 | 15 | 14 |
| `no_hands` | 62 | 61 | 42 | 10 | 9 |
| **Total** | **189** | **188** | **131** | **30** | **27** |

These are the counts from the real run (method `class_and_capture_format`, seed 42). The per-format counts are in `outputs/reports/hand_dataset_split.md`. That report and the manifest also record how many images moved compared with the previous split (93 of 188 on the first re-split), plus the limitations. Two limitations matter most:

- many images are near-consecutive video frames, so near-duplicates cross splits and test metrics are optimistic;
- the split does **not** address the camera-domain shift to the demo video.

Outputs: `data/processed/hand_train.json`, `hand_val.json`, `hand_test.json`, `hand_dataset_manifest.json`, and `outputs/reports/hand_dataset_split.md`.

### Auditing it

```bash
python scripts/audit_dataset.py --root "G:/My Drive/dataset"   # or set DATASET_ROOT in .env
```

The audit is read-only (nothing is renamed, moved, deleted or resized). It writes:

- `data/processed/dataset_manifest.json`: exact per-folder counts, image/video metadata, corrupt/empty/unsupported files, SHA-256 exact-duplicate groups, and one record per file. Paths are relative to the dataset root; only the root's folder name is stored.
- `outputs/reports/dataset_audit.md`: the human-readable report.

The audit itself creates no split, resizing, augmentation, frame extraction or synthetic data. The hand-state split is a separate step (above).

## Planned outputs

- `data/events/physical_risk_events.parquet`: one row per risk event. Schema in plan §13.
- `data/events/dataset_manifest.json` (event-dataset manifest; separate from the local-dataset audit manifest in `data/processed/`): collection window, cadence, row count, source breakdown, schema, evidence paths.
- `outputs/evidence/event_XXXX/`: `front.mp4`, `driver.mp4`, `metadata.json` (±5 s around each event).
