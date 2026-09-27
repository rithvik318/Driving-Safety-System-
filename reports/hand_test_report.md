# Hand-state classifier — test evaluation

> **Small-data caveat.** The dataset contains only **188 unique usable images**, and this test set contains only **27 images**. One image changes accuracy by about 3.7 percentage points. These numbers describe this prototype on this data. They are **not** production or real-world accuracy claims.

- Model: mobilenet_v3_small (ImageNet-pretrained: True), checkpoint epoch 25 selected on validation macro-F1 (1.0).
- Evaluated 2026-09-26T21:41:00+00:00 on `test` split (never used for training or model selection).
- Metrics use the arg-max class. The confidence threshold is analysed separately below.

## Overall

| Metric | Value |
| --- | ---: |
| Accuracy | 1.0000 |
| Macro precision | 1.0000 |
| Macro recall | 1.0000 |
| Macro F1 | 1.0000 |
| Samples | 27 |

## Per class

| Class | Hand state | Test samples | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `both_hands_on_steering` | BOTH_HANDS | 4 | 1.0000 | 1.0000 | 1.0000 |
| `one_hand_on_steering` | ONE_HAND | 14 | 1.0000 | 1.0000 | 1.0000 |
| `no_hands` | NO_HANDS | 9 | 1.0000 | 1.0000 | 1.0000 |

## Confusion matrix (rows = true, columns = predicted)

| True \ Predicted | BOTH_HANDS | ONE_HAND | NO_HANDS |
| --- | ---: | ---: | ---: |
| BOTH_HANDS | 4 | 0 | 0 |
| ONE_HAND | 0 | 14 | 0 |
| NO_HANDS | 0 | 0 | 9 |

![confusion matrix](../plots/hand_confusion_matrix.png)

## Confidence threshold

At threshold 0.60, 0 of 27 images would be reported as `UNKNOWN` (coverage 100.00%). Accuracy on the accepted images: 100.00%.

## Misclassified images

None.
