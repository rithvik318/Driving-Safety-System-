"""Hand-state model logic that needs no PyTorch: thresholding, metrics, class weights,
checkpoint selection, preprocessing geometry, manifests and the report. Torch-dependent
tests are in test_hand_model_torch.py. No trained model is needed anywhere."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from app.driver.hand_model import (
    DEFAULT_THRESHOLD,
    HAND_STATES,
    HandModelError,
    TrainConfig,
    compute_class_weights,
    compute_metrics,
    is_better,
    load_split,
    pad_to_square,
    preprocess_base,
    probabilities_to_prediction,
    render_test_report,
)

def test_hand_state_order_matches_class_indices():
    assert HAND_STATES == ("BOTH_HANDS", "ONE_HAND", "NO_HANDS")


@pytest.mark.parametrize("probs,expected", [
    ([0.8, 0.15, 0.05], "BOTH_HANDS"),
    ([0.1, 0.7, 0.2], "ONE_HAND"),
    ([0.05, 0.05, 0.9], "NO_HANDS"),
    ([0.5, 0.3, 0.2], "UNKNOWN"),  # below 0.60
    ([0.6, 0.3, 0.1], "BOTH_HANDS"),  # exactly at threshold is accepted
])
def test_threshold_mapping(probs, expected):
    out = probabilities_to_prediction(probs, DEFAULT_THRESHOLD)
    assert out["hand_state"] == expected
    assert set(out) >= {"hand_state", "confidence", "class_probabilities"}
    assert set(out["class_probabilities"]) == set(HAND_STATES)
    assert out["confidence"] == pytest.approx(max(probs), abs=1e-4)


def test_unknown_still_reports_argmax_and_threshold_configurable():
    out = probabilities_to_prediction([0.45, 0.35, 0.2], 0.6)
    assert out["hand_state"] == "UNKNOWN" and out["predicted_class"] == "BOTH_HANDS"
    assert probabilities_to_prediction([0.45, 0.35, 0.2], 0.4)["hand_state"] == "BOTH_HANDS"


def test_default_threshold_is_060():
    assert DEFAULT_THRESHOLD == 0.60


@pytest.mark.parametrize("bad", [[0.5, 0.5], [np.nan, 0.5, 0.5], [-0.1, 0.6, 0.5], [0, 0, 0]])
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(ValueError):
        probabilities_to_prediction(bad)


def test_probabilities_are_renormalised():
    out = probabilities_to_prediction([2.0, 1.0, 1.0])
    assert sum(out["class_probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_class_weights_inverse_frequency_on_real_train_counts():
    labels = [0] * 21 + [1] * 68 + [2] * 43
    w = compute_class_weights(labels)
    assert w == pytest.approx([132 / (3 * 21), 132 / (3 * 68), 132 / (3 * 43)], abs=1e-5)
    assert w[0] > w[2] > w[1]  # rarest class weighted most


def test_class_weights_missing_class_raises():
    with pytest.raises(HandModelError, match="no examples of: no_hands"):
        compute_class_weights([0, 0, 1])


def test_metrics_hand_computed_example():
    y_true = [0, 0, 1, 1, 1, 2, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 2, 2, 0]
    m = compute_metrics(y_true, y_pred)
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 2, 1], [1, 0, 2]]
    assert m["accuracy"] == pytest.approx(5 / 8, abs=1e-4)
    both = m["per_class"]["both_hands_on_steering"]
    assert (both["precision"], both["recall"], both["support"]) == (0.5, 0.5, 2)
    one = m["per_class"]["one_hand_on_steering"]
    assert one["precision"] == pytest.approx(2 / 3, abs=1e-4) and one["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["macro_f1"] == pytest.approx(np.mean([0.5, 2 / 3, 2 / 3]), abs=1e-3)


def test_metrics_zero_division_is_reported():
    m = compute_metrics([0, 1, 2], [1, 1, 1])
    assert m["per_class"]["both_hands_on_steering"]["precision"] == 0.0
    assert any("never predicted" in z for z in m["zero_division"])


def test_best_checkpoint_selected_by_macro_f1_not_accuracy():
    best = {"val_macro_f1": 0.60, "val_loss": 0.9, "val_accuracy": 0.90}
    higher_f1_lower_acc = {"val_macro_f1": 0.70, "val_loss": 1.1, "val_accuracy": 0.70}
    assert is_better(higher_f1_lower_acc, best)
    assert not is_better({"val_macro_f1": 0.55, "val_loss": 0.1, "val_accuracy": 0.99}, best)
    assert is_better({"val_macro_f1": 0.60, "val_loss": 0.8, "val_accuracy": 0.1}, best)  # tie -> lower loss
    assert is_better(best, None)


def test_pad_to_square_keeps_whole_frame():
    img = Image.new("RGB", (90, 160), (255, 0, 0))
    sq = pad_to_square(img)
    assert sq.size == (160, 160)
    arr = np.asarray(sq)
    assert (arr[80, 35:125] == [255, 0, 0]).all()  # original pixels intact, centred
    assert tuple(arr[80, 0]) != (255, 0, 0)  # padding at the sides
    assert preprocess_base(img).size == (224, 224)


def test_load_split_validates(tmp_path):
    good = tmp_path / "hand_train.json"
    good.write_text(json.dumps([{"path": "a.jpg", "label": "no_hands", "class_index": 2, "split": "train"}]))
    assert load_split(good)[0]["label"] == "no_hands"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"path": "a.jpg", "label": "no_hands", "class_index": 0, "split": "train"}]))
    with pytest.raises(HandModelError, match="unexpected label/class_index"):
        load_split(bad)
    with pytest.raises(HandModelError, match="prepare_hand_dataset"):
        load_split(tmp_path / "missing.json")


def test_train_config_validation():
    with pytest.raises(HandModelError):
        TrainConfig(epochs=0).validate()
    with pytest.raises(HandModelError):
        TrainConfig(hflip_prob=1.5).validate()
    TrainConfig().validate()


def test_report_states_dataset_size_and_all_metrics():
    m = compute_metrics([0, 1, 1, 2], [0, 1, 2, 2])
    m.update(threshold_analysis={"threshold": 0.6, "reported_unknown": 1, "coverage": 0.75, "accuracy_on_accepted": 1.0},
             predictions=[{"path": "x.jpg", "true": "one_hand_on_steering", "predicted": "no_hands", "confidence": 0.7, "correct": False}])
    text = render_test_report(m, {"arch": "mobilenet_v3_small", "epoch": 7, "val_metrics": {"macro_f1": 0.8},
                                  "train_config": {"pretrained": True}}, {"total_usable": 188})
    assert "only **188 unique usable images**" in text and "only **4 images**" in text
    assert "not** production" in text
    for label in ["Accuracy", "Macro precision", "Macro recall", "Macro F1", "Test samples", "Confusion matrix"]:
        assert label in text
    assert "x.jpg" in text
