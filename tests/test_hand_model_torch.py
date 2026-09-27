"""Torch-dependent tests for the hand-state model (skipped if torch/torchvision are not installed).

Uses a randomly initialised MobileNetV3-Small and a fixed-logit fake model on tiny generated
fixture images: no trained model and no downloaded weights are needed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.driver.hand_model import (
    DEFAULT_THRESHOLD,
    HandModelError,
    TrainConfig,
    load_split,
    preprocess_base,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from app.driver.hand_model import (  # noqa: E402
    HandStateClassifier,
    build_model,
    build_transforms,
    evaluate_split,
    load_checkpoint,
    plot_confusion_matrix,
    predict_hand_state,
    resolve_device,
    save_checkpoint,
    train_hand_model,
)


def test_model_head_has_exactly_three_classes():
    model = build_model(pretrained=False).eval()
    assert model.classifier[-1].out_features == 3
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 3)


def test_eval_transform_is_deterministic_and_train_differs():
    img = preprocess_base(Image.new("RGB", (120, 200), (30, 120, 200)))
    ev = build_transforms(False)
    assert torch.equal(ev(img), ev(img))
    torch.manual_seed(0)
    tr = build_transforms(True, TrainConfig(hflip_prob=1.0))
    assert tr(img).shape == (3, 224, 224)


def test_device_resolution(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    with pytest.raises(HandModelError, match="CUDA was requested"):
        resolve_device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
    with pytest.raises(HandModelError):
        resolve_device("tpu")


class FakeModel(torch.nn.Module):
    """Returns fixed logits, so predictions are known without training."""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor([logits], dtype=torch.float32)

    def forward(self, x):
        return self.logits.repeat(x.shape[0], 1)


def fake_classifier(logits, threshold=DEFAULT_THRESHOLD):
    return HandStateClassifier(FakeModel(logits), {"image_size": 224}, "cpu", threshold)


def test_predict_hand_state_with_fake_model_all_input_types(tmp_path):
    clf = fake_classifier([0.0, 3.0, 0.0])  # softmax ~ [0.045, 0.91, 0.045]
    path = tmp_path / "x.jpg"
    Image.new("RGB", (64, 100)).save(path)
    for image in [path, str(path), Image.open(path), np.zeros((100, 64, 3), np.uint8), np.zeros((100, 64), np.uint8)]:
        out = predict_hand_state(image, classifier=clf)
        assert out["hand_state"] == "ONE_HAND" and out["confidence"] == pytest.approx(0.9094, abs=1e-3)


def test_predict_low_confidence_returns_unknown():
    out = predict_hand_state(np.zeros((50, 50, 3), np.uint8), classifier=fake_classifier([0.3, 0.2, 0.1]))
    assert out["hand_state"] == "UNKNOWN" and out["confidence"] < 0.6


def test_predict_requires_model_or_checkpoint():
    with pytest.raises(HandModelError):
        predict_hand_state(np.zeros((10, 10, 3), np.uint8))


def test_checkpoint_roundtrip_and_incompatible_file(tmp_path):
    model = build_model(pretrained=False)
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, TrainConfig(pretrained=False), 3, {"macro_f1": 0.5})
    loaded, meta = load_checkpoint(path)
    assert meta["epoch"] == 3 and meta["class_names"][2] == "no_hands" and "state_dict" not in meta
    x = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(model.eval()(x), loaded(x), atol=1e-6)
    torch.save({"format": "other"}, tmp_path / "bad.pt")
    with pytest.raises(HandModelError, match="not a compatible"):
        load_checkpoint(tmp_path / "bad.pt")
    with pytest.raises(HandModelError, match="checkpoint not found"):
        load_checkpoint(tmp_path / "none.pt")


# --- tiny end-to-end fixture -------------------------------------------------------------------

COLORS = {"both_hands_on_steering": (220, 40, 40), "one_hand_on_steering": (40, 220, 40), "no_hands": (40, 40, 220)}


@pytest.fixture
def tiny_split(tmp_path):
    root = tmp_path / "Data"
    manifests = tmp_path / "processed"
    manifests.mkdir()
    counts = {"train": 4, "val": 2, "test": 2}
    k = 0
    for split, n in counts.items():
        records = []
        for idx, (label, color) in enumerate(COLORS.items()):
            for i in range(n):
                k += 1
                rel = f"drivercamera/{label}/{split}_{i}.jpg"
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shade = tuple(min(255, c + (k % 7) * 3) for c in color)
                Image.new("RGB", (60, 100), shade).save(root / rel)
                records.append({"path": rel, "label": label, "class_index": idx, "split": split})
        (manifests / f"hand_{split}.json").write_text(json.dumps(records))
    return root, manifests


def test_training_writes_best_last_history_and_never_reads_test(tiny_split, tmp_path, monkeypatch):
    root, manifests = tiny_split
    (manifests / "hand_test.json").unlink()  # training must not need the test split
    out = tmp_path / "out"
    cfg = TrainConfig(epochs=2, batch_size=4, pretrained=False, device="cpu", patience=0)
    history = train_hand_model(root, manifests, out, cfg, log=lambda s: None)
    assert (out / "models" / "hand_state_best.pt").is_file() and (out / "models" / "hand_state_last.pt").is_file()
    saved = json.loads((out / "reports" / "hand_training_history.json").read_text())
    assert saved["selection_metric"].startswith("validation macro-F1")
    assert len(saved["epochs"]) == 2 and saved["best_epoch"] in (1, 2)
    assert set(saved["class_weights"]) == set(COLORS)
    assert history["device"] == "cpu"


def test_training_is_deterministic_for_a_seed(tiny_split, tmp_path):
    root, manifests = tiny_split
    cfg = dict(epochs=1, batch_size=4, pretrained=False, device="cpu", patience=0, seed=7)
    a = train_hand_model(root, manifests, tmp_path / "a", TrainConfig(**cfg), log=lambda s: None)
    b = train_hand_model(root, manifests, tmp_path / "b", TrainConfig(**cfg), log=lambda s: None)
    assert a["epochs"][0]["train_loss"] == b["epochs"][0]["train_loss"]
    assert a["epochs"][0]["val_loss"] == b["epochs"][0]["val_loss"]


def test_missing_images_reported_clearly(tiny_split, tmp_path):
    root, manifests = tiny_split
    (root / "drivercamera/no_hands/train_0.jpg").unlink()
    with pytest.raises(HandModelError, match="missing under Data/"):
        train_hand_model(root, manifests, tmp_path / "o", TrainConfig(epochs=1, pretrained=False, device="cpu"), log=lambda s: None)


def test_evaluate_split_and_confusion_png(tiny_split, tmp_path):
    root, manifests = tiny_split
    records = load_split(manifests / "hand_test.json")
    m = evaluate_split(fake_classifier([0.0, 5.0, 0.0]), records, root)
    assert m["num_samples"] == 6
    assert m["confusion_matrix"] == [[0, 2, 0], [0, 2, 0], [0, 2, 0]]
    assert m["threshold_analysis"]["coverage"] == 1.0
    png = tmp_path / "cm.png"
    plot_confusion_matrix(m["confusion_matrix"], png)
    assert png.stat().st_size > 1000


# --- CLI ---------------------------------------------------------------------------------------------

def _cli(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def no_dotenv(monkeypatch):
    monkeypatch.setattr("app.config.settings.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("DATASET_ROOT", raising=False)


def test_cli_train_then_evaluate(no_dotenv, tiny_split, tmp_path, capsys):
    root, manifests = tiny_split
    out = tmp_path / "outputs"
    common = ["--root", str(root), "--manifest-dir", str(manifests), "--output-dir", str(out)]
    assert _cli("train_hand_model").main(common + ["--epochs", "1", "--batch-size", "4", "--no-pretrained", "--device", "cpu"]) == 0
    assert _cli("evaluate_hand_model").main(common + ["--device", "cpu"]) == 0
    for rel in ["reports/hand_test_metrics.json", "reports/hand_test_report.md", "plots/hand_confusion_matrix.png"]:
        assert (out / rel).is_file(), rel
    metrics = json.loads((out / "reports" / "hand_test_metrics.json").read_text())
    for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "per_class", "confusion_matrix"]:
        assert key in metrics
    assert "Not a production accuracy claim" in metrics["note"]


def test_cli_errors(no_dotenv, tiny_split, tmp_path, capsys):
    root, manifests = tiny_split
    assert _cli("evaluate_hand_model").main(["--root", str(root), "--manifest-dir", str(manifests),
                                             "--checkpoint", str(tmp_path / "none.pt")]) == 2
    assert "checkpoint not found" in capsys.readouterr().err
    assert _cli("train_hand_model").main(["--root", str(root), "--manifest-dir", str(tmp_path / "nope"),
                                          "--no-pretrained", "--output-dir", str(tmp_path / "o")]) == 2
    assert "prepare_hand_dataset" in capsys.readouterr().err
