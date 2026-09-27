"""Hand-state classifier: MobileNetV3-Small (ImageNet-pretrained) fine-tuned to 3 classes.

Classes (index order fixed by app.driver.hand_dataset):
    0 both_hands_on_steering -> BOTH_HANDS
    1 one_hand_on_steering   -> ONE_HAND
    2 no_hands               -> NO_HANDS

Preprocessing (identical for train / val / test / inference):
    EXIF orientation applied in memory -> RGB -> pad to square with a neutral grey
    (whole frame kept, nothing cropped) -> resize to 224x224 -> ImageNet normalisation.

Augmentation (TRAIN ONLY, mild): horizontal flip (label-preserving: it mirrors which
hand is on the wheel, not how many), rotation up to ±7°, shift up to 5 %, scale 0.95-1.05,
brightness and contrast ±20 %. Validation, test and inference use no augmentation.

Class imbalance: class-weighted cross-entropy with weights N / (K * n_c) computed from the
TRAIN split only. No oversampling.

The parts that do not need PyTorch (metrics, thresholding, weights, padding, reports) work
without it; torch/torchvision are imported only by the functions that need them.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from app.driver.hand_dataset import CLASS_NAMES, CLASS_TO_HAND_STATE, load_training_image

HAND_STATES: tuple[str, ...] = tuple(CLASS_TO_HAND_STATE[c] for c in CLASS_NAMES)  # BOTH_HANDS, ONE_HAND, NO_HANDS
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAD_FILL = (124, 116, 104)  # ImageNet mean colour; padding normalises to ~0
DEFAULT_THRESHOLD = 0.60
CHECKPOINT_FORMAT = "hand_state_mobilenet_v3_small/v1"
ARCH = "mobilenet_v3_small"


class HandModelError(Exception):
    """Configuration, data or checkpoint problem. The message says what to do."""


# =========================================================================== config


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 16
    lr: float = 1e-3  # classifier head learning rate
    backbone_lr_factor: float = 0.1  # pretrained backbone learns at lr * factor
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    hflip_prob: float = 0.5
    rotation_degrees: float = 7.0
    translate: float = 0.05
    scale_min: float = 0.95
    scale_max: float = 1.05
    brightness: float = 0.2
    contrast: float = 0.2
    patience: int = 10  # early stop after this many epochs without val macro-F1 improvement (0 = off)
    pretrained: bool = True
    num_workers: int = 0  # 0 is the most portable (Windows) and deterministic choice
    image_size: int = IMAGE_SIZE

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise HandModelError("epochs and batch_size must be >= 1")
        if not (self.lr > 0 and self.weight_decay >= 0 and 0 < self.backbone_lr_factor <= 1):
            raise HandModelError("lr must be > 0, weight_decay >= 0, backbone_lr_factor in (0, 1]")
        if not 0 <= self.hflip_prob <= 1:
            raise HandModelError("hflip_prob must be between 0 and 1")


# =========================================================================== pure helpers (no torch)


def pad_to_square(img: Image.Image, fill=PAD_FILL) -> Image.Image:
    """Pad the shorter side so the whole frame is kept (no cropping of hands or wheel)."""
    w, h = img.size
    side = max(w, h)
    if w == h:
        return img
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def preprocess_base(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Deterministic geometry shared by training and inference: RGB -> pad square -> size x size."""
    return pad_to_square(img.convert("RGB")).resize((size, size), Image.Resampling.BILINEAR)


def compute_class_weights(labels: list[int], num_classes: int = NUM_CLASSES) -> list[float]:
    """Inverse-frequency weights N / (K * n_c). A class absent from `labels` is an error."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes)
    if (counts == 0).any():
        missing = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i) for i in np.flatnonzero(counts == 0)]
        raise HandModelError(f"training split has no examples of: {', '.join(missing)}")
    return [round(float(len(labels) / (num_classes * c)), 6) for c in counts]


def compute_metrics(y_true: list[int], y_pred: list[int], num_classes: int = NUM_CLASSES) -> dict:
    """Accuracy, per-class and macro precision/recall/F1, support, confusion matrix (rows = true).

    A metric whose denominator is 0 is reported as 0.0 and listed in `zero_division`.
    """
    y_true_a, y_pred_a = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if y_true_a.shape != y_pred_a.shape:
        raise ValueError("y_true and y_pred must have the same length")
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true_a, y_pred_a):
        cm[t, p] += 1
    per_class, zero_div = {}, []
    for i in range(num_classes):
        tp = int(cm[i, i])
        predicted, actual = int(cm[:, i].sum()), int(cm[i, :].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)
        if not predicted:
            zero_div.append(f"{name}: precision (never predicted)")
        if not actual:
            zero_div.append(f"{name}: recall (no samples)")
        per_class[name] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": actual}
    n = len(y_true_a)
    return {
        "num_samples": n,
        "accuracy": round(float((y_true_a == y_pred_a).mean()) if n else 0.0, 4),
        "macro_precision": round(float(np.mean([m["precision"] for m in per_class.values()])), 4),
        "macro_recall": round(float(np.mean([m["recall"] for m in per_class.values()])), 4),
        "macro_f1": round(float(np.mean([m["f1"] for m in per_class.values()])), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "zero_division": zero_div,
    }


def is_better(candidate: dict, best: dict | None) -> bool:
    """Checkpoint selection: higher validation macro-F1; ties broken by lower validation loss."""
    if best is None:
        return True
    if candidate["val_macro_f1"] != best["val_macro_f1"]:
        return candidate["val_macro_f1"] > best["val_macro_f1"]
    return candidate["val_loss"] < best["val_loss"]


def probabilities_to_prediction(probs, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Map class probabilities (order = CLASS_NAMES) to the public prediction dict.

    If the highest probability is below `threshold`, hand_state is "UNKNOWN".
    """
    p = np.asarray(probs, dtype=float).reshape(-1)
    if p.shape[0] != NUM_CLASSES or not np.all(np.isfinite(p)) or (p < 0).any():
        raise ValueError(f"expected {NUM_CLASSES} finite non-negative probabilities, got {probs!r}")
    total = p.sum()
    if total <= 0:
        raise ValueError("probabilities sum to 0")
    p = p / total
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    best = int(p.argmax())
    confidence = float(p[best])
    return {
        "hand_state": HAND_STATES[best] if confidence >= threshold else "UNKNOWN",
        "confidence": round(confidence, 4),
        "class_probabilities": {HAND_STATES[i]: round(float(p[i]), 4) for i in range(NUM_CLASSES)},
        "predicted_class": HAND_STATES[best],  # argmax, even when reported as UNKNOWN
        "threshold": threshold,
    }


def load_split(path: Path) -> list[dict]:
    """Load one hand_{split}.json manifest and check it against the fixed class mapping."""
    if not path.is_file():
        raise HandModelError(f"split manifest not found: {path}. Run scripts/prepare_hand_dataset.py first.")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise HandModelError(f"split manifest is empty or malformed: {path}")
    for r in records:
        if r.get("label") not in CLASS_NAMES or CLASS_NAMES.index(r["label"]) != r.get("class_index"):
            raise HandModelError(f"record with unexpected label/class_index in {path.name}: {r}")
    return records


# =========================================================================== torch parts


def resolve_device(name: str = "auto") -> str:
    import torch

    name = (name or "auto").lower()
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise HandModelError("CUDA was requested but torch.cuda.is_available() is False. Use --device auto or cpu.")
        return name
    if name == "cpu":
        return "cpu"
    raise HandModelError(f"unknown device {name!r}; use auto, cpu or cuda")


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_model(pretrained: bool = True, num_classes: int = NUM_CLASSES):
    """MobileNetV3-Small with its last Linear layer replaced by a num_classes head."""
    import torch.nn as nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        model = mobilenet_v3_small(weights=weights)
    except Exception as exc:  # usually no internet on first download
        raise HandModelError(
            f"could not load ImageNet weights for MobileNetV3-Small ({type(exc).__name__}: {exc}). "
            "They download once (~10 MB) from download.pytorch.org into the torch cache; check the connection."
        ) from exc
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_transforms(train: bool, cfg: TrainConfig | None = None):
    """Tensor transforms applied AFTER preprocess_base (which already did pad + resize)."""
    from torchvision import transforms as T

    cfg = cfg or TrainConfig()
    tail = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return T.Compose(tail)
    return T.Compose([
        T.RandomHorizontalFlip(p=cfg.hflip_prob),
        T.RandomAffine(degrees=cfg.rotation_degrees, translate=(cfg.translate, cfg.translate),
                       scale=(cfg.scale_min, cfg.scale_max), fill=PAD_FILL),
        T.ColorJitter(brightness=cfg.brightness, contrast=cfg.contrast),
        *tail,
    ])


class HandImageDataset:
    """Map-style dataset over manifest records. Images are decoded once and cached after
    preprocess_base, so augmentation (if any) runs on the cached 224x224 image each epoch."""

    def __init__(self, records: list[dict], root: Path, transform, image_size: int = IMAGE_SIZE):
        missing = [r["path"] for r in records if not (root / r["path"]).is_file()]
        if missing:
            raise HandModelError(
                f"{len(missing)} image(s) listed in the manifest are missing under {root.name}/, e.g. {missing[0]}. "
                "Check --root / DATASET_ROOT."
            )
        self.records = records
        self.transform = transform
        self.images = [preprocess_base(load_training_image(root / r["path"]), image_size) for r in records]
        self.labels = [int(r["class_index"]) for r in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        return self.transform(self.images[i]), self.labels[i]


def _loader(dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int):
    import torch
    from torch.utils.data import DataLoader

    g = torch.Generator()
    g.manual_seed(seed)

    def worker_init(worker_id: int) -> None:
        s = seed + worker_id
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g,
                      num_workers=num_workers, worker_init_fn=worker_init)


def _run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, list[int], list[int]]:
    import torch

    training = optimizer is not None
    model.train(training)
    total_loss, y_true, y_pred = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * len(y)
            y_true += y.tolist()
            y_pred += logits.argmax(1).tolist()
    return total_loss / max(1, len(y_true)), y_true, y_pred


def save_checkpoint(path: Path, model, cfg: TrainConfig, epoch: int, val_metrics: dict, extra: dict | None = None) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "arch": ARCH,
        "num_classes": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "hand_states": list(HAND_STATES),
        "image_size": cfg.image_size,
        "pad_fill": list(PAD_FILL),
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "epoch": epoch,
        "val_metrics": val_metrics,
        "train_config": asdict(cfg),
        "torch_version": str(torch.__version__),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
        "state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path: Path, device: str = "cpu"):
    """Return (model in eval mode on device, checkpoint metadata without the weights)."""
    import torch

    if not Path(path).is_file():
        raise HandModelError(f"checkpoint not found: {path}. Train first with scripts/train_hand_model.py.")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if ckpt.get("format") != CHECKPOINT_FORMAT or ckpt.get("class_names") != list(CLASS_NAMES):
        raise HandModelError(f"{path} is not a compatible hand-state checkpoint (format/classes differ)")
    model = build_model(pretrained=False, num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, {k: v for k, v in ckpt.items() if k != "state_dict"}


def train_hand_model(root: Path, manifest_dir: Path, output_dir: Path, cfg: TrainConfig,
                     log: Callable[[str], None] = print) -> dict:
    """Fine-tune on train, select on validation macro-F1. Never reads the test split."""
    import torch
    import torch.nn as nn

    cfg.validate()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    train_records = load_split(manifest_dir / "hand_train.json")
    val_records = load_split(manifest_dir / "hand_val.json")

    log(f"Loading {len(train_records)} train and {len(val_records)} validation images (EXIF applied, padded to square)...")
    train_ds = HandImageDataset(train_records, root, build_transforms(True, cfg), cfg.image_size)
    val_ds = HandImageDataset(val_records, root, build_transforms(False, cfg), cfg.image_size)
    train_loader = _loader(train_ds, cfg.batch_size, True, cfg.seed, cfg.num_workers)
    val_loader = _loader(val_ds, cfg.batch_size, False, cfg.seed, cfg.num_workers)

    class_weights = compute_class_weights(train_ds.labels)
    model = build_model(pretrained=cfg.pretrained).to(device)
    head = list(model.classifier.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": cfg.lr * cfg.backbone_lr_factor}, {"params": head, "lr": cfg.lr}],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    models_dir = output_dir / "models"
    best_path, last_path = models_dir / "hand_state_best.pt", models_dir / "hand_state_last.pt"
    history: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device,
        "torch_version": str(torch.__version__),
        "arch": ARCH,
        "pretrained_weights": "IMAGENET1K_V1" if cfg.pretrained else None,
        "config": asdict(cfg),
        "class_names": list(CLASS_NAMES),
        "class_weights": dict(zip(CLASS_NAMES, class_weights)),
        "train_counts": dict(zip(CLASS_NAMES, np.bincount(train_ds.labels, minlength=NUM_CLASSES).tolist())),
        "val_counts": dict(zip(CLASS_NAMES, np.bincount(val_ds.labels, minlength=NUM_CLASSES).tolist())),
        "selection_metric": "validation macro-F1 (ties: lower validation loss)",
        "epochs": [],
    }
    log(f"Device: {device} · class weights: {history['class_weights']} · pretrained: {cfg.pretrained}")

    best, since_best = None, 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        lrs = [g["lr"] for g in optimizer.param_groups]
        train_loss, ty, tp = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, vy, vp = _run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        vm = compute_metrics(vy, vp)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_accuracy": round(float(np.mean(np.array(ty) == np.array(tp))), 4),
            "val_loss": round(val_loss, 5),
            "val_accuracy": vm["accuracy"],
            "val_macro_f1": vm["macro_f1"],
            "val_per_class_f1": {c: vm["per_class"][c]["f1"] for c in CLASS_NAMES},
            "lr_backbone": lrs[0],
            "lr_head": lrs[1],
            "seconds": round(time.time() - t0, 2),
        }
        improved = is_better(row, best)
        row["is_best"] = improved
        history["epochs"].append(row)
        if improved:
            best, since_best = row, 0
            save_checkpoint(best_path, model, cfg, epoch, vm)
        else:
            since_best += 1
        log(f"epoch {epoch:3d}/{cfg.epochs}  train_loss {row['train_loss']:.4f}  train_acc {row['train_accuracy']:.3f}  "
            f"val_loss {row['val_loss']:.4f}  val_acc {row['val_accuracy']:.3f}  val_macroF1 {row['val_macro_f1']:.3f}"
            f"{'  *best' if improved else ''}  ({row['seconds']}s)")
        if cfg.patience and since_best >= cfg.patience:
            log(f"Early stop: no validation macro-F1 improvement for {cfg.patience} epochs.")
            history["early_stopped_at"] = epoch
            break

    save_checkpoint(last_path, model, cfg, history["epochs"][-1]["epoch"], vm)
    history["best_epoch"] = best["epoch"]
    history["best_val_macro_f1"] = best["val_macro_f1"]
    history["best_checkpoint"] = best_path.name
    history["last_checkpoint"] = last_path.name
    report_path = output_dir / "reports" / "hand_training_history.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


# =========================================================================== inference


class HandStateClassifier:
    """Loads a trained checkpoint and predicts hand state for one image."""

    def __init__(self, model, meta: dict, device: str = "cpu", threshold: float = DEFAULT_THRESHOLD):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.model, self.meta, self.device = model, meta, device
        self.threshold = threshold
        self.image_size = int(meta.get("image_size", IMAGE_SIZE))
        self.transform = build_transforms(train=False)

    @classmethod
    def from_checkpoint(cls, path: Path, device: str = "auto", threshold: float = DEFAULT_THRESHOLD):
        dev = resolve_device(device)
        model, meta = load_checkpoint(path, dev)
        return cls(model, meta, dev, threshold)

    def _to_pil(self, image, color_order: str) -> Image.Image:
        if isinstance(image, (str, Path)):
            return load_training_image(Path(image))
        if isinstance(image, Image.Image):
            from PIL import ImageOps

            return ImageOps.exif_transpose(image).convert("RGB")
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.ndim != 3 or arr.shape[2] not in (3, 4):
                raise ValueError(f"unsupported image array shape {image.shape}")
            arr = arr[:, :, :3]
            if color_order.upper() == "BGR":
                arr = arr[:, :, ::-1]
            return Image.fromarray(np.ascontiguousarray(arr.astype(np.uint8)), "RGB")
        raise TypeError(f"unsupported image type {type(image).__name__}")

    def predict_proba(self, image, color_order: str = "BGR") -> np.ndarray:
        import torch

        x = self.transform(preprocess_base(self._to_pil(image, color_order), self.image_size)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.softmax(self.model(x), dim=1)[0].cpu().numpy()

    def predict(self, image, color_order: str = "BGR", threshold: float | None = None) -> dict:
        return probabilities_to_prediction(self.predict_proba(image, color_order),
                                           self.threshold if threshold is None else threshold)


def predict_hand_state(image, classifier: HandStateClassifier | None = None, checkpoint: Path | None = None,
                       threshold: float = DEFAULT_THRESHOLD, color_order: str = "BGR") -> dict:
    """Predict hand state for one image (path, PIL image, or numpy array; arrays default to OpenCV BGR).

    Returns {"hand_state": BOTH_HANDS|ONE_HAND|NO_HANDS|UNKNOWN, "confidence", "class_probabilities", ...}.
    Pass a loaded `classifier` for repeated calls; `checkpoint` loads one (slower).
    """
    if classifier is None:
        if checkpoint is None:
            raise HandModelError("pass a HandStateClassifier or a checkpoint path")
        classifier = HandStateClassifier.from_checkpoint(checkpoint, threshold=threshold)
    return classifier.predict(image, color_order=color_order, threshold=threshold)


# =========================================================================== evaluation


def evaluate_split(classifier: HandStateClassifier, records: list[dict], root: Path, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Metrics on one split (argmax), plus how the confidence threshold would behave."""
    missing = [r["path"] for r in records if not (root / r["path"]).is_file()]
    if missing:
        raise HandModelError(f"{len(missing)} image(s) missing under {root.name}/, e.g. {missing[0]}. Check --root.")
    predictions, y_true, y_pred = [], [], []
    for r in records:
        probs = classifier.predict_proba(root / r["path"])
        pred = probabilities_to_prediction(probs, threshold)
        idx = HAND_STATES.index(pred["predicted_class"])
        y_true.append(int(r["class_index"]))
        y_pred.append(idx)
        predictions.append({"path": r["path"], "true": r["label"], "predicted": CLASS_NAMES[idx],
                            "confidence": pred["confidence"], "reported_state": pred["hand_state"],
                            "correct": idx == int(r["class_index"])})
    metrics = compute_metrics(y_true, y_pred)
    accepted = [p for p in predictions if p["reported_state"] != "UNKNOWN"]
    metrics["threshold_analysis"] = {
        "threshold": threshold,
        "reported_unknown": len(predictions) - len(accepted),
        "coverage": round(len(accepted) / len(predictions), 4) if predictions else 0.0,
        "accuracy_on_accepted": round(sum(p["correct"] for p in accepted) / len(accepted), 4) if accepted else None,
    }
    metrics["predictions"] = predictions
    return metrics


def plot_confusion_matrix(cm: list[list[int]], path: Path, title: str = "Hand-state confusion matrix (test)") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=150)
    ax.imshow(arr, cmap="Blues")
    labels = [s.replace("_", "\n") for s in HAND_STATES]
    ax.set_xticks(range(NUM_CLASSES), labels)
    ax.set_yticks(range(NUM_CLASSES), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11)
    threshold = arr.max() / 2 if arr.size and arr.max() else 0.5
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if arr[i, j] > threshold else "black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def render_test_report(metrics: dict, meta: dict, dataset_counts: dict, split: str = "test") -> str:
    n_total = dataset_counts.get("total_usable")
    n_split = metrics["num_samples"]
    pc = metrics["per_class"]
    ta = metrics.get("threshold_analysis", {})
    lines = [
        f"# Hand-state classifier — {split} evaluation",
        "",
        f"> **Small-data caveat.** The dataset contains only **{n_total} unique usable images**, and this {split} set "
        f"contains only **{n_split} images**. One image changes accuracy by about {100 / max(n_split, 1):.1f} percentage points. "
        "These numbers describe this prototype on this data. They are **not** production or real-world accuracy claims.",
        "",
        f"- Model: {meta.get('arch', ARCH)} (ImageNet-pretrained: {meta.get('train_config', {}).get('pretrained')}), "
        f"checkpoint epoch {meta.get('epoch')} selected on validation macro-F1 "
        f"({meta.get('val_metrics', {}).get('macro_f1')}).",
        f"- Evaluated {metrics.get('evaluated_at', '')} on `{split}` split (never used for training or model selection).",
        "- Metrics use the arg-max class. The confidence threshold is analysed separately below.",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Macro precision | {metrics['macro_precision']:.4f} |",
        f"| Macro recall | {metrics['macro_recall']:.4f} |",
        f"| Macro F1 | {metrics['macro_f1']:.4f} |",
        f"| Samples | {n_split} |",
        "",
        "## Per class",
        "",
        "| Class | Hand state | Test samples | Precision | Recall | F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, hs in zip(CLASS_NAMES, HAND_STATES):
        m = pc[name]
        lines.append(f"| `{name}` | {hs} | {m['support']} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")
    lines += ["", "## Confusion matrix (rows = true, columns = predicted)", "",
              "| True \\ Predicted | " + " | ".join(HAND_STATES) + " |",
              "| --- | " + " | ".join(["---:"] * NUM_CLASSES) + " |"]
    for hs, row in zip(HAND_STATES, metrics["confusion_matrix"]):
        lines.append(f"| {hs} | " + " | ".join(str(v) for v in row) + " |")
    lines += ["", "![confusion matrix](../plots/hand_confusion_matrix.png)", ""]
    if ta:
        acc = ta["accuracy_on_accepted"]
        acc_text = "n/a" if acc is None else f"{acc:.2%}"
        lines += [
            "## Confidence threshold",
            "",
            f"At threshold {ta['threshold']:.2f}, {ta['reported_unknown']} of {n_split} images would be reported as `UNKNOWN` "
            f"(coverage {ta['coverage']:.2%}). Accuracy on the accepted images: {acc_text}.",
            "",
        ]
    if metrics.get("zero_division"):
        lines += ["Metrics set to 0 because the denominator was 0: " + "; ".join(metrics["zero_division"]), ""]
    wrong = [p for p in metrics.get("predictions", []) if not p["correct"]]
    lines += ["## Misclassified images", ""]
    lines += [f"- `{p['path']}`: true {p['true']}, predicted {p['predicted']} ({p['confidence']:.2f})" for p in wrong] or ["None."]
    lines.append("")
    return "\n".join(lines)
