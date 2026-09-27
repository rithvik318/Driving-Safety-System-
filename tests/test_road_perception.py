"""Front-camera road perception: models, YOLO provider (faked and real), video sampling, script.

Most tests use a fake YOLO model (no ultralytics / torch needed). Tests marked `real_yolo` run
the real pretrained model on CPU when ultralytics and the weights (models/road/yolo26n.pt or
ROAD_MODEL_PATH) are available; they never download anything.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.config.road_config import DEFAULT_ROAD_CLASSES, RoadConfig, load_road_config, parse_classes
from app.road import Detection, RoadFrameStatus, RoadModelError, YoloRoadDetector, build_detections
from app.road import detector as detector_module
from app.road.runner import read_image_bgr, run_image, run_video
from app.road.visualize import draw_detections
from app.sensors.video import FrameSampler, VideoInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe"]
NAMES = dict(enumerate(COCO))
ID = {v: k for k, v in NAMES.items()}


def det(**kw) -> Detection:
    base = dict(timestamp=1.0, class_name="car", class_id=2, confidence=0.8, bbox_x1=10, bbox_y1=20,
                bbox_x2=50, bbox_y2=100, source="yolo26n")
    base.update(kw)
    return Detection(**base)


# --------------------------------------------------------------------------- fakes


class FakeBoxes:
    def __init__(self, rows):
        rows = np.array(rows, dtype=np.float32).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = rows[:, :4], rows[:, 4], rows[:, 5]

    def __len__(self):
        return len(self.conf)


class FakeResult:
    def __init__(self, rows):
        self.boxes = FakeBoxes(rows) if rows is not None else None


class FakeYolo:
    """Mimics ultralytics.YOLO.predict: returns scripted boxes [x1, y1, x2, y2, conf, cls] and counts calls."""

    def __init__(self, rows=(), raises=False):
        self.names = NAMES
        self.rows = list(rows)
        self.raises = raises
        self.calls = 0
        self.kwargs = []

    def predict(self, frame, **kw):
        self.calls += 1
        self.kwargs.append(kw)
        if self.raises:
            raise RuntimeError("boom")
        return [FakeResult(self.rows if self.rows else np.zeros((0, 6)))]


def fake_detector(rows=(), **kw) -> YoloRoadDetector:
    kw.setdefault("warmup", False)
    return YoloRoadDetector("models/road/yolo26n.pt", device="cpu", model=FakeYolo(rows), **kw)


FRAME = np.zeros((480, 640, 3), np.uint8)


# --------------------------------------------------------------------------- A. dataclass validation


def test_valid_detection_and_required_fields():
    d = det()
    record = d.to_dict()
    for key in ["timestamp", "class_name", "class_id", "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                "bbox_center_x", "bbox_center_y", "bbox_width", "bbox_height", "source"]:
        assert key in record, key
    assert record["source"] == "yolo26n" and record["class_name"] == "car"


@pytest.mark.parametrize("bad", [
    dict(bbox_x2=5),  # x2 < x1
    dict(bbox_y2=10),  # y2 < y1
    dict(confidence=1.5),
    dict(confidence=-0.1),
    dict(confidence=float("nan")),
    dict(class_id=-1),
    dict(class_id=True),
    dict(class_id=2.0),
    dict(class_name=""),
    dict(timestamp=float("inf")),
    dict(bbox_x1=float("nan")),
    dict(source=""),
])
def test_invalid_detection_rejected(bad):
    with pytest.raises(ValueError):
        det(**bad)


def test_image_detection_may_have_no_timestamp():
    assert det(timestamp=None).to_dict()["timestamp"] is None


# --------------------------------------------------------------------------- B. geometry


def test_bbox_geometry():
    d = det(bbox_x1=10, bbox_y1=20, bbox_x2=50, bbox_y2=100)
    assert (d.bbox_width, d.bbox_height) == (40, 80)
    assert (d.bbox_center_x, d.bbox_center_y) == (30, 60)
    assert d.bbox_area == 3200
    r = d.to_dict()
    assert (r["bbox_width"], r["bbox_height"], r["bbox_center_x"], r["bbox_center_y"]) == (40, 80, 30, 60)


def test_zero_size_box_is_valid_geometry():
    d = det(bbox_x2=10, bbox_y2=20)
    assert d.bbox_width == 0 and d.bbox_height == 0 and d.bbox_center_x == 10


# --------------------------------------------------------------------------- C. threshold filtering


def test_build_detections_threshold_classes_clipping_and_order():
    rows = np.array([
        [10, 10, 50, 50, 0.90, ID["car"]],
        [10, 10, 50, 50, 0.34, ID["car"]],  # below threshold
        [0, 0, 20, 20, 0.35, ID["person"]],  # exactly at threshold: kept
        [5, 5, 9, 9, 0.99, ID["bench"]],  # class not allowed
        [-30, -5, 700, 500, 0.60, ID["bus"]],  # clipped to the 640x480 image
        [30, 30, 30, 60, 0.95, ID["dog"]],  # zero width: dropped
    ], dtype=np.float32)
    allowed = {ID[c] for c in ("car", "person", "bus", "dog")}
    out = build_detections(rows[:, :4], rows[:, 4], rows[:, 5], NAMES, 0.35, allowed, (640, 480), "yolo26n", 2.5, 7)
    assert [d.class_name for d in out] == ["car", "bus", "person"]  # sorted by confidence
    bus = out[1]
    assert (bus.bbox_x1, bus.bbox_y1, bus.bbox_x2, bus.bbox_y2) == (0, 0, 640, 480)
    assert all(d.timestamp == 2.5 and d.frame_index == 7 for d in out)
    assert out[0].category == "vehicle" and out[2].category == "person"


def test_detector_applies_threshold_even_if_model_returns_low_scores():
    d = fake_detector([[10, 10, 60, 60, 0.9, ID["car"]], [10, 10, 60, 60, 0.2, ID["dog"]]], confidence_threshold=0.5)
    r = d.detect(FRAME, 1.0, 3)
    assert [x.class_name for x in r.detections] == ["car"]
    assert d.model.kwargs[-1]["conf"] == 0.5  # also passed to the model


def test_configurable_threshold_and_image_size_passed_to_model():
    d = fake_detector(confidence_threshold=0.6, image_size=320)
    d.detect(FRAME)
    kw = d.model.kwargs[-1]
    assert kw["conf"] == 0.6 and kw["imgsz"] == 320 and kw["device"] == "cpu" and kw["verbose"] is False
    assert sorted(kw["classes"]) == sorted(ID[c] for c in DEFAULT_ROAD_CLASSES)


# --------------------------------------------------------------------------- D. provider initialisation


def test_provider_init_resolves_classes_and_name():
    d = fake_detector()
    assert d.name == "yolo26n" and d.device == "cpu"
    assert set(d.class_names) == set(DEFAULT_ROAD_CLASSES)
    assert "bench" not in d.class_names


def test_provider_all_classes_and_custom_classes():
    assert fake_detector(classes=("all",)).class_ids is None
    assert fake_detector(classes=("car", "Dog")).class_names == ["car", "dog"]


def test_provider_rejects_unknown_class_and_bad_settings():
    with pytest.raises(RoadModelError, match="not in the model: unicorn"):
        fake_detector(classes=("car", "unicorn"))
    with pytest.raises(RoadModelError):
        fake_detector(confidence_threshold=1.5)
    with pytest.raises(RoadModelError):
        fake_detector(image_size=100)
    with pytest.raises(RoadModelError):
        YoloRoadDetector("x.pt", device="tpu", model=FakeYolo(), warmup=False)


def test_from_config():
    cfg = RoadConfig(confidence_threshold=0.5, image_size=416, classes=("car", "person"), device="cpu")
    d = YoloRoadDetector.from_config(cfg, model=FakeYolo(), warmup=False)
    assert d.confidence_threshold == 0.5 and d.image_size == 416 and d.class_names == ["person", "car"]


def test_missing_weights_without_download_is_a_clear_error(tmp_path, monkeypatch):
    pytest.importorskip("ultralytics")
    with pytest.raises(RoadModelError, match="not found"):
        YoloRoadDetector(tmp_path / "custom_model.pt", device="cpu", allow_download=False)


# --------------------------------------------------------------------------- E. model loads once


def test_model_loaded_once_and_reused(monkeypatch):
    loads = []
    model = FakeYolo([[10, 10, 60, 60, 0.9, ID["car"]]])

    def fake_load(path, allow_download=True):
        loads.append(path)
        return model

    monkeypatch.setattr(detector_module, "load_yolo_model", fake_load)
    d = YoloRoadDetector("models/road/yolo26n.pt", device="cpu")  # default warmup=True
    for i in range(5):
        d.detect(FRAME, float(i), i)
    assert len(loads) == 1
    assert model.calls == 1 + 5  # one warm-up + one call per frame
    assert d.calls == 5 and d.warmup_seconds >= 0


# --------------------------------------------------------------------------- G. empty / invalid frames


def test_empty_frame_gives_ok_with_no_detections():
    r = fake_detector([]).detect(FRAME, 0.0, 0)
    assert r.status is RoadFrameStatus.OK and r.detections == () and r.detection_count == 0
    assert (r.image_width, r.image_height) == (640, 480) and r.inference_ms is not None


def test_result_without_boxes_object():
    d = fake_detector()
    d.model.predict = lambda frame, **kw: [FakeResult(None)]
    assert d.detect(FRAME).detections == ()


@pytest.mark.parametrize("frame", [None, np.zeros((480, 640), np.uint8), np.zeros((480, 640, 4), np.uint8),
                                   np.zeros((480, 640, 3), np.float32), np.zeros((4, 4, 3), np.uint8), "frame"])
def test_invalid_frame_not_sent_to_model(frame):
    d = fake_detector()
    r = d.detect(frame, 1.0, 1)
    assert r.status is RoadFrameStatus.INVALID_FRAME and r.error and d.model.calls == 0


def test_provider_exception_becomes_provider_error():
    d = YoloRoadDetector("m.pt", device="cpu", model=FakeYolo(raises=True), warmup=False)
    r = d.detect(FRAME, 1.0, 1)
    assert r.status is RoadFrameStatus.PROVIDER_ERROR and "boom" in r.error and r.detections == ()


# --------------------------------------------------------------------------- H. video sampling


class FakeReader:
    """Mimics app.sensors.video.VideoReader: n frames at fps with container timestamps."""

    def __init__(self, n=90, fps=30.0):
        self.n, self.fps, self.i = n, fps, -1
        self.retrieved = 0

    def info(self):
        return VideoInfo("fake.mp4", 640, 480, self.fps, self.n)

    def grab(self):
        self.i += 1
        return self.i < self.n

    def retrieve(self):
        self.retrieved += 1
        return True, FRAME

    def position_seconds(self):
        return self.i / self.fps


class CountingDetector:
    name = "fake"

    def __init__(self):
        self.calls = []

    def detect(self, frame, timestamp=None, frame_index=None):
        from app.road.models import RoadPerceptionResult

        self.calls.append((timestamp, frame_index))
        return RoadPerceptionResult(timestamp, frame_index, 640, 480, (), RoadFrameStatus.OK, "fake", 1.0)


def test_video_sampling_does_not_call_detector_on_every_frame():
    reader, detector = FakeReader(90, 30.0), CountingDetector()
    stats = run_video(reader, detector, sample_fps=5.0)
    assert stats.frames_read == 90
    assert len(detector.calls) == 15 == stats.frames_processed  # 3 s x 5 fps, not 90
    assert reader.retrieved == 15  # only sampled frames are decoded
    ts = [t for t, _ in detector.calls]
    assert ts == pytest.approx([i * 0.2 for i in range(15)], abs=1e-6)  # timestamps preserved
    idx = [i for _, i in detector.calls]
    assert idx == sorted(idx) and idx[:3] == [0, 6, 12]


def test_video_every_frame_and_max_seconds():
    assert len(run_video(FakeReader(30, 30.0), (d := CountingDetector()), sample_fps=0).results) == 30 == len(d.calls)
    d2 = CountingDetector()
    run_video(FakeReader(90, 30.0), d2, sample_fps=10.0, max_seconds=1.0)
    assert max(t for t, _ in d2.calls) <= 1.0 + 1e-6 and len(d2.calls) == 11


def test_video_sampling_with_real_provider_counts_model_calls():
    d = fake_detector([[10, 10, 60, 60, 0.9, ID["dog"]]])
    stats = run_video(FakeReader(60, 30.0), d, sample_fps=5.0)
    assert d.model.calls == 10 and stats.class_counts == {"dog": 10}
    assert all(r.detections[0].timestamp == r.timestamp for r in stats.results)


def test_frame_sampler_grid():
    s = FrameSampler(5.0)
    picked = [t for t in [i / 30 for i in range(31)] if s.should_process(t)]
    assert picked == pytest.approx([0, 0.2, 0.4, 0.6, 0.8, 1.0], abs=1e-6)
    with pytest.raises(ValueError):
        FrameSampler(-1)


# --------------------------------------------------------------------------- images, visualisation, config


def test_image_runner_applies_exif_and_leaves_file_untouched(tmp_path):
    p = tmp_path / "rot.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (80, 40), (10, 20, 30)).save(p, exif=exif)
    before = p.read_bytes()
    assert read_image_bgr(p).shape == (80, 40, 3)  # rotated upright
    stats = run_image(p, fake_detector([[1, 1, 30, 30, 0.9, ID["person"]]]))
    assert stats.frames_processed == 1 and stats.class_counts == {"person": 1}
    assert stats.results[0].timestamp is None
    assert p.read_bytes() == before


def test_draw_detections_returns_annotated_copy():
    r = fake_detector([[100, 100, 300, 300, 0.9, ID["car"]]]).detect(FRAME, 0.0, 0)
    out = draw_detections(FRAME, r, "caption")
    assert out is not FRAME and FRAME.sum() == 0 and out.sum() > 0


def test_road_config_env_and_validation(monkeypatch):
    monkeypatch.setattr("app.config.road_config.load_dotenv", lambda *a, **k: False)
    for k in ["ROAD_MODEL_PATH", "ROAD_CONFIDENCE_THRESHOLD", "ROAD_IMAGE_SIZE", "ROAD_SAMPLE_FPS", "ROAD_CLASSES",
              "ROAD_DEVICE", "DEVICE"]:
        monkeypatch.delenv(k, raising=False)
    c = load_road_config()
    assert c.model_path.name == "yolo26n.pt" and c.confidence_threshold == 0.35 and c.image_size == 640
    assert c.sample_fps == 5.0 and c.device == "auto" and "dog" in c.classes
    monkeypatch.setenv("ROAD_CONFIDENCE_THRESHOLD", "0.5")
    monkeypatch.setenv("ROAD_CLASSES", "car, person")
    monkeypatch.setenv("DEVICE", "cpu")
    c = load_road_config()
    assert c.confidence_threshold == 0.5 and c.classes == ("car", "person") and c.device == "cpu"
    monkeypatch.setenv("ROAD_IMAGE_SIZE", "100")
    with pytest.raises(ValueError, match="multiple of 32"):
        load_road_config()
    assert parse_classes("ALL") == ("all",)


# --------------------------------------------------------------------------- script


def _script():
    spec = importlib.util.spec_from_file_location("road_cli", PROJECT_ROOT / "scripts" / "test_road_perception.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_video(path: Path, n=20, fps=10.0):
    cv2 = pytest.importorskip("cv2")
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), i * 10, np.uint8))
    w.release()


def test_script_prints_rows_and_summary(tmp_path, monkeypatch, capsys):
    import app.road

    monkeypatch.setattr("app.config.settings.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr("app.config.road_config.load_dotenv", lambda *a, **k: False)
    folder = tmp_path / "vehicles"
    folder.mkdir()
    Image.new("RGB", (64, 48)).save(folder / "a.jpg")
    _write_video(folder / "clip.avi")
    rows = [[5, 5, 40, 40, 0.8, ID["car"]]]

    class ScriptDetector(YoloRoadDetector):
        def __init__(self, *a, **k):
            super().__init__("models/road/yolo26n.pt", device="cpu", model=FakeYolo(rows), warmup=False)

    monkeypatch.setattr(app.road, "YoloRoadDetector", ScriptDetector)
    code = _script().main(["--source", str(folder), "--sample-fps", "5", "--jsonl", str(tmp_path / "out.jsonl"),
                           "--save-annotated", str(tmp_path / "dbg")])
    out = capsys.readouterr().out
    assert code == 0
    assert "folder: vehicles (human context, not ground truth)" in out
    assert "| car           | 0.80 |" in out
    assert "Images processed     : 1" in out and "Video frames         : 20 read, 10 sent to YOLO" in out
    assert "car" in out.split("Class counts")[1] and "Inference time" in out
    lines = (tmp_path / "out.jsonl").read_text().splitlines()
    assert len(lines) == 11
    assert (tmp_path / "dbg" / "a_annotated.jpg").is_file() and (tmp_path / "dbg" / "clip_annotated.mp4").is_file()


def test_script_missing_source(capsys, monkeypatch):
    monkeypatch.setattr("app.config.settings.load_dotenv", lambda *a, **k: False)
    assert _script().main(["--source", "does/not/exist.jpg"]) == 2
    assert "not found" in capsys.readouterr().err


# --------------------------------------------------------------------------- F. real pretrained model on CPU


def _real_weights() -> Path | None:
    if importlib.util.find_spec("ultralytics") is None:
        return None
    p = Path(os.environ.get("ROAD_MODEL_PATH") or PROJECT_ROOT / "models" / "road" / "yolo26n.pt")
    return p if p.is_file() else None


real_yolo = pytest.mark.skipif(_real_weights() is None, reason="ultralytics or YOLO weights not available")


@pytest.fixture(scope="module")
def real_detector():
    return YoloRoadDetector(_real_weights(), device="cpu", allow_download=False)


@real_yolo
def test_real_model_loads_on_cpu_and_runs(real_detector):
    assert real_detector.device == "cpu" and len(real_detector.names) == 80
    r = real_detector.detect(np.zeros((720, 1280, 3), np.uint8), 0.0, 0)
    assert r.status is RoadFrameStatus.OK and r.detections == ()  # blank frame: nothing found
    assert r.inference_ms is not None and r.inference_ms > 0


@real_yolo
def test_real_model_detects_a_drawn_scene_structurally(real_detector):
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (640, 640, 3), dtype=np.uint8)
    r = real_detector.detect(frame, 1.5, 3)
    assert r.status is RoadFrameStatus.OK
    for d in r.detections:  # whatever it finds must be valid and road-relevant
        assert d.class_name in DEFAULT_ROAD_CLASSES and d.confidence >= real_detector.confidence_threshold
        assert 0 <= d.bbox_x1 <= d.bbox_x2 <= 640 and d.timestamp == 1.5 and not math.isnan(d.confidence)


@real_yolo
@pytest.mark.skipif(not os.environ.get("DATASET_ROOT"), reason="DATASET_ROOT not set")
def test_real_front_camera_image_if_available(real_detector):
    images = sorted(Path(os.environ["DATASET_ROOT"]).glob("frontcamera*/frontcamera/*/*.jpg"))
    if not images:
        pytest.skip("no front-camera images under DATASET_ROOT")
    stats = run_image(images[0], real_detector)
    assert stats.status_counts == {"OK": 1}
