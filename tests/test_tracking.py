"""IoU tracker: association, IDs, miss handling, image-space motion features, label changes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.config.tracking_config import TrackingConfig, load_tracking_config
from app.road.models import Detection
from app.tracking import IoUTracker, iou
from app.tracking.visualize import draw_tracks, track_label


def D(x1, y1, x2, y2, cls="car", conf=0.9, t=None):
    return Detection(timestamp=t, class_name=cls, class_id=0, confidence=conf, bbox_x1=x1, bbox_y1=y1,
                     bbox_x2=x2, bbox_y2=y2, source="test")


def box(cx, cy, w=40, h=40, cls="car", conf=0.9):
    return D(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cls, conf)


def by_id(tracks):
    return {t.track_id: t for t in tracks}


# --------------------------------------------------------------------------- IoU helper


def test_iou_values():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


# --------------------------------------------------------------------------- A-D association and IDs


def test_A_first_detection_creates_track_1():
    tracks = IoUTracker().update([box(100, 100)], 0.0, 0)
    assert len(tracks) == 1
    t = tracks[0]
    assert t.track_id == 1 and t.active and t.detection_count == 1 and t.class_name == "car"
    assert t.velocity_x_pixels_per_second is None and t.area_change_rate is None  # needs two observations
    assert t.persistence_seconds == 0 and t.age_seconds == 0 and t.missed_seconds == 0


def test_B_same_object_keeps_id_across_frames():
    tr = IoUTracker()
    ids = [tr.update([box(100 + 5 * i, 100)], i * 0.1, i)[0].track_id for i in range(10)]
    assert ids == [1] * 10 and tr.tracks_created == 1


def test_C_two_objects_get_distinct_ids():
    tracks = IoUTracker().update([box(100, 100), box(400, 100)], 0.0)
    assert [t.track_id for t in tracks] == [1, 2]


def test_D_moving_objects_keep_their_ids():
    tr = IoUTracker()
    for i in range(8):  # A moves right, B moves down; input order swapped on odd frames
        a, b = box(100 + 8 * i, 100), box(400, 100 + 8 * i, cls="car")
        tracks = tr.update([a, b] if i % 2 == 0 else [b, a], i * 0.1)
    t = by_id(tracks)
    assert set(t) == {1, 2}
    assert t[1].center_x == pytest.approx(156) and t[1].center_y == pytest.approx(100)
    assert t[2].center_x == pytest.approx(400) and t[2].center_y == pytest.approx(156)


def test_greedy_prefers_highest_iou():
    tr = IoUTracker()
    tr.update([box(100, 100), box(130, 100)], 0.0)  # two overlapping cars
    tracks = by_id(tr.update([box(132, 100), box(101, 100)], 0.1))  # swapped order, small moves
    assert tracks[1].center_x == pytest.approx(101) and tracks[2].center_x == pytest.approx(132)


# --------------------------------------------------------------------------- E. class handling


def test_E_class_mismatch_prevents_association():
    tr = IoUTracker(TrackingConfig(iou_threshold=0.3, class_switch_iou=0.6))
    tr.update([box(100, 100, cls="car")], 0.0)
    tracks = by_id(tr.update([box(115, 100, cls="person")], 0.1))  # IoU ~0.45: enough for same class, not for a switch
    assert iou((80, 80, 120, 120), (95, 80, 135, 120)) == pytest.approx(1000 / 2200)  # 0.45: between the two thresholds
    assert set(tracks) == {1, 2}
    assert not tracks[1].active and tracks[2].class_name == "person"


def test_label_flicker_on_same_box_is_recorded_not_rewritten():
    tr = IoUTracker()
    tr.update([box(100, 100, cls="dog")], 0.0)
    tr.update([box(101, 100, cls="person")], 0.2)  # same box, YOLO relabelled it
    t = tr.update([box(102, 100, cls="dog")], 0.4)[0]
    assert t.track_id == 1 and tr.tracks_created == 1
    assert t.class_history == ("dog", "person", "dog") and t.class_changes == 2
    assert t.observed_class == "dog" and t.class_name == "dog"
    t2 = tr.update([box(103, 100, cls="person")], 0.6)[0]
    assert t2.observed_class == "person"  # detector label kept as observed
    assert t2.class_name == "dog"  # track label = majority (2 dog vs 2 person -> earliest)


def test_cross_class_matching_can_be_disabled():
    tr = IoUTracker(TrackingConfig(class_switch_iou=1.5))
    tr.update([box(100, 100, cls="dog")], 0.0)
    assert [t.track_id for t in tr.update([box(100, 100, cls="person")], 0.2)] == [1, 2]


def test_same_class_match_preferred_over_label_switch():
    tr = IoUTracker()
    tr.update([box(100, 100, cls="dog")], 0.0)
    tracks = by_id(tr.update([box(100, 100, cls="person"), box(104, 100, cls="dog")], 0.2))  # duplicate boxes
    assert tracks[1].observed_class == "dog" and tracks[1].class_changes == 0
    assert tracks[2].observed_class == "person"


# --------------------------------------------------------------------------- F-G. miss handling


def test_F_short_miss_keeps_track():
    tr = IoUTracker(TrackingConfig(max_missed_seconds=1.0))
    tr.update([box(100, 100)], 0.0)
    coasting = tr.update([], 0.4)
    assert len(coasting) == 1 and not coasting[0].active and coasting[0].missed_seconds == pytest.approx(0.4)
    assert coasting[0].missed_frames == 1
    back = tr.update([box(110, 100)], 0.8)[0]
    assert back.track_id == 1 and back.active and back.detection_count == 2 and back.missed_frames == 0
    assert back.velocity_x_pixels_per_second == pytest.approx(10 / 0.8)  # divided by the real gap


def test_G_track_expires_after_configured_missed_time():
    tr = IoUTracker(TrackingConfig(max_missed_seconds=0.5))
    tr.update([box(100, 100)], 0.0)
    assert len(tr.update([], 0.5)) == 1  # exactly at the limit: still alive
    assert tr.update([], 0.6) == []
    assert [t.track_id for t in tr.ended_last_update] == [1] and tr.ended_last_update[0].ended
    assert tr.update([box(100, 100)], 0.7)[0].track_id == 2  # same place later = new track


def test_G_track_expires_after_configured_missed_frames():
    tr = IoUTracker(TrackingConfig(max_missed_seconds=10, max_missed_frames=2))
    tr.update([box(100, 100)], 0.0)
    assert len(tr.update([], 0.1)) == 1 and len(tr.update([], 0.2)) == 1
    assert tr.update([], 0.3) == []


# --------------------------------------------------------------------------- H-J. motion features


def test_H_center_velocity():
    tr = IoUTracker()
    tr.update([box(100, 200)], 1.0)
    t = tr.update([box(110, 194)], 1.2)[0]
    assert t.previous_center_x == 100 and t.previous_center_y == 200
    assert t.velocity_x_pixels_per_second == pytest.approx(50.0)
    assert t.velocity_y_pixels_per_second == pytest.approx(-30.0)
    assert t.speed_pixels_per_second == pytest.approx(math.hypot(50, 30))


def test_I_area_change_rate():
    tr = IoUTracker()
    tr.update([box(100, 100, w=40, h=40)], 0.0)  # area 1600
    t = tr.update([box(100, 100, w=50, h=48)], 0.5)[0]  # area 2400
    assert t.previous_area == 1600 and t.area == 2400
    assert t.area_change_rate == pytest.approx(1600.0)  # (2400 - 1600) / 0.5 px^2/s
    assert t.relative_area_change_rate == pytest.approx(1.0)  # +50 % per 0.5 s


def test_J_persistence_and_age():
    tr = IoUTracker()
    for t_ in (2.0, 2.2, 2.4, 2.6):
        last = tr.update([box(100, 100)], t_)[0]
    assert last.first_seen_timestamp == 2.0 and last.last_seen_timestamp == 2.6
    assert last.persistence_seconds == pytest.approx(0.6) and last.age_seconds == pytest.approx(0.6)
    coasting = tr.update([], 3.0)[0]
    assert coasting.persistence_seconds == pytest.approx(0.6)  # not extended while unseen
    assert coasting.age_seconds == pytest.approx(1.0) and coasting.missed_seconds == pytest.approx(0.4)


def test_mean_and_latest_confidence():
    tr = IoUTracker()
    tr.update([box(100, 100, conf=0.8)], 0.0)
    t = tr.update([box(101, 100, conf=0.6)], 0.1)[0]
    assert t.class_confidence == pytest.approx(0.6) and t.mean_confidence == pytest.approx(0.7)


# --------------------------------------------------------------------------- K. timestamps


def test_K_irregular_timestamps_use_real_dt():
    tr = IoUTracker()
    tr.update([box(100, 100)], 0.0)
    assert tr.update([box(105, 100)], 0.05)[0].velocity_x_pixels_per_second == pytest.approx(100)
    assert tr.update([box(125, 100)], 0.45)[0].velocity_x_pixels_per_second == pytest.approx(50)


def test_K_long_gap_ends_tracks_before_matching():
    tr = IoUTracker(TrackingConfig(max_missed_seconds=1.0))
    tr.update([box(100, 100)], 0.0)
    t = tr.update([box(100, 100)], 5.0)  # identical box after 5 s: not linked across the gap
    assert [x.track_id for x in t] == [2] and [x.track_id for x in tr.ended_last_update] == [1]


def test_K_backwards_timestamp_starts_new_timeline():
    tr = IoUTracker()
    tr.update([box(100, 100)], 5.0)
    t = tr.update([box(100, 100)], 1.0)
    assert tr.timeline_resets == 1 and [x.track_id for x in t] == [2]


def test_K_same_timestamp_does_not_divide_by_zero():
    tr = IoUTracker()
    tr.update([box(100, 100)], 1.0)
    t = tr.update([box(120, 100)], 1.0)[0]
    assert t.velocity_x_pixels_per_second is None and t.detection_count == 2


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "1.0", True])
def test_K_invalid_timestamp_rejected(bad):
    with pytest.raises(ValueError):
        IoUTracker().update([], bad)


# --------------------------------------------------------------------------- L-N


def test_L_empty_frame():
    tr = IoUTracker()
    assert tr.update([], 0.0) == [] and tr.tracks_created == 0
    assert tr.all_tracks() == []


def test_M_multiple_simultaneous_tracks():
    tr = IoUTracker()
    classes = ["car", "person", "dog", "truck", "bicycle"]
    for i in range(5):
        tracks = tr.update([box(100 + 150 * k + 3 * i, 100 + 2 * i, cls=c) for k, c in enumerate(classes)], i * 0.2)
    assert [t.track_id for t in tracks] == [1, 2, 3, 4, 5]
    assert [t.class_name for t in tracks] == classes
    assert all(t.detection_count == 5 and t.velocity_x_pixels_per_second == pytest.approx(15) for t in tracks)


def _scenario(tr):
    rng = np.random.default_rng(3)
    out = []
    for i in range(30):
        dets = [box(100 + 4 * i + rng.normal(0, 1), 100, cls="car"), box(300, 100 + 3 * i, cls="person")]
        if i % 7 == 3:
            dets = dets[:1]  # person missed now and then
        if i % 11 == 5:
            dets.append(box(500, 400, cls="dog"))
        out.append([t.to_dict() for t in tr.update(dets, i * 0.2, i)])
    return out


def test_N_deterministic_ids_and_values():
    assert _scenario(IoUTracker()) == _scenario(IoUTracker())
    tr = IoUTracker()
    _scenario(tr)
    tr.reset()
    assert tr.update([box(1, 1, w=10, h=10)], 0.0)[0].track_id == 1  # reset restarts IDs


def test_finish_and_all_tracks():
    tr = IoUTracker()
    tr.update([box(100, 100), box(400, 400, cls="person")], 0.0)
    tr.update([box(102, 100)], 0.2)
    ended = tr.finish()
    assert [t.track_id for t in ended] == [1, 2] and all(t.ended for t in ended)
    allt = tr.all_tracks()
    assert [t.track_id for t in allt] == [1, 2] and allt[0].detection_count == 2


def test_to_dict_uses_pixel_units_only():
    tr = IoUTracker()
    tr.update([box(100, 100)], 0.0)
    d = tr.update([box(110, 100)], 0.2)[0].to_dict()
    for key in ["track_id", "class_name", "class_confidence", "first_seen_timestamp", "last_seen_timestamp",
                "age_seconds", "missed_seconds", "bbox", "center_x", "center_y", "bbox_width", "bbox_height",
                "previous_center_x", "previous_center_y", "velocity_x_pixels_per_second",
                "velocity_y_pixels_per_second", "speed_pixels_per_second", "area", "previous_area",
                "area_change_rate", "detection_count", "persistence_seconds", "active", "observed_class",
                "class_history"]:
        assert key in d, key
    assert not any(u in k for k in d for u in ("meter", "kmh", "km_h", "ttc", "distance"))


# --------------------------------------------------------------------------- config, visualisation, speed


def test_tracking_config_env_and_validation(monkeypatch):
    monkeypatch.setattr("app.config.tracking_config.load_dotenv", lambda *a, **k: False)
    for k in ["TRACK_IOU_THRESHOLD", "TRACK_CLASS_SWITCH_IOU", "TRACK_MAX_MISSED_SECONDS", "TRACK_MAX_MISSED_FRAMES"]:
        monkeypatch.delenv(k, raising=False)
    assert load_tracking_config() == TrackingConfig()
    monkeypatch.setenv("TRACK_MAX_MISSED_SECONDS", "0.5")
    assert load_tracking_config().max_missed_seconds == 0.5
    monkeypatch.setenv("TRACK_IOU_THRESHOLD", "0.9")  # stricter than class_switch_iou 0.6
    with pytest.raises(ValueError):
        load_tracking_config()
    with pytest.raises(ValueError):
        IoUTracker(TrackingConfig(iou_threshold=0))


def test_draw_tracks_and_label():
    tr = IoUTracker()
    tr.update([box(100, 100, cls="dog")], 0.0)
    t = tr.update([box(104, 100, cls="person", conf=0.82)], 0.4)[0]
    assert track_label(t) == "ID 1 | person* | 0.82 | 0.4s"
    frame = np.zeros((300, 400, 3), np.uint8)
    out = draw_tracks(frame, [t] + tr.update([], 0.6), "cap")
    assert frame.sum() == 0 and out.sum() > 0


def test_tracker_is_lightweight():
    import time

    tr = IoUTracker()
    dets = [box(60 * k + 30, 100 + (k % 3) * 60, cls=["car", "person"][k % 2]) for k in range(20)]
    start = time.perf_counter()
    for i in range(100):
        tr.update(dets, i * 0.1)
    per_frame_ms = (time.perf_counter() - start) * 10
    assert per_frame_ms < 20  # 20 objects per frame, far below the ~70 ms YOLO call


# --------------------------------------------------------------------------- script (fake YOLO, real video file)


def test_tracking_script_end_to_end(tmp_path, monkeypatch, capsys):
    import importlib.util
    import json
    import sys
    from pathlib import Path

    cv2 = pytest.importorskip("cv2")
    import app.road
    from app.road import YoloRoadDetector

    monkeypatch.setattr("app.config.settings.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr("app.config.road_config.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr("app.config.tracking_config.load_dotenv", lambda *a, **k: False)
    folder = tmp_path / "vehicles"
    folder.mkdir()
    w = cv2.VideoWriter(str(folder / "clip.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (320, 240))
    for i in range(40):  # 2 s at 20 fps
        w.write(np.zeros((240, 320, 3), np.uint8))
    w.release()

    class Boxes:
        def __init__(self, rows):
            a = np.array(rows, dtype=np.float32).reshape(-1, 6)
            self.xyxy, self.conf, self.cls = a[:, :4], a[:, 4], a[:, 5]

        def __len__(self):
            return len(self.conf)

    class MovingCarYolo:  # a car moving right by 4 px per call and growing; a person that vanishes
        names = {0: "person", 2: "car"}

        def __init__(self):
            self.calls = 0

        def predict(self, frame, **kw):
            i = self.calls
            self.calls += 1
            rows = [[20 + 4 * i, 50, 80 + 5 * i, 110 + i, 0.9, 2]]
            if i < 5:
                rows.append([250, 60, 280, 140, 0.8, 0])
            return [type("R", (), {"boxes": Boxes(rows)})()]

    class ScriptDetector(YoloRoadDetector):
        def __init__(self, *a, **k):
            super().__init__("models/road/yolo26n.pt", device="cpu", model=MovingCarYolo(), warmup=False,
                             classes=("car", "person"))

    monkeypatch.setattr(app.road, "YoloRoadDetector", ScriptDetector)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("track_cli", root / "scripts" / "test_road_tracking.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    out_jsonl = tmp_path / "tracks.jsonl"
    code = mod.main(["--source", str(folder), "--sample-fps", "10", "--jsonl", str(out_jsonl),
                     "--save-annotated", str(tmp_path / "dbg")])
    out = capsys.readouterr().out
    assert code == 0
    assert "folder: vehicles (human context, not ground truth)" in out
    assert "| car           | 0.90 |" in out and "NEW" in out and "END" in out
    assert "Tracks created        : 2" in out
    assert "Growing bbox area     : 1 track(s)" in out
    assert "image-space only" in out and "Time per frame" in out
    frames = [json.loads(l) for l in out_jsonl.read_text().splitlines()]
    assert len(frames) == 20  # 2 s sampled at 10 fps, not all 40 frames
    car = [t for t in frames[-1]["tracks"] if t["class_name"] == "car"][0]
    assert car["track_id"] == 1 and car["detection_count"] == 20
    assert car["velocity_x_pixels_per_second"] == pytest.approx(45, abs=1)  # 4.5 px center shift per 0.1 s
    assert (tmp_path / "dbg" / "clip_tracks.mp4").is_file()
