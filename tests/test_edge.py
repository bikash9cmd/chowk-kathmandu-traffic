from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from PIL import Image

from chowk.counting import CountLine
from chowk.edge import CameraBox, Outbox


@dataclass
class FakeDet:
    box: tuple
    features: np.ndarray


class FakeDetector:
    """One vehicle driving down through the frame, again and again."""

    def __init__(self):
        self.k = 0

    def detect(self, _frame):
        y = 150 + 10 * (self.k % 30)
        self.k += 1
        return [FakeDet((100.0, float(y), 160.0, float(y + 60)), np.zeros(4, dtype=np.float32))]


class FakeClassifier:
    classes_ = np.array(["heavy", "light", "medium", "two_wheeler"])

    def predict_proba(self, X):
        return np.tile([0.0, 0.9, 0.1, 0.0], (len(X), 1))


def make_box(tmp_path):
    return CameraBox("maitighar", [CountLine("from_thapathali", 300)], FakeDetector(), FakeClassifier(), "v1",
                     Outbox(tmp_path / "outbox"))


def test_interval_rollover_writes_one_record_per_approach(tmp_path):
    box = make_box(tmp_path)
    frame = Image.new("RGB", (640, 480))
    t = datetime(2026, 10, 1, 8, 0, 0)
    closed = []
    for i in range(int(15 * 60 / 0.5) + 2):  # 2 fps, just past the 15-minute boundary
        closed += box.process(frame, t + timedelta(seconds=0.5 * i))
    assert len(closed) == 1
    rec = closed[0]
    assert rec["interval_start"] == "2026-10-01T08:00:00"
    assert rec["counts"]["light"] == 60  # one crossing every 30 frames
    assert abs(rec["minutes_observed"] - 15) < 0.05
    assert len(box.outbox.pending()) == 1


def test_camera_outage_shows_up_as_fewer_minutes_observed(tmp_path):
    box = make_box(tmp_path)
    frame = Image.new("RGB", (640, 480))
    t = datetime(2026, 10, 1, 8, 0, 0)
    for i in range(600):  # 5 minutes at 2 fps
        box.process(frame, t + timedelta(seconds=0.5 * i))
    # camera dead from 08:05 to 08:14, then one frame in the next interval
    closed = box.process(frame, datetime(2026, 10, 1, 8, 15, 1))
    rec = closed[0]
    assert 4.9 < rec["minutes_observed"] < 5.1
    # the rate is scaled from what was seen (about 3x the raw count), not reported as a 15-minute total
    assert abs(rec["pcu_per_hour"] - rec["pcu"] * 60 / 5) < 1


def test_outbox_keeps_order_and_stops_at_first_failure(tmp_path):
    ob = Outbox(tmp_path / "o")
    for m in ("00", "15", "30"):
        ob.put({"junction": "j", "approach": "a", "interval_start": f"2026-10-01T08:{m}:00"})
    sent = []

    def flaky(rec):
        if rec["interval_start"].endswith("30:00"):
            return False  # link drops
        sent.append(rec["interval_start"])
        return True

    assert ob.flush(flaky) == 2
    assert sent == ["2026-10-01T08:00:00", "2026-10-01T08:15:00"]
    assert len(ob.pending()) == 1
    assert ob.flush(lambda r: True) == 1


def test_model_swap_changes_version_on_next_record(tmp_path):
    box = make_box(tmp_path)
    frame = Image.new("RGB", (640, 480))
    box.process(frame, datetime(2026, 10, 1, 8, 0, 0))
    box.swap_model(FakeClassifier(), "v2")
    closed = box.process(frame, datetime(2026, 10, 1, 8, 15, 0))
    assert closed[0]["model_version"] == "v2"


def test_dark_frames_are_flagged_in_the_record(tmp_path):
    box = make_box(tmp_path)
    dark = Image.new("RGB", (640, 480), (20, 20, 20))
    t = datetime(2026, 10, 1, 18, 0, 0)
    for i in range(100):
        box.process(dark, t + timedelta(seconds=0.5 * i))
    rec = box.process(dark, datetime(2026, 10, 1, 18, 15, 0))[0]
    assert rec["image_quality"]["dark"] == 1.0


def test_quality_flags_on_real_like_images():
    from chowk import quality

    rng = np.random.default_rng(0)
    textured = Image.fromarray(rng.integers(30, 230, (48, 64), dtype=np.uint8)).resize((320, 240)).convert("RGB")
    assert quality.flags(textured) == set()
    assert "dark" in quality.flags(Image.new("RGB", (320, 240), (10, 10, 10)))
    assert "blurry" in quality.flags(Image.new("RGB", (320, 240), (128, 128, 128)))
