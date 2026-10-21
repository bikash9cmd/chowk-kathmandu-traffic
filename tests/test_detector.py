import numpy as np
import pytest
from PIL import Image

from chowk import config
from chowk.detector import Detector, letterbox, nms, zero_shot_class


def test_letterbox_keeps_aspect_and_pads():
    lb = letterbox(Image.new("RGB", (100, 200)), 416)
    assert lb.tensor.shape == (1, 3, 416, 416)
    assert (lb.width, lb.height) == (208, 416)
    assert lb.tensor[0, :, 0, 300].tolist() == [114.0, 114.0, 114.0]  # padding on the right


def test_nms_drops_overlapping_lower_score():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], dtype=float)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.45)
    assert keep == [0, 2]


def test_pool_never_returns_an_empty_slice():
    fmap = np.arange(2 * 13 * 13, dtype=np.float32).reshape(2, 13, 13)
    v = Detector.pool(fmap, 32, 410, 410, 416, 416)  # box in the last cell
    assert np.isfinite(v).all() and v.shape == (2,)


def test_zero_shot_mapping_and_threshold():
    scores = {"bicycle": 0.0, "car": 0.2, "motorcycle": 0.1, "bus": 0.6, "truck": 0.1}
    assert zero_shot_class(scores) == "heavy"
    assert zero_shot_class(dict.fromkeys(scores, 0.1)) is None
    assert "medium" not in config.COCO_TO_CLASS.values()  # COCO can never say "microbus"


@pytest.mark.skipif(not (config.ARTIFACT_DIR / "weights" / "yolox_nano.onnx").exists(),
                    reason="detector weights not downloaded (python -m chowk download)")
def test_detector_returns_256_features_for_a_crop():
    det = Detector()
    f, scores = det.crop_features(Image.new("RGB", (120, 200), (128, 128, 128)))
    assert f.shape == (256,) and set(scores) >= set(config.COCO_TO_CLASS)
    assert det.detect(Image.new("RGB", (640, 480))) == []  # a blank frame has no vehicles
