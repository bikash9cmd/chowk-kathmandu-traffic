"""YOLOX-Nano on ONNX Runtime: vehicle boxes plus a feature vector for each box.

One forward pass does two jobs. The detector's boxes are kept (finding vehicles transfers well
from COCO to Kathmandu), but its class labels are not trusted (they do not: see README, Results).
Instead, two intermediate feature maps are pooled under each box and handed to a small classifier
trained on Kathmandu vehicles (chowk/model.py). No second network, no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from chowk import config
from chowk.data import fetch

PAD_VALUE = 114  # YOLOX letterbox padding colour
STRIDES = (8, 16, 32)


def model_path(artifact_dir: Path = config.ARTIFACT_DIR) -> Path:
    return fetch(config.DETECTOR_URL, artifact_dir / "weights" / "yolox_nano.onnx", config.DETECTOR_SHA256)


def with_feature_outputs(onnx_path: Path) -> bytes:
    """The detector graph with the two feature tensors added as extra outputs."""
    import onnx

    model = onnx.load(str(onnx_path))
    existing = {o.name for o in model.graph.output}
    for name in config.FEATURE_TENSORS:
        if name not in existing:
            model.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
    return model.SerializeToString()


@dataclass
class Letterbox:
    tensor: np.ndarray  # 1x3xSxS float32, BGR, 0-255 (YOLOX ONNX expects no normalisation)
    scale: float
    width: int  # resized image size inside the square
    height: int


def letterbox(img: Image.Image, size: int = config.DETECTOR_INPUT) -> Letterbox:
    # Every training crop is greyscale (bug #12). A colour camera would be a distribution shift the
    # classifier has never seen, so frames are made greyscale before anything else.
    img = img.convert("L").convert("RGB")
    r = min(size / img.width, size / img.height)
    w, h = max(1, int(img.width * r)), max(1, int(img.height * r))
    canvas = np.full((size, size, 3), PAD_VALUE, np.uint8)
    canvas[:h, :w] = np.asarray(img.resize((w, h), Image.BILINEAR))[:, :, ::-1]
    return Letterbox(canvas.transpose(2, 0, 1)[None].astype(np.float32), r, w, h)


def _grids(size: int) -> tuple[np.ndarray, np.ndarray]:
    grids, strides = [], []
    for s in STRIDES:
        n = size // s
        yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        ious = inter / (area(boxes[i]) + area(boxes[order[1:]]) - inter + 1e-9)
        order = order[1:][ious <= iou]
    return keep


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in original image pixels
    score: float
    coco_class: str
    features: np.ndarray  # pooled feature vector for the classifier


class Detector:
    def __init__(self, onnx_path: Path | None = None, threads: int = 2):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(with_feature_outputs(onnx_path or model_path()), opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size = config.DETECTOR_INPUT
        self.grid, self.stride = _grids(self.size)
        self._coco_names = {v: k for k, v in config.COCO_VEHICLE_IDS.items()}

    def _forward(self, lb: Letterbox):
        raw, f32, f16 = self.session.run(None, {self.input_name: lb.tensor})
        return raw[0], f32[0], f16[0]

    @staticmethod
    def pool(fmap: np.ndarray, stride: int, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        """Mean of a feature map over a box given in letterboxed-input pixels."""
        n = fmap.shape[1]
        c1, r1 = int(np.clip(x1 // stride, 0, n - 1)), int(np.clip(y1 // stride, 0, n - 1))
        c2, r2 = int(np.clip(np.ceil(x2 / stride), c1 + 1, n)), int(np.clip(np.ceil(y2 / stride), r1 + 1, n))
        return fmap[:, r1:r2, c1:c2].mean(axis=(1, 2))

    def features(self, fm32: np.ndarray, fm16: np.ndarray, box_in: tuple[float, float, float, float]) -> np.ndarray:
        return np.concatenate([self.pool(fm32, 32, *box_in), self.pool(fm16, 16, *box_in)]).astype(np.float32)

    def crop_features(self, img: Image.Image) -> tuple[np.ndarray, dict[str, float]]:
        """For an image that is already one vehicle: its feature vector and the best COCO score per class."""
        lb = letterbox(img, self.size)
        raw, f32, f16 = self._forward(lb)
        scores = raw[:, 4:5] * raw[:, 5:]
        best = {name: float(scores[:, i].max()) for name, i in config.COCO_VEHICLE_IDS.items()}
        best["person"] = float(scores[:, 0].max())
        return self.features(f32, f16, (0, 0, lb.width, lb.height)), best

    def detect(self, img: Image.Image, score_min: float = config.DETECT_SCORE_MIN) -> list[Detection]:
        """Vehicles in a full camera frame, class-agnostic NMS over the five COCO vehicle classes."""
        lb = letterbox(img, self.size)
        raw, f32, f16 = self._forward(lb)
        xy = (raw[:, :2] + self.grid) * self.stride
        wh = np.exp(raw[:, 2:4]) * self.stride
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)
        ids = np.array(list(config.COCO_VEHICLE_IDS.values()))
        cls_scores = raw[:, 4:5] * raw[:, 5:][:, ids]
        best = cls_scores.argmax(1)
        score = cls_scores.max(1)
        mask = score >= score_min
        boxes, score, best = boxes[mask], score[mask], best[mask]
        out = []
        for i in nms(boxes, score, config.NMS_IOU):
            b = boxes[i]
            orig = tuple(float(v / lb.scale) for v in b)
            out.append(Detection(orig, float(score[i]), self._coco_names[int(ids[best[i]])],
                                 self.features(f32, f16, tuple(b))))
        return out


def zero_shot_class(coco_scores: dict[str, float], score_min: float = config.DETECT_SCORE_MIN) -> str | None:
    """The off-the-shelf answer: the best-scoring COCO vehicle class, mapped to a PCU bucket.

    None means the detector found no vehicle at all above its usual threshold.
    """
    name = max(config.COCO_TO_CLASS, key=lambda k: coco_scores[k])
    if coco_scores[name] < score_min:
        return None
    return config.COCO_TO_CLASS[name]
