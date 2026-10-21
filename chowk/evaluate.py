"""Cross-validation by recording session, the zero-shot baselines, and the robustness experiments.

The question every number here answers: if a camera at a Kathmandu junction counted the traffic
for 15 minutes, how far would its total Passenger Car Units be from the truth?
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from chowk import config, model
from chowk.detector import zero_shot_class

CLASSES = list(config.CLASSES)
COCO = ["bicycle", "car", "motorcycle", "bus", "truck"]
# A session needs at least this many vehicles for its PCU error to mean anything.
MIN_VEHICLES_PER_SESSION = 20


def labels(df: pd.DataFrame) -> np.ndarray:
    # pandas 3 stores strings as Arrow arrays, which scikit-learn's fancy indexing can't slice (bug #9).
    return df["label"].to_numpy(dtype=object)


def grouped_cv_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray, c: float = model.DEFAULT_C,
                       balanced: bool = False, folds: int = config.CV_FOLDS) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions and probabilities, never training on a test image's session."""
    pred = np.empty(len(y), dtype=object)
    proba = np.zeros((len(y), len(CLASSES)))
    for train, test in GroupKFold(folds).split(X, y, groups):
        clf = model.make_classifier(c, balanced).fit(X[train], y[train])
        pred[test] = clf.predict(X[test])
        p = clf.predict_proba(X[test])
        proba[np.ix_(test, [CLASSES.index(k) for k in clf.classes_])] = p
    return pred, proba


def random_cv_predict(X: np.ndarray, y: np.ndarray, c: float = model.DEFAULT_C) -> np.ndarray:
    pred = np.empty(len(y), dtype=object)
    for train, test in StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED).split(X, y):
        pred[test] = model.make_classifier(c).fit(X[train], y[train]).predict(X[test])
    return pred


def zero_shot_predict(df: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
    """The off-the-shelf detector's own class. With threshold=None, always pick the best vehicle class
    (fairest to the detector: it is told there is a vehicle). With a threshold, below it -> None."""
    out = []
    for _, r in df[["coco_" + k for k in COCO]].iterrows():
        scores = {k: r["coco_" + k] for k in COCO}
        out.append(zero_shot_class(scores, threshold if threshold is not None else -1.0))
    return np.array(out, dtype=object)


def pcu_of(classes: np.ndarray, pcu: dict[str, float] = config.PCU) -> np.ndarray:
    return np.array([pcu.get(c, 0.0) if c is not None else 0.0 for c in classes])


@dataclass
class PcuError:
    median_session_error: float  # median over sessions of |predicted - true| / true
    worst_session_error: float
    total_bias: float  # (sum predicted - sum true) / sum true, over all sessions
    sessions: int


def pcu_error(df: pd.DataFrame, pred: np.ndarray, pcu: dict[str, float] = config.PCU) -> PcuError:
    t = pd.DataFrame({"session": df["session"].to_numpy(), "true": pcu_of(labels(df), pcu),
                      "pred": pcu_of(pred, pcu)})
    s = t.groupby("session").agg(true=("true", "sum"), pred=("pred", "sum"), n=("true", "size"))
    s = s[s["n"] >= MIN_VEHICLES_PER_SESSION]
    err = (s["pred"] - s["true"]).abs() / s["true"]
    return PcuError(round(float(err.median()), 4), round(float(err.max()), 4),
                    round(float((s["pred"].sum() - s["true"].sum()) / s["true"].sum()), 4), len(s))


def scores(df: pd.DataFrame, pred: np.ndarray, pcu: dict[str, float] = config.PCU) -> dict:
    y = labels(df)
    mask = np.array([p is not None for p in pred])
    filled = np.where(mask, pred, "none")
    return {
        "accuracy": round(float((filled == y).mean()), 4),
        "macro_f1": round(float(f1_score(y, filled, labels=CLASSES, average="macro")), 4),
        "f1": dict(zip(CLASSES, np.round(f1_score(y, filled, labels=CLASSES, average=None), 4).tolist(),
                       strict=True)),
        "pcu": pcu_error(df, pred, pcu).__dict__,
        "confusion": confusion_matrix(y, filled, labels=CLASSES).tolist(),
    }


# ---------------------------------------------------------------- robustness
# Each perturbation is a thing that really happens to a junction camera in Kathmandu.
def dusk(img: Image.Image) -> Image.Image:
    """Evening light: a third of the brightness, half the contrast, sensor noise."""
    out = ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(0.35)).enhance(0.5)
    arr = np.asarray(out, dtype=np.float32)
    rng = np.random.default_rng(abs(hash(img.size)) % (2**32))
    arr = arr + rng.normal(0, 8, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def far_away(img: Image.Image) -> Image.Image:
    """The same vehicle at a quarter of the pixels: a cheap camera mounted high, or a far lane."""
    small = img.resize((max(8, img.width // 4), max(8, img.height // 4)), Image.BILINEAR)
    return small.resize(img.size, Image.BILINEAR)


def dusty_lens(img: Image.Image) -> Image.Image:
    """Dust and monsoon spray on the housing: blur plus a grey veil."""
    veil = Image.new("RGB", img.size, (170, 170, 170))
    return Image.blend(img.filter(ImageFilter.GaussianBlur(3)), veil, 0.35)


def low_bitrate(img: Image.Image) -> Image.Image:
    """Heavy JPEG compression, as from a camera streaming over a weak 4G link."""
    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=10)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


PERTURBATIONS = {"dusk": dusk, "far_away": far_away, "dusty_lens": dusty_lens, "low_bitrate": low_bitrate}


def perturbed_features(df: pd.DataFrame, detector, archive) -> dict[str, np.ndarray]:
    from chowk import data

    out = {}
    for name, fn in PERTURBATIONS.items():
        out[name] = np.stack([detector.crop_features(fn(data.load_image(archive, p)))[0] for p in df["path"]])
    return out


def robustness(df: pd.DataFrame, X_clean: np.ndarray, perturbed: dict[str, np.ndarray]) -> dict:
    """Test on degraded crops of held-out sessions (same folds as the main CV), three ways:

      trained_clean          the model as first built: clean crops only
      trained_augmented      clean + every degradation, including the one tested (an upper bound:
                             it has seen this exact synthetic degradation)
      trained_without_this   clean + the *other* three degradations: does it generalise to a kind
                             of damage it never saw? (the fairer test)
    Plus "clean": does augmentation cost anything on undamaged images?
    """
    y, groups = labels(df), df["session"].to_numpy()
    folds = list(GroupKFold(config.CV_FOLDS).split(X_clean, y, groups))

    def fit_aug(train, names):
        Xa = np.concatenate([X_clean[train]] + [perturbed[n][train] for n in names])
        return model.make_classifier().fit(Xa, np.concatenate([y[train]] * (1 + len(names))))

    results = {}
    for name, Xp in perturbed.items():
        preds = {k: np.empty(len(y), dtype=object) for k in ("trained_clean", "trained_augmented",
                                                              "trained_without_this")}
        others = [n for n in perturbed if n != name]
        for train, test in folds:
            preds["trained_clean"][test] = model.make_classifier().fit(X_clean[train], y[train]).predict(Xp[test])
            preds["trained_augmented"][test] = fit_aug(train, list(perturbed)).predict(Xp[test])
            preds["trained_without_this"][test] = fit_aug(train, others).predict(Xp[test])
        results[name] = {k: scores(df, v) for k, v in preds.items()}
    clean_aug = np.empty(len(y), dtype=object)
    for train, test in folds:
        clean_aug[test] = fit_aug(train, list(perturbed)).predict(X_clean[test])
    results["clean"] = {"trained_augmented": scores(df, clean_aug)}
    return results


def input_check_rates(df: pd.DataFrame, archive, n: int = 500) -> dict:
    """How often chowk.quality flags a clean crop, and each kind of degraded crop."""
    from chowk import data, quality

    idx = np.random.default_rng(1).choice(len(df), min(n, len(df)), replace=False)
    out = {}
    for name, fn in {"clean": lambda im: im, **PERTURBATIONS}.items():
        f = [quality.flags(fn(data.load_image(archive, df["path"].iat[i]))) for i in idx]
        out[name] = {"any": round(float(np.mean([len(x) > 0 for x in f])), 3),
                     **{k: round(float(np.mean([k in x for x in f])), 3) for k in quality.FLAGS}}
    return out


def pcu_sensitivity(df: pd.DataFrame, preds: dict[str, np.ndarray]) -> dict:
    """How much the headline depends on my microbus assumption (medium = 1.5 PCU)."""
    out = {}
    for medium in (1.0, 1.5, 3.0):
        pcu = dict(config.PCU, medium=medium)
        out[str(medium)] = {name: pcu_error(df, p, pcu).__dict__ for name, p in preds.items()}
    return out


def write(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
