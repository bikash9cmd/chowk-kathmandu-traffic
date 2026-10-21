"""The Kathmandu vehicle classifier: a linear model on the detector's own features.

Deliberately small. 256 inputs, four classes, about 1,000 weights. It trains in under a
second, and runs as a few multiply-adds on top of the detector pass the camera box already does.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from chowk import config

# Chosen by session-grouped cross-validation over {0.01, 0.03, 0.1, 0.3, 1}; see docs/CHALLENGES_AND_BUGS.md #8.
DEFAULT_C = 0.1


def make_classifier(c: float = DEFAULT_C, balanced: bool = False) -> Pipeline:
    # No class reweighting. I started with class_weight="balanced" because medium vehicles are only 8% of
    # the data; measured, it made both macro-F1 and the PCU error worse by pushing ~170 cars into
    # "medium" (bug #7).
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=3000, class_weight="balanced" if balanced else None),
    )


def fit(X: np.ndarray, y: np.ndarray, c: float = DEFAULT_C) -> Pipeline:
    clf = make_classifier(c)
    clf.fit(X, y)
    return clf


@dataclass
class ModelMeta:
    version: str
    trained_on: int
    classes: list[str]
    cv_macro_f1: float
    cv_pcu_error: float
    detector_sha256: str = config.DETECTOR_SHA256

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def save(clf: Pipeline, meta: ModelMeta, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, directory / "classifier.joblib")
    (directory / "meta.json").write_text(meta.to_json())
    return directory


def load(directory: Path) -> tuple[Pipeline, ModelMeta]:
    clf = joblib.load(directory / "classifier.joblib")  # noqa: S301 (our own artifact)
    meta = ModelMeta(**json.loads((directory / "meta.json").read_text()))
    return clf, meta


def predict_with_review(clf: Pipeline, X: np.ndarray,
                        min_conf: float = config.MIN_CONFIDENCE) -> tuple[np.ndarray, np.ndarray]:
    """Class per row, plus a mask of low-confidence rows whose crops go to the review queue.

    Low-confidence vehicles are still counted with the model's best guess: it is right about half
    the time on them, and the alternative (the detector's COCO class) is worse. Their value is as
    the next labels to collect (docs/ARCHITECTURE.md, "The labelling loop").
    """
    proba = clf.predict_proba(X)
    return clf.classes_[proba.argmax(1)].astype(object), proba.max(1) < min_conf
