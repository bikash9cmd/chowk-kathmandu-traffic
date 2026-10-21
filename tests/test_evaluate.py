import numpy as np
import pandas as pd

from chowk import config
from chowk import evaluate as ev


def test_pcu_error_is_relative_per_session():
    df = pd.DataFrame({"session": [0] * 20 + [1] * 20, "label": ["light"] * 40})
    pred = np.array(["light"] * 20 + ["heavy"] * 20, dtype=object)
    e = ev.pcu_error(df, pred)
    assert e.sessions == 2
    assert e.worst_session_error == 2.0  # 20 x 3.0 vs 20 x 1.0
    assert e.median_session_error == 1.0
    assert e.total_bias == 1.0


def test_missed_vehicles_count_zero_pcu():
    assert ev.pcu_of(np.array([None, "two_wheeler", "heavy"], dtype=object)).tolist() == [0.0, 0.5, 3.0]


def test_grouped_cv_cannot_learn_from_the_session_itself(synthetic):
    """If every session has one label and the features only identify the session, a random split
    looks perfect (it has seen the session) while a grouped split must fail. This is the leak the
    session grouping exists to stop."""
    df, _ = synthetic
    df = df.copy()
    df["label"] = [config.CLASSES[s % 4] for s in df["session"]]
    X = np.eye(df["session"].nunique())[df["session"]].astype(np.float32)
    y = ev.labels(df)
    grouped, _ = ev.grouped_cv_predict(X, y, df["session"].to_numpy(), folds=4)
    random = ev.random_cv_predict(X, y)
    assert (random == y).mean() > 0.95
    assert (grouped == y).mean() < 0.5


def test_classifier_separates_synthetic_classes(synthetic):
    df, X = synthetic
    pred, proba = ev.grouped_cv_predict(X, ev.labels(df), df["session"].to_numpy(), folds=4)
    assert (pred == ev.labels(df)).mean() > 0.95
    assert np.allclose(proba.sum(1), 1)


def test_perturbations_keep_size():
    from PIL import Image

    img = Image.new("RGB", (90, 140), (120, 120, 120))
    for fn in ev.PERTURBATIONS.values():
        assert fn(img).size == img.size
