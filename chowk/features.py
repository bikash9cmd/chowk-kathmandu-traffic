"""Feature vectors for every labelled crop, computed once and cached.

For a crop the vehicle's box is the whole image (the person who cut it out acted as the
detector), so the features are the detector's feature maps pooled over the image area.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from chowk import config, data
from chowk.detector import Detector


def labelled_manifest(rebuild: bool = False) -> pd.DataFrame:
    """Manifest joined to the hand labels, with exact duplicate files removed."""
    df = data.clean(data.load_manifest(rebuild=rebuild))
    labels = pd.read_csv(config.LABELS_CSV).rename(columns={"image": "path"})
    df = df.merge(labels, on="path", how="left", validate="one_to_one")
    missing = df["label"].isna().sum()
    if missing:
        raise ValueError(f"{missing} images have no label in {config.LABELS_CSV}")
    bad = set(df["label"]) - set(config.CLASSES)
    if bad:
        raise ValueError(f"unknown labels {bad}")
    # A two-wheeler folder image can never be a four-wheeler class and vice versa.
    wrong = ((df["source_class"] == "two_wheeler") != (df["label"] == "two_wheeler")).sum()
    if wrong:
        raise ValueError(f"{wrong} labels contradict the source folder")
    return df.reset_index(drop=True)


def compute(df: pd.DataFrame, detector: Detector | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Features plus the zero-shot COCO scores for each crop, in df order."""
    det = detector or Detector(threads=4)
    feats, scores = [], []
    with data.open_archive() as archive:
        for path in df["path"]:
            f, s = det.crop_features(data.load_image(archive, path))
            feats.append(f)
            scores.append(s)
    return np.stack(feats), pd.DataFrame(scores, index=df.index)


def load(rebuild: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    """(manifest with labels and COCO scores, feature matrix), cached under artifacts/."""
    fpath = config.ARTIFACT_DIR / "features.npy"
    mpath = config.ARTIFACT_DIR / "labelled.parquet"
    if fpath.exists() and mpath.exists() and not rebuild:
        df = pd.read_parquet(mpath)
    else:
        df = labelled_manifest()
        X, scores = compute(df)
        df = pd.concat([df, scores.add_prefix("coco_")], axis=1)
        np.save(fpath, X)
        df.assign(dhash=df["dhash"].astype("uint64")).to_parquet(mpath, index=False)
    # Labels are re-read every time: a label fix must never be hidden by a stale cache (bug #10).
    current = pd.read_csv(config.LABELS_CSV).rename(columns={"image": "path", "label": "current"})
    df = df.merge(current, on="path", how="left", validate="one_to_one")
    df["label"] = df.pop("current")
    return df, np.load(fpath)
