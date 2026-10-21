"""Training with a promotion gate, and the replay that turns recorded clips into 15-minute flows.

Independent of where it runs (laptop, Docker, Lambda).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from chowk import config, model
from chowk.counting import aggregate
from chowk.evaluate import labels, pcu_error
from chowk.store import FlowStore

log = logging.getLogger("chowk")


@dataclass
class GateResult:
    passed: bool
    candidate_macro_f1: float
    candidate_pcu_error: float
    champion_macro_f1: float | None
    champion_pcu_error: float | None
    reason: str

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _cv_scores(df: pd.DataFrame, X: np.ndarray, extra: list[np.ndarray]) -> tuple[float, float]:
    """Session-grouped CV scored on clean crops; `extra` (degraded copies, row-aligned with X) are added
    to each training fold only."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold

    y = labels(df)
    pred = np.empty(len(y), dtype=object)
    for train, test in GroupKFold(config.CV_FOLDS).split(X, y, df["session"].to_numpy()):
        Xa = np.concatenate([X[train]] + [E[train] for E in extra])
        clf = model.make_classifier().fit(Xa, np.concatenate([y[train]] * (1 + len(extra))))
        pred[test] = clf.predict(X[test])
    return float(f1_score(y, pred, average="macro")), pcu_error(df, pred).median_session_error


def train_with_gate(df: pd.DataFrame, X: np.ndarray, champion_meta: model.ModelMeta | None = None,
                    version: str | None = None, extra: list[np.ndarray] | None = None):
    """Score a candidate by session-grouped CV and only return it if it is not worse than the champion.

    The champion's score is the one recorded when it was promoted, on the data it had then. New
    labels usually make the problem harder, not easier (they come from the crops the old model was
    unsure about), so the gate allows a small tolerance rather than demanding strict improvement.
    """
    f1, pcu = _cv_scores(df, X, extra or [])
    if champion_meta is None:
        passed, reason = True, "no champion yet"
    elif f1 < champion_meta.cv_macro_f1 - config.GATE_MAX_F1_DROP:
        passed, reason = False, f"macro-F1 {f1:.3f} < champion {champion_meta.cv_macro_f1:.3f} - tolerance"
    elif pcu > champion_meta.cv_pcu_error * (1 + config.GATE_MAX_PCU_ERROR_RISE) + 0.005:
        passed, reason = False, f"PCU error {pcu:.3f} > champion {champion_meta.cv_pcu_error:.3f} + tolerance"
    else:
        passed, reason = True, "within tolerance of champion"
    gate = GateResult(passed, round(f1, 4), round(pcu, 4),
                      champion_meta.cv_macro_f1 if champion_meta else None,
                      champion_meta.cv_pcu_error if champion_meta else None, reason)
    if not passed:
        return None, None, gate
    y = labels(df)
    clf = model.fit(np.concatenate([X] + list(extra or [])), np.concatenate([y] * (1 + len(extra or []))))
    meta = model.ModelMeta(version or datetime.now().strftime("%Y%m%dT%H%M%S"), len(df),
                           [str(c) for c in clf.classes_], round(f1, 4), round(pcu, 4))
    return clf, meta, gate


def replay_sessions(df: pd.DataFrame, X: np.ndarray, clf, version: str, store: FlowStore,
                    junction: str = "replay", start: datetime | None = None) -> pd.DataFrame:
    """Each recorded session becomes one 15-minute interval at a demo junction.

    The crops were cut from short clips, so this is not real junction traffic, but it runs the same
    code path as a camera would: classify, aggregate into a FlowRecord, store idempotently.
    The true PCU from the hand labels is kept alongside for comparison.
    """
    start = start or datetime(2016, 6, 1, 8, 0)
    pred = clf.predict(X)
    rows = []
    for i, (session, g) in enumerate(df.groupby("session", sort=True)):
        at = start + timedelta(minutes=config.FLOW_INTERVAL_MINUTES * i)
        rec = aggregate(junction, "recorded_clips", at, list(pred[g.index]), version)
        store.put(rec.as_dict())
        truth = aggregate(junction, "recorded_clips", at, list(labels(g)), "hand_labels")
        rows.append({"session": int(session), "interval_start": rec.interval_start, "vehicles": len(g),
                     "pcu_predicted": rec.pcu, "pcu_true": truth.pcu,
                     **{f"n_{c}": rec.counts[c] for c in config.CLASSES}})
        log.info(json.dumps({"event": "flow", **rec.as_dict()}))
    return pd.DataFrame(rows)
