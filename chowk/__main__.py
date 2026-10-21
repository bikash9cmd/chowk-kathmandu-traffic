"""Command line: python -m chowk <command>."""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np

from chowk import config

REPORTS = config.ARTIFACT_DIR / "reports"
IMG = config.ROOT / "docs" / "img"


def cmd_download(_):
    from chowk import data, detector

    print("data:", data.download())
    print("detector:", detector.model_path())


def cmd_quality(_):
    from chowk import data

    report = data.quality_report(data.load_manifest(rebuild=True))
    print("\n".join(report.lines()))
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "data_quality.json").write_text(json.dumps(report.__dict__, indent=2))


def cmd_evaluate(args):
    from chowk import data, features, report
    from chowk import evaluate as ev

    df, X = features.load(rebuild=args.rebuild)
    y, groups = ev.labels(df), df["session"].to_numpy()
    pred, proba = ev.grouped_cv_predict(X, y, groups)
    balanced, _ = ev.grouped_cv_predict(X, y, groups, balanced=True)
    zero_shot = ev.zero_shot_predict(df)
    zero_shot_thr = ev.zero_shot_predict(df, config.DETECT_SCORE_MIN)
    naive = np.where(df["source_class"] == "two_wheeler", "two_wheeler", "light").astype(object)
    out = {
        "images": len(df), "sessions": int(df["session"].nunique()),
        "class_counts": df["label"].value_counts().to_dict(),
        "chowk_grouped_cv": ev.scores(df, pred),
        "chowk_random_cv": ev.scores(df, ev.random_cv_predict(X, y)),
        "chowk_class_balanced": ev.scores(df, balanced),
        "zero_shot_best_class": ev.scores(df, zero_shot),
        "zero_shot_at_0.25": ev.scores(df, zero_shot_thr),
        "two_classes_only": ev.scores(df, naive),
        "c_search": {str(c): ev.scores(df, ev.grouped_cv_predict(X, y, groups, c)[0])["macro_f1"]
                     for c in (0.01, 0.03, 0.1, 0.3, 1.0)},
        "pcu_sensitivity": ev.pcu_sensitivity(df, {"chowk": pred, "zero_shot": zero_shot, "two_classes": naive}),
        "abstain": {},
    }
    conf = proba.max(1)
    for t in (0.45, 0.55, 0.65):
        low = conf < t
        out["abstain"][str(t)] = {"rate": round(float(low.mean()), 4),
                                  "accuracy_kept": round(float((pred[~low] == y[~low]).mean()), 4),
                                  "accuracy_abstained": round(float((pred[low] == y[low]).mean()), 4)
                                  if low.any() else None}
    ev.write(out, REPORTS / "evaluation.json")
    np.save(REPORTS / "cv_predictions.npy", pred.astype(str))
    # Charts
    from chowk.pipeline import replay_sessions
    from chowk.store import LocalStore

    class _Fixed:  # out-of-fold predictions dressed as a classifier, so the replay shows held-out numbers
        def predict(self, _X):
            return pred

    rows = replay_sessions(df, X, _Fixed(), "cv", LocalStore(REPORTS / "replay_flows.jsonl")).to_dict("records")
    zs_pcu = {int(s): float(ev.pcu_of(zero_shot[g.index]).sum()) for s, g in df.groupby("session")}
    IMG.mkdir(parents=True, exist_ok=True)
    report.session_pcu_chart(rows, zs_pcu, IMG / "session_pcu.png")
    report.confusion_chart({"Off-the-shelf YOLOX (COCO classes)": out["zero_shot_best_class"]["confusion"],
                            "Chowk (held-out sessions)": out["chowk_grouped_cv"]["confusion"]},
                           IMG / "confusion.png")
    with data.open_archive() as archive:
        report.example_sheet(archive, df, pred, IMG / "examples.png")
    print(json.dumps({k: out[k]["pcu"] | {"macro_f1": out[k]["macro_f1"]} for k in
                      ("chowk_grouped_cv", "chowk_random_cv", "zero_shot_best_class", "two_classes_only")}, indent=2))


def cmd_robustness(_):
    from chowk import data, features, report
    from chowk import evaluate as ev
    from chowk.detector import Detector

    df, X = features.load()
    cache = config.ARTIFACT_DIR / "features_degraded.npz"
    if cache.exists():
        perturbed = dict(np.load(cache))
    else:
        with data.open_archive() as archive:
            perturbed = ev.perturbed_features(df, Detector(threads=4), archive)
        np.savez(cache, **perturbed)
    res = ev.robustness(df, X, perturbed)
    with data.open_archive() as archive:
        res["input_checks"] = ev.input_check_rates(df, archive)
    ev.write(res, REPORTS / "robustness.json")
    clean = json.loads((REPORTS / "evaluation.json").read_text())["chowk_grouped_cv"]["pcu"]
    report.robustness_chart({k: v for k, v in res.items() if k in ev.PERTURBATIONS}, clean["median_session_error"],
                            IMG / "robustness.png")
    for k, v in res.items():
        if k in ev.PERTURBATIONS or k == "clean":
            print(k, {m: (s["pcu"]["median_session_error"], s["macro_f1"]) for m, s in v.items()})
    print("input checks", res["input_checks"])


def cmd_train(args):
    from chowk import features, model
    from chowk.pipeline import train_with_gate

    df, X = features.load()
    champion = None
    champ_dir = config.ARTIFACT_DIR / "models" / "champion"
    if (champ_dir / "meta.json").exists() and not args.bootstrap:
        _, champion = model.load(champ_dir)
    # Degraded copies of every crop (made by `robustness`) are added to the training set. They cost
    # about 0.6 points of PCU error on clean images and fix compressed / dusty / dark frames
    # (docs/FAILURE_MODES.md); --no-augment trains on clean crops only.
    cache = config.ARTIFACT_DIR / "features_degraded.npz"
    extra = list(np.load(cache).values()) if cache.exists() and not args.no_augment else []
    clf, meta, gate = train_with_gate(df, X, champion, args.version, extra)
    print(json.dumps(gate.as_dict(), indent=2))
    if clf is not None:
        model.save(clf, meta, config.ARTIFACT_DIR / "models" / meta.version)
        model.save(clf, meta, champ_dir)
        print("promoted", meta.version)


def cmd_replay(_):
    from chowk import features, model
    from chowk.pipeline import replay_sessions
    from chowk.store import LocalStore

    df, X = features.load()
    clf, meta = model.load(config.ARTIFACT_DIR / "models" / "champion")
    rows = replay_sessions(df, X, clf, meta.version, LocalStore())
    print(rows.to_string(index=False))
    print("\nNote: the champion was trained on these same crops, so these are in-sample numbers;"
          " see `evaluate` for held-out ones.")


def cmd_serve(args):
    import uvicorn

    from chowk.api import create_app

    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(prog="chowk")
    sub = p.add_subparsers(required=True)
    sub.add_parser("download").set_defaults(fn=cmd_download)
    sub.add_parser("quality").set_defaults(fn=cmd_quality)
    e = sub.add_parser("evaluate")
    e.add_argument("--rebuild", action="store_true", help="recompute detector features (about a minute)")
    e.set_defaults(fn=cmd_evaluate)
    sub.add_parser("robustness").set_defaults(fn=cmd_robustness)
    t = sub.add_parser("train")
    t.add_argument("--bootstrap", action="store_true", help="ignore any existing champion")
    t.add_argument("--version", default=None)
    t.add_argument("--no-augment", action="store_true", help="train on clean crops only")
    t.set_defaults(fn=cmd_train)
    sub.add_parser("replay").set_defaults(fn=cmd_replay)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
