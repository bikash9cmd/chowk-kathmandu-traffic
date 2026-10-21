from fastapi.testclient import TestClient

from chowk import model
from chowk.api import create_app
from chowk.counting import aggregate
from chowk.pipeline import replay_sessions, train_with_gate
from chowk.store import LocalStore


def test_gate_promotes_first_model_and_rejects_a_worse_one(synthetic):
    df, X = synthetic
    clf, meta, gate = train_with_gate(df, X, None, "v1")
    assert gate.passed and clf is not None and meta.cv_macro_f1 > 0.9
    # A champion that scored perfectly: a noisy candidate must not replace it.
    perfect = model.ModelMeta("v0", len(df), list(meta.classes), 1.0, 0.0)
    noisy = X + __import__("numpy").random.default_rng(1).normal(0, 6, X.shape).astype(X.dtype)
    clf2, _, gate2 = train_with_gate(df, noisy, perfect, "v2")
    assert not gate2.passed and clf2 is None


def test_model_round_trip(tmp_path, synthetic):
    df, X = synthetic
    clf, meta, _ = train_with_gate(df, X, None, "v1")
    model.save(clf, meta, tmp_path)
    clf2, meta2 = model.load(tmp_path)
    assert meta2 == meta and (clf2.predict(X) == clf.predict(X)).all()


def test_replay_and_store_are_idempotent(tmp_path, synthetic):
    df, X = synthetic
    clf, meta, _ = train_with_gate(df, X, None, "v1")
    store = LocalStore(tmp_path / "f.jsonl")
    rows = replay_sessions(df, X, clf, "v1", store)
    assert len(rows) == 12 and (rows["pcu_predicted"] - rows["pcu_true"]).abs().max() < 3
    again = LocalStore(tmp_path / "f.jsonl")
    replay_sessions(df, X, clf, "v1", again)
    assert len(again.query("replay", "recorded_clips", "", "9999")) == 12


def test_api_now_and_flows(tmp_path):
    from datetime import datetime

    store = LocalStore(tmp_path / "f.jsonl")
    store.put(aggregate("maitighar", "from_thapathali", datetime(2026, 10, 1, 8, 0),
                        ["two_wheeler"] * 300 + ["light"] * 80 + ["heavy"] * 10, "v1").as_dict())
    c = TestClient(create_app(store))
    now = c.get("/junctions/maitighar/now").json()
    first = {a["approach"]: a for a in now["approaches"]}
    assert first["from_thapathali"]["pcu_per_hour"] == 1040.0
    assert first["from_thapathali"]["level"] == "near_capacity"
    assert first["from_babarmahal"]["status"] == "no_data"
    flows = c.get("/junctions/maitighar/from_thapathali/flows", params={"since": "2026-10-01T07:00", "hours": 2})
    assert len(flows.json()["flows"]) == 1
    assert c.get("/junctions/nowhere/now").status_code == 404
    assert c.get("/junctions/maitighar/from_thapathali/flows", params={"hours": 0}).status_code == 400


def test_api_marks_degraded_images(tmp_path):
    from datetime import datetime

    store = LocalStore(tmp_path / "f.jsonl")
    store.put(aggregate("maitighar", "from_thapathali", datetime(2026, 10, 1, 18, 0), ["light"] * 50, "v1",
                        image_quality={"dark": 0.9, "blurry": 0.2}).as_dict())
    now = TestClient(create_app(store)).get("/junctions/maitighar/now").json()
    a = {x["approach"]: x for x in now["approaches"]}["from_thapathali"]
    assert a["status"] == "degraded_image" and a["image_problems"] == ["dark"]
