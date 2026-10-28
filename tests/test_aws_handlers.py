"""Lambda handlers against mocked AWS (moto). Same resources as infra/template.yaml."""

import io
import json
from datetime import datetime, timedelta

import boto3
import numpy as np
import pytest
from moto import mock_aws

from chowk import config
from chowk.aws import handlers
from chowk.counting import aggregate

BUCKET, TABLE, PARAM = "chowk-test", "chowk-flows", "/chowk/champion"


@pytest.fixture
def aws(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("DATA_BUCKET", BUCKET)
    monkeypatch.setenv("FLOWS_TABLE", TABLE)
    monkeypatch.setenv("CHAMPION_PARAM", PARAM)
    with mock_aws():
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "ap-south-1"})
        s3.put_object(Bucket=BUCKET, Key="config/junctions.json",
                      Body=(config.DATA_DIR / "junctions.json").read_bytes())
        boto3.client("dynamodb").create_table(
            TableName=TABLE, BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"},
                                  {"AttributeName": "sk", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}])
        yield s3


def record(minute=0, **over):
    start = (datetime.now() + handlers.NPT - timedelta(hours=6)).replace(minute=minute, second=0, microsecond=0)
    rec = aggregate("maitighar", "from_thapathali", start, ["two_wheeler"] * 40 + ["light"] * 10 + ["heavy"], "v1")
    return rec.as_dict() | over


def test_valid_record_is_archived_and_stored_once(aws):
    r = record()
    assert handlers.ingest_handler(r) == {"accepted": True, "new": True}
    assert handlers.ingest_handler(r) == {"accepted": True, "new": False}  # re-send after reconnect
    keys = [o["Key"] for o in aws.list_objects_v2(Bucket=BUCKET, Prefix="flows/")["Contents"]]
    assert len(keys) == 1 and keys[0].startswith("flows/junction=maitighar/")


@pytest.mark.parametrize(("over", "reason"), [
    ({"pcu": 999.0}, "does not match"),
    ({"counts": {"tempo": 3}}, "unknown classes"),
    ({"interval_start": "2026-10-01T08:07:00"}, "15-minute boundary"),
    ({"interval_start": "2999-01-01T08:00:00"}, "future"),
    ({"interval_start": "2026-10-01T08:00:00+05:45"}, "without a UTC offset"),
    ({"interval_start": "2026-10-01T02:15:00Z"}, "without a UTC offset"),
    ({"minutes_observed": 0}, "out of range"),
    ({"junction": None}, "missing"),
    ({"image_quality": {"dark": 3}}, "image_quality"),
    ({"camera": "kalanki-01"}, "may not report"),
])
def test_bad_records_are_rejected(aws, over, reason):
    out = handlers.ingest_handler(record(**over))
    assert not out["accepted"] and any(reason in p for p in out["problems"])


def test_rejected_record_is_not_stored(aws):
    out = handlers.ingest_handler(record(interval_start="2026-10-01T08:00:00+05:45"))
    assert not out["accepted"]
    assert "Contents" not in aws.list_objects_v2(Bucket=BUCKET, Prefix="flows/")
    assert boto3.client("dynamodb").scan(TableName=TABLE)["Count"] == 0


def test_camera_may_report_for_its_own_junction(aws):
    assert handlers.ingest_handler(record(camera="maitighar-01"))["accepted"]


def test_watchdog_reports_quiet_cameras(aws):
    r = record()
    handlers.ingest_handler(r)
    now = datetime.fromisoformat(r["interval_start"]) + timedelta(minutes=20)
    stale = handlers.watchdog_handler({"now": now.isoformat()})["stale"]
    assert "maitighar#from_thapathali" not in stale and "maitighar#from_babarmahal" in stale
    later = handlers.watchdog_handler({"now": (now + timedelta(hours=1)).isoformat()})["stale"]
    assert "maitighar#from_thapathali" in later


def test_api_now_and_flows(aws):
    r = record()
    handlers.ingest_handler(r)
    now = handlers.api_handler({"rawPath": "/junctions/maitighar/now", "pathParameters": {"junction": "maitighar"}})
    body = json.loads(now["body"])
    assert now["statusCode"] == 200
    assert body["approaches"][0]["counts"]["two_wheeler"] == 40
    since = (datetime.fromisoformat(r["interval_start"]) - timedelta(hours=1)).isoformat()
    flows = handlers.api_handler({"rawPath": "/junctions/maitighar/from_thapathali/flows",
                                  "pathParameters": {"junction": "maitighar", "approach": "from_thapathali"},
                                  "queryStringParameters": {"since": since, "hours": "2"}})
    assert len(json.loads(flows["body"])["flows"]) == 1
    missing = handlers.api_handler({"rawPath": "/junctions/x/now", "pathParameters": {"junction": "x"}})
    assert missing["statusCode"] == 404
    bad = handlers.api_handler({"rawPath": "/junctions/maitighar/from_thapathali/flows",
                                "pathParameters": {"junction": "maitighar", "approach": "from_thapathali"},
                                "queryStringParameters": {"hours": "many"}})
    assert bad["statusCode"] == 400


def test_retrain_bootstraps_then_gate_blocks_a_worse_model(aws, synthetic):
    df, X = synthetic

    def upload(X):
        buf = io.BytesIO()
        np.save(buf, X)
        aws.put_object(Bucket=BUCKET, Key="training/features.npy", Body=buf.getvalue())
        aws.put_object(Bucket=BUCKET, Key="training/labelled.parquet", Body=df.to_parquet())

    upload(X)
    first = handlers.retrain_handler({"bootstrap": True})
    assert first["promoted"]
    assert boto3.client("ssm").get_parameter(Name=PARAM)["Parameter"]["Value"] == first["version"]
    upload(X + np.random.default_rng(2).normal(0, 6, X.shape).astype(X.dtype))
    second = handlers.retrain_handler({})
    assert not second["promoted"]
    assert boto3.client("ssm").get_parameter(Name=PARAM)["Parameter"]["Value"] == first["version"]
