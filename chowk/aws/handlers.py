"""AWS Lambda entry points. Same core code as the local pipeline; only I/O differs.

  ingest_handler    IoT Core rule on topic chowk/flows/+: validate a camera's 15-minute record,
                    archive it to S3, upsert into DynamoDB
  watchdog_handler  every 15 minutes (EventBridge Scheduler): which cameras have gone quiet?
  retrain_handler   weekly: features + labels from S3 -> train_with_gate -> new champion if it passes
  api_handler       API Gateway HTTP API: GET /junctions/{id}/now, /junctions/{id}/{approach}/flows, /health

Environment variables are set by infra/template.yaml.
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import boto3

from chowk import config
from chowk.store import DynamoStore

log = logging.getLogger()
log.setLevel(logging.INFO)

NPT = timedelta(hours=5, minutes=45)  # Nepal time, UTC+05:45
# A camera that has sent nothing for this long is reported as stale. Two missed intervals, not one:
# a record can legitimately arrive a few minutes late after a 4G drop.
STALE_AFTER = timedelta(minutes=2 * config.FLOW_INTERVAL_MINUTES + 5)
# A single approach above this is a bug or a tampered message, not traffic: the busiest Ring Road
# approaches carry a few thousand PCU an hour.
MAX_PLAUSIBLE_PCU_PER_HOUR = 10_000


def _aws(service: str, kind: str = "client"):
    return getattr(boto3, kind)(service)


def _metric(name: str, value: float, unit: str = "Count", **props: str) -> None:
    """CloudWatch Embedded Metric Format: a structured log line becomes a metric, no API call.

    No dimensions on purpose: every distinct dimension value is a separately billed custom metric,
    and an alarm only matches the exact dimension set it names. Context goes in `props`.
    """
    print(json.dumps({
        "_aws": {"Timestamp": int(time.time() * 1000), "CloudWatchMetrics": [{
            "Namespace": "Chowk", "Dimensions": [[]], "Metrics": [{"Name": name, "Unit": unit}]}]},
        name: value, **props}))


def _now_npt() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None) + NPT


# ---------- ingest ----------

REQUIRED = {"junction": str, "approach": str, "interval_start": str, "counts": dict,
            "pcu": (int, float), "pcu_per_hour": (int, float), "minutes_observed": (int, float),
            "model_version": str}


def validate(record: dict) -> list[str]:
    """Reasons to reject a record. Empty list = valid. Recomputes PCU rather than trusting the camera."""
    problems = [f"missing or wrong type: {k}" for k, t in REQUIRED.items() if not isinstance(record.get(k), t)]
    if problems:
        return problems
    counts = record["counts"]
    if set(counts) - set(config.CLASSES):
        problems.append(f"unknown classes {sorted(set(counts) - set(config.CLASSES))}")
    if any(not isinstance(v, int) or v < 0 for v in counts.values()):
        problems.append("counts must be non-negative integers")
    # The IoT rule adds `camera` (the thing name from the topic, which the certificate pins). A box
    # named maitighar-01 may only report for maitighar, so a stolen or misconfigured box can't
    # write another junction's counts.
    camera = record.get("camera")
    if camera is not None and str(camera).rsplit("-", 1)[0] != record["junction"]:
        problems.append(f"camera {camera} may not report for junction {record['junction']}")
    if not 0 < record["minutes_observed"] <= config.FLOW_INTERVAL_MINUTES:
        problems.append("minutes_observed out of range")
    try:
        start = datetime.fromisoformat(record["interval_start"])
        if start.minute % config.FLOW_INTERVAL_MINUTES or start.second:
            problems.append("interval_start not on a 15-minute boundary")
        if start > _now_npt() + timedelta(minutes=5):
            problems.append("interval_start in the future (camera clock wrong?)")
    except ValueError:
        problems.append("interval_start is not ISO 8601")
    if not problems:
        pcu = sum(config.PCU[c] * n for c, n in counts.items())
        if abs(pcu - record["pcu"]) > 0.51:
            problems.append(f"pcu {record['pcu']} does not match counts ({pcu})")
        if record["pcu_per_hour"] > MAX_PLAUSIBLE_PCU_PER_HOUR:
            problems.append("pcu_per_hour implausibly high")
        iq = record.get("image_quality", {})
        if not isinstance(iq, dict) or any(not isinstance(v, int | float) or not 0 <= v <= 1 for v in iq.values()):
            problems.append("image_quality must map flags to shares between 0 and 1")
    return problems


def ingest_handler(event: dict, _ctx=None) -> dict:
    """The IoT rule passes the MQTT payload as the event. One record per message."""
    problems = validate(event)
    if problems:
        _metric("RecordsRejected", 1, reasons="; ".join(problems), junction=str(event.get("junction")))
        return {"accepted": False, "problems": problems}
    s3 = _aws("s3")
    # Archive first: S3 is the source of truth; DynamoDB is a query cache that can be rebuilt from it.
    day = event["interval_start"][:10]
    key = (f"flows/junction={event['junction']}/date={day}/"
           f"{event['approach']}_{event['interval_start'][11:16].replace(':', '')}.json")
    s3.put_object(Bucket=os.environ["DATA_BUCKET"], Key=key, Body=json.dumps(event).encode())
    new = DynamoStore().put(event)
    _metric("RecordsIngested", 1)
    if not new:
        _metric("RecordsDuplicate", 1, junction=event["junction"])
    return {"accepted": True, "new": new}


# ---------- watchdog ----------

def watchdog_handler(event: dict, _ctx=None) -> dict:
    """Count cameras whose latest record is older than STALE_AFTER. Alarmed on in CloudWatch."""
    junctions = json.loads(_aws("s3").get_object(Bucket=os.environ["DATA_BUCKET"],
                                                 Key="config/junctions.json")["Body"].read())["junctions"]
    now = datetime.fromisoformat(event["now"]) if "now" in event else _now_npt()
    store = DynamoStore()
    stale = []
    for jid, j in junctions.items():
        for aid in j["approaches"]:
            rec = store.latest(jid, aid)
            last = datetime.fromisoformat(rec["interval_start"]) if rec else None
            # interval_start is the start; the record is due at start + 15 minutes.
            if last is None or now - (last + timedelta(minutes=config.FLOW_INTERVAL_MINUTES)) > STALE_AFTER:
                stale.append(f"{jid}#{aid}")
    _metric("StaleApproaches", len(stale), approaches=",".join(stale))
    return {"stale": stale}


# ---------- retrain ----------

def retrain_handler(event: dict, _ctx=None) -> dict:
    """Weekly. Inputs in S3: training/features.npy, training/labelled.parquet (current labels), and
    optionally training/features_degraded.npz (degraded copies, added to training only)."""
    import numpy as np
    import pandas as pd

    from chowk import model
    from chowk.pipeline import train_with_gate

    s3, ssm = _aws("s3"), _aws("ssm")
    bucket = os.environ["DATA_BUCKET"]
    X = np.load(io.BytesIO(s3.get_object(Bucket=bucket, Key="training/features.npy")["Body"].read()))
    df = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=bucket, Key="training/labelled.parquet")["Body"].read()))
    extra = []
    try:
        body = s3.get_object(Bucket=bucket, Key="training/features_degraded.npz")["Body"].read()
        extra = list(np.load(io.BytesIO(body)).values())
    except s3.exceptions.NoSuchKey:
        log.info("no degraded features; training on clean crops only")
    champion_meta = None
    if not event.get("bootstrap"):
        version = ssm.get_parameter(Name=os.environ["CHAMPION_PARAM"])["Parameter"]["Value"]
        meta = json.loads(s3.get_object(Bucket=bucket, Key=f"models/{version}/meta.json")["Body"].read())
        champion_meta = model.ModelMeta(**meta)
    clf, meta, gate = train_with_gate(df, X, champion_meta, extra=extra)
    _metric("RetrainGatePassed", int(gate.passed))
    if clf is None:
        log.warning(json.dumps({"event": "gate_failed", **gate.as_dict()}))
        return {"promoted": False, "gate": gate.as_dict()}
    d = Path(tempfile.mkdtemp())
    model.save(clf, meta, d)
    for name in ("classifier.joblib", "meta.json"):
        s3.upload_file(str(d / name), bucket, f"models/{meta.version}/{name}")
    ssm.put_parameter(Name=os.environ["CHAMPION_PARAM"], Value=meta.version, Type="String", Overwrite=True)
    # Retained message: a camera that reconnects later still learns about the newest model.
    _aws("iot-data").publish(topic="chowk/model/champion", qos=1, retain=True,
                             payload=json.dumps({"version": meta.version,
                                                 "key": f"models/{meta.version}/classifier.joblib"}))
    return {"promoted": True, "version": meta.version, "gate": gate.as_dict()}


# ---------- api ----------

def _response(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"content-type": "application/json",
                                              "cache-control": "max-age=60"},
            "body": json.dumps(body, ensure_ascii=False)}


def api_handler(event: dict, _ctx=None) -> dict:
    """HTTP API payload v2. Same logic as chowk/api.py, without FastAPI (smaller package, faster cold start)."""
    from chowk.status import approach_status

    path = event.get("rawPath", "/")
    params = event.get("pathParameters") or {}
    qs = event.get("queryStringParameters") or {}
    if path == "/health":
        return _response(200, {"ok": True})
    junctions = json.loads(_aws("s3").get_object(Bucket=os.environ["DATA_BUCKET"],
                                                 Key="config/junctions.json")["Body"].read())["junctions"]
    jid = params.get("junction")
    if jid not in junctions:
        return _response(404, {"error": f"unknown junction {jid}"})
    store = DynamoStore()
    if path.endswith("/now"):
        return _response(200, {"junction": jid, "approaches": [
            approach_status(store, jid, a, g) for a, g in junctions[jid]["approaches"].items()]})
    aid = params.get("approach")
    if aid not in junctions[jid]["approaches"]:
        return _response(404, {"error": f"unknown approach {aid}"})
    try:
        hours = int(qs.get("hours", 24))
        since = qs.get("since") or (_now_npt() - timedelta(hours=hours)).isoformat(timespec="minutes")
        until = (datetime.fromisoformat(since) + timedelta(hours=hours)).isoformat()
    except ValueError:
        return _response(400, {"error": "since must be ISO 8601 and hours an integer"})
    if not 1 <= hours <= 24 * 31:
        return _response(400, {"error": "hours must be between 1 and 744"})
    return _response(200, {"junction": jid, "approach": aid, "flows": store.query(jid, aid, since, until)})
