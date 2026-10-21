"""Where 15-minute flow records live. Locally a JSON-lines file; on AWS a DynamoDB table.

Both implement the same three methods, so the API and the pipeline never know which one they use.
"""

from __future__ import annotations

import json
import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Protocol

from chowk import config


class FlowStore(Protocol):
    def put(self, record: dict) -> bool: ...
    def query(self, junction: str, approach: str, since: str, until: str) -> list[dict]: ...
    def latest(self, junction: str, approach: str) -> dict | None: ...


def key(record: dict) -> tuple[str, str]:
    return f"{record['junction']}#{record['approach']}", record["interval_start"]


class LocalStore:
    def __init__(self, path: Path = config.ARTIFACT_DIR / "flows.jsonl") -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: dict[tuple[str, str], dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                self._rows[key(r)] = r

    def put(self, record: dict) -> bool:
        """Idempotent: a camera re-sending an interval after a reconnect overwrites, never double-counts."""
        k = key(record)
        new = k not in self._rows
        self._rows[k] = record
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in self._rows.values()))
        tmp.replace(self.path)
        return new

    def query(self, junction: str, approach: str, since: str, until: str) -> list[dict]:
        pk = f"{junction}#{approach}"
        return sorted((r for (p, s), r in self._rows.items() if p == pk and since <= s < until),
                      key=lambda r: r["interval_start"])

    def latest(self, junction: str, approach: str) -> dict | None:
        rows = self.query(junction, approach, "", "9999")
        return rows[-1] if rows else None


def _to_dynamo(obj):
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _to_dynamo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_dynamo(v) for v in obj]
    return obj


def _from_dynamo(obj):
    if isinstance(obj, Decimal):
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, dict):
        return {k: _from_dynamo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_dynamo(v) for v in obj]
    return obj


class DynamoStore:
    """pk = "junction#approach", sk = interval_start. Items expire after ~13 months (TTL), enough
    for a year-on-year comparison; the S3 archive keeps everything."""

    TTL_SECONDS = 400 * 24 * 3600

    def __init__(self, table_name: str | None = None, resource=None) -> None:
        import boto3

        resource = resource or boto3.resource("dynamodb")
        self.table = resource.Table(table_name or os.environ["FLOWS_TABLE"])

    def put(self, record: dict) -> bool:
        pk, sk = key(record)
        item = _to_dynamo(record) | {"pk": pk, "sk": sk, "expires_at": int(time.time()) + self.TTL_SECONDS}
        resp = self.table.put_item(Item=item, ReturnValues="ALL_OLD")
        return "Attributes" not in resp

    def query(self, junction: str, approach: str, since: str, until: str) -> list[dict]:
        from boto3.dynamodb.conditions import Key

        cond = Key("pk").eq(f"{junction}#{approach}") & Key("sk").between(since, until)
        items, kwargs = [], {"KeyConditionExpression": cond}
        while True:
            resp = self.table.query(**kwargs)
            items += resp["Items"]
            if "LastEvaluatedKey" not in resp:
                break
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        out = []
        for i in items:
            r = _from_dynamo(i)
            if r["sk"] >= until:  # between() is inclusive at both ends; the local store is not
                continue
            for k in ("pk", "sk", "expires_at"):
                r.pop(k, None)
            out.append(r)
        return out

    def latest(self, junction: str, approach: str) -> dict | None:
        from boto3.dynamodb.conditions import Key

        resp = self.table.query(KeyConditionExpression=Key("pk").eq(f"{junction}#{approach}"),
                                ScanIndexForward=False, Limit=1)
        if not resp.get("Items"):
            return None
        r = _from_dynamo(resp["Items"][0])
        for k in ("pk", "sk", "expires_at"):
            r.pop(k, None)
        return r
