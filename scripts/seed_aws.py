"""Seed a freshly deployed stack: junction config, training data, and the local champion model.

    python scripts/seed_aws.py --bucket <DataBucketName>

After this, the weekly retrain has something to train on, the API knows the junctions, and
cameras can download the model the SSM parameter points at.
"""

from __future__ import annotations

import argparse
import json

import boto3

from chowk import config, model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--param", default="/chowk/champion-model-version")
    args = p.parse_args()
    s3, ssm = boto3.client("s3"), boto3.client("ssm")
    s3.upload_file(str(config.DATA_DIR / "junctions.json"), args.bucket, "config/junctions.json")
    s3.upload_file(str(config.ARTIFACT_DIR / "features.npy"), args.bucket, "training/features.npy")
    s3.upload_file(str(config.ARTIFACT_DIR / "labelled.parquet"), args.bucket, "training/labelled.parquet")
    degraded = config.ARTIFACT_DIR / "features_degraded.npz"
    if degraded.exists():
        s3.upload_file(str(degraded), args.bucket, "training/features_degraded.npz")
    champion = config.ARTIFACT_DIR / "models" / "champion"
    _, meta = model.load(champion)
    for name in ("classifier.joblib", "meta.json"):
        s3.upload_file(str(champion / name), args.bucket, f"models/{meta.version}/{name}")
    ssm.put_parameter(Name=args.param, Value=meta.version, Type="String", Overwrite=True)
    print(json.dumps({"bucket": args.bucket, "champion": meta.version}))


if __name__ == "__main__":
    main()
