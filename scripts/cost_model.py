"""Monthly AWS cost for N junctions, from list prices. Every number in docs/COST.md comes from here.

Prices are on-demand list prices from the AWS pricing pages (checked October 2026; US East figures
where the page only gives those; ap-south-1 is within a few percent). Re-check before quoting.
"""

from __future__ import annotations

import argparse

APPROACHES_PER_JUNCTION = 4
INTERVALS_PER_MONTH = 4 * 24 * 30
MINUTES_PER_MONTH = 60 * 24 * 30

PRICE = {
    "iot_connect_per_million_min": 0.08,
    "iot_message_per_million": 1.00,      # 0 when publishing through Basic Ingest ($aws/rules/...)
    "iot_rule_per_million": 0.15,
    "iot_action_per_million": 0.15,
    "lambda_request_per_million": 0.20,
    "lambda_arm_gb_second": 0.0000133334,
    "ddb_write_per_million": 1.25,
    "ddb_read_per_million": 0.25,
    "s3_put_per_1000": 0.005,
    "s3_gb_month": 0.025,
    "cw_custom_metric": 0.30,
    "cw_alarm": 0.10,
    "cw_logs_ingest_gb": 0.50,
    # Alternatives
    "kvs_ingest_gb": 0.0085,
    "kvs_store_gb_month": 0.023,
    "rekognition_stored_video_per_min": 0.10,
    "sagemaker_ml_t3_medium_hour": 0.05,
}


def chowk(junctions: int, basic_ingest: bool = True) -> dict[str, float]:
    records = junctions * APPROACHES_PER_JUNCTION * INTERVALS_PER_MONTH
    reviews = junctions * 5 * INTERVALS_PER_MONTH          # up to 5 crops per interval per box
    watchdog_runs = INTERVALS_PER_MONTH
    c = {
        "IoT Core connectivity": junctions * MINUTES_PER_MONTH / 1e6 * PRICE["iot_connect_per_million_min"],
        "IoT Core messages": 0.0 if basic_ingest else records / 1e6 * PRICE["iot_message_per_million"],
        "IoT rules + actions": records / 1e6 * (PRICE["iot_rule_per_million"] + PRICE["iot_action_per_million"]),
        "Lambda (ingest, watchdog, API, retrain)": (
            (records + watchdog_runs + 50_000) / 1e6 * PRICE["lambda_request_per_million"]
            + (records * 0.1 * 0.25 + watchdog_runs * 1.0 * 0.25 + 50_000 * 0.05 * 0.25 + 4 * 60 * 2)
            * PRICE["lambda_arm_gb_second"]),
        "DynamoDB on-demand": (records / 1e6 * PRICE["ddb_write_per_million"]
                               + (watchdog_runs * junctions * APPROACHES_PER_JUNCTION + 50_000 * 2) / 1e6
                               * PRICE["ddb_read_per_million"]),
        "S3 (archive PUTs, review crops)": ((records + reviews) / 1000 * PRICE["s3_put_per_1000"]
                                            + (records * 0.5e-6 + reviews * 10e-6) * 3 * PRICE["s3_gb_month"]),
        "CloudWatch (5 metrics, 5 alarms, logs)": (5 * PRICE["cw_custom_metric"] + 5 * PRICE["cw_alarm"]
                                                   + records * 1.5e-6 * PRICE["cw_logs_ingest_gb"]),
    }
    return {k: round(v, 2) for k, v in c.items()}


def stream_to_cloud(junctions: int, mbps: float = 1.0) -> dict[str, float]:
    """The design I did not build: send video to AWS and run the model there."""
    gb = mbps / 8 * MINUTES_PER_MONTH * 60 / 1000 * junctions
    return {
        "video uploaded per month (GB)": round(gb),
        "Kinesis Video Streams ingest + 1 day hot storage": round(gb * PRICE["kvs_ingest_gb"]
                                                                  + gb / 30 * PRICE["kvs_store_gb_month"], 2),
        "Rekognition stored-video labels, 24/7": round(junctions * MINUTES_PER_MONTH
                                                       * PRICE["rekognition_stored_video_per_min"], 2),
        "SageMaker real-time endpoint (1 x ml.t3.medium, idle or not)": round(
            24 * 30 * PRICE["sagemaker_ml_t3_medium_hour"], 2),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--junctions", type=int, nargs="+", default=[1, 20, 200])
    args = p.parse_args()
    for n in args.junctions:
        c = chowk(n)
        print(f"\n{n} junction(s), {n * APPROACHES_PER_JUNCTION} approaches")
        for k, v in c.items():
            print(f"  {k:45s} ${v:8.2f}")
        print(f"  {'TOTAL (Basic Ingest)':45s} ${sum(c.values()):8.2f}")
        print(f"  {'TOTAL (normal MQTT topic)':45s} ${sum(chowk(n, basic_ingest=False).values()):8.2f}")
        for k, v in stream_to_cloud(n).items():
            print(f"  alt: {k:58s} {v:>10}")


if __name__ == "__main__":
    main()
