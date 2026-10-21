# Cost

All figures from [`scripts/cost_model.py`](../scripts/cost_model.py) (`make cost`), using on-demand list prices checked in October 2026 (US East where the pricing page only shows that; ap-south-1 is within a few percent). Re-check before quoting.

## AWS, per month

| | 1 junction | **20 junctions** | 200 junctions |
|---|---:|---:|---:|
| IoT Core connectivity | $0.00 | $0.07 | $0.69 |
| IoT Core messages (Basic Ingest) | $0.00 | $0.00 | $0.00 |
| IoT rules + actions | $0.00 | $0.07 | $0.69 |
| Lambda (ingest, watchdog, API, retrain) | $0.04 | $0.16 | $1.26 |
| DynamoDB on-demand | $0.04 | $0.37 | $3.48 |
| S3 (archive PUTs, review crops) | $0.14 | $2.82 | $28.17 |
| CloudWatch (5 custom metrics, 5 alarms, logs) | $2.01 | $2.17 | $3.73 |
| **Total** | **$2.23** | **$5.66** | **$38.02** |

Four approaches per junction, one record per approach every 15 minutes, up to five review crops per box per interval.

**What it says:** at small scale the bill is monitoring (fixed CloudWatch charges); at large scale it is S3 PUT requests for the per-record archive. The ML costs almost nothing in AWS, because it doesn't run there.

## The edge, per camera box

Not in the AWS bill, and larger than it:

| Item | Rough cost |
|---|---|
| Raspberry Pi 5 (8 GB), case, power supply, SD card | ~$110 one-off (more after import duty in Nepal) |
| Camera (USB or CSI, weatherproof housing) | ~$40-80 one-off |
| Mobile data, ~0.6 GB/month (counts + review crops) | the smallest data pack an operator sells |
| Power | ~5-8 W continuous; a small UPS for load-shedding |

Mounting, permission from KVTPO or the municipality, and maintenance visits will cost more than the hardware.

## Compared with the designs I didn't build

| Design | Data out of each camera per month | AWS cost per camera per month |
|---|---:|---:|
| Stream video to Rekognition Video (label detection on stored video, $0.10/min) | ~324 GB | ~$4,320 |
| Stream video to Kinesis Video Streams + own model on a SageMaker endpoint | ~324 GB | ~$3 ingest + $36 endpoint (shared) |
| **Chowk: inference on the box, counts to IoT Core** | **~0.6 GB** | **~$0.28** (at 20 junctions) |

Rekognition's cheaper "streaming video events" ($0.0082/min) only detect people, pets and packages, so it can't do this job at any price. And 324 GB a month per camera over Nepali 4G is the real blocker, before any AWS price.

## Cost decisions in the code

1. **Inference on the box** (ADR-004). Removes the video bandwidth, the endpoint and the per-minute vision API in one decision.
2. **A linear head on the detector's own features** (ADR-002). No second network per vehicle, so a $80 board keeps up; no GPU anywhere.
3. **Basic Ingest** (`$aws/rules/chowk_flows/...`). Messages go straight to the rule: no IoT messaging charge.
4. **One small record per approach per 15 minutes**, not per vehicle. About 300 bytes; IoT and DynamoDB both bill in units far larger than that.
5. **Retrain on Lambda**, weekly, seconds long (ADR-005). No SageMaker training job or notebook left running.
6. **Graviton (arm64) Lambdas**, about 20% cheaper per GB-second.
7. **DynamoDB on-demand with a TTL** of ~13 months. No idle capacity; old rows delete themselves for free.
8. **No IA/Glacier transition on the archive.** Records are ~400 bytes, far below the 128 KB minimum billable object size for IA, so a transition would cost more than it saves.
9. **Review crops expire after 90 days**, which is also a privacy control.
10. **Log retention** of 7-30 days per function; CloudWatch keeps logs forever by default.
11. **HTTP API with throttling** (10 req/s), so a scraper can't run up the bill; responses carry `cache-control: max-age=60`.
12. **A budget alarm** at 80% of actual and 100% of forecast monthly spend.

## What I'd change at 200 junctions

The per-record S3 archive becomes three quarters of the bill. Sending records through Kinesis Data Firehose to S3 (one object every few minutes, Parquet) would cut ~$28 to under $1 and make the archive queryable with Athena. Below about 50 junctions it isn't worth the extra moving part.
