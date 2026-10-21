# Alternatives considered

Each record: the decision, what else I looked at, and why I chose what I did. "Revisit when" is the condition that would change my mind.

## ADR-001: Which traffic problem

**Decision:** count the vehicle mix at junctions in Passenger Car Units (PCU), from cameras.

**Options:**
1. **Congestion / travel-time forecasting.** The classic ML traffic project. It needs speed or travel-time data, and Kathmandu has no open feed: Uber Movement never covered it, Google and Waze data are not open, and OpenStreetMap GPS traces are too sparse to estimate speeds by road and hour.
2. **Crash hotspot prediction.** Nepal Police publish monthly crash totals by district, not geolocated crash records, so there is nothing to train a spatial model on.
3. **Signal timing optimisation.** Needs a simulator calibrated to real counts, which is exactly the data that doesn't exist.
4. **Vehicle-mix counting (chosen).** Every one of the above needs *counts by vehicle type* first, and today those come from people with clipboards. Nepal's road standard plans capacity in PCU, where a motorcycle is 0.5 and a bus is 3.0, so a count is only useful if the vehicle types are right. There is a public dataset of real Kathmandu vehicles to build it on.

**Revisit when:** a year of counts exists. Then forecasting the next hour's PCU per approach (the Hawa pattern) becomes possible, and that is the natural second project.

## ADR-002: Fine-tune a detector, or classify on its features

**Decision:** keep the COCO-trained detector for *finding* vehicles, discard its class labels, and train a logistic regression on two of its internal feature maps pooled over each box.

| Option | Training data needed | Runs on a Pi-class box | Effort | Notes |
|---|---|---|---|---|
| Use the COCO classes as they are | none | yes | none | Calls 65% of Kathmandu cars a truck or bus (README, Results) |
| **Linear head on detector features (chosen)** | class labels only | yes, ~1k extra multiply-adds | low | One forward pass does both jobs |
| Separate CNN classifier on each crop (e.g. MobileNet) | class labels | slower: a second network per vehicle | medium | Could be more accurate on van vs microbus |
| Fine-tune the detector with 4 Kathmandu classes | **bounding boxes on full frames** | yes | high | The right end state, but needs labelled frames I don't have |
| Amazon Rekognition Custom Labels | labels | no (cloud only) | low | Pay per inference-hour; video has to reach AWS (ADR-004) |

The dataset is crops, not frames, so there are no boxes to fine-tune a detector on. A linear head trains in under a second, is easy to inspect, and its mistakes can be read off a confusion matrix. The detector's own features turned out to be enough: macro-F1 0.88 on held-out sessions.

**Revisit when:** someone labels a few hundred full junction frames. Then fine-tuning YOLOX-Nano with the four classes is the upgrade, and the classifier head becomes the baseline it must beat.

## ADR-003: Four PCU classes, and where the microbus goes

**Decision:** two-wheeler 0.5, light 1.0, medium 1.5, heavy 3.0, following Nepal Road Standard 2070, Table 4-1. The three-wheeled electric Safa tempo is an auto rickshaw (light). The 12-15 seat microbus goes in **medium** (1.5).

The standard has no microbus row. By the letter of it, a Hiace is a "light van" (1.0). But a microbus is longer, and it stops anywhere to load passengers, which costs a junction far more than a private van does. A minibus is 3.0, and a microbus is clearly smaller. 1.5 is my judgement, not the standard. The model card reports the headline with the microbus at 1.0, 1.5 and 3.0: Chowk stays within 1.4-3.4% either way; the off-the-shelf detector is 34-51% off either way.

**Revisit when:** a Kathmandu PCU study measures the microbus (the Pulchowk Campus PCU studies measure buses, trucks, LCVs and two-wheelers, not microbuses).

## ADR-004: Run the model on the camera box, not in the cloud

**Decision:** inference at the edge; only 15-minute counts (about 300 bytes per approach) go to AWS.

| Option | Data sent per camera per month | AWS cost per camera per month | Works through a 4G outage |
|---|---|---|---|
| Stream video, Rekognition Video labels | ~324 GB (1 Mbps) | ~$4,300 (stored-video labels at $0.10/min) | no |
| Stream video (Kinesis Video Streams), own model on SageMaker | ~324 GB | ~$3 ingest + ~$36 endpoint | no |
| Upload sampled frames to S3, Lambda inference | ~5-30 GB | a few dollars | partly |
| **Edge inference, send counts (chosen)** | ~0.6 GB, mostly review crops | ~$0.28 at 20 junctions | **yes: counts queue on disk** |

Bandwidth decides it before AWS prices do: 324 GB a month on a Nepali 4G data plan is not a realistic cost for a junction camera, and fibre to every pole doesn't exist. It is also better for privacy: video never leaves the box; only counts and a few low-confidence vehicle crops do. Rekognition's streaming video events (the cheap per-minute option) only detect people, pets and packages, not vehicle types.

**Revisit when:** junctions get fibre and a city-wide video platform exists for other reasons; then cloud inference on existing streams avoids buying boxes.

## ADR-005: Retrain on Lambda, not SageMaker

**Decision:** a weekly Lambda runs five-fold session-grouped CV on the logistic regression, applies the promotion gate, and publishes the model.

Training takes seconds on 5,000 rows. A SageMaker training job would add minutes of instance start-up for a few seconds of work, plus a second set of IAM and image plumbing. Inference doesn't happen in AWS at all (ADR-004), so there is no endpoint to host.

**Revisit when:** the training set reaches the hundreds of thousands, or the head becomes a CNN that wants a GPU. Then SageMaker Training with spot instances (about 70% cheaper) is the move.

## ADR-006: IoT Core, not API Gateway, for camera traffic

**Decision:** cameras publish MQTT to AWS IoT Core through Basic Ingest; an IoT rule invokes the ingest Lambda.

API Gateway with API keys would work for a demo. For a fleet of boxes on poles, IoT Core gives a certificate per device (revoke one stolen box, not everyone's key), a policy that lets each box publish only to its own topic, an MQTT connection that tolerates flaky 4G, retained messages for model updates, and the credentials provider so boxes never hold long-lived AWS keys. Basic Ingest skips the broker and the per-message charge.

**Revisit when:** there are only one or two cameras and nobody will manage certificates; then an HTTPS endpoint is simpler.

## ADR-007: DynamoDB for flows, S3 as the archive

**Decision:** DynamoDB (pk `junction#approach`, sk `interval_start`) for the API, S3 for the raw records.

The access pattern is "the latest interval" and "a day of intervals for one approach", both single-partition queries. The key makes ingest idempotent: a re-sent record overwrites itself. Timestream was the other candidate, and suits time series well, but costs more at this size and adds nothing the two query shapes need. S3 is the source of truth; the table can be rebuilt from it.

**Revisit when:** analysts want city-wide aggregations across all junctions. Then Athena over the S3 archive (after compacting it into daily Parquet, docs/COST.md) is the answer, not a bigger table.

## ADR-008: Infrastructure as code

**Decision:** AWS SAM. Serverless functions, schedules and the HTTP API are one line each; it's CloudFormation underneath, so IoT resources sit in the same template. CDK would be the choice for a team that prefers TypeScript; Terraform for multi-cloud. Neither buys anything here.
