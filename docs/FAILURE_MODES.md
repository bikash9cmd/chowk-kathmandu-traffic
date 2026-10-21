# Failure modes

What breaks, how I'd know, and what the system does about it. The first section is measured; the rest is designed and tested with unit tests and mocked AWS (moto), not on a live deployment.

## 1. The image gets worse (measured)

A junction camera in Kathmandu doesn't see the clean daytime crops in the training set for long. The light goes at 5:30 pm in winter, dust settles on the lens within days in the dry season, and a camera on a weak 4G link will be streamed at a low bitrate. I simulated four degradations on every crop (`chowk/evaluate.py`, `PERTURBATIONS`):

| Degradation | What it simulates | How |
|---|---|---|
| dusk | Evening light, sensor noise | Brightness × 0.35, contrast × 0.5, Gaussian noise (σ = 8) |
| far away | A far lane, or a camera mounted high | Downscale to a quarter of the width and height, then back up |
| dusty lens | Dust or spray on the housing | Gaussian blur (radius 3) blended 35% with a grey veil |
| low bitrate | A cheap camera on a bad link | JPEG quality 10 |

Then the same session-grouped cross-validation as the main evaluation, three ways: the classifier trained on clean crops only; trained with degraded copies of the *other three* degradations (to see whether augmentation generalises to something it hasn't seen); and trained with all four.

![Robustness chart](img/robustness.png)

Median session PCU error on held-out sessions (`artifacts/reports/robustness.json`):

| Degradation | Trained on clean | With the other 3 | With all 4 |
|---|---:|---:|---:|
| dusk | 19.9% | 60.0% | **4.3%** |
| far away | 7.4% | 6.4% | **1.9%** |
| dusty lens | 31.8% | 18.9% | **4.0%** |
| low bitrate | 106.4% | 57.3% | **2.5%** |
| *clean images* | *1.3%* | | *1.9%* |

What this says:

1. **A model trained on clean crops falls apart on bad images.** At low bitrate it more than doubles the PCU (106%) by calling 1,731 of 1,969 cars and 701 of 1,820 two-wheelers heavy. At dusk it calls 1,377 of 1,969 cars two-wheelers.
2. **Augmentation fixes what it has seen, not what it hasn't.** Training on the other three made dusk *worse* (60%). So augmentation isn't a general robustness switch, and a fifth kind of bad image (rain on the lens, headlight glare at night) should be assumed broken until it is measured.
3. **It isn't free.** On clean images the augmented model is at 1.9% instead of 1.3% (macro-F1 0.862 instead of 0.884). I ship the augmented one anyway, because a junction camera spends a lot more time in bad light than the clean evaluation suggests.

### Detecting it: the input checks

Since the model can't be trusted on images it hasn't seen, the camera box measures every tenth frame with four cheap checks (`chowk/quality.py`): mean brightness (dark), standard deviation (low contrast), Laplacian variance (blurry) and 8×8 block-edge strength (compressed). Each 15-minute record carries the share of checked frames that tripped each flag. The API marks an approach `degraded_image` when any flag is above 50%, so a user sees "counts may be wrong" instead of a quiet wrong number.

Share of crops flagged by at least one check:

| Input | Flagged | Main flag |
|---|---:|---|
| clean | 4.0% | (false alarms) |
| dusk | 100% | dark |
| far away | 79.4% | blurry |
| dusty lens | 98.2% | blurry |
| low bitrate | 99.6% | compressed |

The thresholds were set on the crops, so on full junction frames they'd need re-tuning, which is part of the first-deployment hour (see the end of this doc).

## 2. Everything else

| # | Failure | How I'd know | What happens |
|---|---|---|---|
| 1 | **Camera or box crashes** for part of an interval | Record's `minutes_observed` < 15 | Rate is scaled from the minutes actually observed (gaps over 5 s don't count). The API marks the approach `partial` below 7.5 minutes rather than showing half the traffic as a quiet road. |
| 2 | **Box goes dark** (power cut, load-shedding, theft) | Watchdog Lambda every 15 min; `StaleCameraAlarm` after 35 minutes with no record, twice in a row | SNS email. The API shows the last record with its age; it never fills the gap with a guess. |
| 3 | **4G link drops** for hours | Records arrive late, all at once; `RecordsDuplicate` rises on reconnect | Records are written to an on-disk outbox before sending and deleted only after the broker acks. Ingest is idempotent on (junction, approach, interval_start), so re-sends are harmless. A 1-day outage is under 100 KB of disk. |
| 4 | **Clock drift** on the box (no RTC battery, NTP blocked) | `RecordsRejected` with "future" or "15-minute boundary" | Rejected, not stored. A record with a wrong time is worse than no record: it puts rush hour in the middle of the night. |
| 5 | **Bad firmware or a tampered record** (counts and PCU don't add up, impossible rates) | `RecordsRejected` with the reason in the log | Ingest recomputes PCU from the counts instead of trusting the camera, and rejects over 10,000 PCU/h per approach. |
| 6 | **Stolen box or copied certificate** | `RecordsRejected` "may not report for junction" | Each box has its own certificate; the IoT policy only lets it connect as itself and publish on its own topic; the rule stamps the thing name onto the record and ingest checks it against the junction. A stolen box can only lie about its own junction, and revoking one certificate cuts it off. |
| 7 | **Night** | Input check: `dark` near 100% on every evening record | No fix. There is no night data at all. The API flags it; the honest answer is to label night footage before claiming night counts. |
| 8 | **New vehicle types** (electric microbuses, Chinese EV cars that weren't on the road in 2016) | Review-queue rate rises (2.5% of vehicles today); the crops in it show what's new | Low-confidence crops (max probability < 0.55) are saved, up to 5 per interval, uploaded under the box's own S3 prefix, and labelled. They're still counted with the model's best guess, because dropping them would undercount. |
| 9 | **A retrain makes the model worse** | `RetrainGateAlarm` | The weekly retrain only promotes a model if session-grouped CV macro-F1 is within 0.01 and PCU error within 10% of the champion's. Otherwise the cameras keep the old model. |
| 10 | **A bad model gets promoted anyway** | Next week's PCU totals jump at one moment across all junctions | Every record carries `model_version`, so a step change is attributable. Rollback is writing the old version into the SSM parameter and republishing the retained `chowk/model/champion` message; every box swaps at its next frame. |
| 11 | **Label errors** (one annotator, me) | Model/label disagreements, reviewed by hand | I audited 60 disagreements and fixed 7 labels (CHALLENGES #11). Estimated 1-2% still wrong. A second annotator on a sample is the fix I'd want before trusting medium-class numbers. |
| 12 | **Ingest Lambda failing or throttled** | `IngestErrorsAlarm` | The IoT rule's error action writes the raw record to `s3://…/errors/`, so nothing is lost; replay them through the handler once it's fixed. |
| 13 | **DynamoDB lost or corrupted** | API returns nothing | S3 is the source of truth: every accepted record is archived there first. DynamoDB is a query cache that can be rebuilt from the archive, which is why point-in-time recovery is off. |
| 14 | **The watchdog itself stops** (scheduler, IAM change) | `StaleCameraAlarm` uses `TreatMissingData: breaching`, so no metric at all is an alarm; dropped scheduled runs land in a DLQ with its own alarm | Same SNS email as a dead camera. |
| 15 | **Cost runaway** (a buggy box publishing in a loop, a public API being scraped) | AWS Budget emails at 80% of actual and 100% of forecast | HTTP API throttled at 10 req/s (burst 20). One record per approach per 15 minutes is the normal rate; anything above that is a bug and shows up in `RecordsDuplicate` long before the bill. |

## 3. What isn't covered (and what I'd do first)

- **Detection on full frames.** Everything above the classifier is unmeasured on real junction video: detector recall in dense traffic, occlusion (a tempo hidden behind a bus), and the tracker double-counting a vehicle that stops on the line. **First step on a real deployment:** record one hour at one junction, count it by hand per class, and compare. That hour also tunes the count-line position and the image-check thresholds.
- **Rain.** Not simulated. Water drops on the lens are a different failure from dust (sharp local blur, reflections), and given the result in section 1, I'd assume it's broken until measured.
- **Box hardware.** Detector latency is 24 ms per 1280×720 frame on one core of a 2.1 GHz Xeon. A Raspberry Pi 5 is not measured; I'd estimate 3-5× slower, which is still enough for the 2-5 fps the tracker needs. Heat in a metal box on a pole in May is the risk I'd worry about more.
