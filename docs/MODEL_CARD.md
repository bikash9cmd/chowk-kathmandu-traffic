# Model card: Chowk vehicle classifier

## What it is

A logistic regression (256 inputs, 4 classes, L2, C = 0.1, no class weights) on features taken from inside a COCO-trained YOLOX-Nano detector: the mean of two feature maps under a vehicle's box. Output: one of `two_wheeler`, `light`, `medium`, `heavy`, mapped to 0.5 / 1.0 / 1.5 / 3.0 PCU (Nepal Road Standard 2070, Table 4-1, with the microbus rule in ADR-003).

## Intended use

Counting vehicles by type at Kathmandu junctions, aggregated into 15-minute PCU flows per approach, for traffic engineers planning signal timing and capacity, and for before/after comparisons of interventions. The unit of use is a 15-minute total, not an individual vehicle.

**Not for:** enforcement or anything about an individual vehicle or person; places unlike the training data (highways, other cities) without re-validation; night-time counts (no night data at all).

## Training data

4,785 crops (4,797 minus 12 exact duplicates) cut by hand from 30 daytime street videos in Kathmandu, 2016, all greyscale (data/README.md). Two-wheeler labels come from the source; the four-wheeler classes were labelled by one person. Class mix: 1,820 two-wheelers, 1,969 light, 378 medium, 618 heavy.

## Evaluation

Five-fold cross-validation grouped by recording session (31 sessions recovered from file save times), so no session is ever in both train and test. PCU error is computed per session (26 sessions with at least 20 vehicles).

| | Chowk | Off-the-shelf YOLOX (COCO classes) | Two classes only (4W = car) |
|---|---:|---:|---:|
| Accuracy | 93.9% | 63.2% | 79.2% |
| Macro-F1 | 0.884 | 0.467 | 0.449 |
| Median session PCU error | **1.3%** | 45.7% | 25.5% |
| Worst session PCU error | 9.0% | 116.6% | 46.8% |
| Overall PCU bias | -0.7% | +54.7% | -26.8% |

Per class F1: two-wheeler 0.999, light 0.940, medium 0.696, heavy 0.903.

**The model that ships is the augmented one.** The table above is the model trained on clean crops only, which is the fair comparison with the off-the-shelf detector. The champion in `artifacts/models/champion` (version 2026-10-21) is also trained on four degraded copies of every crop (dusk, distance, dusty lens, low bitrate). That costs a little on clean images, **1.9% median session PCU error instead of 1.3%, macro-F1 0.862 instead of 0.884**, and buys 2-5% error instead of 7-106% when the image gets worse (docs/FAILURE_MODES.md). On a street camera that sees dusk every evening and dust every dry season, that is the right trade.

**Sensitivity to my microbus assumption** (median session PCU error if "medium" were weighted differently):

| medium = | 1.0 | **1.5** | 3.0 |
|---|---:|---:|---:|
| Chowk | 1.4% | 1.3% | 3.4% |
| Off-the-shelf | 50.7% | 45.7% | 34.1% |

The conclusion doesn't depend on the assumption.

**Confidence and the review queue:** at a 0.55 probability threshold, 2.5% of vehicles are queued for review; the model is right on 52% of those and 95.0% of the rest.

## Limitations, and who the errors affect

- **Crops, not frames.** Every test vehicle was found and cut out by a person. Detection recall on real junction frames, occlusion in dense traffic, and double counting by the tracker are **not measured**. These are the first things to measure on a deployment (one hour of video, counted by hand).
- **Medium is the weak class** (F1 0.70). Boxy vans and jeeps get called microbuses, and microbuses get called vans. Microbuses are Kathmandu's main public transport, so an approach dominated by them is the case where counts are least reliable.
- **Errors partly cancel** in totals (light-to-medium and medium-to-light confusions go in opposite directions). An approach with an unusual mix (say, all microbuses) won't get that cancellation.
- **2016 vehicles.** No electric microbuses or the newer Chinese EV cars that are common in Kathmandu now. Expect drift; the review queue is how it would show up.
- **Daytime, dry, greyscale.** Robustness to dusk, distance, dust and compression is measured with simulated degradations in docs/FAILURE_MODES.md, not on real footage.
- **One annotator.** I estimate 1-2% of the four-wheeler labels are still wrong (docs/CHALLENGES_AND_BUGS.md #11).
- **Small crops:** 35 crops are under 64 px on their short side; distant vehicles are the hardest.

## Privacy

Video never leaves the camera box. Review crops (vehicles the model is unsure about) can include faces and number plates; they are uploaded only under the box's own S3 prefix, kept for at most 90 days, and used only for labelling.
