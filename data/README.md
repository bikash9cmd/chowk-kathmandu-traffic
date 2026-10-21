# Data

## Vehicle crops (the training data)

**Source:** [sdevkota007/vehicles-nepal-dataset](https://github.com/sdevkota007/vehicles-nepal-dataset), `VehiclesNepal.zip`, pinned to commit `719dd82` and verified by SHA-256 (`chowk/data.py`). Downloaded by `python -m chowk download` into `data/raw/` (not committed).

The dataset's author and project mates cut 4,797 vehicles out of 30 traffic videos, each about 4 minutes, filmed on streets in Kathmandu in 2016 for their undergraduate project *Vehicle Detection and Road Traffic Congestion Mapping Using Image Processing*. It has two folders: `2wheelers` (1,823) and `4wheelers` (2,974). All images are greyscale. The repository has no license file; this project downloads it from its own URL rather than redistributing it, and credits the authors (GitHub user sdevkota007 and project mates Anup Adhikari, Binish Koirala and Sparsha Bhattarai).

What I found in it, and what the code does about each (details in [docs/CHALLENGES_AND_BUGS.md](../docs/CHALLENGES_AND_BUGS.md)):

| Finding | Handling |
|---|---|
| No video IDs, but the zip keeps each file's save time | Files split into 31 "sessions" at every pause over 5 minutes; sessions are the cross-validation groups |
| 12 files saved twice under consecutive numbers (identical bytes) | Dropped (`exact_duplicate_of`) |
| 2 files stored as 8-bit greyscale, the rest as RGB (all greyscale in content) | Everything converted to 3-channel; camera frames made greyscale too |
| 35 crops under 64 px on the short side | Kept; reported separately in the model card |
| Only two classes, but PCU needs four | Four-wheelers hand-labelled into light / medium / heavy (below) |

## Labels: `labels/vehicle_classes.csv`

One row per image: `image` (path inside the zip) and `label`, one of `two_wheeler`, `light`, `medium`, `heavy`.

Two-wheelers come straight from the source folder. The 2,974 four-wheelers were labelled by me, by eye, in October 2026:

1. Cluster the four-wheelers into 30 groups on the detector's features, so similar vehicles sit together.
2. Render each group as numbered contact sheets of 60 crops (64 sheets).
3. For each sheet, set the majority label and list the exceptions by number.
4. Audit: train the classifier, look at 60 random disagreements between model and label, and fix the label where I was wrong (7 fixed).

**Rules** (from Nepal Road Standard 2070, Table 4-1, plus two of mine for vehicles it doesn't name):

| Label | PCU | Includes |
|---|---:|---|
| `two_wheeler` | 0.5 | motorcycles, scooters, bicycles |
| `light` | 1.0 | cars, SUVs and jeeps (Bolero, Scorpio, Sumo), pick-ups, light vans (Maruti Omni), taxis, **Safa tempos** (as auto rickshaws) |
| `medium` | 1.5 | mini trucks, **12-15 seat microbuses** (Hiace, Winger, Traveller) - my assumption, see ADR-003 |
| `heavy` | 3.0 | buses, minibuses, trucks, tankers |

**Known weaknesses:** one annotator, no second opinion; pick-up vs mini truck and Omni van vs microbus are judgement calls on a front view; I estimate 1-2% of four-wheeler labels are still wrong (7 of 60 audited disagreements were my errors; disagreements are 8% of four-wheelers).

## Junctions: `junctions.json`

Three real Kathmandu junctions with two approaches each, used by the API's congestion level. **The lane counts, green ratios and saturation flows are illustrative, not surveyed.**

## Not used, and why

- **OpenStreetMap GPS traces** for speeds: too sparse in Kathmandu for per-road, per-hour estimates.
- **Nepal Police crash statistics:** monthly totals by district, not records with locations.
- **Department of Roads traffic counts (AADT):** annual averages for highways, published in reports, not per-vehicle data.
