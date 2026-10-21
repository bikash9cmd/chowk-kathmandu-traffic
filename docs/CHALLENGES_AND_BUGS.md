# Challenges and bugs

A log of what went wrong while building this, roughly in the order I hit it, with the evidence and the fix. Every number can be reproduced with `make evaluate` and `make robustness`.

## Finding a problem with real data

### 1. Kathmandu has almost no open traffic data

**Symptom:** my first plan was congestion forecasting. There is nothing to forecast from: no open speed or travel-time feed for Kathmandu, OpenStreetMap GPS traces are too sparse for per-road, per-hour speeds, and Nepal Police crash data is published as monthly district totals, not located records.

**What I did:** worked backwards from what does exist. A public dataset of 4,797 real vehicles cut from 30 Kathmandu street videos does, and every traffic question (signal timing, capacity, forecasting) starts with classified counts, which today are done by hand. So the project became the counting step, done properly. ADR-001 has the full comparison.

**Lesson:** pick the problem the data can answer, then make that answer production-grade, rather than picking an impressive problem and faking the data.

### 2. The dataset has two classes; the road standard needs four

**Symptom:** the source folders are "2wheelers" and "4wheelers". Nepal Road Standard 2070 weights a car 1.0 PCU and a bus or truck 3.0, so "four-wheeler" is useless for capacity: a stream of buses and a stream of cars look identical.

**Fix:** I labelled the 2,974 four-wheelers myself into light, medium and heavy. To make that feasible, I clustered them on the detector's features (similar vehicles together), rendered 64 numbered contact sheets, and wrote one default label per sheet plus the exceptions. Two vehicles the standard doesn't name needed rules (data/README.md): the Safa tempo is an auto rickshaw (1.0), and the microbus goes in the 1.5 bucket (ADR-003, my assumption, with a sensitivity check in the model card).

## The off-the-shelf model

### 3. A COCO detector calls two in three Kathmandu cars a truck or bus

**Symptom:** YOLOX-Nano, trained on COCO, finds vehicles but labels them badly here. Of 1,969 light vehicles (cars, jeeps, vans, tempos), it labels 1,287 (65%) as "truck" or "bus". Every one of those counts as 3.0 PCU instead of 1.0, so its 15-minute PCU totals are 55% too high overall and up to 117% too high in one session.

**Cause (my reading, not proven):** COCO's "car" is mostly side and rear views of sedans at street level. These crops are greyscale, filmed from above, mostly front views, and full of boxy Suzuki vans, Mahindra jeeps and roof racks. There is also no COCO class for a microbus, mini truck or tempo at all, so "medium" can never be predicted.

**Fix:** keep the detector's boxes and features, drop its labels, and train a Kathmandu classifier on its features (ADR-002).

### 4. At its default threshold the detector looks only 11% wrong, because two big errors cancel

**Symptom:** with the usual 0.25 confidence threshold, the detector's median session PCU error is 10.6%, which looks almost acceptable.

**Cause:** it misses 30% of two-wheelers and 43% of light vehicles entirely (score below 0.25 on a crop that contains nothing but the vehicle), which pushes the total down; and it triples the PCU of most cars it does find, which pushes the total up. In some sessions they roughly cancel. The worst session is still 82% off, and per-vehicle accuracy is 43%.

**Lesson:** an aggregate metric can hide two large errors in opposite directions. Report the per-class confusion next to every total.

## Evaluation

### 5. The dataset has no video IDs, so a random split leaks

**Symptom:** a vehicle appears in several consecutive frames, and the crops were cut frame by frame. A random train/test split puts near-copies of a test vehicle in the training set.

**Fix:** the zip file still holds each file's modification time, which is when it was saved during cropping. Saves come in bursts with long pauses between them. Splitting at every pause over 5 minutes gives 31 sessions, against the 30 videos the author describes, and both class folders fall into the same sessions. Cross-validation is grouped by session (`GroupKFold`), so a session is either all train or all test. `test_grouped_cv_cannot_learn_from_the_session_itself` checks that this split really blocks a session-identity leak.

**How much it mattered:** less than I expected. A random split gives macro-F1 0.896 against 0.884 grouped, and its worst session is 5.6% off against 9.0%. The leak flatters the worst case most, which is the number a traffic engineer would ask about.

### 6. Twelve files were saved twice

24 files are 12 byte-identical pairs under consecutive numbers (for example `4wheelers/346.jpg` and `347.jpg`): the same crop saved twice. Both copies would have counted as two vehicles. Dropped by MD5 (`exact_duplicate_of`).

### 7. Class weighting made the model worse

**Symptom:** medium vehicles are only 8% of the data, so I started with `class_weight="balanced"`, expecting the model would otherwise ignore them.

**Measured:** balanced weighting scored worse on everything: macro-F1 0.870 vs 0.884, medium-class F1 0.664 vs 0.696, and median session PCU error 2.5% vs 1.3%. It pushed about 170 cars into "medium". Light vehicles really are two thirds of the four-wheelers in this traffic, so the unweighted prior is the right one.

**Fix:** no reweighting. **Lesson:** "imbalanced, so reweight" is a reflex, not a result; measure it on the metric you actually care about.

### 8. I chose two settings on the same folds I report

The regularisation strength `C` (from 0.01 to 1) and the class weighting were both picked on the same session-grouped CV I report. With only two choices the optimism is small (macro-F1 across `C` ranges 0.865-0.884), but it is not zero, and the honest fix is a separate held-out set of sessions, which with 31 sessions would leave too few to test on. The next round of labelled data from a real junction is that held-out set.

### 9. pandas 3 strings broke scikit-learn

**Symptom:** `TypeError: only integer scalar arrays can be converted to a scalar index` inside `cross_val_predict`.

**Cause:** pandas 3 stores strings as Arrow arrays by default; scikit-learn indexes `y` with an integer array, which Arrow-backed arrays don't support. **Fix:** `df["label"].to_numpy(dtype=object)` everywhere labels leave the DataFrame (`evaluate.labels`).

### 10. A label fix could have been hidden by the feature cache

The first version cached labels together with the features (the slow part, about a minute). After the audit in #11 I edited the label CSV, and the cached file still held the old labels. **Fix:** features are cached, labels never are: `features.load` re-reads the CSV and merges it in on every call.

### 11. Some of the model's "mistakes" were mine

**What I did:** after the first model, I looked at 60 random four-wheelers where model and label disagreed. 7 were my labelling errors (a Maruti 800 marked medium, a microbus marked heavy, pick-ups marked medium), which I fixed. Since disagreements are 8% of four-wheelers, I estimate 1-2% of my labels are still wrong.

**The real model errors** cluster in two places: Maruti Omni vans called microbuses, and Mahindra Bolero/Scorpio and Tata Sumo jeeps called microbuses. From the front, a boxy van and a microbus look alike; the difference is length, which a front-view crop barely shows. That is the boundary the review queue targets (docs/ARCHITECTURE.md, "The labelling loop").

### 12. Errors partly cancel in the totals

The two main confusions go in opposite directions: 61 light vehicles called medium (+0.5 PCU each) and 107 medium called light (-0.5 each). Some of the 1.3% headline comes from that cancelling. The per-vehicle accuracy (94%) and the worst session (9.0%) are the honest companions to the headline, and both are in the README.

## Engineering

### 13. Getting features out of a detector that wasn't built to give them

The ONNX export only outputs final detections. The intermediate tensors have export-generated names; I found the two I wanted by running ONNX shape inference and picking by shape (`890`: 128x13x13, the stride-32 FPN lateral map; `841`: 128x26x26, a stride-16 backbone map), and added them as extra graph outputs at load time. Those names are specific to that exact file, so the file is pinned by SHA-256 and a test checks the feature vector has 256 values.

### 14. A box smaller than one feature cell would pool to NaN

Caught while writing the pooling code, before it bit. A distant motorcycle can be smaller than one 32-pixel cell. Slicing `fmap[:, r1:r2, c1:c2]` with `r1 == r2` gives an empty array and `mean()` returns NaN, which then poisons the classifier. **Fix:** the slice always covers at least one cell (`test_pool_never_returns_an_empty_slice`).

### 15. Every training image is greyscale

All 4,797 crops are greyscale, so the classifier has never seen colour. A colour camera at a junction would be a distribution shift on day one. **Fix:** the detector converts every frame to greyscale before anything else. Cheap insurance, and it costs nothing on the training crops.
