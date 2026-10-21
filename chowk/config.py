"""Project-wide constants. Everything a reviewer might question lives here, with a reason."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CHOWK_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
LABELS_CSV = DATA_DIR / "labels" / "vehicle_classes.csv"
ARTIFACT_DIR = Path(os.environ.get("CHOWK_ARTIFACT_DIR", ROOT / "artifacts"))

# ---------------------------------------------------------------- vehicle classes and PCU
# Nepal Road Standard 2070 (Department of Roads), Table 4-1 "Vehicle types, Equivalency Factors".
# A junction's capacity is planned in Passenger Car Units, so a count is only useful once every
# vehicle has been put into one of these buckets.
#
#   Bicycle, Motorcycle ......................................... 0.5
#   Car, Auto Rickshaw, SUV, Light Van, Pick Up ................. 1.0
#   Light (Mini) Truck, Tractor, Rickshaw ....................... 1.5
#   Truck, Bus, Minibus, Tractor with trailer ................... 3.0
#
# Two Kathmandu vehicles are not named in the table, so the labels follow these rules
# (docs/ALTERNATIVES.md, ADR-003; sensitivity in docs/MODEL_CARD.md):
#   * the three-wheeled electric "Safa tempo" is an auto rickshaw: light, 1.0;
#   * the 12-15 seat microbus (Hiace, Tata Winger, Force Traveller) is a light van in the standard, but
#     it is longer, stops anywhere to pick up passengers, and sits between a van and a minibus. It goes
#     in the 1.5 bucket with mini trucks. This is my assumption, not the standard.
CLASSES = ("two_wheeler", "light", "medium", "heavy")
PCU = {"two_wheeler": 0.5, "light": 1.0, "medium": 1.5, "heavy": 3.0}

# How the COCO classes of an off-the-shelf detector map onto PCU buckets (the zero-shot baseline).
# COCO has no microbus, mini truck or tempo, so "medium" can never be predicted this way.
COCO_TO_CLASS = {"bicycle": "two_wheeler", "motorcycle": "two_wheeler", "car": "light",
                 "bus": "heavy", "truck": "heavy"}
COCO_VEHICLE_IDS = {"bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}

# ---------------------------------------------------------------- data
# The source crops were saved one by one while cutting vehicles out of 30 street videos.
# A pause of more than this between two saved files starts a new "session"; at 5 minutes the
# files fall into 31 sessions, matching the 30 videos the dataset's author describes. Sessions are
# the unit of cross-validation: the same vehicle appears in several consecutive frames, so a random
# split puts near-copies of a test image in the training set (docs/CHALLENGES_AND_BUGS.md #4).
SESSION_GAP_SECONDS = 300
CV_FOLDS = 5

# Crops smaller than this on their short side are distant vehicles a camera would not classify
# reliably either. They are kept for training but reported separately.
SMALL_CROP_PX = 64

# ---------------------------------------------------------------- detector
# YOLOX-Nano (Megvii, Apache-2.0), COCO-trained, exported to ONNX by its authors.
# 3.7 MB, runs on a Raspberry Pi-class CPU. Pinned by URL and SHA-256.
DETECTOR_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"
DETECTOR_SHA256 = "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"
DETECTOR_INPUT = 416
# Intermediate tensors exposed as extra graph outputs: the stride-32 FPN lateral map and the
# stride-16 backbone map, 128 channels each. Pooled over a box, they are the classifier's input.
FEATURE_TENSORS = ("890", "841")
DETECT_SCORE_MIN = 0.25
NMS_IOU = 0.45

# ---------------------------------------------------------------- classifier
# Below this probability a vehicle's crop is queued for human review (it is still counted with the
# model's best guess). At 0.55, 2.5% of vehicles are queued and the model is right on only about half
# of them, so the queue is mostly real mistakes: the most useful next labels (docs/MODEL_CARD.md).
MIN_CONFIDENCE = 0.55
# Promotion gate: a retrained classifier replaces the champion only if its session-grouped macro-F1
# is no more than this much worse, and its PCU error no more than this fraction worse.
GATE_MAX_F1_DROP = 0.01
GATE_MAX_PCU_ERROR_RISE = 0.10

# ---------------------------------------------------------------- flows
# Counts are reported per 15 minutes, the interval traffic engineers use for peak-hour factors.
FLOW_INTERVAL_MINUTES = 15
# Saturation flow per lane for a signalised approach, PCU per hour of green. 1,800 is the common
# planning default (HCM / IRC:SP:41); Kathmandu's narrow lanes and mixed traffic are lower, so it is a
# parameter per approach in the junction config, not a constant in the code.
DEFAULT_SATURATION_PCU_PER_LANE_HOUR = 1800
# Volume-to-capacity bands used for the congestion level in the API.
VC_BANDS = ((0.6, "free"), (0.85, "busy"), (1.0, "near_capacity"), (float("inf"), "over_capacity"))

SEED = 42
