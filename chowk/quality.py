"""Is this camera frame like the images the model was trained on?

The robustness experiments (docs/FAILURE_MODES.md) show the classifier degrades badly on dark,
blurred and heavily compressed images, and that training on degraded copies does not carry over to
kinds of damage it hasn't seen. So the box also *detects* bad input and says so in every record,
rather than quietly reporting wrong counts.

Thresholds are about the 1st (or 99th) percentile of each measure over the training crops. Measured
by `python -m chowk robustness` on 500 held-out crops: a clean crop trips at least one flag 4% of the
time; simulated dusk 100%, dusty lens 98%, heavy compression 99.6%, a distant vehicle 79%.
They were calibrated on crops, so they need re-calibrating on a site's own full frames.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

DARK_BELOW = 70.0          # mean grey level, 0-255
FLAT_BELOW = 25.0          # standard deviation of grey levels (contrast)
BLURRY_BELOW = 120.0       # variance of the Laplacian at 128x128
BLOCKY_ABOVE = 1.7         # JPEG 8x8 block-edge gradient / other gradients

FLAGS = ("dark", "low_contrast", "blurry", "compressed")


def measures(img: Image.Image) -> dict[str, float]:
    g = np.asarray(img.convert("L"), dtype=np.float32)
    small = np.asarray(img.convert("L").resize((128, 128), Image.BILINEAR), dtype=np.float32)
    lap = 4 * small[1:-1, 1:-1] - small[:-2, 1:-1] - small[2:, 1:-1] - small[1:-1, :-2] - small[1:-1, 2:]
    dx = np.abs(np.diff(g, axis=1))
    cols = np.arange(dx.shape[1])
    edge, inner = dx[:, cols % 8 == 7], dx[:, cols % 8 != 7]
    block = float(edge.mean() / (inner.mean() + 1e-6)) if edge.size and inner.size else 1.0
    return {"brightness": float(small.mean()), "contrast": float(small.std()),
            "sharpness": float(lap.var()), "blockiness": block}


def flags(img: Image.Image) -> set[str]:
    m = measures(img)
    out = set()
    if m["brightness"] < DARK_BELOW:
        out.add("dark")
    if m["contrast"] < FLAT_BELOW:
        out.add("low_contrast")
    if m["sharpness"] < BLURRY_BELOW:
        out.add("blurry")
    if m["blockiness"] > BLOCKY_ABOVE:
        out.add("compressed")
    return out
