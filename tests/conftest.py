import io
import zipfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from chowk import config


@pytest.fixture
def synthetic():
    """Features where each class is a separable blob, 12 sessions, all four classes in every session."""
    rng = np.random.default_rng(0)
    rows, feats = [], []
    centres = {c: rng.normal(0, 1, 32) * 3 for c in config.CLASSES}
    for s in range(12):
        for c in config.CLASSES:
            for _ in range(8):
                rows.append({"session": s, "label": c, "path": f"x/{len(rows)}.jpg",
                             "source_class": "two_wheeler" if c == "two_wheeler" else "four_wheeler"})
                feats.append(centres[c] + rng.normal(0, 1, 32))
    return pd.DataFrame(rows), np.array(feats, dtype=np.float32)


@pytest.fixture
def tiny_zip(tmp_path):
    """A miniature copy of the source archive: two folders, real JPEGs, controlled save times."""
    path = tmp_path / "VehiclesNepal.zip"
    t0 = datetime(2016, 6, 1, 9, 0, 0)
    entries = [("2wheelers/1.jpg", t0, 10), ("2wheelers/2.jpg", t0 + timedelta(seconds=20), 20),
               ("4wheelers/1.jpg", t0 + timedelta(seconds=40), 30),
               ("4wheelers/2.jpg", t0 + timedelta(seconds=42), 30),  # same bytes as 4wheelers/1
               ("4wheelers/3.jpg", t0 + timedelta(minutes=30), 200)]
    with zipfile.ZipFile(path, "w") as z:
        for name, t, shade in entries:
            buf = io.BytesIO()
            mode = "L" if name == "2wheelers/2.jpg" else "RGB"
            img = Image.new(mode, (80, 120), shade if mode == "L" else (shade, shade, shade))
            img.save(buf, format="JPEG")
            info = zipfile.ZipInfo(name, date_time=t.timetuple()[:6])
            z.writestr(info, buf.getvalue())
    return path
