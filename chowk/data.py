"""Download the Kathmandu vehicle crops, build a manifest, and check data quality.

The source is a public dataset of 4,797 vehicles cut out of 30 street videos filmed in Kathmandu
(two-wheelers and four-wheelers). The rules here come from problems found in the real archive;
each one is documented in docs/CHALLENGES_AND_BUGS.md and covered by a test.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from chowk import config

# Pinned to a commit so the data can never change underneath the results.
SOURCE_COMMIT = "719dd8281134bc6b9223a79d1f3ecf0a0a735c59"
SOURCE_URL = (f"https://raw.githubusercontent.com/sdevkota007/vehicles-nepal-dataset/"
              f"{SOURCE_COMMIT}/VehiclesNepal.zip")
SOURCE_SHA256 = "613dcd37b30a5b2ef12efd2adac81493011625a5eecdc1fe5f1f923c04d4f66f"
ZIP_NAME = "VehiclesNepal.zip"
SOURCE_DIRS = {"2wheelers": "two_wheeler", "4wheelers": "four_wheeler"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected_sha256: str) -> Path:
    """Download once, then verify the checksum every time."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (pinned https URL)
            tmp.write_bytes(resp.read())
        tmp.rename(dest)
    actual = sha256(dest)
    if actual != expected_sha256:
        raise ValueError(f"checksum mismatch for {dest.name}: {actual}")
    return dest


def download(raw_dir: Path = config.RAW_DIR) -> Path:
    return fetch(SOURCE_URL, raw_dir / ZIP_NAME, SOURCE_SHA256)


def open_archive(raw_dir: Path = config.RAW_DIR) -> zipfile.ZipFile:
    return zipfile.ZipFile(raw_dir / ZIP_NAME)


def load_image(archive: zipfile.ZipFile, path: str) -> Image.Image:
    """Every crop as 3-channel RGB.

    Two files are stored as 8-bit greyscale ("L"), the rest as RGB, although all 4,797 are
    greyscale in content. The detector expects three channels, so convert everything.
    """
    return Image.open(io.BytesIO(archive.read(path))).convert("RGB")


def dhash(img: Image.Image) -> int:
    """64-bit difference hash: near-identical crops have hashes a few bits apart."""
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def build_manifest(archive: zipfile.ZipFile) -> pd.DataFrame:
    """One row per crop: id, source class, saved time, size, content hash and perceptual hash."""
    rows = []
    for info in archive.infolist():
        if info.is_dir() or not info.filename.lower().endswith(".jpg"):
            continue
        folder, name = info.filename.split("/", 1)
        raw = archive.read(info)
        img = Image.open(io.BytesIO(raw))
        rows.append({
            "path": info.filename,
            "source_class": SOURCE_DIRS[folder],
            "number": int(Path(name).stem),
            # The zip keeps each file's modification time: when it was saved during cropping.
            "saved_at": datetime(*info.date_time),
            "width": img.width,
            "height": img.height,
            "md5": hashlib.md5(raw).hexdigest(),  # noqa: S324 (duplicate detection, not security)
            "dhash": dhash(img),
        })
    df = pd.DataFrame(rows).sort_values(["saved_at", "path"]).reset_index(drop=True)
    df["image_id"] = df["source_class"].str[:2] + "_" + df["number"].astype(str).str.zfill(4)
    df["session"] = assign_sessions(df["saved_at"])
    df["exact_duplicate_of"] = _first_with_same(df, "md5")
    return df


def assign_sessions(saved_at: pd.Series, gap_seconds: int = config.SESSION_GAP_SECONDS) -> pd.Series:
    """Split the save times into sessions at every pause longer than gap_seconds.

    Both class folders share one timeline (the author saved two- and four-wheelers from the same
    video), so sessions are assigned over all files together, in time order.
    """
    order = saved_at.sort_values(kind="stable")
    gaps = order.diff().dt.total_seconds().fillna(0)
    session = (gaps > gap_seconds).cumsum().astype(int)
    return session.reindex(saved_at.index)


def _first_with_same(df: pd.DataFrame, column: str) -> pd.Series:
    first = df.groupby(column)["image_id"].transform("first")
    return first.where(first != df["image_id"])


def near_duplicate_pairs(df: pd.DataFrame, max_bits: int = 4) -> pd.DataFrame:
    """Pairs of crops whose perceptual hashes differ by at most max_bits (excluding exact copies)."""
    h = np.array(df["dhash"].tolist(), dtype=np.uint64)
    bits = np.unpackbits(h.view(np.uint8).reshape(-1, 8), axis=1).astype(bool)
    out = []
    for i in range(len(bits)):
        d = (bits[i + 1:] != bits[i]).sum(1)
        for j in np.nonzero(d <= max_bits)[0]:
            out.append((df["image_id"].iat[i], df["image_id"].iat[i + 1 + j], int(d[j]),
                        bool(df["session"].iat[i] == df["session"].iat[i + 1 + j])))
    return pd.DataFrame(out, columns=["a", "b", "bits", "same_session"])


@dataclass
class QualityReport:
    images: int = 0
    by_class: dict[str, int] = field(default_factory=dict)
    sessions: int = 0
    session_sizes: tuple[int, int, int] = (0, 0, 0)
    exact_duplicates: int = 0
    near_duplicate_pairs: int = 0
    near_duplicates_same_session: float = 0.0
    small_crops: int = 0
    out_of_order_saves: int = 0

    def lines(self) -> list[str]:
        lo, med, hi = self.session_sizes
        return [
            f"images: {self.images}  {self.by_class}",
            f"sessions (pause > {config.SESSION_GAP_SECONDS}s): {self.sessions}; images per session min/median/max "
            f"{lo}/{med}/{hi}",
            f"exact duplicate files: {self.exact_duplicates} (dropped)",
            f"near-duplicate pairs (dHash <= 4 bits): {self.near_duplicate_pairs}, "
            f"{self.near_duplicates_same_session:.0%} of them inside one session",
            f"crops under {config.SMALL_CROP_PX}px on the short side: {self.small_crops}",
            f"files whose number is out of save order: {self.out_of_order_saves}",
        ]


def quality_report(df: pd.DataFrame) -> QualityReport:
    near = near_duplicate_pairs(df[df["exact_duplicate_of"].isna()])
    sizes = df.groupby("session").size()
    ooo = 0
    for _, g in df.groupby("source_class"):
        ooo += int((g.sort_values("number")["saved_at"].diff().dt.total_seconds() < 0).sum())
    return QualityReport(
        images=len(df),
        by_class=df["source_class"].value_counts().to_dict(),
        sessions=int(df["session"].nunique()),
        session_sizes=(int(sizes.min()), int(sizes.median()), int(sizes.max())),
        exact_duplicates=int(df["exact_duplicate_of"].notna().sum()),
        near_duplicate_pairs=len(near),
        near_duplicates_same_session=float(near["same_session"].mean()) if len(near) else 0.0,
        small_crops=int((df[["width", "height"]].min(axis=1) < config.SMALL_CROP_PX).sum()),
        out_of_order_saves=ooo,
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate files (the same crop saved twice under consecutive numbers)."""
    return df[df["exact_duplicate_of"].isna()].reset_index(drop=True)


def manifest_path() -> Path:
    return config.ARTIFACT_DIR / "manifest.parquet"


def load_manifest(rebuild: bool = False) -> pd.DataFrame:
    path = manifest_path()
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    download()
    with open_archive() as archive:
        df = build_manifest(archive)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.assign(dhash=df["dhash"].astype("uint64")).to_parquet(path, index=False)
    return df
