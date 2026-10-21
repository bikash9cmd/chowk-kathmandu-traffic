import zipfile
from datetime import datetime

import pandas as pd
import pytest

from chowk import data


def test_sessions_split_on_pauses_across_both_folders():
    t = pd.Series(pd.to_datetime(["2016-06-01 09:00:00", "2016-06-01 09:00:30", "2016-06-01 09:20:00",
                                  "2016-06-01 09:00:10"]))
    s = data.assign_sessions(t, gap_seconds=300)
    # The 09:00:10 file (other folder) belongs with the 09:00 files, not on its own.
    assert list(s) == [0, 0, 1, 0]


def test_manifest_reads_save_times_and_flags_exact_duplicates(tiny_zip):
    with zipfile.ZipFile(tiny_zip) as z:
        df = data.build_manifest(z)
    assert len(df) == 5
    assert df["saved_at"].min() == datetime(2016, 6, 1, 9, 0, 0)
    dup = df[df["exact_duplicate_of"].notna()]
    assert list(dup["path"]) == ["4wheelers/2.jpg"]
    assert df["session"].nunique() == 2
    assert len(data.clean(df)) == 4


def test_greyscale_files_are_loaded_as_three_channels(tiny_zip):
    with zipfile.ZipFile(tiny_zip) as z:
        img = data.load_image(z, "2wheelers/2.jpg")
    assert img.mode == "RGB"


def test_checksum_mismatch_is_an_error(tmp_path):
    f = tmp_path / "x.zip"
    f.write_bytes(b"not the dataset")
    with pytest.raises(ValueError, match="checksum mismatch"):
        data.fetch("https://example.invalid/never-called", f, "0" * 64)


def test_near_duplicates_are_found(tiny_zip):
    with zipfile.ZipFile(tiny_zip) as z:
        df = data.build_manifest(z)
    pairs = data.near_duplicate_pairs(df, max_bits=0)
    assert len(pairs) >= 1
