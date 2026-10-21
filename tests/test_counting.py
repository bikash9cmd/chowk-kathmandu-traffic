from datetime import datetime

import pytest

from chowk.counting import CountLine, LineCounter, aggregate, congestion, interval_start, iou


def moving_box(y):
    return (100.0, y, 160.0, y + 60.0)


def test_vehicle_crossing_line_is_counted_once_with_majority_class():
    lc = LineCounter([CountLine("north", 300)])
    seen = []
    for k, y in enumerate(range(150, 400, 10)):
        cls = "medium" if k == 3 else "light"  # one bad frame
        seen += lc.update([(moving_box(y), cls)])
    assert seen == [("north", "light")]


def test_two_vehicles_and_a_parked_one():
    lc = LineCounter([CountLine("north", 300)])
    total = []
    for y in range(150, 400, 10):
        total += lc.update([(moving_box(y), "two_wheeler"), ((400.0, y + 20, 470.0, y + 80), "heavy"),
                            ((600.0, 280.0, 660.0, 340.0), "light")])  # parked across the line: never crosses
    assert sorted(total) == [("north", "heavy"), ("north", "two_wheeler")]


def test_track_lost_for_too_long_is_dropped():
    lc = LineCounter([CountLine("north", 300)], max_missed_frames=2)
    lc.update([(moving_box(250), "light")])
    for _ in range(3):
        lc.update([])
    assert lc.tracks == []


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_aggregate_pcu_and_rate_scale_with_observed_minutes():
    rec = aggregate("j", "a", datetime(2026, 10, 1, 8, 0), ["two_wheeler"] * 10 + ["heavy"] * 2, "v1",
                    minutes_observed=7.5)
    assert rec.pcu == 11.0
    assert rec.pcu_per_hour == 88.0  # 11 PCU in 7.5 min, not 11 x 4
    assert rec.counts == {"two_wheeler": 10, "light": 0, "medium": 0, "heavy": 2}


def test_interval_start():
    assert interval_start(datetime(2026, 10, 1, 8, 44, 59)) == datetime(2026, 10, 1, 8, 30)


@pytest.mark.parametrize(("pcu", "band"), [(300, "free"), (800, "busy"), (1000, "near_capacity"),
                                           (2000, "over_capacity")])
def test_congestion_bands(pcu, band):
    # 2 lanes x 1500 x 0.35 green = 1050 PCU/h capacity
    assert congestion(pcu, 2, 0.35, 1500)[1] == band
