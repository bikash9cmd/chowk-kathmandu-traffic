"""From per-frame detections to 15-minute PCU flows per approach.

Runs on the camera box. Every frame: match detections to tracks (greedy IoU), and when a
track's centre crosses the approach's count line, count it once with the class it was given most
often over its life (a vehicle seen 20 times gets 20 votes, which smooths single-frame mistakes).
Only the 15-minute totals leave the box: a few hundred bytes, not video.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from chowk import config


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    box: tuple[float, float, float, float]
    votes: Counter = field(default_factory=Counter)
    last_seen: int = 0
    prev_cy: float | None = None
    counted: bool = False

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2

    def vehicle_class(self) -> str:
        return self.votes.most_common(1)[0][0] if self.votes else "light"


@dataclass
class CountLine:
    """A horizontal line across one approach, in frame pixels; vehicles moving down it are counted."""
    approach: str
    y: float
    x_min: float = 0.0
    x_max: float = float("inf")


class LineCounter:
    def __init__(self, lines: list[CountLine], iou_match: float = 0.3, max_missed_frames: int = 5):
        self.lines = lines
        self.iou_match = iou_match
        self.max_missed = max_missed_frames
        self.tracks: list[Track] = []
        self._next_id = 0
        self.frame = 0
        self.events: list[tuple[int, str, str]] = []  # (frame, approach, class)

    def update(self, detections: list[tuple[tuple[float, float, float, float], str]]) -> list[tuple[str, str]]:
        """Feed one frame of (box, class) pairs; returns the (approach, class) counts made this frame."""
        self.frame += 1
        unmatched = list(range(len(detections)))
        # Greedy: best IoU pairs first. Good enough at 2-5 fps where vehicles move a fraction of a box per frame.
        pairs = sorted(((iou(t.box, detections[j][0]), i, j) for i, t in enumerate(self.tracks)
                        for j in unmatched), reverse=True)
        used_t, used_d = set(), set()
        for score, i, j in pairs:
            if score < self.iou_match or i in used_t or j in used_d:
                continue
            t = self.tracks[i]
            t.prev_cy = t.cy
            t.box = detections[j][0]
            t.votes[detections[j][1]] += 1
            t.last_seen = self.frame
            used_t.add(i)
            used_d.add(j)
        for j in unmatched:
            if j not in used_d:
                box, cls = detections[j]
                self.tracks.append(Track(self._next_id, box, Counter({cls: 1}), self.frame))
                self._next_id += 1
        made = []
        for t in self.tracks:
            if t.counted or t.prev_cy is None or t.last_seen != self.frame:
                continue
            cx = (t.box[0] + t.box[2]) / 2
            for line in self.lines:
                if t.prev_cy < line.y <= t.cy and line.x_min <= cx <= line.x_max:
                    t.counted = True
                    made.append((line.approach, t.vehicle_class()))
                    self.events.append((self.frame, line.approach, t.vehicle_class()))
                    break
        self.tracks = [t for t in self.tracks if self.frame - t.last_seen <= self.max_missed]
        return made


def interval_start(ts: datetime, minutes: int = config.FLOW_INTERVAL_MINUTES) -> datetime:
    return ts.replace(minute=ts.minute - ts.minute % minutes, second=0, microsecond=0)


@dataclass
class FlowRecord:
    """What one camera sends every 15 minutes for one approach."""
    junction: str
    approach: str
    interval_start: str  # ISO 8601, Nepal time (UTC+05:45)
    counts: dict[str, int]
    pcu: float
    pcu_per_hour: float
    minutes_observed: float  # < 15 when the camera was down for part of the interval
    model_version: str
    # Fraction of checked frames with each image-quality flag (chowk/quality.py). Empty = not checked.
    image_quality: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def aggregate(junction: str, approach: str, start: datetime, classes: list[str], model_version: str,
              minutes_observed: float = config.FLOW_INTERVAL_MINUTES,
              image_quality: dict[str, float] | None = None) -> FlowRecord:
    counts = Counter(classes)
    pcu = sum(config.PCU[c] * n for c, n in counts.items())
    rate = pcu * 60 / minutes_observed if minutes_observed > 0 else 0.0
    return FlowRecord(junction, approach, start.isoformat(), {c: counts.get(c, 0) for c in config.CLASSES},
                      round(pcu, 1), round(rate, 1), round(minutes_observed, 2), model_version,
                      dict(image_quality or {}))


def congestion(pcu_per_hour: float, lanes: int, green_ratio: float,
               saturation: float = config.DEFAULT_SATURATION_PCU_PER_LANE_HOUR) -> tuple[float, str]:
    """Volume-to-capacity ratio of a signalised approach and its band."""
    capacity = lanes * saturation * green_ratio
    vc = pcu_per_hour / capacity if capacity > 0 else float("inf")
    for upper, name in config.VC_BANDS:
        if vc < upper:
            return round(vc, 3), name
    return round(vc, 3), config.VC_BANDS[-1][1]


def windows(start: datetime, end: datetime, minutes: int = config.FLOW_INTERVAL_MINUTES):
    t = interval_start(start, minutes)
    while t < end:
        yield t
        t += timedelta(minutes=minutes)
