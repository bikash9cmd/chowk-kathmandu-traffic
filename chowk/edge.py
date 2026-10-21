"""The loop that runs on the camera box at a junction.

frame -> detector boxes + features -> classifier -> tracker/line counter -> 15-minute FlowRecord
-> outbox on disk -> MQTT to AWS IoT Core when the link is up.

The parts that matter when things break:
  * minutes_observed: frames stop (camera crash, power cut) -> the record says how long it actually saw,
    and the rate is scaled from that, instead of a silent undercount.
  * outbox: records are written to disk before sending and deleted only after the broker acks,
    so a 4G outage delays data instead of losing it. The ingest side is idempotent on
    (junction, approach, interval_start), so re-sends after a reconnect are harmless.
  * review queue: low-confidence crops are kept (a few per interval) for labelling.
  * image_quality: the share of checked frames that were dark, low-contrast, blurry or compressed,
    so a dirty lens or dusk shows up in the data instead of as silently wrong counts.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from chowk import config, model, quality
from chowk.counting import CountLine, LineCounter, aggregate, interval_start

# A gap between frames longer than this counts as "not observing" (normal spacing is 0.2-0.5 s).
MAX_FRAME_GAP = timedelta(seconds=5)
MAX_REVIEW_CROPS_PER_INTERVAL = 5
# Image checks are cheap but not free; one frame in this many is enough to see a dirty lens or dusk.
QUALITY_EVERY_N_FRAMES = 10


@dataclass
class Outbox:
    root: Path

    def __post_init__(self):
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, record: dict) -> Path:
        name = f"{record['junction']}_{record['approach']}_{record['interval_start'].replace(':', '')}.json"
        path = self.root / name
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record))
        tmp.replace(path)
        return path

    def pending(self) -> list[Path]:
        return sorted(self.root.glob("*.json"))

    def flush(self, send: Callable[[dict], bool]) -> int:
        """Send oldest first; stop at the first failure so order is kept. Returns how many were sent."""
        sent = 0
        for path in self.pending():
            if not send(json.loads(path.read_text())):
                break
            path.unlink()
            sent += 1
        return sent


@dataclass
class CameraBox:
    junction: str
    lines: list[CountLine]
    detector: object  # chowk.detector.Detector, or a fake in tests
    classifier: object
    model_version: str
    outbox: Outbox
    review_dir: Path | None = None
    counter: LineCounter = field(init=False)
    _interval: datetime | None = field(default=None, init=False)
    _classes: dict[str, list[str]] = field(default_factory=dict, init=False)
    _observed: timedelta = field(default=timedelta(0), init=False)
    _last_frame: datetime | None = field(default=None, init=False)
    _reviews: int = field(default=0, init=False)
    _frames: int = field(default=0, init=False)
    _checked: int = field(default=0, init=False)
    _flagged: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.counter = LineCounter(self.lines)

    def process(self, frame, ts: datetime) -> list[dict]:
        """One frame. Returns any FlowRecords closed by this frame (at an interval boundary)."""
        closed = []
        start = interval_start(ts)
        if self._interval is None:
            self._interval = start
        elif start != self._interval:
            closed = self.close_interval()
            self._interval = start
        if self._last_frame is not None:
            gap = ts - self._last_frame
            if gap <= MAX_FRAME_GAP:
                self._observed += gap
        self._last_frame = ts
        self._frames += 1
        if self._frames % QUALITY_EVERY_N_FRAMES == 1:
            self._checked += 1
            for f in quality.flags(frame):
                self._flagged[f] = self._flagged.get(f, 0) + 1

        dets = self.detector.detect(frame)
        if dets:
            X = np.stack([d.features for d in dets])
            classes, review = model.predict_with_review(self.classifier, X)
            if self.review_dir is not None:
                self._queue_reviews(frame, dets, review, ts)
            made = self.counter.update([(d.box, str(c)) for d, c in zip(dets, classes, strict=True)])
        else:
            made = self.counter.update([])
        for approach, cls in made:
            self._classes.setdefault(approach, []).append(cls)
        return closed

    def close_interval(self) -> list[dict]:
        records = []
        # If the box restarted mid-interval, it saw less than 15 minutes; the record says so.
        minutes = min(self._observed.total_seconds() / 60, config.FLOW_INTERVAL_MINUTES)
        iq = {f: round(self._flagged.get(f, 0) / self._checked, 3) for f in quality.FLAGS} if self._checked else {}
        for line in self.lines:
            rec = aggregate(self.junction, line.approach, self._interval, self._classes.get(line.approach, []),
                            self.model_version, minutes_observed=max(minutes, 0.01), image_quality=iq).as_dict()
            self.outbox.put(rec)
            records.append(rec)
        self._classes, self._observed, self._reviews = {}, timedelta(0), 0
        self._checked, self._flagged = 0, {}
        return records

    def swap_model(self, classifier, version: str) -> None:
        """Called when a new champion arrives on chowk/model/champion. Takes effect at the next frame;
        the current interval's record carries the version that counted most of it."""
        self.classifier, self.model_version = classifier, version

    def _queue_reviews(self, frame, dets, review, ts: datetime) -> None:
        for d, flag in zip(dets, review, strict=True):
            if not flag or self._reviews >= MAX_REVIEW_CROPS_PER_INTERVAL:
                continue
            crop = frame.crop(tuple(int(v) for v in d.box))
            self.review_dir.mkdir(parents=True, exist_ok=True)
            crop.save(self.review_dir / f"{self.junction}_{ts:%Y%m%dT%H%M%S}_{self._reviews}.jpg", quality=85)
            np.save(self.review_dir / f"{self.junction}_{ts:%Y%m%dT%H%M%S}_{self._reviews}.npy", d.features)
            self._reviews += 1
