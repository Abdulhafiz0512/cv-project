"""Multi-object tracking (ByteTrack) and trajectory storage.

Coordinates are converted to a canonical 1920x1080 image space in which the
scene layout (``scene.py``) is defined, so every stage is independent of the
input and processing resolutions.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from .detector import Detections
from .labels import COCO_KIND

CANON_W, CANON_H = 1920.0, 1080.0

TRACKER_ARGS = SimpleNamespace(
    tracker_type="bytetrack",
    track_high_thresh=0.30,   # first association stage
    track_low_thresh=0.10,    # second stage recovers weak (occluded / dark) boxes
    new_track_thresh=0.40,    # a new id needs a confident box
    track_buffer=40,          # updates a lost track survives (4 s at 10 Hz)
    match_thresh=0.80,
    fuse_score=True,
)


def _contained(inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
    """Fraction of each inner box's area covered by each outer box, (N, M)."""
    x1 = np.maximum(inner[:, None, 0], outer[None, :, 0])
    y1 = np.maximum(inner[:, None, 1], outer[None, :, 1])
    x2 = np.minimum(inner[:, None, 2], outer[None, :, 2])
    y2 = np.minimum(inner[:, None, 3], outer[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (inner[:, 2] - inner[:, 0]) * (inner[:, 3] - inner[:, 1])
    return inter / np.maximum(area[:, None], 1e-6)


def drop_riders(det: Detections) -> Detections:
    """Remove persons riding two-wheelers or visible inside vehicles.

    A rider is part of the two-wheeler road user; a driver seen through a
    windscreen is not a pedestrian. Both would otherwise trigger jaywalking.
    """
    person = det.cls == 0
    carrier = np.isin(det.cls, [1, 2, 3, 5, 7])
    if not person.any() or not carrier.any():
        return det
    p_idx, c_idx = np.flatnonzero(person), np.flatnonzero(carrier)
    cover = _contained(det.xyxy[p_idx], det.xyxy[c_idx])
    ride = cover.max(axis=1) >= 0.45
    keep = np.ones(len(det), bool)
    keep[p_idx[ride]] = False
    return Detections(det.xyxy[keep], det.conf[keep], det.cls[keep])


@dataclass
class Track:
    tid: int
    cls: int
    kind: str
    t: np.ndarray       # (N,) seconds
    box: np.ndarray     # (N, 4) canonical pixels x1 y1 x2 y2
    conf: np.ndarray    # (N,)

    @property
    def foot(self) -> np.ndarray:
        """Ground contact point: bottom-centre of the box, (N, 2)."""
        return np.stack([(self.box[:, 0] + self.box[:, 2]) / 2, self.box[:, 3]], axis=1)

    @property
    def size(self) -> np.ndarray:
        """Box height in canonical pixels — the local perspective scale."""
        return self.box[:, 3] - self.box[:, 1]

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0

    def smooth_foot(self, window_s: float = 0.5) -> np.ndarray:
        return smooth(self.t, self.foot, window_s)

    def velocity(self, window_s: float = 0.6) -> np.ndarray:
        """Foot-point velocity in canonical px/s (central differences on the smoothed path)."""
        p = self.smooth_foot(window_s)
        if len(p) < 2:
            return np.zeros_like(p)
        v = np.gradient(p, self.t, axis=0)
        return v


def smooth(t: np.ndarray, x: np.ndarray, window_s: float) -> np.ndarray:
    """Centred moving average over a time window (non-causal, Part A only)."""
    x = np.asarray(x, dtype=np.float64)
    if len(t) < 3:
        return x.copy()
    x2 = x.reshape(len(t), -1)
    half = window_s / 2
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    csum = np.vstack([np.zeros((1, x2.shape[1])), np.cumsum(x2, axis=0)])
    return ((csum[hi] - csum[lo]) / (hi - lo)[:, None]).reshape(x.shape)


@dataclass
class TrackStore:
    """Feeds per-frame detections to ByteTrack and accumulates trajectories."""
    frame_w: int
    frame_h: int
    rows: dict[int, list] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.tracker = BYTETracker(TRACKER_ARGS)
        self.sx = CANON_W / self.frame_w
        self.sy = CANON_H / self.frame_h

    def update(self, t: float, det: Detections) -> np.ndarray:
        """Track one frame; returns rows [x1 y1 x2 y2 tid conf cls] in canonical px."""
        from ultralytics.engine.results import Boxes

        det = drop_riders(det)
        data = np.concatenate([det.xyxy, det.conf[:, None], det.cls[:, None].astype(np.float32)], axis=1)
        out = self.tracker.update(Boxes(data, (self.frame_h, self.frame_w)))
        if len(out) == 0:
            return np.zeros((0, 7), np.float32)
        out = out[:, :7].copy()
        out[:, [0, 2]] *= self.sx
        out[:, [1, 3]] *= self.sy
        for x1, y1, x2, y2, tid, conf, cls in out:
            self.rows.setdefault(int(tid), []).append((t, x1, y1, x2, y2, conf, int(cls)))
        return out

    def tracks(self, min_len: int = 3) -> list[Track]:
        return tracks_from_rows(self.rows, min_len)


def tracks_from_rows(rows_by_id: dict[int, list], min_len: int = 3) -> list[Track]:
    """Accumulated rows (t, x1, y1, x2, y2, conf, cls) -> Track objects."""
    result = []
    for tid, rows in rows_by_id.items():
        if len(rows) < min_len:
            continue
        a = np.asarray([r[:6] for r in rows], dtype=np.float64)
        votes: Counter = Counter()
        for r in rows:  # majority class, weighted by confidence
            votes[r[6]] += r[5]
        cls = votes.most_common(1)[0][0]
        result.append(Track(tid, cls, COCO_KIND.get(cls, "other"), a[:, 0], a[:, 1:5], a[:, 5]))
    return result
