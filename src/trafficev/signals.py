"""Signal-phase evidence for the inbound approach.

The inbound signal heads face away from the camera, so the phase is read
from behaviour: a vehicle waiting at the front of the queue, just before the
stop line, for more than a couple of seconds means red. Visible lamp heads
(``scene.SIGNAL_HEADS``) are also measured every sampled frame; their
colour series is exported for EDA and used as a secondary cue.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import scene
from .kinematics import Kin, runs

MIN_WAIT_S = 2.0        # a stop this long at the line is a red phase, not hesitation
REACTION_S = 1.0        # leader moves ~1 s after green


@dataclass
class LampMeter:
    """Per-frame fraction of lit red / green pixels in each visible signal head."""
    times: list[float] = field(default_factory=list)
    values: dict[str, list[tuple[float, float]]] = field(default_factory=dict)

    def update(self, t: float, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        s = np.array([w / scene.CANON[0], h / scene.CANON[1]])
        self.times.append(t)
        for name, box in scene.SIGNAL_HEADS.items():
            (x0, y0), (x1, y1) = (box * s).astype(int)
            roi = frame[max(0, y0):y1, max(0, x0):x1]
            if roi.size == 0:
                self.values.setdefault(name, []).append((0.0, 0.0))
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            lit = (sat > 90) & (val > 150)
            red = lit & ((hue < 10) | (hue > 170))
            green = lit & (hue > 55) & (hue < 95)
            self.values.setdefault(name, []).append((float(red.mean()), float(green.mean())))

    def series(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.values.items()}


def distance_upstream(foot: np.ndarray) -> np.ndarray:
    """Signed distance (px) of points before the stop line; negative once past it."""
    a, b = scene.STOP_LINE[0], scene.STOP_LINE[-1]
    d = b - a
    n = np.array([d[1], -d[0]]) / np.linalg.norm(d)   # normal pointing upstream
    if np.dot(n, scene.INBOUND.direction) > 0:
        n = -n
    return (np.atleast_2d(foot) - a) @ n


def red_intervals(kins: list[Kin]) -> list[tuple[float, float, int]]:
    """(start, end, tid) spans in which an inbound vehicle waited at the stop line."""
    out = []
    for k in kins:
        if k.kind not in ("vehicle", "two_wheeler"):
            continue
        up = distance_upstream(k.foot)
        near_line = ((up > -0.3 * k.scale) & (up < 2.2 * k.scale)
                     & scene.points_in_poly(k.foot, scene.QUEUE_ZONE) & ~scene.points_in_poly(k.foot, scene.PARKING))
        for i, j in runs(k.still() & near_line):
            if k.t[j] - k.t[i] >= MIN_WAIT_S:
                out.append((float(k.t[i]), float(k.t[j]) - REACTION_S, k.tid))
    return out


def is_red(t: float, reds: list[tuple[float, float, int]], exclude_tid: int | None = None) -> bool:
    return any(s <= t <= e and tid != exclude_tid for s, e, tid in reds)
