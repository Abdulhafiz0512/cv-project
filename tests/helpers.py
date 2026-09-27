"""Shared helpers for the test-suite: synthetic trajectories in the real scene layout."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from trafficev import events, scene, signals  # noqa: E402
from trafficev.kinematics import kinematics  # noqa: E402
from trafficev.labels import COCO_KIND  # noqa: E402
from trafficev.tracking import Track  # noqa: E402

HZ = 10.0
SIZES = {0: (22.0, 60.0), 2: (90.0, 60.0), 3: (30.0, 45.0), 5: (220.0, 110.0)}  # w, h in canonical px


def norm(x: float, y: float) -> np.ndarray:
    return np.array([x, y]) * scene.CANON


def make_track(tid: int, cls: int, t0: float, waypoints: list[tuple[float, np.ndarray]]) -> Track:
    """Piecewise-linear foot path through (time, canonical point) waypoints, sampled at 10 Hz."""
    t_end = waypoints[-1][0]
    t = np.arange(t0, t_end + 1e-9, 1 / HZ)
    wt = np.array([w[0] for w in waypoints])
    wp = np.stack([w[1] for w in waypoints])
    fx, fy = np.interp(t, wt, wp[:, 0]), np.interp(t, wt, wp[:, 1])
    w, h = SIZES[cls]
    box = np.stack([fx - w / 2, fy - h, fx + w / 2, fy], axis=1)
    return Track(tid, cls, COCO_KIND[cls], t, box, np.full(len(t), 0.9))


def context(tracks: list[Track], duration: float) -> events.Context:
    kins = [kinematics(tr) for tr in tracks]
    return events.Context(kins, duration, reds=signals.red_intervals(kins))
