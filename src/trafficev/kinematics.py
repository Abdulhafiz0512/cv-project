"""Per-track kinematics in a perspective-normalised form.

Speeds are expressed in *box heights per second* (bh/s): dividing image
speed by the object's own apparent size cancels most of the perspective
scale change across this elevated view, so one threshold works in the
foreground and near the horizon. A parked car jitters at < 0.05 bh/s, a car
at walking pace moves ~0.6 bh/s, free-flowing traffic 3-8 bh/s.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tracking import Track, smooth

STILL_BHS = 0.12   # below this a road user counts as stationary


@dataclass
class Kin:
    track: Track
    t: np.ndarray
    foot: np.ndarray       # (N, 2) smoothed ground point, canonical px
    center: np.ndarray     # (N, 2) smoothed box centre
    scale: np.ndarray      # (N,) smoothed box height, px
    vel: np.ndarray        # (N, 2) px/s
    rel_speed: np.ndarray  # (N,) bh/s
    accel: np.ndarray      # (N,) d(rel_speed)/dt, bh/s^2

    @property
    def kind(self) -> str:
        return self.track.kind

    @property
    def tid(self) -> int:
        return self.track.tid

    def heading(self) -> np.ndarray:
        """Unit direction of motion (zero vector while stationary)."""
        n = np.linalg.norm(self.vel, axis=1, keepdims=True)
        return np.where(n > 1e-6, self.vel / np.maximum(n, 1e-6), 0.0)

    def still(self) -> np.ndarray:
        return self.rel_speed < STILL_BHS

    def at(self, t: float) -> int:
        """Index of the sample nearest to time t."""
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))


def kinematics(track: Track, pos_window: float = 0.8) -> Kin:
    t = track.t
    foot = smooth(t, track.foot, pos_window)
    center = smooth(t, (track.box[:, :2] + track.box[:, 2:]) / 2, pos_window)
    scale = smooth(t, track.size, 2.0)
    scale = np.maximum(scale, 8.0)
    if len(t) >= 2:
        vel = np.gradient(foot, t, axis=0)
        vel = smooth(t, vel, pos_window)
    else:
        vel = np.zeros_like(foot)
    rel = np.linalg.norm(vel, axis=1) / scale
    acc = smooth(t, np.gradient(rel, t), pos_window) if len(t) >= 2 else np.zeros_like(rel)
    return Kin(track, t, foot, center, scale, vel, rel, acc)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges [i, j] (inclusive) of consecutive True values."""
    out = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out
