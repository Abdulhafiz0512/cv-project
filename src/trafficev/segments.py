"""Temporal segment utilities: per-sample flags -> clean [start, end] intervals."""
from __future__ import annotations

import numpy as np

Interval = tuple[float, float]


def flags_to_intervals(t: np.ndarray, flags: np.ndarray, dt: float) -> list[Interval]:
    """Runs of True samples -> intervals; each sample covers [t, t + dt)."""
    out: list[Interval] = []
    start = None
    for ti, f in zip(t, flags):
        if f and start is None:
            start = ti
        elif not f and start is not None:
            out.append((float(start), float(ti)))
            start = None
    if start is not None:
        out.append((float(start), float(t[-1] + dt)))
    return out


def union(intervals: list[Interval]) -> list[Interval]:
    """Merge overlapping or touching intervals (same class = one segment)."""
    out: list[list[float]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def close_gaps(intervals: list[Interval], max_gap: float) -> list[Interval]:
    out: list[list[float]] = []
    for s, e in sorted(intervals):
        if out and s - out[-1][1] <= max_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def drop_short(intervals: list[Interval], min_len: float) -> list[Interval]:
    return [(s, e) for s, e in intervals if e - s >= min_len]


def clip(intervals: list[Interval], duration: float) -> list[Interval]:
    out = []
    for s, e in intervals:
        s, e = max(0.0, s), min(duration, e)
        if e - s > 1e-3:
            out.append((s, e))
    return out


def tiou(a: Interval, b: Interval) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    uni = max(a[1], b[1]) - min(a[0], b[0])
    return inter / uni if uni > 0 else 0.0
