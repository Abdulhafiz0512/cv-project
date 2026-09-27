"""Part A orchestration: video -> perception (detect + track) -> rules -> events.

Time budget: the harness allows 3 x duration for Part A and Part B together,
and its Part B loop decodes every 4K frame again. Part A therefore gets
PART_A_SHARE x duration. Decoding runs on a producer thread so it overlaps
inference; if the projected finish is late, frames are thinned, and at the
hard deadline the rules run on whatever has been perceived so far.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import events, runtime, signals
from .detector import Detector
from .kinematics import kinematics
from .tracking import TrackStore, tracks_from_rows
from .video import PROC_H, PROC_W, VideoInfo, probe, sample_frames

PART_A_SHARE = 1.1   # x video duration
TARGET_FPS = 10.0    # I/P frames of the camera's IBBP GOP at 29.97 fps
MAX_THIN = 4         # thin to at most every 4th sampled frame when behind schedule
ENABLED = tuple(events.RULES)

_detector: Detector | None = None


def get_detector() -> Detector:
    """One model instance per process, shared by Part A and Part B."""
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector


@dataclass
class Perception:
    info: VideoInfo
    times: np.ndarray
    rows: dict[int, list]
    lamps: dict[str, np.ndarray]
    complete: bool

    def tracks(self):
        return tracks_from_rows(self.rows)


def _frames_async(path: str, target_fps: float, maxsize: int = 32):
    """Decode on a producer thread; yields (t, frame) and finally None."""
    q: queue.Queue = queue.Queue(maxsize=maxsize)
    stop = threading.Event()

    def produce():
        try:
            for item in sample_frames(path, target_fps):
                if stop.is_set():
                    break
                q.put(item)
        finally:
            q.put(None)

    th = threading.Thread(target=produce, daemon=True)
    th.start()
    try:
        while (item := q.get()) is not None:
            yield item
    finally:
        stop.set()
        while th.is_alive():  # drain so the producer can exit
            try:
                q.get(timeout=0.1)
            except queue.Empty:
                pass


def perceive(path: str, budget_s: float | None = None, target_fps: float = TARGET_FPS,
             detector: Detector | None = None, on_frame=None) -> Perception:
    """Detect and track road users over the whole video.

    on_frame(t, tracked_rows) is called after every tracker update, in time
    order (the live demo derives its causal risk curve and progress from it).
    """
    info = probe(path)
    cached = _cache_load(path) if on_frame is None else None
    if cached is not None:
        return cached
    det = detector or get_detector()
    store = TrackStore(PROC_W, PROC_H)
    lamps = signals.LampMeter()
    budget = runtime.Budget(budget_s if budget_s is not None else PART_A_SHARE * max(info.duration, 1.0))
    batch_size = 8 if det.device.startswith("cuda") else 1
    batch: list[tuple[float, np.ndarray]] = []
    times: list[float] = []
    thin, n, complete = 1, 0, True

    def flush():
        dets = det([f for _, f in batch])
        for (t, frame), d in zip(batch, dets):
            rows = store.update(t, d)
            lamps.update(t, frame)
            times.append(t)
            if on_frame is not None:
                on_frame(t, rows)
        batch.clear()

    for t, frame in _frames_async(path, target_fps):
        n += 1
        if n % thin:
            continue
        batch.append((t, frame))
        if len(batch) >= batch_size:
            flush()
        if n % 20 == 0 and not runtime.EXACT:
            if budget.remaining() < 0:
                complete = False
                break
            if not budget.on_track(t / max(info.duration, 1e-6)) and thin < MAX_THIN:
                thin += 1
    if batch:
        flush()
    per = Perception(info, np.asarray(times), store.rows, lamps.series(), complete)
    _cache_save(path, per)
    return per


def detect_events(path: str) -> list[list]:
    return events_from(perceive(path))


def events_from(per: Perception) -> list[list]:
    kins = [kinematics(tr) for tr in per.tracks()]
    ctx = events.Context(kins, per.info.duration, reds=signals.red_intervals(kins))
    return events.run_rules(ctx, ENABLED)


# --------------------------------------------------------------------------- dev cache
# Set TRAFFICEV_CACHE=<dir> to store perception per video, so rules can be
# iterated without re-running the detector. Never set in the official run.

def _cache_path(path: str) -> Path | None:
    root = os.environ.get("TRAFFICEV_CACHE")
    if not root:
        return None
    st = Path(path).stat()
    key = hashlib.sha1(f"{Path(path).name}:{st.st_size}".encode()).hexdigest()[:12]
    return Path(root) / f"{Path(path).stem}_{key}.pkl"


def _cache_load(path: str) -> Perception | None:
    p = _cache_path(path)
    if p is None or not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def _cache_save(path: str, per: Perception) -> None:
    p = _cache_path(path)
    if p is None:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as fh:
        pickle.dump(per, fh)
