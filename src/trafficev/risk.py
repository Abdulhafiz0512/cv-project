"""Part B: causal accident-anticipation risk.

The estimator sees each frame once, in order. At RISK_HZ it downsizes the
frame, detects and tracks road users online (ByteTrack is causal), keeps a
short history per track, and turns conflict cues into a probability that an
accident starts within the next 5 s:

- predicted footprint overlap: the ground footprints (lower part of the
  boxes) of two road users on crossing paths, extrapolated at constant
  velocity, overlap within ~1.5 s while they are apart now. Working with
  footprints instead of point distances keeps the cue robust to the strong
  perspective of this elevated view, where cars in adjacent lanes look close;
- hard braking with another road user close ahead;
- wrong-way motion on a one-way carriageway;
- crossing the stop line while another vehicle waits at it (red).

Each cue is a hand-set logistic so that 0.5 corresponds to a conflict an
operator would call alarming. The fused score rises instantly and decays with
a 2 s time constant, which keeps alarms contiguous without stretching them.
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import NamedTuple

import numpy as np

from . import runtime, scene, signals
from .labels import COCO_KIND
from .tracking import TrackStore
from .video import PROC_H, PROC_W, to_proc

RISK_HZ = 5.0          # perception rate on a GPU
CPU_RISK_HZ = 2.0      # the same on CPU-only machines
HISTORY_S = 1.6        # per-track memory
DECAY_TAU_S = 2.0
STEP_BUDGET_S = 0.012  # target amortised cost of step() per video frame
HORIZON_S = 1.6        # footprint extrapolation horizon
MOVERS = ("vehicle", "two_wheeler")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Motion(NamedTuple):
    kind: str
    foot: np.ndarray    # ground point, canonical px
    vel: np.ndarray     # px/s
    scale: float        # box height, px
    decel: float        # bh/s^2 (positive = slowing down)
    box: np.ndarray     # latest box x1 y1 x2 y2

    @property
    def speed(self) -> float:
        """bh/s"""
        return float(np.linalg.norm(self.vel)) / self.scale

    def footprint(self, dt: float = 0.0) -> np.ndarray:
        x1, y1, x2, y2 = self.box
        fp = np.array([x1, y2 - 0.4 * (y2 - y1), x2, y2])
        return fp + np.tile(self.vel * dt, 2)


class _TrackState:
    __slots__ = ("hist", "kind", "stopped_since")

    def __init__(self, kind: str):
        self.hist: deque = deque()
        self.kind = kind
        self.stopped_since: float | None = None

    def push(self, t: float, box: np.ndarray) -> None:
        self.hist.append((t, box))
        while self.hist and t - self.hist[0][0] > HISTORY_S:
            self.hist.popleft()

    def motion(self) -> Motion | None:
        """Least-squares velocity of the ground point over the recent history."""
        if len(self.hist) < 3:
            return None
        t = np.array([h[0] for h in self.hist])
        b = np.array([h[1] for h in self.hist])
        p = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], axis=1)
        s = float(max(np.median(b[:, 3] - b[:, 1]), 8.0))
        tc = t - t.mean()
        v = (tc[:, None] * (p - p.mean(axis=0))).sum(axis=0) / (float((tc ** 2).sum()) or 1e-6)
        half = len(t) // 2
        v1 = np.linalg.norm(p[half] - p[0]) / max(t[half] - t[0], 1e-3) / s
        v2 = np.linalg.norm(p[-1] - p[half]) / max(t[-1] - t[half], 1e-3) / s
        decel = (v1 - v2) / max((t[-1] - t[0]) / 2, 1e-3)
        return Motion(self.kind, p[-1], v, s, float(decel), b[-1])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class CausalRiskModel:
    def __init__(self, detector_factory):
        self._detector_factory = detector_factory
        self.det = None

    def reset(self, meta: dict) -> None:
        if self.det is None:
            self.det = self._detector_factory()
        on_gpu = self.det.device.startswith("cuda") or runtime.EXACT
        fps = float(meta.get("fps") or 25.0)
        self.stride = max(1, round(fps / (RISK_HZ if on_gpu else CPU_RISK_HZ)))
        self.store = TrackStore(PROC_W, PROC_H)
        self.states: dict[int, _TrackState] = {}
        self.n = 0
        self.score = 0.0
        self.last_t = 0.0
        self.cost = 0.0

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        self.n += 1
        if (self.n - 1) % self.stride:
            return self._decayed(t_sec)
        t0 = time.perf_counter()
        det = self.det([to_proc(frame)])[0]
        rows = self.store.update(t_sec, det)
        raw = self.cues(t_sec, rows)["risk"]
        self.score = max(raw, self._decayed(t_sec))
        self.last_t = t_sec
        self._adapt(time.perf_counter() - t0)
        return self.score

    def cues(self, t: float, rows: np.ndarray) -> dict[str, float]:
        """Update track states with one frame of tracks and score every cue."""
        seen = set()
        for x1, y1, x2, y2, tid, _conf, cls in rows:
            tid = int(tid)
            seen.add(tid)
            st = self.states.get(tid)
            if st is None:
                st = self.states[tid] = _TrackState(COCO_KIND.get(int(cls), "other"))
            st.push(t, np.array([x1, y1, x2, y2], dtype=np.float64))
        for tid in [k for k, s in self.states.items() if t - s.hist[-1][0] > HISTORY_S]:
            del self.states[tid]
        motions = {tid: m for tid in seen if (m := self.states[tid].motion()) is not None}
        cues = {
            "conflict": self._conflict(motions),
            "braking": self._braking(motions),
            "wrong_way": self._wrong_way(motions),
            "red": self._red(t, motions),
        }
        cues["risk"] = 1.0 - float(np.prod([1.0 - c for c in cues.values()]))
        return cues

    # ------------------------------------------------------------------ internals

    def _decayed(self, t: float) -> float:
        return self.score * math.exp(-(t - self.last_t) / DECAY_TAU_S)

    def _adapt(self, spent: float) -> None:
        """Lower the perception rate if step() costs too much per video frame."""
        if runtime.EXACT:
            return
        self.cost = 0.9 * self.cost + 0.1 * spent if self.cost else spent
        if self.cost / self.stride > STEP_BUDGET_S and self.stride < 30:
            self.stride += 1

    @staticmethod
    def _conflict(motions: dict[int, Motion]) -> float:
        best = 0.0
        items = list(motions.items())
        for i, (_, a) in enumerate(items):
            for _, b in items[i + 1:]:
                if a.kind not in MOVERS and b.kind not in MOVERS:
                    continue
                if a.kind == "other" or b.kind == "other":
                    continue
                if a.kind in MOVERS and b.kind in MOVERS:
                    if max(a.speed, b.speed) < 1.0 or min(a.speed, b.speed) < 0.3:
                        continue
                    ha, hb = a.vel / np.linalg.norm(a.vel), b.vel / np.linalg.norm(b.vel)
                    if abs(float(ha @ hb)) > 0.82:
                        continue  # same or opposite direction: lane following / passing
                else:
                    veh, ped = (a, b) if a.kind in MOVERS else (b, a)
                    if veh.speed < 1.0 or not scene.on_road(ped.foot[None])[0]:
                        continue
                if _iou(a.footprint(), b.footprint()) > 0.02:
                    continue  # already overlapping in the image: occlusion, not a prediction
                for dt in np.arange(0.2, HORIZON_S + 1e-6, 0.2):
                    if _iou(a.footprint(dt), b.footprint(dt)) > 0.2:
                        closing = float(np.linalg.norm(a.vel - b.vel)) / (0.5 * (a.scale + b.scale))
                        best = max(best, _sigmoid((1.0 - dt) / 0.2) * _sigmoid((closing - 2.0) / 0.4))
                        break
        return best

    @staticmethod
    def _braking(motions: dict[int, Motion]) -> float:
        best = 0.0
        for ta, a in motions.items():
            if a.kind not in MOVERS or a.decel < 3.0:
                continue
            speed = float(np.linalg.norm(a.vel))
            if speed < 1e-6:
                continue
            h = a.vel / speed
            for tb, b in motions.items():
                if tb == ta:
                    continue
                rel = b.foot - a.foot
                ahead, side = float(rel @ h), abs(float(h[0] * rel[1] - h[1] * rel[0]))
                if 0 < ahead < 2.0 * a.scale and side < 0.8 * a.scale:
                    best = max(best, 0.6 * _sigmoid((a.decel - 5.0) / 1.0))
        return best

    @staticmethod
    def _wrong_way(motions: dict[int, Motion]) -> float:
        best = 0.0
        for m in motions.values():
            if m.kind not in MOVERS or m.speed < 0.5:
                continue
            z = scene.one_way_zone(m.foot[None])[0]
            if z < 0:
                continue
            cos = float(m.vel @ scene.ONE_WAY[z].direction) / (m.speed * m.scale)
            if cos < -0.6:
                best = max(best, 0.35 * _sigmoid((m.speed - 1.5) / 0.5))
        return best

    def _red(self, t: float, motions: dict[int, Motion]) -> float:
        waiting, crossing = False, 0.0
        for tid, m in motions.items():
            if m.kind not in MOVERS:
                continue
            up = float(signals.distance_upstream(m.foot)[0])
            st = self.states[tid]
            near = -0.3 * m.scale < up < 2.2 * m.scale and scene.points_in_poly(m.foot[None], scene.QUEUE_ZONE)[0]
            if near and m.speed < 0.12:
                st.stopped_since = st.stopped_since or t
                waiting |= t - st.stopped_since > signals.MIN_WAIT_S
            else:
                st.stopped_since = None
            along = float(m.vel @ scene.INBOUND.direction) / m.scale
            if -1.5 * m.scale < up < 0.5 * m.scale and along > 2.0:
                crossing = max(crossing, _sigmoid((along - 3.0) / 0.7))
        return 0.3 * crossing if waiting else 0.0
