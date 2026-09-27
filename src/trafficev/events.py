"""Event rules: trajectories + scene layout -> segments per class.

Every rule is precision-first. The metric averages F1 over the classes that
occur in the ground truth *or* in the predictions, so a spurious prediction
of a class that never occurs costs a whole class at F1 = 0. Where a rule
cannot be made reliable from this view (illegal_turn needs lane-level turn
restrictions we cannot read from the scene), it is not emitted at all.

Segment boundaries follow the annotation conventions of the task: a strong
"core" condition must hold for a minimum time, and the reported segment is the
surrounding run of a weaker condition (e.g. jaywalking: the person is deep in
the carriageway for >= 1 s; the segment spans stepping on -> stepping off).
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field

import numpy as np

from . import scene, signals
from .kinematics import Kin, runs
from .segments import Interval, clip, close_gaps, drop_short, union

VEHICLES = ("vehicle", "two_wheeler")


@dataclass
class Context:
    kins: list[Kin]
    duration: float
    reds: list[signals.Red] = field(default_factory=list)
    # (label, track id, start, end) of every road user a rule fired on: for review
    # renders and error analysis only, never part of the output.
    evidence: list[tuple[str, int, float, float]] = field(default_factory=list)
    label: str = ""

    def of(self, *kinds: str) -> list[Kin]:
        return [k for k in self.kins if k.kind in kinds]

    def blame(self, k: Kin, s: float, e: float) -> None:
        self.evidence.append((self.label, int(k.tid), float(s), float(e)))


# --------------------------------------------------------------------------- helpers

def _grown_runs(k: Kin, weak: np.ndarray, strong: np.ndarray, min_strong_s: float) -> list[Interval]:
    """Runs of `weak` that contain a run of `strong` lasting >= min_strong_s."""
    out = []
    for i, j in runs(weak):
        ok = any(k.t[i + b] - k.t[i + a] >= min_strong_s for a, b in runs(strong[i:j + 1]))
        if ok:
            out.append((float(k.t[i]), float(k.t[j])))
    return out


def _clear(k: Kin, margin: float = 4.0) -> np.ndarray:
    """Samples whose box is clear of the frame border. A box cut by the edge (a car
    entering under the camera) has a false foot point and a false velocity."""
    b = k.track.box
    return ((b[:, 0] > margin) & (b[:, 1] > margin)
            & (b[:, 2] < scene.CANON[0] - margin) & (b[:, 3] < scene.CANON[1] - margin))


def _finish(intervals: list[Interval], duration: float, max_gap: float, min_len: float) -> list[Interval]:
    return drop_short(clip(close_gaps(union(intervals), max_gap), duration), min_len)


# --------------------------------------------------------------------------- pedestrians

def jaywalking(ctx: Context) -> list[Interval]:
    out = []
    for k in ctx.of("person"):
        on_road = scene.on_road(k.foot) & ~scene.on_crosswalk(k.foot, margin_px=0.6 * k.scale)
        # faster than ~4 m/s is a rider on something COCO has no class for (e-scooter), not a pedestrian
        on_foot = k.rel_speed < 2.5
        deep = on_road & on_foot & (scene.road_depth(k.foot) > 0.45 * k.scale)
        found = _grown_runs(k, on_road, deep, min_strong_s=1.2)
        for s, e in found:
            ctx.blame(k, s, e)
        out += found
    return _finish(out, ctx.duration, max_gap=2.0, min_len=1.2)  # stepping over the median is one crossing


def failure_to_yield(ctx: Context) -> list[Interval]:
    peds = ctx.of("person")
    ped_cw = {p.tid: scene.crosswalk_of(p.foot) for p in peds}
    out = []
    for k in ctx.of(*VEHICLES):
        cw = np.where(_clear(k), scene.crosswalk_of(k.foot), -1)
        for c in np.unique(cw[cw >= 0]):
            for i, j in runs(cw == c):
                t0, t1 = k.t[i], k.t[j]
                min_speed = 1.2 if k.kind == "two_wheeler" else 0.6  # a pushed bike is a pedestrian
                if t1 - t0 < 0.2 or k.rel_speed[i:j + 1].mean() < min_speed:
                    continue  # stopped on the zebra is not "driving through"
                if _pedestrian_in_path(k, i, j, c, peds, ped_cw):
                    out.append((float(t0), float(t1)))
                    ctx.blame(k, t0, t1)
    return _finish(out, ctx.duration, max_gap=0.3, min_len=0.4)


def _pedestrian_in_path(k: Kin, i: int, j: int, c: int, peds: list[Kin], ped_cw: dict) -> bool:
    for p in peds:
        sel = np.flatnonzero((p.t >= k.t[i] - 0.5) & (p.t <= k.t[j]) & (ped_cw[p.tid] == c))
        for s in sel:
            ki = k.at(p.t[s])
            if np.linalg.norm(p.foot[s] - k.foot[ki]) < 3.0 * k.scale[ki]:
                return True
    return False


# --------------------------------------------------------------------------- signal violations

def red_light(ctx: Context) -> list[Interval]:
    out = []
    for k in ctx.of(*VEHICLES):
        up = signals.distance_upstream(k.foot)
        inbound = scene.points_in_poly(k.foot, scene.INBOUND.poly)
        for n in range(1, len(k.t)):
            if not (up[n - 1] > 0 >= up[n] and inbound[max(0, n - 3):n].any()):
                continue
            along = float(np.dot(k.vel[n], scene.INBOUND.direction)) / k.scale[n]
            if along < 0.8:
                continue
            tc = float(k.t[n - 1] + (k.t[n] - k.t[n - 1]) * up[n - 1] / max(up[n - 1] - up[n], 1e-6))
            # another vehicle of the same phase group must be waiting at the line both before
            # and after the crossing (the median lane moves on its own arrow)
            group = int(signals.phase_group(k.foot[n])[0])
            if not (signals.is_red(tc - 0.5, ctx.reds, k.tid, group) and signals.is_red(tc + 1.5, ctx.reds, k.tid, group)):
                continue
            after = np.flatnonzero(k.t >= tc)
            if len(after) == 0:
                continue
            inside = scene.points_in_poly(k.foot[after], scene.JUNCTION) | scene.points_in_poly(
                k.foot[after], scene.STOP_BOX)
            leave = after[np.argmin(inside)] if (~inside).any() else after[-1]
            out.append((tc, max(float(k.t[leave]), tc + 0.5)))
            ctx.blame(k, *out[-1])
    return _finish(out, ctx.duration, max_gap=0.0, min_len=0.5)


def stop_line(ctx: Context) -> list[Interval]:
    """Stopped past the stop line on red. A vehicle held there by a stationary vehicle
    right ahead is spill-back from a blocked junction, not a stop-line violation."""
    out = []
    vehicles = ctx.of(*VEHICLES)
    for k in vehicles:
        up = signals.distance_upstream(k.foot)
        past = (up < -0.15 * k.scale) & scene.points_in_poly(k.foot, scene.STOP_BOX)
        for i, j in runs(k.still() & past):
            if k.t[j] - k.t[i] < 3.0:
                continue
            tm = 0.5 * (k.t[i] + k.t[j])
            group = int(signals.phase_group(k.foot[i])[0])
            if not signals.is_red(tm, ctx.reds, k.tid, group) or _blocked_ahead(k, i, vehicles):
                continue
            out.append((float(k.t[i]), float(k.t[j])))
            ctx.blame(k, *out[-1])
    return _finish(out, ctx.duration, max_gap=1.0, min_len=3.0)


def _blocked_ahead(k: Kin, n: int, vehicles: list[Kin]) -> bool:
    """A stationary vehicle within 2.5 box heights ahead, in the same lane, when k stops."""
    d = scene.INBOUND.direction
    for o in vehicles:
        if o is k or not (o.t[0] <= k.t[n] <= o.t[-1]):
            continue
        m = o.at(k.t[n])
        rel = o.foot[m] - k.foot[n]
        ahead, lateral = float(rel @ d), abs(float(d[0] * rel[1] - d[1] * rel[0]))
        if 0.3 * k.scale[n] < ahead < 2.5 * k.scale[n] and lateral < 0.5 * k.scale[n] and o.still()[m]:
            return True
    return False


# --------------------------------------------------------------------------- vehicle behaviour

def wrong_way(ctx: Context) -> list[Interval]:
    out = []
    for k in ctx.of(*VEHICLES):
        zone = scene.one_way_zone(k.foot)
        if (zone < 0).all():
            continue
        dirs = np.stack([scene.ONE_WAY[z].direction if z >= 0 else np.zeros(2) for z in zone])
        cos = (k.heading() * dirs).sum(axis=1)
        moving = k.rel_speed > 0.35
        against = (zone >= 0) & moving & (cos < -0.5)
        for i, j in runs(against):
            dist = np.linalg.norm(k.foot[j] - k.foot[i]) / np.median(k.scale[i:j + 1])
            if k.t[j] - k.t[i] >= 1.5 and dist >= 2.5:
                # segment: entering the opposing carriageway -> leaving it
                in_zone = zone >= 0
                a = i
                while a > 0 and in_zone[a - 1]:
                    a -= 1
                b = j
                while b < len(zone) - 1 and in_zone[b + 1]:
                    b += 1
                out.append((float(k.t[a]), float(k.t[b])))
                ctx.blame(k, *out[-1])
    return _finish(out, ctx.duration, max_gap=1.0, min_len=1.5)


def illegal_u_turn(ctx: Context) -> list[Interval]:
    """Heading reversal while clearly moving. Headings come only from boxes clear of the
    frame border (a box cut by the edge drags the foot point sideways) and from samples
    faster than 0.8 bh/s (a creeping car in a queue has a jittery heading), and the turn
    itself must take at most 12 s at a real driving speed."""
    out = []
    for k in ctx.of(*VEHICLES):
        moving = (k.rel_speed > 0.8) & _clear(k)
        if moving.sum() < 8 or k.t[-1] - k.t[0] < 3.0:
            continue
        idx = np.flatnonzero(moving)
        steps = np.linalg.norm(np.diff(k.foot[idx], axis=0), axis=1) / k.scale[idx[1:]]
        if (steps > 2.0).any():
            continue  # a jump means an identity switch, not a manoeuvre
        ang = np.unwrap(np.arctan2(k.vel[idx, 1], k.vel[idx, 0]))
        turn = ang - ang[0]
        total = turn[-1]
        if abs(total) < np.deg2rad(150):
            continue
        h0, h1 = k.heading()[idx[0]], k.heading()[idx[-1]]
        if np.dot(h0, h1) > -0.7:
            continue
        sgn = np.sign(total)
        start = idx[np.argmax(sgn * turn > np.deg2rad(20))]
        end = idx[np.argmax(sgn * turn > abs(total) - np.deg2rad(20))]
        if k.t[end] - k.t[start] > 12.0 or np.median(k.rel_speed[start:end + 1]) < 0.8:
            continue
        if scene.on_road(k.foot[start:end + 1]).mean() > 0.8:
            out.append((float(k.t[start]), float(k.t[end])))
            ctx.blame(k, *out[-1])
    return _finish(out, ctx.duration, max_gap=0.5, min_len=1.5)


def solid_line_crossing(ctx: Context) -> list[Interval]:
    out = []
    for k in ctx.of(*VEHICLES):
        for line in scene.SOLID_LINES:
            side = scene.signed_side(k.foot, line)
            for n in range(1, len(k.t)):
                p, q = k.foot[n - 1], k.foot[n]
                if np.sign(side[n - 1]) == np.sign(side[n]) or not scene.crosses_polyline(p, q, line):
                    continue
                if k.rel_speed[n] < 0.5:
                    continue
                # "wheel crosses" ~ one box-width before the foot point crosses; "fully in" ~ after
                lat = abs(float(np.dot(k.vel[n], _normal(line)))) + 1e-6
                half_w = 0.5 * (k.track.box[n, 2] - k.track.box[n, 0])
                lead = min(1.5, half_w / lat)
                out.append((float(k.t[n] - lead), float(k.t[n] + lead)))
                ctx.blame(k, *out[-1])
    return _finish(out, ctx.duration, max_gap=0.5, min_len=0.5)


def _normal(line: np.ndarray) -> np.ndarray:
    d = line[-1] - line[0]
    n = np.array([-d[1], d[0]])
    return n / np.linalg.norm(n)


def stopped_vehicle(ctx: Context) -> list[Interval]:
    """Stationary >= 10 s on a road link: outside the signal queue, the junction box
    (turning / yielding traffic), the zebras' approaches, parking and the bus bay,
    and not part of a queue (another vehicle standing close by most of the time).

    Stationary spells are pooled across track fragments at the same spot, since
    a long-parked car often loses and regains its identity.
    """
    vehicles = ctx.of("vehicle")
    by_tid = {k.tid: k for k in vehicles}
    spells = []
    for k in vehicles:
        allowed = (scene.on_road(k.foot) & ~scene.points_in_poly(k.foot, scene.PARKING)
                   & ~scene.points_in_poly(k.foot, scene.QUEUE_ZONE) & ~scene.points_in_poly(k.foot, scene.STOP_BOX)
                   & ~scene.points_in_poly(k.foot, scene.JUNCTION)
                   & ~scene.on_crosswalk(k.foot, margin_px=1.0 * k.scale))
        if k.track.cls == 5:  # buses dwell at the stop by design
            allowed &= ~scene.points_in_poly(k.foot, scene.BUS_BAY)
        for i, j in runs(k.still() & allowed):
            if k.t[j] - k.t[i] >= 2.0:
                spells.append([float(k.t[i]), float(k.t[j]), k.foot[i:j + 1].mean(axis=0),
                               float(np.median(k.scale[i:j + 1])), {k.tid}])
    spells.sort(key=lambda s: s[0])
    merged: list[list] = []
    for s in spells:
        for m in merged:
            if s[0] - m[1] <= 4.0 and np.linalg.norm(s[2] - m[2]) < 0.6 * m[3]:
                m[1] = max(m[1], s[1])
                m[4] |= s[4]
                break
        else:
            merged.append(list(s))
    out = []
    for s, e, pos, scale, tids in merged:
        if e - s >= 10.0 and not _queued(vehicles, tids, s, e, pos, scale):
            out.append((s, ctx.duration if ctx.duration - e < 1.0 else e))
            for tid in tids:
                ctx.blame(by_tid[tid], *out[-1])
    return _finish(out, ctx.duration, max_gap=2.0, min_len=10.0)


def _queued(vehicles: list[Kin], own: set, s: float, e: float, pos: np.ndarray, scale: float) -> bool:
    """True if another vehicle stands within 3 box heights for most of [s, e]
    (a bus dwelling in the bay is not a queue)."""
    grid = np.arange(s, e, 1.0)
    near = np.zeros(len(grid), bool)
    for o in vehicles:
        if o.tid in own or o.t[-1] < s or o.t[0] > e:
            continue
        still = o.still()
        for g, tg in enumerate(grid):
            if not o.t[0] <= tg <= o.t[-1]:
                continue
            n = o.at(tg)
            if not still[n]:
                continue
            if o.track.cls == 5 and scene.points_in_poly(o.foot[n:n + 1], scene.BUS_BAY)[0]:
                continue
            d = np.linalg.norm(o.foot[n] - pos)
            # a second box on the very same car (car + truck detections) is not a queue
            near[g] |= bool(0.3 * scale < d < 3.0 * scale)
    return near.mean() > 0.5 if len(grid) else False


def congestion(ctx: Context) -> list[Interval]:
    """Standstill/crawl across a whole one-way carriageway for longer than a signal cycle."""
    if not ctx.kins:
        return []
    grid = np.arange(0.0, ctx.duration, 1.0)
    out = []
    for zone, min_len in ((scene.INBOUND, 90.0), (scene.OUTBOUND, 30.0)):
        present = np.zeros(len(grid))
        slow = np.zeros(len(grid))
        for k in ctx.of("vehicle"):
            inside = scene.points_in_poly(k.foot, zone.poly)
            if not inside.any():
                continue
            gi = np.clip(np.searchsorted(grid, k.t[inside], side="right") - 1, 0, len(grid) - 1)
            spd = k.rel_speed[inside]
            for g in np.unique(gi):
                present[g] += 1
                slow[g] += np.median(spd[gi == g]) < 0.4
        jam = (present >= 6) & (slow >= 0.8 * np.maximum(present, 1))
        for i, j in runs(jam):
            if grid[j] - grid[i] >= min_len:
                out.append((float(grid[i]), float(grid[j] + 1.0)))
    return _finish(out, ctx.duration, max_gap=5.0, min_len=30.0)


# --------------------------------------------------------------------------- conflicts

def _footprint(k: Kin, n: int) -> np.ndarray:
    """Lower part of the box: the image region closest to the object's ground contact."""
    x1, y1, x2, y2 = k.track.box[n]
    return np.array([x1, y2 - 0.4 * (y2 - y1), x2, y2])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def accident(ctx: Context) -> list[Interval]:
    """Contact followed by an abrupt stop of the striker, outside the signal queue.

    A rear-end arrival in a queue also produces touching boxes, so the striker
    must lose > 2.5 bh/s within a second of contact, outside the queue zone, and
    it (and a struck vehicle) must then stay still for 3 s. The segment runs from
    first contact to the moment the striker has stopped.
    """
    users = ctx.of(*VEHICLES, "person")
    out = []
    for (i, j), times in _contacts(users).items():
        a, b = users[i], users[j]
        onsets = [t for n, t in enumerate(times) if n == 0 or t - times[n - 1] > 0.5]  # a collision starts at contact
        for tc in onsets:
            hit = _collision_end(a, b, tc) or _collision_end(b, a, tc)
            if hit is not None:
                out.append((tc, hit))
                ctx.blame(a, tc, hit)
                ctx.blame(b, tc, hit)
                break
    return _finish(out, ctx.duration, max_gap=2.0, min_len=0.5)


def _contacts(users: list[Kin], step: float = 0.2, thr: float = 0.2) -> dict[tuple[int, int], list[float]]:
    """Times at which two road users' footprints overlap (IoU >= thr), for pairs
    that include at least one vehicle. One vectorised IoU matrix per time step
    keeps this linear in video length even in crowded scenes."""
    if len(users) < 2:
        return {}
    t0 = min(k.t[0] for k in users)
    t1 = max(k.t[-1] for k in users)
    n_steps = int((t1 - t0) / step) + 1
    grid = t0 + step * np.arange(n_steps)
    members: list[list[int]] = [[] for _ in range(n_steps)]
    boxes: list[list[np.ndarray]] = [[] for _ in range(n_steps)]
    for u, k in enumerate(users):
        g0, g1 = int(np.ceil((k.t[0] - t0) / step)), int(np.floor((k.t[-1] - t0) / step))
        if g1 < g0:
            continue
        fps = _footprints(k)[_idx(k, grid[g0:g1 + 1])]
        for g, f in zip(range(g0, g1 + 1), fps):
            members[g].append(u)
            boxes[g].append(f)
    mover = np.array([k.kind in VEHICLES for k in users])
    out: dict[tuple[int, int], list[float]] = {}
    for g in range(n_steps):
        if len(members[g]) < 2:
            continue
        ids = np.asarray(members[g])
        f = np.stack(boxes[g])
        ix = np.clip(np.minimum(f[:, None, 2], f[None, :, 2]) - np.maximum(f[:, None, 0], f[None, :, 0]), 0, None)
        iy = np.clip(np.minimum(f[:, None, 3], f[None, :, 3]) - np.maximum(f[:, None, 1], f[None, :, 1]), 0, None)
        area = (f[:, 2] - f[:, 0]) * (f[:, 3] - f[:, 1])
        iou = ix * iy / np.maximum(area[:, None] + area[None, :] - ix * iy, 1e-9)
        ii, jj = np.nonzero(np.triu(iou >= thr, k=1))
        for a, b in zip(ids[ii], ids[jj]):
            if mover[a] or mover[b]:
                out.setdefault((int(a), int(b)), []).append(float(grid[g]))
    return out


def _collision_end(striker: Kin, struck: Kin, tc: float) -> float | None:
    if striker.kind not in VEHICLES:
        return None
    # a track that simply ends (occluded by a bus) is no evidence of a stop
    if striker.t[0] > tc - 1.0 or striker.t[-1] < tc + 4.0:
        return None
    if struck.kind in VEHICLES and struck.t[-1] < tc + 4.0:
        return None
    n = striker.at(tc)
    if scene.points_in_poly(striker.foot[n:n + 1], scene.QUEUE_ZONE)[0]:
        return None
    before, after = striker.rel_speed[striker.at(tc - 1.0)], striker.rel_speed[striker.at(tc + 1.0)]
    if before < 2.5 or before - after < 2.5:
        return None
    a0, a1 = striker.at(tc + 1.0), striker.at(tc + 4.0)
    if not striker.still()[a0:a1 + 1].all():
        return None
    if struck.kind in VEHICLES and not struck.still()[struck.at(tc + 1.0):struck.at(tc + 4.0) + 1].all():
        return None
    stopped = np.flatnonzero(striker.still()[n:a0 + 1])
    return float(striker.t[n + stopped[0]]) if len(stopped) and striker.t[n + stopped[0]] > tc + 0.3 else tc + 1.0


def _footprints(k: Kin) -> np.ndarray:
    """Lower 40% of every box: the image region closest to the ground contact, (N, 4)."""
    b = k.track.box
    return np.stack([b[:, 0], b[:, 3] - 0.4 * (b[:, 3] - b[:, 1]), b[:, 2], b[:, 3]], axis=1)


def _idx(k: Kin, times: np.ndarray) -> np.ndarray:
    """Nearest sample index for each time."""
    hi = np.clip(np.searchsorted(k.t, times), 0, len(k.t) - 1)
    lo = np.clip(hi - 1, 0, len(k.t) - 1)
    return np.where(np.abs(k.t[lo] - times) <= np.abs(k.t[hi] - times), lo, hi)


def near_miss(ctx: Context) -> list[Interval]:
    """Hard braking of a moving vehicle with another road user close ahead and no contact.

    The braking box must be inside the frame (a box shrinking at the edge reads as a
    stop), and the road user ahead must be a pedestrian or moving: slowing down behind
    a parked or queued car is ordinary driving."""
    users = [k for k in ctx.kins if k.kind in (*VEHICLES, "person")]
    out = []
    for k in ctx.of(*VEHICLES):
        hard = (k.accel < -3.0) & (k.rel_speed > 0.5) & _clear(k)
        for i, j in runs(hard):
            v0 = k.rel_speed[max(0, i - 3)]
            if v0 < 2.0 or k.rel_speed[j] > 0.5 * v0:
                continue
            h = k.heading()[max(0, i - 3)]
            for o in users:
                if o is k or not (o.t[0] <= k.t[i] <= o.t[-1]):
                    continue
                oi = o.at(k.t[i])
                if o.kind in VEHICLES and o.still()[oi]:
                    continue
                rel = o.foot[oi] - k.foot[i]
                ahead = float(np.dot(rel, h))
                side = abs(float(h[0] * rel[1] - h[1] * rel[0]))
                if not (0 < ahead < 2.5 * k.scale[i] and side < 1.0 * k.scale[i]):
                    continue
                contact = any(_iou(_footprint(k, k.at(t)), _footprint(o, o.at(t))) > 0.2
                              for t in np.arange(k.t[i], min(k.t[j] + 2.0, o.t[-1], k.t[-1]), 0.2))
                if not contact:
                    out.append((float(k.t[max(0, i - 2)]), float(k.t[j] + 1.0)))
                    ctx.blame(k, *out[-1])
                    ctx.blame(o, *out[-1])
                    break
    return _finish(out, ctx.duration, max_gap=1.0, min_len=1.0)


def road_obstacle(ctx: Context) -> list[Interval]:
    out = []
    for k in ctx.of("animal"):
        on = scene.on_road(k.foot)
        found = _grown_runs(k, on, on, min_strong_s=2.0)
        for s, e in found:
            ctx.blame(k, s, e)
        out += found
    return _finish(out, ctx.duration, max_gap=2.0, min_len=2.0)


RULES = {
    "accident": accident,
    "near_miss": near_miss,
    "red_light": red_light,
    "wrong_way": wrong_way,
    "illegal_u_turn": illegal_u_turn,
    "stopped_vehicle": stopped_vehicle,
    "jaywalking": jaywalking,
    "failure_to_yield": failure_to_yield,
    "solid_line_crossing": solid_line_crossing,
    "stop_line": stop_line,
    "congestion": congestion,
    "road_obstacle": road_obstacle,
}


def run_rules(ctx: Context, enabled: tuple[str, ...] | None = None) -> list[list]:
    """All enabled rules; a rule that raises loses only its own class."""
    events = []
    for label, rule in RULES.items():
        if enabled is not None and label not in enabled:
            continue
        ctx.label = label
        try:
            found = rule(ctx)
        except Exception:  # noqa: BLE001 - isolate faults per class
            traceback.print_exc(file=sys.stderr)
            continue
        for s, e in found:
            events.append([round(s, 2), round(e, 2), label])
    return sorted(events)
