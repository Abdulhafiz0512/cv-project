"""Hand-measured layout of the fixed camera view (the rules' "camera.md").

The view is a T-junction seen from an elevated position looking north-west:
- west leg: a divided main road. The NEAR carriageway (below the median)
  carries inbound traffic towards the camera and ends at a stop line followed
  by the west zebra; the FAR carriageway (above the median) carries outbound
  traffic away from the camera and has a bus bay by the bus stop;
- east leg: the main road continues to the right edge, with the east zebra;
- south leg: a side road under the camera with three channelising islands
  and a long diagonal zebra.
Traffic keeps right. Polygons are measured on a 4K frame in normalised
coordinates and stored in a canonical 1920x1080 space (see tracking.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CANON = np.array([1920.0, 1080.0])


def _poly(*pts: tuple[float, float]) -> np.ndarray:
    return np.asarray(pts, dtype=np.float64) * CANON


def _unit(dx: float, dy: float) -> np.ndarray:
    v = np.array([dx, dy]) * CANON
    return v / np.linalg.norm(v)


@dataclass(frozen=True)
class Zone:
    name: str
    poly: np.ndarray                     # (K, 2) canonical px
    direction: np.ndarray | None = None  # legal travel direction (unit), if one-way


# Drivable surface: outer boundary; holes (median, islands) are removed below.
ROAD_OUTER = _poly(
    (0.000, 0.100), (0.060, 0.090), (0.330, 0.125), (0.460, 0.180), (0.620, 0.250), (0.660, 0.280),
    (0.700, 0.320), (0.740, 0.370), (0.790, 0.420), (0.850, 0.440), (0.935, 0.445), (0.965, 0.462),
    (1.000, 0.470), (1.000, 1.000), (0.000, 1.000),
    (0.000, 0.780), (0.100, 0.700), (0.170, 0.635), (0.190, 0.600), (0.150, 0.520), (0.100, 0.400),
    (0.050, 0.250), (0.020, 0.160), (0.000, 0.140),
)
MEDIAN = _poly((0.050, 0.112), (0.625, 0.462), (0.625, 0.495), (0.595, 0.495), (0.050, 0.138))
ISLANDS = [
    MEDIAN,
    _poly((0.620, 0.505), (0.690, 0.505), (0.690, 0.535), (0.620, 0.535)),              # keep-right sign
    _poly((0.065, 0.880), (0.165, 0.800), (0.215, 0.860), (0.200, 0.885)),              # south-west
    _poly((0.265, 0.725), (0.330, 0.640), (0.395, 0.700), (0.380, 0.715)),              # south, top
    _poly((0.360, 0.840), (0.440, 0.790), (0.485, 0.790), (0.500, 0.845), (0.400, 0.860)),  # south-east
]

CROSSWALKS = {
    "west": _poly((0.170, 0.585), (0.600, 0.495), (0.625, 0.530), (0.190, 0.640)),
    "east": _poly((0.640, 0.485), (0.905, 0.435), (0.935, 0.465), (0.665, 0.525)),
    "south": _poly((0.090, 0.720), (0.175, 0.700), (0.470, 1.000), (0.330, 1.000)),
}

# One-way carriageways of the divided west leg (wrong-way reference).
INBOUND = Zone("near_inbound", _poly(
    (0.000, 0.135), (0.050, 0.140), (0.500, 0.428), (0.485, 0.445), (0.150, 0.515), (0.100, 0.400),
    (0.050, 0.250), (0.020, 0.160)), _unit(0.575, 0.355))
OUTBOUND = Zone("far_outbound", _poly(
    (0.000, 0.100), (0.060, 0.090), (0.330, 0.125), (0.460, 0.180), (0.620, 0.250), (0.660, 0.280),
    (0.700, 0.300), (0.625, 0.462), (0.050, 0.112)), _unit(-0.575, -0.355))
ONE_WAY = [INBOUND, OUTBOUND]

# Stop line of the inbound approach; vehicles cross it moving along INBOUND.direction.
STOP_LINE = _poly((0.150, 0.515), (0.485, 0.445))
# Where inbound vehicles wait for the signal (a queue here is not "stopped_vehicle").
QUEUE_ZONE = _poly((0.080, 0.330), (0.300, 0.300), (0.500, 0.428), (0.485, 0.445), (0.150, 0.515))
# Between the stop line and the far edge of the west zebra: stopping here on red is "stop_line".
STOP_BOX = _poly((0.150, 0.515), (0.485, 0.445), (0.610, 0.510), (0.190, 0.640))
# Junction box between the stop line / zebras.
JUNCTION = _poly((0.150, 0.515), (0.485, 0.445), (0.625, 0.462), (0.935, 0.465), (1.000, 0.520),
                 (1.000, 0.700), (0.500, 0.845), (0.190, 0.640))
BUS_BAY = _poly((0.300, 0.110), (0.500, 0.165), (0.500, 0.230), (0.300, 0.172))
PARKING = _poly((0.000, 0.300), (0.100, 0.330), (0.185, 0.440), (0.185, 0.520), (0.000, 0.540))
# Solid markings: the yellow edge line along the median (crossing it = entering the median / oncoming side).
SOLID_LINES = [
    _poly((0.050, 0.142), (0.300, 0.300), (0.590, 0.488)),
]
# Signal heads visible from the camera (lamp ROI boxes x0 y0 x1 y1), used for phase estimation.
SIGNAL_HEADS = {
    "pole_west": _poly((0.130, 0.465), (0.145, 0.505)),
    "median_nose": _poly((0.588, 0.325), (0.612, 0.395)),
}


def points_in_poly(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd point-in-polygon test; pts (N, 2) -> (N,) bool."""
    pts = np.atleast_2d(pts)
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = poly[:, 0][None, :], poly[:, 1][None, :]
    x2, y2 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    cond = (y1 > y) != (y2 > y)
    xint = x1 + (y - y1) * (x2 - x1) / np.where(y2 == y1, 1e-9, y2 - y1)
    return (cond & (x < xint)).sum(axis=1) % 2 == 1


def on_road(pts: np.ndarray) -> np.ndarray:
    inside = points_in_poly(pts, ROAD_OUTER)
    for hole in ISLANDS:
        inside &= ~points_in_poly(pts, hole)
    return inside


def on_crosswalk(pts: np.ndarray, margin_px: float | np.ndarray = 0.0) -> np.ndarray:
    """On a zebra, or within `margin_px` of one (people cross a metre beside the paint)."""
    pts = np.atleast_2d(pts)
    hit = np.zeros(len(pts), bool)
    for poly in CROSSWALKS.values():
        hit |= points_in_poly(pts, poly)
        if np.any(np.asarray(margin_px) > 0):
            hit |= edge_distance(pts, poly) <= margin_px
    return hit


def edge_distance(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest polygon edge (px)."""
    pts = np.atleast_2d(pts)
    a = poly
    ab = np.roll(poly, -1, axis=0) - a
    ap = pts[:, None, :] - a[None, :, :]
    tt = np.clip((ap * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9)[None], 0, 1)
    proj = a[None] + tt[..., None] * ab[None]
    return np.linalg.norm(pts[:, None, :] - proj, axis=-1).min(axis=1)


def road_depth(pts: np.ndarray) -> np.ndarray:
    """How far inside the carriageway a point is, in px (0 when off-road)."""
    pts = np.atleast_2d(pts)
    d = edge_distance(pts, ROAD_OUTER)
    for hole in ISLANDS:
        d = np.minimum(d, edge_distance(pts, hole))
    return np.where(on_road(pts), d, 0.0)


def crosswalk_of(pts: np.ndarray) -> np.ndarray:
    """Index of the crosswalk each point is on (-1 if none)."""
    out = np.full(len(np.atleast_2d(pts)), -1)
    for i, poly in enumerate(CROSSWALKS.values()):
        out[points_in_poly(pts, poly) & (out < 0)] = i
    return out


def signed_side(pts: np.ndarray, line: np.ndarray) -> np.ndarray:
    """>0 on the left of the directed line a->b (image coords, y down)."""
    a, b = line[0], line[-1]
    d = b - a
    rel = np.atleast_2d(pts) - a
    return d[0] * rel[:, 1] - d[1] * rel[:, 0]


def crosses_polyline(p: np.ndarray, q: np.ndarray, line: np.ndarray) -> bool:
    """Does segment p->q intersect any piece of the polyline?"""
    for a, b in zip(line[:-1], line[1:]):
        if _seg_intersect(p, q, a, b):
            return True
    return False


def _seg_intersect(p, q, a, b) -> bool:
    def orient(u, v, w):
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
    d1, d2 = orient(a, b, p), orient(a, b, q)
    d3, d4 = orient(p, q, a), orient(p, q, b)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def one_way_zone(pts: np.ndarray) -> np.ndarray:
    """Index into ONE_WAY for each point, -1 outside every one-way carriageway."""
    out = np.full(len(np.atleast_2d(pts)), -1)
    for i, z in enumerate(ONE_WAY):
        out[points_in_poly(pts, z.poly) & (out < 0)] = i
    return out
