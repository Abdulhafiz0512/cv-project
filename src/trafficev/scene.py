"""Hand-measured layout of the fixed camera view (the rules' "camera.md").

The view is a T-junction seen from an elevated position looking north-west:
- west leg: a divided main road. The NEAR carriageway (below the median)
  carries inbound traffic towards the camera and ends at a stop line followed
  by the west zebra; the FAR carriageway (above the median) carries outbound
  traffic away from the camera and has a bus bay by the bus stop;
- east leg: the main road continues to the right edge, with the east zebra;
- south leg: a side road under the camera with three channelising islands
  and a long diagonal zebra.
Traffic keeps right. Polygons are measured on the C3896 4K frame (the base
view) in normalised coordinates and stored in a canonical 1920x1080 space
(see tracking.py). Recordings are framed slightly differently, so every video
is registered to the base view and ``set_view`` warps this layout into it
(see view.py). Vertices on the image border sit just outside it (-0.02 /
1.02) so a warp cannot pull a road edge into the picture.
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
    (-0.020, 0.077), (0.0661, 0.0696), (0.3326, 0.1146), (0.4604, 0.1739), (0.6177, 0.2491), (0.6569, 0.2802),
    (0.6960, 0.3212), (0.7350, 0.3721), (0.7838, 0.4233), (0.8428, 0.4453), (0.9266, 0.4535), (0.9560, 0.4714),
    (1.020, 0.4806), (1.020, 1.020), (-0.020, 1.020),
    (-0.020, 0.7487), (0.1000, 0.6735), (0.1698, 0.6119), (0.1899, 0.5781), (0.1511, 0.4976), (0.1028, 0.3772),
    (0.0548, 0.2272), (0.0259, 0.1372), (-0.020, 0.1167),
)
MEDIAN = _poly((0.0560, 0.0909), (0.6207, 0.4585), (0.6204, 0.4910), (0.5908, 0.4899), (0.0558, 0.1166))
_SE_ISLAND_TIP = (0.4990, 0.8111)
ISLANDS = [
    MEDIAN,
    _poly((0.6154, 0.5007), (0.6844, 0.5033), (0.6842, 0.5329), (0.6151, 0.5303)),                    # keep-right sign
    _poly((0.0637, 0.8500), (0.1633, 0.7747), (0.2122, 0.8358), (0.1971, 0.8600)),                    # south-west
    _poly((0.2628, 0.7044), (0.3278, 0.6229), (0.3915, 0.6846), (0.3765, 0.6989)),                    # south, top
    _poly((0.3555, 0.7778), (0.4375, 0.7519), (0.4583, 0.7519), _SE_ISLAND_TIP, (0.4922, 0.8222),
          (0.3604, 0.8306)),                                                                          # south-east
]

# Zebras fitted to the stripe ends on a median background of C3896. The south
# zebra is fan-shaped (stripes lengthen towards the camera), hence its outline.
_CW_W_NEAR_R, _CW_W_NEAR_L = (0.6016, 0.5131), (0.1773, 0.6056)
_CW_E_NEAR_R = (0.9349, 0.4681)
CROSSWALKS = {
    "west": _poly((0.1749, 0.5611), (0.6000, 0.4811), _CW_W_NEAR_R, _CW_W_NEAR_L),
    "east": _poly((0.6198, 0.4731), (0.9323, 0.4454), _CW_E_NEAR_R, (0.6198, 0.4998)),
    "south": _poly(
        (0.0911, 0.7009), (0.1719, 0.6593), (0.2135, 0.6963), (0.2526, 0.7343), (0.3333, 0.8009),
        (0.3750, 0.8407), (0.4062, 0.8796), (0.4349, 0.9333), (0.4635, 0.9926), (0.4714, 1.0185),
        (0.3240, 1.0185), (0.3187, 0.9861), (0.3021, 0.9444), (0.2839, 0.9000), (0.2682, 0.8630),
        (0.2302, 0.8287), (0.1849, 0.7917), (0.1458, 0.7546), (0.1094, 0.7222)),
}

# Stop line of the inbound approach, fitted to the paint; vehicles cross it along INBOUND.direction.
_SL_L, _SL_R = (0.1484, 0.4862), (0.4833, 0.4237)
STOP_LINE = _poly(_SL_L, _SL_R)

# One-way carriageways of the divided west leg (wrong-way reference).
INBOUND = Zone("near_inbound", _poly(
    (-0.020, 0.1117), (0.0558, 0.1186), (0.4976, 0.4201), _SL_R, _SL_L, (0.1028, 0.3772),
    (0.0548, 0.2272), (0.0259, 0.1372)), _unit(0.0242, 0.0159))
OUTBOUND = Zone("far_outbound", _poly(
    (-0.020, 0.0772), (0.0661, 0.0696), (0.3326, 0.1146), (0.4604, 0.1739), (0.6177, 0.2491), (0.6569, 0.2802),
    (0.6962, 0.3015), (0.6207, 0.4585), (0.0560, 0.0909)), _unit(-0.0242, -0.0159))
ONE_WAY = [INBOUND, OUTBOUND]

# Where inbound vehicles wait for the signal (a queue here is not "stopped_vehicle").
QUEUE_ZONE = _poly((0.0837, 0.3073), (0.3013, 0.2862), (0.4976, 0.4201), _SL_R, _SL_L)
# Between the stop line and the far edge of the west zebra: stopping here on red is "stop_line".
STOP_BOX = _poly(_SL_L, _SL_R, _CW_W_NEAR_R, _CW_W_NEAR_L)
# Junction box between the stop line / zebras.
JUNCTION = _poly(_SL_L, _SL_R, (0.6207, 0.4585), _CW_E_NEAR_R, (1.020, 0.5299),
                 (1.020, 0.7074), _SE_ISLAND_TIP, _CW_W_NEAR_L)
BUS_BAY = _poly((0.3031, 0.0986), (0.5000, 0.1606), (0.4994, 0.2248), (0.3025, 0.1598))
PARKING = _poly((-0.020, 0.2746), (0.1034, 0.3081), (0.1864, 0.4200), (0.1857, 0.4990), (-0.020, 0.5117))
# Solid markings: the yellow edge line along the median (crossing it = entering the median / oncoming side).
SOLID_LINES = [
    _poly((0.0558, 0.1206), (0.3013, 0.2862), (0.5859, 0.4828)),
]
# Signal heads visible from the camera (lamp ROI boxes x0 y0 x1 y1), used for phase estimation.
SIGNAL_HEADS = {
    "pole_west": _poly((0.1315, 0.4425), (0.1467, 0.4826)),
    "median_nose": _poly((0.5848, 0.3219), (0.6091, 0.3919)),
}

# Bump when any geometry above changes: it keys the perception cache.
LAYOUT_VERSION = "c3896-base-1"


# --------------------------------------------------------------------------- per-video view

def _warp(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return hom[:, :2] / hom[:, 2:3]


def _warp_zone(z: Zone, H: np.ndarray) -> Zone:
    c = z.poly.mean(axis=0)
    a, b = _warp(np.array([c, c + 20.0 * z.direction]), H)
    return Zone(z.name, _warp(z.poly, H), (b - a) / np.linalg.norm(b - a))


def _warp_box(box: np.ndarray, H: np.ndarray) -> np.ndarray:
    (x0, y0), (x1, y1) = box
    c = _warp(np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]]), H)
    return np.array([c.min(axis=0), c.max(axis=0)])


_BASE = dict(
    ROAD_OUTER=ROAD_OUTER, ISLANDS=ISLANDS, CROSSWALKS=CROSSWALKS, STOP_LINE=STOP_LINE, INBOUND=INBOUND,
    OUTBOUND=OUTBOUND, QUEUE_ZONE=QUEUE_ZONE, STOP_BOX=STOP_BOX, JUNCTION=JUNCTION, BUS_BAY=BUS_BAY,
    PARKING=PARKING, SOLID_LINES=SOLID_LINES, SIGNAL_HEADS=SIGNAL_HEADS,
)
VIEW_H = np.eye(3)   # base layout -> current video, canonical px


def set_view(H: np.ndarray | None) -> None:
    """Warp the base layout into a video's view (None or identity = base view)."""
    global ROAD_OUTER, MEDIAN, ISLANDS, CROSSWALKS, STOP_LINE, INBOUND, OUTBOUND, ONE_WAY
    global QUEUE_ZONE, STOP_BOX, JUNCTION, BUS_BAY, PARKING, SOLID_LINES, SIGNAL_HEADS, VIEW_H
    H = np.eye(3) if H is None else np.asarray(H, dtype=np.float64)
    if np.allclose(H, VIEW_H):
        return
    b = _BASE
    ROAD_OUTER = _warp(b["ROAD_OUTER"], H)
    ISLANDS = [_warp(p, H) for p in b["ISLANDS"]]
    MEDIAN = ISLANDS[0]
    CROSSWALKS = {k: _warp(p, H) for k, p in b["CROSSWALKS"].items()}
    STOP_LINE = _warp(b["STOP_LINE"], H)
    INBOUND, OUTBOUND = _warp_zone(b["INBOUND"], H), _warp_zone(b["OUTBOUND"], H)
    ONE_WAY = [INBOUND, OUTBOUND]
    QUEUE_ZONE, STOP_BOX, JUNCTION = _warp(b["QUEUE_ZONE"], H), _warp(b["STOP_BOX"], H), _warp(b["JUNCTION"], H)
    BUS_BAY, PARKING = _warp(b["BUS_BAY"], H), _warp(b["PARKING"], H)
    SOLID_LINES = [_warp(p, H) for p in b["SOLID_LINES"]]
    SIGNAL_HEADS = {k: _warp_box(p, H) for k, p in b["SIGNAL_HEADS"].items()}
    VIEW_H = H.copy()


def reset_view() -> None:
    set_view(None)


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
