"""Per-video camera view: register a frame to the reference views, warp the layout.

The organizers' recordings share the camera but not the exact framing (C3896
is zoomed and panned relative to C3905, which moves the stop line by ~25 px).
The layout in ``scene.py`` is measured once in the base view; every video is
registered to the closest reference image (SIFT on static structure, MAGSAC
homography) and the layout is warped into that video's pixels.

A frame that cannot be registered falls back to the base layout unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import scene
from .scene import CANON

ASSETS = Path(__file__).resolve().parent / "assets"
REG_W, REG_H = 1280, 720          # registration resolution
MIN_INLIERS = 60
# Reference images (median backgrounds) and the homography from the base
# (layout) view into each, in canonical 1920x1080 px. C3896 is the base view;
# C3905 (dusk, zoomed out) was registered to it with 172 inlier matches.
REFERENCES: dict[str, np.ndarray] = {
    "ref_C3896.jpg": np.eye(3),
    "ref_C3905.jpg": np.array([
        [1.01118332, 0.01699225, -14.4904878],
        [-0.02240763, 1.01249499, 23.92124681],
        [-0.00000086, 0.00000035, 1.0],
    ]),
}


@dataclass(frozen=True)
class ViewFit:
    H: np.ndarray        # base layout -> this video, canonical px
    reference: str
    inliers: int

    @property
    def registered(self) -> bool:
        return self.reference != ""


IDENTITY = ViewFit(np.eye(3), "", 0)

_S = np.diag([REG_W / CANON[0], REG_H / CANON[1], 1.0])    # canonical -> registration px


def _gray(frame: np.ndarray) -> np.ndarray:
    g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if g.shape[:2] != (REG_H, REG_W):
        g = cv2.resize(g, (REG_W, REG_H), interpolation=cv2.INTER_AREA)
    # CLAHE evens out day / dusk contrast before matching
    return cv2.createCLAHE(3.0, (8, 8)).apply(g)


def _features(gray: np.ndarray):
    return cv2.SIFT_create(4000).detectAndCompute(gray, None)


@lru_cache(maxsize=None)
def _reference(name: str):
    img = cv2.imread(str(ASSETS / name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(ASSETS / name)
    return _features(_gray(img))


def _plausible(H: np.ndarray) -> bool:
    """Same camera, small zoom / pan: reject degenerate or wild fits."""
    if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-9:
        return False
    A = H[:2, :2] / H[2, 2]
    det = float(np.linalg.det(A))
    return 0.6 < det < 1.7 and np.abs(H[2, :2] / H[2, 2]).max() < 2e-3 * max(REG_W, REG_H) / CANON[0]


def register(frame: np.ndarray) -> ViewFit:
    """Fit the base layout onto one frame (BGR, any size). Deterministic."""
    cv2.setRNGSeed(0)
    kf, df = _features(_gray(frame))
    if df is None or len(kf) < MIN_INLIERS:
        return IDENTITY
    best = IDENTITY
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    for name, H_base_ref in REFERENCES.items():
        kr, dr = _reference(name)
        pairs = matcher.knnMatch(dr, df, k=2)
        good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
        if len(good) < MIN_INLIERS:
            continue
        src = np.float32([kr[m.queryIdx].pt for m in good])
        dst = np.float32([kf[m.trainIdx].pt for m in good])
        H_ref_frame, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 2.0, maxIters=5000, confidence=0.999)
        if H_ref_frame is None:
            continue
        inliers = int(mask.sum())
        if inliers < MIN_INLIERS or inliers <= best.inliers or not _plausible(H_ref_frame):
            continue
        H = np.linalg.inv(_S) @ H_ref_frame @ _S @ H_base_ref
        best = ViewFit(H / H[2, 2], name, inliers)
    return best


class ViewLock:
    """Registers the view on a video's first frames and applies it to ``scene``.

    Retries once a second for a few seconds if a frame cannot be matched (a bus
    covering the junction, a dark first frame). Uses only frames already seen,
    so Part B stays causal.
    """
    RETRIES, EVERY_S = 5, 1.0

    def __init__(self) -> None:
        self.fit: ViewFit | None = None
        self.tries, self.next_t = 0, 0.0

    def update(self, t: float, frame: np.ndarray) -> None:
        if (self.fit is not None and self.fit.registered) or self.tries >= self.RETRIES or t < self.next_t:
            return
        self.tries += 1
        self.next_t = t + self.EVERY_S
        fit = register(frame)
        if self.fit is None or fit.registered:
            self.fit = fit
            scene.set_view(fit.H)

    @property
    def H(self) -> np.ndarray:
        return self.fit.H if self.fit is not None else np.eye(3)


def warp_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    flat = pts.reshape(-1, 2)
    hom = np.hstack([flat, np.ones((len(flat), 1))]) @ H.T
    return (hom[:, :2] / hom[:, 2:3]).reshape(pts.shape)
