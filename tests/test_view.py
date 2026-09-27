import cv2
import numpy as np
from helpers import scene  # noqa: F401  (sets sys.path)

from trafficev import view

REF = view.ASSETS / "ref_C3896.jpg"


def _canon_to_px(w: int, h: int) -> np.ndarray:
    return np.diag([w / scene.CANON[0], h / scene.CANON[1], 1.0])


def test_register_recovers_a_zoom_and_pan():
    img = cv2.imread(str(REF))
    h, w = img.shape[:2]
    true = np.array([[1.05, 0.01, -40.0], [-0.01, 1.05, 18.0], [0.0, 0.0, 1.0]])   # canonical px
    S = _canon_to_px(w, h)
    warped = cv2.warpPerspective(img, S @ true @ np.linalg.inv(S), (w, h))
    fit = view.register(warped)
    assert fit.registered and fit.inliers >= view.MIN_INLIERS
    err = view.warp_points(scene.STOP_LINE, fit.H) - view.warp_points(scene.STOP_LINE, true)
    assert np.abs(err).max() < 2.0


def test_register_matches_the_dusk_reference():
    fit = view.register(cv2.imread(str(view.ASSETS / "ref_C3905.jpg")))
    assert fit.registered
    expected = view.warp_points(scene.STOP_LINE, view.REFERENCES["ref_C3905.jpg"])
    assert np.abs(view.warp_points(scene.STOP_LINE, fit.H) - expected).max() < 2.0


def test_unmatchable_frame_keeps_the_base_layout():
    noise = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    fit = view.register(noise)
    assert not fit.registered and np.allclose(fit.H, np.eye(3))


def test_set_view_warps_and_restores_exactly():
    base = scene.STOP_LINE.copy(), scene.CROSSWALKS["west"].copy(), scene.INBOUND.direction.copy()
    H = np.array([[1.02, 0.0, 10.0], [0.0, 1.02, -12.0], [0.0, 0.0, 1.0]])
    scene.set_view(H)
    try:
        assert np.allclose(scene.STOP_LINE, view.warp_points(base[0], H))
        assert scene.on_crosswalk(view.warp_points(base[1].mean(axis=0, keepdims=True), H))[0]
        assert np.isclose(np.linalg.norm(scene.INBOUND.direction), 1.0)
    finally:
        scene.reset_view()
    assert np.array_equal(scene.STOP_LINE, base[0]) and np.array_equal(scene.CROSSWALKS["west"], base[1])
    assert np.allclose(scene.INBOUND.direction, base[2])
