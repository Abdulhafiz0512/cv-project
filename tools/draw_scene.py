"""Render the hand-measured scene layout over a frame (sanity check + website figure).

    python tools/draw_scene.py --frame data/frame_4k_0.jpg --out web/assets/scene_layout.jpg
    python tools/draw_scene.py --frame other_video_frame.jpg --out check.jpg --register

--register first fits the layout to the frame's view (as the pipeline does per video).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trafficev import scene  # noqa: E402

COLORS = {
    "road": (80, 80, 80), "island": (60, 60, 200), "crosswalk": (255, 255, 255), "inbound": (0, 200, 0),
    "outbound": (255, 160, 0), "stop": (0, 0, 255), "queue": (0, 200, 255), "junction": (200, 0, 200),
    "bus": (0, 255, 255), "parking": (160, 160, 255), "solid": (0, 255, 255),
}


def draw(frame: np.ndarray, register: bool = False) -> np.ndarray:
    if register:
        from trafficev import view
        fit = view.register(frame)
        scene.set_view(fit.H)
        print(f"registered to {fit.reference or 'nothing (base layout)'} with {fit.inliers} inliers")
    h, w = frame.shape[:2]
    s = np.array([w / scene.CANON[0], h / scene.CANON[1]])
    img = cv2.convertScaleAbs(frame, alpha=1.5, beta=10)
    over = img.copy()

    def poly(p, color, fill=False, thick=3):
        q = (p * s).astype(np.int32)
        if fill:
            cv2.fillPoly(over, [q], color)
        cv2.polylines(img, [q], True, color, thick, cv2.LINE_AA)

    poly(scene.ROAD_OUTER, COLORS["road"], fill=True)
    for isl in scene.ISLANDS:
        poly(isl, COLORS["island"], fill=True)
    for cw in scene.CROSSWALKS.values():
        poly(cw, COLORS["crosswalk"], fill=True)
    img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
    poly(scene.INBOUND.poly, COLORS["inbound"])
    poly(scene.OUTBOUND.poly, COLORS["outbound"])
    poly(scene.QUEUE_ZONE, COLORS["queue"], thick=2)
    poly(scene.JUNCTION, COLORS["junction"], thick=2)
    poly(scene.BUS_BAY, COLORS["bus"], thick=2)
    poly(scene.PARKING, COLORS["parking"], thick=2)
    for line in scene.SOLID_LINES:
        cv2.polylines(img, [(line * s).astype(np.int32)], False, COLORS["solid"], 4, cv2.LINE_AA)
    a, b = (scene.STOP_LINE * s).astype(int)
    cv2.line(img, tuple(a), tuple(b), COLORS["stop"], 6, cv2.LINE_AA)
    for z in scene.ONE_WAY:
        c = z.poly.mean(axis=0) * s
        d = z.direction * 0.08 * w
        cv2.arrowedLine(img, tuple((c - d).astype(int)), tuple((c + d).astype(int)), (255, 255, 255), 8,
                        cv2.LINE_AA, tipLength=0.3)
    for box in scene.SIGNAL_HEADS.values():
        (x0, y0), (x1, y1) = (box * s).astype(int)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 3)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--register", action="store_true", help="warp the layout to this frame's view first")
    args = ap.parse_args()
    frame = cv2.imread(args.frame)
    img = draw(frame, register=args.register)
    img = cv2.resize(img, (args.width, int(img.shape[0] * args.width / img.shape[1])), interpolation=cv2.INTER_AREA)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return 0


if __name__ == "__main__":
    sys.exit(main())
