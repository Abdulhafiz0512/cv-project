"""Contact sheets for reviewing predicted events by eye (dev-set building).

    TRAFFICEV_CACHE=data/cache python tools/review_events.py --video data/samples/C3905.mp4 \
        --pred predictions_samples.json --out data/review

For every predicted event: frames at 1 s before the start, the start, the middle and
the end, with all tracked boxes drawn, in one image named <video>_<n>_<class>.jpg.
Verdicts (TP / FP / boundary shift) go into labels/dev_labels.json by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trafficev import pipeline, scene  # noqa: E402
from trafficev.labels import COCO_KIND  # noqa: E402

COLORS = {"vehicle": (0, 200, 255), "two_wheeler": (255, 128, 0), "person": (0, 255, 0), "animal": (255, 0, 255)}
TILE_W, TILE_H = 800, 450


def frame_at(cap: cv2.VideoCapture, fps: float, t: float) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
    ok, f = cap.read()
    return f if ok else np.zeros((TILE_H, TILE_W, 3), np.uint8)


def draw(img: np.ndarray, rows: list, label: str) -> np.ndarray:
    img = cv2.convertScaleAbs(cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA), alpha=1.3, beta=8)
    sx, sy = TILE_W / scene.CANON[0], TILE_H / scene.CANON[1]
    for tid, x1, y1, x2, y2, cls in rows:
        c = COLORS.get(COCO_KIND.get(int(cls), ""), (200, 200, 200))
        cv2.rectangle(img, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), c, 1)
        cv2.putText(img, str(tid), (int(x1 * sx), int(y1 * sy) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)
    cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    per = pipeline.perceive(args.video)
    by_t: dict[float, list] = {}
    for tid, rows in per.rows.items():
        for t, x1, y1, x2, y2, conf, cls in rows:
            by_t.setdefault(round(t, 2), []).append((tid, x1, y1, x2, y2, cls))
    times = np.array(sorted(by_t))
    evts = json.loads(Path(args.pred).read_text())["videos"][Path(args.video).name]["events"]
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for n, (s, e, lab) in enumerate(evts):
        if args.only and lab not in args.only:
            continue
        tiles = []
        for name, t in (("start-1s", s - 1.0), ("start", s), ("middle", (s + e) / 2), ("end", e)):
            t = float(np.clip(t, 0, per.info.duration - 0.05))
            near = times[np.argmin(np.abs(times - t))] if len(times) else None
            rows = by_t.get(near, []) if near is not None else []
            tiles.append(draw(frame_at(cap, fps, t), rows, f"{lab} {name} t={t:.1f}s"))
        sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        cv2.imwrite(str(out / f"{Path(args.video).stem}_{n:03d}_{lab}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"{len(evts)} events -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
