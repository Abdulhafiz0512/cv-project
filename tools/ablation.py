"""Ablations on real footage: detector / input size, and decode strategy.

    python tools/ablation.py --video data/excerpts/C3905_first12s.mp4 --out web/data/ablations.json

Detector rows: latency per frame on this machine, detections per frame by kind,
tracks formed, and events produced by the unchanged rules. Decode rows: video
seconds decoded per wall-clock second for each reader.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trafficev import pipeline, runtime  # noqa: E402
from trafficev.detector import Detector  # noqa: E402
from trafficev.labels import COCO_KIND  # noqa: E402
from trafficev.video import PROC_H, PROC_W, probe, sample_frames  # noqa: E402

VARIANTS = [("yolo26s.pt", 1280), ("yolo26s.pt", 960), ("yolo11s.pt", 1280), ("yolo26m.pt", 1280)]


def detector_row(video: str, weights: str, imgsz: int, weights_dir: Path) -> dict:
    runtime.WEIGHTS_DIR = weights_dir if (weights_dir / weights).exists() else runtime.WEIGHTS_DIR
    det = Detector(weights=weights, imgsz=imgsz)
    det([np.zeros((PROC_H, PROC_W, 3), np.uint8)])  # warm-up
    times, kinds = [], {"vehicle": 0, "two_wheeler": 0, "person": 0}
    t0 = time.perf_counter()
    per = pipeline.perceive(video, budget_s=1e9, detector=_Timed(det, times, kinds))  # run without TRAFFICEV_CACHE
    wall = time.perf_counter() - t0
    frames = len(per.times)
    tracks = per.tracks()
    events = pipeline.events_from(per)
    return {
        "detector": weights.replace(".pt", ""), "imgsz": imgsz, "device": det.device,
        "ms_per_frame": round(1000 * float(np.median(times)), 1),
        "per_frame": {k: round(v / max(frames, 1), 1) for k, v in kinds.items()},
        "tracks": {k: sum(1 for tr in tracks if tr.kind == k) for k in kinds},
        "events": len(events), "wall_s": round(wall, 1),
    }


class _Timed:
    """Detector wrapper recording latency and detections by kind."""

    def __init__(self, det: Detector, times: list, kinds: dict):
        self.det, self.times, self.kinds, self.device = det, times, kinds, det.device

    def __call__(self, frames):
        t = time.perf_counter()
        out = self.det(frames)
        self.times.append((time.perf_counter() - t) / len(frames))
        for d in out:
            for c in d.cls[d.conf >= 0.3]:
                k = COCO_KIND.get(int(c))
                if k in self.kinds:
                    self.kinds[k] += 1
        return out


def decode_rows(video: str) -> list[dict]:
    dur = probe(video).duration
    rows = []
    for name, fn in [
        ("PyAV, B-frames skipped (submission)", lambda: sum(1 for _ in sample_frames(video, 10.0, skip_bframes=True))),
        ("PyAV, full decode, 10 fps kept", lambda: sum(1 for _ in sample_frames(video, 10.0, skip_bframes=False))),
        ("OpenCV grab every frame, retrieve every 3rd", lambda: _cv2(video, 3)),
        ("OpenCV read every frame (the harness's Part B loop)", lambda: _cv2(video, 1)),
    ]:
        t = time.perf_counter()
        n = fn()
        wall = time.perf_counter() - t
        rows.append({"reader": name, "frames_out": n, "realtime_factor": round(dur / wall, 2)})
    return rows


def _cv2(video: str, every: int) -> int:
    cap = cv2.VideoCapture(video)
    i = n = 0
    while cap.grab():
        if i % every == 0:
            cap.retrieve()
            n += 1
        i += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights-dir", default="data/ablation_weights", help="where the non-shipped weights live")
    args = ap.parse_args()
    shipped = runtime.WEIGHTS_DIR
    rows = []
    for weights, imgsz in VARIANTS:
        runtime.WEIGHTS_DIR = shipped
        rows.append(detector_row(args.video, weights, imgsz, Path(args.weights_dir)))
        print(rows[-1], flush=True)
    result = {"video": Path(args.video).name, "machine": "4-core Intel i5-10210U laptop, no GPU",
              "detectors": rows, "decode": decode_rows(args.video)}
    print(result["decode"])
    Path(args.out).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
