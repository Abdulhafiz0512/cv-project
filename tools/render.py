"""Render an annotated review video: tracks, scene layout, active events, timeline and risk.

    python tools/render.py --video data/samples/C3905.mp4 --pred predictions_samples.json \
        --out web/media/C3905_annotated.mp4

Uses the perception cache (TRAFFICEV_CACHE) when present, so rendering after a
submission run does not re-run the detector.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trafficev import pipeline, scene  # noqa: E402
from trafficev.labels import COCO_KIND  # noqa: E402
from trafficev.video import sample_frames  # noqa: E402

KIND_COLORS = {"vehicle": (0, 200, 255), "two_wheeler": (255, 128, 0), "person": (0, 255, 0), "animal": (255, 0, 255)}
CLASS_COLORS = {
    "accident": (0, 0, 255), "near_miss": (0, 128, 255), "red_light": (0, 0, 200), "wrong_way": (255, 0, 255),
    "illegal_u_turn": (255, 0, 128), "stopped_vehicle": (0, 255, 255), "jaywalking": (0, 255, 0),
    "failure_to_yield": (255, 255, 0), "illegal_turn": (128, 0, 255), "solid_line_crossing": (255, 128, 128),
    "stop_line": (128, 128, 255), "congestion": (0, 160, 160), "road_obstacle": (160, 80, 0), "fire_smoke": (0, 80, 255),
}
OUT_W, OUT_H = 1280, 720
PANEL_H = 110


def _rows_by_time(per) -> dict[float, list]:
    by_t: dict[float, list] = {}
    for tid, rows in per.rows.items():
        for t, x1, y1, x2, y2, conf, cls in rows:
            by_t.setdefault(round(t, 3), []).append((tid, x1, y1, x2, y2, cls))
    return by_t


def _draw_scene(img: np.ndarray, sx: float, sy: float) -> None:
    s = np.array([sx, sy])
    for cw in scene.CROSSWALKS.values():
        cv2.polylines(img, [(cw * s).astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
    a, b = (scene.STOP_LINE * s).astype(int)
    cv2.line(img, tuple(a), tuple(b), (0, 0, 255), 2, cv2.LINE_AA)


def _panel(duration: float, events: list, risk: list, t: float) -> np.ndarray:
    panel = np.full((PANEL_H, OUT_W, 3), 24, np.uint8)
    labels = sorted({e[2] for e in events})
    x0, x1 = 150, OUT_W - 20

    def tx(sec):
        return int(x0 + (x1 - x0) * sec / max(duration, 1e-6))

    rows = max(1, len(labels))
    lane_h = max(8, min(16, (PANEL_H - 40) // rows))
    for i, lab in enumerate(labels):
        y = 6 + i * lane_h
        cv2.putText(panel, lab, (6, y + lane_h - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        for s, e, l in events:
            if l == lab:
                cv2.rectangle(panel, (tx(s), y + 1), (max(tx(e), tx(s) + 2), y + lane_h - 2), CLASS_COLORS.get(l, (200, 200, 200)), -1)
    if risk:
        r = np.asarray(risk)
        ys = PANEL_H - 6 - (r[:, 1] * 26).astype(int)
        pts = np.stack([np.array([tx(v) for v in r[:, 0]]), ys], axis=1).astype(np.int32)
        cv2.polylines(panel, [pts], False, (80, 80, 255), 1, cv2.LINE_AA)
        cv2.line(panel, (x0, PANEL_H - 19), (x1, PANEL_H - 19), (90, 90, 90), 1)
        cv2.putText(panel, "risk", (6, PANEL_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 255), 1, cv2.LINE_AA)
    cv2.line(panel, (tx(t), 0), (tx(t), PANEL_H), (255, 255, 255), 1)
    return panel


def _culprit(evidence: list, tid: int, t: float) -> str | None:
    """The event a road user is currently blamed for (the most specific, i.e. shortest, one)."""
    hits = [(e - s, lab) for lab, k, s, e in evidence if k == tid and s - 0.3 <= t <= e + 0.3]
    return min(hits)[1] if hits else None


def render(video: str, per, events: list, risk: list, out: str, fps: float = 10.0, progress=None,
           evidence: list | None = None, crf: int = 27) -> None:
    """Write an annotated H.264 review video; `per` is a pipeline.Perception.

    Road users are coloured by kind; one that an event rule fired on is drawn
    thick in that event's colour with the event name, while the event is active.
    """
    if evidence is None:
        evidence = pipeline.explain(per)[1]
    by_t = _rows_by_time(per)
    duration = per.info.duration
    trails: dict[int, list] = {}
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H + PANEL_H}",
         "-r", str(fps), "-i", "pipe:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    sx, sy = OUT_W / scene.CANON[0], OUT_H / scene.CANON[1]
    risk_t = np.asarray([r[0] for r in risk]) if risk else None
    last, tick = None, 0.0
    for t, frame in sample_frames(video, fps):
        img = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        img = cv2.convertScaleAbs(img, alpha=1.25, beta=8)
        _draw_scene(img, sx, sy)
        for tid, x1, y1, x2, y2, cls in by_t.get(round(t, 3), []):
            kind = COCO_KIND.get(int(cls), "other")
            c = KIND_COLORS.get(kind, (200, 200, 200))
            p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
            blamed = _culprit(evidence, int(tid), t)
            if blamed:
                c = CLASS_COLORS.get(blamed, (255, 255, 255))
                cv2.rectangle(img, p1, p2, c, 4)
                label = f"{tid} {blamed}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (p1[0], p1[1] - th - 8), (p1[0] + tw + 6, p1[1]), c, -1)
                cv2.putText(img, label, (p1[0] + 3, p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(img, p1, p2, c, 1)
                cv2.putText(img, str(tid), (p1[0], p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
            trail = trails.setdefault(tid, [])
            trail.append((t, int((x1 + x2) / 2 * sx), int(y2 * sy)))
            while trail and t - trail[0][0] > 3.0:
                trail.pop(0)
            if len(trail) > 1:
                cv2.polylines(img, [np.array([(x, y) for _, x, y in trail], np.int32)], False, c, 1, cv2.LINE_AA)
        active = [l for s, e, l in events if s <= t <= e]
        for i, lab in enumerate(sorted(set(active))):
            cv2.rectangle(img, (8, 8 + 26 * i), (260, 30 + 26 * i), CLASS_COLORS.get(lab, (200, 200, 200)), -1)
            cv2.putText(img, lab, (14, 25 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        if risk_t is not None:
            r = risk[int(np.clip(np.searchsorted(risk_t, t), 0, len(risk) - 1))][1]
            col = (0, 0, 255) if r >= 0.5 else (200, 200, 200)
            cv2.putText(img, f"risk {r:.2f}", (OUT_W - 150, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        cv2.putText(img, f"{t:6.1f}s", (OUT_W - 150, OUT_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        # sampled frames are not evenly spaced (I/P frames only): hold each one on a
        # fixed output clock so the review video plays in real time
        out_frame = np.vstack([img, _panel(duration, events, risk, t)]).tobytes()
        while last is not None and tick < t:
            proc.stdin.write(last)
            tick += 1.0 / fps
        last = out_frame
        if progress is not None:
            progress(t / max(duration, 1e-6))
    while last is not None and tick < duration:
        proc.stdin.write(last)
        tick += 1.0 / fps
    proc.stdin.close()
    proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--pred", help="predictions.json from run_submission.py (events + risk)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()
    events, risk = [], []
    if args.pred:
        entry = json.loads(Path(args.pred).read_text())["videos"].get(Path(args.video).name, {})
        events, risk = entry.get("events", []), entry.get("risk", [])
    else:
        events = pipeline.detect_events(args.video)
    render(args.video, pipeline.perceive(args.video), events, risk, args.out, args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
