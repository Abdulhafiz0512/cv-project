"""Exploratory data analysis of the sample videos -> figures + JSON for the website.

    TRAFFICEV_CACHE=data/cache python tools/eda.py --videos data/samples --out web/assets/eda

Per video: metadata (codec, bit depth, GOP), lighting over time, a median
"empty road" background, heatmaps of where vehicles and pedestrians are,
trajectories coloured by heading, the learned flow field (dominant motion per
cell, the evidence behind the hard-coded one-way directions), counts of road
users per second, a speed histogram and the visible signal-lamp series.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trafficev import pipeline, scene  # noqa: E402
from trafficev.kinematics import kinematics  # noqa: E402

OUT_W, OUT_H = 1280, 720
GRID = 32  # flow-field cell size in canonical px


def video_meta(path: Path) -> dict:
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        cc = s.codec_context
        return {
            "file": path.name, "width": cc.width, "height": cc.height, "fps": round(float(s.average_rate), 3),
            "frames": s.frames, "duration_s": round(float(s.duration * s.time_base), 2) if s.duration else None,
            "codec": cc.name, "profile": cc.profile, "pix_fmt": cc.pix_fmt,
            "bitrate_mbps": round((c.bit_rate or 0) / 1e6, 1),
        }


def keyframes(path: Path, every_s: float = 2.0):
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.codec_context.skip_frame = "NONKEY"
        last = -1e9
        for f in c.decode(s):
            t = float(f.time or 0.0)
            if t - last >= every_s:
                last = t
                yield t, f.to_ndarray(width=OUT_W, height=OUT_H, format="bgr24")


def heat_overlay(bg: np.ndarray, pts: np.ndarray, color_map=cv2.COLORMAP_INFERNO) -> np.ndarray:
    h = np.zeros((OUT_H // 4, OUT_W // 4), np.float32)
    if len(pts):
        p = (pts * np.array([OUT_W / scene.CANON[0], OUT_H / scene.CANON[1]]) / 4).astype(int)
        p = p[(p[:, 0] >= 0) & (p[:, 0] < h.shape[1]) & (p[:, 1] >= 0) & (p[:, 1] < h.shape[0])]
        np.add.at(h, (p[:, 1], p[:, 0]), 1.0)
    h = cv2.GaussianBlur(h, (0, 0), 2.0)
    h = np.log1p(h)
    h = (255 * h / max(h.max(), 1e-6)).astype(np.uint8)
    h = cv2.resize(h, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
    col = cv2.applyColorMap(h, color_map)
    alpha = (h.astype(np.float32) / 255.0)[..., None] * 0.85
    return (bg * (1 - alpha) + col * alpha).astype(np.uint8)


def analyse(path: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    meta = video_meta(path)
    frames = list(keyframes(path))
    luma = [(round(t, 1), round(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()), 1)) for t, f in frames]
    bg = np.median(np.stack([f for _, f in frames]), axis=0).astype(np.uint8)
    bg_vis = cv2.convertScaleAbs(bg, alpha=1.4, beta=8)
    cv2.imwrite(str(out / "background.jpg"), bg_vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

    per = pipeline.perceive(str(path))
    kins = [kinematics(t) for t in per.tracks()]
    s = np.array([OUT_W / scene.CANON[0], OUT_H / scene.CANON[1]])

    veh = [k for k in kins if k.kind in ("vehicle", "two_wheeler")]
    ped = [k for k in kins if k.kind == "person"]
    vpts = np.concatenate([k.foot[k.rel_speed > 0.3] for k in veh]) if veh else np.zeros((0, 2))
    ppts = np.concatenate([k.foot for k in ped]) if ped else np.zeros((0, 2))
    cv2.imwrite(str(out / "heat_vehicles.jpg"), heat_overlay(bg_vis, vpts), [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(out / "heat_pedestrians.jpg"), heat_overlay(bg_vis, ppts, cv2.COLORMAP_VIRIDIS),
                [cv2.IMWRITE_JPEG_QUALITY, 85])

    # trajectories coloured by heading (hue = direction of travel)
    traj = (bg_vis * 0.55).astype(np.uint8)
    for k in veh:
        if len(k.t) < 5 or np.linalg.norm(k.foot[-1] - k.foot[0]) < 2 * np.median(k.scale):
            continue
        d = k.foot[-1] - k.foot[0]
        hue = int((np.degrees(np.arctan2(d[1], d[0])) % 360) / 2)
        col = cv2.cvtColor(np.uint8([[[hue, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.polylines(traj, [(k.foot * s).astype(np.int32)], False, col, 2, cv2.LINE_AA)
    cv2.imwrite(str(out / "trajectories.jpg"), traj, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # flow field: mean unit velocity per cell over moving vehicles
    gw, gh = int(scene.CANON[0] // GRID), int(scene.CANON[1] // GRID)
    acc = np.zeros((gh, gw, 2))
    cnt = np.zeros((gh, gw))
    for k in veh:
        m = k.rel_speed > 0.5
        if not m.any():
            continue
        cells = (k.foot[m] // GRID).astype(int)
        ok = (cells[:, 0] < gw) & (cells[:, 1] < gh) & (cells >= 0).all(axis=1)
        v = k.vel[m][ok] / np.linalg.norm(k.vel[m][ok], axis=1, keepdims=True)
        np.add.at(acc, (cells[ok, 1], cells[ok, 0]), v)
        np.add.at(cnt, (cells[ok, 1], cells[ok, 0]), 1)
    flow = (bg_vis * 0.55).astype(np.uint8)
    for gy in range(gh):
        for gx in range(gw):
            if cnt[gy, gx] < 3:
                continue
            v = acc[gy, gx] / cnt[gy, gx]
            coherence = float(np.linalg.norm(v))
            if coherence < 0.5:
                continue
            c = (np.array([gx + 0.5, gy + 0.5]) * GRID) * s
            tip = c + v / max(coherence, 1e-6) * GRID * s * 0.9
            cv2.arrowedLine(flow, tuple(c.astype(int)), tuple(tip.astype(int)), (60, 220, 255), 2, cv2.LINE_AA,
                            tipLength=0.4)
    cv2.imwrite(str(out / "flow_field.jpg"), flow, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # road users per second, by kind
    dur = per.info.duration
    secs = np.arange(0, int(np.ceil(dur)))
    counts = {}
    for kind in ("vehicle", "two_wheeler", "person"):
        c = np.zeros(len(secs), int)
        for k in kins:
            if k.kind != kind:
                continue
            present = np.unique(np.clip(k.t.astype(int), 0, len(secs) - 1))
            c[present] += 1
        counts[kind] = c.tolist()
    speeds = np.concatenate([k.rel_speed[k.rel_speed > 0.3] for k in veh]) if veh else np.zeros(0)
    hist, edges = np.histogram(speeds, bins=24, range=(0, 12))
    stats = {
        "meta": meta,
        "lighting": luma,
        "counts_per_second": counts,
        "tracks": {kind: sum(1 for k in kins if k.kind == kind) for kind in ("vehicle", "two_wheeler", "person", "animal")},
        "speed_hist_bhs": {"counts": hist.tolist(), "edges": edges.round(2).tolist()},
        "lamps": {name: np.asarray(v).round(4).tolist() for name, v in per.lamps.items()},
        "lamp_times": np.asarray(per.times).round(2).tolist(),
    }
    (out / "stats.json").write_text(json.dumps(stats))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src = Path(args.videos)
    vids = [src] if src.is_file() else sorted(src.glob("*.mp4"))
    index = {}
    for v in vids:
        st = analyse(v, Path(args.out) / v.stem)
        index[v.name] = {"meta": st["meta"], "tracks": st["tracks"]}
        print(v.name, st["meta"], st["tracks"])
    (Path(args.out) / "index.json").write_text(json.dumps(index, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
