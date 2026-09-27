"""Build the website's data: EDA figures, annotated videos and a site.json manifest.

    TRAFFICEV_CACHE=data/cache python tools/build_site.py --videos data/samples \
        --pred predictions_samples.json --out web

Everything the page shows about the sample videos comes from this script and the
submission's own predictions file, so the site can be regenerated in one step.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import cv2  # noqa: E402
import eda  # noqa: E402
from render import render  # noqa: E402
from trafficev import pipeline, view  # noqa: E402
from trafficev.labels import CLASSES  # noqa: E402
from trafficev.video import sample_frames  # noqa: E402

RISK_HZ = 5.0
WEB_CRF = 30          # web renders: ~1/2 the size of review renders, still sharp at 1280 px


def downsample(risk: list, hz: float) -> list:
    out, next_t = [], 0.0
    for t, s in risk:
        if t >= next_t:
            out.append([round(t, 2), round(s, 3)])
            next_t = t + 1.0 / hz
    return out


def view_summary(video: Path) -> dict:
    """Which reference view the video registered to, and how its framing differs from the base."""
    _, frame = next(iter(sample_frames(str(video), 1.0)))
    fit = view.register(frame)
    H = fit.H
    return {"reference": fit.reference.removesuffix(".jpg").removeprefix("ref_") or None, "inliers": fit.inliers,
            "zoom": round(float(abs(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0])) ** 0.5, 3),
            "shift_px": [round(float(H[0, 2]), 1), round(float(H[1, 2]), 1)]}


def summary(entry: dict, log: dict) -> dict:
    counts: dict[str, int] = {}
    for _, _, lab in entry["events"]:
        counts[lab] = counts.get(lab, 0) + 1
    risk = [s for _, s in entry["risk"]]
    duration = float(log.get("duration") or 0.0)
    return {
        "events": len(entry["events"]),
        "per_class": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "event_seconds": round(sum(e - s for s, e, _ in entry["events"]), 1),
        "max_risk": round(max(risk), 3) if risk else 0.0,
        "alarm_seconds": round(sum(1 for s in risk if s >= 0.5) / max(len(risk) / max(duration, 1e-6), 1e-6), 1),
        "runtime": {k: log.get(k) for k in ("part_a_sec", "part_b_sec", "total_sec", "budget_sec")},
        "runtime_x": round(float(log.get("total_sec") or 0.0) / duration, 2) if duration else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--pred", required=True, help="predictions_samples.json from run_submission.py")
    ap.add_argument("--out", default="web")
    ap.add_argument("--demo-url", default="", help="URL of the live demo (Hugging Face Space)")
    ap.add_argument("--repo-url", default="")
    ap.add_argument("--no-render", action="store_true", help="skip the annotated videos")
    args = ap.parse_args()

    out = Path(args.out)
    pred = json.loads(Path(args.pred).read_text())
    src = Path(args.videos)
    vids = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() == ".mp4")
    (out / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(args.pred, out / "data" / "predictions_samples.json")

    videos = []
    for v in vids:
        entry = pred["videos"].get(v.name, {"events": [], "risk": []})
        eda_dir = out / "assets" / "eda" / v.stem
        stats = eda.analyse(v, eda_dir)
        media = None
        if not args.no_render:
            media = out / "media" / f"{v.stem}.mp4"
            render(str(v), pipeline.perceive(str(v)), entry["events"], entry["risk"], str(media), crf=WEB_CRF)
        videos.append({
            "name": v.name,
            "stem": v.stem,
            "meta": stats["meta"],
            "tracks": stats["tracks"],
            "events": entry["events"],
            "risk": downsample(entry["risk"], RISK_HZ),
            "annotated": f"media/{v.stem}.mp4" if media else None,
            "eda": {k: f"assets/eda/{v.stem}/{k}.jpg" for k in
                    ("background", "heat_vehicles", "heat_pedestrians", "trajectories", "flow_field")},
            "stats": f"assets/eda/{v.stem}/stats.json",
            "summary": summary(entry, pred.get("log", {}).get(v.name, {})),
            "view": view_summary(v),
        })
        print(f"{v.name}: {len(entry['events'])} events", flush=True)

    previous = {}
    if (out / "data" / "site.json").exists():  # keep links set earlier with tools/set_site_links.py
        previous = json.loads((out / "data" / "site.json").read_text(encoding="utf-8"))
    site = {
        "generated": time.strftime("%Y-%m-%d"),
        "team": pred.get("team", ""),
        "demo_url": args.demo_url or previous.get("demo_url", ""),
        "repo_url": args.repo_url or previous.get("repo_url", ""),
        "classes": CLASSES,
        "videos": videos,
    }
    (out / "data" / "site.json").write_text(json.dumps(site))
    return 0


if __name__ == "__main__":
    sys.exit(main())
