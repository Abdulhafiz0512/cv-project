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

import eda  # noqa: E402
from render import render  # noqa: E402
from trafficev import pipeline  # noqa: E402
from trafficev.labels import CLASSES  # noqa: E402

RISK_HZ = 5.0


def downsample(risk: list, hz: float) -> list:
    out, next_t = [], 0.0
    for t, s in risk:
        if t >= next_t:
            out.append([round(t, 2), round(s, 3)])
            next_t = t + 1.0 / hz
    return out


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
    vids = [src] if src.is_file() else sorted(src.glob("*.mp4"))
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
            render(str(v), pipeline.perceive(str(v)), entry["events"], entry["risk"], str(media))
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
