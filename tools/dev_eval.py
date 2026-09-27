"""Fast rule iteration on the dev set: cached perception -> rules -> official Part A metric.

    TRAFFICEV_CACHE=data/cache python tools/dev_eval.py --videos data/samples --gt labels/dev_labels.json

The first run fills the perception cache (detector + tracker); later runs only
re-evaluate the rules, which takes seconds. Scores use evaluate.py's own code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import evaluate  # noqa: E402  (official metric, unchanged)
from trafficev import pipeline  # noqa: E402


def predict(video: Path, only: tuple[str, ...] | None) -> list[list]:
    evts = pipeline.events_from(pipeline.perceive(str(video)))
    return [e for e in evts if only is None or e[2] in only]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--only", nargs="*", help="evaluate a subset of rules")
    ap.add_argument("--per-video", action="store_true")
    args = ap.parse_args()
    if not os.environ.get("TRAFFICEV_CACHE"):
        print("hint: set TRAFFICEV_CACHE=data/cache to avoid re-running perception", file=sys.stderr)

    gt = json.loads(Path(args.gt).read_text())
    gt = gt.get("videos", gt)
    videos = {p.name: p for p in sorted(Path(args.videos).glob("*.mp4")) if p.name in gt}
    pred = {name: {"events": predict(path, tuple(args.only) if args.only else None)} for name, path in videos.items()}
    gt = {k: v for k, v in gt.items() if k in videos}
    rep = evaluate.evaluate_part_a(gt, pred, per_video=args.per_video)
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
