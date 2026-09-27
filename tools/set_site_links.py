"""Set the website's external links without rebuilding the sample-video data.

    python tools/set_site_links.py --demo-url https://xxxx.gradio.live
    python tools/set_site_links.py --repo-url https://github.com/Abdulhafiz0512/cv-project

Edits web/data/site.json in place; commit and push to publish (GitHub Pages redeploys).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "web" / "data" / "site.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo-url")
    ap.add_argument("--repo-url")
    args = ap.parse_args()
    site = json.loads(SITE.read_text(encoding="utf-8"))
    for key in ("demo_url", "repo_url"):
        value = getattr(args, key)
        if value is not None:
            site[key] = value.rstrip("/")
    SITE.write_text(json.dumps(site), encoding="utf-8")
    print(f"demo_url={site.get('demo_url')!r} repo_url={site.get('repo_url')!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
