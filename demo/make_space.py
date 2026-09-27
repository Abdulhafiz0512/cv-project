"""Assemble a ready-to-push Hugging Face Space from this repository.

    python demo/make_space.py --out build/space
    cd build/space && git init && git remote add origin https://huggingface.co/spaces/<user>/<space>
    git add . && git commit -m "Junction Watch demo" && git push -u origin main

or, after `hf auth login`, in one step:

    python demo/make_space.py --push <user>/junction-watch

The Space gets app.py at its root plus the package, the renderer and the weights.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FRONT_MATTER = """---
title: Junction Watch
emoji: 🚦
colorFrom: blue
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
license: agpl-3.0
---

Live demo of the Junction Watch traffic-event detector (WIUT Hackathon 2026, CV track).
Upload a clip from the junction camera to get the events, an annotated playback and the accident-risk curve.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="build/space")
    ap.add_argument("--push", metavar="USER/SPACE", help="create/update this Hugging Face Space "
                    "(needs `hf auth login` first)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "tools").mkdir(parents=True)
    shutil.copytree(ROOT / "src", out / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (out / "weights").mkdir()
    shutil.copy(ROOT / "weights" / "yolo26s.pt", out / "weights" / "yolo26s.pt")
    shutil.copy(ROOT / "tools" / "render.py", out / "tools" / "render.py")
    shutil.copy(ROOT / "demo" / "app.py", out / "app.py")
    shutil.copy(ROOT / "demo" / "requirements.txt", out / "requirements.txt")
    if (ROOT / "demo" / "examples").exists():
        shutil.copytree(ROOT / "demo" / "examples", out / "demo" / "examples")
    (out / "README.md").write_text(FRONT_MATTER, encoding="utf-8")
    (out / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n*.mp4 filter=lfs diff=lfs merge=lfs -text\n")
    print(f"Space assembled in {out}")
    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.push, repo_type="space",
                          commit_message="Deploy Junction Watch demo")
        print(f"Deployed: https://huggingface.co/spaces/{args.push}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
