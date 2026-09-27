"""Shrink a local camera file to a small 1080p copy (for sharing over a slow link).

    python scripts/shrink_video.py C3896.MP4 C3896.mp4

Same settings as fetch_samples.py: 1080p H.264, original frame timing, no audio.
A 5-minute 5 GB camera file becomes roughly 150-250 MB, which the pipeline
processes the same way (it downsizes every frame to 1280x720 anyway).
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import imageio_ffmpeg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--crf", type=int, default=24)
    args = ap.parse_args()
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-y", "-i", args.src, "-map", "0:v:0",
           "-vf", f"scale=-2:{args.height}:flags=area", "-fps_mode", "passthrough",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", args.dst]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
