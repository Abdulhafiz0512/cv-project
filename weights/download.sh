#!/usr/bin/env bash
# Fetch model weights (run once, with internet, before the offline evaluation).
# weights/yolo26s.pt is committed to the repository, so this is only needed if
# it is missing (e.g. a shallow checkout without LFS).
set -euo pipefail
cd "$(dirname "$0")"
URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt"
SHA256="646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
if [ ! -s yolo26s.pt ]; then
  curl -fL --retry 5 -o yolo26s.pt "$URL"
fi
echo "${SHA256}  yolo26s.pt" | sha256sum -c -
