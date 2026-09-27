#!/usr/bin/env bash
# Regenerate everything derived from the sample videos, in order:
#   1. fetch the organizers' samples (1080p transcodes, see fetch_samples.py)
#   2. predictions_samples.json through the unchanged harness (exact mode = GPU-identical settings)
#   3. the website data: EDA figures, annotated videos, site.json
# Usage: bash scripts/reproduce_samples.sh [samples_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
SAMPLES="${1:-data/samples}"
export TRAFFICEV_CACHE="${TRAFFICEV_CACHE:-data/cache}"

if [ -z "$(ls "$SAMPLES"/*.mp4 2>/dev/null)" ]; then
  python scripts/fetch_samples.py --out "$SAMPLES"
fi
TRAFFICEV_EXACT=1 python run_submission.py --videos "$SAMPLES" --out predictions_samples.json \
  --team junction-watch --time-factor 30
python evaluate.py --pred predictions_samples.json --validate-only
python tools/build_site.py --videos "$SAMPLES" --pred predictions_samples.json --out web \
  --demo-url "${DEMO_URL:-}" --repo-url "${REPO_URL:-}"
