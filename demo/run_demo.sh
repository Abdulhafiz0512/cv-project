#!/usr/bin/env bash
# Run the Junction Watch live demo on this Linux/macOS machine.
# First run creates .venv-demo and installs dependencies (needs internet).
#   bash demo/run_demo.sh           -> public https://*.gradio.live link + http://<this-host>:7860
#   bash demo/run_demo.sh --local   -> this machine only
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
if [ ! -x .venv-demo/bin/python ]; then
  echo "Creating .venv-demo ..."
  "$PY" -m venv .venv-demo
  .venv-demo/bin/python -m pip install --upgrade pip
  .venv-demo/bin/python -m pip install -r demo/requirements.txt
fi
if [ "${1:-}" = "--local" ]; then
  exec .venv-demo/bin/python demo/app.py
fi
exec .venv-demo/bin/python demo/app.py --host 0.0.0.0 --share
