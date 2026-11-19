#!/usr/bin/env bash
# One-time setup: a Python virtual environment with InkSolve's packages.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if ! command -v "$PY" >/dev/null; then echo "Python 3 is needed (brew install python)." >&2; exit 1; fi
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || { echo "Python 3.10 or newer is needed." >&2; exit 1; }
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt
mkdir -p data
echo "InkSolve is set up. Start it with ./run.sh"
