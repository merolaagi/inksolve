#!/usr/bin/env bash
# Start InkSolve on port 48631, or the next free port. Writes data/server.json so scripts can find it.
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || ./setup.sh
PORT=${INKSOLVE_PORT:-48631}
HOST=${INKSOLVE_HOST:-127.0.0.1}
while .venv/bin/python - "$PORT" "$HOST" <<'PY'
import socket, sys
s = socket.socket()
try:
    s.bind((sys.argv[2], int(sys.argv[1])))
except OSError:
    sys.exit(0)        # taken
sys.exit(1)            # free
PY
do PORT=$((PORT + 1)); done
mkdir -p data
printf '{"port": %s, "host": "%s", "pid": %s, "started": %s}\n' "$PORT" "$HOST" "$$" "$(date +%s)" > data/server.json
echo "InkSolve: http://$HOST:$PORT"
( sleep 1.5; command -v open >/dev/null && [ -z "${INKSOLVE_NO_OPEN:-}" ] && open "http://$HOST:$PORT" ) >/dev/null 2>&1 &
exec .venv/bin/python -m uvicorn server.app:app --host "$HOST" --port "$PORT" --log-level warning
