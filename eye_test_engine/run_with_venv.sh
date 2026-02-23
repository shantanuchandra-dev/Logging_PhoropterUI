#!/bin/bash
# Run Eye Test Engine backend and frontend using project venv.
# Stops any process already using ports 5000 and 8080.

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_ROOT/venv"
cd "$REPO_ROOT"

# Stop existing servers on 5000 and 8080
echo "Stopping any existing servers on 5000 and 8080..."
for port in 5000 8080; do
  pid=$(lsof -ti :$port 2>/dev/null || true)
  if [ -n "$pid" ]; then
    kill -9 $pid 2>/dev/null || true
    echo "  Killed PID $pid on port $port"
  fi
done
echo ""

# Ensure venv exists
if [ ! -d "$VENV" ]; then
  echo "Creating venv..."
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install -q flask flask-cors pyyaml 2>/dev/null || true

echo "Starting backend on http://localhost:5000 ..."
cd "$REPO_ROOT/eye_test_engine"
"$VENV/bin/python" -c "
from api_server import app
app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
" &
BACKEND_PID=$!
cd "$REPO_ROOT"

sleep 2
echo "Starting frontend on http://localhost:8080 ..."
cd "$REPO_ROOT/eye_test_engine/frontend"
"$VENV/bin/python" -m http.server 8080 &
FRONTEND_PID=$!
cd "$REPO_ROOT"

echo ""
echo "=========================================="
echo "Eye Test Engine (venv) is running"
echo "=========================================="
echo "  Backend:  http://localhost:5000"
echo "  Frontend: http://localhost:8080"
echo "  Press Ctrl+C to stop both servers"
echo ""

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
