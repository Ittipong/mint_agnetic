#!/bin/bash
# Restart LangGraph Studio / FastAPI Server
# Usage: ./restart_studio.sh [port]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PORT="${1:-8000}"
PID_FILE="/tmp/langgraph_studio_${PORT}.pid"
LOG_FILE="$PROJECT_DIR/logs/studio_$(date +%Y-%m-%d).log"

echo "=== Restarting LangGraph Studio ==="
echo "Project: $PROJECT_DIR"
echo "Port: $PORT"
echo "Log: $LOG_FILE"
echo ""

# Kill existing process
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping existing process (PID: $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$OLD_PID" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
fi

# Kill by port
EXISTING_PID=$(lsof -ti:$PORT 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
    echo "Killing process on port $PORT (PID: $EXISTING_PID)..."
    kill "$EXISTING_PID" 2>/dev/null || true
    sleep 1
fi

# Create logs dir
mkdir -p "$PROJECT_DIR/logs"

# Start server
echo "Starting server on port $PORT..."
cd "$PROJECT_DIR"
source .venv/bin/activate

nohup python -m uvicorn src.server:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    > "$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

echo "Started (PID: $NEW_PID)"
echo "Log: $LOG_FILE"
echo ""

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..30}; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server is ready!"
        exit 0
    fi
    sleep 1
done

echo "Server may not be ready. Check log:"
tail -50 "$LOG_FILE"
exit 1
