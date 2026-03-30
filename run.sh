#!/bin/bash

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $(jobs -p) 2>/dev/null
    wait 2>/dev/null
    echo "Done."
    exit
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT SIGTERM

# Start ADK API Server in background on port 8000
echo "Starting ADK API Server on port 8000..."
adk api_server --port 8000 &
ADK_PID=$!

# Wait for the API server to be ready (poll every 0.5s, timeout after 30s)
echo "Waiting for API Server to be ready..."
MAX_WAIT=30
WAITED=0
while ! curl -s http://localhost:8000 >/dev/null 2>&1; do
    sleep 0.5
    WAITED=$((WAITED + 1))
    if [ $WAITED -ge $((MAX_WAIT * 2)) ]; then
        echo "ERROR: ADK API server did not start within ${MAX_WAIT}s"
        kill $ADK_PID 2>/dev/null
        exit 1
    fi
done
echo "ADK API Server is ready."

# Start Streamlit App (foreground)
echo "Starting Streamlit App..."
streamlit run app.py

# If streamlit exits, cleanup
cleanup
