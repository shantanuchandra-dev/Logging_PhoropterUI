#!/bin/bash

# Eye Test Engine - Frontend Launcher
# This script starts both the backend API and opens the frontend

echo "=========================================="
echo "Eye Test Engine - Interactive Frontend"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "api_server.py" ]; then
    echo "Error: Please run this script from the eye_test_engine directory"
    exit 1
fi

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import flask, flask_cors, yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip3 install flask flask-cors pyyaml
fi

echo "✓ Dependencies OK"
echo ""

# Start backend server in background
echo "Starting backend API server on port 5000..."
python3 -m eye_test_engine.api_server &
BACKEND_PID=$!
echo "✓ Backend started (PID: $BACKEND_PID)"
echo ""

# Wait for backend to start
sleep 2

# Start frontend server
echo "Starting frontend server on port 8080..."
cd frontend
python3 -m http.server 8080 &
FRONTEND_PID=$!
echo "✓ Frontend started (PID: $FRONTEND_PID)"
echo ""

# Open browser
echo "Opening browser..."
sleep 1
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open http://localhost:8080
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open http://localhost:8080
fi

echo ""
echo "=========================================="
echo "Eye Test Engine is now running!"
echo "=========================================="
echo ""
echo "Backend API:  http://localhost:5000"
echo "Frontend UI:  http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for Ctrl+C
trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
