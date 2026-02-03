#!/bin/bash
# Launcher script for the recommender FastAPI service
# Used by Aspire to start the recommender as an executable

cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Check if uvicorn is available (part of requirements_api.txt)
if ! python3 -c "import uvicorn" 2>/dev/null; then
    echo "Warning: uvicorn not installed. Installing dependencies..."
    pip3 install -r requirements_api.txt
fi

# Get port from environment or use default
PORT=${PORT:-8000}

echo "Starting recommender service on port $PORT..."
python3 -m uvicorn api:app --host 0.0.0.0 --port "$PORT"
