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

# Set environment variables to prevent TensorFlow Metal plugin issues on macOS
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""
# Disable Metal plugin which can cause segfaults on some macOS systems
export TF_ENABLE_ONEDNN_OPTS=0

echo "Starting recommender service on port $PORT..."
echo "TensorFlow will use CPU (Metal disabled to prevent crashes)"

# Run with better error handling
python3 -m uvicorn api:app --host 0.0.0.0 --port "$PORT" 2>&1 || {
    echo "Error: Recommender service crashed!"
    echo "Checking TensorFlow installation..."
    python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')" || echo "TensorFlow import failed"
    exit 1
}
