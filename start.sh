#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate market-research-genai

# Set environment variables with all three Gemini API keys
export GEMINI_API_KEY_1=AIzaSyCqjq6SqxpWEOBYouBeBDB4Opv2s9EPp94
export GEMINI_API_KEY_2=AIzaSyDjvcU0N3LHG1ydS9TcAM9z-lzEETLSZwM
export GEMINI_API_KEY_3=AIzaSyDNORJ99OA1bCKsB54ICevw7Ac_lKaJ6xQ
export REDIS_URL=redis://localhost:6379/0

# Start Redis if not running
if ! pgrep redis-server > /dev/null; then
    echo "Starting Redis server..."
    redis-server &
    sleep 2
fi

# Start FastAPI in the background
echo "Starting FastAPI server..."
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload &
FASTAPI_PID=$!

# Start Streamlit in the background
echo "Starting Streamlit app..."
streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0 &
STREAMLIT_PID=$!

# Wait for both processes
wait $FASTAPI_PID $STREAMLIT_PID

# Cleanup on exit
trap 'kill $FASTAPI_PID $STREAMLIT_PID' EXIT 