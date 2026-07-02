#!/bin/bash

# Find and kill any process running on port 8000
PORT=8000
PID=$(lsof -t -i :$PORT)

if [ -n "$PID" ]; then
  echo "Found process $PID running on port $PORT. Killing it..."
  kill -9 $PID
  sleep 1
else
  echo "No process running on port $PORT."
fi

echo "Starting the SHL Recommender application on port $PORT..."
uv run uvicorn app.main:app --host 127.0.0.1 --port $PORT --reload
