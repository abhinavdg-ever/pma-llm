#!/bin/bash

# Start FastAPI backend in background
cd /app
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 &

# Start nginx in foreground
nginx -g "daemon off;"

