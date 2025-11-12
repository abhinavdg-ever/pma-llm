#!/bin/bash

# Start FastAPI backend in background on port 8081 (nginx proxies from 8080)
cd /app
uvicorn app:app --host 0.0.0.0 --port 8081 --workers 1 &
UVICORN_PID=$!

# Wait for FastAPI to be ready
echo "Waiting for FastAPI to start..."
for i in {1..30}; do
  if curl -f http://localhost:8081/health > /dev/null 2>&1; then
    echo "FastAPI is ready!"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "Warning: FastAPI did not start within 30 seconds"
  fi
  sleep 1
done

# Start nginx in foreground on port 8080 (matches docker-compose.yml 8080:8080)
nginx -g "daemon off;"

