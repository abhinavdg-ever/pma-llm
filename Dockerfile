# Multi-stage build for Contracts Copilot

# Frontend builder stage
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files first for better caching
COPY frontend/package*.json ./
COPY frontend/package-lock.json* ./

RUN npm ci

# Copy all frontend files
# Build context is /apps/local-llm/, so frontend/ refers to /apps/local-llm/frontend/
# IMPORTANT: Copy src/lib FIRST before the main copy to ensure it's not excluded
COPY frontend/src/lib ./src/lib
COPY frontend/ .

# Debug: Verify files were copied correctly - FAIL BUILD if utils.ts is missing
RUN echo "=== Verifying copied files ===" && \
    pwd && \
    echo "=== Root directory contents ===" && \
    ls -la && \
    echo "=== Checking if src exists ===" && \
    (test -d src && echo "✓ src/ exists" && ls -la src/ || (echo "✗ src/ does not exist" && exit 1)) && \
    echo "=== Checking if src/lib exists ===" && \
    (test -d src/lib && echo "✓ src/lib/ exists" && ls -la src/lib/ || (echo "✗ src/lib/ does not exist" && exit 1)) && \
    echo "=== Verifying utils.ts exists ===" && \
    (test -f src/lib/utils.ts && echo "✓ utils.ts EXISTS" || (echo "✗ utils.ts MISSING!" && find . -name "utils.ts" -type f 2>/dev/null && exit 1)) && \
    echo "=== Full src/lib contents ===" && \
    ls -la src/lib/ && \
    echo "=== File structure check ===" && \
    find src -type f -name "*.ts" 2>/dev/null | head -10

# Build frontend
RUN npm run build

# Production stage
FROM python:3.11-slim

# Install system dependencies including nginx
RUN apt-get update && apt-get install -y \
    gcc \
    default-libmysqlclient-dev \
    pkg-config \
    curl \
    nginx \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Python backend files
COPY app.py ./
COPY contract_llm.py ./

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/frontend/dist /var/www/html

# Configure nginx to serve frontend and proxy API requests
# Nginx listens on 8080 internally; docker-compose maps host 8080 → 8080
RUN echo 'server { \
    listen 8080; \
    server_name _; \
    root /var/www/html; \
    index index.html; \
    \
    # Serve frontend static files \
    location / { \
        try_files $uri $uri/ /index.html; \
    } \
    \
    # Proxy API requests to FastAPI on port 8001 (strip /api prefix) \
    location /api/ { \
        rewrite ^/api/(.*) /$1 break; \
        proxy_pass http://localhost:8081; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto $scheme; \
        proxy_connect_timeout 60s; \
        proxy_send_timeout 60s; \
        proxy_read_timeout 60s; \
    } \
    \
    # Also proxy non-/api routes that are API endpoints (for backward compatibility) \
    location ~ ^/(health|query|wearable|knowledge|stats|trends|docs|redoc|openapi.json) { \
        proxy_pass http://localhost:8081; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto $scheme; \
    } \
}' > /etc/nginx/sites-available/default

# Create directories
RUN mkdir -p logs

# Create startup script
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# Note: nginx needs to run as root to bind to ports, so we keep root user
# The FastAPI process runs as root but this is acceptable for containerized apps

# Expose port (matches docker-compose.yml mapping 8080:8080)
EXPOSE 8080

# Health check (matches docker-compose.yml)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["./docker-entrypoint.sh"]
