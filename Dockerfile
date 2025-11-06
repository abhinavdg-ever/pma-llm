# Multi-stage build for React frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files first for better caching
COPY frontend/package*.json ./
COPY frontend/package-lock.json* ./
RUN npm ci

# Copy all frontend files (including src/lib/utils.ts)
COPY frontend/ .

# Verify utils.ts exists before build
RUN ls -la src/lib/ || echo "Warning: src/lib directory not found"

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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Python backend files
COPY app.py .
COPY sleep_coach_llm.py .

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/frontend/dist /var/www/html

# Configure nginx to serve frontend and proxy API requests
RUN echo 'server { \
    listen 80; \
    server_name _; \
    root /var/www/html; \
    index index.html; \
    \
    # Serve frontend static files \
    location / { \
        try_files $uri $uri/ /index.html; \
    } \
    \
    # Proxy API requests to FastAPI (strip /api prefix) \
    location /api/ { \
        rewrite ^/api/(.*) /$1 break; \
        proxy_pass http://localhost:8000; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto $scheme; \
    } \
    \
    # Also proxy non-/api routes that are API endpoints (for backward compatibility) \
    location ~ ^/(health|query|wearable|knowledge|stats|trends|docs|redoc|openapi.json) { \
        proxy_pass http://localhost:8000; \
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

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost/api/health || exit 1

CMD ["./docker-entrypoint.sh"]
