# Multi-stage build for Sleep Coach LLM

# Frontend builder stage
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files first for better caching
COPY frontend/package*.json ./
COPY frontend/package-lock.json* ./

RUN npm ci

# Copy all frontend files
COPY frontend/ .

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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Python backend files
COPY app.py ./
COPY sleep_coach_llm.py ./

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/frontend/dist /var/www/html

# Configure nginx to serve frontend and proxy API requests
# Note: docker-compose.yml maps 8015:8000, so nginx listens on 8000 internally
RUN echo 'server { \
    listen 8000; \
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
        proxy_pass http://localhost:8001; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto $scheme; \
    } \
    \
    # Also proxy non-/api routes that are API endpoints (for backward compatibility) \
    location ~ ^/(health|query|wearable|knowledge|stats|trends|docs|redoc|openapi.json) { \
        proxy_pass http://localhost:8001; \
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

# Expose port (matches docker-compose.yml mapping 8015:8000)
EXPOSE 8000

# Health check (matches docker-compose.yml)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

CMD ["./docker-entrypoint.sh"]
