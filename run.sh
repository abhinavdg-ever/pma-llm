#!/bin/bash

# Run FastAPI with reload, excluding venv and other unnecessary directories
uvicorn app:app \
  --host 0.0.0.0 \
  --port 8001 \
  --reload \
  --reload-exclude "venv/*" \
  --reload-exclude "*.pyc" \
  --reload-exclude "__pycache__/*" \
  --reload-exclude ".git/*" \
  --reload-exclude "*.log"

# To run WITHOUT reload, just use:
# uvicorn app:app --host 0.0.0.0 --port 8001

