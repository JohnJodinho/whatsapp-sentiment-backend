#!/bin/bash
set -e

# 1. Start Local Redis (Fallback Broker)
redis-server --daemonize yes --port 6379 --save "" --maxmemory 256mb --maxmemory-policy allkeys-lru

# 2. Start Unified Celery Worker
IS_CELERY_WORKER=true celery -A src.app.celery_app worker -Q sentiment,embeddings --loglevel=info --concurrency=2 &

# 3. Dynamic PORT handling for Heroku and container environments
PORT="${PORT:-8000}"
echo "Starting FastAPI on 0.0.0.0:$PORT"

# 4. Start FastAPI (Foreground)
exec uvicorn src.app.main:app --host 0.0.0.0 --port "$PORT"