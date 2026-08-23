#!/bin/bash

# 1. Start Local Redis (Fallback Broker)
# daemonize yes allows this to run in the background.
# maxmemory-policy allkeys-lru ensures it doesn't OOM if many signals pile up.
redis-server --daemonize yes --port 6379 --save "" --maxmemory 256mb --maxmemory-policy allkeys-lru

# 2. Start Unified Celery Worker
# IS_CELERY_WORKER env var can be used for internal logic/routing.
# Concurrency 2 matches the 2 vCPUs usually available on HF standard spaces.
IS_CELERY_WORKER=true celery -A src.app.celery_app worker -Q sentiment,embeddings --loglevel=info --concurrency=2 &

# 3. Start FastAPI (Foreground)
# exec ensures that uvicorn receives OS signals (like SIGTERM) for clean shutdown.
exec uvicorn src.app.main:app --host 0.0.0.0 --port 7860