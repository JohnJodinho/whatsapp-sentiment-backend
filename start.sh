#!/bin/bash
set -e

PORT="${PORT:-8000}"

# Only attempt local Redis/Celery if GitHub Actions worker is not enabled
if [ "$USE_GITHUB_ACTIONS_WORKER" != "true" ]; then
    if command -v redis-server >/dev/null 2>&1; then
        echo "Starting local Redis broker..."
        redis-server --daemonize yes --port 6379 --save "" --maxmemory 256mb --maxmemory-policy allkeys-lru || true
    fi

    if [ "$ENABLE_LOCAL_CELERY" = "true" ]; then
        echo "Starting local Celery worker..."
        IS_CELERY_WORKER=true celery -A src.app.celery_app worker -Q sentiment,embeddings --loglevel=info --concurrency=1 &
    fi
else
    echo "GitHub Actions Distributed Workers enabled. Running in lightweight API mode."
fi

echo "Starting FastAPI on 0.0.0.0:$PORT"
exec uvicorn src.app.main:app --host 0.0.0.0 --port "$PORT"