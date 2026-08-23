# src/app/celery_app.py

import os
import ssl
from celery import Celery
from src.app.config import settings


# Priority list for the broker
BROKER_FAILOVER_LIST = [str(settings.CELERY_BROKER_URL), "redis://localhost:6379/0"]

celery_app = Celery(
    "chat_processor",
    broker=BROKER_FAILOVER_LIST, # Celery will try these in order
    include=["src.app.services.sentiment_worker", "src.app.services.embedding_worker"]
)

ssl_options = {
    'ssl_cert_reqs': ssl.CERT_NONE
}

celery_app.conf.update(
    # Broker & Backend Settings
    result_backend=settings.CELERY_RESULT_BACKEND,
    broker_use_ssl=ssl_options,
    redis_backend_use_ssl=ssl_options,
    
    # Critical Connection Resilience
    broker_transport_options={
        'visibility_timeout': 3600,
        'socket_timeout': 30,
        'socket_keepalive': True,
        'health_check_interval': 15, # Ping every 15s to keep Upstash alive
        'retry_on_timeout': True,
    },

    # Task Settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True, # Critical: allows continuation after fallback
    
    # Worker Settings
    worker_prefetch_multiplier=1,
    worker_cancel_long_running_tasks_on_connection_loss=True,
    
    # Routing
    task_routes={
        'src.app.services.embedding_worker.generate_embeddings_task': {'queue': 'embeddings'},
        'src.app.services.sentiment_worker.analyze_sentiment_task': {'queue': 'sentiment'},
    }
)