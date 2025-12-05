# src/app/celery_app.py

import os
import ssl
from celery import Celery
from src.app.config import settings

# 1. Define the Celery App Instance
celery_app = Celery("chat_processor")

# 2. Robust Azure Redis Configuration
# These settings are critical to prevent "Connection closed by server"
broker_transport_options = {
    'visibility_timeout': 3600,  # 1 hour
    'socket_timeout': 30,        # Fail fast if socket hangs
    'socket_connect_timeout': 30,
    'socket_keepalive': True,    # TCP Keepalive
    'health_check_interval': 15  # Ping Redis every 15s (Crucial for Azure)
}

celery_app.conf.update(
    broker_url=settings.CELERY_BROKER_URL,
    result_backend=settings.CELERY_RESULT_BACKEND,
    broker_transport_options=broker_transport_options,
    
    # Task Settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    
    # Worker Settings
    worker_cancel_long_running_tasks_on_connection_loss=True,
    worker_prefetch_multiplier=1,
    
    # 3. Explicit Task Routing
    # This ensures tasks automatically go to the right queue
    task_routes={
        'src.app.services.embedding_worker.generate_embeddings_task': {'queue': 'embeddings'},
        'src.app.services.sentiment_worker.analyze_sentiment_task': {'queue': 'sentiment'},
    }
)