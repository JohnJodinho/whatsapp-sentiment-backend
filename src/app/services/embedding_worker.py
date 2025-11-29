# src/app/services/embedding_worker.py

import asyncio
import logging
from src.app.services.sentiment_worker import celery_app # Re-use the app instance!
from src.app.services.embedding_service import run_embedding_job_with_retries 

log = logging.getLogger(__name__)

@celery_app.task(name="generate_embeddings", bind=True)
def generate_embeddings_task(self, chat_id: int):
    """
    Celery wrapper for the embedding service.
    """
    try:
        log.info(f"Starting embeddings for Chat {chat_id}")
        # Run the async service function
        asyncio.run(run_embedding_job_with_retries(chat_id))
        log.info(f"✅ Embeddings finished for Chat {chat_id}")
    except Exception as e:
        log.error(f"Embedding task failed: {e}")
        # Optional: Set DB status to failed here if your service doesn't handle it
        raise e