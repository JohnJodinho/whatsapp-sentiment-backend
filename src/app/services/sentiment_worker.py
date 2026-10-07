# src/app/services/sentiment_worker.py

import asyncio
import logging
import json
import os
import redis
from redis import ConnectionPool
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from src.app import crud, models
from src.app.celery_app import celery_app
from src.app.config import settings
from src.app.utils.redis_helper import get_redis_client
from src.app.services.sentiment_service import get_sentiment_classifier, AfroXLMRMiniSentimentClassifier

log = logging.getLogger(__name__)

# --- CONFIG ---
DATABASE_URL = str(settings.DATABASE_URL)
BROKER_URL = settings.CELERY_BROKER_URL


def should_stop(chat_id: int) -> bool:
    """Checks the currently active Redis for a stop signal."""
    try:
        r = get_redis_client()
        if r.exists(f"stop_signal_{chat_id}"):
            log.info("🛑 Stop signal detected for chat %s", chat_id)
            return True
    except Exception as e:
        log.warning("Could not check stop signal: %s", e)
    return False


def publish_progress(chat_id: int, status_key: str, data: dict):
    """Resilient publishing to active Redis instance."""
    try:
        r = get_redis_client()
        r.publish(f"chat_progress_{chat_id}", json.dumps({"status": status_key, "data": data}))
    except Exception as e:
        log.error("Failed to publish progress: %s", e)


async def _process_batch(
    db,
    chat_id: int,
    buffer: list,
    create_func,
    get_text_func,
    classifier: AfroXLMRMiniSentimentClassifier,
):
    if not buffer:
        return

    # Sort to minimize padding (speedup)
    buffer.sort(key=lambda x: len(get_text_func(x)))
    BATCH_SIZE = 32

    for i in range(0, len(buffer), BATCH_SIZE):
        if should_stop(chat_id):
            raise Exception("Cancelled by user")

        batch_items = buffer[i : i + BATCH_SIZE]
        texts = [get_text_func(item) for item in batch_items]

        # Run INT8 ONNX inference
        preds = classifier.predict_sync(texts)

        for item_obj, pred in zip(batch_items, preds):
            payload = {
                "overall_label": pred["overall_label"],
                "overall_label_score": pred["overall_label_score"],
                "score_positive": pred.get("score_positive"),
                "score_negative": pred.get("score_negative"),
                "score_neutral": pred.get("score_neutral"),
            }
            await create_func(db, item_obj.id, payload, should_commit=False)

    # Commit once per buffer
    await db.commit()

    # Update Progress
    progress = await crud.get_sentiment_progress(db, chat_id)
    total = progress["messages_total"] + progress["segments_total"]
    done = progress["messages_scored"] + progress["segments_scored"]
    percent = int(100 * (done / total)) if total > 0 else 0

    publish_progress(
        chat_id,
        "progress",
        {
            "percent": percent,
            "messages_done": progress["messages_scored"],
            "messages_total": progress["messages_total"],
            "segments_done": progress["segments_scored"],
            "segments_total": progress["segments_total"],
            "total": total,
        },
    )


async def process_chat_logic(chat_id: int):
    classifier = get_sentiment_classifier()
    worker_engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    WorkerSession = async_sessionmaker(worker_engine, expire_on_commit=False)

    try:
        async with WorkerSession() as db:
            await crud.update_chat_status(db, chat_id, models.SentimentStatusEnum.processing.value)

            try:
                # 1. Process Messages
                buffer = []
                async for item in crud.stream_unscored_messages(db, chat_id):
                    buffer.append(item)
                    if len(buffer) >= 100:
                        await _process_batch(
                            db, chat_id, buffer, crud.create_message_sentiment, lambda x: x.content, classifier
                        )
                        buffer = []
                if buffer:
                    await _process_batch(
                        db, chat_id, buffer, crud.create_message_sentiment, lambda x: x.content, classifier
                    )

                # 2. Process Segments
                buffer = []
                async for item in crud.stream_unscored_sender_segments(db, chat_id):
                    buffer.append(item)
                    if len(buffer) >= 100:
                        await _process_batch(
                            db, chat_id, buffer, crud.create_segment_sentiment, lambda x: x.combined_text, classifier
                        )
                        buffer = []
                if buffer:
                    await _process_batch(
                        db, chat_id, buffer, crud.create_segment_sentiment, lambda x: x.combined_text, classifier
                    )

                # 3. Complete
                await crud.update_chat_status(db, chat_id, models.SentimentStatusEnum.completed.value)
                publish_progress(chat_id, "completed", {"percent": 100, "status": "done"})

            except Exception as e:
                await db.rollback()
                if str(e) == "Cancelled by user":
                    log.info("Chat %s Cancelled.", chat_id)
                    await crud.update_chat_status(db, chat_id, models.SentimentStatusEnum.cancelled.value)
                else:
                    log.error("Sentiment processing error: %s", e, exc_info=True)
                    try:
                        await crud.update_chat_status(db, chat_id, models.SentimentStatusEnum.failed.value)
                    except Exception as update_err:
                        log.error("Failed to update status to failed for Chat %s: %s", chat_id, update_err)
                    publish_progress(chat_id, "error", {"error": str(e)})
                    raise
    finally:
        await worker_engine.dispose()


@celery_app.task(name="src.app.services.sentiment_worker.analyze_sentiment_task", bind=True)
def analyze_sentiment_task(self, chat_id: int):
    try:
        asyncio.run(process_chat_logic(chat_id))
    except Exception as e:
        log.error("Sentiment task failed for Chat %s: %s", chat_id, e, exc_info=True)
        raise