# src/app/services/sentiment_worker_local.py

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
from src.app.services.sentiment_service import get_sentiment_classifier, AfroXLMRMiniSentimentClassifier

log = logging.getLogger(__name__)

DATABASE_URL = str(settings.DATABASE_URL)
BROKER_URL = settings.CELERY_BROKER_URL
redis_pool = ConnectionPool.from_url(BROKER_URL, decode_responses=True)


def get_redis_sync():
    """Get a sync redis client from pool."""
    return redis.Redis(connection_pool=redis_pool)


def should_stop(chat_id: int) -> bool:
    try:
        r = get_redis_sync()
        if r.exists(f"stop_signal_{chat_id}"):
            return True
    except Exception:
        pass
    return False


def publish_progress(chat_id: int, status_key: str, data: dict):
    try:
        r = get_redis_sync()
        channel = f"chat_progress_{chat_id}"
        message = {"status": status_key, "data": data}
        r.publish(channel, json.dumps(message))
    except Exception as e:
        log.error("Redis Publish Error: %s", e)


async def _process_smart_buffer(db, chat_id, buffer, create_func, get_text_func, classifier: AfroXLMRMiniSentimentClassifier):
    if not buffer:
        return

    buffer.sort(key=lambda x: len(get_text_func(x)))
    BATCH_SIZE = 32
    processed_count = 0
    UPDATE_FREQUENCY = 25

    for i in range(0, len(buffer), BATCH_SIZE):
        if should_stop(chat_id):
            raise Exception("Cancelled by user")

        batch_items = buffer[i : i + BATCH_SIZE]
        texts = [get_text_func(item) for item in batch_items]

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

        processed_count += len(batch_items)

        if processed_count % UPDATE_FREQUENCY == 0 or processed_count == len(buffer):
            await db.commit()
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


async def _process_stage_batch(db, chat_id, stream_func, create_func, get_text_func, classifier, buffer_size=100):
    buffer = []
    async for item in stream_func(db, chat_id):
        if len(buffer) % 50 == 0 and should_stop(chat_id):
            raise Exception("Cancelled by user")

        buffer.append(item)
        if len(buffer) >= buffer_size:
            await _process_smart_buffer(db, chat_id, buffer, create_func, get_text_func, classifier)
            buffer = []

    if buffer:
        await _process_smart_buffer(db, chat_id, buffer, create_func, get_text_func, classifier)


async def process_chat_logic(chat_id: int):
    classifier = get_sentiment_classifier()
    worker_engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    WorkerSession = async_sessionmaker(worker_engine, expire_on_commit=False)

    try:
        async with WorkerSession() as db:
            chat = await crud.get_chat(db, chat_id)
            if not chat:
                return

            if should_stop(chat_id) or chat.cancel_requested:
                raise Exception("Cancelled by user")

            if chat.sentiment_status != models.SentimentStatusEnum.processing.value:
                chat.sentiment_status = models.SentimentStatusEnum.processing.value
                db.add(chat)
                await db.commit()

            try:
                await _process_stage_batch(
                    db=db,
                    chat_id=chat_id,
                    stream_func=crud.stream_unscored_messages,
                    create_func=crud.create_message_sentiment,
                    get_text_func=lambda x: x.content,
                    classifier=classifier,
                    buffer_size=100,
                )

                await _process_stage_batch(
                    db=db,
                    chat_id=chat_id,
                    stream_func=crud.stream_unscored_sender_segments,
                    create_func=crud.create_segment_sentiment,
                    get_text_func=lambda x: x.combined_text,
                    classifier=classifier,
                    buffer_size=100,
                )

                chat.sentiment_status = models.SentimentStatusEnum.completed.value
                db.add(chat)
                await db.commit()

                publish_progress(chat_id, "completed", {"percent": 100, "status": "done"})

            except Exception as e:
                await db.rollback()
                if str(e) == "Cancelled by user":
                    log.info("Chat %s stopped via Kill Switch.", chat_id)
                else:
                    log.error("Processing failed: %s", e, exc_info=True)
                    chat.sentiment_status = models.SentimentStatusEnum.failed.value
                    db.add(chat)
                    await db.commit()
                    publish_progress(chat_id, "error", {"error": str(e)})
                raise e
    finally:
        await worker_engine.dispose()


@celery_app.task(name="src.app.services.sentiment_worker.analyze_sentiment_task", bind=True, max_retries=3)
def analyze_sentiment_task(self, chat_id: int):
    try:
        asyncio.run(process_chat_logic(chat_id))
    except Exception as exc:
        if str(exc) == "Cancelled by user":
            return
        self.retry(exc=exc, countdown=5)