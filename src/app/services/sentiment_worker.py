import time
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

_worker_redis_client = None


def get_worker_redis():
    """Returns a reused Redis client for worker progress and cancellation checks."""
    global _worker_redis_client
    if _worker_redis_client is not None:
        try:
            _worker_redis_client.ping()
            return _worker_redis_client
        except Exception:
            _worker_redis_client = None

    _worker_redis_client = get_redis_client(use_async=False)
    return _worker_redis_client


def should_stop(chat_id: int) -> bool:
    """Checks the currently active Redis for a stop signal using persistent client."""
    try:
        r = get_worker_redis()
        if r.exists(f"stop_signal_{chat_id}"):
            log.info("🛑 Stop signal detected for chat %s", chat_id)
            return True
    except Exception as e:
        log.warning("Could not check stop signal: %s", e)
    return False


def publish_progress(chat_id: int, status_key: str, data: dict):
    """
    Dual-layer publishing to active Redis instance:
    1. Real-time PubSub broadcast for connected SSE listeners.
    2. Durable state snapshot (HSET) so reconnecting clients never lose progress.
    """
    try:
        r = get_worker_redis()
        # 1. Live stream broadcast
        r.publish(f"chat_progress_{chat_id}", json.dumps({"status": status_key, "data": data}))

        # 2. Durable state snapshot
        mapping = {
            "status": status_key,
            "percent": str(data.get("percent", 0)),
            "updated_at": str(time.time()),
        }
        for key in ["messages_done", "messages_total", "segments_done", "segments_total", "total", "error", "message"]:
            if key in data:
                mapping[key] = str(data[key])

        r.hset(f"chat_progress_state:{chat_id}", mapping=mapping)
    except Exception as e:
        log.error("Failed to publish progress: %s", e)


async def _process_batch(
    db,
    chat_id: int,
    buffer: list,
    create_func,
    get_text_func,
    classifier: AfroXLMRMiniSentimentClassifier,
    item_type: str,
    tracker: dict,
):
    if not buffer:
        return

    # Check cancellation before batch compute
    if should_stop(chat_id):
        raise Exception("Cancelled by user")

    # Sort to minimize padding (speedup)
    buffer.sort(key=lambda x: len(get_text_func(x)))
    BATCH_SIZE = 32

    for i in range(0, len(buffer), BATCH_SIZE):
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

    # Update Progress in-memory (0 database queries per batch)
    if item_type == "message":
        tracker["messages_scored"] += len(buffer)
    else:
        tracker["segments_scored"] += len(buffer)

    total = tracker["total"]
    done = tracker["messages_scored"] + tracker["segments_scored"]
    percent = int(100 * (done / total)) if total > 0 else 0

    publish_progress(
        chat_id,
        "progress",
        {
            "percent": percent,
            "messages_done": tracker["messages_scored"],
            "messages_total": tracker["messages_total"],
            "segments_done": tracker["segments_scored"],
            "segments_total": tracker["segments_total"],
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

            # Query baseline totals once from DB for in-memory tracker
            initial_progress = await crud.get_sentiment_progress(db, chat_id)
            tracker = {
                "messages_total": initial_progress["messages_total"],
                "messages_scored": initial_progress["messages_scored"],
                "segments_total": initial_progress["segments_total"],
                "segments_scored": initial_progress["segments_scored"],
                "total": initial_progress["messages_total"] + initial_progress["segments_total"],
            }

            try:
                # 1. Process Messages
                buffer = []
                async for item in crud.stream_unscored_messages(db, chat_id):
                    buffer.append(item)
                    if len(buffer) >= 100:
                        await _process_batch(
                            db, chat_id, buffer, crud.create_message_sentiment, lambda x: x.content, classifier, "message", tracker
                        )
                        buffer = []
                if buffer:
                    await _process_batch(
                        db, chat_id, buffer, crud.create_message_sentiment, lambda x: x.content, classifier, "message", tracker
                    )

                # 2. Process Segments
                buffer = []
                async for item in crud.stream_unscored_sender_segments(db, chat_id):
                    buffer.append(item)
                    if len(buffer) >= 100:
                        await _process_batch(
                            db, chat_id, buffer, crud.create_segment_sentiment, lambda x: x.combined_text, classifier, "segment", tracker
                        )
                        buffer = []
                if buffer:
                    await _process_batch(
                        db, chat_id, buffer, crud.create_segment_sentiment, lambda x: x.combined_text, classifier, "segment", tracker
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