import asyncio
import logging
import json
import os
import psutil  
from typing import List, Any
from celery import Celery
import redis
import time
from src.app.config import settings
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline
from onnxruntime import SessionOptions, GraphOptimizationLevel
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool 
from src.app import crud, models


try:
    from src.app.config import settings
    DATABASE_URL = str(settings.DATABASE_URL)
except ImportError:
    DATABASE_URL = os.getenv("DATABASE_URL")

from src.app import crud, models

log = logging.getLogger(__name__)


BROKER_URL = settings.CELERY_BROKER_URL
BACKEND_URL = settings.CELERY_RESULT_BACKEND
celery_app = Celery(
    "sentiment_worker",
    broker=BROKER_URL,
    backend=BACKEND_URL,
    include=['src.app.services.embedding_worker']
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1 
)

# Optimized Model Loading 
_PIPELINE = None
_MODEL_PATH = "./onnx_model_optimized"
_REDIS_CLIENT = None

def get_pipeline():
    """
    Loads model with CPU-Specific Optimizations.
    """
    global _PIPELINE
    if _PIPELINE is None:
        log.info(f"Loading ONNX model from {_MODEL_PATH}...")
        try:
            
            n_threads = os.cpu_count() or 4
            
            sess_options = SessionOptions()
            sess_options.intra_op_num_threads = n_threads
            sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
            
            ort_model = ORTModelForSequenceClassification.from_pretrained(
                _MODEL_PATH,
                session_options=sess_options
            )
            tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH)
            
            _PIPELINE = pipeline(
                "text-classification",
                model=ort_model,
                tokenizer=tokenizer,
                top_k=None,
                device=-1, # CPU
                truncation=True,
                max_length=512
            )
            log.info(f"ONNX Model loaded (Threads: {n_threads}).")
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise e
    return _PIPELINE

# Redis Pub/Sub Helper 
def publish_progress(chat_id: int, status_key: str, data: dict):
    global _REDIS_CLIENT
    try:
        if _REDIS_CLIENT is None:
            _REDIS_CLIENT = redis.from_url(BROKER_URL, decode_responses=True)

        
        channel = f"chat_progress_{chat_id}"
        message = {"status": status_key, "data": data}
        listeners = _REDIS_CLIENT.publish(channel, json.dumps(message))

        if listeners > 0:
            log.info(f"SENT {status_key} to Chat {chat_id} (Listeners: {listeners}) | {data.get('percent', '')}%")
        else:
            log.warning(f"SENT {status_key} to Chat {chat_id} but NO LISTENERS connected (Frontend might be disconnected)")
    except Exception as e:
        log.error(f"Redis Publish Error: {e}")
        _REDIS_CLIENT = None # Reset on error


# SMART BATCHING LOGIC (speed optimized)
async def _process_smart_buffer(db, chat_id, buffer, create_func, get_text_func, pipe):
    """
    Sorts the buffer by text length and processes in mini-batches.
    To enables effective Dynamic Padding.
    """
    # Sort by length of content (Shortest -> Longest)
    # This ensures a batch of 5-word messages is only padded to ~5 words, not 512.
    buffer.sort(key=lambda x: len(get_text_func(x)))
    
    BATCH_SIZE = 8
    start_time = time.time()

    for i in range(0, len(buffer), BATCH_SIZE):
        batch_items = buffer[i : i + BATCH_SIZE]
        texts = [get_text_func(item) for item in batch_items]

        preds = pipe(texts, batch_size=len(texts), padding=True, truncation=True)

        for item_obj, pred in zip(batch_items, preds):
            p = pred[0] if isinstance(pred, list) else pred
            payload = {
                "overall_label": p.get("label"),
                "overall_label_score": p.get("score_positive", p["score"]) if isinstance(p, dict) else p["score"],
            }
            await create_func(db, item_obj.id, payload, should_commit=False)
    inference_time = time.time() - start_time
    log.info(f"Inference finished in {inference_time:.2f}s. Committing to DB...")
    
    await db.commit()
    
    # Progress rep sending...
    progress = await crud.get_sentiment_progress(db, chat_id)
    total = progress["messages_total"] + progress["segments_total"]
    done = progress["messages_scored"] + progress["segments_scored"]
    percent = int(100 * (done / total)) if total > 0 else 0
    
    publish_progress(chat_id, "progress", {
        "percent": percent,
        "messages_done": progress["messages_scored"],
        "messages_total": progress["messages_total"],
        "segments_done": progress["segments_scored"],
        "segments_total": progress["segments_total"]
    })

async def _process_stage_batch(db, chat_id, stream_func, create_func, get_text_func, pipe, buffer_size=1000):

    buffer = []
    
    # Fetch chat once for cancel check
    chat = await crud.get_chat(db, chat_id)
    
    item_count = 0
    async for item in stream_func(db, chat_id):
        item_count += 1
        if len(buffer) % 50 == 0:
            await db.refresh(chat)
            if chat.cancel_requested:
                log.warning(f"Cancellation flag detected for Chat {chat_id}")
                raise Exception("Cancelled by user")

        buffer.append(item)
        
        if len(buffer) >= buffer_size:
            log.info(f"Buffere full ({len(buffer)}). Triggering batch process.")
            await _process_smart_buffer(db, chat_id, buffer, create_func, get_text_func, pipe)
            buffer = [] 

    # Take care of remiaining
    if buffer:
        log.info(f"Processing final buffer of {len(buffer)} items.")
        await db.refresh(chat)
        if chat.cancel_requested:
             raise Exception("Cancelled by user")
        await _process_smart_buffer(db, chat_id, buffer, create_func, get_text_func, pipe)
    if item_count == 0:
        log.info(f"No items found in this stage for chat {chat_id}.")


async def process_chat_logic(chat_id: int):
    pipe = get_pipeline()
    
    worker_engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    WorkerSession = async_sessionmaker(worker_engine, expire_on_commit=False)

    try:
        async with WorkerSession() as db:
            chat = await crud.get_chat(db, chat_id)
            if not chat: 
                log.error(f"Chat {chat_id} not found in DB.")
                return

            if chat.sentiment_status != models.SentimentStatusEnum.processing.value:
                log.info(f"Setting Chat {chat_id} status to PROCESSING.")
                chat.sentiment_status = models.SentimentStatusEnum.processing.value
                db.add(chat)
                await db.commit()

            try:
                log.info(f"Starting Messages for Chat {chat_id} (Smart Batching)")
                await _process_stage_batch(
                    db=db, 
                    chat_id=chat_id,
                    stream_func=crud.stream_unscored_messages,
                    create_func=crud.create_message_sentiment,
                    get_text_func=lambda x: x.content,
                    pipe=pipe,
                    buffer_size=1000 
                )

                log.info(f"🚀 Starting Segments for Chat {chat_id}")
                await _process_stage_batch(
                    db=db, 
                    chat_id=chat_id,
                    stream_func=crud.stream_unscored_sender_segments,
                    create_func=crud.create_segment_sentiment,
                    get_text_func=lambda x: x.combined_text,
                    pipe=pipe,
                    buffer_size=200 
                )
                log.info(f"[Chat {chat_id}] All stages done. Finalizing...")
                final_progress = await crud.get_sentiment_progress(db, chat_id)
                chat.sentiment_status = models.SentimentStatusEnum.completed.value
                db.add(chat)
                await db.commit()
                
                publish_progress(chat_id, "completed", {
                    "percent": 100, 
                    "status": "done",
                    "messages_done": final_progress["messages_scored"],
                    "messages_total": final_progress["messages_total"],
                    "segments_done": final_progress["segments_scored"],
                    "segments_total": final_progress["segments_total"]
                })
                log.info(f"Chat {chat_id} DONE.")

            except Exception as e:
                log.error(f"Error processing chat {chat_id}: {e}", exc_info=True)
                await db.rollback()
                if str(e) == "Cancelled by user":
                    log.info(f"Chat {chat_id} stopped due to cancellation.")
                else:
                    chat.sentiment_status = models.SentimentStatusEnum.failed.value
                    db.add(chat)
                    await db.commit()
                    publish_progress(chat_id, "error", {"error": str(e)})
                raise e
    finally:
        
        await worker_engine.dispose()

# Entry point
@celery_app.task(name="analyze_sentiment", bind=True, max_retries=3)
def analyze_sentiment_task(self, chat_id: int):
    log.info(f"Received task: analyze_sentiment for Chat {chat_id}")
    try:
        asyncio.run(process_chat_logic(chat_id))
    except Exception as exc:
        if str(exc) == "Cancelled by user":
            return 
        log.warning(f"Task failed,  retrying... Error: {exc}")
        self.retry(exc=exc, countdown=5 * (self.request.retries + 1))