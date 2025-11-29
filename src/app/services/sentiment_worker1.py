import asyncio
from asyncio import Queue
from typing import List, AsyncGenerator, TypeVar, Callable, Coroutine, Any
from sqlalchemy.ext.asyncio import AsyncSession
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline
from sqlalchemy.exc import DBAPIError
from asyncpg.exceptions import ConnectionDoesNotExistError
from src.app.db.session import AsyncSessionLocal
from src.app import crud, models
import time
import logging

log = logging.getLogger(__name__)

# --- Model & Pipeline Config ---
_PIPELINE = None
_MODEL_PATH = "./onnx_model_optimized"
_DEVICE = "cpu"

# --- Performance Tuning ---
# Max batches to pre-fetch from the DB. Prevents high memory use.
# This provides "backpressure" on the database streaming.
QUEUE_MAX_SIZE = 10 

# How many batches to process before committing to the DB.
# This makes the SSE endpoint responsive.
COMMIT_CHUNK_SIZE = 20

# --- Custom Exception ---
class AnalysisCancelledException(Exception):
    """Custom exception to signal a safe cancellation."""
    pass

# --- Batch Generator (Unchanged) ---
T = TypeVar('T')
async def batch_generator(stream: AsyncGenerator[T, None], size: int) -> AsyncGenerator[List[T], None]:
    """Yields batches of a given size from an async stream."""
    batch = []
    async for item in stream:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch: 
        yield batch

# --- ML Model Loading (Unchanged) ---
def _load_pipeline():
    """Loads the ONNX pipeline, caching it globally."""
    global _PIPELINE
    if _PIPELINE is None:
        log.info(f"Loading ONNX model from {_MODEL_PATH} on {_DEVICE.upper()}...")
        ort_model = ORTModelForSequenceClassification.from_pretrained(_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH)
        _PIPELINE = pipeline(
            "text-classification", 
            model=ort_model, 
            tokenizer=tokenizer, 
            top_k=None, 
            device=-1 if _DEVICE=="cpu" else 0, 
            truncation=True, 
            max_length=512
        )
        log.info("✅ Model loaded successfully.")
    return _PIPELINE

# --- ML Inference (Unchanged) ---
def _predict_batch_sync(pipe, texts: List[str], batch_size: int):
    """Runs the CPU-bound inference in a separate thread."""
    return pipe(texts, batch_size=batch_size)

async def _predict_batch(pipe, texts: List[str], batch_size: int):
    """Async wrapper for the inference function."""
    return await asyncio.to_thread(_predict_batch_sync, pipe, texts, batch_size)


# -----------------------------------------------------------------
# 🚀 NEW: REUSABLE PROCESSING STAGE HELPER
# -----------------------------------------------------------------
async def _process_stage(
    db: AsyncSession,
    pipe: Any,
    stream: AsyncGenerator[Any, None],
    create_sentiment_func: Callable[..., Coroutine[Any, Any, None]],
    get_content_func: Callable[[Any], str],
    chat_id: int,
    batch_size: int,
    stage_name: str,
    check_cancel_func: Callable[[], Coroutine[Any, Any, None]]
):
    """
    A generic, parallelized worker for processing a stream of data.
    
    This function NO LONGER has its own try/except block. It lets the
    main 'queue_sentiment_analysis' function handle all exceptions,
    which correctly manages the session rollback.
    """
    log.info(f"[Chat {chat_id}] Starting parallel {stage_name} processing...")
    queue: Queue[List[Any] | None | Exception] = Queue(maxsize=QUEUE_MAX_SIZE)
    
    # --- 1. Producer (Fetcher) Task ---
    async def _fetcher():
        """Fetches batches from the DB stream and puts them in the queue."""
        try:
            async for batch in batch_generator(stream, batch_size):
                await queue.put(batch)
            await queue.put(None) # Signal "Done"
        except Exception as e:
            log.error(f"[Chat {chat_id}] Error during DB fetch in {stage_name}: {e}")
            await queue.put(e) # Send exception to worker
            
    fetch_task = asyncio.create_task(_fetcher())
    batch_index = 0
    total_processed = 0

    # --- 2. Consumer (Worker) Loop ---
    # This loop gets pre-fetched batches from the queue and processes them.
    while True:
        batch = await queue.get()

        if batch is None:
            break # Fetcher is done
        if isinstance(batch, Exception):
            raise batch # Raise the error from the fetcher

        if batch_index % 20 == 0:
            await check_cancel_func()

        texts = [get_content_func(item) for item in batch]
        preds = await _predict_batch(pipe, texts, batch_size=len(texts))
        print(f"[Chat {chat_id}] Processed {total_processed + len(batch)} {stage_name}...")

        for item, pred_list in zip(batch, preds):
            p = pred_list[0] if isinstance(pred_list, list) else pred_list
            payload = {
                "overall_label": p.get("label"),
                "overall_label_score": p.get("score_positive", p["score"]) if isinstance(p, dict) else p["score"],
                "score_negative": None, "score_neutral": None, "score_positive": None,
                "sentences_summary": None, "opinions": None, "api_version": "onnx_local",
                "analysis_timestamp": None, "error_code": None, "error_message": None
            }
            
            # This line can now fail (e.g., ConnectionDoesNotExistError)
            # and the exception will be caught by the *main* handler.
            await create_sentiment_func(db, item.id, payload, should_commit=False)
        
        total_processed += len(batch)
        
        if (batch_index + 1) % COMMIT_CHUNK_SIZE == 0:
            log.info(f"[Chat {chat_id}] Committing {stage_name} chunk... (Processed {total_processed} total)")
            await db.commit()

        batch_index += 1

    # --- 3. Cleanup & Final Commit ---
    if not fetch_task.done():
        fetch_task.cancel()
    
    # Commit any remaining items that didn't make a full chunk
    log.info(f"[Chat {chat_id}] Committing final {stage_name} batch...")
    await db.commit() 
    
    log.info(f"[Chat {chat_id}] {stage_name.capitalize()} stage complete. Processed {total_processed} items.")
# -----------------------------------------------------------------
# REFACTORED MAIN WORKER
# -----------------------------------------------------------------
async def queue_sentiment_analysis(chat_id: int, batch_size: int = 32):
    """
    Runs the full sentiment analysis job for a chat.
    This function now throws exceptions (DBAPIError, etc.) on failure,
    to be caught by the retry supervisor.
    """
    pipe = _load_pipeline()
    start_time = time.time()

    async def _check_cancel():
        async with AsyncSessionLocal() as check_db:
            cancelled = await crud.is_chat_cancelled(check_db, chat_id)
            if cancelled:
                log.info(f"Cancellation requested for chat with id={chat_id}. Aborting safely...")
                raise AnalysisCancelledException()

    # This session is now managed by the worker.
    # If it fails, the session is discarded, and the supervisor creates a new one.
    async with AsyncSessionLocal() as db:
        try:
            chat = await crud.get_chat(db, chat_id=chat_id)
            if not chat:
                log.warning("queue_sentiment_analysis: chat %s not found", chat_id)
                return
            
            # This is key to resuming: we only run if not completed.
            if chat.sentiment_status == models.SentimentStatusEnum.completed.value:
                log.info("Chat %s is already completed. Skipping.", chat_id)
                return
            
            # If status is "failed", we are retrying, so set to "processing"
            if chat.sentiment_status != models.SentimentStatusEnum.processing.value:
                chat.sentiment_status = models.SentimentStatusEnum.processing.value
                db.add(chat)
                await db.commit()

            # --- Message-level inference ---
            msg_stream = crud.stream_unscored_messages(db, chat_id)
            await _process_stage(
                db=db, pipe=pipe, stream=msg_stream,
                create_sentiment_func=crud.create_message_sentiment,
                get_content_func=lambda m: m.content,
                chat_id=chat_id, batch_size=batch_size,
                stage_name="messages", check_cancel_func=_check_cancel
            )

            # --- Segment-level inference ---
            seg_stream = crud.stream_unscored_sender_segments(db, chat_id)
            await _process_stage(
                db=db, pipe=pipe, stream=seg_stream,
                create_sentiment_func=crud.create_segment_sentiment,
                get_content_func=lambda s: s.combined_text,
                chat_id=chat_id, batch_size=batch_size,
                stage_name="segments", check_cancel_func=_check_cancel
            )

            # --- Final Status Update ---
            progress = await crud.get_sentiment_progress(db, chat_id)
            if progress["messages_total"] == progress["messages_scored"] and \
               progress["segments_total"] == progress["segments_scored"]:
                chat.sentiment_status = models.SentimentStatusEnum.completed.value
            else:
                chat.sentiment_status = models.SentimentStatusEnum.processing.value
            
            db.add(chat)
            await db.commit()
            
            end_time = time.time()
            log.info(f"✅ Finished sentiment job for chat %s in %.2f seconds", chat_id, end_time - start_time)

        except AnalysisCancelledException:
            # --- FIX 2: This is the ONLY exception we catch here ---
            # We must roll back the session to discard any partial work
            await db.rollback()
            log.info(f"Job for chat {chat_id} was cancelled. Setting status to 'cancelled'.")
            async with AsyncSessionLocal() as update_db:
                await crud.update_chat_status(
                    update_db, 
                    chat_id, 
                    models.SentimentStatusEnum.cancelled.value
                )
            # Re-raise so the supervisor knows it was a clean exit
            raise


async def run_sentiment_job_with_retries(chat_id: int, batch_size: int = 32):
    """
    A supervisor function that runs the sentiment worker with a retry policy.
    This is the function that should be passed to BackgroundTasks.
    """
    MAX_RETRIES = 3
    BASE_DELAY_SECONDS = 10

    for attempt in range(MAX_RETRIES):
        try:
            # Try to run the whole job
            await queue_sentiment_analysis(chat_id, batch_size)
            
            # If it returns without error, it's done.
            log.info(f"[Supervisor Chat {chat_id}] Job completed successfully.")
            return # Success!

        except (DBAPIError, ConnectionDoesNotExistError) as e:
            # --- This is a retriable network error ---
            log.warning(
                f"[Supervisor Chat {chat_id}] Network error on attempt {attempt + 1}/{MAX_RETRIES}: {e}"
            )
            if attempt == MAX_RETRIES - 1:
                log.error(f"[Supervisor Chat {chat_id}] Job failed after {MAX_RETRIES} attempts.")
                break # Break loop to fall through to final failure logic
            
            delay = BASE_DELAY_SECONDS * (2 ** attempt) # 10s, 20s, 40s
            log.info(f"[Supervisor Chat {chat_id}] Retrying in {delay} seconds...")
            await asyncio.sleep(delay)

        except AnalysisCancelledException:
            # --- The user cancelled ---
            log.info(f"[Supervisor Chat {chat_id}] Job was cancelled by user. Stopping.")
            return # This is a clean exit, not a failure.

        except Exception as e:
            # --- This is a non-retriable logic error ---
            log.error(
                f"[Supervisor Chat {chat_id}] A non-retriable error occurred: {e}", 
                exc_info=True
            )
            break # Break loop to fall through to final failure logic

    # --- If we break the loop (all retries failed or non-retriable error) ---
    log.error(f"[Supervisor Chat {chat_id}] Setting chat status to 'failed'.")
    try:
        async with AsyncSessionLocal() as error_db:
            await crud.update_chat_status(
                error_db, 
                chat_id, 
                models.SentimentStatusEnum.failed.value
            )
    except Exception as e:
        log.critical(
            f"[Supervisor Chat {chat_id}] CRITICAL: FAILED TO SET FAILED STATUS. {e}", 
            exc_info=True
        )