# src/app/services/embedding_worker.py

import asyncio
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, AsyncGenerator

from src.app.config import settings
from src.app.celery_app import celery_app
from src.app.db.session import AsyncSessionLocal
from src.app import crud, schemas
from src.app.services.vector_store import get_vector_store, VectorStore
from src.app.services.embedding_service import embed_texts

log = logging.getLogger(__name__)

# --- Configuration ---
VECTOR_SIZE = 384  # Davlan/afro-xlmr-mini INT8
HARD_CAP_LIMIT = 5000
HISTORY_LIMIT_DAYS = 730
MIN_WORD_COUNT = 4
DB_BATCH_SIZE = 1000
EMBED_BATCH_SIZE = 32


@celery_app.task(name="src.app.services.embedding_worker.generate_embeddings_task", bind=True, acks_late=True)
def generate_embeddings_task(self, chat_id: int):
    """Celery task to orchestrate vector ingestion into VectorStore (ChromaDB)."""
    try:
        log.info("Starting Vector Ingestion for Chat %s", chat_id)
        asyncio.run(process_chat_ingestion(chat_id))
        log.info("✅ Vector Ingestion finished for Chat %s", chat_id)
    except Exception as e:
        log.error("Embedding task failed for Chat %s: %s", chat_id, e, exc_info=True)


async def process_chat_ingestion(chat_id: int):
    vector_store = get_vector_store()

    try:
        # 1. Hard Cap Check
        existing_count = await vector_store.count(where={"chat_id": {"$eq": chat_id}})
        if existing_count >= HARD_CAP_LIMIT:
            log.warning("[Hard Cap] Chat %s already has %d vectors. Skipping.", chat_id, existing_count)
            return

        # Calculate remaining quota
        quota = HARD_CAP_LIMIT - existing_count
        cutoff_aware = datetime.now(timezone.utc) - timedelta(days=HISTORY_LIMIT_DAYS)
        cutoff_date = cutoff_aware.replace(tzinfo=None)

        async with AsyncSessionLocal() as db:
            # Update status to processing
            await crud.update_chat_embedding_status(
                db, chat_id, schemas.EmbeddingStatusEnum.processing.value, should_commit=True
            )

            total_ingested = 0

            # Stream data in batches and upload
            async for batch in data_stream_generator(db, chat_id, cutoff_date, quota):
                processed_count = await process_and_upload_batch(vector_store, batch, chat_id)
                total_ingested += processed_count

                quota -= processed_count
                if quota <= 0:
                    break

            # Final Success Status
            await crud.update_chat_embedding_status(
                db, chat_id, schemas.EmbeddingStatusEnum.completed.value, should_commit=True
            )
            log.info("✅ Ingestion Complete. Total %d vectors for Chat %s", total_ingested, chat_id)

    except Exception as e:
        log.error("Error during chat vector ingestion: %s", e, exc_info=True)
        async with AsyncSessionLocal() as db:
            await crud.update_chat_embedding_status(
                db, chat_id, schemas.EmbeddingStatusEnum.failed.value, should_commit=True
            )
        raise e


async def data_stream_generator(
    db, chat_id: int, cutoff_date: datetime, max_items: int
) -> AsyncGenerator[List[Dict[str, Any]], None]:
    """Yields batches of data (Messages + Segments) to be processed."""
    current_batch: List[Dict[str, Any]] = []
    total_processed_count = 0

    # PHASE 1: Process Messages
    msg_offset = 0
    while total_processed_count < max_items:
        raw_msgs = await crud.get_messages_batch(
            db, chat_id=chat_id, limit=DB_BATCH_SIZE, offset=msg_offset, min_date=cutoff_date
        )
        if not raw_msgs:
            break

        for msg in raw_msgs:
            if total_processed_count >= max_items:
                break

            if not msg.content or len(msg.content.split()) < MIN_WORD_COUNT:
                continue

            ts = msg.timestamp if msg.timestamp.tzinfo else msg.timestamp.replace(tzinfo=timezone.utc)
            participant_id = msg.participant_id
            participant = msg.participant.name if msg.participant else None

            current_batch.append({
                "type": "message",
                "id": msg.id,
                "text": msg.content,
                "timestamp": ts,
                "source_table": "messages",
                "participant_id": participant_id,
                "sender_name": participant,
            })
            total_processed_count += 1

            if len(current_batch) >= EMBED_BATCH_SIZE:
                yield current_batch
                current_batch = []

        msg_offset += DB_BATCH_SIZE

    # PHASE 2: Process Segments
    if total_processed_count < max_items:
        seg_offset = 0
        while total_processed_count < max_items:
            raw_segs = await crud.get_segments_batch(
                db, chat_id=chat_id, limit=DB_BATCH_SIZE, offset=seg_offset, min_date=cutoff_date
            )
            if not raw_segs:
                break

            for seg in raw_segs:
                if total_processed_count >= max_items:
                    break

                if not seg.combined_text:
                    continue

                seg_ts = seg.time_segment.start_time if seg.time_segment else datetime.now(timezone.utc)
                if seg_ts.tzinfo is None:
                    seg_ts = seg_ts.replace(tzinfo=timezone.utc)

                participant_id = seg.sender_id
                participant = seg.participant.name if seg.participant else None

                current_batch.append({
                    "type": "segment",
                    "id": seg.id,
                    "text": seg.combined_text,
                    "timestamp": seg_ts,
                    "source_table": "segments_sender",
                    "participant_id": participant_id,
                    "sender_name": participant,
                })
                total_processed_count += 1

                if len(current_batch) >= EMBED_BATCH_SIZE:
                    yield current_batch
                    current_batch = []

            seg_offset += DB_BATCH_SIZE

    # PHASE 3: Flush Remaining
    if current_batch:
        yield current_batch


async def process_and_upload_batch(vector_store: VectorStore, batch: List[Dict[str, Any]], chat_id: int) -> int:
    """Embeds texts and upserts them to VectorStore."""
    if not batch:
        return 0

    texts = [item["text"] for item in batch]
    try:
        embeddings = await embed_texts(texts)
    except Exception as e:
        log.error("Embedding generation failed: %s", e)
        return 0

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    valid_embeddings: List[List[float]] = []

    for i, item in enumerate(batch):
        embedding = embeddings[i]
        if not embedding:
            continue

        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{item['source_table']}_{item['id']}"))
        metadata = {
            "source_table": str(item["source_table"]),
            "source_id": int(item["id"]),
            "chat_id": int(chat_id),
            "text": str(item["text"]),
            "timestamp": item["timestamp"].isoformat(),
            "participant_id": int(item.get("participant_id") or 0),
            "sender_name": str(item.get("sender_name") or "unknown"),
        }

        ids.append(point_id)
        documents.append(item["text"])
        metadatas.append(metadata)
        valid_embeddings.append(embedding)

    if ids:
        await vector_store.upsert(
            ids=ids,
            embeddings=valid_embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    return len(ids)