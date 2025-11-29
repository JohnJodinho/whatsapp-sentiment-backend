import asyncio
import os
import logging
import json
from typing import List, Dict, Any

from openai import AsyncAzureOpenAI
from src.app.config import settings 
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text
from asyncpg.exceptions import ConnectionDoesNotExistError
from sqlalchemy.exc import DBAPIError

from src.app.db.session import AsyncSessionLocal
from src.app import crud, models


log = logging.getLogger(__name__)

client = AsyncAzureOpenAI(
    api_version=settings.AZURE_OPENAI_API_VERSION,
    azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT),
    api_key=settings.AZURE_OPENAI_API_KEY
)

EMBEDDING_DIM = 1536 
OPENAI_BATCH_SIZE = 64


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Calls the Azure OpenAI Embedding API in batches.
    """
    results: List[List[float]] = []
    
    for i in range(0, len(texts), OPENAI_BATCH_SIZE):
        batch = texts[i:i + OPENAI_BATCH_SIZE]
        try:
            log.info(f"Sending batch {i//OPENAI_BATCH_SIZE + 1} to Azure OpenAI ({len(batch)} items)...")
            print(f"Sending batch {i//OPENAI_BATCH_SIZE + 1} to Azure OpenAI ({len(batch)} items)...")
            resp = await client.embeddings.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT, 
                input=batch
            )
            
            batch_embs = [r.embedding for r in resp.data]
            results.extend(batch_embs)
            
        except Exception as e:
            log.error(f"Azure OpenAI API call failed: {e}")
            results.extend([[]] * len(batch)) 
            
    return results


async def persist_embeddings(
    session: AsyncSession, 
    chat_id: int, 
    items: List[Dict[str, Any]], 
    embeddings: List[List[float]]
):
    """
    Persists embeddings from multiple sources.
    """
    log.info(f"Persisting {len(embeddings)} multi-source embeddings for chat {chat_id}...")
    
    # Use CAST(:embedding AS vector) for asyncpg compatibility
    stmt = text("""
        INSERT INTO public.embeddings 
            (chat_id, source_table, source_id, embedding, text_excerpt, chunk_metadata)
        VALUES 
            (:chat_id, :source_table, :source_id, CAST(:embedding AS vector), :text_excerpt, :chunk_metadata)
    """)
    
    insert_params: List[Dict[str, Any]] = []
    
    for item, emb in zip(items, embeddings):
        if not emb or len(emb) != EMBEDDING_DIM: 
            log.warning(f"Skipping chunk for chat {chat_id} (table: {item['source_table']}, id: {item['source_id']}) due to failed/malformed embedding.")
            continue
        
        params = {
            "chat_id": chat_id,
            "source_table": item['source_table'],
            "source_id": item['source_id'],
  
            "embedding": "[" + ",".join(map(str, emb)) + "]",
            "text_excerpt": item['text'][:1000],
            "chunk_metadata": json.dumps({}) 
        }
        insert_params.append(params)

    if not insert_params:
        log.warning(f"No valid embeddings to persist for chat {chat_id}.")
        return

    try:
        await session.execute(stmt, insert_params)
        
        await session.commit() 
        log.info(f"Successfully persisted {len(insert_params)} embeddings for chat {chat_id}.")
        
    except Exception as e:
        log.error(f"Failed to persist embeddings for chat {chat_id}: {e}", exc_info=True)
        await session.rollback()
        raise

async def queue_embedding_job(chat_id: int):
    """
    Orchestrates the multi-source embedding job.
    """
    log.info(f"[Embedding Job {chat_id}] Starting multi-source embedding...")
    items_to_embed: List[Dict[str, Any]] = []
    
    async with AsyncSessionLocal() as db:
        try:
            
            messages = await crud.get_messages_by_chat(db, chat_id=chat_id)
            for m in messages:
                if m.content:
                    items_to_embed.append({
                        "text": m.content,
                        "source_table": "messages",
                        "source_id": m.id
                    })
                    
            sender_segments = await crud.get_all_sender_segments_for_chat(db, chat_id=chat_id)
            for ss in sender_segments:
                if ss.combined_text:
                    items_to_embed.append({
                        "text": ss.combined_text,
                        "source_table": "segments_sender",
                        "source_id": ss.id
                    })
            
            if not items_to_embed:
                log.warning(f"[Embedding Job {chat_id}] No text content found to embed. Exiting.")
                return

            log.info(f"[Embedding Job {chat_id}] Found {len(items_to_embed)} items to embed across all sources.")
            
           
            texts = [item['text'] for item in items_to_embed]
            embeddings = await embed_texts(texts)

            # C) Persist embeddings
            await persist_embeddings(db, chat_id, items_to_embed, embeddings)
            
            log.info(f"✅ [Embedding Job {chat_id}] Successfully completed multi-source embedding.")
        
        except Exception as e:
            log.error(f"[Embedding Job {chat_id}] Job failed: {e}", exc_info=True)
            await db.rollback()
            raise # Re-raise for the supervisor

# --- 4. Supervisor (Unchanged) ---
async def run_embedding_job_with_retries(chat_id: int):
    """
    A supervisor function that runs the embedding worker with a retry policy.
    """
    MAX_RETRIES = 3
    BASE_DELAY_SECONDS = 10
    
    log.info(f"[Embedding Supervisor {chat_id}] Job queued.")

    for attempt in range(MAX_RETRIES):
        try:
            await queue_embedding_job(chat_id)
            log.info(f"[Embedding Supervisor {chat_id}] Job completed successfully.")
            return # Success!

        except (DBAPIError, ConnectionDoesNotExistError) as e:
            log.warning(
                f"[Embedding Supervisor {chat_id}] Network error on attempt {attempt + 1}/{MAX_RETRIES}: {e}"
            )
            if attempt == MAX_RETRIES - 1:
                log.error(f"[Embedding Supervisor {chat_id}] Job failed after {MAX_RETRIES} attempts.")
                break 
            
            delay = BASE_DELAY_SECONDS * (2 ** attempt)
            log.info(f"[Embedding Supervisor {chat_id}] Retrying in {delay} seconds...")
            await asyncio.sleep(delay)

        except Exception as e:
            log.error(
                f"[Embedding Supervisor {chat_id}] A non-retriable error occurred: {e}", 
                exc_info=True
            )
            break 

    log.error(f"[Embedding Supervisor {chat_id}] Job FAILED.")