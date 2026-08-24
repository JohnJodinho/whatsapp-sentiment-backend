# src/app/services/summary_service.py

import asyncio
import os
import logging
import re
from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from src.app.config import settings
from src.app.services.llm_factory import get_context_llm

from sqlalchemy.ext.asyncio import AsyncSession
from asyncpg.exceptions import ConnectionDoesNotExistError
from sqlalchemy.exc import DBAPIError
from src.app.db.session import AsyncSessionLocal
from src.app import crud, models

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant. Summarize the following chat conversation
in a concise paragraph. Focus on the main topics and any conclusions or actions."""

MAX_CONCURRENT_REQUESTS = 10
MAX_INTERNAL_RETRIES = 3
INTERNAL_RETRY_DELAY = 2


async def summarize_text(
    text_to_summarize: str,
    semaphore: asyncio.Semaphore,
    segment_id: int,
) -> str:
    """Calls Groq Chat Completion to get a concise segment summary."""
    if not text_to_summarize.strip():
        log.warning("[Segment %s] Received empty text, skipping.", segment_id)
        return ""

    async with semaphore:
        llm = get_context_llm()
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=text_to_summarize),
        ]

        for attempt in range(MAX_INTERNAL_RETRIES):
            try:
                response = await llm.ainvoke(messages)
                content = response.content if hasattr(response, "content") else str(response)
                return content.strip() if content else ""
            except Exception as e:
                delay = INTERNAL_RETRY_DELAY * (2 ** attempt)
                if attempt == MAX_INTERNAL_RETRIES - 1:
                    log.error("[Segment %s] Summary generation failed: %s", segment_id, e)
                    break
                log.warning("[Segment %s] Summary attempt %d failed: %s. Retrying in %ds...", segment_id, attempt + 1, e, delay)
                await asyncio.sleep(delay)

        return ""


async def get_test_for_segments(db: AsyncSession, chat_id: int) -> List[models.TimeSegment]:
    log.info("Fetching segments to summarize for chat %s...", chat_id)
    segments = await crud.get_segments_for_summarization(db, chat_id=chat_id)
    return segments or []


async def queue_summary_job(chat_id: int):
    """The main worker function that orchestrates the summarization job."""
    log.info("[Summary Job %s] Starting...", chat_id)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with AsyncSessionLocal() as db:
        try:
            segments_to_process = await get_test_for_segments(db, chat_id=chat_id)
            if not segments_to_process:
                log.info("[Summary Job %s] No segments found needing summarization.", chat_id)
                return

            log.info("[Summary Job %s] Found %d segments to summarize.", chat_id, len(segments_to_process))

            tasks = []
            segments_to_update = []

            for segment in segments_to_process:
                full_segment_text = "\n".join(
                    [s.combined_text for s in segment.sender_segments if s.combined_text]
                )
                if full_segment_text.strip():
                    tasks.append(
                        summarize_text(
                            text_to_summarize=full_segment_text,
                            semaphore=semaphore,
                            segment_id=segment.id,
                        )
                    )
                    segments_to_update.append(segment)
                else:
                    log.warning("Segment %s has no text. Skipping.", segment.id)

            if not tasks:
                log.warning("[Summary Job %s] No valid text found to summarize.", chat_id)
                return

            summaries = await asyncio.gather(*tasks)

            for segment, summary in zip(segments_to_update, summaries):
                if summary:
                    segment.summary = summary
                    await db.merge(segment)

            await db.commit()
            log.info("✅ [Summary Job %s] Successfully completed.", chat_id)

        except Exception as e:
            log.error("[Summary Job %s] Job failed: %s", chat_id, e, exc_info=True)
            await db.rollback()
            raise


async def run_summary_job_with_retries(chat_id: int):
    MAX_RETRIES = 3
    BASE_DELAY_SECONDS = 5

    for attempt in range(MAX_RETRIES):
        try:
            await queue_summary_job(chat_id)
            return
        except (DBAPIError, ConnectionDoesNotExistError) as e:
            if attempt == MAX_RETRIES - 1:
                log.error("[Summary Supervisor %s] Job failed after %d attempts: %s", chat_id, MAX_RETRIES, e)
                break
            delay = BASE_DELAY_SECONDS * (2 ** attempt)
            await asyncio.sleep(delay)
        except Exception as e:
            log.error("[Summary Supervisor %s] Non-retriable error: %s", chat_id, e, exc_info=True)
            break