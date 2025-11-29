import asyncio
import os
import logging
import json
from typing import List, Dict, Any

from openai import (
    AsyncAzureOpenAI,
    RateLimitError,
    APITimeoutError,
    APIError,
    BadRequestError
)
from src.app.config import settings 


from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text
from asyncpg.exceptions import ConnectionDoesNotExistError
from sqlalchemy.exc import DBAPIError
import re
from src.app.db.session import AsyncSessionLocal
from src.app import crud, models 


log = logging.getLogger(__name__)


client = AsyncAzureOpenAI(
    api_version=settings.AZURE_OPENAI_API_VERSION_SUMMARY,
    azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT_SUMMARY), 
    api_key=settings.AZURE_OPENAI_API_KEY_SUMMARY,
    max_retries=0
)


SYSTEM_PROMPT = """You are a helpful assistant. Summarize the following chat conversation
in a concise paragraph. Focus on the main topics and any conclusions or actions."""

MAX_CONCURRENT_REQUESTS = 20  # "Bouncer": Only 20 requests at a time
MAX_INTERNAL_RETRIES = 3    # "Patience": Try 3 times on flaky errors
INTERNAL_RETRY_DELAY = 5    # "Cooldown": Wait 5s between retries
RETRY_AFTER_REGEX = re.compile(r"retry after (\d+) seconds")


async def summarize_text(
        text_to_summarize: str, 
        semaphore: asyncio.Semaphore, 
        segment_id: int
) -> str:
    """
    Calls the Azure OpenAI Chat Completion API to get a summary.
    """
    if not text_to_summarize.strip():
        log.warning(f"[Segment {segment_id}] Received empty text, skipping.")
        return ""

    # This is the "bouncer"
    async with semaphore:
        for attempt in range(MAX_INTERNAL_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=settings.AZURE_OPENAI_DEPLOYMENT_SUMMARY,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text_to_summarize}
                    ],
                    max_completion_tokens=1024,
                    timeout=30.0 # Set a 30-second timeout
                )
                summary = resp.choices[0].message.content
                return summary.strip() if summary else ""

            except RateLimitError as e:
                # 429 Error: API is busy. Wait and retry.
                delay = INTERNAL_RETRY_DELAY * (2 ** attempt)
                re_match = RETRY_AFTER_REGEX.search(e.message)
                if re_match:
                    try:
                        delay = int(re_match.group(1)) + 1
                    except ValueError:
                        pass


                if attempt == MAX_INTERNAL_RETRIES - 1:
                    log.error(f"[Segment {segment_id}] Rate limit. all retires failed. Giving up mehn!")
                    break

                log.warning(f"[Segment {segment_id}] Rate limit hit (Attempt {attempt + 1}). Retrying in {delay}s... Error: {e}")
                await asyncio.sleep(delay)



            except BadRequestError as e:
            
                if "content_filter" in e.message:
                    log.error(f"[Segment {segment_id}] PERMANENT 400 ERROR (Content Filter). Giving up. Error: {e.message}")
                elif "max_tokens" in e.message:
                    log.error(f"[Segment {segment_id}] PERMANENT 400 ERROR (Max Tokens). Our limit (1024) is *still* too small. Giving up. Error: {e.message}")
                else:
                    log.error(f"[Segment {segment_id}] PERMANENT 400 ERROR (Unknown). Giving up. Error: {e.message}")
                return "" 
            
            except (APITimeoutError, APIError) as e:
                if attempt == MAX_INTERNAL_RETRIES - 1:
                    log.error(f"[Segment {segment_id}] Temporary API Error. All retries failed. I give up guy")
                    break

                delay = INTERNAL_RETRY_DELAY * (2 ** attempt)
                log.warning(f"[Segment {segment_id}] Temporary API Error (Attempt {attempt + 1}). Retrying in {delay}s... Error: {e}")
                await asyncio.sleep(delay)

            except Exception as e:
               
                log.error(f"[Segment {segment_id}] Unexpected error. Giving up. Error: {e}", exc_info=True)
                return ""


        log.error(f"[Segment {segment_id}] All {MAX_INTERNAL_RETRIES} retry attempts failed.")
        return ""
    
async def    get_test_for_segments(db: AsyncSession, chat_id: int) -> List[models.TimeSegment]:
    """
    Fetches all TimeSegments for a chat that don't have a summary.
    """
    log.info(f"Fetching segments to summarize for chat {chat_id}...")
    
    segments = await crud.get_segments_for_summarization(db, chat_id=chat_id)
    
    if not segments:
        log.info(f"No segments found for chat {chat_id} needing summarization.")
        return []
        
    return segments


async def queue_summary_job(chat_id: int):
    """
    The main worker function that orchestrates the summarization job.
    Runs all summarization API calls in parallel.
    """
    log.info(f"[Summary Job {chat_id}] Starting...")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async with AsyncSessionLocal() as db:
        try:
  
            segments_to_process = await get_test_for_segments(db, chat_id=chat_id)
            
            if not segments_to_process:
                log.info(f"[Summary Job {chat_id}] No segments found needing summarization. Exiting.")
                return

            log.info(f"[Summary Job {chat_id}] Found {len(segments_to_process)} segments to summarize.")

            tasks = []
            segments_to_update = [] 

            for segment in segments_to_process:

                full_segment_text = "\n".join(
                    [s.combined_text for s in segment.sender_segments if s.combined_text]
                )
                
                if full_segment_text.strip():
                    
                    tasks.append(summarize_text(
                        text_to_summarize=full_segment_text,
                        semaphore=semaphore,
                        segment_id=segment.id
                    ))
                    # Add the corresponding segment to map results back
                    segments_to_update.append(segment)
                else:
                    log.warning(f"Segment {segment.id} has no text. Skipping.")
            
            if not tasks:
                log.warning(f"[Summary Job {chat_id}] No valid text found to summarize. Exiting.")
                return

            # C) Run all summarization tasks in parallel
            log.info(f"[Summary Job {chat_id}] Sending {len(tasks)} summary requests (batched {MAX_CONCURRENT_REQUESTS} at a time)...")
            summaries = await asyncio.gather(*tasks)
            log.info(f"[Summary Job {chat_id}] Received {len(summaries)} summary results.")
    

            # D) Persist all results
            for segment, summary in zip(segments_to_update, summaries):
                if summary:
                    segment.summary = summary # Update the model instance
                    await db.merge(segment)
                    log.info(f"[Segment {segment.id}] Successfully summarized and saved.")
                else:
                     log.warning(f"[Segment {segment.id}] Failed to generate summary (check logs for permanent 400 error).")
            await db.commit()
            log.info(f"✅ [Summary Job {chat_id}] Successfully completed.")
            
        except Exception as e:
            log.error(f"[Summary Job {chat_id}] Job failed: {e}", exc_info=True)
            await db.rollback()
            raise # Re-raise for the supervisor

# --- 3. Supervisor (Copied from embedding_service) ---

async def run_summary_job_with_retries(chat_id: int):
    """
    A supervisor function that runs the summary worker with a retry policy.
    This is the function that should be passed to BackgroundTasks.
    """
    MAX_RETRIES = 3
    BASE_DELAY_SECONDS = 10
    
    log.info(f"[Summary Supervisor {chat_id}] Job queued.")

    for attempt in range(MAX_RETRIES):
        try:
            # Call the main job
            await queue_summary_job(chat_id)
            
            log.info(f"[Summary Supervisor {chat_id}] Job completed successfully.")
            return # Success!

        except (DBAPIError, ConnectionDoesNotExistError) as e:
            log.warning(
                f"[Summary Supervisor {chat_id}] Network error on attempt {attempt + 1}/{MAX_RETRIES}: {e}"
            )
            if attempt == MAX_RETRIES - 1:
                log.error(f"[Summary Supervisor {chat_id}] Job failed after {MAX_RETRIES} attempts.")
                break 
            
            delay = BASE_DELAY_SECONDS * (2 ** attempt)
            log.info(f"[Summary Supervisor {chat_id}] Retrying in {delay} seconds...")
            await asyncio.sleep(delay)

        except Exception as e:
            log.error(
                f"[Summary Supervisor {chat_id}] A non-retriable error occurred: {e}", 
                exc_info=True
            )
            break # Do not retry on unknown errors

    log.error(f"[Summary Supervisor {chat_id}] Job FAILED.")