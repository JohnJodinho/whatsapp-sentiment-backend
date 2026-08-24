# src/app/services/llm_factory.py

import os
import random
import logging
import asyncio
from typing import List, Union, AsyncGenerator, Any

from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from src.app.config import settings

log = logging.getLogger(__name__)


def create_groq_chat_llm(
    model_name: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    streaming: bool = False,
) -> ChatGroq:
    """Create a native ChatGroq instance targeting Groq Cloud endpoints."""
    api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "dummy_key")
    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )


# --- Initialized LLMs ---
def get_router_llm():
    """Qwen 3.6 27B for fast routing, safe SQL generation & filter extraction."""
    return create_groq_chat_llm(
        model_name=settings.GROQ_MODEL_ROUTER,
        temperature=0.0,
        max_tokens=500,
    )


def get_context_llm():
    """GPT-OSS-20B for question contextualization and standalone rephrasing."""
    return create_groq_chat_llm(
        model_name=settings.GROQ_MODEL_FALLBACK,
        temperature=0.1,
        max_tokens=500,
    )


def get_main_llm_primary(streaming: bool = True):
    """GPT-OSS-120B for primary RAG answer synthesis and high-reasoning tasks."""
    return create_groq_chat_llm(
        model_name=settings.GROQ_MODEL_PRIMARY,
        temperature=0.2,
        max_tokens=2048,
        streaming=streaming,
    )


def get_main_llm_fallback(streaming: bool = True):
    """GPT-OSS-20B fallback when primary model is rate-limited or unavailable."""
    return create_groq_chat_llm(
        model_name=settings.GROQ_MODEL_FALLBACK,
        temperature=0.2,
        max_tokens=2048,
        streaming=streaming,
    )


async def execute_resilient_llm(
    messages: List[BaseMessage],
    max_retries: int = 3,
    stream: bool = False,
) -> Union[AsyncGenerator[str, None], Any]:
    """
    Executes Groq LLM invocation with exponential backoff and automatic
    model fallback from Primary (openai/gpt-oss-120b) to Fallback (openai/gpt-oss-20b).
    """
    primary_llm = get_main_llm_primary(streaming=stream)
    fallback_llm = get_main_llm_fallback(streaming=stream)

    for attempt in range(max_retries):
        try:
            if stream:
                return primary_llm.astream(messages)
            return await primary_llm.ainvoke(messages)
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate limit" in str(e).lower()
            if is_rate_limit or attempt == max_retries - 1:
                log.warning(
                    "[Groq Resilient LLM] Primary model %s hit error on attempt %d: %s. Falling back to %s",
                    settings.GROQ_MODEL_PRIMARY,
                    attempt + 1,
                    e,
                    settings.GROQ_MODEL_FALLBACK,
                )
                try:
                    if stream:
                        return fallback_llm.astream(messages)
                    return await fallback_llm.ainvoke(messages)
                except Exception as fb_err:
                    log.error("[Groq Resilient LLM] Fallback model also failed: %s", fb_err)
                    raise fb_err

            delay = (2 ** attempt) + random.uniform(0.5, 1.5)
            log.warning("[Groq Resilient LLM] Retrying in %.2fs due to error: %s", delay, e)
            await asyncio.sleep(delay)

    raise RuntimeError("All Groq LLM retry attempts failed.")
