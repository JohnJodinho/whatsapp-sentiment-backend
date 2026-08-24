import os
import pytest
from src.app.services.llm_factory import get_router_llm, get_main_llm_primary
from langchain_core.messages import HumanMessage


@pytest.mark.asyncio
async def test_groq_endpoints():
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set in environment, skipping live API call.")

    router = get_router_llm()
    resp = await router.ainvoke([HumanMessage(content="Say hello in one word.")])
    assert resp.content is not None