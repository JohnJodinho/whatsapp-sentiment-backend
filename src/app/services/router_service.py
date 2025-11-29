import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

from src.app.config import settings
from src.app.schemas import QueryRoute

log = logging.getLogger(__name__)


router_llm = AzureChatOpenAI(
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_ROUTER    ,
    api_version=settings.AZURE_OPENAI_API_VERSION_ROUTER,
    azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT_ROUTER),
    api_key=settings.AZURE_OPENAI_API_KEY_ROUTER,
    
    temperature=0.0,
    max_tokens=20
)


router_prompt_template = """
You are an expert query classifier. Your job is to read a user's
question and return exactly one of three labels: CHAT, ANALYTICS, or BOTH.

- CHAT: Use this for questions about the *content* of messages.
  (e.g., "What was said about X?", "Who mentioned Y?", "Find the message about Z")

- ANALYTICS: Use this for questions about *structured data, metrics, or counts*.
  (e.g., "Who sent the most messages?", "How many media files?",
  "What was the overall sentiment?", "What time is most active?")

- BOTH: Use this for questions that *correlate content with analytics*.
  (e.g., "Which messages caused the sentiment dip?",
  "Why was John so negative at 3 PM?")

--- EXAMPLES ---
Q: "Who said 'foobar' in the chat?"
A: CHAT

Q: "Show sentiment distribution"
A: ANALYTICS

Q: "Which messages caused the sentiment dip at 10:00?"
A: BOTH

Q: "who is the most active participant?"
A: ANALYTICS

Q: "What did we say about the restaurant?"
A: CHAT

Q: "How many voice notes did Jane send?"
A: ANALYTICS
---

User question:
{user_question}

Return one word only: CHAT, ANALYTICS, or BOTH.
"""


router_chain = (
    ChatPromptTemplate.from_template(router_prompt_template)
    | router_llm
    | StrOutputParser()
)


async def route_query(query: str) -> QueryRoute:
    """
    Classifies a user query into CHAT, ANALYTICS or BOTH
    """

    log.info(f"Routting query: '{query}'")

    try:
        result = await router_chain.ainvoke({"user_question": query})

        cleaned_res = result.strip().replace(".", "").upper()

        if cleaned_res == "CHAT":
            log.info("Route: CHAT (Vector Search)")
            return "CHAT"
        elif cleaned_res == "ANALYTICS":
            log.info("Route: ANALYTICS (Structured Query)")
            return "ANALYTICS"
        elif cleaned_res == "BOTH":
            log.info("Route: BOTH (Hybrid)")
            return "BOTH"
        else: 
            log.warning(f"Router returned unkown result '{result}'. Defaulting to CHAT.")
            return "CHAT"
    except Exception as e:
        log.error(f"Error in router_chain: {e}. Defaulting to CHAT.")
        return "CHAT"