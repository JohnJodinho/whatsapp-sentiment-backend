import pytest
import pytest_asyncio
import logging

# --- Project Imports
from src.app.db.session import AsyncSessionLocal
from src.app.services.retrieval_service import SQLAlchemyVectorRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- TEST CONFIGURATION ---

# 1. SET THE CHAT ID TO TEST
LIVE_CHAT_ID = 57

# 2. SET A REAL QUERY FOR THAT CHAT
#    (e.g., "What did we say about the project?")
USER_QUERY = "CHANGE THIS TO A REAL QUERY FOR CHAT 57" 

# ---

@pytest.mark.asyncio
async def test_live_retriever_on_existing_chat():
    """
    Tests the full E2E retrieval pipeline on an existing chat
    by making real API and DB calls.
    """
    USER_QUERY = input("Enter a query: ")
    if USER_QUERY == "CHANGE THIS TO A REAL QUERY FOR CHAT 57":
        log.warning("Please update USER_QUERY in the test file.")
        assert False, "Test query is not set."

    log.info(f"--- Running Live Retriever Test ---")
    log.info(f"Chat ID: {LIVE_CHAT_ID}")
    log.info(f"Query:   '{USER_QUERY}'")

    # 1. Instantiate the retriever
    retriever = SQLAlchemyVectorRetriever(
        db_session_factory=AsyncSessionLocal,
        top_k=5 # Let's get the top 5 results
    )

    # 2. Run the retrieval (Real API call + Real DB search)
    try:
        documents = await retriever._aget_relevant_documents(
            query=USER_QUERY,
            chat_id=LIVE_CHAT_ID,
            run_manager=None
        )
    except Exception as e:
        log.error(f"Retriever failed with an exception: {e}", exc_info=True)
        assert False, f"Retriever raised an exception: {e}"

    # 3. Assert and Print Results
    assert documents is not None, "Retriever returned None"
    assert len(documents) > 0, "Retriever found 0 documents. Does chat 57 have embeddings?"

    log.info(f"\n--- SUCESS! Found {len(documents)} relevant documents ---")

    for i, doc in enumerate(documents):
        print("\n")
        log.info(f"--- Document {i+1} (Distance: {doc.metadata.get('distance'):.4f}) ---")
        log.info(f"Source: {doc.metadata.get('source_table')}:{doc.metadata.get('source_id')}")
        log.info(f"Content: {doc.page_content}")
        print("\n")


    assert True