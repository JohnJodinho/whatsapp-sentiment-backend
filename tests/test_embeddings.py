import asyncio
import sys
import os
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


from src.app.services.embedding_service import run_embedding_job_with_retries
from src.app.db.session import engine  


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


async def run_test_job():
    """
    Runs the embedding job for a specific chat ID.
    """
    
    chat_id_to_test = 57 
    
    log.info(f"--- Starting embeddings job integration test for chat_id: {chat_id_to_test} ---")
    
    try:
  
        await run_embedding_job_with_retries(chat_id_to_test)
        
        log.info(f"--- Finished embeddings job integration test for chat_id: {chat_id_to_test} ---")
    except Exception as e:
        log.error(f"Test job failed: {e}", exc_info=True)
    finally:
        
        log.info("Cleaning up database connections...")
        await engine.dispose()
        log.info("Cleanup complete.")

if __name__ == "__main__":

    asyncio.run(run_test_job())