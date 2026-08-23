import asyncio
import sys
import os
import logging

# --- Add project root to path ---
# This allows this script to find and import from 'src'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
# --- End path setup ---

from src.app.services.summary_service import run_summary_job_with_retries
from src.app.db.session import engine  # Import engine for proper shutdown

# --- Configure logging ---
# This ensures you see all the log output from the service
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)
# --- End logging ---

async def run_test_job():
    """
    Runs the summary job for a specific chat ID.
    """
    # =================== WARNING ===================
    # This is an INTEGRATION TEST. It will:
    # 1. Connect to your REAL database.
    # 2. Make REAL API calls to Azure OpenAI (which cost money).
    # 3. REALLY write data to your database.
    #
    # Only run this if you intend to test the full, live integration.
    # ===============================================
    
    chat_id_to_test = 77 # The ID you wanted to test
    
    log.info(f"--- Starting summary job integration test for chat_id: {chat_id_to_test} ---")
    
    try:
        # This is the function you wanted to test
        await run_summary_job_with_retries(chat_id_to_test)
        
        log.info(f"--- Finished summary job integration test for chat_id: {chat_id_to_test} ---")
    except Exception as e:
        log.error(f"Test job failed: {e}", exc_info=True)
    finally:
        # Always dispose of the engine to close connections
        log.info("Cleaning up database connections...")
        await engine.dispose()
        log.info("Cleanup complete.")

if __name__ == "__main__":
    # This makes the script runnable with:
    # python tests/test_time_seg_summary.py
    asyncio.run(run_test_job())