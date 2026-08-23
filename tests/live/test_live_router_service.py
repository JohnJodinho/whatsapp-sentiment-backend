import pytest
import pytest_asyncio
import logging

from src.app.services.router_service import route_query

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_live_router_service():
    while True:
        user_query = input("Enter a query ('Q' to quit): ")
        if user_query.strip().upper() == "Q":
            log.info("Ending test..")
            break

        if user_query.strip().upper() == "":
            log.warning("No query entered!")
            log.info("Please enter a query...")
            continue
        
        log.info(f"--- Running Live Query Routing Test ---")
        log.info(f"Query: '{user_query}'")

        try:
            response = await route_query(user_query)
            log.info(f"Query router resolved to: {response}")
        except Exception as e:
            log.error(f"Query router failed with an exception: {e}", exc_info=True)
            assert False, f"Router rased an exception: {e}"

        assert response in {"CHAT", "ANALYTICS", "HYBRID"}, f"invalid reponse: {response}"

        log.info(f"Successfuly run router query")

    assert True




    