import asyncio
from src.app.services.vector_store import get_vector_store

TARGET_CHAT_ID = 141


async def check_chat_payload(chat_id: int):
    store = get_vector_store()
    print(f"🔎 Checking Chroma vectors for Chat {chat_id}...")
    count = await store.count(where={"chat_id": {"$eq": chat_id}})
    print(f"Total vectors for Chat {chat_id}: {count}")


if __name__ == "__main__":
    asyncio.run(check_chat_payload(TARGET_CHAT_ID))
