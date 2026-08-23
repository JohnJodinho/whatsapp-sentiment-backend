# import asyncio
# from qdrant_client import AsyncQdrantClient, models as qmodels
# from src.app.config import settings  # Ensure this import works, or hardcode your URL/KEY

# # Configuration
# QDRANT_COLLECTION = "chat_vectors"

# async def apply_indices():
#     client = AsyncQdrantClient(
#         url=str(settings.QDRANT_URL),
#         api_key=settings.QDRANT_API_KEY,
#     )

#     print(f"Connecting to Qdrant at {settings.QDRANT_URL}...")
    
#     try:
#         # 1. Create Index for Sender Name (KEYWORD is required for exact filtering)
#         print("Creating index for 'sender_name'...")
#         await client.create_payload_index(
#             collection_name=QDRANT_COLLECTION,
#             field_name="sender_name",
#             field_schema=qmodels.PayloadSchemaType.TEXT
#         )
#         print("✅ 'sender_name' index created.")

#         # 2. Create Index for Timestamp (DATETIME is required for range filtering)
#         print("Creating index for 'timestamp'...")
#         await client.create_payload_index(
#             collection_name=QDRANT_COLLECTION,
#             field_name="timestamp",
#             field_schema=qmodels.PayloadSchemaType.DATETIME
#         )
#         print("✅ 'timestamp' index created.")

#     except Exception as e:
#         print(f"❌ Error creating indices: {e}")
#     finally:
#         await client.close()

# if __name__ == "__main__":
#     asyncio.run(apply_indices())



import asyncio
from qdrant_client import AsyncQdrantClient, models as qmodels
from src.app.config import settings

# --- CONFIGURATION ---
TARGET_CHAT_ID = 141  # <--- Change this to the chat ID you want to check

async def check_chat_payload(chat_id: int):
    client = AsyncQdrantClient(
        url=str(settings.QDRANT_URL),
        api_key=settings.QDRANT_API_KEY,
    )
    
    print(f"🔎 Searching for one vector in Chat {chat_id}...")

    # Create a filter to only look at this specific chat_id
    chat_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="chat_id",
                match=qmodels.MatchValue(value=chat_id)
            )
        ]
    )
    
    # Fetch just 1 point that matches the filter
    results, _ = await client.scroll(
        collection_name="chat_vectors",
        scroll_filter=chat_filter,
        limit=1,
        with_payload=True
    )
    
    if results:
        point = results[0]
        payload = point.payload
        print(f"✅ Found vector ID: {point.id}")
        print(f"Has 'sender_name'? {'✅ YES' if 'sender_name' in payload else '❌ NO'}")
        print(f"Has 'timestamp'?   {'✅ YES' if 'timestamp' in payload else '❌ NO'}")
        print("-" * 20)
        print("Full Payload:", payload)
    else:
        print(f"⚠️ No vectors found for Chat ID {chat_id}.")

    await client.close()

if __name__ == "__main__":
    asyncio.run(check_chat_payload(TARGET_CHAT_ID))
