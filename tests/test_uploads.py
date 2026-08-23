
import pytest
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.app import models
from pathlib import Path

# @pytest.mark.asyncio
# async def test_upload_real_chat_file(client):
#     """
#     Tests uploading a real-world WhatsApp .txt file from the filesystem.
#     """
#     file_path = Path(__file__).parent / "data" / "WhatsApp Chat with Mira Neigh.txt"

#     with open(file_path, "rb") as f:
#         files = {
#             'file': ('WhatsApp Chat with Mira Neigh.txt', f, 'text/plain')
#         }
#         response = client.post('/uploads/whatsapp', files=files)

#     assert response.status_code == 200
#     data = response.json()

#     assert data["title"] == "Mira Neigh"
#     assert "id" in data
#     assert data["sentiment_status"] == "pending"

@pytest.mark.asyncio
async def test_upload_triggers_worker_and_completes(client, db_session: AsyncSession):
    """
    Tests the full end-to-end flow and verifies results in the database.
    """
    # ARRANGE: Upload a valid chat file
    sample_chat_content = "10/12/2025, 9:07 PM - Alice: I love this product!\n"
    files = {
        'file': ('WhatsApp Chat with E2E Test.txt', sample_chat_content, 'text/plain')
    }
    response = await client.post('/uploads/whatsapp', files=files)
    assert response.status_code == 200
    chat_id = response.json()["id"]

    # ACT & ASSERT: Poll the status endpoint until the job is done
    final_status = ""
    for _ in range(40): # Increased timeout for slower machines/models
        status_response = await client.get(f"/chats/{chat_id}/sentiment-status")
        assert status_response.status_code == 200
        
        status_data = status_response.json()
        final_status = status_data["status"]

        if final_status == "completed":
            break
        
        await asyncio.sleep(0.5)

    assert final_status == "completed"

    # FINAL VERIFICATION: Query the database directly to confirm worker success
    result = await db_session.execute(
        select(models.MessageSentiment)
        .join(models.Message)
        .where(models.Message.chat_id == chat_id)
    )
    sentiments = result.scalars().all()
    assert len(sentiments) == 1
    assert sentiments[0].overall_label is not None

    
# @pytest.mark.asyncio
# async def test_upload_valid_txt_file(client): # <-- 1. Add client as an argument
#     """
#     Tests a successful upload of a valid WhatsApp chat file.
#     """
#     # A simple, valid chat line
#     sample_chat_content = "10/12/2025, 9:07 PM - Alice: This is a test message.\n"
    
#     # The 'files' parameter simulates a multipart/form-data upload
#     files = {
#         'file': ('WhatsApp Chat with Test Chat.txt', sample_chat_content, 'text/plain')
#     }
    
#     response = client.post('/uploads/whatsapp', files=files)
    
#     # 1. Check for a successful HTTP status code
#     assert response.status_code == 200
    
#     # 2. Parse the JSON response
#     data = response.json()
    
#     # 3. Assert that the response contains the expected fields from your ChatRead schema
#     assert "id" in data
#     assert "title" in data
#     assert "sentiment_status" in data
    
#     # 4. Assert the values are correct
#     assert data["title"] == "Test Chat"
#     assert data["sentiment_status"] == "pending"

# @pytest.mark.asyncio
# async def test_upload_invalid_file_type(client): # <-- 2. Also add client here
#     """
#     Tests uploading a non-txt file, which should be rejected.
#     """
#     files = {'file': ('image.jpg', b'someimagedata', 'image/jpeg')}
    
#     response = client.post('/uploads/whatsapp', files=files)
    
#     assert response.status_code == 400 # Bad Request
#     assert "Invalid file" in response.json()["detail"]