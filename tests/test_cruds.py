from src.app import crud
from src.app.db.session import AsyncSessionLocal
import asyncio
async def test_participant_relationships():
    async with AsyncSessionLocal() as db:
        segs = await crud.get_segments_batch(db, 163, limit=10, offset=0)
    
        for seg in segs:
            participant_id = seg.sender_id
            participant = seg.participant.name

            print(participant_id, participant)

        msgs = await crud.get_messages_batch(db, 163, limit=10, offset=0 )

        for msg in msgs:
            participant_id = msg.participant_id
            participant = msg.participant.name

            print(participant_id, participant)



asyncio.run(test_participant_relationships())