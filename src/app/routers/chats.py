from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from  sqlalchemy import select
from src.app.db.session import get_db
from src.app import crud, schemas, models
from src.app.security import get_current_user
from src.app.limiter import limiter

router = APIRouter()

@router.delete("/{chat_id}", response_model=dict, status_code=status.HTTP_200_OK)
@limiter.limit("5/minute")
async def delete_chat(
    request: Request, 
    chat_id: int, 
    db: AsyncSession = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    """
    Delete a chat and all its related data. 
   
    """
    try:
        # 1. Lock Row (For Update) to prevent race conditions
        # This works within the EXISTING transaction started by auth
        stmt = select(models.Chat).where(models.Chat.id == chat_id).with_for_update()
        result = await db.execute(stmt)
        chat = result.scalars().first()

        if not chat:
            # We must commit/rollback to release the implicit transaction lock if we error out
            await db.rollback() 
            raise HTTPException(status_code=404, detail="Chat not found")

        # 2. Verify Ownership
        if chat.owner_id != current_user.id:
            await db.rollback()
            raise HTTPException(status_code=404, detail="Chat not found") 

        # 3. Check Status
        if chat.sentiment_status == "processing":
            await db.rollback()
            raise HTTPException(
                status_code=409,
                detail="Sentiment analysis is still running. Please cancel the job first."
            )

        # 4. Delete
        # We pass should_commit=False so we can commit strictly after everything succeeds
        await crud.delete_chat(db, chat_id, should_commit=False)
        
        # 5. Final Commit
        await db.commit()

        return {"ok": True, "message": f"Chat {chat_id} deleted successfully"}

    except Exception:
        # Ensure we rollback on any unexpected error to release locks
        await db.rollback()
        raise