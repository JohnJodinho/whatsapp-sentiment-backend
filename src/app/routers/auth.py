from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta, timezone
import jwt

from src.app.config import settings
from src.app.db.session import get_db
from src.app import models
from src.app.security import ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter()

@router.post("/guest", response_model=dict)
async def create_guest_session(db: AsyncSession = Depends(get_db)):
    """
    Creates a new anonymous user and returns a long-lived JWT.
    The frontend should call this ONCE on first load and store the token.
    """
    # 1. Create a new Anonymous User
    new_user = models.User()
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    # 2. GeneratT
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # "sub" holds the User UUID
    to_encode = {"sub": str(new_user.id), "exp": expire}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)

    return {
        "access_token": encoded_jwt, 
        "token_type": "bearer",
        "user_id": str(new_user.id)
    }