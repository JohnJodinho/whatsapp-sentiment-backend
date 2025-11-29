from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
import jwt
from datetime import datetime, timedelta, timezone
import uuid
import logging

from src.app.config import settings
from src.app.db.session import get_db
from src.app import models

# Configuration
ALGORITHM = settings.JWT_ALGORITHM
ACCESS_TOKEN_EXPIRE_DAYS = settings.JWT_ACCESS_TOKEN_EXPIRE_DAYS

# The URL where the frontend gets the token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/guest")

logger = logging.getLogger(__name__)

async def get_current_user(
    token: str = Depends(oauth2_scheme), 
    db: AsyncSession = Depends(get_db)
) -> models.User:
    """
    Validates the JWT and ensures the anonymous user exists in the DB.
    If expired, deletes the User and all their data immediately. Then raises 401.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Decode and Verify Signature
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str: str = payload.get("sub")
        
        if user_id_str is None:
            raise credentials_exception
            
        try:
            user_uuid = uuid.UUID(user_id_str)
        except ValueError:
            raise credentials_exception

        # Check user exists in DB
        result = await db.execute(select(models.User).where(models.User.id == user_uuid))
        user = result.scalars().first()
        
        if user is None:
            raise credentials_exception
            
        return user

    except jwt.ExpiredSignatureError:
        logger.warning("Token expired. Initiating immediate user cleanup.")
        try:
         
            unverified_payload = jwt.decode(token, options={"verify_signature": False})
            expired_user_id = unverified_payload.get("sub")
            
            if expired_user_id:
                user_uuid = uuid.UUID(expired_user_id)
    
                await db.execute(delete(models.User).where(models.User.id == user_uuid))
                await db.commit()
                logger.info(f"Cleanup successful: Deleted expired user {user_uuid}")
                
        except Exception as e:
            
            logger.error(f"Failed to clean up expired user: {e}")

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    except jwt.PyJWTError:
        # Malformed tokens, bad signatures, etc.
        raise credentials_exception

async def get_current_user_ws(token: str, db: AsyncSession) -> models.User | None:
    """
    Special helper for WebSockets/SSE where we cannot use Depends() easily.
    Passes the session manually.
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str = payload.get("sub")
        if not user_id_str:
            return None
        
        user_uuid = uuid.UUID(user_id_str)
        result = await db.execute(select(models.User).where(models.User.id == user_uuid))
        return result.scalars().first()
    except jwt.ExpiredSignatureError:
        try: 
            unverified = jwt.decode(token, options={"verify_signature": False})
            uid = unverified.get("sub")
            if uid:
                await db.execute(delete(models.User).where(models.User.id == uuid.UUID(uid)))
                await db.execute()
                logger.info(f"WS/SSE Cleanup: Dleted expired user {uid}")
        except Exception:
            pass
        return 
    except Exception:
        return None
        
    