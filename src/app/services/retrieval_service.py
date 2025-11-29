# src/app/services/retrieval_service.py

import asyncio
import logging
from typing import List, Dict, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text
from sqlalchemy.orm import selectinload
from sqlalchemy import insert, func, delete, select

# --- LangChain Imports ---
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

# --- Project Imports ---
from src.app.db.session import AsyncSessionLocal
from src.app.config import settings
from src.app import models
from src.app.services.embedding_service import client, EMBEDDING_DIM

log = logging.getLogger(__name__)


async def embed_query(text: str) -> Optional[List[float]]:
    """
    Embeds a single query string using the Azure OpenAI client.
    """
    try:
        resp = await client.embeddings.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT,
            input=[text]
        )
        return resp.data[0].embedding
    except Exception as e:
        log.error(f"Failed to embed query: {e}")
        return None



async def _fetch_message_context(db_session_factory: Any, ids: List[int]) -> Dict[int, str]:
    """
    Bulk fetches context for 'messages' sources.
    """
    async with db_session_factory() as db: 
        stmt = (
            select(models.Message)
            .options(selectinload(models.Message.participant)) 
            .where(models.Message.id.in_(ids))
        )
        result = await db.execute(stmt)
        messages = result.scalars().all()
    
    # Format the page_content string
    context_map = {}
    for m in messages:
        sender = m.participant.name if m.participant else "Unknown"
        time_str = m.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        context_map[m.id] = f"Message from {sender} at {time_str}:\n{m.content}"
    return context_map



async def _fetch_sender_segment_context(db_session_factory: Any, ids: List[int]) -> Dict[int, str]:
    """
    Bulk fetches context for 'segments_sender' sources.
    """
    async with db_session_factory() as db:
        stmt = (
            select(models.SenderSegment)
            .options(
                selectinload(models.SenderSegment.participant), # Eager load participant
                selectinload(models.SenderSegment.time_segment) # Eager load time segment
            )
            .where(models.SenderSegment.id.in_(ids))
        )
        result = await db.execute(stmt)
        segments = result.scalars().all()
    
    context_map = {}
    for s in segments:
        sender = s.participant.name if s.participant else "Unknown"
        start_str = s.time_segment.start_time.strftime('%Y-%m-%d %H:%M')
        context_map[s.id] = f"Summary of messages from {sender} around {start_str}:\n{s.combined_text}"
    return context_map


# --- 3. The Retriever Class (from PDF Guide ) ---

class SQLAlchemyVectorRetriever(BaseRetriever):
    """
    A LangChain custom retriever that uses raw SQLAlchemy
    to perform vector search and enriches the results.
    """
    db_session_factory: Any # e.g., AsyncSessionLocal
    top_k: int = 8
    
    class Config:
        arbitrary_types_allowed = True

    # --- ADD THIS 4-LINE METHOD ---
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs
    ) -> List[Document]:
        """Synchronous not supported. Use async version."""
        raise NotImplementedError("This retriever only supports async operation")
    # --

    async def _aget_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun,
        chat_id: int # Our custom, required parameter
    ) -> List[Document]:
        """
        Asynchronously get the relevant documents.
        """
        
        # 1. Embed the user's query
        q_emb = await embed_query(query)
        if not q_emb:
            log.error("Cannot get relevant documents, query embedding failed.")
            return []
            
        # Format for pgvector string
        q_emb_str = "[" + ",".join(map(str, q_emb)) + "]"

        # 2. Run the Vector Search (Phase 3 SQL [cite: 135-140])
        # We use '<=>' (L2 distance), which the 'vector_l2_ops' index supports
        sql_query = text("""
            SELECT 
                id, 
                source_table, 
                source_id,
                text_excerpt,
                chunk_metadata,
                embedding <=> CAST(:query_embedding AS vector) AS distance
            FROM public.embeddings
            WHERE chat_id = :chat_id
            ORDER BY distance
            LIMIT :k;
        """)
        
        retrieved_rows = []
        async with self.db_session_factory() as session:
            try:
                result = await session.execute(sql_query, {
                    "query_embedding": q_emb_str,
                    "chat_id": chat_id,
                    "k": self.top_k
                })
                retrieved_rows = result.fetchall()
            except Exception as e:
                log.error(f"Vector search failed for chat {chat_id}: {e}")
                return [] # Return empty list on failure

        # 3. Perform "Second-Hop" Bulk Lookups
        # Group IDs by their source table
        source_ids_map = {
            "messages": [],
        
            "segments_sender": []
        }
        for row in retrieved_rows:
            if row.source_table in source_ids_map:
                source_ids_map[row.source_table].append(row.source_id)

        # Run the bulk fetches in parallel
        context_maps = {}
        
        # REMOVED: async with self.db_session_factory() as db:
        
        tasks = []
        if source_ids_map["messages"]:
            # Pass the factory, not the session
            tasks.append(_fetch_message_context(self.db_session_factory, source_ids_map["messages"]))
        
        if source_ids_map["segments_sender"]:
            tasks.append(_fetch_sender_segment_context(self.db_session_factory, source_ids_map["segments_sender"]))
        
        # Wait for all context fetches to complete
        if tasks: # Only run if there are tasks
            results = await asyncio.gather(*tasks)
        else:
            results = []
            
            # Combine results into a single map
            for res_map in results:
                context_maps.update(res_map)

        # 4. Assemble the final LangChain Documents
        documents: List[Document] = []
        for row in retrieved_rows:
            # Get the enriched page content from our second-hop lookup
            page_content = context_maps.get(row.source_id)
            
            # Fallback in case lookup failed (should not happen)
            if not page_content:
                page_content = row.text_excerpt 
            
            doc = Document(
                page_content=page_content,
                metadata={
                    "source_table": row.source_table,
                    "source_id": row.source_id,
                    "distance": row.distance
                }
            )
            documents.append(doc)
            
        return documents