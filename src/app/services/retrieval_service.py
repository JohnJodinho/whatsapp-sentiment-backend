# src/app/services/retrieval_service.py

import logging
from typing import List, Optional, Tuple
from datetime import datetime
from qdrant_client import AsyncQdrantClient, models as qmodels

from src.app.config import settings
from src.app.services.embedding_service import embed_query
from src.app.schemas import RagSource

log = logging.getLogger(__name__)

# --- Qdrant Configuration ---
QDRANT_COLLECTION = "chat_vectors"
qdrant = AsyncQdrantClient(
    url=str(settings.QDRANT_URL),
    api_key=settings.QDRANT_API_KEY
)

class QdrantVectorRetriever:
    """
    Retrieves semantic context from Qdrant Cloud.
    """
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    async def aget_relevant_documents(
            self, 
            query: str, 
            chat_id: int,
            sender_names: Optional[List[str]] = None,
            time_ranges: Optional[List[Tuple[datetime, datetime]]] = None
    ) -> List[RagSource]:
        """
        Performs vector search in Qdrant scoped to the specific chat_id.
        """
        # 1. Embed Query
        query_vectors = await embed_query([query])
        if not query_vectors or not query_vectors[0]:
            log.warning("Embedding failed for query. Returning empty results.")
            return []

        query_vector = query_vectors[0]

        # Base Filter: Chat ID
        must_conditions = [
            qmodels.FieldCondition(
                key="chat_id",
                match=qmodels.MatchValue(value=chat_id)
            )
        ]

        # --- FIX FOR TEXT INDEX COMPATIBILITY ---
        # If index is TEXT, we cannot use MatchAny (which requires KEYWORD).
        # We must use MatchText inside a 'Should' clause to simulate "OR".
        if sender_names:
            sender_conditions = []
            for name in sender_names:
                sender_conditions.append(
                    qmodels.FieldCondition(
                        key="sender_name",
                        match=qmodels.MatchText(text=name)  # <--- CHANGED TO MatchText
                    )
                )
            
            # Wrap in 'must' -> 'should' (logical OR between names)
            must_conditions.append(
                qmodels.Filter(
                    should=sender_conditions
                )
            )

        # Conditional: Filter by Multiple Non-Contiguous Time Ranges (Logical OR)
        if time_ranges:
            time_range_conditions = []
            for start_date, end_date in time_ranges:
                range_filter = qmodels.Range(
                    gte=start_date.isoformat() if start_date else None,
                    lte=end_date.isoformat() if end_date else None
                )
                time_range_conditions.append(
                    qmodels.FieldCondition(
                        key="timestamp",
                        range=range_filter
                    )
                )
            
            # Use a 'Should' clause to logically OR the time range conditions
            must_conditions.append(
                qmodels.Filter( # Changed from HasIdCondition which was incorrect for this structure
                    should=time_range_conditions
                )
            )

        try:
            results = await qdrant.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_vector,
                limit=self.top_k,
                query_filter=qmodels.Filter(
                    must=must_conditions
                )
            )
            hits = results.points
        except Exception as e:
            log.error(f"Qdrant search failed: {e}")
            return []

        # 3. Map to RagSource Schema
        sources = []
        for hit in hits:
            payload = hit.payload or {}
            
            if "text" not in payload:
                continue
                
            sources.append(RagSource(
                source_table=payload.get("source_table", "unknown"),
                source_id=payload.get("source_id", 0),
                sender_name=payload.get("sender_name"),
                timestamp=payload.get("timestamp"),
                distance=hit.score,
                text=payload.get("text")
            ))
            
        return sources

# Singleton instance for easy import
retriever = QdrantVectorRetriever(top_k=5)