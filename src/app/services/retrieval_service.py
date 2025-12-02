# src/app/services/retrieval_service.py

import logging
from typing import List, Optional
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

    async def aget_relevant_documents(self, query: str, chat_id: int) -> List[RagSource]:
        """
        Performs vector search in Qdrant scoped to the specific chat_id.
        """
        # 1. Embed Query
        query_vectors = await embed_query([query])
        if not query_vectors or not query_vectors[0]:
            log.warning("Embedding failed for query. Returning empty results.")
            return []

        query_vector = query_vectors[0]
        try:
            # FIX: Use query_points (Unified API) and await it
            results = await qdrant.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_vector, # Argument is now 'query', not 'query_vector'
                limit=self.top_k,
                query_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="chat_id",
                            match=qmodels.MatchValue(value=chat_id)
                        )
                    ]
                )
            )
            # FIX: The actual list is inside .points
            hits = results.points
        except Exception as e:
            log.error(f"Qdrant search failed: {e}")
            return []

        # 3. Map to RagSource Schema
        # Note: We stored the text in the payload during ingestion, so no need for a 2nd DB hop!
        sources = []
        for hit in hits:
            payload = hit.payload or {}
            
            # Defensive check for required fields
            if "text" not in payload:
                continue
                
            sources.append(RagSource(
                source_table=payload.get("source_table", "unknown"),
                source_id=payload.get("source_id", 0),
                distance=hit.score, # Qdrant returns cosine similarity score
                text=payload.get("text")
            ))
            
        return sources

# Singleton instance for easy import
retriever = QdrantVectorRetriever(top_k=5)