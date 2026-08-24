# src/app/services/retrieval_service.py

import logging
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime

from src.app.services.vector_store import get_vector_store, VectorStore
from src.app.services.embedding_service import embed_query
from src.app.schemas import RagSource

log = logging.getLogger(__name__)


class VectorStoreRetriever:
    """Retrieves semantic context using the unified VectorStore abstraction (Chroma Local / Cloud)."""

    def __init__(self, top_k: int = 5, vector_store: Optional[VectorStore] = None):
        self.top_k = top_k
        self._vector_store = vector_store

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store

    def _build_where_filter(
        self,
        chat_id: int,
        sender_names: Optional[List[str]] = None,
        time_ranges: Optional[List[Tuple[datetime, datetime]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Construct ChromaDB-compatible query filter dictionary."""
        and_conditions: List[Dict[str, Any]] = [
            {"chat_id": {"$eq": chat_id}}
        ]

        if sender_names:
            clean_senders = [s.strip() for s in sender_names if s and s.strip()]
            if len(clean_senders) == 1:
                and_conditions.append({"sender_name": {"$eq": clean_senders[0]}})
            elif len(clean_senders) > 1:
                and_conditions.append({
                    "$or": [{"sender_name": {"$eq": name}} for name in clean_senders]
                })

        if time_ranges:
            range_conditions = []
            for start_date, end_date in time_ranges:
                clause = []
                if start_date:
                    clause.append({"timestamp": {"$gte": start_date.isoformat()}})
                if end_date:
                    clause.append({"timestamp": {"$lte": end_date.isoformat()}})
                if len(clause) == 1:
                    range_conditions.append(clause[0])
                elif len(clause) > 1:
                    range_conditions.append({"$and": clause})

            if len(range_conditions) == 1:
                and_conditions.append(range_conditions[0])
            elif len(range_conditions) > 1:
                and_conditions.append({"$or": range_conditions})

        if len(and_conditions) == 1:
            return and_conditions[0]
        return {"$and": and_conditions}

    async def aget_relevant_documents(
        self,
        query: str,
        chat_id: int,
        sender_names: Optional[List[str]] = None,
        time_ranges: Optional[List[Tuple[datetime, datetime]]] = None,
    ) -> List[RagSource]:
        """Perform vector search scoped to specific chat_id, senders, and date ranges."""
        query_vectors = await embed_query(query)
        if not query_vectors or not query_vectors[0]:
            log.warning("Embedding generation failed for query. Returning empty results.")
            return []

        query_vector = query_vectors[0]
        where_filter = self._build_where_filter(chat_id, sender_names, time_ranges)

        try:
            results = await self.vector_store.query(
                query_embedding=query_vector,
                n_results=self.top_k,
                where=where_filter,
            )
        except Exception as e:
            log.error("Vector store query failed: %s", e)
            return []

        sources: List[RagSource] = []
        for res in results:
            meta = res.get("metadata") or {}
            doc_text = res.get("document") or meta.get("text") or ""
            
            if not doc_text:
                continue

            raw_ts = meta.get("timestamp")
            parsed_ts = None
            if raw_ts:
                try:
                    parsed_ts = datetime.fromisoformat(str(raw_ts))
                except (ValueError, TypeError):
                    parsed_ts = None

            sources.append(
                RagSource(
                    source_table=str(meta.get("source_table", "messages")),
                    source_id=int(meta.get("source_id", 0)),
                    sender_name=meta.get("sender_name"),
                    timestamp=parsed_ts,
                    distance=float(res.get("distance", 0.0)),
                    text=doc_text,
                )
            )

        return sources


# Singleton retriever for backward compatibility
retriever = VectorStoreRetriever(top_k=5)