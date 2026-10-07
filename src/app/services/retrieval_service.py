# src/app/services/retrieval_service.py

import logging
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timezone

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
    ) -> Optional[Dict[str, Any]]:
        """Construct ChromaDB-compatible query filter dictionary scoped to chat_id and senders."""
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
        where_filter = self._build_where_filter(chat_id, sender_names)
        fetch_k = self.top_k * 4 if time_ranges else self.top_k

        try:
            results = await self.vector_store.query(
                query_embedding=query_vector,
                n_results=fetch_k,
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
                    if parsed_ts.tzinfo is None:
                        parsed_ts = parsed_ts.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    parsed_ts = None

            # Accurate Python-based date filtering prevents ChromaDB operator string crashes
            if time_ranges and parsed_ts:
                in_range = False
                for start_date, end_date in time_ranges:
                    s_date = start_date if (start_date is None or start_date.tzinfo) else start_date.replace(tzinfo=timezone.utc)
                    e_date = end_date if (end_date is None or end_date.tzinfo) else end_date.replace(tzinfo=timezone.utc)
                    if (s_date is None or parsed_ts >= s_date) and (e_date is None or parsed_ts <= e_date):
                        in_range = True
                        break
                if not in_range:
                    continue

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

            if len(sources) >= self.top_k:
                break

        return sources


# Singleton retriever for backward compatibility
retriever = VectorStoreRetriever(top_k=5)