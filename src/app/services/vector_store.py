# src/app/services/vector_store.py

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import asyncio

import chromadb
from chromadb.config import Settings as ChromaSettings
from src.app.config import settings

log = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract Vector Store interface decoupling application logic from vector database vendors."""

    @abstractmethod
    async def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str],
    ) -> int:
        """Upsert documents with their embeddings and metadata."""
        pass

    @abstractmethod
    async def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query nearest vector neighbors with optional metadata filtering."""
        pass

    @abstractmethod
    async def delete(
        self,
        where: Optional[Dict[str, Any]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete items by ID list or metadata filter."""
        pass

    @abstractmethod
    async def count(self, where: Optional[Dict[str, Any]] = None) -> int:
        """Count items in the collection, optionally filtered by metadata."""
        pass

    @abstractmethod
    async def health(self) -> bool:
        """Verify vector database connectivity."""
        pass


class ChromaVectorStore(VectorStore):
    """Unified ChromaDB vector store supporting Local persistence and Chroma Cloud."""

    def __init__(
        self,
        mode: Optional[str] = None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        api_key: Optional[str] = None,
        tenant: Optional[str] = None,
        database: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        self.mode = (mode or settings.CHROMA_MODE).lower()
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.api_key = api_key or settings.CHROMA_API_KEY
        self.tenant = tenant or settings.CHROMA_TENANT or "default_tenant"
        self.database = database or settings.CHROMA_DATABASE or "default_database"
        self.host = host or settings.CHROMA_HOST
        self.port = port or settings.CHROMA_PORT

        self._client = None
        self._collection = None
        self._init_client()

    def _init_client(self):
        """Instantiate appropriate Chroma client based on execution mode."""
        if self.mode == "cloud":
            log.info(
                "Initializing Chroma Cloud client for tenant=%s, database=%s",
                self.tenant,
                self.database,
            )
            if hasattr(chromadb, "CloudClient") and self.api_key:
                self._client = chromadb.CloudClient(
                    tenant=self.tenant,
                    database=self.database,
                    api_key=self.api_key,
                )
            elif self.host:
                self._client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port or 8000,
                    ssl=True,
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else None,
                    tenant=self.tenant,
                    database=self.database,
                )
            else:
                # Fallback to CloudClient or HttpClient
                self._client = chromadb.HttpClient(
                    ssl=True,
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else None,
                    tenant=self.tenant,
                    database=self.database,
                )
        else:
            log.info(
                "Initializing Local ChromaDB persistent client at %s",
                self.persist_directory,
            )
            os.makedirs(self.persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_directory)

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("Chroma collection '%s' ready.", self.collection_name)

    async def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str],
    ) -> int:
        if not ids:
            return 0

        # Run synchronous Chroma client call in worker threadpool to avoid blocking event loop
        def _sync_upsert():
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            return len(ids)

        return await asyncio.to_thread(_sync_upsert)

    async def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not query_embedding:
            return []

        def _sync_query():
            kwargs: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "include": ["metadatas", "documents", "distances"],
            }
            if where:
                kwargs["where"] = where

            res = self._collection.query(**kwargs)
            results = []

            ids = res.get("ids", [[]])[0]
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]

            for i in range(len(ids)):
                results.append({
                    "id": ids[i],
                    "document": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else 0.0,
                })
            return results

        return await asyncio.to_thread(_sync_query)

    async def delete(
        self,
        where: Optional[Dict[str, Any]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        def _sync_delete():
            kwargs: Dict[str, Any] = {}
            if ids:
                kwargs["ids"] = ids
            if where:
                kwargs["where"] = where
            if kwargs:
                self._collection.delete(**kwargs)

        await asyncio.to_thread(_sync_delete)

    async def count(self, where: Optional[Dict[str, Any]] = None) -> int:
        def _sync_count():
            if where:
                res = self._collection.get(where=where, include=[])
                return len(res.get("ids", []))
            return self._collection.count()

        return await asyncio.to_thread(_sync_count)

    async def health(self) -> bool:
        try:
            def _sync_health():
                self._client.heartbeat()
                return True

            return await asyncio.to_thread(_sync_health)
        except Exception as e:
            log.warning("Chroma health check failed: %s", e)
            return False


_VECTOR_STORE_INSTANCE: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return a singleton VectorStore instance."""
    global _VECTOR_STORE_INSTANCE
    if _VECTOR_STORE_INSTANCE is None:
        _VECTOR_STORE_INSTANCE = ChromaVectorStore()
    return _VECTOR_STORE_INSTANCE
