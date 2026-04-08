from __future__ import annotations

from langchain_core.documents import Document

from backend.rag.vector_store import VectorStoreManager


class TravelRetriever:
    def __init__(self, vector_store: VectorStoreManager) -> None:
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        location: str | None = None,
        k: int = 5,
    ) -> list[Document]:
        if not location:
            return self.vector_store.similarity_search(query=query, k=k, metadata_filter=None)

        wanted = location.strip().lower()
        filtered = self.vector_store.similarity_search(
            query=query,
            k=k,
            metadata_filter={"location": wanted},
        )
        if filtered:
            return filtered

        # Fallback: run unfiltered retrieval and keep near-match locations first.
        broad = self.vector_store.similarity_search(query=query, k=max(12, k * 2), metadata_filter=None)
        narrowed = [
            doc
            for doc in broad
            if wanted in str(doc.metadata.get("location", "")).strip().lower()
        ]
        return (narrowed or broad)[:k]
