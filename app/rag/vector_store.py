from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.core.config import settings
from app.rag.embedder import get_embeddings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self, index_dir: str | None = None) -> None:
        self.index_dir = index_dir or settings.faiss_index_dir
        self.embeddings = get_embeddings()
        self._store: FAISS | None = None
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)

    def index_exists(self) -> bool:
        return Path(self.index_dir, "index.faiss").exists() and Path(
            self.index_dir, "index.pkl"
        ).exists()

    def load(self) -> FAISS | None:
        if self._store is not None:
            return self._store
        if not self.index_exists():
            return None
        self._store = FAISS.load_local(
            self.index_dir,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return self._store

    def save(self) -> None:
        if self._store is None:
            return
        self._store.save_local(self.index_dir)

    def add_documents(self, docs: Iterable[Document]) -> None:
        docs_list = list(docs)
        if not docs_list:
            logger.info("No documents provided for indexing.")
            return
        if self._store is None:
            self._store = self.load()
        if self._store is None:
            self._store = FAISS.from_documents(docs_list, self.embeddings)
        else:
            self._store.add_documents(docs_list)
        self.save()
        logger.info("Indexed %s document chunks.", len(docs_list))

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[Document]:
        store = self.load()
        if store is None:
            return []
        return store.similarity_search(query, k=k, filter=metadata_filter)

    def has_location(self, location: str) -> bool:
        store = self.load()
        if store is None:
            return False
        wanted = location.strip().lower()
        for _, doc in store.docstore._dict.items():
            doc_location = str(doc.metadata.get("location", "")).strip().lower()
            if doc_location == wanted:
                return True
        return False
