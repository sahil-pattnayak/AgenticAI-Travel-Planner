from __future__ import annotations

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from app.rag.embedder import get_embeddings


def categorize_chunk(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ["safe", "crime", "alert", "warning", "scam"]):
        return "safety"
    if any(word in lowered for word in ["pack", "luggage", "carry", "clothes"]):
        return "packing"
    return "itinerary"


def chunk_text(
    text: str,
    location: str,
    source: str,
    source_url: str | None = None,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[Document]:
    _ = (chunk_size, chunk_overlap)  # kept for call compatibility
    base_meta = {"location": location.strip().lower(), "source": source, "source_url": source_url or ""}
    semantic_splitter = SemanticChunker(get_embeddings(), breakpoint_threshold_type="percentile")
    semantic_docs = semantic_splitter.split_documents([Document(page_content=text, metadata=base_meta)])
    for doc in semantic_docs:
        doc.metadata["category"] = categorize_chunk(doc.page_content)
    return semantic_docs
