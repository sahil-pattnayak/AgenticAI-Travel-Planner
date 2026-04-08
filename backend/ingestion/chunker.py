from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(text)
    documents: list[Document] = []
    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "location": location.strip().lower(),
                    "category": categorize_chunk(chunk),
                    "source": source,
                    "source_url": source_url or "",
                },
            )
        )
    return documents
