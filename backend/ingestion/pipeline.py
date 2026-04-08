from __future__ import annotations

import logging
from urllib.parse import quote_plus

from langchain_core.documents import Document

from backend.ingestion.chunker import chunk_text
from backend.ingestion.cleaner import clean_html, clean_text
from backend.ingestion.crawler import TravelCrawler
from backend.rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


def _build_sources(location_name: str) -> list[tuple[str, str]]:
    title = location_name.strip().replace(" ", "_")
    q = quote_plus(location_name.strip())
    return [
        (f"https://en.wikivoyage.org/wiki/{title}", "wikivoyage"),
        (f"https://en.wikipedia.org/wiki/{title}", "wiki"),
        (f"https://www.nomadicmatt.com/?s={q}", "blog"),
    ]


async def ingest_location(location_name: str, vector_store: VectorStoreManager) -> int:
    crawler = TravelCrawler()
    sources = _build_sources(location_name)
    urls = [url for url, _ in sources]
    source_map = {url: source for url, source in sources}

    results = await crawler.crawl_urls(urls)
    all_docs: list[Document] = []

    for item in results:
        if not item["success"]:
            logger.warning("Failed crawl for %s: %r", item["url"], item["error"])
            continue
        if item.get("error"):
            logger.warning("Crawler warning for %s: %r", item["url"], item["error"])
        text = clean_text(item["text"])
        if len(text) < 200:
            text = clean_html(item["html"])
        if len(text) < 200:
            logger.warning("Skipping low-content page: %s", item["url"])
            continue
        source = source_map.get(item["url"], "unknown")
        docs = chunk_text(
            text=text,
            location=location_name,
            source=source,
            source_url=item["url"],
            chunk_size=400,
            chunk_overlap=50,
        )
        all_docs.extend(docs)

    vector_store.add_documents(all_docs)
    logger.info("Ingestion complete for %s with %s chunks.", location_name, len(all_docs))
    return len(all_docs)
