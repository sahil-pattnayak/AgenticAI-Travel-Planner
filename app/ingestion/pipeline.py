from __future__ import annotations

import logging
import re
from urllib.parse import quote_plus

from langchain_core.documents import Document

from app.ingestion.chunker import chunk_text
from app.ingestion.cleaner import clean_html, clean_text
from app.ingestion.crawler import TravelCrawler
from app.rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


def slugify_location_part(value: str) -> str:
    normalized = value.strip().lower().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    return re.sub(r"-{2,}", "-", normalized).strip("-")


def build_incredible_india_urls(location_name: str) -> list[str]:
    # Supports patterns like:
    # /en/maharashtra/pune
    # /en/madhya-pradesh/gwalior/kuno-national-park
    raw_parts = [part.strip() for part in re.split(r",|/|\\|\\|", location_name) if part.strip()]
    parts = [slugify_location_part(part) for part in raw_parts if slugify_location_part(part)]
    urls: list[str] = []
    if parts:
        for i in range(1, len(parts) + 1):
            urls.append(f"https://www.incredibleindia.gov.in/en/{'/'.join(parts[:i])}")
    urls.append(f"https://www.incredibleindia.gov.in/en/search?query={quote_plus(location_name.strip())}")
    return list(dict.fromkeys(urls))


def build_sources(location_name: str) -> list[tuple[str, str]]:
    title = location_name.strip().replace(" ", "_")
    q = quote_plus(location_name.strip())
    sources: list[tuple[str, str]] = [
        (f"https://en.wikivoyage.org/wiki/{title}", "wikivoyage"),
        (f"https://en.wikipedia.org/wiki/{title}", "wiki"),
        (f"https://www.nomadicmatt.com/?s={q}", "blog"),
    ]
    sources.extend((url, "incredibleindia") for url in build_incredible_india_urls(location_name))
    return sources


async def ingest_location(location_name: str, vector_store: VectorStoreManager) -> int:
    crawler = TravelCrawler()
    sources = build_sources(location_name)
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
