from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

import requests
from crawl4ai import AsyncWebCrawler

from app.core.config import settings

logger = logging.getLogger(__name__)


class TravelCrawler:
    @staticmethod
    def fallback_fetch(url: str) -> dict[str, Any]:
        try:
            resp = requests.get(
                url,
                timeout=25,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                    )
                },
            )
            resp.raise_for_status()
            return {
                "url": url,
                "success": True,
                "html": resp.text,
                "text": "",
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "url": url,
                "success": False,
                "html": "",
                "text": "",
                "error": f"fallback_failed: {exc!r}",
            }

    async def crawl_url(self, url: str) -> dict[str, Any]:
        if sys.platform.startswith("win") and not settings.crawl4ai_enabled:
            fallback = await asyncio.to_thread(self.fallback_fetch, url)
            if fallback["success"]:
                fallback["error"] = "crawl4ai_disabled_on_windows_fallback_used"
            return fallback

        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
            html = getattr(result, "html", "") or ""
            markdown = getattr(result, "markdown", "") or ""
            success = bool(html or markdown)
            if success:
                return {
                    "url": url,
                    "success": True,
                    "html": html,
                    "text": markdown,
                    "error": None,
                }
            fallback = await asyncio.to_thread(self.fallback_fetch, url)
            if fallback["success"]:
                fallback["error"] = "crawl4ai_empty_content_fallback_used"
            return fallback
        except Exception as exc:  # noqa: BLE001
            fallback = await asyncio.to_thread(self.fallback_fetch, url)
            if fallback["success"]:
                fallback["error"] = f"crawl4ai_failed_fallback_used: {exc!r}"
            else:
                fallback["error"] = f"crawl4ai_failed: {exc!r}; {fallback['error']}"
            return fallback

    async def crawl_urls(self, urls: list[str]) -> list[dict[str, Any]]:
        logger.info("Crawling %s sources.", len(urls))
        tasks = [self.crawl_url(url) for url in urls]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        processed: list[dict[str, Any]] = []
        for url, item in zip(urls, raw, strict=True):
            if isinstance(item, Exception):
                processed.append(
                    {
                        "url": url,
                        "success": False,
                        "html": "",
                        "text": "",
                        "error": str(item),
                    }
                )
            else:
                processed.append(item)
        return processed
