from __future__ import annotations

import requests

from app.core.config import settings


class NewsTool:
    base_url = "https://newsapi.org/v2/everything"

    def run(self, location: str) -> dict:
        if not settings.news_api_key:
            raise ValueError("News API key missing. Set NEWS_API_KEY in .env.")

        params = {
            "q": f"{location} travel news",
            "sortBy": "publishedAt",
            "pageSize": 5,
            "apiKey": settings.news_api_key,
            "language": "en",
        }
        response = requests.get(self.base_url, params=params, timeout=20)
        if response.status_code in {401, 403}:
            raise ValueError("NewsAPI authentication failed. Check NEWS_API_KEY.")
        response.raise_for_status()
        payload = response.json()
        articles = payload.get("articles", [])
        return {
            "location": location,
            "articles": [
                {
                    "title": article.get("title"),
                    "source": (article.get("source") or {}).get("name"),
                    "published_at": article.get("publishedAt"),
                    "url": article.get("url"),
                    "description": article.get("description"),
                }
                for article in articles
            ],
        }
