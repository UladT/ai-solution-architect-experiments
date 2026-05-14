"""
GNews.io News MCP Server
Provides news search and top-headline tools using the GNews.io free-tier API.

Run standalone:  python news_server.py
MCP transport:   stdio (default)

Requires: GNEWS_API_KEY in .env  (free key at https://gnews.io)
"""

import json
import os
import re
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("gnews-news")

GNEWS_BASE = "https://gnews.io/api/v4"
_API_KEY = os.getenv("GNEWS_API_KEY", "")

VALID_CATEGORIES = [
    "general", "world", "nation", "business",
    "technology", "entertainment", "sports", "science", "health",
]


@mcp.tool()
async def search_news(
    query: str,
    language: str = "en",
    max_articles: int = 5,
) -> str:
    """
    Search for the latest news articles matching a query using GNews.io.

    Args:
        query: Search query (e.g. 'artificial intelligence', 'climate change')
        language: Language code (default 'en')
        max_articles: Number of articles to return (1–10, default 5)

    Returns:
        JSON string with a list of articles (title, description, source, date, url).
    """
    if not _API_KEY:
        return json.dumps({"error": "GNEWS_API_KEY is not set. Add it to .env."})

    query = (query or "").strip()[:120]
    if not query:
        return json.dumps({"error": "Query must be a non-empty string."})
    if not re.fullmatch(r"[A-Za-z0-9 .,'-]{1,120}", query):
        return json.dumps({"error": "Query contains unsupported characters."})

    max_articles = max(1, min(10, int(max_articles)))

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{GNEWS_BASE}/search",
            params={
                "q": query,
                "lang": language,
                "max": max_articles,
                "apikey": _API_KEY,
                "sortby": "publishedAt",
            }
        )

    if resp.status_code != 200:
        return json.dumps({"error": f"GNews API error {resp.status_code}: {resp.text}"})

    data = resp.json()

    if "errors" in data:
        return json.dumps({"error": data["errors"]})

    articles = data.get("articles", [])
    return json.dumps({
        "query": query,
        "total_results": data.get("totalArticles", 0),
        "articles": [
            {
                "title": a.get("title"),
                "description": a.get("description"),
                "source": a.get("source", {}).get("name"),
                "published_at": a.get("publishedAt"),
                "url": a.get("url"),
            }
            for a in articles
        ],
    }, indent=2)


@mcp.tool()
async def get_top_headlines(
    category: str = "general",
    language: str = "en",
    max_articles: int = 5,
) -> str:
    """
    Get the top headline news articles by category using GNews.io.

    Args:
        category: News category – one of: general, world, nation, business,
                  technology, entertainment, sports, science, health
        language: Language code (default 'en')
        max_articles: Number of articles to return (1–10, default 5)

    Returns:
        JSON string with a list of headline articles.
    """
    if not _API_KEY:
        return json.dumps({"error": "GNEWS_API_KEY is not set. Add it to .env."})

    if category not in VALID_CATEGORIES:
        category = "general"

    max_articles = max(1, min(10, int(max_articles)))

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{GNEWS_BASE}/top-headlines",
            params={
                "category": category,
                "lang": language,
                "max": max_articles,
                "apikey": _API_KEY,
            }
        )

    if resp.status_code != 200:
        return json.dumps({"error": f"GNews API error {resp.status_code}: {resp.text}"})

    data = resp.json()

    if "errors" in data:
        return json.dumps({"error": data["errors"]})

    articles = data.get("articles", [])
    return json.dumps({
        "category": category,
        "total_results": data.get("totalArticles", 0),
        "articles": [
            {
                "title": a.get("title"),
                "description": a.get("description"),
                "source": a.get("source", {}).get("name"),
                "published_at": a.get("publishedAt"),
                "url": a.get("url"),
            }
            for a in articles
        ],
    }, indent=2)


if __name__ == "__main__":
    mcp.run()
