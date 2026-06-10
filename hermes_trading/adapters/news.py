import httpx
import time
import os

SCHEMA_VERSION = "1.0"


class SchemaError(Exception):
    pass


async def fetch() -> dict:
    news_api_key = os.getenv("NEWS_API_KEY", "")
    headlines = []

    if news_api_key:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": "bitcoin crypto",
                        "sortBy": "publishedAt",
                        "pageSize": 5,
                        "apiKey": news_api_key,
                    },
                )
                r.raise_for_status()
                data = r.json()
                headlines = [a["title"] for a in data.get("articles", [])]
        except Exception:
            pass

    result = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": int(time.time()),
        "headlines": headlines,
        "source": "newsapi" if headlines else "unavailable",
    }

    if "schema_version" not in result:
        raise SchemaError("news adapter missing schema_version")

    return result
