"""
News context for pair scouting — fetches recent headlines per token
from CryptoPanic and returns a sentiment score + headlines.

Setup: set CRYPTOPANIC_TOKEN in Railway environment variables.
Free token: https://cryptopanic.com/developers/api/

Without a token the module is a no-op — scanner works normally, news
bonuses are simply skipped.

Sentiment scoring:
  Each post has a vote tally: positive / negative / important / lol / saved.
  We compute a net sentiment = (positive + important*2) - (negative*2)
  Normalised to a -1 .. +1 range across recent posts for that token.

Scanner conviction bonus:
  +10  strong bullish news  (net sentiment > 0.5, ≥2 posts)
  +5   mild bullish news    (net sentiment > 0.2)
  -10  strong bearish news  (net sentiment < -0.5, ≥2 posts)
  -5   mild bearish news    (net sentiment < -0.2)
   0   neutral / no data
"""
from __future__ import annotations
import os
import time
import asyncio
import httpx
from pathlib import Path

CRYPTOPANIC_TOKEN = os.getenv("CRYPTOPANIC_TOKEN", "")
API_BASE          = "https://cryptopanic.com/api/free/v1/posts/"

# Cache news responses for 15 minutes to avoid hammering the API
_cache: dict[str, tuple[float, dict]] = {}   # {symbol: (fetched_at, result)}
CACHE_TTL = 900   # 15 minutes

STATE_DIR  = Path(__file__).parent.parent / "state"
NEWS_LOG   = STATE_DIR / "news_cache.json"


async def fetch_news(symbol: str) -> dict:
    """
    Fetch recent news for a token symbol (e.g. 'VET', 'SOL').
    Returns:
      {
        sentiment:   float   -1.0 to +1.0
        post_count:  int
        headlines:   list[str]   up to 3 most recent
        conviction_bonus: int    scanner conviction delta
        label:       str         'bullish' | 'bearish' | 'neutral' | 'no_data'
      }
    """
    _empty = {"sentiment": 0.0, "post_count": 0, "headlines": [],
              "conviction_bonus": 0, "label": "no_data"}

    if not CRYPTOPANIC_TOKEN:
        return _empty

    # Check cache
    cached = _cache.get(symbol)
    if cached and (time.time() - cached[0]) < CACHE_TTL:
        return cached[1]

    try:
        result = await _fetch_from_api(symbol)
        _cache[symbol] = (time.time(), result)
        return result
    except Exception as e:
        print(f"[news] {symbol} fetch failed: {e}", flush=True)
        return _empty


async def fetch_news_batch(symbols: list[str]) -> dict[str, dict]:
    """Fetch news for multiple symbols concurrently."""
    results = await asyncio.gather(*[fetch_news(s) for s in symbols], return_exceptions=True)
    return {
        sym: (r if isinstance(r, dict) else {"sentiment": 0.0, "post_count": 0,
              "headlines": [], "conviction_bonus": 0, "label": "no_data"})
        for sym, r in zip(symbols, results)
    }


async def _fetch_from_api(symbol: str) -> dict:
    """Hit CryptoPanic API and parse the response."""
    params = {
        "auth_token": CRYPTOPANIC_TOKEN,
        "currencies":  symbol,
        "kind":        "news",
        "filter":      "hot",
        "public":      "true",
    }
    async with httpx.AsyncClient(timeout=8.0) as client:
        resp = await client.get(API_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()

    posts = data.get("results", [])[:10]   # cap at 10 most recent
    if not posts:
        return {"sentiment": 0.0, "post_count": 0, "headlines": [],
                "conviction_bonus": 0, "label": "no_data"}

    # Score each post
    scores = []
    headlines = []
    for p in posts:
        votes    = p.get("votes", {})
        positive = votes.get("positive", 0) + votes.get("important", 0) * 2
        negative = votes.get("negative", 0) * 2
        net      = positive - negative
        scores.append(net)
        title = p.get("title", "")
        if title and len(headlines) < 3:
            headlines.append(title[:100])

    # Normalise: divide by max absolute score (or 10 as floor)
    max_abs = max(max(abs(s) for s in scores), 10)
    norm_scores = [s / max_abs for s in scores]
    avg_sentiment = sum(norm_scores) / len(norm_scores)

    # Conviction bonus
    n = len(posts)
    if avg_sentiment > 0.5 and n >= 2:
        bonus, label = 10, "bullish"
    elif avg_sentiment > 0.2:
        bonus, label = 5,  "bullish"
    elif avg_sentiment < -0.5 and n >= 2:
        bonus, label = -10, "bearish"
    elif avg_sentiment < -0.2:
        bonus, label = -5,  "bearish"
    else:
        bonus, label = 0, "neutral"

    return {
        "sentiment":       round(avg_sentiment, 3),
        "post_count":      n,
        "headlines":       headlines,
        "conviction_bonus": bonus,
        "label":           label,
    }


def symbol_from_pair(pair: str) -> str:
    """Extract token symbol from pair string, e.g. 'VET/USDT' → 'VET'."""
    return pair.split("/")[0].upper()
