"""
News context for pair scouting — fetches recent crypto headlines from
free RSS feeds (CoinTelegraph, Decrypt, CoinDesk) and scores sentiment
by matching token mentions against a simple bullish/bearish keyword list.

No API key required. Feeds are cached for 20 minutes to avoid hammering.

Sentiment scoring:
  Each headline mentioning the token is scored:
    +2  per bullish keyword (surge, rally, breakout, partnership, launch, ...)
    -2  per bearish keyword (hack, crash, ban, lawsuit, exploit, ...)
    +1  if token is in title (higher relevance)
  Net score normalised to -1..+1 across matching headlines.

Scanner conviction bonus:
  +10  strong bullish  (net > 0.5, ≥2 headlines)
  +5   mild bullish    (net > 0.2)
  -10  strong bearish  (net < -0.5, ≥2 headlines)
  -5   mild bearish    (net < -0.2)
   0   neutral / no matching headlines
"""
from __future__ import annotations
import re
import time
import asyncio
import httpx
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------------------------
# RSS sources — all free, no auth
# ---------------------------------------------------------------------------

RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://bitcoinmagazine.com/.rss/full/",
]

# ---------------------------------------------------------------------------
# Keyword lists for simple sentiment scoring
# ---------------------------------------------------------------------------

BULLISH_WORDS = {
    "surge", "surges", "surging", "rally", "rallies", "rallying",
    "breakout", "breaks out", "breakthrough", "soar", "soars", "soaring",
    "launch", "launches", "launched", "partnership", "partners", "integrates",
    "adoption", "adopts", "approved", "approval", "upgrade", "upgrades",
    "bullish", "bull", "gains", "gained", "rises", "rise", "rising",
    "record", "high", "milestone", "listing", "listed", "invest", "investment",
    "accumulate", "accumulation", "buy", "buying", "long", "upside",
    "mainnet", "staking", "yield", "airdrop", "reward",
}

BEARISH_WORDS = {
    "crash", "crashes", "crashing", "hack", "hacked", "hacking", "exploit",
    "exploited", "ban", "banned", "banning", "lawsuit", "sued", "sec",
    "investigation", "fraud", "scam", "rug", "rugpull", "dump", "dumping",
    "fall", "falls", "falling", "drop", "drops", "dropping", "plunge",
    "plunges", "bearish", "bear", "sell", "selling", "short", "downside",
    "breach", "stolen", "theft", "fine", "penalty", "warning", "delist",
    "delisted", "vulnerability", "attack", "liquidation", "liquidated",
    "flashloan", "flash loan", "ponzi", "insolvency", "insolvent",
}

# ---------------------------------------------------------------------------
# In-memory cache: {symbol: (fetched_at, result)}
# ---------------------------------------------------------------------------

_feed_cache:   tuple[float, list[dict]] | None = None   # shared feed cache
_result_cache: dict[str, tuple[float, dict]]   = {}     # per-symbol result cache

FEED_CACHE_TTL   = 1200   # 20 min — refresh feed list
RESULT_CACHE_TTL = 900    # 15 min — per-symbol result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def fetch_news(symbol: str) -> dict:
    """
    Return news sentiment for a token symbol (e.g. 'VET', 'SOL').
    Result shape:
      {sentiment, post_count, headlines, conviction_bonus, label}
    """
    cached = _result_cache.get(symbol)
    if cached and (time.time() - cached[0]) < RESULT_CACHE_TTL:
        return cached[1]

    try:
        articles = await _get_feed_articles()
        result   = _score_symbol(symbol, articles)
        _result_cache[symbol] = (time.time(), result)
        return result
    except Exception as e:
        print(f"[news] {symbol} scoring failed: {e}", flush=True)
        return _empty()


async def fetch_news_batch(symbols: list[str]) -> dict[str, dict]:
    """
    Score multiple symbols from the same cached feed — one feed fetch
    for the whole batch regardless of universe size.
    """
    try:
        articles = await _get_feed_articles()
    except Exception as e:
        print(f"[news] feed fetch failed: {e}", flush=True)
        return {s: _empty() for s in symbols}

    results = {}
    for symbol in symbols:
        cached = _result_cache.get(symbol)
        if cached and (time.time() - cached[0]) < RESULT_CACHE_TTL:
            results[symbol] = cached[1]
        else:
            r = _score_symbol(symbol, articles)
            _result_cache[symbol] = (time.time(), r)
            results[symbol] = r
    return results


def symbol_from_pair(pair: str) -> str:
    """'VET/USDT' → 'VET'"""
    return pair.split("/")[0].upper()


# ---------------------------------------------------------------------------
# Feed fetching
# ---------------------------------------------------------------------------

async def _get_feed_articles() -> list[dict]:
    """
    Return a flat list of articles from all RSS feeds, cached for FEED_CACHE_TTL.
    Each article: {title, summary, published_ts}
    """
    global _feed_cache
    if _feed_cache and (time.time() - _feed_cache[0]) < FEED_CACHE_TTL:
        return _feed_cache[1]

    cutoff = time.time() - 86400   # only last 24h articles matter

    tasks   = [_fetch_feed(url) for url in RSS_FEEDS]
    batches = await asyncio.gather(*tasks, return_exceptions=True)

    articles: list[dict] = []
    for batch in batches:
        if isinstance(batch, list):
            articles.extend(a for a in batch if a["published_ts"] >= cutoff)

    _feed_cache = (time.time(), articles)
    print(f"[news] loaded {len(articles)} articles from {len(RSS_FEEDS)} feeds", flush=True)
    return articles


async def _fetch_feed(url: str) -> list[dict]:
    """Fetch and parse one RSS feed. Returns list of article dicts."""
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True,
                                     headers={"User-Agent": "Hermes-Trading/1.0"}) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
    except Exception as e:
        print(f"[news] feed {url} failed: {e}", flush=True)
        return []

    articles = []
    # Handle both RSS <channel><item> and Atom <entry>
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items = root.findall(".//item") or root.findall(".//atom:entry", ns)
    for item in items[:30]:   # cap per feed
        title   = _text(item, ["title"])
        summary = _text(item, ["description", "summary", "atom:summary"], ns)
        pub_str = _text(item, ["pubDate", "published", "atom:published"], ns)
        ts      = _parse_date(pub_str)
        # Extract URL — RSS uses <link>, Atom uses href attribute on <link>
        link_el = item.find("link")
        link = ""
        if link_el is not None:
            link = (link_el.text or "").strip() or link_el.get("href", "")
        if not link:
            link = _text(item, ["atom:link"], ns) or ""
        if title:
            articles.append({"title": title, "summary": summary or "", "published_ts": ts, "url": link})
    return articles


def _text(el, tags: list[str], ns: dict = None) -> str:
    for tag in tags:
        child = el.find(tag, ns or {}) if ns else el.find(tag)
        if child is not None and child.text:
            return child.text.strip()
    return ""


def _parse_date(s: str) -> float:
    """Parse RSS/Atom date string to unix timestamp. Returns 0 on failure."""
    if not s:
        return 0.0
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
    ):
        try:
            dt = datetime.strptime(s.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    return 0.0


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------

def _score_symbol(symbol: str, articles: list[dict]) -> dict:
    """
    Find articles mentioning this token and compute net sentiment.
    Uses full-name aliases for common tokens to improve recall.
    """
    names = _token_names(symbol)
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(n) for n in names) + r')\b',
        re.IGNORECASE,
    )

    matched_headlines = []
    matched_urls      = []
    scores = []

    for article in articles:
        text = article["title"] + " " + article["summary"]
        if not pattern.search(text):
            continue

        title_lower = article["title"].lower()
        words       = set(re.findall(r'\b\w+\b', title_lower))
        bull_hits   = len(words & BULLISH_WORDS)
        bear_hits   = len(words & BEARISH_WORDS)
        # +1 relevance if symbol/name appears in title specifically
        in_title    = 1 if pattern.search(article["title"]) else 0
        net         = (bull_hits * 2) - (bear_hits * 2) + in_title
        scores.append(net)
        if len(matched_headlines) < 3:
            matched_headlines.append(article["title"][:100])
            matched_urls.append(article.get("url", ""))

    if not scores:
        return _empty()

    max_abs = max(max(abs(s) for s in scores), 1)
    avg     = sum(scores) / len(scores) / max_abs
    n       = len(scores)

    if avg > 0.5 and n >= 2:
        bonus, label = 10, "bullish"
    elif avg > 0.2:
        bonus, label = 5, "bullish"
    elif avg < -0.5 and n >= 2:
        bonus, label = -10, "bearish"
    elif avg < -0.2:
        bonus, label = -5, "bearish"
    else:
        bonus, label = 0, "neutral"

    return {
        "sentiment":        round(avg, 3),
        "post_count":       n,
        "headlines":        matched_headlines,
        "headline_urls":    matched_urls,
        "conviction_bonus": bonus,
        "label":            label,
    }


def _token_names(symbol: str) -> list[str]:
    """Return symbol + common full names for better headline matching."""
    aliases: dict[str, list[str]] = {
        "BTC":  ["Bitcoin"],
        "ETH":  ["Ethereum"],
        "SOL":  ["Solana"],
        "BNB":  ["Binance", "BNB"],
        "XRP":  ["Ripple", "XRP"],
        "ADA":  ["Cardano"],
        "DOT":  ["Polkadot"],
        "LINK": ["Chainlink"],
        "AVAX": ["Avalanche"],
        "MATIC":["Polygon", "MATIC"],
        "UNI":  ["Uniswap"],
        "ATOM": ["Cosmos"],
        "NEAR": ["NEAR Protocol", "NEAR"],
        "APT":  ["Aptos"],
        "OP":   ["Optimism"],
        "ARB":  ["Arbitrum"],
        "INJ":  ["Injective"],
        "SUI":  ["Sui"],
        "VET":  ["VeChain", "VET"],
        "TIA":  ["Celestia"],
        "LDO":  ["Lido"],
        "AAVE": ["Aave"],
        "CRV":  ["Curve"],
        "MKR":  ["Maker", "MakerDAO"],
        "DOGE": ["Dogecoin"],
        "SHIB": ["Shiba", "SHIB"],
        "PEPE": ["Pepe"],
        "WIF":  ["dogwifhat", "WIF"],
        "ENA":  ["Ethena"],
        "PENDLE":["Pendle"],
    }
    names = aliases.get(symbol, [])
    # Always include the raw symbol itself
    if symbol not in names:
        names = [symbol] + names
    return names


def _empty() -> dict:
    return {"sentiment": 0.0, "post_count": 0, "headlines": [],
            "conviction_bonus": 0, "label": "no_data"}
