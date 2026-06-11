"""
Price adapter.

Paper mode:  Binance spot ticker (public, no auth, works from all Railway regions)
             Falls back to CoinGecko for any symbol Binance spot doesn't carry.
Live mode:   Binance spot ticker (same endpoint, swapped to futures in exchange.py)

Batch fetch uses a single Binance request per symbol via asyncio.gather with
a semaphore to avoid hammering the API when tracking 100+ pairs.
"""
import os
import time
import httpx
import asyncio

SCHEMA_VERSION = "1.0"
MODE = os.getenv("HERMES_TRADING_MODE", "paper").lower()

# CoinGecko IDs — used only as a last-resort fallback for symbols not on Binance spot
COINGECKO_IDS = {
    "BTC/USDT":  "bitcoin",
    "ETH/USDT":  "ethereum",
    "SOL/USDT":  "solana",
    "BNB/USDT":  "binancecoin",
    "XRP/USDT":  "ripple",
    "HYPE/USDT": "hyperliquid",
    "DOGE/USDT": "dogecoin",
    "ADA/USDT":  "cardano",
    "AVAX/USDT": "avalanche-2",
    "DOT/USDT":  "polkadot",
    "LINK/USDT": "chainlink",
    "LTC/USDT":  "litecoin",
    "BCH/USDT":  "bitcoin-cash",
    "UNI/USDT":  "uniswap",
    "ATOM/USDT": "cosmos",
}

# Simple per-asset cache (TTL 50s) to limit API calls
_cache: dict[str, dict] = {}
_cache_ts: dict[str, float] = {}
_CACHE_TTL = 12


class SchemaError(Exception):
    pass


async def fetch(asset: str = "BTC/USDT") -> dict:
    return await _fetch_binance_spot(asset)


async def fetch_all(assets: list[str]) -> dict[str, dict]:
    """Fetch prices for all assets in parallel (max 20 concurrent requests)."""
    sem = asyncio.Semaphore(20)

    async def _guarded(asset):
        async with sem:
            return asset, await _fetch_binance_spot(asset)

    results = await asyncio.gather(*[_guarded(a) for a in assets], return_exceptions=True)
    out = {}
    for item in results:
        if isinstance(item, Exception):
            continue
        asset, data = item
        out[asset] = data
    return out


async def _fetch_binance_spot(asset: str) -> dict:
    """
    Fetch from Binance spot ticker/24hr — public endpoint, accessible globally.
    Falls back to CoinGecko if Binance returns an error for this symbol.
    """
    now = time.time()
    cached = _cache.get(asset)
    if cached and (now - _cache_ts.get(asset, 0)) < _CACHE_TTL:
        return cached

    symbol = asset.replace("/", "")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}")
            r.raise_for_status()
            data = r.json()

        result = _schema(asset, {
            "price":               float(data["lastPrice"]),
            "volume_24h":          float(data["quoteVolume"]),
            "price_change_pct_24h": float(data["priceChangePercent"]),
            "high_24h":            float(data["highPrice"]),
            "low_24h":             float(data["lowPrice"]),
        })
    except Exception:
        # Symbol not on Binance spot (e.g. HYPE) — fall back to CoinGecko
        result = await _fetch_coingecko(asset)

    _cache[asset] = result
    _cache_ts[asset] = now
    return result


async def _fetch_coingecko(asset: str) -> dict:
    cg_id = COINGECKO_IDS.get(asset)
    if not cg_id:
        # Return zeroed-out schema so the loop doesn't crash
        return _schema(asset, {"price": 0.0, "volume_24h": 0.0,
                                "price_change_pct_24h": 0.0, "high_24h": None, "low_24h": None})

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": cg_id, "vs_currencies": "usd",
                    "include_24hr_vol": "true", "include_24hr_change": "true"},
        )
        r.raise_for_status()
        data = r.json().get(cg_id, {})

    return _schema(asset, {
        "price":               float(data.get("usd", 0)),
        "volume_24h":          float(data.get("usd_24h_vol", 0)),
        "price_change_pct_24h": float(data.get("usd_24h_change", 0)),
        "high_24h":            None,
        "low_24h":             None,
    })


def _schema(asset: str, fields: dict) -> dict:
    result = {"schema_version": SCHEMA_VERSION, "asset": asset,
              "timestamp": int(time.time()), **fields}
    missing = {"schema_version", "asset", "price", "timestamp"} - result.keys()
    if missing:
        raise SchemaError(f"price adapter missing fields: {missing}")
    return result
