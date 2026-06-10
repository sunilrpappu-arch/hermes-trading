"""
Candle (OHLCV) adapter — fetches from Binance public REST API.
No authentication required. Works for both paper and live mode.

Falls back gracefully: futures endpoint first, then spot, then empty list.
Results are cached per (symbol, interval) to avoid hammering the API.
"""
import time
import httpx
from typing import Optional

SCHEMA_VERSION = "1.0"

# Cache TTLs by interval (seconds)
_CACHE_TTL = {
    "15m": 13 * 60,
    "1h":  55 * 60,
    "4h":  3 * 55 * 60,
}
_DEFAULT_TTL = 10 * 60

# Cache store: (symbol, interval) -> {"ts": float, "candles": list}
_cache: dict = {}

BINANCE_FUTURES_BASE = "https://fapi.binance.com"
BINANCE_SPOT_BASE    = "https://api.binance.com"


def _symbol(asset: str) -> str:
    return asset.replace("/", "")


def _ttl(interval: str) -> float:
    return _CACHE_TTL.get(interval, _DEFAULT_TTL)


async def fetch(asset: str, interval: str = "1h", limit: int = 100) -> list[dict]:
    """
    Fetch OHLCV candles for `asset` at `interval`.
    Returns list of dicts: {ts, open, high, low, close, volume}
    Most-recent candle last. Returns [] on failure.
    """
    key = (asset, interval)
    now = time.time()
    cached = _cache.get(key)
    if cached and (now - cached["ts"]) < _ttl(interval):
        return cached["candles"]

    sym = _symbol(asset)
    candles = await _fetch_futures(sym, interval, limit)
    if not candles:
        candles = await _fetch_spot(sym, interval, limit)

    _cache[key] = {"ts": now, "candles": candles}
    return candles


async def fetch_multi(asset: str, intervals: list[str] = ("15m", "1h", "4h"), limit: int = 100) -> dict[str, list[dict]]:
    """Fetch multiple intervals for one asset. Returns {interval: [candles]}."""
    import asyncio
    results = await asyncio.gather(*[fetch(asset, iv, limit) for iv in intervals], return_exceptions=True)
    out = {}
    for iv, result in zip(intervals, results):
        if isinstance(result, Exception):
            print(f"[candles] {asset} {iv} fetch failed: {result}", flush=True)
            out[iv] = []
        else:
            out[iv] = result
    return out


async def fetch_all_multi(assets: list[str], intervals: list[str] = ("15m", "1h", "4h"), limit: int = 100) -> dict[str, dict]:
    """Fetch all intervals for all assets. Returns {asset: {interval: [candles]}}."""
    import asyncio
    results = await asyncio.gather(*[fetch_multi(a, intervals, limit) for a in assets], return_exceptions=True)
    out = {}
    for asset, result in zip(assets, results):
        if isinstance(result, Exception):
            print(f"[candles] {asset} multi-fetch failed: {result}", flush=True)
            out[asset] = {iv: [] for iv in intervals}
        else:
            out[asset] = result
    return out


async def _fetch_futures(symbol: str, interval: str, limit: int) -> list[dict]:
    url = f"{BINANCE_FUTURES_BASE}/fapi/v1/klines"
    return await _get_klines(url, symbol, interval, limit)


async def _fetch_spot(symbol: str, interval: str, limit: int) -> list[dict]:
    url = f"{BINANCE_SPOT_BASE}/api/v3/klines"
    return await _get_klines(url, symbol, interval, limit)


async def _get_klines(url: str, symbol: str, interval: str, limit: int) -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, params={"symbol": symbol, "interval": interval, "limit": limit})
            r.raise_for_status()
            raw = r.json()
        return [_parse(k) for k in raw]
    except Exception:
        return []


def _parse(k: list) -> dict:
    return {
        "ts":     int(k[0]),
        "open":   float(k[1]),
        "high":   float(k[2]),
        "low":    float(k[3]),
        "close":  float(k[4]),
        "volume": float(k[5]),
    }


def closes(candles: list[dict]) -> list[float]:
    return [c["close"] for c in candles]


def highs(candles: list[dict]) -> list[float]:
    return [c["high"] for c in candles]


def lows(candles: list[dict]) -> list[float]:
    return [c["low"] for c in candles]
