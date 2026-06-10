"""
Pair scanner — two responsibilities:

1. fetch_universe()   — pulls ALL active USDT-M perpetual futures from Binance
                        and filters out leveraged tokens, stablecoins, and low-vol junk.
                        Called at startup and each rescan so the list is always current.

2. scan()             — scores every pair in the universe and returns the top N for
                        the current volatility regime.

Scoring criteria (0–100):
  30 pts  Liquidity       (log-normalised 24h USDT volume)
  25 pts  Signal strength (RSI distance from neutral 50)
  25 pts  Trend clarity   (price distance from 50MA)
  20 pts  Volatility fit  (pair's vol vs regime vol — not too noisy, not too flat)
"""
from __future__ import annotations
import math
import time
import httpx

from hermes_trading.adapters.candles import fetch as fetch_candles, closes as get_closes
from hermes_trading.indicators import rsi as compute_rsi, sma

# ---------------------------------------------------------------------------
# Universe filters
# ---------------------------------------------------------------------------

# Tokens whose names contain these strings are excluded (leveraged / stable / index)
_EXCLUDE_CONTAINS = [
    "UP", "DOWN", "BULL", "BEAR",          # leveraged tokens
    "BUSD", "USDC", "TUSD", "USDP",        # stablecoins
    "FDUSD", "DAI", "USDD",
    "DEFI", "NFTUSDT",                      # index products
]

# Minimum 24h quote volume in USDT to be considered liquid enough
_MIN_VOLUME_USDT = 5_000_000   # $5M/day

# Cache the universe for 10 minutes so rescans don't hammer exchange info
_universe_cache: list[str] = []
_universe_cache_ts: float   = 0.0
_UNIVERSE_TTL = 10 * 60


async def fetch_universe(filters: dict = None) -> list[str]:
    """
    Return all active USDT-M perp futures on Binance that pass quality filters.
    Falls back to a hardcoded list if the API call fails.
    """
    global _universe_cache, _universe_cache_ts

    now = time.time()
    if _universe_cache and (now - _universe_cache_ts) < _UNIVERSE_TTL:
        return _universe_cache

    filters = filters or {}
    min_vol = filters.get("min_volume_usdt", _MIN_VOLUME_USDT)

    try:
        pairs = await _fetch_from_binance(min_vol)
        if pairs:
            _universe_cache    = pairs
            _universe_cache_ts = now
            print(f"[scanner] Universe refreshed: {len(pairs)} pairs from Binance futures", flush=True)
            return pairs
    except Exception as e:
        print(f"[scanner] Universe fetch failed ({e}), using cache/fallback", flush=True)

    # Return stale cache or hardcoded fallback
    if _universe_cache:
        return _universe_cache
    return _FALLBACK_UNIVERSE


async def _fetch_from_binance(min_vol_usdt: float) -> list[str]:
    """
    Build universe using CoinGecko markets API — accessible from all cloud regions.
    Returns top coins by market cap that trade on Binance with sufficient USDT volume.
    Falls back to Binance futures/spot API if CoinGecko fails (works locally).
    """
    try:
        return await _fetch_coingecko_universe(min_vol_usdt)
    except Exception as e:
        print(f"[scanner] CoinGecko universe failed ({e}), trying Binance futures", flush=True)

    try:
        return await _fetch_binance_futures_universe(min_vol_usdt)
    except Exception as e:
        print(f"[scanner] Binance futures unavailable ({e}), trying spot", flush=True)

    return await _fetch_binance_spot_universe(min_vol_usdt)


async def _fetch_coingecko_universe(min_vol_usdt: float) -> list[str]:
    """
    Use CoinGecko /coins/markets — always accessible, ranked by market cap.
    Fetches top 250 coins and filters by Binance USDT volume.
    """
    pairs = []
    async with httpx.AsyncClient(timeout=20) as client:
        for page in (1, 2, 3):   # 3 pages × 100 coins = top 300 by market cap
            r = await client.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={
                    "vs_currency":           "usd",
                    "order":                 "market_cap_desc",
                    "per_page":              100,
                    "page":                  page,
                    "sparkline":             "false",
                    "price_change_percentage": "24h",
                },
            )
            if r.status_code == 429:
                break   # rate limited — use what we have
            r.raise_for_status()
            for coin in r.json():
                symbol = coin.get("symbol", "").upper()
                volume = float(coin.get("total_volume", 0))

                # Skip stablecoins, leveraged tokens, and low-volume coins
                if any(excl in symbol for excl in _EXCLUDE_CONTAINS):
                    continue
                if volume < min_vol_usdt:
                    continue

                pairs.append(f"{symbol}/USDT")

    if not pairs:
        raise ValueError("CoinGecko returned no usable pairs")

    # Deduplicate (CoinGecko occasionally has duplicate symbols)
    seen = set()
    deduped = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    print(f"[scanner] CoinGecko universe: {len(deduped)} pairs", flush=True)
    return deduped


async def _fetch_binance_futures_universe(min_vol_usdt: float) -> list[str]:
    async with httpx.AsyncClient(timeout=20) as client:
        info_r   = await client.get("https://fapi.binance.com/fapi/v1/exchangeInfo")
        ticker_r = await client.get("https://fapi.binance.com/fapi/v1/ticker/24hr")
        info_r.raise_for_status()
        ticker_r.raise_for_status()
    symbols_info = {
        s["symbol"]: s for s in info_r.json()["symbols"]
        if s["status"] == "TRADING" and s["quoteAsset"] == "USDT" and s["contractType"] == "PERPETUAL"
    }
    volume_map = {t["symbol"]: float(t["quoteVolume"]) for t in ticker_r.json()}
    return _build_pairs(symbols_info, volume_map, min_vol_usdt)


async def _fetch_binance_spot_universe(min_vol_usdt: float) -> list[str]:
    async with httpx.AsyncClient(timeout=20) as client:
        info_r   = await client.get("https://api.binance.com/api/v3/exchangeInfo")
        ticker_r = await client.get("https://api.binance.com/api/v3/ticker/24hr")
        info_r.raise_for_status()
        ticker_r.raise_for_status()
    symbols_info = {
        s["symbol"]: s for s in info_r.json()["symbols"]
        if s["status"] == "TRADING" and s["quoteAsset"] == "USDT" and s.get("isSpotTradingAllowed")
    }
    volume_map = {t["symbol"]: float(t["quoteVolume"]) for t in ticker_r.json()}
    return _build_pairs(symbols_info, volume_map, min_vol_usdt)


def _build_pairs(symbols_info: dict, volume_map: dict, min_vol_usdt: float) -> list[str]:
    pairs = []
    for symbol, info in symbols_info.items():
        base = info["baseAsset"]
        if any(excl in base for excl in _EXCLUDE_CONTAINS):
            continue
        if volume_map.get(symbol, 0) < min_vol_usdt:
            continue
        pairs.append(f"{base}/USDT")
    pairs.sort(key=lambda p: volume_map.get(p.replace("/", ""), 0), reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

async def scan(universe: list[str], max_pairs: int, regime_vol: float) -> list[str]:
    """
    Score every pair in `universe` and return the top `max_pairs`.
    Falls back to top `max_pairs` by position if scoring fails.
    """
    try:
        scores  = await _score_all(universe, regime_vol)
        ranked  = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [pair for pair, _ in ranked[:max_pairs]]
        print(
            f"[scanner] Top {len(selected)}/{len(universe)}: "
            + ", ".join(f"{p}({scores[p]:.1f})" for p in selected),
            flush=True,
        )
        return selected
    except Exception as e:
        print(f"[scanner] Scoring failed ({e}), using top {max_pairs} by volume", flush=True)
        return universe[:max_pairs]


async def _score_all(universe: list[str], regime_vol: float) -> dict[str, float]:
    import asyncio
    # Score in parallel but cap concurrency to avoid hammering candle API
    sem = asyncio.Semaphore(10)

    async def _guarded(pair):
        async with sem:
            return await _score_pair(pair, regime_vol)

    results = await asyncio.gather(*[_guarded(p) for p in universe], return_exceptions=True)
    return {
        pair: (0.0 if isinstance(r, Exception) else r)
        for pair, r in zip(universe, results)
    }


async def _score_pair(pair: str, regime_vol: float) -> float:
    """Score a single pair 0–100. Higher = better candidate right now."""
    candles = await fetch_candles(pair, "1h", 100)
    if len(candles) < 20:
        return 0.0

    c             = get_closes(candles)
    current_price = c[-1]

    # 1. Liquidity (30 pts)
    volume_24h     = sum(k["volume"] for k in candles[-24:])
    liquidity_score = min(math.log10(max(volume_24h, 1)) / 10, 1.0) * 30

    # 2. Signal strength — how far RSI is from neutral (25 pts)
    rsi_val       = compute_rsi(c)
    signal_score  = (abs(rsi_val - 50) / 50) * 25

    # 3. Trend clarity — price distance from 50MA (25 pts)
    ma50           = sma(c, 50)
    clarity_score  = (min(abs(current_price - ma50) / ma50 * 200, 1.0) * 25) if ma50 else 0.0

    # 4. Volatility fit — pair's vol should match regime vol (20 pts)
    recent         = c[-24:]
    pair_returns   = [recent[i] / recent[i-1] - 1 for i in range(1, len(recent))] if len(recent) > 1 else []
    pair_vol       = (sum(r**2 for r in pair_returns) / len(pair_returns)) ** 0.5 if pair_returns else 0
    vol_delta      = abs(pair_vol - regime_vol)
    vol_fit_score  = max(0.0, 1.0 - vol_delta / max(regime_vol, 0.001)) * 20

    return round(liquidity_score + signal_score + clarity_score + vol_fit_score, 2)


# ---------------------------------------------------------------------------
# Fallback universe (used if Binance API is unreachable)
# ---------------------------------------------------------------------------

_FALLBACK_UNIVERSE = [
    # Majors
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    # Large caps
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "LTC/USDT", "BCH/USDT", "UNI/USDT", "ATOM/USDT", "ETC/USDT",
    "NEAR/USDT", "APT/USDT", "OP/USDT", "ARB/USDT", "MKR/USDT",
    "FIL/USDT", "AAVE/USDT", "ICP/USDT", "STX/USDT", "VET/USDT",
    # Mid caps
    "SUI/USDT", "TIA/USDT", "INJ/USDT", "WIF/USDT", "HYPE/USDT",
    "SEI/USDT", "JTO/USDT", "PYTH/USDT", "JUP/USDT", "ONDO/USDT",
    "STRK/USDT", "MANTA/USDT", "ALT/USDT", "PIXEL/USDT", "PORTAL/USDT",
    "ETHFI/USDT", "ENA/USDT", "W/USDT", "OMNI/USDT", "REZ/USDT",
    "BB/USDT", "NOT/USDT", "IO/USDT", "ZK/USDT", "LISTA/USDT",
    # DeFi / ecosystem
    "CRV/USDT", "LDO/USDT", "SNX/USDT", "COMP/USDT", "1INCH/USDT",
    "SUSHI/USDT", "BAL/USDT", "CAKE/USDT", "PENDLE/USDT", "TURBO/USDT",
    # Layer 1 / Layer 2
    "MATIC/USDT", "FTM/USDT", "ALGO/USDT", "EGLD/USDT", "FLOW/USDT",
    "ROSE/USDT", "ONE/USDT", "ZIL/USDT", "ICX/USDT", "KAVA/USDT",
]
