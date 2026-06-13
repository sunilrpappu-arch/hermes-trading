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
from hermes_trading.news import fetch_news_batch, symbol_from_pair

from hermes_trading.adapters.candles import fetch as fetch_candles, closes as get_closes, highs as get_highs, lows as get_lows
from hermes_trading.indicators import (
    rsi as compute_rsi, sma, rsi_divergence,
    liquidity_grab as detect_liquidity_grab,
    breakout_detector, candlestick_patterns, chart_patterns,
    bb_squeeze, macd as compute_macd, vwap as compute_vwap,
)

# ---------------------------------------------------------------------------
# Universe filters
# ---------------------------------------------------------------------------

# Tokens whose names contain these strings are excluded (leveraged / stable / index)
_EXCLUDE_CONTAINS = [
    "UP", "DOWN", "BULL", "BEAR",          # leveraged tokens
    "BUSD", "USDC", "TUSD", "USDP",        # stablecoins
    "FDUSD", "DAI", "USDD", "STABLE",      # stablecoins / stable-pegged tokens
    "DEFI", "NFTUSDT",                      # index products
]

# Exact symbols to always exclude — tokenised commodities/metals that behave
# differently from crypto and cause candle fetch issues on Binance perps
_EXCLUDE_EXACT = {
    "XAUT/USDT",   # tokenised gold — illiquid candles, stale data
    "PAXG/USDT",   # tokenised gold — same issues
    "AEUR/USDT",   # EUR stablecoin
    "EURI/USDT",   # EUR stablecoin
}

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
            pairs = [p for p in pairs if p not in _EXCLUDE_EXACT]
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

                # Skip malformed, stablecoins, leveraged tokens, low-volume
                if len(symbol) < 2:
                    continue
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

async def scan(universe: list[str], max_pairs: int, regime_vol: float,
               regime_info: dict = None) -> list[str]:
    """
    Score every pair in `universe` and return the top `max_pairs`.
    Falls back to top `max_pairs` by position if scoring fails.

    regime_info (optional): full output of volatility.detect() — used to apply
    Total2/Total3 macro biases (alt_season / btc_dom_rising).
    """
    try:
        scores  = await _score_all(universe, regime_vol, regime_info or {})
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


async def _score_all(universe: list[str], regime_vol: float, regime_info: dict) -> dict[str, float]:
    import asyncio
    # Fetch news for all pairs in one batch before scoring (15m cache, graceful if no token)
    symbols   = [symbol_from_pair(p) for p in universe]
    news_data = await fetch_news_batch(symbols)   # {symbol: news_dict}

    # Score in parallel but cap concurrency to avoid hammering candle API
    sem = asyncio.Semaphore(10)

    async def _guarded(pair):
        async with sem:
            symbol = symbol_from_pair(pair)
            return await _score_pair(pair, regime_vol, regime_info, news_data.get(symbol, {}))

    results = await asyncio.gather(*[_guarded(p) for p in universe], return_exceptions=True)
    return {
        pair: (0.0 if isinstance(r, Exception) else r)
        for pair, r in zip(universe, results)
    }


async def _score_pair(pair: str, regime_vol: float, regime_info: dict,
                      news: dict = None) -> float:
    """
    Score a single pair 0–100 base + conviction + macro bonuses + news bonus.

    Base score (0–100):
      30 pts  Liquidity       (log-normalised 24h USDT volume)
      25 pts  Signal strength (RSI distance from neutral 50)
      25 pts  Trend clarity   (price distance from 50MA)
      20 pts  Volatility fit  (pair's vol vs regime vol)

    Conviction bonus:
      +10  RSI extreme (>70 or <30)
      +10  RSI divergence (1H)
      +10  Breakout / breakdown (1H)
      +10  Liquidity grab (15m wick sweep)
      +10  MACD crossover 1H aligned with RSI direction (−10 opposing)
      +15  MACD crossover 15m aligned with RSI direction (−10 opposing)
      +10  Candlestick patterns
      +10/+20  BB squeeze → expansion
      +5/+15   VWAP alignment / band touch
      +15  Chart patterns (confidence-weighted)

    News bonus (CryptoPanic, 15m cached, optional):
      +10  Strong bullish sentiment (≥2 posts, net sentiment > 0.5)
      +5   Mild bullish sentiment
      -10  Strong bearish sentiment (≥2 posts, net sentiment < -0.5)
      -5   Mild bearish sentiment

    Macro bonus (Total2/Total3 layer):
      +15  alt_season=True and pair is not BTC/ETH (alts outperforming)
      -15  btc_dom_rising=True and pair is not BTC/ETH (alts losing share)
      +10  macro_sentiment agrees with pair's RSI direction (bull → rsi<50, bear → rsi>50)
    """
    candles_1h  = await fetch_candles(pair, "1h", 100)
    candles_15m = await fetch_candles(pair, "15m", 60)
    if len(candles_1h) < 20:
        return 0.0

    c             = get_closes(candles_1h)
    c15m          = get_closes(candles_15m)
    current_price = c[-1]

    # 1. Liquidity (30 pts)
    volume_24h     = sum(k["volume"] for k in candles_1h[-24:])
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

    base_score = liquidity_score + signal_score + clarity_score + vol_fit_score

    # 5. Conviction bonus — multiple aligned signals boost ranking
    conviction = 0
    signals    = []

    # RSI approaching tradeable threshold (+10).
    # Fire for RSI 30-40 (bull dip setup) and RSI 60-70 (bear short setup).
    # Penalise RSI > 75 or < 25 — price is so extended the engine can't enter.
    if (30 <= rsi_val <= 40) or (60 <= rsi_val <= 70):
        conviction += 10
        signals.append(f"rsi_approach({rsi_val:.0f})")
    elif rsi_val > 75 or rsi_val < 25:
        conviction -= 10
        signals.append(f"rsi_too_extended({rsi_val:.0f})")

    # RSI divergence on 1H (+10)
    try:
        div = rsi_divergence(c)
        if div["bullish"] or div["bearish"]:
            conviction += 10
            signals.append("rsi_div")
    except Exception:
        pass

    # Breakout / breakdown on 1H (+10)
    try:
        bo = breakout_detector(candles_1h)
        if bo["breakout"] or bo["breakdown"] or bo["false_breakout"] or bo["false_breakdown"]:
            conviction += 10
            signals.append("breakout" if bo["breakout"] or bo["breakdown"] else "false_bo")
    except Exception:
        pass

    # Liquidity grab on 1H (+10)
    try:
        lq = detect_liquidity_grab(candles_1h)
        if lq["bullish"] or lq["bearish"]:
            conviction += 10
            signals.append(f"lq_grab")
    except Exception:
        pass

    # MACD crossover on 1H — direction-aware
    # Bullish crossover aligns with dip setups (rsi<50); bearish with bounce setups (rsi>50)
    # Opposing crossover = momentum headwind → penalise
    try:
        m1h = compute_macd(c)
        if m1h:
            bull_setup = rsi_val < 50   # likely long candidate
            if m1h["crossover_bullish"] and bull_setup:
                conviction += 10
                signals.append("macd1h_bull✓")
            elif m1h["crossover_bearish"] and not bull_setup:
                conviction += 10
                signals.append("macd1h_bear✓")
            elif m1h["crossover_bullish"] and not bull_setup:
                conviction -= 10
                signals.append("macd1h_bull✗")
            elif m1h["crossover_bearish"] and bull_setup:
                conviction -= 10
                signals.append("macd1h_bear✗")
    except Exception:
        pass

    # MACD crossover on 15m — faster signal, same direction-awareness (+15 aligned, -10 opposing)
    # 15m crossover is the entry-timing signal; 1H is the trend filter
    try:
        m15 = compute_macd(c15m) if len(c15m) >= 35 else None
        if m15:
            bull_setup = rsi_val < 50
            if m15["crossover_bullish"] and bull_setup:
                conviction += 15
                signals.append("macd15m_bull✓")
            elif m15["crossover_bearish"] and not bull_setup:
                conviction += 15
                signals.append("macd15m_bear✓")
            elif m15["crossover_bullish"] and not bull_setup:
                conviction -= 10
                signals.append("macd15m_bull✗")
            elif m15["crossover_bearish"] and bull_setup:
                conviction -= 10
                signals.append("macd15m_bear✗")
    except Exception:
        pass

    # Candlestick pattern confirmation (+10)
    try:
        cs = candlestick_patterns(candles_1h)
        if cs["bullish_signals"] or cs["bearish_signals"]:
            conviction += 10
            cs_names = cs["bullish_signals"] + cs["bearish_signals"]
            signals.append(f"cs({'|'.join(cs_names)})")
    except Exception:
        pass

    # BB squeeze → expansion on 1H (+20 — highest conviction timing signal)
    # Squeeze without expansion = skip (low conviction, no direction)
    try:
        bb = bb_squeeze(c)
        if bb["expanding"] and bb["expansion_dir"]:
            # Squeeze→expansion is the best breakout timing signal
            bonus = 20 if bb["was_squeezing"] else 10
            conviction += bonus
            signals.append(f"bb_squeeze→{bb['expansion_dir']}" if bb["was_squeezing"] else f"bb_expand_{bb['expansion_dir']}")
    except Exception:
        pass

    # VWAP alignment on 1H (+10 if price is aligned with direction bias)
    # VWAP band touch (+15 if at ±1σ or ±2σ — precision mean-reversion level)
    try:
        vw = compute_vwap(candles_1h)
        if vw:
            # General alignment: above/below VWAP = directional bias (+5)
            conviction += 5
            signals.append("↑VWAP" if vw["price_above"] else "↓VWAP")
            # Band touch: price at ±1σ or ±2σ = highest-precision entry level (+10 more)
            if vw.get("at_upper_1") or vw.get("at_upper_2") or vw.get("at_lower_1") or vw.get("at_lower_2"):
                conviction += 10
                band = ("+2σ" if vw.get("at_upper_2") else "+1σ" if vw.get("at_upper_1")
                        else "-2σ" if vw.get("at_lower_2") else "-1σ")
                signals.append(f"VWAP_band({band})")
    except Exception:
        pass

    # Chart patterns on 1H (+15 for high-confidence patterns like H&S, triangle)
    # Pairs with a pattern in play are higher priority — clear setup = clear entry
    try:
        cp = chart_patterns(candles_1h, lookback=50)
        all_patterns = cp["bullish_patterns"] + cp["bearish_patterns"]
        if all_patterns:
            # Weight by confidence of best pattern
            best = cp["best_bullish"] or cp["best_bearish"]
            bonus = int(15 * (best["confidence"] if best else 0.5))
            conviction += bonus
            signals.append(f"pattern({'|'.join(all_patterns[:2])})")
    except Exception:
        pass

    # ── Macro bonus: Total2 / Total3 layer ────────────────────────────────
    # These bonuses shift the ranking of the whole pair, not just a single signal.
    macro_bonus = 0
    macro_notes = []
    alt_season      = regime_info.get("alt_season",      False)
    btc_dom_rising  = regime_info.get("btc_dom_rising",  False)
    macro_sentiment = regime_info.get("macro_sentiment", "neutral")

    is_btc = pair.startswith("BTC/")
    is_eth = pair.startswith("ETH/")
    is_alt = not is_btc and not is_eth

    if is_alt and alt_season:
        # Alt season: money rotating into alts — boost all non-BTC/ETH pairs
        macro_bonus += 15
        macro_notes.append("alt_season(+15)")
    elif is_alt and btc_dom_rising:
        # BTC dominance rising: alts losing share to BTC — penalise alt pairs
        macro_bonus -= 15
        macro_notes.append("btc_dom(-15)")

    if is_btc and btc_dom_rising:
        # BTC taking market share — BTC setups get a boost
        macro_bonus += 10
        macro_notes.append("btc_dom(+10)")

    # Macro sentiment alignment: if macro is bull and RSI is oversold, high-conviction long
    if macro_sentiment == "bull" and rsi_val < 45:
        macro_bonus += 10
        macro_notes.append("macro_bull_dip(+10)")
    elif macro_sentiment == "bear" and rsi_val > 55:
        macro_bonus += 10
        macro_notes.append("macro_bear_bounce(+10)")

    # ── News bonus (CryptoPanic sentiment, optional) ──────────────────────
    news_bonus = 0
    news_notes = []
    if news:
        news_bonus = news.get("conviction_bonus", 0)
        label      = news.get("label", "no_data")
        posts      = news.get("post_count", 0)
        headlines  = news.get("headlines", [])
        if news_bonus != 0:
            news_notes.append(f"news_{label}({news_bonus:+d}, {posts} posts)")
            if headlines:
                # Print top headline for visibility in logs
                print(f"  [news] {pair}: {headlines[0][:80]}", flush=True)

    total = round(base_score + conviction + macro_bonus + news_bonus, 2)
    all_notes = signals + macro_notes + news_notes
    if all_notes:
        print(
            f"[scanner] {pair} base={base_score:.1f} "
            f"+{conviction}conv +{macro_bonus}macro +{news_bonus}news "
            f"({', '.join(all_notes)}) → {total}",
            flush=True,
        )
    return total


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
