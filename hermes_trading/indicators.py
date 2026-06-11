"""
Pure indicator functions — no I/O, easy to test.

All functions operate on plain lists of floats (closes, highs, lows).
Returns None / empty when there's insufficient data rather than crashing.
"""
from __future__ import annotations
import math


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def rsi(closes: list[float], period: int = 14) -> float:
    """Standard Wilder RSI. Returns 50.0 when insufficient data."""
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [d for d in deltas[-period:] if d > 0]
    losses = [-d for d in deltas[-period:] if d < 0]
    avg_gain = sum(gains)  / period if gains  else 0.0
    avg_loss = sum(losses) / period if losses else 1e-9
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# EMA / MACD
# ---------------------------------------------------------------------------

def ema_series(values: list[float], period: int) -> list[float]:
    """Exponential moving average — returns same-length list (first `period-1` values seeded with SMA)."""
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    result = [sum(values[:period]) / period]
    for v in values[period:]:
        result.append(v * k + result[-1] * (1 - k))
    # Pad front with NaN-equivalents (we use None here, callers handle it)
    return [None] * (period - 1) + result


def _clean(series: list) -> list[float]:
    """Strip leading Nones."""
    return [v for v in series if v is not None]


def macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> dict | None:
    """
    Returns dict with keys:
      macd_line, signal_line, histogram,
      crossover_bullish, crossover_bearish  (True if crossover on the last bar)
    Returns None if insufficient data.
    """
    min_len = slow + signal_period + 1
    if len(closes) < min_len:
        return None

    fast_ema  = _clean(ema_series(closes, fast))
    slow_ema  = _clean(ema_series(closes, slow))

    # Align lengths (fast EMA is longer)
    n = min(len(fast_ema), len(slow_ema))
    macd_line_series = [fast_ema[-n + i] - slow_ema[-n + i] for i in range(n)]

    signal_series = _clean(ema_series(macd_line_series, signal_period))
    if len(signal_series) < 2:
        return None

    macd_val   = macd_line_series[-1]
    signal_val = signal_series[-1]
    hist_now   = macd_val - signal_val
    hist_prev  = macd_line_series[-2] - signal_series[-2]

    return {
        "macd_line":         round(macd_val, 6),
        "signal_line":       round(signal_val, 6),
        "histogram":         round(hist_now, 6),
        "crossover_bullish": hist_prev < 0 and hist_now >= 0,   # crossed above zero
        "crossover_bearish": hist_prev > 0 and hist_now <= 0,   # crossed below zero
        "histogram_rising":  hist_now > hist_prev,
        "histogram_falling": hist_now < hist_prev,
    }


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------

def sma(values: list[float], period: int) -> float | None:
    """Simple moving average. Returns None if not enough data."""
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def trend_vs_ma(price: float, closes: list[float], period: int = 50) -> str:
    """
    Returns 'uptrend', 'downtrend', or 'warming_up' based on price vs MA.
    """
    ma = sma(closes, period)
    if ma is None:
        return "warming_up"
    return "uptrend" if price > ma else "downtrend"


# ---------------------------------------------------------------------------
# RSI Divergence
# ---------------------------------------------------------------------------

def _local_minima(values: list[float], window: int = 3) -> list[int]:
    """Indices of local minima within `window` bars either side."""
    indices = []
    for i in range(window, len(values) - window):
        if values[i] == min(values[i - window: i + window + 1]):
            indices.append(i)
    return indices


def _local_maxima(values: list[float], window: int = 3) -> list[int]:
    indices = []
    for i in range(window, len(values) - window):
        if values[i] == max(values[i - window: i + window + 1]):
            indices.append(i)
    return indices


def rsi_divergence(closes: list[float], rsi_period: int = 14, lookback: int = 40, swing_window: int = 3) -> dict:
    """
    Detect RSI divergence over the last `lookback` bars.

    Returns:
      {
        "bullish": bool,   # price lower low, RSI higher low (hidden strength)
        "bearish": bool,   # price higher high, RSI lower high (hidden weakness)
      }
    """
    if len(closes) < lookback + rsi_period:
        return {"bullish": False, "bearish": False}

    window = closes[-lookback:]
    rsi_vals = [rsi(window[:i+1], rsi_period) for i in range(len(window))]

    # Bullish divergence: look at recent lows
    min_idx = _local_minima(window, swing_window)
    bullish = False
    if len(min_idx) >= 2:
        i1, i2 = min_idx[-2], min_idx[-1]
        if window[i2] < window[i1] and rsi_vals[i2] > rsi_vals[i1]:
            bullish = True  # price lower low, RSI higher low

    # Bearish divergence: look at recent highs
    max_idx = _local_maxima(window, swing_window)
    bearish = False
    if len(max_idx) >= 2:
        i1, i2 = max_idx[-2], max_idx[-1]
        if window[i2] > window[i1] and rsi_vals[i2] < rsi_vals[i1]:
            bearish = True  # price higher high, RSI lower high

    return {"bullish": bullish, "bearish": bearish}


# ---------------------------------------------------------------------------
# ATR (Average True Range) — used by volatility engine
# ---------------------------------------------------------------------------

def atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float | None:
    """Average True Range. Returns None if insufficient data."""
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
        trs.append(tr)
    return sum(trs[-period:]) / period


# ---------------------------------------------------------------------------
# Liquidity Grab (Stop Hunt / Wick Reversal)
# ---------------------------------------------------------------------------

def liquidity_grab(
    candles: list[dict],
    lookback: int = 20,
    wick_ratio: float = 2.0,
    sweep_pct: float = 0.002,
) -> dict:
    """
    Detect a liquidity grab (stop hunt) on the most recent candles.

    A bullish liquidity grab (sweep of lows):
      - A candle wicks below the recent swing low by at least `sweep_pct`
      - The lower wick is >= `wick_ratio` × the candle body
      - The candle closes back above the swept low (recovery)

    A bearish liquidity grab (sweep of highs):
      - Mirror of above on the upside

    Returns:
      {
        "bullish": bool,   # swept lows → likely long reversal
        "bearish": bool,   # swept highs → likely short reversal
        "wick_pct": float, # size of the sweep wick as % of price
      }
    """
    if len(candles) < lookback + 2:
        return {"bullish": False, "bearish": False, "wick_pct": 0.0}

    recent    = candles[-(lookback + 1):-1]   # lookback reference candles (exclude last)
    last      = candles[-1]                    # candle to evaluate

    swing_low  = min(c["low"]  for c in recent)
    swing_high = max(c["high"] for c in recent)

    o, h, l, c_close = last["open"], last["high"], last["low"], last["close"]
    body   = abs(c_close - o)
    lower_wick = min(o, c_close) - l   # wick below body
    upper_wick = h - max(o, c_close)   # wick above body

    bullish = False
    bearish = False
    wick_pct = 0.0

    # Bullish: wick swept below swing low, candle closes back above it
    if (l < swing_low * (1 - sweep_pct)          # actually swept the low
            and c_close > swing_low               # recovered above it
            and body > 0                          # not a doji
            and lower_wick >= wick_ratio * body): # wick dominates body
        bullish  = True
        wick_pct = (swing_low - l) / swing_low

    # Bearish: wick swept above swing high, candle closes back below it
    if (h > swing_high * (1 + sweep_pct)
            and c_close < swing_high
            and body > 0
            and upper_wick >= wick_ratio * body):
        bearish  = True
        wick_pct = (h - swing_high) / swing_high

    return {"bullish": bullish, "bearish": bearish, "wick_pct": round(wick_pct, 6)}


# ---------------------------------------------------------------------------
# ADX (Average Directional Index) — trend strength 0-100
# ---------------------------------------------------------------------------

def adx(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float | None:
    """
    Wilder's Average Directional Index.
      < 20  →  ranging / sideways
      20-25 →  weakly trending
      > 25  →  trending
    Returns None if insufficient data.
    """
    n = len(closes)
    if n < 2 * period + 1:
        return None

    tr_vals, plus_dm_vals, minus_dm_vals = [], [], []
    for i in range(1, n):
        tr = max(
            highs[i]   - lows[i],
            abs(highs[i]   - closes[i - 1]),
            abs(lows[i]    - closes[i - 1]),
        )
        up_move   = highs[i]    - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm   = up_move   if up_move   > down_move and up_move   > 0 else 0.0
        minus_dm  = down_move if down_move > up_move   and down_move > 0 else 0.0
        tr_vals.append(tr)
        plus_dm_vals.append(plus_dm)
        minus_dm_vals.append(minus_dm)

    def _wilder(data: list[float], p: int) -> list[float]:
        s = [sum(data[:p])]
        for x in data[p:]:
            s.append(s[-1] - s[-1] / p + x)
        return s

    str_  = _wilder(tr_vals,      period)
    sdm_p = _wilder(plus_dm_vals, period)
    sdm_m = _wilder(minus_dm_vals, period)

    dx_vals = []
    for i in range(len(str_)):
        if str_[i] == 0:
            continue
        pdi  = 100 * sdm_p[i] / str_[i]
        mdi  = 100 * sdm_m[i] / str_[i]
        dsum = pdi + mdi
        dx_vals.append(100 * abs(pdi - mdi) / dsum if dsum else 0.0)

    if len(dx_vals) < period:
        return None

    # ADX = Wilder's smoothed average of DX values
    adx_val = sum(dx_vals[:period]) / period
    for dx in dx_vals[period:]:
        adx_val = (adx_val * (period - 1) + dx) / period
    return round(adx_val, 2)


# ---------------------------------------------------------------------------
# Range / Sideways levels
# ---------------------------------------------------------------------------

def prev_day_levels(candles_1h: list[dict]) -> dict | None:
    """
    Previous day's high, low, open, close derived from 1H candles.
    Returns None if yesterday's candles are unavailable.
    """
    from datetime import datetime, timezone, timedelta
    if not candles_1h:
        return None

    now_utc   = datetime.now(timezone.utc)
    yesterday = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")

    yday = [
        c for c in candles_1h
        if datetime.fromtimestamp(c["ts"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d") == yesterday
    ]
    if not yday:
        return None

    yday.sort(key=lambda x: x["ts"])
    pdh = max(c["high"]  for c in yday)
    pdl = min(c["low"]   for c in yday)
    return {
        "pdh":      pdh,
        "pdl":      pdl,
        "pdo":      yday[0]["open"],
        "pdc":      yday[-1]["close"],
        "pd_range": round(pdh - pdl, 8),
    }


def opening_range(candles_15m: list[dict], bars: int = 4) -> dict | None:
    """
    Opening range: high/low of the first `bars` × 15-min candles of the current UTC day.
    Default: 4 bars = first 60 minutes of the UTC day.
    Returns None if today's candles are not yet available.
    """
    from datetime import datetime, timezone
    if not candles_15m:
        return None

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_c = sorted(
        [c for c in candles_15m
         if datetime.fromtimestamp(c["ts"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d") == today],
        key=lambda x: x["ts"],
    )
    if not today_c:
        return None

    or_c      = today_c[:bars]
    or_high   = max(c["high"] for c in or_c)
    or_low    = min(c["low"]  for c in or_c)
    return {
        "or_high":  or_high,
        "or_low":   or_low,
        "or_mid":   round((or_high + or_low) / 2, 8),
        "or_range": round(or_high - or_low, 8),
    }


def swing_levels(candles: list[dict], lookback: int = 20) -> dict:
    """Recent swing high and low over the last `lookback` candles."""
    recent = candles[-lookback:] if len(candles) > lookback else candles
    return {
        "swing_high": max(c["high"] for c in recent),
        "swing_low":  min(c["low"]  for c in recent),
    }


def classify_pair_regime(candles_4h: list[dict]) -> str:
    """
    Classify a pair's own trend regime from its 4H candles.
    Independent of the macro BTC regime — each pair gets its own classification.

    Returns:
      'bull'     — price above 50MA and ADX >= 20 (trending up)
      'bear'     — price below 50MA and ADX >= 20 (trending down)
      'sideways' — ADX < 20 (ranging regardless of MA position)
      'neutral'  — insufficient data
    """
    if len(candles_4h) < 30:
        return "neutral"

    c = [x["close"] for x in candles_4h]
    h = [x["high"]  for x in candles_4h]
    l = [x["low"]   for x in candles_4h]

    adx_val = adx(h, l, c, period=14)
    ma50    = sma(c, 50)
    price   = c[-1]

    if adx_val is not None and adx_val < 20:
        return "sideways"

    if ma50 is None:
        return "neutral"

    return "bull" if price > ma50 else "bear"


def range_position(price: float, range_high: float, range_low: float) -> float:
    """
    Where is `price` within [range_low, range_high]?
    0.0 = at the very bottom of the range.
    1.0 = at the very top.
    0.5 = mid-range.
    """
    if range_high <= range_low or range_low <= 0:
        return 0.5
    return max(0.0, min(1.0, (price - range_low) / (range_high - range_low)))


def realized_vol(closes: list[float], period: int = 24) -> float | None:
    """
    Realized volatility as std-dev of log returns over `period` bars.
    Returned as a fraction (e.g. 0.02 = 2%).
    """
    if len(closes) < period + 1:
        return None
    log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(len(closes) - period, len(closes))]
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
    return math.sqrt(variance)
