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
    """Simple-average RSI (not Wilder's EMA-smoothed variant — internally consistent
    across live and backtest, but values differ from charting tools by ~3-8pts in trends).
    Returns 50.0 when insufficient data."""
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


def vwap(candles: list[dict], reset_daily: bool = True) -> dict | None:
    """
    Volume Weighted Average Price — resets at UTC midnight by default (intraday VWAP).

    Uses typical price = (high + low + close) / 3, volume-weighted.
    Bands are ±1σ and ±2σ of the volume-weighted standard deviation.

    Returns:
      {
        "vwap":          float   — current VWAP
        "upper_1":       float   — VWAP + 1σ  (overbought zone)
        "lower_1":       float   — VWAP - 1σ  (oversold zone)
        "upper_2":       float   — VWAP + 2σ  (extreme overbought)
        "lower_2":       float   — VWAP - 2σ  (extreme oversold)
        "std":           float
        "price_above":   bool    — price > VWAP (bullish intraday bias)
        "pct_from_vwap": float   — % deviation (+ above, - below)
        "at_upper_1":    bool    — price near +1σ band (potential short)
        "at_lower_1":    bool    — price near -1σ band (potential long)
        "at_upper_2":    bool    — price near +2σ band (extreme short)
        "at_lower_2":    bool    — price near -2σ band (extreme long)
        "candles_used":  int
      }
    Returns None if no data or zero volume.
    """
    from datetime import datetime, timezone

    if not candles:
        return None

    # Intraday: use only today's UTC candles; fall back to last 20 if none yet
    if reset_daily:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        data  = [
            c for c in candles
            if datetime.fromtimestamp(c["ts"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d") == today
        ] or candles[-20:]
    else:
        data = candles

    if not data:
        return None

    cum_vol  = 0.0
    cum_pv   = 0.0
    cum_pv2  = 0.0  # for variance: E[P²] - E[P]²

    for c in data:
        tp  = (c["high"] + c["low"] + c["close"]) / 3.0
        vol = float(c.get("volume") or 0)
        cum_vol  += vol
        cum_pv   += tp * vol
        cum_pv2  += tp * tp * vol

    if cum_vol == 0:
        return None

    vwap_val = cum_pv / cum_vol
    variance = max(0.0, (cum_pv2 / cum_vol) - vwap_val ** 2)
    std      = variance ** 0.5

    price     = data[-1]["close"]
    dev_pct   = (price - vwap_val) / vwap_val * 100 if vwap_val > 0 else 0.0
    band_tol  = 0.003   # within 0.3% of band = "at" the band

    def _near(level: float) -> bool:
        return vwap_val > 0 and abs(price - level) / vwap_val <= band_tol

    return {
        "vwap":          round(vwap_val,          8),
        "upper_1":       round(vwap_val + std,     8),
        "lower_1":       round(vwap_val - std,     8),
        "upper_2":       round(vwap_val + 2 * std, 8),
        "lower_2":       round(vwap_val - 2 * std, 8),
        "std":           round(std,                8),
        "price_above":   price > vwap_val,
        "pct_from_vwap": round(dev_pct, 4),
        "at_upper_1":    _near(vwap_val + std),
        "at_lower_1":    _near(vwap_val - std),
        "at_upper_2":    _near(vwap_val + 2 * std),
        "at_lower_2":    _near(vwap_val - 2 * std),
        "candles_used":  len(data),
    }


def bollinger_bands(
    closes: list[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> dict | None:
    """
    Bollinger Bands: middle SMA ± N standard deviations.
    Returns None if insufficient data.
    """
    if len(closes) < period:
        return None
    recent   = closes[-period:]
    middle   = sum(recent) / period
    variance = sum((x - middle) ** 2 for x in recent) / period
    std      = variance ** 0.5
    upper    = middle + std_dev * std
    lower    = middle - std_dev * std
    bandwidth = (upper - lower) / middle if middle > 0 else 0.0
    price    = closes[-1]
    return {
        "upper":          round(upper, 8),
        "middle":         round(middle, 8),
        "lower":          round(lower, 8),
        "bandwidth":      round(bandwidth, 6),
        "std":            round(std, 8),
        "pct_b":          round((price - lower) / (upper - lower), 4) if upper > lower else 0.5,
        # %B: 0 = at lower band, 1 = at upper band, >1 above, <0 below
    }


def bb_squeeze(
    closes: list[float],
    period: int = 20,
    std_dev: float = 2.0,
    squeeze_pct: float = 0.25,
    lookback: int = 50,
) -> dict:
    """
    Detect Bollinger Band squeeze and expansion.

    Squeeze: bands are historically tight (low-vol consolidation).
             A big directional move is building — don't enter yet.
    Expanding: bandwidth widening after a squeeze — breakout is starting NOW.
               This is the highest-conviction entry timing signal.

    squeeze_pct: bandwidth below this percentile of recent history = squeeze (default 25th).

    Returns:
      {
        "bb":             dict (upper/middle/lower/bandwidth/pct_b) | None
        "squeeze":        bool  — currently in low-vol consolidation
        "expanding":      bool  — bandwidth growing (breakout starting)
        "was_squeezing":  bool  — was in squeeze recently (expansion after squeeze = best signal)
        "expansion_dir":  "up" | "down" | None
        "price_above_mid": bool
        "at_upper_band":  bool  — price ≥ upper BB (overbought / breakout)
        "at_lower_band":  bool  — price ≤ lower BB (oversold / breakdown)
      }
    """
    empty = {
        "bb": None, "squeeze": False, "expanding": False,
        "was_squeezing": False, "expansion_dir": None,
        "price_above_mid": False, "at_upper_band": False, "at_lower_band": False,
    }
    min_len = period + lookback
    if len(closes) < min_len:
        return empty

    # Rolling bandwidth history over lookback window
    bw_history = []
    for i in range(lookback, 0, -1):
        window = closes[-(period + i):len(closes) - i]
        if len(window) < period:
            continue
        mid = sum(window) / period
        if mid <= 0:
            continue
        std = (sum((x - mid) ** 2 for x in window) / period) ** 0.5
        bw_history.append((2 * std_dev * std) / mid)

    if not bw_history:
        return empty

    current_bb = bollinger_bands(closes, period, std_dev)
    if not current_bb:
        return empty

    current_bw    = current_bb["bandwidth"]
    current_price = closes[-1]
    sorted_bw     = sorted(bw_history)
    thresh_idx    = max(0, int(len(sorted_bw) * squeeze_pct) - 1)
    squeeze_thresh = sorted_bw[thresh_idx]

    squeeze   = current_bw <= squeeze_thresh
    above_mid = current_price > current_bb["middle"]

    # Expansion: bandwidth has been increasing for the last 3 readings
    expanding = False
    if len(bw_history) >= 3:
        expanding = all(bw_history[-i] > bw_history[-(i + 1)]
                        for i in range(1, 3))

    # Was squeezing: any of the last 5 readings were in squeeze territory
    was_squeezing = any(bw <= squeeze_thresh for bw in bw_history[-5:])

    # Expansion direction: use %B and price vs bands
    expansion_dir = None
    if expanding:
        pct_b = current_bb["pct_b"]
        if pct_b >= 0.8 or current_price >= current_bb["upper"]:
            expansion_dir = "up"
        elif pct_b <= 0.2 or current_price <= current_bb["lower"]:
            expansion_dir = "down"
        else:
            expansion_dir = "up" if above_mid else "down"

    return {
        "bb":              current_bb,
        "squeeze":         squeeze,
        "expanding":       expanding,
        "was_squeezing":   was_squeezing,
        "expansion_dir":   expansion_dir,
        "price_above_mid": above_mid,
        "at_upper_band":   current_price >= current_bb["upper"],
        "at_lower_band":   current_price <= current_bb["lower"],
    }


def candlestick_patterns(candles: list[dict]) -> dict:
    """
    Detect key candlestick patterns on the most recent 1-2 candles.

    Patterns detected:
      hammer          — small body near top, lower wick ≥ 2× body (bullish reversal)
      shooting_star   — small body near bottom, upper wick ≥ 2× body (bearish reversal)
      bullish_engulf  — current bullish body fully engulfs prior bearish body (bullish momentum)
      bearish_engulf  — current bearish body fully engulfs prior bullish body (bearish momentum)
      bull_marubozu   — large bullish body, wicks < 15% of body (strong bull momentum)
      bear_marubozu   — large bearish body, wicks < 15% of body (strong bear momentum)
      doji            — body < 10% of candle range (indecision — confirms reversal at extremes)

    Returns dict with bool for each pattern, plus summary lists:
      "bullish_signals": list of bullish pattern names found
      "bearish_signals": list of bearish pattern names found
    """
    result = {
        "hammer":         False,
        "shooting_star":  False,
        "bullish_engulf": False,
        "bearish_engulf": False,
        "bull_marubozu":  False,
        "bear_marubozu":  False,
        "doji":           False,
        "bullish_signals": [],
        "bearish_signals": [],
    }
    if len(candles) < 2:
        return result

    cur  = candles[-1]
    prev = candles[-2]

    c_open  = cur["open"]
    c_close = cur["close"]
    c_high  = cur["high"]
    c_low   = cur["low"]
    c_range = c_high - c_low

    if c_range < 1e-12:
        return result

    c_body        = abs(c_close - c_open)
    c_body_top    = max(c_open, c_close)
    c_body_bot    = min(c_open, c_close)
    c_upper_wick  = c_high - c_body_top
    c_lower_wick  = c_body_bot - c_low
    c_is_bull     = c_close > c_open
    c_is_bear     = c_close < c_open

    p_open  = prev["open"]
    p_close = prev["close"]
    p_body_top = max(p_open, p_close)
    p_body_bot = min(p_open, p_close)
    p_is_bull  = p_close > p_open
    p_is_bear  = p_close < p_open

    # --- Doji: body is tiny relative to candle range ---
    if c_body < 0.10 * c_range:
        result["doji"] = True
        # Doji is neutral — counted as both signals at extremes
        result["bullish_signals"].append("doji")
        result["bearish_signals"].append("doji")

    # --- Hammer: small body at top of candle, big lower wick ---
    # Lower wick ≥ 2× body, upper wick ≤ 30% of body, body in upper 40% of range
    if (c_body > 0
            and c_lower_wick >= 2.0 * c_body
            and c_upper_wick <= 0.3 * c_body
            and c_body_bot >= c_low + 0.55 * c_range):
        result["hammer"] = True
        result["bullish_signals"].append("hammer")

    # --- Shooting star: small body at bottom of candle, big upper wick ---
    if (c_body > 0
            and c_upper_wick >= 2.0 * c_body
            and c_lower_wick <= 0.3 * c_body
            and c_body_top <= c_low + 0.45 * c_range):
        result["shooting_star"] = True
        result["bearish_signals"].append("shooting_star")

    # --- Bullish engulfing: bull candle body fully covers prior bear body ---
    if (c_is_bull and p_is_bear
            and c_body_bot <= p_body_bot
            and c_body_top >= p_body_top
            and c_body > 0 and p_body_top > p_body_bot):
        result["bullish_engulf"] = True
        result["bullish_signals"].append("bull_engulf")

    # --- Bearish engulfing: bear candle body fully covers prior bull body ---
    if (c_is_bear and p_is_bull
            and c_body_bot <= p_body_bot
            and c_body_top >= p_body_top
            and c_body > 0 and p_body_top > p_body_bot):
        result["bearish_engulf"] = True
        result["bearish_signals"].append("bear_engulf")

    # --- Bull Marubozu: big bullish body, wicks < 15% of body each ---
    if (c_is_bull
            and c_body >= 0.7 * c_range           # body dominates the candle
            and c_upper_wick <= 0.15 * c_body
            and c_lower_wick <= 0.15 * c_body):
        result["bull_marubozu"] = True
        result["bullish_signals"].append("bull_marubozu")

    # --- Bear Marubozu: big bearish body, wicks < 15% of body each ---
    if (c_is_bear
            and c_body >= 0.7 * c_range
            and c_upper_wick <= 0.15 * c_body
            and c_lower_wick <= 0.15 * c_body):
        result["bear_marubozu"] = True
        result["bearish_signals"].append("bear_marubozu")

    return result


def _candle_swing_highs(candles: list[dict], window: int = 3) -> list[tuple[int, float]]:
    """Swing highs: index + value pairs where candle high is local maximum."""
    highs = [c["high"] for c in candles]
    result = []
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i - window: i + window + 1]):
            result.append((i, highs[i]))
    return result


def _candle_swing_lows(candles: list[dict], window: int = 3) -> list[tuple[int, float]]:
    """Swing lows: index + value pairs where candle low is local minimum."""
    lows = [c["low"] for c in candles]
    result = []
    for i in range(window, len(lows) - window):
        if lows[i] == min(lows[i - window: i + window + 1]):
            result.append((i, lows[i]))
    return result


def _linreg_slope(values: list[float]) -> float:
    """Normalised linear regression slope (slope / mean). Returns 0 if insufficient data."""
    n = len(values)
    if n < 2:
        return 0.0
    y_mean = sum(values) / n
    if y_mean == 0:
        return 0.0
    x_mean = (n - 1) / 2.0
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return (num / den) / y_mean if den > 0 else 0.0


def chart_patterns(candles: list[dict], lookback: int = 50, swing_window: int = 3) -> dict:
    """
    Detect classic chart patterns on the provided candles.
    Works on any timeframe — use 4H for trend patterns, 1H for entry-level patterns.

    Patterns detected:
      Reversal:     double_top, double_bottom, head_shoulders, inv_head_shoulders
      Continuation: bull_flag, bear_flag
      Triangle:     ascending_triangle, descending_triangle, symmetric_triangle
      Channel:      ascending_channel, descending_channel

    Returns:
      {
        "patterns":         list of pattern dicts
        "bullish_patterns": list of bullish pattern names
        "bearish_patterns": list of bearish pattern names
        "best_bullish":     highest-confidence bullish pattern | None
        "best_bearish":     highest-confidence bearish pattern | None
      }

    Each pattern dict:
      { "name", "bias", "key_level", "confidence", "description" }
    """
    empty = {
        "patterns": [], "bullish_patterns": [], "bearish_patterns": [],
        "best_bullish": None, "best_bearish": None,
    }
    data = candles[-lookback:] if len(candles) > lookback else candles
    if len(data) < 20:
        return empty

    highs  = [c["high"]  for c in data]
    lows   = [c["low"]   for c in data]
    closes = [c["close"] for c in data]

    swing_h = _candle_swing_highs(data, swing_window)
    swing_l = _candle_swing_lows(data, swing_window)
    current = closes[-1]
    tol     = 0.025   # 2.5% tolerance for level comparisons
    detected: list[dict] = []

    # ── Double Top ─────────────────────────────────────────────────────────
    # Two peaks at similar level, valley between them, price near/below neckline
    if len(swing_h) >= 2:
        (h1_i, h1), (h2_i, h2) = swing_h[-2], swing_h[-1]
        if h2_i > h1_i and abs(h1 - h2) / max(h1, 1e-9) <= tol:
            valley   = min(lows[h1_i: h2_i + 1]) if h1_i < h2_i else h1
            depth    = (max(h1, h2) - valley) / max(h1, 1e-9)
            if depth > 0.02:
                conf = 0.75 if current <= valley * 1.01 else 0.40
                detected.append({
                    "name": "double_top", "bias": "bearish",
                    "key_level": round(valley, 8), "confidence": conf,
                    "description": f"Double top peaks≈{(h1+h2)/2:.4g} neckline={valley:.4g}",
                })

    # ── Double Bottom ───────────────────────────────────────────────────────
    if len(swing_l) >= 2:
        (l1_i, l1), (l2_i, l2) = swing_l[-2], swing_l[-1]
        if l2_i > l1_i and abs(l1 - l2) / max(l1, 1e-9) <= tol:
            peak     = max(highs[l1_i: l2_i + 1]) if l1_i < l2_i else l1
            depth    = (peak - min(l1, l2)) / max(peak, 1e-9)
            if depth > 0.02:
                conf = 0.75 if current >= peak * 0.99 else 0.40
                detected.append({
                    "name": "double_bottom", "bias": "bullish",
                    "key_level": round(peak, 8), "confidence": conf,
                    "description": f"Double bottom troughs≈{(l1+l2)/2:.4g} neckline={peak:.4g}",
                })

    # ── Triple Top ──────────────────────────────────────────────────────────
    if len(swing_h) >= 3:
        (h1_i, h1), (h2_i, h2), (h3_i, h3) = swing_h[-3], swing_h[-2], swing_h[-1]
        if h1_i < h2_i < h3_i:
            avg_peak = (h1 + h2 + h3) / 3
            if all(abs(h - avg_peak) / max(avg_peak, 1e-9) <= tol for h in (h1, h2, h3)):
                valley = min(lows[h1_i: h3_i + 1])
                conf   = 0.80 if current <= valley * 1.01 else 0.45
                detected.append({
                    "name": "triple_top", "bias": "bearish",
                    "key_level": round(valley, 8), "confidence": conf,
                    "description": f"Triple top peaks≈{avg_peak:.4g} neckline={valley:.4g}",
                })

    # ── Triple Bottom ────────────────────────────────────────────────────────
    if len(swing_l) >= 3:
        (l1_i, l1), (l2_i, l2), (l3_i, l3) = swing_l[-3], swing_l[-2], swing_l[-1]
        if l1_i < l2_i < l3_i:
            avg_trough = (l1 + l2 + l3) / 3
            if all(abs(l - avg_trough) / max(avg_trough, 1e-9) <= tol for l in (l1, l2, l3)):
                peak = max(highs[l1_i: l3_i + 1])
                conf = 0.80 if current >= peak * 0.99 else 0.45
                detected.append({
                    "name": "triple_bottom", "bias": "bullish",
                    "key_level": round(peak, 8), "confidence": conf,
                    "description": f"Triple bottom troughs≈{avg_trough:.4g} neckline={peak:.4g}",
                })

    # ── Head and Shoulders ──────────────────────────────────────────────────
    if len(swing_h) >= 3:
        (ls_i, ls), (hd_i, hd), (rs_i, rs) = swing_h[-3], swing_h[-2], swing_h[-1]
        if ls_i < hd_i < rs_i and hd > ls and hd > rs:
            if abs(ls - rs) / max(ls, 1e-9) <= 0.20:
                lt = min(lows[ls_i: hd_i + 1]) if ls_i < hd_i else ls
                rt = min(lows[hd_i: rs_i + 1]) if hd_i < rs_i else rs
                neckline = (lt + rt) / 2
                conf = 0.75 if current <= neckline * 1.01 else 0.45
                detected.append({
                    "name": "head_shoulders", "bias": "bearish",
                    "key_level": round(neckline, 8), "confidence": conf,
                    "description": f"H&S head={hd:.4g} shoulders={ls:.4g}/{rs:.4g} neck={neckline:.4g}",
                })

    # ── Inverse Head and Shoulders ──────────────────────────────────────────
    if len(swing_l) >= 3:
        (ls_i, ls), (hd_i, hd), (rs_i, rs) = swing_l[-3], swing_l[-2], swing_l[-1]
        if ls_i < hd_i < rs_i and hd < ls and hd < rs:
            if abs(ls - rs) / max(ls, 1e-9) <= 0.20:
                lp = max(highs[ls_i: hd_i + 1]) if ls_i < hd_i else ls
                rp = max(highs[hd_i: rs_i + 1]) if hd_i < rs_i else rs
                neckline = (lp + rp) / 2
                conf = 0.75 if current >= neckline * 0.99 else 0.45
                detected.append({
                    "name": "inv_head_shoulders", "bias": "bullish",
                    "key_level": round(neckline, 8), "confidence": conf,
                    "description": f"Inv H&S head={hd:.4g} shoulders={ls:.4g}/{rs:.4g} neck={neckline:.4g}",
                })

    # ── Triangles ─────────────────────────────────────────────────────────
    if len(swing_h) >= 3 and len(swing_l) >= 3:
        slope_h = _linreg_slope([v for _, v in swing_h[-3:]])
        slope_l = _linreg_slope([v for _, v in swing_l[-3:]])
        res     = sum(v for _, v in swing_h[-3:]) / 3
        sup     = sum(v for _, v in swing_l[-3:]) / 3
        apex    = (res + sup) / 2

        # Ascending: flat top + rising bottom → bullish breakout
        if abs(slope_h) < 0.005 and slope_l > 0.005:
            conf = min(0.80, 0.50 + slope_l * 8)
            detected.append({
                "name": "ascending_triangle", "bias": "bullish",
                "key_level": round(res, 8), "confidence": conf,
                "description": f"Ascending triangle resistance={res:.4g} (breakout target)",
            })
        # Descending: flat bottom + falling top → bearish breakdown
        elif abs(slope_l) < 0.005 and slope_h < -0.005:
            conf = min(0.80, 0.50 + abs(slope_h) * 8)
            detected.append({
                "name": "descending_triangle", "bias": "bearish",
                "key_level": round(sup, 8), "confidence": conf,
                "description": f"Descending triangle support={sup:.4g} (breakdown target)",
            })
        # Symmetric: converging → neutral, watch breakout direction
        elif slope_h < -0.003 and slope_l > 0.003:
            detected.append({
                "name": "symmetric_triangle", "bias": "neutral",
                "key_level": round(apex, 8), "confidence": 0.50,
                "description": f"Symmetric triangle apex≈{apex:.4g} — watch breakout direction",
            })

    # ── Channels ──────────────────────────────────────────────────────────
    if len(swing_h) >= 2 and len(swing_l) >= 2:
        slope_h  = _linreg_slope([v for _, v in swing_h[-2:]])
        slope_l  = _linreg_slope([v for _, v in swing_l[-2:]])
        ch_top   = max(v for _, v in swing_h[-2:])
        ch_bot   = min(v for _, v in swing_l[-2:])
        parallel = (slope_h != 0 and slope_l != 0 and
                    abs(slope_h - slope_l) / max(abs(slope_h), abs(slope_l), 1e-9) < 0.60)
        if parallel:
            if slope_h > 0.003 and slope_l > 0.003:
                detected.append({
                    "name": "ascending_channel", "bias": "bullish",
                    "key_level": round(ch_bot, 8), "confidence": 0.55,
                    "description": f"Ascending channel top={ch_top:.4g} bot={ch_bot:.4g} — long at bot",
                })
            elif slope_h < -0.003 and slope_l < -0.003:
                detected.append({
                    "name": "descending_channel", "bias": "bearish",
                    "key_level": round(ch_top, 8), "confidence": 0.55,
                    "description": f"Descending channel top={ch_top:.4g} bot={ch_bot:.4g} — short at top",
                })

    # ── Bull Flag ──────────────────────────────────────────────────────────
    # Strong upward pole in first half, tight pullback consolidation in second half
    n = len(data)
    if n >= 20:
        mid        = n // 2
        pole_c     = closes[:mid]
        flag_c     = closes[mid:]
        pole_move  = (pole_c[-1] - pole_c[0]) / max(pole_c[0], 1e-9)
        flag_slope = _linreg_slope(flag_c)
        flag_range = max(flag_c) - min(flag_c)
        pole_range = max(pole_c) - min(pole_c)

        if (pole_move > 0.05
                and flag_slope < 0
                and pole_range > 0
                and flag_range < pole_range * 0.50):
            conf = min(0.80, 0.50 + pole_move)
            detected.append({
                "name": "bull_flag", "bias": "bullish",
                "key_level": round(max(closes[-5:]), 8), "confidence": conf,
                "description": f"Bull flag pole={pole_move:.1%} — breakout above flag",
            })
        # Bear Flag
        elif (pole_move < -0.05
                and flag_slope > 0
                and pole_range > 0
                and flag_range < pole_range * 0.50):
            conf = min(0.80, 0.50 + abs(pole_move))
            detected.append({
                "name": "bear_flag", "bias": "bearish",
                "key_level": round(min(closes[-5:]), 8), "confidence": conf,
                "description": f"Bear flag pole={pole_move:.1%} — breakdown below flag",
            })

    # ── Cup and Handle ──────────────────────────────────────────────────────
    # U-shaped base (rounded bottom) + small handle pullback → bullish breakout
    if n >= 30:
        cup      = closes[:int(n * 0.75)]
        handle   = closes[int(n * 0.75):]
        cup_top  = max(cup[0], cup[-1])
        cup_bot  = min(cup)
        cup_mid  = len(cup) // 2
        cup_depth = (cup_top - cup_bot) / max(cup_top, 1e-9)
        # Cup: price recovers back toward the rim
        cup_recovery = (cup[-1] - cup_bot) / max(cup_top - cup_bot, 1e-9)
        # Handle: small pullback (< 50% of cup depth)
        handle_pullback = (max(handle) - min(handle)) / max(cup_top, 1e-9)
        if (cup_depth > 0.10
                and cup_recovery > 0.70
                and handle_pullback < cup_depth * 0.50
                and _linreg_slope(handle) < 0):   # handle drifts slightly down
            rim = cup_top
            conf = min(0.75, 0.50 + cup_depth)
            detected.append({
                "name": "cup_and_handle", "bias": "bullish",
                "key_level": round(rim, 8), "confidence": conf,
                "description": f"Cup & handle rim={rim:.4g} depth={cup_depth:.1%} — breakout above rim",
            })

    # ── Summarise ──────────────────────────────────────────────────────────
    bull = [p for p in detected if p["bias"] == "bullish"]
    bear = [p for p in detected if p["bias"] == "bearish"]
    if detected:
        print(f"[patterns] {', '.join(p['name'] for p in detected)}", flush=True)
    return {
        "patterns":         detected,
        "bullish_patterns": [p["name"] for p in bull],
        "bearish_patterns": [p["name"] for p in bear],
        "best_bullish":     max(bull, key=lambda p: p["confidence"]) if bull else None,
        "best_bearish":     max(bear, key=lambda p: p["confidence"]) if bear else None,
    }


def dynamic_levels(
    direction: str,
    entry_price: float,
    candles_15m: list[dict],
    candles_1h: list[dict],
    rng_high: float = None,
    rng_low: float = None,
    min_rr: float = 1.0,
    max_sl_pct: float = 0.04,   # never risk more than 4% on a single trade
    min_sl_pct: float = 0.003,  # never place SL inside 0.3% noise band
    sl_buffer_pct: float = 0.003,  # buffer just beyond the structural level
) -> dict:
    """
    Compute dynamic stop-loss and take-profit levels from price structure.

    Priority order
    ──────────────
    Stop-loss:
      1. Nearest structural level on the losing side (swing high/low, range extreme)
      2. ATR-based fallback  (entry ± 1.5 × ATR_15m)

    Take-profit:
      1. Nearest structural level on the winning side that satisfies min_rr
      2. Fibonacci extension from risk:  1.0×, 1.618×, 2.0×, 2.618×, 3.0×

    Returns
    ───────
    {
      sl_price  float   absolute SL level
      tp_price  float   absolute TP level
      sl_pct    float   risk %  (e.g. 0.018 = 1.8%)
      tp_pct    float   reward % (e.g. 0.036 = 3.6%)
      rr_ratio  float   reward / risk  (1.0 = 1:1)
      sl_method str     where the SL was placed
      tp_method str     where the TP was placed
      valid     bool    True if rr_ratio >= min_rr
    }
    """
    is_long = direction == "long"

    # ── Collect structural levels ─────────────────────────────────────────
    # 15m swing levels (short-term structure)
    sh_15m = _candle_swing_highs(candles_15m[-40:], window=3) if len(candles_15m) >= 10 else []
    sl_15m = _candle_swing_lows (candles_15m[-40:], window=3) if len(candles_15m) >= 10 else []

    # 1H swing levels (medium-term structure)
    sh_1h  = _candle_swing_highs(candles_1h[-60:],  window=3) if len(candles_1h) >= 10 else []
    sl_1h  = _candle_swing_lows (candles_1h[-60:],  window=3) if len(candles_1h) >= 10 else []

    # All swing highs / lows as flat lists of prices
    all_highs = sorted({v for _, v in sh_15m + sh_1h}, reverse=True)
    all_lows  = sorted({v for _, v in sl_15m + sl_1h})

    # ATR for noise-floor estimate
    h15 = [c["high"]  for c in candles_15m] if candles_15m else []
    l15 = [c["low"]   for c in candles_15m] if candles_15m else []
    c15 = [c["close"] for c in candles_15m] if candles_15m else []
    atr_val = atr(h15, l15, c15, period=14) or (entry_price * 0.008)

    # ── Stop-loss placement ───────────────────────────────────────────────
    sl_price  = None
    sl_method = "atr_fallback"

    if is_long:
        # SL below entry: highest structural low that is still below entry.
        # Buffer is relative to the structural level itself (not entry_price).
        candidates = [p - p * sl_buffer_pct
                      for p in all_lows if p < entry_price * (1 - min_sl_pct)]
        if rng_low and rng_low < entry_price * (1 - min_sl_pct):
            candidates.append(rng_low - rng_low * sl_buffer_pct)
        # Pick closest (highest) candidate below entry
        valid_sl = [p for p in candidates if p < entry_price]
        if valid_sl:
            sl_price       = max(valid_sl)
            _range_sl_low  = (rng_low  - entry_price * sl_buffer_pct) if rng_low  else None
            sl_method      = "range_low"  if (_range_sl_low  and abs(sl_price - _range_sl_low)  < entry_price * 0.0005) else "swing_low"
    else:
        # SL above entry: lowest structural high that is still above entry.
        # Buffer is relative to the structural level itself (not entry_price).
        candidates = [p + p * sl_buffer_pct
                      for p in all_highs if p > entry_price * (1 + min_sl_pct)]
        if rng_high and rng_high > entry_price * (1 + min_sl_pct):
            candidates.append(rng_high + rng_high * sl_buffer_pct)
        valid_sl = [p for p in candidates if p > entry_price]
        if valid_sl:
            sl_price        = min(valid_sl)
            _range_sl_high  = (rng_high + entry_price * sl_buffer_pct) if rng_high else None
            sl_method       = "range_high" if (_range_sl_high and abs(sl_price - _range_sl_high) < entry_price * 0.0005) else "swing_high"

    # ATR fallback if no structural SL found
    if sl_price is None:
        sl_price  = entry_price - 1.5 * atr_val if is_long else entry_price + 1.5 * atr_val
        sl_method = "atr_1.5x"

    # Clamp SL to [min_sl_pct, max_sl_pct] range
    sl_dist = abs(entry_price - sl_price)
    if sl_dist < entry_price * min_sl_pct:
        sl_dist  = entry_price * min_sl_pct
        sl_price = entry_price - sl_dist if is_long else entry_price + sl_dist
        sl_method += "+min_clamp"
    if sl_dist > entry_price * max_sl_pct:
        sl_dist  = entry_price * max_sl_pct
        sl_price = entry_price - sl_dist if is_long else entry_price + sl_dist
        sl_method += "+max_clamp"

    sl_pct = sl_dist / entry_price

    # ── Take-profit placement ─────────────────────────────────────────────
    tp_price  = None
    tp_method = "fib_1.618"

    # Fibonacci extensions from risk (always available as fallback)
    _fib_levels = [
        (1.0,   "fib_1:1"),
        (1.618, "fib_1.618"),
        (2.0,   "fib_2:1"),
        (2.618, "fib_2.618"),
        (3.0,   "fib_3:1"),
    ]
    fib_tps = []
    for mult, label in _fib_levels:
        fib_tp = (entry_price + sl_dist * mult) if is_long else (entry_price - sl_dist * mult)
        if fib_tp > 0:
            fib_tps.append((fib_tp, label))

    if is_long:
        # Structural TPs: swing highs and range high above entry, satisfying min_rr
        struct_tps = [p for p in all_highs if p > entry_price]
        if rng_high and rng_high > entry_price:
            struct_tps.append(rng_high)
        # Nearest structural level that achieves min_rr
        for tp in sorted(struct_tps):
            if (tp - entry_price) / sl_dist >= min_rr:
                tp_price  = tp
                tp_method = "range_high" if (rng_high and abs(tp - rng_high) < entry_price * 0.001) else "swing_high"
                break
    else:
        struct_tps = [p for p in all_lows if p < entry_price]
        if rng_low and rng_low < entry_price:
            struct_tps.append(rng_low)
        for tp in sorted(struct_tps, reverse=True):
            if (entry_price - tp) / sl_dist >= min_rr:
                tp_price  = tp
                tp_method = "range_low" if (rng_low and abs(tp - rng_low) < entry_price * 0.001) else "swing_low"
                break

    # Fibonacci fallback: pick first fib level that satisfies min_rr
    if tp_price is None:
        for fib_tp, label in fib_tps:
            fib_dist = abs(fib_tp - entry_price)
            if fib_dist / sl_dist >= min_rr:
                tp_price  = fib_tp
                tp_method = label
                break

    # Final fallback: 1:1 from SL
    if tp_price is None:
        tp_price  = entry_price + sl_dist if is_long else entry_price - sl_dist
        tp_method = "fib_1:1_fallback"

    tp_dist  = abs(tp_price - entry_price)
    tp_pct   = tp_dist / entry_price
    rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0.0

    return {
        "sl_price":  round(sl_price, 8),
        "tp_price":  round(tp_price, 8),
        "sl_pct":    round(sl_pct * 100, 4),    # e.g. 1.8 (not 0.018)
        "tp_pct":    round(tp_pct * 100, 4),    # e.g. 3.6
        "rr_ratio":  round(rr_ratio, 3),
        "sl_method": sl_method,
        "tp_method": tp_method,
        "valid":     rr_ratio >= min_rr,
    }


def breakout_detector(
    candles: list[dict],
    lookback: int = 20,
    confirm_pct: float = 0.001,
) -> dict:
    """
    Detect price breakouts above resistance or breakdowns below support.

    Uses the highest high / lowest low of the prior `lookback` candles
    (excluding the most recent candle being evaluated).

    'breakout'        — candle closes above prior resistance (bullish momentum)
    'breakdown'       — candle closes below prior support   (bearish momentum)
    'false_breakout'  — wick above resistance, closed back below with bearish body
                        → failed breakout = short signal
    'false_breakdown' — wick below support, closed back above with bullish body
                        → failed breakdown = long signal
    """
    empty = {
        "breakout": False, "breakdown": False,
        "false_breakout": False, "false_breakdown": False,
        "resistance": 0.0, "support": 0.0,
    }
    if len(candles) < lookback + 2:
        return empty

    prior      = candles[-(lookback + 1):-1]
    last       = candles[-1]
    resistance = max(c["high"] for c in prior)
    support    = min(c["low"]  for c in prior)

    close = last["close"]
    high  = last["high"]
    low   = last["low"]
    open_ = last["open"]

    breakout  = close > resistance * (1 + confirm_pct)
    breakdown = close < support    * (1 - confirm_pct)

    # False breakout: spiked above resistance but closed back below with a bearish candle
    false_breakout = (
        high  > resistance * (1 + confirm_pct)
        and close < resistance
        and close < open_
    )

    # False breakdown: spiked below support but closed back above with a bullish candle
    false_breakdown = (
        low   < support * (1 - confirm_pct)
        and close > support
        and close > open_
    )

    return {
        "breakout":        breakout,
        "breakdown":       breakdown,
        "false_breakout":  false_breakout,
        "false_breakdown": false_breakdown,
        "resistance":      round(resistance, 8),
        "support":         round(support, 8),
    }


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
