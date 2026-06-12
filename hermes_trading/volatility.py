"""
Volatility regime engine — v17 (Total2 / Total3 macro layer).

Three-layer macro classification:
  Layer 1 — BTC:    vol + ADX → calm / normal / volatile / extreme / sideways
  Layer 2 — Total2: ETH as proxy (excl. BTC) → alt market bias
  Layer 3 — Total3: SOL/BNB/ADA basket (excl. BTC+ETH) → small-cap momentum

Composite signals:
  alt_season       = ETH bull  +  majority of Total3 alts bull
  btc_dom_rising   = BTC above MA while ETH below MA (BTC eating alt share)
  macro_sentiment  = "bull" | "bear" | "neutral" aggregate
"""
from __future__ import annotations
from hermes_trading.indicators import realized_vol, atr, adx as compute_adx, sma
from hermes_trading.adapters.candles import closes as get_closes, highs as get_highs, lows as get_lows

# Thresholds for 24-bar realized vol on 1H candles (≈ daily vol)
REGIME_THRESHOLDS = {
    "calm":     0.010,   # < 1%
    "normal":   0.030,   # 1–3%
    "volatile": 0.050,   # 3–5%
    # extreme:  > 5%
}

# Risk params per regime
REGIME_PARAMS: dict[str, dict] = {
    "calm": {
        "max_pairs":       5,
        "capital_per_pair": 200.0,
        "position_size_r":  0.05,
        "stop_loss_pct":    1.8,
        "take_profit_pct":  3.0,
        "leverage":         2.0,   # calm trend → 2x amplification
        "label":           "🟢 calm",
    },
    "normal": {
        "max_pairs":       4,
        "capital_per_pair": 175.0,
        "position_size_r":  0.04,
        "stop_loss_pct":    1.5,
        "take_profit_pct":  3.0,
        "leverage":         1.5,   # moderate trend → 1.5x
        "label":           "🟡 normal",
    },
    "volatile": {
        "max_pairs":       3,
        "capital_per_pair": 150.0,
        "position_size_r":  0.03,
        "stop_loss_pct":    1.2,
        "take_profit_pct":  2.5,
        "leverage":         1.0,   # high vol → no leverage
        "label":           "🟠 volatile",
    },
    "extreme": {
        "max_pairs":       2,
        "capital_per_pair": 100.0,
        "position_size_r":  0.02,
        "stop_loss_pct":    0.8,
        "take_profit_pct":  2.0,
        "leverage":         1.0,   # extreme → no leverage, capital preservation
        "label":           "🔴 extreme",
    },
    # Sideways / ranging: ADX < 20 regardless of absolute vol level
    # Mean-reversion at range extremes — tighter stops, shorter TP target
    "sideways": {
        "max_pairs":        5,
        "capital_per_pair": 175.0,
        "position_size_r":  0.04,
        "stop_loss_pct":    1.5,
        "take_profit_pct":  2.5,
        "leverage":         2.0,   # ranging market → 2x (mean-reversion high probability)
        "label":            "↔️  sideways",
    },
}

# ADX threshold below which market is classified as ranging / sideways
ADX_SIDEWAYS_THRESHOLD = 20.0


def classify(vol: float) -> str:
    if vol < REGIME_THRESHOLDS["calm"]:
        return "calm"
    elif vol < REGIME_THRESHOLDS["normal"]:
        return "normal"
    elif vol < REGIME_THRESHOLDS["volatile"]:
        return "volatile"
    else:
        return "extreme"


def params(regime: str) -> dict:
    return REGIME_PARAMS.get(regime, REGIME_PARAMS["normal"])


async def detect(
    btc_candles_1h:  list[dict],
    eth_candles_1h:  list[dict] | None = None,
    alt_candles_1h:  dict[str, list[dict]] | None = None,
) -> dict:
    """
    Detect volatility + macro regime.

    Parameters
    ----------
    btc_candles_1h  : BTC 1H candles — primary vol + ADX source (required)
    eth_candles_1h  : ETH 1H candles — Total2 proxy (excl. BTC)
    alt_candles_1h  : {symbol: candles} for Total3 basket, e.g.
                      {"SOL/USDT": [...], "BNB/USDT": [...], "ADA/USDT": [...]}

    Returns
    -------
    {
      # ── Layer 1: BTC-based vol regime ──────────────────────────────
      "regime":          str,    # "calm"|"normal"|"volatile"|"extreme"|"sideways"
      "vol_regime":      str,    # vol classification before ADX override
      "vol":             float,
      "adx":             float | None,
      "is_sideways":     bool,

      # ── Layer 2: Total2 — ETH proxy ────────────────────────────────
      "total2_bias":     str,    # "bull"|"bear"|"neutral"
      "total2_vol":      float,
      "eth_above_ma50":  bool | None,
      "eth_vs_btc":      str,    # "outperforming"|"underperforming"|"neutral"

      # ── Layer 3: Total3 — small-cap alt basket ─────────────────────
      "total3_bias":     str,    # "bull"|"bear"|"neutral"
      "alt_bulls":       int,    # number of Total3 alts in uptrend
      "alt_total":       int,    # total alts checked
      "alt_bull_pct":    float,  # fraction in uptrend

      # ── Composite signals ──────────────────────────────────────────
      "alt_season":       bool,  # ETH + majority alts bullish
      "btc_dom_rising":   bool,  # BTC in uptrend, ETH in downtrend
      "macro_sentiment":  str,   # "bull"|"bear"|"neutral"

      **risk_params              # max_pairs, capital_per_pair, …
    }
    """
    # ── Layer 1: BTC vol + ADX ─────────────────────────────────────────────
    c   = get_closes(btc_candles_1h)
    h   = get_highs(btc_candles_1h)
    l   = get_lows(btc_candles_1h)

    vol     = realized_vol(c, period=24)
    adx_val = compute_adx(h, l, c, period=14)

    if vol is None:
        vol_regime = "normal"
        vol        = 0.0
    else:
        vol_regime = classify(vol)

    is_sideways  = adx_val is not None and adx_val < ADX_SIDEWAYS_THRESHOLD
    final_regime = "sideways" if is_sideways else vol_regime
    p            = REGIME_PARAMS.get(final_regime, REGIME_PARAMS["normal"])

    # BTC 50MA position (used for dom check)
    btc_ma50      = sma(c, 50) if len(c) >= 50 else None
    btc_above_ma  = (btc_ma50 is not None and c[-1] > btc_ma50)

    # ── Layer 2: Total2 — ETH as alt-market proxy ──────────────────────────
    total2_bias   = "neutral"
    total2_vol    = 0.0
    eth_above_ma  = None
    eth_vs_btc    = "neutral"

    if eth_candles_1h and len(eth_candles_1h) >= 50:
        ec          = get_closes(eth_candles_1h)
        eth_vol_raw = realized_vol(ec, period=24)
        total2_vol  = eth_vol_raw or 0.0
        eth_ma50    = sma(ec, 50)
        if eth_ma50:
            eth_above_ma = ec[-1] > eth_ma50
            total2_bias  = "bull" if eth_above_ma else "bear"

        # ETH/BTC relative performance over last 24 bars
        # Compute % change of ETH vs % change of BTC — positive = ETH outperforming
        if len(c) >= 24 and len(ec) >= 24:
            btc_ret = (c[-1]  / c[-24]  - 1) if c[-24]  > 0 else 0
            eth_ret = (ec[-1] / ec[-24] - 1) if ec[-24] > 0 else 0
            rel     = eth_ret - btc_ret
            if rel > 0.005:       # ETH outperforming by >0.5%
                eth_vs_btc = "outperforming"
            elif rel < -0.005:
                eth_vs_btc = "underperforming"

    # ── Layer 3: Total3 — small-cap alt basket ─────────────────────────────
    total3_bias = "neutral"
    alt_bulls   = 0
    alt_total   = 0

    if alt_candles_1h:
        for sym, candles in alt_candles_1h.items():
            if len(candles) >= 50:
                ac      = get_closes(candles)
                alt_ma  = sma(ac, 50)
                if alt_ma:
                    alt_total += 1
                    if ac[-1] > alt_ma:
                        alt_bulls += 1

    alt_bull_pct = alt_bulls / alt_total if alt_total > 0 else 0.5
    if alt_total > 0:
        total3_bias = ("bull"    if alt_bull_pct >= 0.60
                       else "bear" if alt_bull_pct <= 0.35
                       else "neutral")

    # ── Composite signals ──────────────────────────────────────────────────
    # Alt season: ETH + majority of Total3 alts in uptrend → rotate into alts
    alt_season = (total2_bias == "bull") and (total3_bias in ("bull", "neutral"))

    # BTC dominance rising: BTC trending up while alts lagging/falling
    btc_dom_rising = btc_above_ma and (total2_bias == "bear" or eth_vs_btc == "underperforming")

    # Composite sentiment — majority vote across three layers
    sentiments = [
        "bull" if btc_above_ma else "bear",   # Layer 1
        total2_bias,                           # Layer 2
        total3_bias,                           # Layer 3
    ]
    bull_votes = sentiments.count("bull")
    bear_votes = sentiments.count("bear")
    if bull_votes >= 2:
        macro_sentiment = "bull"
    elif bear_votes >= 2:
        macro_sentiment = "bear"
    else:
        macro_sentiment = "neutral"

    adx_str = f"{adx_val:.1f}" if adx_val is not None else "n/a"
    alt_str = (f"T2={total2_bias} T3={total3_bias} "
               f"alts={alt_bulls}/{alt_total} eth_vs_btc={eth_vs_btc} "
               f"alt_season={'✅' if alt_season else '❌'} "
               f"btc_dom={'↑' if btc_dom_rising else '→'}")
    print(
        f"[volatility] regime={p['label']} vol={vol:.3%} adx={adx_str} "
        f"sideways={is_sideways} macro={macro_sentiment} | {alt_str}",
        flush=True,
    )
    return {
        # Layer 1 (BTC)
        "regime":         final_regime,
        "vol_regime":     vol_regime,
        "vol":            round(vol, 6),
        "adx":            adx_val,
        "is_sideways":    is_sideways,
        # Layer 2 (Total2 / ETH)
        "total2_bias":    total2_bias,
        "total2_vol":     round(total2_vol, 6),
        "eth_above_ma50": eth_above_ma,
        "eth_vs_btc":     eth_vs_btc,
        # Layer 3 (Total3 / alt basket)
        "total3_bias":    total3_bias,
        "alt_bulls":      alt_bulls,
        "alt_total":      alt_total,
        "alt_bull_pct":   round(alt_bull_pct, 3),
        # Composite
        "alt_season":      alt_season,
        "btc_dom_rising":  btc_dom_rising,
        "macro_sentiment": macro_sentiment,
        **p,
    }
