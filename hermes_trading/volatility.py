"""
Volatility regime engine.

Uses BTC/USDT 1H candles as the market-wide volatility proxy.
Classifies the market into one of four regimes and returns
the corresponding risk parameters Hermes should apply.
"""
from __future__ import annotations
from hermes_trading.indicators import realized_vol, atr, adx as compute_adx
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
        "label":           "🟢 calm",
    },
    "normal": {
        "max_pairs":       4,
        "capital_per_pair": 175.0,
        "position_size_r":  0.04,
        "stop_loss_pct":    1.5,
        "take_profit_pct":  3.0,
        "label":           "🟡 normal",
    },
    "volatile": {
        "max_pairs":       3,
        "capital_per_pair": 150.0,
        "position_size_r":  0.03,
        "stop_loss_pct":    1.2,
        "take_profit_pct":  2.5,
        "label":           "🟠 volatile",
    },
    "extreme": {
        "max_pairs":       2,
        "capital_per_pair": 100.0,
        "position_size_r":  0.02,
        "stop_loss_pct":    0.8,
        "take_profit_pct":  2.0,
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


async def detect(btc_candles_1h: list[dict]) -> dict:
    """
    Detect current volatility + trend-strength regime from BTC 1H candles.

    Returns:
      {
        "regime":      str,   # "calm" | "normal" | "volatile" | "extreme" | "sideways"
        "vol_regime":  str,   # underlying vol regime (unaffected by ADX)
        "vol":         float,
        "adx":         float | None,
        "is_sideways": bool,
        **risk_params         # max_pairs, capital_per_pair, position_size_r, stop/tp pct
      }
    """
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

    # Sideways: ADX below threshold → market is ranging regardless of vol level
    is_sideways = adx_val is not None and adx_val < ADX_SIDEWAYS_THRESHOLD

    final_regime = "sideways" if is_sideways else vol_regime
    p            = REGIME_PARAMS.get(final_regime, REGIME_PARAMS["normal"])

    adx_str = f"{adx_val:.1f}" if adx_val is not None else "n/a"
    print(
        f"[volatility] regime={p['label']} vol={vol:.3%} adx={adx_str} "
        f"sideways={is_sideways} max_pairs={p['max_pairs']} "
        f"pos_size={p['position_size_r']} stop={p['stop_loss_pct']}%",
        flush=True,
    )
    return {
        "regime":      final_regime,
        "vol_regime":  vol_regime,
        "vol":         round(vol, 6),
        "adx":         adx_val,
        "is_sideways": is_sideways,
        **p,
    }
