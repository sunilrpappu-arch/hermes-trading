"""
Backtesting engine — replays historical candles through the core trading pipeline.

Fetches N days of candles for a pair, steps through time windows one candle at a
time, applies the same entry/exit logic as loop.py, and returns simulated trades
with full performance metrics.

Used by self_improve.py to validate proposed parameter changes before applying them.
"""
from __future__ import annotations
import time
from hermes_trading.indicators import (
    rsi as compute_rsi,
    macd as compute_macd,
    sma,
    rsi_divergence,
    breakout_detector,
    candlestick_patterns,
    bb_squeeze,
    dynamic_levels,
    classify_pair_regime,
    range_position,
    swing_levels as compute_swing_levels,
)
from hermes_trading.adapters.candles import closes as get_closes, fetch as fetch_candles


# ---------------------------------------------------------------------------
# Core backtester
# ---------------------------------------------------------------------------

async def run_backtest(
    pair: str,
    strategy: dict,
    lookback_days: int = 30,
) -> dict:
    """
    Backtest a strategy config against `lookback_days` of 1H candles.

    Returns
    -------
    {
      trades       list   simulated closed trades
      win_rate     float  0–100
      total_pnl_pct float cumulative pnl_pct sum
      avg_rr       float  average realised R:R
      max_drawdown float  max peak-to-trough drawdown
      total_trades int
      wins         int
      losses       int
      pair         str
      days         int
    }
    """
    # Fetch enough candles: lookback_days × 24 + 100 warm-up bars
    limit = min(lookback_days * 24 + 120, 1000)
    try:
        candles_1h = await fetch_candles(pair, "1h", limit)
        candles_4h = await fetch_candles(pair, "4h", 200)
    except Exception as e:
        return _empty_result(pair, lookback_days, error=str(e))

    if len(candles_1h) < 80:
        return _empty_result(pair, lookback_days, error="insufficient candles")

    # Warm-up: need 60 bars before we start simulating
    WARMUP = 60
    trades       = []
    open_pos     = None   # {direction, entry_price, sl_price, tp_price, entry_i}
    peak_equity  = 0.0
    equity       = 0.0
    max_dd       = 0.0

    # Mirror live leverage so PnL magnitudes match (self_improve compares backtest vs live)
    lev_cfg   = strategy.get("leverage", {})
    _bt_lev   = float(lev_cfg.get("normal", 1.5))

    bull_cfg  = strategy.get("bull",  {})
    bear_cfg  = strategy.get("bear",  {})
    sw_cfg    = strategy.get("sideways", {})
    mtf_req   = int(strategy.get("mtf", {}).get("require_signals", 1))
    sw_entry  = sw_cfg.get("range_entry_pct", 0.20)

    for i in range(WARMUP, len(candles_1h)):
        window    = candles_1h[:i + 1]       # candles up to and including bar i
        candle    = candles_1h[i]
        c_high    = candle["high"]
        c_low     = candle["low"]
        c_close   = candle["close"]
        closes    = get_closes(window)
        current   = c_close

        # ── Manage open position ──────────────────────────────────────────
        if open_pos:
            direction = open_pos["direction"]
            sl        = open_pos["sl_price"]
            tp        = open_pos["tp_price"]
            entry     = open_pos["entry_price"]
            hit_sl    = (direction == "long"  and c_low  <= sl) or \
                        (direction == "short" and c_high >= sl)
            hit_tp    = (direction == "long"  and c_high >= tp) or \
                        (direction == "short" and c_low  <= tp)

            if hit_sl or hit_tp:
                exit_price   = sl if hit_sl else tp
                mult         = 1 if direction == "long" else -1
                pnl_pct      = (exit_price - entry) / entry * mult * _bt_lev
                close_reason = "stop_loss" if hit_sl else "take_profit"
                sl_dist      = abs(entry - sl)
                tp_dist      = abs(tp   - entry)
                actual_rr    = abs(exit_price - entry) / sl_dist if sl_dist > 0 else 0

                trades.append({
                    "direction":    direction,
                    "entry_price":  entry,
                    "exit_price":   exit_price,
                    "pnl_pct":      round(pnl_pct, 6),
                    "close_reason": close_reason,
                    "actual_rr":    round(actual_rr, 3),
                    "planned_rr":   open_pos.get("planned_rr", 0),
                    "pair_regime":  open_pos.get("pair_regime", "?"),
                    "signal":       open_pos.get("signal", "?"),
                    "bar_i":        i,
                })
                equity += pnl_pct
                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / (1 + peak_equity) if peak_equity > 0 else abs(min(equity, 0))
                max_dd = max(max_dd, dd)
                open_pos = None
            else:
                continue   # still in position — don't look for new entries

        # ── Entry logic (simplified faithful replica of loop.py) ──────────
        if len(closes) < 55:
            continue

        rsi_val    = compute_rsi(closes)
        ma50       = sma(closes, 50)
        if ma50 is None:
            continue

        # Pair regime from 4H context (use portion of 4H candles)
        # Approximate: use ratio to pick 4H window proportional to current 1H position
        frac       = i / len(candles_1h)
        c4h_end    = max(10, int(len(candles_4h) * frac))
        c4h_window = candles_4h[:c4h_end]
        pair_regime = classify_pair_regime(c4h_window) if len(c4h_window) >= 20 else "neutral"

        # Breakout on recent 1H window
        bo = breakout_detector(window[-25:]) if len(window) >= 25 else \
             {"breakout": False, "breakdown": False, "false_breakout": False, "false_breakdown": False}

        # RSI divergence on 1H
        rsi_div = rsi_divergence(closes) if len(closes) >= 54 else {"bullish": False, "bearish": False}

        # BB squeeze
        bb = bb_squeeze(closes) if len(closes) >= 70 else \
             {"squeeze": False, "expanding": False, "expansion_dir": None}

        # Candlestick patterns
        cs = candlestick_patterns(window[-5:]) if len(window) >= 5 else \
             {"bullish_signals": [], "bearish_signals": [], "bull_marubozu": False, "bear_marubozu": False}

        cs_bull = bool(cs["bullish_signals"]) and not cs["bear_marubozu"]
        cs_bear = bool(cs["bearish_signals"]) and not cs["bull_marubozu"]

        # Swing levels for range position
        sw_lvls  = compute_swing_levels(window[-30:], 20)
        rng_high = sw_lvls["swing_high"] if sw_lvls else None
        rng_low  = sw_lvls["swing_low"]  if sw_lvls else None
        rng_pos  = range_position(current, rng_high, rng_low) if (rng_high and rng_low) else None

        # MTF signal count (simplified)
        htf_count = 0
        m1h = compute_macd(closes)
        if m1h and m1h["crossover_bullish"]:
            htf_count += 1
        if current > ma50:
            htf_count += 1
        if rsi_div["bullish"]:
            htf_count += 1

        new_direction = None
        signal        = ""

        if pair_regime == "bull":
            thr = bull_cfg.get("long_threshold", 35)
            if rsi_val < thr and current > ma50 and cs_bull:
                new_direction, signal = "long", f"bull_rsi_dip({rsi_val:.0f})"
            elif bo["breakout"] and not cs["bear_marubozu"]:
                new_direction, signal = "long", "bull_breakout"
            elif not bo["breakout"]:
                if bo["false_breakout"] and cs_bear:
                    new_direction, signal = "short", "false_breakout"
                elif rsi_div["bearish"] and cs_bear:
                    new_direction, signal = "short", "bear_rsi_div"

        elif pair_regime == "bear":
            thr = bear_cfg.get("short_threshold", 60)
            if rsi_val > thr and current < ma50 and cs_bear:
                new_direction, signal = "short", f"bear_rsi_bounce({rsi_val:.0f})"
            elif bo["breakdown"] and not cs["bull_marubozu"]:
                new_direction, signal = "short", "bear_breakdown"
            elif not bo["breakdown"]:
                if bo["false_breakdown"] and cs_bull:
                    new_direction, signal = "long", "false_breakdown"
                elif rsi_div["bullish"] and cs_bull:
                    new_direction, signal = "long", "bull_rsi_div"

        elif pair_regime == "sideways" and rng_pos is not None:
            if rng_pos <= sw_entry and cs_bull:
                new_direction, signal = "long",  f"range_bottom({rng_pos:.0%})"
            elif rng_pos >= (1.0 - sw_entry) and cs_bear:
                new_direction, signal = "short", f"range_top({rng_pos:.0%})"

        # BB direction gate
        if new_direction and bb["squeeze"] and not bb["expanding"]:
            new_direction = None  # wait for expansion
        if new_direction and bb["expanding"] and bb["expansion_dir"]:
            if (new_direction == "long"  and bb["expansion_dir"] == "down") or \
               (new_direction == "short" and bb["expansion_dir"] == "up"):
                new_direction = None

        # MTF gate
        if new_direction and mtf_req > 0 and htf_count < mtf_req:
            new_direction = None

        if not new_direction:
            continue

        # Dynamic levels
        try:
            lvls = dynamic_levels(
                new_direction, current,
                window[-40:], window[-60:],
                rng_high, rng_low,
                min_rr=1.0,
            )
            if not lvls["valid"]:
                continue
        except Exception:
            continue

        open_pos = {
            "direction":   new_direction,
            "entry_price": current,
            "sl_price":    lvls["sl_price"],
            "tp_price":    lvls["tp_price"],
            "planned_rr":  lvls["rr_ratio"],
            "pair_regime": pair_regime,
            "signal":      signal,
            "entry_i":     i,
        }

    # Close any open position at last bar
    if open_pos:
        last  = candles_1h[-1]["close"]
        entry = open_pos["entry_price"]
        mult  = 1 if open_pos["direction"] == "long" else -1
        pnl   = (last - entry) / entry * mult
        trades.append({
            **open_pos,
            "exit_price":   last,
            "pnl_pct":      round(pnl, 6),
            "close_reason": "end_of_data",
            "actual_rr":    0,
        })

    return _metrics(trades, pair, lookback_days, max_dd)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metrics(trades: list[dict], pair: str, days: int, max_dd: float) -> dict:
    if not trades:
        return _empty_result(pair, days)
    wins   = [t for t in trades if t.get("pnl_pct", 0) > 0]
    losses = [t for t in trades if t.get("pnl_pct", 0) <= 0]
    wr     = len(wins) / len(trades) * 100 if trades else 0
    total_pnl = sum(t.get("pnl_pct", 0) for t in trades)
    rr_vals   = [t.get("actual_rr", 0) for t in trades if t.get("actual_rr", 0) > 0]
    avg_rr    = sum(rr_vals) / len(rr_vals) if rr_vals else 0
    return {
        "pair":          pair,
        "days":          days,
        "total_trades":  len(trades),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      round(wr, 1),
        "total_pnl_pct": round(total_pnl * 100, 3),
        "avg_rr":        round(avg_rr, 3),
        "max_drawdown":  round(max_dd * 100, 3),
        "trades":        trades,
        "error":         None,
    }


def _empty_result(pair: str, days: int, error: str = None) -> dict:
    return {
        "pair": pair, "days": days, "total_trades": 0, "wins": 0, "losses": 0,
        "win_rate": 0.0, "total_pnl_pct": 0.0, "avg_rr": 0.0,
        "max_drawdown": 0.0, "trades": [], "error": error,
    }
