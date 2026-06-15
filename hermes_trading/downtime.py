"""
Downtime engine — productive work during quiet markets.

Triggered when no trades have closed in >8 hours. Runs three passes:

  1. Idle diagnosis   — explains per-pair why entries are being blocked
  2. OOS backtest     — validates current strategy on held-out recent data
                        (train window: days 31–60, test window: days 1–30)
  3. Shadow trading   — logs what the engine *would* have entered and tracks
                        whether those shadow trades hit TP/SL forward in time

Results are written to state/downtime_log.jsonl and state/shadow_trades.jsonl.
A summary is sent via the reflection notification channel.
"""
from __future__ import annotations
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

STATE_DIR         = Path(__file__).parent.parent / "state"
DOWNTIME_LOG      = STATE_DIR / "downtime_log.jsonl"
SHADOW_TRADES     = STATE_DIR / "shadow_trades.jsonl"
LAST_DOWNTIME_FILE = STATE_DIR / ".last_downtime_run"

# How many hours of trade silence before downtime engine fires
IDLE_THRESHOLD_HOURS = 8

# OOS split: total lookback, and how many days are held out for testing
OOS_TOTAL_DAYS  = 60
OOS_TEST_DAYS   = 30   # most recent 30 days = test; older 30 days = train basis


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run_downtime(
    all_trades:   list[dict],
    strategy:     dict,
    active_pairs: list[str],
    heartbeats:   dict,          # {asset: heartbeat_dict}
) -> str:
    """
    Run all three downtime passes. Returns a human-readable summary string.
    Writes detailed results to state files.
    Respects a cooldown — won't fire more than once per IDLE_THRESHOLD_HOURS.
    """
    if not _cooldown_expired():
        return ""

    _mark_run()

    lines = [f"🌙 <b>Downtime Analysis</b> — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"]

    # 1. Idle diagnosis
    diag = _idle_diagnosis(heartbeats, strategy)
    lines.append("\n📋 <b>Entry blockers</b>")
    for pair, reasons in diag.items():
        lines.append(f"  {pair}: {' · '.join(reasons)}")

    # 2. OOS backtest
    oos_result = await _oos_backtest(active_pairs, strategy)
    if oos_result:
        lines.append("\n🧪 <b>Out-of-sample validation</b>")
        lines.append(
            f"  Train WR {oos_result['train_wr']:.0f}% → Test WR {oos_result['test_wr']:.0f}%  "
            f"({'✅ holding up' if oos_result['test_wr'] >= oos_result['train_wr'] - 10 else '⚠️ degrading'})"
        )
        lines.append(
            f"  Train PnL {oos_result['train_pnl']:+.1f}%  Test PnL {oos_result['test_pnl']:+.1f}%  "
            f"({oos_result['test_trades']} OOS trades)"
        )
        if oos_result["overfit_warning"]:
            lines.append("  ⚠️ Possible overfitting — test WR is >15pp below train WR")

    # 3. Feature cohort analysis
    cohort_summary = _feature_cohort_analysis(all_trades)
    if cohort_summary:
        lines.append("\n🏷️ <b>Feature cohorts</b> (trades since each change)")
        for label, stats in cohort_summary.items():
            wr   = stats["win_rate"]
            n    = stats["count"]
            pnl  = stats["total_pnl"]
            tick = "✅" if wr >= 55 else "⚠️" if wr >= 40 else "❌"
            lines.append(f"  {tick} {label}: {n} trades · WR {wr:.0f}% · PnL {pnl:+.2f}%")

    # 4. R:R bleed analysis
    rr_summary = _rr_bleed_analysis(all_trades)
    if rr_summary:
        lines.append("\n📐 <b>R:R bleed check</b>")
        for bucket, stats in rr_summary["buckets"].items():
            wr   = stats["win_rate"]
            n    = stats["count"]
            pnl  = stats["total_pnl"]
            tick = "✅" if wr >= 55 else "⚠️" if wr >= 40 else "❌"
            lines.append(f"  {tick} R:R {bucket}: {n} trades · WR {wr:.0f}% · PnL {pnl:+.2f}%")
        if rr_summary.get("pattern_vs_structural"):
            pv = rr_summary["pattern_vs_structural"]
            lines.append(
                f"  Pattern TP: WR {pv['pattern_wr']:.0f}% ({pv['pattern_n']} trades) "
                f"vs Structural TP: WR {pv['structural_wr']:.0f}% ({pv['structural_n']} trades)"
            )
            if pv["pattern_wr"] < pv["structural_wr"] - 10 and pv["pattern_n"] >= 5:
                lines.append("  ⚠️ Pattern TPs underperforming — relaxed R:R may be bleeding WR")
            elif pv["pattern_wr"] > pv["structural_wr"] + 5 and pv["pattern_n"] >= 5:
                lines.append("  ✅ Pattern TPs outperforming structural levels")

    # 5. Shadow trade evaluation
    shadow_summary = _evaluate_shadow_trades(all_trades)
    if shadow_summary:
        lines.append("\n👻 <b>Shadow trade outcomes</b>")
        lines.append(
            f"  {shadow_summary['resolved']} resolved — "
            f"shadow WR {shadow_summary['shadow_wr']:.0f}%  "
            f"vs live WR {shadow_summary['live_wr']:.0f}%"
        )
        if shadow_summary["gap"] > 15:
            lines.append("  ⚠️ Shadow WR >> live WR — system may be filtering good setups")
        elif shadow_summary["gap"] < -15:
            lines.append("  ✅ Live filters outperforming shadows — gates are adding value")

    # 6. Trailing SL analysis
    trail_summary = _trailing_sl_analysis(all_trades)
    if trail_summary:
        lines.append("\n🎯 <b>Trailing SL analysis</b>")
        lines.append(
            f"  Trailed: {trail_summary['trailed_n']} trades · "
            f"WR {trail_summary['trailed_wr']:.0f}% · PnL {trail_summary['trailed_pnl']:+.2f}%"
        )
        lines.append(
            f"  Non-trailed: {trail_summary['normal_n']} trades · "
            f"WR {trail_summary['normal_wr']:.0f}% · PnL {trail_summary['normal_pnl']:+.2f}%"
        )
        if trail_summary["stopped_at_breakeven"] > 0:
            lines.append(f"  ⚠️ {trail_summary['stopped_at_breakeven']} trades stopped at breakeven (trail may be too tight)")
        if trail_summary["trailed_n"] >= 5:
            if trail_summary["trailed_pnl"] > trail_summary["normal_pnl"] + 1:
                lines.append("  ✅ Trailing SL locking in more profit than static SL")
            elif trail_summary["trailed_pnl"] < trail_summary["normal_pnl"] - 1:
                lines.append("  ⚠️ Trailing SL stopping out too early — consider loosening trail_pct")

    summary = "\n".join(lines)

    # Persist full results
    _log_downtime({
        "timestamp":    time.time(),
        "diagnosis":    diag,
        "oos":          oos_result,
        "shadow":       shadow_summary,
        "pairs":        active_pairs,
        "trades_n":     len(all_trades),
    })

    # Write strategy notes for dashboard modal
    _write_strategy_notes(diag, oos_result, rr_summary, shadow_summary)

    return summary


def _write_strategy_notes(diag, oos_result, rr_summary, shadow_summary):
    notes = []
    ts    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Idle blockers
    all_blockers = [r for reasons in (diag or {}).values() for r in reasons]
    if all_blockers:
        top = all_blockers[:3]
        notes.append({"icon": "🚫", "text": "Top blockers: " + " · ".join(top), "ts": ts})

    # OOS validation
    if oos_result:
        train_wr = oos_result.get("train_wr", 0)
        test_wr  = oos_result.get("test_wr",  0)
        gap      = train_wr - test_wr
        if gap > 15:
            notes.append({"icon": "⚠️", "text": f"OOS gap {gap:.0f}pp (train {train_wr:.0f}% vs test {test_wr:.0f}%) — possible overfitting", "ts": ts})
        else:
            notes.append({"icon": "✅", "text": f"OOS healthy — train {train_wr:.0f}% vs test {test_wr:.0f}%", "ts": ts})

    # R:R bleed
    if rr_summary and rr_summary.get("pattern_vs_structural"):
        pv = rr_summary["pattern_vs_structural"]
        if pv["pattern_n"] >= 5:
            if pv["pattern_wr"] < pv["structural_wr"] - 10:
                notes.append({"icon": "⚠️", "text": f"Pattern TPs underperforming structural ({pv['pattern_wr']:.0f}% vs {pv['structural_wr']:.0f}%) — relaxed R:R may be bleeding WR", "ts": ts})
            elif pv["pattern_wr"] > pv["structural_wr"] + 5:
                notes.append({"icon": "✅", "text": f"Pattern TPs outperforming structural ({pv['pattern_wr']:.0f}% vs {pv['structural_wr']:.0f}%)", "ts": ts})

    # Shadow trades
    if shadow_summary and shadow_summary.get("shadow_n", 0) >= 5:
        s_wr = shadow_summary.get("shadow_wr", 0)
        l_wr = shadow_summary.get("live_wr",   0)
        if s_wr > l_wr + 10:
            notes.append({"icon": "⚠️", "text": f"Shadow WR {s_wr:.0f}% >> live WR {l_wr:.0f}% — filters may be blocking good setups", "ts": ts})
        else:
            notes.append({"icon": "✅", "text": f"Live filters beating shadows ({l_wr:.0f}% vs {s_wr:.0f}%)", "ts": ts})

    if not notes:
        return

    notes_file = STATE_DIR / "strategy_notes.json"
    existing   = []
    if notes_file.exists():
        try:
            existing = json.loads(notes_file.read_text())
        except Exception:
            existing = []

    # Prepend new notes, keep last 20
    combined = notes + [n for n in existing if n not in notes]
    notes_file.write_text(json.dumps(combined[:20], indent=2))


# ---------------------------------------------------------------------------
# 1. Idle diagnosis
# ---------------------------------------------------------------------------

def _idle_diagnosis(heartbeats: dict, strategy: dict) -> dict[str, list[str]]:
    """
    For each active pair, explain why no entry fired on the last heartbeat.
    Returns {pair: [reason, ...]}
    """
    bull_thr = strategy.get("bull",  {}).get("long_threshold",  35)
    bear_thr = strategy.get("bear",  {}).get("short_threshold", 60)
    fg_min   = strategy.get("session_breakout", {}).get("min_fg_score", 18)
    fg_max   = strategy.get("session_breakout", {}).get("max_fg_score", 85)

    result = {}
    for asset, hb in heartbeats.items():
        reasons = []
        rsi     = hb.get("rsi_15m")
        regime  = hb.get("pair_regime", "neutral")
        bb_sq   = hb.get("bb_squeeze", False)
        bb_exp  = hb.get("bb_expanding", False)
        trend   = hb.get("trend")
        fg      = hb.get("fear_greed_score")

        if rsi is not None:
            if regime == "bull" and rsi >= bull_thr:
                reasons.append(f"RSI {rsi:.0f} ≥ {bull_thr} (need dip)")
            elif regime == "bear" and rsi <= bear_thr:
                reasons.append(f"RSI {rsi:.0f} ≤ {bear_thr} (need bounce)")
            elif regime in ("neutral", "sideways"):
                reasons.append(f"regime={regime}")

        if bb_sq and not bb_exp:
            reasons.append("BB squeezing, no expansion yet")

        if trend and trend not in ("uptrend", "downtrend"):
            reasons.append(f"trend={trend}")

        if fg is not None:
            if fg < fg_min:
                reasons.append(f"fear/greed {fg:.0f} (extreme fear gate)")
            elif fg > fg_max:
                reasons.append(f"fear/greed {fg:.0f} (extreme greed gate)")

        if not reasons:
            reasons.append("no clear setup on last tick")

        result[asset] = reasons

    return result


# ---------------------------------------------------------------------------
# 2. Out-of-sample backtest
# ---------------------------------------------------------------------------

async def _oos_backtest(pairs: list[str], strategy: dict) -> dict | None:
    """
    Backtest the current strategy on two non-overlapping windows:
      - Train: days 31–60 (older, used historically for hypothesis generation)
      - Test:  days 1–30  (recent, never used for training — true OOS)

    Returns aggregated metrics for both windows.
    """
    try:
        from hermes_trading.backtest import run_backtest
    except Exception:
        return None

    bt_pairs = pairs[:4] or ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    train_tasks = [run_backtest(p, strategy, lookback_days=OOS_TOTAL_DAYS) for p in bt_pairs]
    test_tasks  = [run_backtest(p, strategy, lookback_days=OOS_TEST_DAYS)  for p in bt_pairs]

    all_results = await asyncio.gather(*train_tasks, *test_tasks, return_exceptions=True)
    n = len(bt_pairs)
    train_raw = all_results[:n]
    test_raw  = all_results[n:]

    def _agg(results):
        valid = [r for r in results if isinstance(r, dict) and r.get("total_trades", 0) >= 1]
        if not valid:
            return None
        total = sum(r["total_trades"] for r in valid)
        wins  = sum(r["wins"]         for r in valid)
        pnl   = sum(r["total_pnl_pct"] * r["total_trades"] for r in valid) / total
        return {
            "total_trades": total,
            "win_rate":     round(wins / total * 100, 1) if total else 0,
            "total_pnl_pct": round(pnl, 2),
        }

    train = _agg(train_raw)
    test  = _agg(test_raw)
    if not train or not test:
        return None

    return {
        "train_wr":        train["win_rate"],
        "train_pnl":       train["total_pnl_pct"],
        "train_trades":    train["total_trades"],
        "test_wr":         test["win_rate"],
        "test_pnl":        test["total_pnl_pct"],
        "test_trades":     test["total_trades"],
        "overfit_warning": test["win_rate"] < train["win_rate"] - 15,
    }


# ---------------------------------------------------------------------------
# 3. Shadow trading
# ---------------------------------------------------------------------------

def record_shadow_trade(asset: str, direction: str, entry_price: float,
                        sl_price: float, tp_price: float, signal: str):
    """
    Log a would-have-entered trade. Called from loop.py when an entry
    signal fires but is blocked by a gate (MTF, BB, fear/greed, etc.).
    """
    record = {
        "asset":       asset,
        "direction":   direction,
        "entry_price": entry_price,
        "sl_price":    sl_price,
        "tp_price":    tp_price,
        "signal":      signal,
        "entered_at":  time.time(),
        "resolved":    False,
        "outcome":     None,
    }
    _append_shadow(record)


def resolve_shadow_trades(current_prices: dict[str, float]):
    """
    Check open shadow trades against current prices and mark as TP/SL/open.
    Called each tick from loop.py with {asset: current_price}.
    """
    if not SHADOW_TRADES.exists():
        return

    try:
        lines  = [l for l in SHADOW_TRADES.read_text().splitlines() if l.strip()]
        trades = [json.loads(l) for l in lines]
    except Exception:
        return

    updated = False
    for t in trades:
        if t.get("resolved"):
            continue
        asset = t["asset"]
        price = current_prices.get(asset)
        if price is None:
            continue

        d  = t["direction"]
        sl = t["sl_price"]
        tp = t["tp_price"]

        hit_tp = (d == "long"  and price >= tp) or (d == "short" and price <= tp)
        hit_sl = (d == "long"  and price <= sl) or (d == "short" and price >= sl)

        # Expire after 48 hours if neither hit
        age_h = (time.time() - t["entered_at"]) / 3600
        if hit_tp:
            t["outcome"], t["resolved"] = "tp", True
        elif hit_sl:
            t["outcome"], t["resolved"] = "sl", True
        elif age_h > 48:
            t["outcome"], t["resolved"] = "expired", True

        if t.get("resolved"):
            t["resolved_at"] = time.time()
            updated = True

    if updated:
        SHADOW_TRADES.write_text("\n".join(json.dumps(t) for t in trades) + "\n")


def _feature_cohort_analysis(all_trades: list[dict]) -> dict | None:
    """
    Group trades by active features and compute WR/PnL per cohort.
    Only includes features that appear in at least 5 trades.
    """
    if not all_trades:
        return None

    cohorts: dict[str, list] = {}
    for t in all_trades:
        for feature in t.get("active_features", []):
            cohorts.setdefault(feature, []).append(t)

    result = {}
    for feature, trades in cohorts.items():
        if len(trades) < 5:
            continue
        wins  = sum(1 for t in trades if (t.get("pnl_pct") or 0) > 0)
        pnl   = sum((t.get("pnl_pct") or 0) * 100 for t in trades)
        result[feature] = {
            "count":      len(trades),
            "win_rate":   wins / len(trades) * 100,
            "total_pnl":  round(pnl, 2),
        }

    return result or None


def _rr_bleed_analysis(all_trades: list[dict]) -> dict | None:
    """
    Check whether lower R:R trades (e.g. pattern-relaxed 0.75×) are dragging
    down overall win rate vs trades that achieved full structural R:R.

    Returns:
      {
        buckets: { "<0.75": {...}, "0.75-1.0": {...}, "1.0-1.5": {...}, "≥1.5": {...} }
        pattern_vs_structural: { pattern_wr, pattern_n, structural_wr, structural_n }
      }
    """
    closed = [t for t in all_trades if t.get("rr_ratio") is not None and t.get("pnl_pct") is not None]
    if len(closed) < 5:
        return None

    def _bucket(rr):
        if rr < 0.75:   return "<0.75"
        if rr < 1.0:    return "0.75–1.0"
        if rr < 1.5:    return "1.0–1.5"
        return "≥1.5"

    buckets: dict = {}
    for t in closed:
        b    = _bucket(t["rr_ratio"])
        win  = (t["pnl_pct"] or 0) > 0
        pnl  = (t["pnl_pct"] or 0) * 100
        buckets.setdefault(b, {"count": 0, "wins": 0, "total_pnl": 0.0})
        buckets[b]["count"]     += 1
        buckets[b]["wins"]      += int(win)
        buckets[b]["total_pnl"] += pnl

    for b, s in buckets.items():
        s["win_rate"]   = s["wins"] / s["count"] * 100 if s["count"] else 0
        s["total_pnl"]  = round(s["total_pnl"], 2)

    # Pattern TP vs structural/fib TP
    pat_trades  = [t for t in closed if (t.get("tp_method") or "").startswith("pattern_")]
    str_trades  = [t for t in closed if not (t.get("tp_method") or "").startswith("pattern_")]

    def _wr(trades):
        if not trades: return 0.0
        return sum(1 for t in trades if (t.get("pnl_pct") or 0) > 0) / len(trades) * 100

    pv = {
        "pattern_wr":    round(_wr(pat_trades), 1),
        "pattern_n":     len(pat_trades),
        "structural_wr": round(_wr(str_trades), 1),
        "structural_n":  len(str_trades),
    }

    return {
        "buckets": {k: buckets[k] for k in ("<0.75", "0.75–1.0", "1.0–1.5", "≥1.5") if k in buckets},
        "pattern_vs_structural": pv if pat_trades or str_trades else None,
    }


def _evaluate_shadow_trades(all_live_trades: list[dict]) -> dict | None:
    """Return win-rate comparison between shadow and live trades."""
    if not SHADOW_TRADES.exists():
        return None
    try:
        lines  = [l for l in SHADOW_TRADES.read_text().splitlines() if l.strip()]
        trades = [json.loads(l) for l in lines]
    except Exception:
        return None

    resolved = [t for t in trades if t.get("resolved") and t.get("outcome") in ("tp", "sl")]
    if len(resolved) < 3:
        return None

    shadow_wins = sum(1 for t in resolved if t["outcome"] == "tp")
    shadow_wr   = shadow_wins / len(resolved) * 100

    live_wins   = sum(1 for t in all_live_trades if (t.get("pnl_pct") or 0) > 0)
    live_wr     = live_wins / len(all_live_trades) * 100 if all_live_trades else 0

    return {
        "resolved":   len(resolved),
        "shadow_wr":  round(shadow_wr, 1),
        "live_wr":    round(live_wr,   1),
        "gap":        round(shadow_wr - live_wr, 1),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def should_run_downtime(all_trades: list[dict]) -> bool:
    """
    Return True if we've been idle long enough to warrant a downtime pass.
    Idle = no closed trade in IDLE_THRESHOLD_HOURS hours.
    """
    if not _cooldown_expired():
        return False
    if not all_trades:
        return True   # no trades ever → definitely idle
    last_trade_ts = max((t.get("close_time") or t.get("entry_time") or 0) for t in all_trades)
    idle_hours = (time.time() - last_trade_ts) / 3600
    return idle_hours >= IDLE_THRESHOLD_HOURS


def _cooldown_expired() -> bool:
    try:
        last = float(LAST_DOWNTIME_FILE.read_text())
        return (time.time() - last) >= IDLE_THRESHOLD_HOURS * 3600
    except Exception:
        return True


def _mark_run():
    LAST_DOWNTIME_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_DOWNTIME_FILE.write_text(str(time.time()))


def _log_downtime(record: dict):
    DOWNTIME_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DOWNTIME_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def _append_shadow(record: dict):
    SHADOW_TRADES.parent.mkdir(parents=True, exist_ok=True)
    with open(SHADOW_TRADES, "a") as f:
        f.write(json.dumps(record) + "\n")


def _trailing_sl_analysis(all_trades: list[dict]) -> dict | None:
    """Compare trades where trailing SL activated vs static SL trades."""
    closed = [t for t in all_trades if t.get("pnl_pct") is not None and t.get("close_reason") != "shutdown"]
    if len(closed) < 3:
        return None

    trailed = [t for t in closed if t.get("sl_method") in ("breakeven", ) or
               (t.get("sl_method") or "").startswith("trail@")]
    normal  = [t for t in closed if t not in trailed]

    def _stats(trades):
        if not trades:
            return 0, 0.0, 0.0
        wr  = sum(1 for t in trades if (t.get("pnl_pct") or 0) > 0) / len(trades) * 100
        pnl = sum((t.get("pnl_pct") or 0) * 100 for t in trades)
        return len(trades), round(wr, 1), round(pnl, 2)

    tn, twr, tpnl = _stats(trailed)
    nn, nwr, npnl = _stats(normal)

    stopped_at_be = sum(1 for t in trailed
                        if t.get("close_reason") == "stop_loss"
                        and t.get("sl_method") == "breakeven")

    return {
        "trailed_n":           tn,
        "trailed_wr":          twr,
        "trailed_pnl":         tpnl,
        "normal_n":            nn,
        "normal_wr":           nwr,
        "normal_pnl":          npnl,
        "stopped_at_breakeven": stopped_at_be,
    }
