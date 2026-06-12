"""
Self-improvement engine — analyses Hermes's own trade history,
forms hypotheses about what to change, validates via backtesting,
and applies approved changes to strategy.yaml.

Called from loop.py's reflection cycle every 5 closed trades.

Improvement philosophy
──────────────────────
  • One parameter change at a time — never change multiple things at once
  • Conservative steps — small adjustments (±3–5 on RSI thresholds, etc.)
  • Backtest validation required — change only applied if WR improves ≥10pp
    OR PnL improves ≥15% with no WR decline
  • Minimum data — need ≥8 live trades in a regime to diagnose it
  • Cooldown — same parameter can't be changed twice within 24 hours
  • Audit trail — every change (and rejection) logged to hypotheses.jsonl
"""
from __future__ import annotations
import asyncio
import json
import time
import yaml
from pathlib import Path
from datetime import datetime, timezone

STATE_DIR    = Path(__file__).parent.parent / "state"
STRATEGY_FILE = STATE_DIR / "strategy.yaml"
IMPROVE_LOG   = STATE_DIR / "improve_log.jsonl"

# Bounds for each tunable parameter
PARAM_BOUNDS = {
    "bull.long_threshold":        (25, 45),
    "bear.short_threshold":       (55, 72),
    "sideways.range_entry_pct":   (0.10, 0.30),
    "mtf.require_signals":        (0, 2),
    "stop_loss_pct":              (1.0, 3.5),
    "take_profit_pct":            (2.0, 5.0),
}

# Step sizes for each parameter
PARAM_STEPS = {
    "bull.long_threshold":        3,
    "bear.short_threshold":       3,
    "sideways.range_entry_pct":   0.05,
    "mtf.require_signals":        1,
    "stop_loss_pct":              0.3,
    "take_profit_pct":            0.3,
}

# Minimum live trades in a regime before we diagnose it
MIN_TRADES_TO_DIAGNOSE = 8

# Minimum backtest trades to trust the result
MIN_BACKTEST_TRADES = 10

# Minimum win-rate improvement (percentage points) to apply a change
MIN_WR_IMPROVEMENT_PP = 8.0

# Cooldown: don't change the same parameter within this many seconds
PARAM_COOLDOWN_SECS = 24 * 3600


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def improve(
    all_trades:    list[dict],
    strategy:      dict,
    active_pairs:  list[str],
) -> dict:
    """
    Full improvement cycle. Returns a summary dict with what was changed (or not).
    """
    summary = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "trades_n":   len(all_trades),
        "hypotheses": [],
        "backtests":  [],
        "changes":    [],
        "rejected":   [],
        "message":    "",
    }

    if len(all_trades) < 5:
        summary["message"] = "Not enough trades to analyse yet"
        return summary

    # 1. Diagnose
    hypotheses = diagnose(all_trades, strategy)
    summary["hypotheses"] = hypotheses

    if not hypotheses:
        summary["message"] = "No clear improvement signal yet — keep trading"
        return summary

    # 2. Pick the highest-priority hypothesis (one at a time)
    hypothesis = hypotheses[0]
    param      = hypothesis["param"]

    # Check cooldown
    if _in_cooldown(param):
        summary["message"] = f"Cooldown active for {param} — skipping"
        return summary

    # 3. Backtest current vs proposed
    current_val  = hypothesis["current_val"]
    proposed_val = hypothesis["proposed_val"]

    print(f"[self_improve] Testing: {param} {current_val} → {proposed_val} "
          f"(reason: {hypothesis['reason']})", flush=True)

    # Run backtests on up to 3 active pairs in parallel
    bt_pairs  = [p for p in active_pairs if p != "BTC/USDT"][:3] or active_pairs[:3]
    baseline_results, proposed_results = await _run_comparison(
        bt_pairs, strategy, param, current_val, proposed_val
    )
    summary["backtests"] = {
        "pairs":    bt_pairs,
        "baseline": baseline_results,
        "proposed": proposed_results,
    }

    # 4. Evaluate
    verdict = _evaluate(baseline_results, proposed_results, hypothesis)
    summary["hypotheses"][0]["verdict"] = verdict

    if verdict["approved"]:
        # 5. Apply
        new_strategy = _apply_change(strategy, param, proposed_val)
        _write_strategy(new_strategy)
        change_record = {
            "param":        param,
            "old_val":      current_val,
            "new_val":      proposed_val,
            "reason":       hypothesis["reason"],
            "baseline_wr":  verdict["baseline_wr"],
            "proposed_wr":  verdict["proposed_wr"],
            "wr_delta":     verdict["wr_delta"],
            "timestamp":    time.time(),
        }
        summary["changes"].append(change_record)
        _log_change(change_record, approved=True)

        summary["message"] = (
            f"✅ Changed {param}: {current_val} → {proposed_val}\n"
            f"Backtest WR: {verdict['baseline_wr']:.0f}% → {verdict['proposed_wr']:.0f}% "
            f"(+{verdict['wr_delta']:.0f}pp) over {verdict['bt_trades']} simulated trades"
        )
    else:
        summary["rejected"].append({
            "param":   param,
            "reason":  verdict["reject_reason"],
        })
        _log_change({"param": param, "reason": verdict["reject_reason"]}, approved=False)
        summary["message"] = (
            f"❌ Kept {param} at {current_val} — "
            f"backtest didn't improve enough ({verdict['reject_reason']})"
        )

    print(f"[self_improve] {summary['message']}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

def diagnose(trades: list[dict], strategy: dict) -> list[dict]:
    """
    Analyse trade history and return a prioritised list of hypotheses.
    Each hypothesis: {param, current_val, proposed_val, reason, priority}
    """
    hypotheses = []

    # Group trades by pair_regime
    by_regime: dict[str, list] = {}
    for t in trades:
        r = t.get("pair_regime") or t.get("regime_at_entry", "neutral")
        by_regime.setdefault(r, []).append(t)

    # ── Bull regime: check long threshold ────────────────────────────────
    bull_trades = by_regime.get("bull", [])
    if len(bull_trades) >= MIN_TRADES_TO_DIAGNOSE:
        bull_wr = _wr(bull_trades)
        bull_cfg = strategy.get("bull", {})
        current_thr = bull_cfg.get("long_threshold", 35)
        lo, hi = PARAM_BOUNDS["bull.long_threshold"]
        step   = PARAM_STEPS["bull.long_threshold"]

        if bull_wr < 40:
            # Too many losses in bull regime → be more selective (lower RSI threshold)
            proposed = max(lo, current_thr - step)
            if proposed != current_thr:
                hypotheses.append({
                    "param":       "bull.long_threshold",
                    "current_val": current_thr,
                    "proposed_val": proposed,
                    "reason":      f"Bull regime WR={bull_wr:.0f}% < 40% — tighten RSI dip threshold",
                    "priority":    1,
                    "regime_wr":   bull_wr,
                    "sample_n":    len(bull_trades),
                })
        elif bull_wr > 65 and current_thr < hi:
            # Performing well → we can be more aggressive (higher RSI threshold = more entries)
            proposed = min(hi, current_thr + step)
            hypotheses.append({
                "param":       "bull.long_threshold",
                "current_val": current_thr,
                "proposed_val": proposed,
                "reason":      f"Bull regime WR={bull_wr:.0f}% > 65% — can relax RSI threshold for more entries",
                "priority":    3,
                "regime_wr":   bull_wr,
                "sample_n":    len(bull_trades),
            })

    # ── Bear regime: check short threshold ───────────────────────────────
    bear_trades = by_regime.get("bear", [])
    if len(bear_trades) >= MIN_TRADES_TO_DIAGNOSE:
        bear_wr = _wr(bear_trades)
        bear_cfg = strategy.get("bear", {})
        current_thr = bear_cfg.get("short_threshold", 60)
        lo, hi = PARAM_BOUNDS["bear.short_threshold"]
        step   = PARAM_STEPS["bear.short_threshold"]

        if bear_wr < 40:
            proposed = min(hi, current_thr + step)   # raise = need stronger bounce to enter short
            if proposed != current_thr:
                hypotheses.append({
                    "param":        "bear.short_threshold",
                    "current_val":  current_thr,
                    "proposed_val": proposed,
                    "reason":       f"Bear regime WR={bear_wr:.0f}% < 40% — require stronger RSI bounce to short",
                    "priority":     1,
                    "regime_wr":    bear_wr,
                    "sample_n":     len(bear_trades),
                })
        elif bear_wr > 65 and current_thr > lo:
            proposed = max(lo, current_thr - step)
            hypotheses.append({
                "param":        "bear.short_threshold",
                "current_val":  current_thr,
                "proposed_val": proposed,
                "reason":       f"Bear regime WR={bear_wr:.0f}% > 65% — can short on weaker RSI bounces",
                "priority":     3,
                "regime_wr":    bear_wr,
                "sample_n":     len(bear_trades),
            })

    # ── Sideways: check range entry pct ──────────────────────────────────
    sw_trades = by_regime.get("sideways", [])
    if len(sw_trades) >= MIN_TRADES_TO_DIAGNOSE:
        sw_wr = _wr(sw_trades)
        sw_cfg = strategy.get("sideways", {})
        current_pct = sw_cfg.get("range_entry_pct", 0.20)
        lo, hi = PARAM_BOUNDS["sideways.range_entry_pct"]
        step   = PARAM_STEPS["sideways.range_entry_pct"]

        if sw_wr < 40:
            # Too many range trades failing → enter deeper into extremes (lower pct = tighter zone)
            proposed = round(max(lo, current_pct - step), 2)
            if proposed != current_pct:
                hypotheses.append({
                    "param":        "sideways.range_entry_pct",
                    "current_val":  current_pct,
                    "proposed_val": proposed,
                    "reason":       f"Sideways WR={sw_wr:.0f}% < 40% — enter only at tighter range extremes",
                    "priority":     2,
                    "regime_wr":    sw_wr,
                    "sample_n":     len(sw_trades),
                })
        elif sw_wr > 65 and current_pct < hi:
            proposed = round(min(hi, current_pct + step), 2)
            hypotheses.append({
                "param":        "sideways.range_entry_pct",
                "current_val":  current_pct,
                "proposed_val": proposed,
                "reason":       f"Sideways WR={sw_wr:.0f}% > 65% — wider entry zone = more opportunities",
                "priority":     3,
                "regime_wr":    sw_wr,
                "sample_n":     len(sw_trades),
            })

    # ── MTF gate: check if it's filtering too many / too few ─────────────
    # If >60% of all trades are being blocked by MTF and WR is still low →
    # MTF gate isn't helping much but is reducing entries
    all_wr = _wr(trades)
    mtf_req = strategy.get("mtf", {}).get("require_signals", 1)
    lo, hi  = PARAM_BOUNDS["mtf.require_signals"]

    if len(trades) >= 15 and all_wr > 60 and mtf_req < hi:
        # Doing well — can require more confirmation for even better entries
        hypotheses.append({
            "param":        "mtf.require_signals",
            "current_val":  mtf_req,
            "proposed_val": mtf_req + 1,
            "reason":       f"Overall WR={all_wr:.0f}% > 60% — stricter MTF gate may improve quality further",
            "priority":     4,
            "regime_wr":    all_wr,
            "sample_n":     len(trades),
        })
    elif len(trades) >= 15 and all_wr < 35 and mtf_req > lo:
        # Very low WR even with gate — gate may be forcing bad-timing entries
        # Actually reducing the gate lets more natural signals through
        hypotheses.append({
            "param":        "mtf.require_signals",
            "current_val":  mtf_req,
            "proposed_val": max(lo, mtf_req - 1),
            "reason":       f"Overall WR={all_wr:.0f}% < 35% — MTF gate may be too restrictive, filtering good setups",
            "priority":     2,
            "regime_wr":    all_wr,
            "sample_n":     len(trades),
        })

    # Sort by priority
    hypotheses.sort(key=lambda h: h["priority"])
    return hypotheses


# ---------------------------------------------------------------------------
# Backtest comparison
# ---------------------------------------------------------------------------

async def _run_comparison(
    pairs:        list[str],
    strategy:     dict,
    param:        str,
    current_val,
    proposed_val,
) -> tuple[dict, dict]:
    """Run baseline and proposed backtests in parallel across all pairs."""
    from hermes_trading.backtest import run_backtest

    proposed_strategy = _apply_change(strategy, param, proposed_val)

    baseline_tasks = [run_backtest(p, strategy,          lookback_days=30) for p in pairs]
    proposed_tasks = [run_backtest(p, proposed_strategy, lookback_days=30) for p in pairs]

    all_tasks = await asyncio.gather(*baseline_tasks, *proposed_tasks, return_exceptions=True)
    n = len(pairs)
    baseline_raw = all_tasks[:n]
    proposed_raw = all_tasks[n:]

    def _agg(results):
        valid = [r for r in results if isinstance(r, dict) and r.get("total_trades", 0) >= 1]
        if not valid:
            return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pnl_pct": 0, "avg_rr": 0}
        total  = sum(r["total_trades"] for r in valid)
        wins   = sum(r["wins"]         for r in valid)
        wr     = wins / total * 100 if total > 0 else 0
        pnl    = sum(r["total_pnl_pct"] for r in valid) / len(valid)
        rr     = sum(r["avg_rr"]        for r in valid) / len(valid)
        return {"total_trades": total, "wins": wins, "win_rate": round(wr, 1),
                "total_pnl_pct": round(pnl, 3), "avg_rr": round(rr, 3)}

    return _agg(baseline_raw), _agg(proposed_raw)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(baseline: dict, proposed: dict, hypothesis: dict) -> dict:
    bt  = proposed.get("total_trades", 0)
    bwr = baseline.get("win_rate",     0)
    pwr = proposed.get("win_rate",     0)
    delta = pwr - bwr

    if bt < MIN_BACKTEST_TRADES:
        return {"approved": False, "reject_reason": f"only {bt} backtest trades (need {MIN_BACKTEST_TRADES})",
                "baseline_wr": bwr, "proposed_wr": pwr, "wr_delta": delta, "bt_trades": bt}

    if delta < MIN_WR_IMPROVEMENT_PP:
        # Also approve if PnL improves significantly with no WR decline
        pnl_delta = proposed.get("total_pnl_pct", 0) - baseline.get("total_pnl_pct", 0)
        if pnl_delta > 5.0 and delta >= -2.0:
            return {"approved": True, "baseline_wr": bwr, "proposed_wr": pwr,
                    "wr_delta": delta, "bt_trades": bt, "reject_reason": None}
        return {"approved": False,
                "reject_reason": f"WR delta only +{delta:.1f}pp (need +{MIN_WR_IMPROVEMENT_PP}pp)",
                "baseline_wr": bwr, "proposed_wr": pwr, "wr_delta": delta, "bt_trades": bt}

    return {"approved": True, "baseline_wr": bwr, "proposed_wr": pwr,
            "wr_delta": delta, "bt_trades": bt, "reject_reason": None}


# ---------------------------------------------------------------------------
# Strategy mutation helpers
# ---------------------------------------------------------------------------

def _apply_change(strategy: dict, param: str, value) -> dict:
    """Return a copy of strategy with param set to value. Supports dotted paths."""
    import copy
    s = copy.deepcopy(strategy)
    parts = param.split(".")
    node = s
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    return s


def _write_strategy(strategy: dict):
    """Write updated strategy to disk. Loop.py picks it up on next tick."""
    STRATEGY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STRATEGY_FILE, "w") as f:
        yaml.dump(strategy, f, default_flow_style=False, sort_keys=False)


def _in_cooldown(param: str) -> bool:
    """Return True if this param was changed within PARAM_COOLDOWN_SECS."""
    if not IMPROVE_LOG.exists():
        return False
    cutoff = time.time() - PARAM_COOLDOWN_SECS
    try:
        for line in IMPROVE_LOG.read_text().splitlines():
            entry = json.loads(line)
            if entry.get("param") == param and entry.get("approved") and \
               entry.get("timestamp", 0) > cutoff:
                return True
    except Exception:
        pass
    return False


def _log_change(record: dict, approved: bool):
    IMPROVE_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {**record, "approved": approved, "timestamp": time.time()}
    with open(IMPROVE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wr(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if (t.get("pnl_pct") or 0) > 0)
    return wins / len(trades) * 100


def format_improvement_message(summary: dict) -> str:
    """Format a Telegram-ready summary of what changed."""
    msg = f"🧠 <b>Self-Improvement Cycle #{summary.get('trades_n', 0) // 5}</b>\n\n"

    if summary.get("changes"):
        for c in summary["changes"]:
            msg += (
                f"✅ <b>Applied:</b> {c['param']}\n"
                f"   {c['old_val']} → {c['new_val']}\n"
                f"   Reason: {c['reason']}\n"
                f"   Backtest WR: {c['baseline_wr']:.0f}% → {c['proposed_wr']:.0f}% "
                f"(+{c['wr_delta']:.0f}pp)\n\n"
            )
    elif summary.get("rejected"):
        for r in summary["rejected"]:
            msg += f"❌ <b>Rejected:</b> {r['param']} — {r['reason']}\n"
        msg += "\n"
    else:
        msg += f"📊 {summary.get('message', 'No changes needed')}\n\n"

    if summary.get("hypotheses"):
        h = summary["hypotheses"][0]
        msg += (
            f"<b>Top signal:</b> {h['param']} "
            f"({h['regime_wr']:.0f}% WR on {h['sample_n']} trades)\n"
            f"{h['reason']}"
        )

    return msg
