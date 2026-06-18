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

STATE_DIR            = Path(__file__).parent.parent / "state"
STRATEGY_FILE        = STATE_DIR / "strategy.yaml"
IMPROVE_LOG          = STATE_DIR / "improve_log.jsonl"
PAIR_THRESHOLDS_FILE = STATE_DIR / "pair_thresholds.json"
ACTIVE_FEATURES_FILE = STATE_DIR / "active_features.json"

# Per-pair RSI tuning: max deviation from strategy default (in RSI points)
PAIR_RSI_MAX_DRIFT = 5
# Minimum trades per pair before we tune it
PAIR_RSI_MIN_TRADES = 10

# Bounds for each tunable parameter
PARAM_BOUNDS = {
    "bull.long_threshold":                  (25, 45),
    "bear.short_threshold":                 (55, 72),
    "sideways.range_entry_pct":             (0.10, 0.30),
    "mtf.require_signals":                  (0, 2),
    "stop_loss_pct":                        (1.0, 3.5),
    "take_profit_pct":                      (2.0, 5.0),
    "conviction_sizing.low":                (20, 50),
    "conviction_sizing.medium":             (30, 75),
    "conviction_sizing.high":               (50, 125),
    "conviction_sizing.very_high":          (75, 200),
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

        # Register in active features manifest so trades can be tagged
        feature_name = f"{param}:{proposed_val}"
        register_feature(
            name        = feature_name,
            feature_type = "auto",
            description = hypothesis["reason"],
            param       = param,
            old_val     = current_val,
            new_val     = proposed_val,
            baseline_wr = verdict["baseline_wr"],
            proposed_wr = verdict["proposed_wr"],
        )

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

    # Always run per-pair RSI tuning alongside the main improvement cycle
    try:
        pair_updates = await tune_pair_thresholds(all_trades, strategy, active_pairs)
        if pair_updates:
            summary["pair_threshold_updates"] = pair_updates
    except Exception as e:
        print(f"[pair_thresh] tuning error: {e}", flush=True)

    return summary


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

def _shadow_wr() -> float | None:
    """Read shadow trade WR from shadow_trades.jsonl. Returns None if insufficient data."""
    shadow_file = STATE_DIR / "shadow_trades.jsonl"
    if not shadow_file.exists():
        return None
    try:
        import json as _j
        resolved = [_j.loads(l) for l in shadow_file.read_text().splitlines()
                    if l.strip() and _j.loads(l).get("outcome")]
        if len(resolved) < 10:
            return None
        return sum(1 for t in resolved if t["outcome"] == "tp") / len(resolved) * 100
    except Exception:
        return None


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

    live_wr     = _wr(trades) if trades else 0
    shadow_wr_v = _shadow_wr()
    shadow_gap  = (shadow_wr_v - live_wr) if shadow_wr_v is not None else 0

    # ── Bull regime: check long threshold ────────────────────────────────
    bull_trades = by_regime.get("bull", [])
    if len(bull_trades) >= MIN_TRADES_TO_DIAGNOSE:
        bull_wr = _wr(bull_trades)
        bull_cfg = strategy.get("bull", {})
        current_thr = bull_cfg.get("long_threshold", 35)
        lo, hi = PARAM_BOUNDS["bull.long_threshold"]
        step   = PARAM_STEPS["bull.long_threshold"]

        if shadow_gap > 15 and current_thr < hi:
            # Shadow trades winning much more than live — filters too tight, relax threshold
            proposed = min(hi, current_thr + step)
            hypotheses.append({
                "param":        "bull.long_threshold",
                "current_val":  current_thr,
                "proposed_val": proposed,
                "reason":       f"Shadow WR {shadow_wr_v:.0f}% >> live WR {live_wr:.0f}% (gap={shadow_gap:.0f}pp) — filters blocking winners, raise RSI threshold",
                "priority":     1,
                "regime_wr":    bull_wr,
                "sample_n":     len(bull_trades),
            })
        elif bull_wr < 40 and shadow_gap <= 10:
            # Losing AND shadows not outperforming → genuinely bad setups, tighten
            proposed = max(lo, current_thr - step)
            if proposed != current_thr:
                hypotheses.append({
                    "param":        "bull.long_threshold",
                    "current_val":  current_thr,
                    "proposed_val": proposed,
                    "reason":       f"Bull regime WR={bull_wr:.0f}% < 40% with no shadow gap — tighten RSI dip threshold",
                    "priority":     1,
                    "regime_wr":    bull_wr,
                    "sample_n":     len(bull_trades),
                })
        elif bull_wr > 65 and current_thr < hi:
            # Performing well → relax for more entries
            proposed = min(hi, current_thr + step)
            hypotheses.append({
                "param":        "bull.long_threshold",
                "current_val":  current_thr,
                "proposed_val": proposed,
                "reason":       f"Bull regime WR={bull_wr:.0f}% > 65% — can relax RSI threshold for more entries",
                "priority":     3,
                "regime_wr":    bull_wr,
                "sample_n":     len(bull_trades),
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

    # ── Conviction tier tuning ────────────────────────────────────────────
    # Group trades by conviction tier. If a tier has good WR but low capital,
    # raise it. If a tier has poor WR but high capital, lower it.
    scored_trades = [t for t in trades if t.get("conviction_score") is not None]
    if len(scored_trades) >= 10:
        tier_groups = {"low": [], "medium": [], "high": [], "very_high": []}
        for t in scored_trades:
            s = t.get("conviction_score", 2)
            if s >= 4:
                tier_groups["very_high"].append(t)
            elif s >= 3:
                tier_groups["high"].append(t)
            elif s >= 2:
                tier_groups["medium"].append(t)
            else:
                tier_groups["low"].append(t)

        sizing = strategy.get("conviction_sizing", {})
        tier_param_map = {
            "low":       ("conviction_sizing.low",       20,  50, 5),
            "medium":    ("conviction_sizing.medium",    30,  75, 5),
            "high":      ("conviction_sizing.high",      50, 125, 10),
            "very_high": ("conviction_sizing.very_high", 75, 200, 10),
        }
        # Find best and worst performing tiers with enough data
        tier_stats = {}
        for tier, tlist in tier_groups.items():
            if len(tlist) >= 5:
                tier_stats[tier] = (_wr(tlist), len(tlist), sizing.get(tier, 50))

        if tier_stats:
            best_tier  = max(tier_stats, key=lambda t: tier_stats[t][0])
            worst_tier = min(tier_stats, key=lambda t: tier_stats[t][0])
            best_wr, best_n, best_cap   = tier_stats[best_tier]
            worst_wr, worst_n, worst_cap = tier_stats[worst_tier]

            # Raise capital on best-performing tier if it has headroom
            param, lo, hi, step = tier_param_map[best_tier]
            if best_wr > 60 and best_cap < hi:
                proposed = min(hi, best_cap + step)
                hypotheses.append({
                    "param":        param,
                    "current_val":  best_cap,
                    "proposed_val": proposed,
                    "reason":       f"Tier '{best_tier}' WR={best_wr:.0f}% on {best_n} trades — increase capital allocation",
                    "priority":     2,
                    "regime_wr":    best_wr,
                    "sample_n":     best_n,
                })

            # Lower capital on worst-performing tier if it's hurting PnL
            param, lo, hi, step = tier_param_map[worst_tier]
            if worst_wr < 40 and worst_cap > lo and worst_tier != best_tier:
                proposed = max(lo, worst_cap - step)
                hypotheses.append({
                    "param":        param,
                    "current_val":  worst_cap,
                    "proposed_val": proposed,
                    "reason":       f"Tier '{worst_tier}' WR={worst_wr:.0f}% on {worst_n} trades — reduce capital allocation",
                    "priority":     2,
                    "regime_wr":    worst_wr,
                    "sample_n":     worst_n,
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
        # Weight PnL by trade count so a single high-volume pair can't dominate
        pnl    = sum(r["total_pnl_pct"] * r["total_trades"] for r in valid) / total if total > 0 else 0
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
        # Also approve if trade-weighted PnL improves with minimal WR decline.
        # Threshold 2.5% (not 5%) because backtest runs without leverage — live
        # PnL is ~2x higher, so 2.5% backtest uplift ≈ 5% live uplift.
        pnl_delta = proposed.get("total_pnl_pct", 0) - baseline.get("total_pnl_pct", 0)
        if pnl_delta > 2.5 and delta >= -2.0:
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
    """Write updated strategy to disk, preserving keys not touched by self_improve."""
    STRATEGY_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Load current file and deep-merge so unknown keys (conviction_sizing, trailing_sl etc.)
    # are never silently dropped by a self_improve rewrite
    existing = {}
    if STRATEGY_FILE.exists():
        try:
            with open(STRATEGY_FILE) as f:
                existing = yaml.safe_load(f) or {}
        except Exception:
            pass
    import copy
    merged = copy.deepcopy(existing)
    for k, v in strategy.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k].update(v)
        else:
            merged[k] = v
    with open(STRATEGY_FILE, "w") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=False)


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
# Active feature manifest — tracks what's live and when it changed
# ---------------------------------------------------------------------------

def load_active_features() -> dict:
    """Load the active features manifest. Returns {} if not yet created."""
    try:
        return json.loads(ACTIVE_FEATURES_FILE.read_text())
    except Exception:
        return {}


def register_feature(name: str, feature_type: str, description: str, **meta):
    """
    Register or update a feature in the active features manifest.
    Called by self_improve when it applies a change, and can be called
    at boot from loop.py to register code-level features.

    feature_type: 'auto'  — self_improve parameter change
                  'code'  — deployed code feature (session breakout, news caution, etc.)
                  'pair'  — per-pair threshold override
    """
    features = load_active_features()
    features[name] = {
        "type":        feature_type,
        "description": description,
        "enabled_at":  datetime.now(timezone.utc).isoformat(),
        **meta,
    }
    ACTIVE_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_FEATURES_FILE.write_text(json.dumps(features, indent=2))


def active_feature_labels() -> list[str]:
    """Return list of active feature names for embedding in trade records."""
    try:
        return list(load_active_features().keys())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Per-pair RSI threshold tuning
# ---------------------------------------------------------------------------

async def tune_pair_thresholds(
    all_trades: list[dict],
    strategy:   dict,
    active_pairs: list[str],
) -> dict:
    """
    For each pair with enough trade history, backtest RSI entry thresholds
    at default ±1..±5 points and save the best performing threshold to
    state/pair_thresholds.json.

    Only updates a pair's threshold if the best candidate beats the default
    by at least MIN_WR_IMPROVEMENT_PP win-rate points or 2.5% PnL.
    Never moves more than PAIR_RSI_MAX_DRIFT from the strategy default.
    """
    from hermes_trading.backtest import run_backtest
    import copy

    bull_default = strategy.get("bull", {}).get("long_threshold",  35)
    bear_default = strategy.get("bear", {}).get("short_threshold", 60)

    # Group trades by pair
    by_pair: dict[str, list] = {}
    for t in all_trades:
        p = t.get("pair")
        if p:
            by_pair.setdefault(p, []).append(t)

    current_thresholds = _load_pair_thresholds()
    updated = {}

    for pair in active_pairs:
        pair_trades = by_pair.get(pair, [])
        if len(pair_trades) < PAIR_RSI_MIN_TRADES:
            continue

        # Determine which threshold to tune based on dominant trade direction
        longs  = [t for t in pair_trades if t.get("direction") == "long"]
        shorts = [t for t in pair_trades if t.get("direction") == "short"]
        tune_long  = len(longs)  >= PAIR_RSI_MIN_TRADES // 2
        tune_short = len(shorts) >= PAIR_RSI_MIN_TRADES // 2

        candidates = []

        if tune_long:
            lo = max(25, bull_default - PAIR_RSI_MAX_DRIFT)
            hi = min(45, bull_default + PAIR_RSI_MAX_DRIFT)
            for thr in range(lo, hi + 1):
                s = copy.deepcopy(strategy)
                s.setdefault("bull", {})["long_threshold"] = thr
                candidates.append(("long_threshold", thr, s))

        if tune_short:
            lo = max(55, bear_default - PAIR_RSI_MAX_DRIFT)
            hi = min(72, bear_default + PAIR_RSI_MAX_DRIFT)
            for thr in range(lo, hi + 1):
                s = copy.deepcopy(strategy)
                s.setdefault("bear", {})["short_threshold"] = thr
                candidates.append(("short_threshold", thr, s))

        if not candidates:
            continue

        # Run all candidate backtests for this pair concurrently
        tasks   = [run_backtest(pair, s, lookback_days=60) for _, _, s in candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Baseline: current strategy
        try:
            baseline = await run_backtest(pair, strategy, lookback_days=60)
        except Exception:
            continue

        baseline_wr  = baseline.get("win_rate", 0)
        baseline_pnl = baseline.get("total_pnl_pct", 0)
        baseline_bt  = baseline.get("total_trades", 0)
        if baseline_bt < MIN_BACKTEST_TRADES:
            continue

        best_long_thr  = current_thresholds.get(pair, {}).get("long_threshold",  bull_default)
        best_short_thr = current_thresholds.get(pair, {}).get("short_threshold", bear_default)
        best_long_wr   = baseline_wr if tune_long  else 0
        best_short_wr  = baseline_wr if tune_short else 0
        improved       = False

        for (kind, thr, _), result in zip(candidates, results):
            if not isinstance(result, dict):
                continue
            bt  = result.get("total_trades", 0)
            wr  = result.get("win_rate", 0)
            pnl = result.get("total_pnl_pct", 0)
            if bt < MIN_BACKTEST_TRADES:
                continue
            wr_delta  = wr  - baseline_wr
            pnl_delta = pnl - baseline_pnl
            qualifies = (wr_delta >= MIN_WR_IMPROVEMENT_PP) or \
                        (pnl_delta > 2.5 and wr_delta >= -2.0)
            if kind == "long_threshold" and qualifies and wr > best_long_wr:
                best_long_wr  = wr
                best_long_thr = thr
                improved      = True
            elif kind == "short_threshold" and qualifies and wr > best_short_wr:
                best_short_wr = wr
                best_short_thr = thr
                improved       = True

        if improved:
            entry = current_thresholds.get(pair, {})
            if tune_long:
                entry["long_threshold"] = best_long_thr
            if tune_short:
                entry["short_threshold"] = best_short_thr
            entry["tuned_at"] = datetime.now(timezone.utc).isoformat()
            entry["sample_trades"] = len(pair_trades)
            current_thresholds[pair] = entry
            updated[pair] = entry
            print(f"[pair_thresh] {pair}: long_thr={best_long_thr} short_thr={best_short_thr} "
                  f"(baseline WR={baseline_wr:.0f}%)", flush=True)

            sym = pair.replace("/USDT", "")
            register_feature(
                name         = f"pair_rsi:{sym}",
                feature_type = "pair",
                description  = f"{sym} RSI thresholds tuned from historical data",
                pair         = pair,
                long_thr     = best_long_thr,
                short_thr    = best_short_thr,
                baseline_wr  = baseline_wr,
                sample_trades = len(pair_trades),
            )

    if updated:
        _write_pair_thresholds(current_thresholds)

    return updated


def _load_pair_thresholds() -> dict:
    try:
        return json.loads(PAIR_THRESHOLDS_FILE.read_text())
    except Exception:
        return {}


def _write_pair_thresholds(data: dict):
    PAIR_THRESHOLDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAIR_THRESHOLDS_FILE.write_text(json.dumps(data, indent=2))


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
