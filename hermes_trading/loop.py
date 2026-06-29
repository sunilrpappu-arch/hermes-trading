"""
TradingLoop — one instance per active pair.

Entry checklist (long):
  1. 15m RSI < long_threshold
  2. Trend filter: price above 50MA  OR  liquidity grab detected (overrides trend filter)
  3. MTF confluence (if enabled): at least N of:
       4H MACD bullish / 4H uptrend / 1H RSI divergence / 1H MACD crossover
  4. Pair drawdown < cap  AND  portfolio drawdown < cap

Entry checklist (short): mirror, inverted.

Risk params come from strategy.yaml, overridden by regime (regime is always tighter or equal).
"""
import asyncio
import json
import time
import os
import yaml
from pathlib import Path
from datetime import datetime, timezone

from hermes_trading.adapters import price as price_adapter
from hermes_trading.adapters.exchange import (
    is_live, open_long, open_short, close_long, close_short
)
from hermes_trading.indicators import (
    rsi as compute_rsi,
    macd as compute_macd,
    sma,
    rsi_divergence,
    liquidity_grab as detect_liquidity_grab,
    breakout_detector,
    candlestick_patterns,
    chart_patterns,
    bb_squeeze,
    vwap as compute_vwap,
    dynamic_levels,
    prev_day_levels,
    opening_range as compute_opening_range,
    swing_levels as compute_swing_levels,
    range_position,
    classify_pair_regime,
)
from hermes_trading.adapters.candles import closes as get_closes, highs as get_highs, lows as get_lows
from hermes_trading.notify import send_trade_email, send_reflection_notification, send_entry_notification
from hermes_trading.session_windows import current_session, session_volume_multiplier
from hermes_trading.news import fetch_news, symbol_from_pair
from hermes_trading.downtime import (
    should_run_downtime, run_downtime,
    record_shadow_trade, resolve_shadow_trades,
)
from hermes_trading.self_improve import register_feature, active_feature_labels

STATE_DIR           = Path(os.getenv("STATE_DIR", Path(__file__).parent.parent / "state"))
TRADES_FILE         = STATE_DIR / "trades.jsonl"


def _load_real_trades() -> list[dict]:
    """Load trades.jsonl excluding shutdown artifacts (redeploy fake-closes)."""
    import json as _j
    trades = []
    try:
        for line in TRADES_FILE.read_text().splitlines():
            if line.strip():
                t = _j.loads(line)
                if t.get("close_reason") != "shutdown":
                    trades.append(t)
    except Exception:
        pass
    return trades
STRATEGY_FILE       = STATE_DIR / "strategy.yaml"
DD_FILE             = STATE_DIR / "drawdown.json"   # portfolio-level drawdown state
CONTROLS_FILE       = STATE_DIR / "controls.json"   # manual override commands
PAIR_THRESHOLDS_FILE = STATE_DIR / "pair_thresholds.json"

# Binance taker fee per leg (configurable via env). Round-trip = 2×.
COMMISSION_PCT = float(os.getenv("COMMISSION_PCT", "0.001"))   # 0.1% default
# Estimated slippage per leg for live orders (0 for paper).
SLIPPAGE_PCT   = float(os.getenv("SLIPPAGE_PCT",   "0.0005"))  # 0.05% default


# ---------------------------------------------------------------------------
# Controls helpers (manual override from dashboard)
# ---------------------------------------------------------------------------

def read_controls() -> dict:
    """Read the controls file — returns defaults if missing or corrupt."""
    try:
        if CONTROLS_FILE.exists():
            return json.loads(CONTROLS_FILE.read_text())
    except Exception:
        pass
    return {"all_stop": False, "manual_exits": [], "pending_entries": [], "leverage_overrides": {}}


def _write_controls(ctrl: dict):
    try:
        CONTROLS_FILE.write_text(json.dumps(ctrl, indent=2))
    except Exception as e:
        print(f"[controls] write failed: {e}", flush=True)


def consume_manual_exit(asset: str):
    ctrl = read_controls()
    exits = [a for a in ctrl.get("manual_exits", []) if a != asset]
    ctrl["manual_exits"] = exits
    _write_controls(ctrl)


def consume_pending_entry(asset: str):
    ctrl = read_controls()
    ctrl["pending_entries"] = [e for e in ctrl.get("pending_entries", []) if e.get("asset") != asset]
    _write_controls(ctrl)

LOOP_INTERVAL          = int(os.getenv("LOOP_INTERVAL_SECONDS", "15"))
MAX_CONSECUTIVE_FAILURES = 5
RETRY_ATTEMPTS         = 3
PRICE_HISTORY_MAX      = 200

CAPITAL_PER_PAIR_USDT = float(os.getenv("CAPITAL_PER_PAIR_USDT", "50"))  # base capital; conviction sizing overrides this

# ---------------------------------------------------------------------------
# Portfolio-level drawdown tracker (shared across all pairs in this process)
# ---------------------------------------------------------------------------

class PortfolioDrawdown:
    """
    Tracks realised PnL across all pairs.
    Halts new entries when portfolio drawdown exceeds the cap.
    """
    _peak_usdt:      float = 0.0
    _current_usdt:   float = 0.0
    _total_capital:  float = 1000.0   # updated on every record_trade call
    _halted:         bool  = False

    @classmethod
    def record_trade(cls, pnl_usdt: float, total_capital: float):
        cls._total_capital = total_capital
        cls._current_usdt += pnl_usdt
        if cls._current_usdt > cls._peak_usdt:
            cls._peak_usdt = cls._current_usdt
        cls._save(total_capital)

    @classmethod
    def drawdown(cls) -> float:
        """Current drawdown as fraction of total capital (0.05 = 5%).
        Always normalises against total_capital so a win-then-loss sequence
        can never produce a drawdown > 100% (the old peak-denominator formula
        gave (30 - (-20)) / 30 = 166% which permanently halted trading)."""
        swing = max(0.0, cls._peak_usdt - cls._current_usdt)
        return swing / max(cls._total_capital, 1)

    @classmethod
    def is_halted(cls, cap: float) -> bool:
        halted = cls.drawdown() >= cap
        if halted and not cls._halted:
            print(f"[PORTFOLIO DD CAP] drawdown={cls.drawdown():.2%} ≥ cap={cap:.2%} — halting all entries", flush=True)
        cls._halted = halted
        return halted

    @classmethod
    def _save(cls, total_capital: float):
        try:
            DD_FILE.write_text(json.dumps({
                "peak_usdt":    cls._peak_usdt,
                "current_usdt": cls._current_usdt,
                "drawdown_pct": round(cls.drawdown() * 100, 3),
                "total_capital": total_capital,
                "updated":      datetime.now(timezone.utc).isoformat(),
            }, indent=2))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Daily loss limit (shared across all pairs in this process)
# ---------------------------------------------------------------------------

class DailyLossGuard:
    """
    Tracks today's realised PnL in USDT.
    Halts new entries for the rest of the UTC day if daily loss exceeds the cap.
    Resets automatically at UTC midnight.
    """
    _date:       str   = ""    # "YYYY-MM-DD" of the current trading day
    _daily_pnl:  float = 0.0   # cumulative PnL since midnight UTC

    @classmethod
    def _today(cls) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @classmethod
    def record(cls, pnl_usdt: float):
        today = cls._today()
        if cls._date != today:
            cls._date      = today
            cls._daily_pnl = 0.0   # reset at midnight
        cls._daily_pnl += pnl_usdt

    @classmethod
    def is_halted(cls, cap_pct: float, total_capital: float) -> bool:
        today = cls._today()
        if cls._date != today:
            cls._date      = today
            cls._daily_pnl = 0.0
        cap_usdt = total_capital * cap_pct
        halted   = cls._daily_pnl <= -cap_usdt
        if halted:
            print(
                f"[DAILY LOSS CAP] daily_pnl=${cls._daily_pnl:.2f} ≤ -${cap_usdt:.2f} "
                f"({cap_pct:.0%} of capital) — no new entries until midnight UTC",
                flush=True,
            )
        return halted

    @classmethod
    def summary(cls) -> dict:
        return {"date": cls._date, "daily_pnl_usdt": round(cls._daily_pnl, 4)}


async def fetch_with_retry(fn, *args, **kwargs):
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            await asyncio.sleep(2 ** attempt)


DEFAULT_STRATEGY = {
    "version": "18",
    # Bull regime: price above 50MA on 4H, ADX >= 20
    # Long bias — buy RSI dips. Short only on confirmed liquidity grabs.
    "bull": {
        "long_threshold":   35,
        "short_on_lq_only": True,
    },
    # Bear regime: price below 50MA on 4H, ADX >= 20
    # Short bias — sell RSI bounces. Long only on confirmed liquidity grabs.
    "bear": {
        "short_threshold":  60,
        "long_on_lq_only":  True,
    },
    # Sideways regime: ADX < 20 on 4H — mean-reversion at range extremes
    "sideways": {
        "adx_threshold":   20.0,
        "range_entry_pct": 0.20,
        "or_bars":         4,
        "swing_lookback":  20,
    },
    # MTF gate: require ≥N HTF signals before entering (0 = disabled).
    # Live strategy.yaml sets require_signals: 1 — this default is only used
    # if strategy.yaml is missing entirely (cold start with no state volume).
    "mtf": {
        "enabled":         True,
        "require_signals": 1,
    },
    "trend_filter": {"enabled": True, "ma_period": 50},
    "liquidity_grab": {
        "enabled":    True,
        "wick_ratio": 2.0,
        "sweep_pct":  0.002,
        "overrides_trend_filter": True,
    },
    "drawdown": {
        "per_pair_cap":  0.10,
        "portfolio_cap": 0.08,
    },
    "cooldown": {
        "after_stop_loss_minutes": 30,   # wait 30 min before re-entering a pair after a stop-loss
    },
    "daily_loss": {
        "max_loss_pct": 0.03,            # halt all new entries if down >3% of total capital on the day
    },
    "leverage": {
        # Default leverage per macro regime (can be overridden per-pair via dashboard)
        "sideways": 2.0,
        "calm":     2.0,
        "normal":   1.5,
        "volatile": 1.0,
        "extreme":  1.0,
        "max_leverage": 3.0,   # hard cap — dashboard cannot exceed this
    },
    "take_profit_pct":  3.0,
    "stop_loss_pct":    1.8,
    "position_size_r":  0.05,
}


def _run_reflection(trades: list[dict], stats: dict):
    """
    Reflection cycle — runs every N closed trades.
    Analyses recent performance and sends a summary via Telegram.
    In future versions this will auto-update strategy parameters.
    """
    n = len(trades)
    recent = trades[-5:]  # last 5 trades

    wins_recent   = sum(1 for t in recent if (t.get("pnl_pct") or 0) > 0)
    losses_recent = len(recent) - wins_recent
    pnl_recent    = sum((t.get("pnl_usdt") or 0) for t in recent)

    # Breakdown by close reason
    reasons = {}
    for t in recent:
        r = t.get("close_reason", "?")
        reasons[r] = reasons.get(r, 0) + 1

    # Best/worst of recent
    best  = max(recent, key=lambda t: t.get("pnl_pct") or 0)
    worst = min(recent, key=lambda t: t.get("pnl_pct") or 0)

    # Regime breakdown
    regimes = {}
    for t in recent:
        reg = t.get("regime_at_entry", "?")
        regimes[reg] = regimes.get(reg, 0) + 1

    reasons_str = "  ".join(f"{r}:{c}" for r, c in reasons.items())
    regimes_str = "  ".join(f"{r}:{c}" for r, c in regimes.items())
    best_pct    = (best.get("pnl_pct") or 0) * 100
    worst_pct   = (worst.get("pnl_pct") or 0) * 100

    # Conviction tier breakdown
    tier_map = {"low": [], "medium": [], "high": [], "very_high": []}
    for t in trades:
        score = t.get("conviction_score")
        if score is None:
            continue
        if score >= 4:
            tier_map["very_high"].append(t)
        elif score >= 3:
            tier_map["high"].append(t)
        elif score >= 2:
            tier_map["medium"].append(t)
        else:
            tier_map["low"].append(t)

    def _tier_summary(tlist: list) -> str:
        if not tlist:
            return "—"
        wr  = 100 * sum(1 for t in tlist if (t.get("pnl_pct") or 0) > 0) / len(tlist)
        pnl = sum((t.get("pnl_usdt") or 0) for t in tlist)
        return f"{len(tlist)}t  WR {wr:.0f}%  ${pnl:+.2f}"

    conviction_str = (
        f"Low:    {_tier_summary(tier_map['low'])}\n"
        f"Med:    {_tier_summary(tier_map['medium'])}\n"
        f"High:   {_tier_summary(tier_map['high'])}\n"
        f"V.High: {_tier_summary(tier_map['very_high'])}"
    )

    summary = (
        f"🔄 <b>Reflection #{n // 5}</b> — last {len(recent)} trades\n\n"
        f"W/L: {wins_recent}W / {losses_recent}L   "
        f"PnL: {'+' if pnl_recent >= 0 else ''}${pnl_recent:.4f}\n"
        f"Best:  {best.get('asset','?')} {best_pct:+.2f}%\n"
        f"Worst: {worst.get('asset','?')} {worst_pct:+.2f}%\n"
        f"Reasons: {reasons_str}\n"
        f"Regimes: {regimes_str}\n\n"
        f"<b>Conviction tiers (all-time):</b>\n{conviction_str}\n\n"
        f"<b>All-time</b>: {stats['total_trades']} trades  "
        f"{stats['wins']}W/{stats['losses']}L  "
        f"WR {stats['win_rate']:.0f}%  "
        f"PnL ${stats['total_pnl_usdt']:+.4f}"
    )

    print(f"[reflection] {summary}", flush=True)
    send_reflection_notification(summary)

    # Log to hypotheses file for future self-improvement
    try:
        hyp = {
            "timestamp":    time.time(),
            "cycle":        n // 5,
            "trades_n":     n,
            "recent_wr":    wins_recent / len(recent) if recent else 0,
            "recent_pnl":   pnl_recent,
            "reasons":      reasons,
            "regimes":      regimes,
            "stats":        stats,
        }
        hyp_file = STATE_DIR / "hypotheses.jsonl"
        with open(hyp_file, "a") as f:
            f.write(json.dumps(hyp) + "\n")
    except Exception as e:
        print(f"[reflection] failed to log hypothesis: {e}", flush=True)


async def _run_self_improvement(all_trades: list[dict], strategy_keys: list[str]):
    """
    Run the full self-improvement cycle asynchronously so it doesn't block trading.
    Analyses trade history, backtests proposed changes, and updates strategy.yaml.
    """
    try:
        from hermes_trading.self_improve import improve, format_improvement_message
        strategy = load_strategy()

        # Collect active pairs from heartbeat files
        active_pairs = []
        for hb_file in STATE_DIR.glob("heartbeat_*.json"):
            try:
                hb = json.loads(hb_file.read_text())
                if hb.get("asset"):
                    active_pairs.append(hb["asset"])
            except Exception:
                pass
        if not active_pairs:
            active_pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

        print(f"[self_improve] Starting improvement cycle — {len(all_trades)} trades, "
              f"pairs: {active_pairs}", flush=True)

        summary = await improve(all_trades, strategy, active_pairs)

        # Send Telegram notification about what changed
        msg = format_improvement_message(summary)
        send_reflection_notification(msg)

    except Exception as e:
        print(f"[self_improve] Cycle failed: {e}", flush=True)


async def _run_downtime_check(all_trades: list[dict]):
    """
    Fires when the system has been idle for IDLE_THRESHOLD_HOURS.
    Runs idle diagnosis, OOS backtest, and shadow trade evaluation.
    """
    try:
        strategy     = load_strategy()
        heartbeats   = {}
        active_pairs = []
        for hb_file in STATE_DIR.glob("heartbeat_*.json"):
            try:
                hb = json.loads(hb_file.read_text())
                if hb.get("asset"):
                    heartbeats[hb["asset"]] = hb
                    active_pairs.append(hb["asset"])
            except Exception:
                pass

        print(f"[downtime] Starting downtime analysis — idle, {len(all_trades)} total trades", flush=True)
        msg = await run_downtime(all_trades, strategy, active_pairs, heartbeats)
        if msg:
            send_reflection_notification(msg)
    except Exception as e:
        print(f"[downtime] Failed: {e}", flush=True)


def load_pair_thresholds() -> dict:
    """Load per-pair RSI threshold overrides written by self_improve."""
    try:
        return json.loads(PAIR_THRESHOLDS_FILE.read_text())
    except Exception:
        return {}


def load_strategy() -> dict:
    if not STRATEGY_FILE.exists():
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(STRATEGY_FILE, "w") as f:
            yaml.dump(DEFAULT_STRATEGY, f)
        return DEFAULT_STRATEGY
    with open(STRATEGY_FILE) as f:
        return yaml.safe_load(f)


class TradingLoop:
    def __init__(self, asset: str, capital_usdt: float = CAPITAL_PER_PAIR_USDT):
        self.asset         = asset
        self.capital_usdt  = capital_usdt
        self.price_history: list[float] = []
        self.open_position: dict | None = None
        self.consecutive_failures       = 0

        # Per-pair drawdown tracking
        self._realised_pnl_usdt: float = 0.0
        self._pair_peak_usdt:    float = 0.0

        # Post-loss cooldown: timestamp of last stop-loss exit (0 = never)
        self._last_stop_loss_ts: float = 0.0

    # ------------------------------------------------------------------
    # Drawdown helpers
    # ------------------------------------------------------------------

    def _trade_costs(self, usdt_deployed: float, leverage: float) -> tuple[float, float]:
        """Return (commission_usdt, slippage_usdt) for a round-trip trade."""
        notional       = usdt_deployed * leverage
        commission     = notional * COMMISSION_PCT * 2          # entry + exit leg
        slip           = notional * (SLIPPAGE_PCT * 2 if is_live() else 0.0)
        return round(commission, 6), round(slip, 6)

    def _record_pnl(self, pnl_pct: float, usdt_deployed: float, close_reason: str = ""):
        pnl_usdt = pnl_pct * usdt_deployed
        self._realised_pnl_usdt += pnl_usdt
        if self._realised_pnl_usdt > self._pair_peak_usdt:
            self._pair_peak_usdt = self._realised_pnl_usdt
        total_capital = float(os.getenv("TOTAL_CAPITAL_USDT", "1000"))
        PortfolioDrawdown.record_trade(pnl_usdt, total_capital)
        DailyLossGuard.record(pnl_usdt)
        # Start cooldown timer on stop-loss exits
        if close_reason == "stop_loss":
            self._last_stop_loss_ts = time.time()

    def _pair_drawdown(self) -> float:
        if self._pair_peak_usdt <= 0:
            return abs(min(self._realised_pnl_usdt, 0)) / self.capital_usdt
        return max(0.0, (self._pair_peak_usdt - self._realised_pnl_usdt) / self.capital_usdt)

    def _pair_halted(self, cap: float) -> bool:
        if self._pair_drawdown() >= cap:
            print(f"[PAIR DD CAP] {self.asset} drawdown={self._pair_drawdown():.2%} ≥ {cap:.2%} — pausing pair", flush=True)
            return True
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _position_file(self) -> Path:
        safe = self.asset.replace("/", "_")
        return STATE_DIR / f"position_{safe}.json"

    def _save_position(self):
        """Persist open position to disk so it survives restarts."""
        pf = self._position_file()
        if self.open_position:
            pf.write_text(json.dumps(self.open_position, indent=2))
        else:
            pf.unlink(missing_ok=True)

    def _load_position(self):
        """Restore open position from disk on startup."""
        pf = self._position_file()
        if pf.exists():
            try:
                self.open_position = json.loads(pf.read_text())
                print(f"[{self.asset}] restored open position from disk: "
                      f"{self.open_position['direction']} @ {self.open_position['entry_price']}", flush=True)
            except Exception as e:
                print(f"[{self.asset}] failed to restore position: {e}", flush=True)
                self.open_position = None

    def _trades_file(self) -> Path:
        safe = self.asset.replace("/", "_")
        return STATE_DIR / f"trades_{safe}.jsonl"

    def write_heartbeat(self, status: str, extra: dict = None):
        data = {
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "status":        status,
            "asset":         self.asset,
            "open_position": self.open_position,
            "pair_dd_pct":   round(self._pair_drawdown() * 100, 2),
            "portfolio_dd_pct": round(PortfolioDrawdown.drawdown() * 100, 2),
            "mode":          "live" if is_live() else "paper",
        }
        if extra:
            data.update(extra)
        safe = self.asset.replace("/", "_")
        (STATE_DIR / f"heartbeat_{safe}.json").write_text(json.dumps(data, indent=2))

    def log_trade(self, trade: dict):
        for path in (self._trades_file(), TRADES_FILE):
            with open(path, "a") as f:
                f.write(json.dumps(trade) + "\n")

    def _deploy_usdt(self, position_size_r: float, regime_params: dict) -> float:
        effective_r       = min(position_size_r, regime_params.get("position_size_r", position_size_r))
        effective_capital = min(self.capital_usdt, regime_params.get("capital_per_pair", self.capital_usdt))
        return effective_capital * effective_r

    def _conviction_capital(self, strategy: dict, htf_reasons: list, lq_note: str | None,
                            rsi: float, patterns: list, direction: str) -> tuple[float, int]:
        """Score conviction 0-5, return (capital_usdt, score).

        Scoring is intentionally strict — most trades should land at 2-3, not 4-5.
        • HTF: only active signals (MACD crossover/divergence, BB squeeze, RSI divergence).
          Passive regime signals ("4H uptrend", "above VWAP") don't count — they're
          already implied by being in bull/bear regime.
        • LQ grab: only a confirmed wick sweep scores; generic entry-type labels don't.
        • RSI extremity: stricter thresholds (<20 / >80) so only genuine extremes score.
        """
        score = 0

        # Only active HTF signals count (crossovers, divergences, BB squeeze breakouts)
        active_htf = [
            r for r in htf_reasons
            if any(kw in r for kw in ("MACD", "RSI divergence", "squeeze"))
        ]
        score += min(len(active_htf), 2)

        # Only real wick-sweep liquidity grabs score, not entry-type labels
        if lq_note and "lq_grab" in lq_note:
            score += 1

        if len(patterns) >= 2:
            score += 1

        # Genuine RSI extremes only (not just the entry RSI threshold)
        if direction == "long" and rsi < 20:
            score += 1
        elif direction == "short" and rsi > 80:
            score += 1

        tiers = strategy.get("conviction_sizing", {})
        if not tiers:
            # No tiers configured — use a safe fixed amount, never raw capital_usdt
            return 50.0, score

        # Tier values are absolute dollar amounts — never fall back to capital_usdt
        if score >= 4:
            return float(tiers.get("very_high", 100)), score
        elif score >= 3:
            return float(tiers.get("high", 75)), score
        elif score >= 2:
            return float(tiers.get("medium", 50)), score
        else:
            return float(tiers.get("low", 30)), score

    # ------------------------------------------------------------------
    # HTF signal evaluation
    # ------------------------------------------------------------------

    def _htf_signals_long(self, candles: dict,
                          patterns_4h: dict = None, patterns_1h: dict = None,
                          bb_1h: dict = None,
                          vwap_data: dict = None) -> tuple[int, list[str]]:
        confirmed = []
        c4h = get_closes(candles.get("4h", []))
        m4h = compute_macd(c4h)
        if m4h and (m4h["crossover_bullish"] or (m4h["histogram"] > 0 and m4h["histogram_rising"])):
            confirmed.append("4H MACD bullish")
        if len(c4h) >= 50:
            ma4h = sma(c4h, 50)
            if ma4h and c4h[-1] > ma4h:
                confirmed.append("4H uptrend")
        c1h = get_closes(candles.get("1h", []))
        div = rsi_divergence(c1h)
        if div["bullish"]:
            confirmed.append("1H RSI divergence")
        m1h = compute_macd(c1h)
        if m1h and m1h["crossover_bullish"]:
            confirmed.append("1H MACD crossover")
        # Chart patterns count as HTF confirmation (Step 2 feeds Step 3)
        if patterns_4h:
            for p in patterns_4h.get("bullish_patterns", []):
                confirmed.append(f"4H {p.replace('_',' ')}")
        if patterns_1h:
            for p in patterns_1h.get("bullish_patterns", []):
                confirmed.append(f"1H {p.replace('_',' ')}")
        # BB squeeze → bullish expansion is a strong breakout signal
        if bb_1h and bb_1h.get("expanding") and bb_1h.get("expansion_dir") == "up":
            label = "1H BB squeeze→up" if bb_1h.get("was_squeezing") else "1H BB expanding up"
            confirmed.append(label)
        # VWAP: price above VWAP = institutional buy-side in control (bullish bias)
        if vwap_data and vwap_data.get("price_above"):
            confirmed.append("above VWAP")
        # VWAP oversold band: price at/below -1σ = mean-reversion long setup
        if vwap_data and (vwap_data.get("at_lower_1") or vwap_data.get("at_lower_2")):
            band = "VWAP -2σ" if vwap_data.get("at_lower_2") else "VWAP -1σ"
            confirmed.append(f"bounce off {band}")
        return len(confirmed), confirmed

    def _htf_signals_short(self, candles: dict,
                           patterns_4h: dict = None, patterns_1h: dict = None,
                           bb_1h: dict = None,
                           vwap_data: dict = None) -> tuple[int, list[str]]:
        confirmed = []
        c4h = get_closes(candles.get("4h", []))
        m4h = compute_macd(c4h)
        if m4h and (m4h["crossover_bearish"] or (m4h["histogram"] < 0 and m4h["histogram_falling"])):
            confirmed.append("4H MACD bearish")
        if len(c4h) >= 50:
            ma4h = sma(c4h, 50)
            if ma4h and c4h[-1] < ma4h:
                confirmed.append("4H downtrend")
        c1h = get_closes(candles.get("1h", []))
        div = rsi_divergence(c1h)
        if div["bearish"]:
            confirmed.append("1H RSI divergence")
        m1h = compute_macd(c1h)
        if m1h and m1h["crossover_bearish"]:
            confirmed.append("1H MACD crossover")
        # Chart patterns count as HTF confirmation
        if patterns_4h:
            for p in patterns_4h.get("bearish_patterns", []):
                confirmed.append(f"4H {p.replace('_',' ')}")
        if patterns_1h:
            for p in patterns_1h.get("bearish_patterns", []):
                confirmed.append(f"1H {p.replace('_',' ')}")
        # BB squeeze → bearish expansion
        if bb_1h and bb_1h.get("expanding") and bb_1h.get("expansion_dir") == "down":
            label = "1H BB squeeze→down" if bb_1h.get("was_squeezing") else "1H BB expanding down"
            confirmed.append(label)
        # VWAP: price below VWAP = institutional sell-side in control (bearish bias)
        if vwap_data and not vwap_data.get("price_above"):
            confirmed.append("below VWAP")
        # VWAP overbought band: price at/above +1σ = mean-reversion short setup
        if vwap_data and (vwap_data.get("at_upper_1") or vwap_data.get("at_upper_2")):
            band = "VWAP +2σ" if vwap_data.get("at_upper_2") else "VWAP +1σ"
            confirmed.append(f"rejected at {band}")
        return len(confirmed), confirmed

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    async def tick(self, market_data: dict = None):
        strategy        = load_strategy()
        pair_thresholds = load_pair_thresholds()
        trend_cfg       = strategy.get("trend_filter", {})
        lq_cfg          = strategy.get("liquidity_grab", {})
        dd_cfg          = strategy.get("drawdown", {})
        lev_cfg         = strategy.get("leverage", {})
        cooldown_cfg    = strategy.get("cooldown", {})
        daily_loss_cfg  = strategy.get("daily_loss", {})

        trend_enabled      = trend_cfg.get("enabled", True)
        ma_period          = trend_cfg.get("ma_period", 50)
        lq_enabled         = lq_cfg.get("enabled", True)
        lq_overrides_trend = lq_cfg.get("overrides_trend_filter", True)
        pair_dd_cap        = dd_cfg.get("per_pair_cap", 0.10)
        portfolio_dd_cap   = dd_cfg.get("portfolio_cap", 0.08)
        max_leverage       = float(lev_cfg.get("max_leverage", 3.0))
        cooldown_minutes   = float(cooldown_cfg.get("after_stop_loss_minutes", 30))
        daily_loss_cap     = float(daily_loss_cfg.get("max_loss_pct", 0.03))

        mtf_cfg            = strategy.get("mtf", {})
        mtf_require        = int(mtf_cfg.get("require_signals", 1))   # ≥1 HTF signal required

        sb_cfg             = strategy.get("session_breakout", {})
        sb_enabled         = sb_cfg.get("enabled", True)
        sb_rsi_relax       = int(sb_cfg.get("rsi_relax_pts", 5))
        sb_vol_min         = float(sb_cfg.get("volume_spike_min", 1.5))
        sb_require_bb      = sb_cfg.get("require_bb_expanding", True)
        sb_min_fg          = float(sb_cfg.get("min_fg_score", 18))
        sb_max_fg          = float(sb_cfg.get("max_fg_score", 85))

        sw_cfg            = strategy.get("sideways", {})
        sw_entry_pct      = sw_cfg.get("range_entry_pct", 0.20)
        sw_or_bars        = sw_cfg.get("or_bars",         4)
        sw_swing_lbk      = sw_cfg.get("swing_lookback",  20)

        regime_params     = (market_data or {}).get("regime_params", {})
        is_sideways       = (market_data or {}).get("is_sideways", False)
        stop_loss_pct     = regime_params.get("stop_loss_pct",   strategy.get("stop_loss_pct",   1.8)) / 100
        take_profit_pct   = regime_params.get("take_profit_pct", strategy.get("take_profit_pct", 3.0)) / 100
        position_size_r   = strategy.get("position_size_r", 0.05)

        # Read manual controls (checked on every tick)
        ctrl = read_controls()
        all_stop = ctrl.get("all_stop", False)

        # Price
        if market_data and "price" in market_data:
            current_price = market_data["price"]
        else:
            pd = await fetch_with_retry(price_adapter.fetch, self.asset)
            current_price = pd["price"]

        if current_price <= 0:
            return  # bad tick — skip

        self.price_history.append(current_price)
        if len(self.price_history) > PRICE_HISTORY_MAX:
            self.price_history = self.price_history[-PRICE_HISTORY_MAX:]

        candles = (market_data or {}).get("candles", {})
        c15m    = get_closes(candles.get("15m", []))
        c1h     = get_closes(candles.get("1h", []))

        rsi_15m = compute_rsi(c15m if len(c15m) >= 15 else self.price_history)
        ma50    = sma(c1h if len(c1h) >= 50 else self.price_history, ma_period)
        trend   = ("warming_up" if ma50 is None
                   else ("uptrend" if current_price > ma50 else "downtrend"))
        regime          = (market_data or {}).get("regime", "normal")
        total2_bias     = (market_data or {}).get("total2_bias",     "neutral")
        total3_bias     = (market_data or {}).get("total3_bias",     "neutral")
        alt_season      = (market_data or {}).get("alt_season",      False)
        btc_dom_rising  = (market_data or {}).get("btc_dom_rising",  False)
        macro_sentiment = (market_data or {}).get("macro_sentiment", "neutral")
        eth_vs_btc      = (market_data or {}).get("eth_vs_btc",      "neutral")

        # Liquidity grab check on 15m candles
        lq = (detect_liquidity_grab(candles.get("15m", []))
              if lq_enabled and len(candles.get("15m", [])) >= 5
              else {"bullish": False, "bearish": False, "wick_pct": 0.0})

        # Breakout / breakdown on 15m candles
        candles_15m_raw = candles.get("15m", [])
        bo = (breakout_detector(candles_15m_raw)
              if len(candles_15m_raw) >= 22
              else {"breakout": False, "breakdown": False,
                    "false_breakout": False, "false_breakdown": False,
                    "resistance": 0.0, "support": 0.0})

        # Candlestick patterns on 15m (entry confirmation)
        cs = (candlestick_patterns(candles_15m_raw)
              if len(candles_15m_raw) >= 2
              else {"hammer": False, "shooting_star": False,
                    "bullish_engulf": False, "bearish_engulf": False,
                    "bull_marubozu": False, "bear_marubozu": False,
                    "doji": False, "bullish_signals": [], "bearish_signals": []})

        # RSI divergence on 15m (entry signal — separate from 1H MTF confluence)
        rsi_div_15m = (rsi_divergence(c15m)
                       if len(c15m) >= 54
                       else {"bullish": False, "bearish": False})

        # BB squeeze on 15m (Step 3: timing signal)
        # Squeeze → expansion = highest-conviction breakout entry
        _bb_empty = {"bb": None, "squeeze": False, "expanding": False,
                     "was_squeezing": False, "expansion_dir": None,
                     "price_above_mid": False, "at_upper_band": False, "at_lower_band": False}
        bb = (bb_squeeze(c15m) if len(c15m) >= 70 else _bb_empty)

        # BB squeeze on 1H for HTF context
        c1h_full = get_closes(candles.get("1h", []))
        bb_1h = (bb_squeeze(c1h_full) if len(c1h_full) >= 70 else _bb_empty)

        # MACD on 15m (informational — written to heartbeat for dashboard display)
        _macd_empty = {"macd": 0, "signal": 0, "histogram": 0,
                       "crossover_bullish": False, "crossover_bearish": False}
        macd_15m = compute_macd(c15m) if len(c15m) >= 35 else _macd_empty

        # Session breakout detection
        active_session  = current_session()                             # None or {name, emoji, minutes_remaining}
        session_vol_mul = session_volume_multiplier(candles_15m_raw)   # ratio vs 24h avg

        # News context (CryptoPanic, 15m cached, no-op if no token)
        pair_news = await fetch_news(symbol_from_pair(self.asset))
        fg_score        = (market_data or {}).get("fear_greed", {}).get("score") if market_data else None
        in_session_window = (
            sb_enabled
            and active_session is not None
            and session_vol_mul >= sb_vol_min
            and (not sb_require_bb or bb["expanding"])
            and (fg_score is None or sb_min_fg <= fg_score <= sb_max_fg)
        )

        # VWAP on 15m candles (intraday, resets at UTC midnight)
        # price_above → bullish intraday bias; band touches → precision entry/exit levels
        vwap_15m = compute_vwap(candles_15m_raw) if len(candles_15m_raw) >= 5 else None

        # Sideways / range level computation
        candles_1h_raw  = candles.get("1h", [])
        range_lvls   = prev_day_levels(candles_1h_raw)
        or_lvls      = compute_opening_range(candles_15m_raw, sw_or_bars)
        sw_lvls      = compute_swing_levels(candles_1h_raw, sw_swing_lbk)

        # Composite range: PDH/PDL tightened with swing levels
        rng_high = rng_low = None
        if range_lvls:
            rng_high = range_lvls["pdh"]
            rng_low  = range_lvls["pdl"]
            if sw_lvls and sw_lvls["swing_high"] < rng_high:
                rng_high = sw_lvls["swing_high"]
            if sw_lvls and sw_lvls["swing_low"] > rng_low:
                rng_low = sw_lvls["swing_low"]
        elif sw_lvls:
            rng_high = sw_lvls["swing_high"]
            rng_low  = sw_lvls["swing_low"]

        rng_pos = range_position(current_price, rng_high, rng_low) if (rng_high and rng_low) else None

        # ------------------------------------------------------------------
        # Manage open position
        # ------------------------------------------------------------------
        if self.open_position:
            entry_price   = self.open_position["entry_price"]
            pos_direction = self.open_position["direction"]
            usdt_deployed = self.open_position.get("usdt_deployed", self.capital_usdt * position_size_r)
            pos_leverage  = float(self.open_position.get("leverage", 1.0))

            # Price PnL (raw % price move)
            price_pct = ((current_price - entry_price) / entry_price) * (
                1 if pos_direction == "long" else -1
            )
            # Capital PnL = price move amplified by leverage
            pnl_pct = price_pct * pos_leverage

            # Stop/TP triggers: price needs to move stop/tp divided by leverage
            stop_trigger = stop_loss_pct / pos_leverage
            tp_trigger   = take_profit_pct / pos_leverage

            should_close = False
            close_reason = ""

            # 1. Manual exit command
            if self.asset in ctrl.get("manual_exits", []):
                should_close = True
                close_reason = "manual_exit"
                consume_manual_exit(self.asset)
                print(f"  [MANUAL] Force-closing {self.asset}", flush=True)
            # 2. All-stop
            elif all_stop:
                should_close = True
                close_reason = "all_stop"
            # 3. Normal stop/TP — use structural price levels when stored, else % fallback
            else:
                sl_lvl = self.open_position.get("sl_price")
                tp_lvl = self.open_position.get("tp_price")

                # ── Trailing SL ──────────────────────────────────────────
                # Phase 1: move to breakeven once price reaches 1R profit
                # Phase 2: trail at 50% of peak gain beyond breakeven
                trail_cfg     = strategy.get("trailing_sl", {})
                trail_enabled = trail_cfg.get("enabled", True)
                trail_pct     = float(trail_cfg.get("trail_pct", 0.70))

                if trail_enabled and sl_lvl and tp_lvl:
                    entry_p  = float(self.open_position.get("entry_price", 0) or 0)
                    orig_sl  = float(self.open_position.get("orig_sl_price", sl_lvl) or sl_lvl)
                    sl_dist  = abs(entry_p - orig_sl)
                    one_r    = (entry_p + sl_dist) if pos_direction == "long" else (entry_p - sl_dist)

                    if sl_dist > 0:
                        if pos_direction == "long" and current_price >= one_r:
                            peak    = max(float(self.open_position.get("peak_price", current_price) or current_price), current_price)
                            self.open_position["peak_price"] = peak
                            trail_sl = peak - trail_pct * (peak - entry_p)
                            new_sl   = max(trail_sl, entry_p, sl_lvl)
                            if new_sl > sl_lvl:
                                moved = "breakeven" if abs(new_sl - entry_p) < entry_p * 0.0001 else f"trail@{trail_pct:.0%}"
                                print(f"  [TRAIL] {self.asset} SL {sl_lvl:.6g}→{new_sl:.6g} ({moved})", flush=True)
                                self.open_position.setdefault("orig_sl_price", sl_lvl)
                                self.open_position["sl_price"]  = new_sl
                                self.open_position["sl_method"] = moved
                                self._save_position()
                                sl_lvl = new_sl

                        elif pos_direction == "short" and current_price <= one_r:
                            peak    = min(float(self.open_position.get("peak_price", current_price) or current_price), current_price)
                            self.open_position["peak_price"] = peak
                            trail_sl = peak + trail_pct * (entry_p - peak)
                            new_sl   = min(trail_sl, entry_p, sl_lvl)
                            if new_sl < sl_lvl:
                                moved = "breakeven" if abs(new_sl - entry_p) < entry_p * 0.0001 else f"trail@{trail_pct:.0%}"
                                print(f"  [TRAIL] {self.asset} SL {sl_lvl:.6g}→{new_sl:.6g} ({moved})", flush=True)
                                self.open_position.setdefault("orig_sl_price", sl_lvl)
                                self.open_position["sl_price"]  = new_sl
                                self.open_position["sl_method"] = moved
                                self._save_position()
                                sl_lvl = new_sl

                if sl_lvl and tp_lvl:
                    # Dynamic level triggers (structural / Fibonacci / trailed)
                    if pos_direction == "long":
                        if current_price <= sl_lvl:
                            should_close, close_reason = True, "stop_loss"
                        elif current_price >= tp_lvl:
                            should_close, close_reason = True, "take_profit"
                    else:
                        if current_price >= sl_lvl:
                            should_close, close_reason = True, "stop_loss"
                        elif current_price <= tp_lvl:
                            should_close, close_reason = True, "take_profit"
                else:
                    # Legacy fallback for positions opened before v18 (no sl_price/tp_price).
                    if price_pct <= -stop_loss_pct:
                        should_close, close_reason = True, "stop_loss"
                    elif price_pct >= take_profit_pct:
                        should_close, close_reason = True, "take_profit"

            if should_close:
                if is_live():
                    qty = self.open_position.get("qty", 0)
                    if qty > 0:
                        (close_long if pos_direction == "long" else close_short)(self.asset, qty)

                self._record_pnl(pnl_pct, usdt_deployed, close_reason)

                _commission_usdt, _slippage_usdt = self._trade_costs(usdt_deployed, pos_leverage)
                _gross_pnl_usdt = round(pnl_pct * usdt_deployed, 4)
                _net_pnl_usdt   = round(_gross_pnl_usdt - _commission_usdt - _slippage_usdt, 4)

                trade = {
                    **self.open_position,
                    "exit_price":         current_price,
                    "exit_time":          int(time.time()),
                    "pnl_pct":            round(pnl_pct, 6),
                    "pnl_usdt":           _gross_pnl_usdt,
                    "commission_usdt":    _commission_usdt,
                    "slippage_usdt":      _slippage_usdt,
                    "net_pnl_usdt":       _net_pnl_usdt,
                    "close_reason":       close_reason,
                    "rsi_at_exit":        round(rsi_15m, 2),
                    "trend_at_exit":      trend,
                    "regime_at_exit":     regime,
                    "pair_dd_pct":        round(self._pair_drawdown() * 100, 3),
                    "portfolio_dd_pct":   round(PortfolioDrawdown.drawdown() * 100, 3),
                    "strategy_version":   strategy.get("version", "unknown"),
                    "mode":               "live" if is_live() else "paper",
                }
                self.log_trade(trade)
                self.open_position = None
                self._save_position()

                # Email notification
                all_trades = _load_real_trades()
                wins   = sum(1 for t in all_trades if t.get("pnl_pct", 0) > 0)
                losses = len(all_trades) - wins
                stats  = {
                    "total_trades":   len(all_trades),
                    "wins":           wins,
                    "losses":         losses,
                    "win_rate":       round(wins / len(all_trades) * 100, 1) if all_trades else 0,
                    "total_pnl_usdt": round(sum(t.get("pnl_usdt", 0) or 0 for t in all_trades), 4),
                }
                send_trade_email(trade, stats)

                # Reflection cycle — every 5 real closed trades
                # Use a persistent counter file to avoid modulo drift from
                # shutdown trades or simultaneous pair closes on the same tick.
                reflection_every = 5
                _ref_file = STATE_DIR / ".reflection_count"
                try:
                    _ref_count = int(_ref_file.read_text().strip()) if _ref_file.exists() else 0
                except Exception:
                    _ref_count = 0
                _ref_count += 1
                _ref_file.write_text(str(_ref_count))
                if _ref_count % reflection_every == 0:
                    _run_reflection(all_trades, stats)
                    asyncio.create_task(
                        _run_self_improvement(all_trades, list(strategy.keys()))
                    )

                print(f"  ← CLOSE {pos_direction.upper()} {self.asset} @ {current_price:.4f} "
                      f"pnl={pnl_pct:+.3%} (lev={pos_leverage}x) [{close_reason}] "
                      f"pair_dd={self._pair_drawdown():.2%} port_dd={PortfolioDrawdown.drawdown():.2%}",
                      flush=True)

        # ------------------------------------------------------------------
        # STEP 2: Trend recognition — per-pair regime + chart patterns
        # ------------------------------------------------------------------
        candles_4h_raw = candles.get("4h", [])
        # classify_pair_regime needs ≥50 bars for MA50; guard matches the internal requirement
        pair_regime    = classify_pair_regime(candles_4h_raw) if len(candles_4h_raw) >= 50 else "neutral"

        # Chart patterns on 4H (trend-level) and 1H (entry-level)
        _empty_pat = {"patterns": [], "bullish_patterns": [], "bearish_patterns": [],
                      "best_bullish": None, "best_bearish": None}
        patterns_4h = (chart_patterns(candles_4h_raw)
                       if len(candles_4h_raw) >= 30 else _empty_pat)
        patterns_1h = (chart_patterns(candles_1h_raw)
                       if len(candles_1h_raw) >= 30 else _empty_pat)

        # Combined pattern bias across both timeframes
        pat_bull_names = patterns_4h["bullish_patterns"] + patterns_1h["bullish_patterns"]
        pat_bear_names = patterns_4h["bearish_patterns"] + patterns_1h["bearish_patterns"]
        pat_bull = bool(pat_bull_names)   # any bullish pattern on 4H or 1H
        pat_bear = bool(pat_bear_names)   # any bearish pattern on 4H or 1H

        # Strong continuation patterns that should BLOCK counter-trend entries
        # e.g. ascending triangle in bull regime → don't short
        _bull_continuation = {"ascending_triangle", "bull_flag", "ascending_channel",
                               "inv_head_shoulders", "cup_and_handle", "double_bottom", "triple_bottom"}
        _bear_continuation = {"descending_triangle", "bear_flag", "descending_channel",
                               "head_shoulders", "double_top", "triple_top"}
        pat_strong_bull = bool(set(pat_bull_names) & _bull_continuation)
        pat_strong_bear = bool(set(pat_bear_names) & _bear_continuation)

        bull_cfg = strategy.get("bull", {})
        bear_cfg = strategy.get("bear", {})

        # ------------------------------------------------------------------
        # Resolve leverage for new entries
        # Priority: per-pair dashboard override > regime param > strategy config
        # ------------------------------------------------------------------
        lev_overrides    = ctrl.get("leverage_overrides", {})
        default_leverage = float(
            lev_overrides.get(self.asset)
            or regime_params.get("leverage")
            or lev_cfg.get(regime, lev_cfg.get("normal", 1.5))
        )
        entry_leverage = min(default_leverage, max_leverage)

        # ------------------------------------------------------------------
        # Entry logic — skip if all_stop is active
        # ------------------------------------------------------------------
        # Cooldown check: how long since last stop-loss on this pair
        cooldown_secs     = cooldown_minutes * 60
        in_cooldown       = (self._last_stop_loss_ts > 0 and
                             (time.time() - self._last_stop_loss_ts) < cooldown_secs)
        cooldown_remaining = max(0, cooldown_secs - (time.time() - self._last_stop_loss_ts)) if in_cooldown else 0

        if in_cooldown:
            print(f"  [COOLDOWN] {self.asset} — {cooldown_remaining/60:.0f}m remaining after stop-loss", flush=True)

        if all_stop:
            pass  # no new entries while all_stop is active
        elif (not self.open_position
                and not in_cooldown
                and not self._pair_halted(pair_dd_cap)
                and not PortfolioDrawdown.is_halted(portfolio_dd_cap)
                and not DailyLossGuard.is_halted(daily_loss_cap, float(os.getenv("TOTAL_CAPITAL_USDT", "1000")))):

            usdt_to_deploy = self._deploy_usdt(position_size_r, regime_params)
            # qty with leverage: same capital, larger notional position
            qty            = usdt_to_deploy * entry_leverage / current_price if not is_live() else 0.0
            new_direction  = None
            htf_reasons    = []
            lq_note        = ""

            lq_bull = lq_enabled and lq["bullish"]
            lq_bear = lq_enabled and lq["bearish"]

            # Check for dashboard manual entry override first
            pending = next((e for e in ctrl.get("pending_entries", []) if e.get("asset") == self.asset), None)
            if pending:
                new_direction  = pending.get("direction", "long")
                lq_note        = "manual_entry"
                # Allow manual leverage override
                manual_lev = pending.get("leverage")
                if manual_lev:
                    entry_leverage = min(float(manual_lev), max_leverage)
                    qty = usdt_to_deploy * entry_leverage / current_price
                consume_pending_entry(self.asset)
                print(f"  [MANUAL] Entry {new_direction.upper()} {self.asset} lev={entry_leverage}x", flush=True)

            else:
                # ── RANGE EXTREME OVERRIDE ─────────────────────────────────
                # Fires regardless of pair_regime when price is at PDH/PDL extreme
                # AND RSI confirms. This catches overbought/oversold pairs that the
                # regime gate would otherwise miss (e.g. DOGE bull-regime rng=87% RSI=74).
                range_override_fired = False
                range_short_rsi = 70   # RSI must be genuinely overbought to short at range top
                range_long_rsi  = 30   # RSI must be genuinely oversold  to long at range bottom
                if rng_pos is not None and rng_high and rng_low and (rng_high - rng_low) > 0:
                    # Pre-compute candlestick helpers for both directions
                    cs_bear_confirms = bool(cs["bearish_signals"])   # shooting star, bear engulf, bear marubozu, doji
                    cs_bull_blocks   = cs["bull_marubozu"]           # strong bull candle = don't short
                    cs_bull_confirms = bool(cs["bullish_signals"])   # hammer, bull engulf, bull marubozu, doji
                    cs_bear_blocks   = cs["bear_marubozu"]           # strong bear candle = don't long
                    # Range extreme short: RSI overbought + bearish candle + not breaking out
                    if (rng_pos >= (1.0 - sw_entry_pct)
                            and rsi_15m > range_short_rsi
                            and not bo["breakout"]
                            and not cs_bull_blocks):
                        cs_note = f"+cs({','.join(cs['bearish_signals'])})" if cs_bear_confirms else ""
                        lq_note             = f"range_extreme_short({rng_pos:.0%} rsi={rsi_15m:.0f}){cs_note}"
                        _, htf_reasons      = self._htf_signals_short(candles, patterns_4h, patterns_1h, bb_1h, vwap_15m)
                        new_direction       = "short"
                        range_override_fired = True
                    # Range extreme long: RSI oversold + bullish candle + not breaking down
                    elif (rng_pos <= sw_entry_pct
                            and rsi_15m < range_long_rsi
                            and not bo["breakdown"]
                            and not cs_bear_blocks):
                        cs_note = f"+cs({','.join(cs['bullish_signals'])})" if cs_bull_confirms else ""
                        lq_note             = f"range_extreme_long({rng_pos:.0%} rsi={rsi_15m:.0f}){cs_note}"
                        _, htf_reasons      = self._htf_signals_long(candles, patterns_4h, patterns_1h, bb_1h, vwap_15m)
                        new_direction       = "long"
                        range_override_fired = True

                if not range_override_fired:
                    # Candlestick conviction helpers (reused across regimes)
                    cs_bull_ok = bool(cs["bullish_signals"]) and not cs["bear_marubozu"]
                    cs_bear_ok = bool(cs["bearish_signals"]) and not cs["bull_marubozu"]
                    cs_bull_str = f"+cs({','.join(cs['bullish_signals'])})" if cs["bullish_signals"] else ""
                    cs_bear_str = f"+cs({','.join(cs['bearish_signals'])})" if cs["bearish_signals"] else ""

                    if pair_regime == "bull":
                        _pair_thr     = pair_thresholds.get(self.asset, {})
                        bull_long_thr = _pair_thr.get("long_threshold",
                                            bull_cfg.get("long_threshold", 35))
                        if in_session_window:
                            bull_long_thr = min(45, bull_long_thr + sb_rsi_relax)
                        # 1. Breakout long — confirmed by bull marubozu or engulfing
                        if bo["breakout"] and trend == "uptrend" and not cs["bear_marubozu"]:
                            lq_note        = f"breakout(res={bo['resistance']:.6g}){cs_bull_str}"
                            _, htf_reasons = self._htf_signals_long(candles, patterns_4h, patterns_1h, bb_1h, vwap_15m)
                            new_direction  = "long"
                        # 2. RSI dip buy — require bullish candle confirmation (hammer/engulf)
                        elif (rsi_15m < bull_long_thr and trend == "uptrend"
                                and (cs_bull_ok or lq_bull)):
                            lq_note        = cs_bull_str
                            _, htf_reasons = self._htf_signals_long(candles, patterns_4h, patterns_1h, bb_1h, vwap_15m)
                            new_direction  = "long"
                        # 3. Short signals — only fire when not breaking out + bearish candle
                        elif not bo["breakout"]:
                            if bo["false_breakout"] and cs_bear_ok:
                                lq_note        = f"false_breakout(res={bo['resistance']:.6g}){cs_bear_str}"
                                _, htf_reasons = self._htf_signals_short(candles, patterns_4h, patterns_1h, bb_1h, vwap_15m)
                                new_direction  = "short"
                            elif rsi_div_15m["bearish"] and cs_bear_ok:
                                lq_note        = f"rsi_div_bearish_15m{cs_bear_str}"
                                _, htf_reasons = self._htf_signals_short(candles, patterns_4h, patterns_1h, bb_1h, vwap_15m)
                                new_direction  = "short"
                            elif bull_cfg.get("short_on_lq_only", True) and lq_bear and cs_bear_ok:
                                lq_note        = f"lq_grab_bear(wick={lq['wick_pct']:.3%}){cs_bear_str}"
                                _, htf_reasons = self._htf_signals_short(candles, patterns_4h, patterns_1h, bb_1h, vwap_15m)
                                new_direction  = "short"

                    elif pair_regime == "bear":
                        _pair_thr      = pair_thresholds.get(self.asset, {})
                        bear_short_thr = _pair_thr.get("short_threshold",
                                             bear_cfg.get("short_threshold", 60))
                        if in_session_window:
                            bear_short_thr = max(55, bear_short_thr - sb_rsi_relax)
                        # 1. Breakdown short — confirmed by bear marubozu or engulfing
                        if bo["breakdown"] and trend == "downtrend" and not cs["bull_marubozu"]:
                            lq_note        = f"breakdown(sup={bo['support']:.6g}){cs_bear_str}"
                            _, htf_reasons = self._htf_signals_short(candles, patterns_4h, patterns_1h, bb_1h, vwap_15m)
                            new_direction  = "short"
                        # 2. RSI bounce short — require bearish candle confirmation
                        elif (rsi_15m > bear_short_thr and trend == "downtrend"
                                and (cs_bear_ok or lq_bear)):
                            lq_note        = cs_bear_str
                            _, htf_reasons = self._htf_signals_short(candles, patterns_4h, patterns_1h, bb_1h, vwap_15m)
                            new_direction  = "short"
                        # 3. Long signals — only fire when not breaking down + bullish candle
                        elif not bo["breakdown"]:
                            if bo["false_breakdown"] and cs_bull_ok:
                                lq_note        = f"false_breakdown(sup={bo['support']:.6g}){cs_bull_str}"
                                _, htf_reasons = self._htf_signals_long(candles, patterns_4h, patterns_1h, bb_1h, vwap_15m)
                                new_direction  = "long"
                            elif rsi_div_15m["bullish"] and cs_bull_ok:
                                lq_note        = f"rsi_div_bullish_15m{cs_bull_str}"
                                _, htf_reasons = self._htf_signals_long(candles, patterns_4h, patterns_1h, bb_1h, vwap_15m)
                                new_direction  = "long"
                            elif bear_cfg.get("long_on_lq_only", True) and lq_bull and cs_bull_ok:
                                lq_note        = f"lq_grab_bull(wick={lq['wick_pct']:.3%}){cs_bull_str}"
                                _, htf_reasons = self._htf_signals_long(candles, patterns_4h, patterns_1h, bb_1h, vwap_15m)
                                new_direction  = "long"

                    elif pair_regime == "sideways":
                        if rng_pos is not None and rng_high and rng_low:
                            rng_total = rng_high - rng_low
                            if rng_total > 0:
                                # Range bottom long — need bullish candle, no strong bear momentum
                                if rng_pos <= sw_entry_pct and (cs_bull_ok or lq_bull):
                                    lq_note        = f"range_bottom({rng_pos:.1%}){cs_bull_str}"
                                    _, htf_reasons = self._htf_signals_long(candles, patterns_4h, patterns_1h, bb_1h, vwap_15m)
                                    new_direction  = "long"
                                # Range top short — need bearish candle, no strong bull momentum
                                elif rng_pos >= (1.0 - sw_entry_pct) and (cs_bear_ok or lq_bear):
                                    lq_note        = f"range_top({rng_pos:.1%}){cs_bear_str}"
                                    _, htf_reasons = self._htf_signals_short(candles, patterns_4h, patterns_1h, bb_1h, vwap_15m)
                                    new_direction  = "short"

                    else:
                        # neutral / warming up fallback
                        if (rsi_div_15m["bullish"] or (rsi_15m < 30 and (trend == "uptrend" or lq_bull))) and cs_bull_ok:
                            if rsi_div_15m["bullish"]:
                                lq_note = f"rsi_div_bullish_15m{cs_bull_str}"
                            elif lq_bull and trend != "uptrend":
                                lq_note = f"lq_grab(wick={lq['wick_pct']:.3%}){cs_bull_str}"
                            _, htf_reasons = self._htf_signals_long(candles, patterns_4h, patterns_1h, bb_1h, vwap_15m)
                            new_direction  = "long"
                        elif (rsi_div_15m["bearish"] or (rsi_15m > 70 and (trend == "downtrend" or lq_bear))) and cs_bear_ok:
                            if rsi_div_15m["bearish"]:
                                lq_note = f"rsi_div_bearish_15m{cs_bear_str}"
                            elif lq_bear and trend != "downtrend":
                                lq_note = f"lq_grab(wick={lq['wick_pct']:.3%}){cs_bear_str}"
                            _, htf_reasons = self._htf_signals_short(candles, patterns_4h, patterns_1h, bb_1h, vwap_15m)
                            new_direction  = "short"

            # ------------------------------------------------------------------
            # STEP 3b: Dynamic SL/TP + R:R gate
            # Compute structural levels before entering.
            # Trades with R:R < 1.0 are blocked; 1.0–1.5 = warn; 2.0+ = high quality.
            # ------------------------------------------------------------------
            _dyn_levels: dict | None = None
            if new_direction:
                try:
                    _dyn_levels = dynamic_levels(
                        direction    = new_direction,
                        entry_price  = current_price,
                        candles_15m  = candles_15m_raw,
                        candles_1h   = candles_1h_raw,
                        rng_high     = rng_high,
                        rng_low      = rng_low,
                        min_rr       = 1.5,      # block R:R < 1:1.5
                        max_sl_pct   = 0.025,    # hard cap: 2.5% → 5% worst-case at 2x lev
                        min_sl_pct   = 0.003,
                        sl_buffer_pct= 0.003,
                        patterns_1h  = patterns_1h,
                        patterns_4h  = patterns_4h,
                    )
                    if _dyn_levels and not _dyn_levels["valid"]:
                        print(
                            f"  [R:R] {self.asset} {new_direction.upper()} blocked — "
                            f"R:R={_dyn_levels['rr_ratio']:.2f} < 1.0 "
                            f"(SL={_dyn_levels['sl_pct']:.2f}% via {_dyn_levels['sl_method']} "
                            f"TP={_dyn_levels['tp_pct']:.2f}% via {_dyn_levels['tp_method']})",
                            flush=True,
                        )
                        new_direction = None
                    elif _dyn_levels:
                        rr_tag = f"R:R={_dyn_levels['rr_ratio']:.2f}"
                        print(
                            f"  [LEVELS] {self.asset} {new_direction.upper()} "
                            f"SL={_dyn_levels['sl_price']:.6g} ({_dyn_levels['sl_pct']:.2f}% {_dyn_levels['sl_method']}) "
                            f"TP={_dyn_levels['tp_price']:.6g} ({_dyn_levels['tp_pct']:.2f}% {_dyn_levels['tp_method']}) "
                            f"{rr_tag}",
                            flush=True,
                        )
                except Exception as _lv_err:
                    print(f"  [LEVELS] {self.asset} level calc failed: {_lv_err}", flush=True)
                    _dyn_levels = None

            # Capture pre-gate direction for shadow trade logging
            _pre_gate_direction = new_direction

            # ------------------------------------------------------------------
            # STEP 3a: BB squeeze gate
            # If bands are in a tight squeeze with NO expansion yet — skip.
            # A squeeze without expansion = consolidation still in progress.
            # A squeeze WITH expansion = highest-conviction breakout entry.
            # ------------------------------------------------------------------
            if new_direction and bb["squeeze"] and not bb["expanding"] and pair_regime != "sideways":
                # Pure squeeze — no directional signal yet, skip entry.
                # Exception: sideways regime *expects* a tight BB; blocking range entries
                # there creates a catch-22 (squeeze IS the sideways condition).
                print(f"  [BB SQUEEZE] {self.asset} {new_direction.upper()} held — "
                      f"bands squeezing (bw={bb['bb']['bandwidth']:.4f}), waiting for expansion",
                      flush=True)
                new_direction = None
            elif new_direction and bb["expanding"] and bb["expansion_dir"]:
                # Expansion firing — check it agrees with intended direction
                if ((new_direction == "long"  and bb["expansion_dir"] == "down") or
                        (new_direction == "short" and bb["expansion_dir"] == "up")):
                    # BB expanding the wrong way — block entry
                    print(f"  [BB DIRECTION] {self.asset} {new_direction.upper()} blocked — "
                          f"BB expanding {bb['expansion_dir']} (opposite direction)",
                          flush=True)
                    new_direction = None

            # ------------------------------------------------------------------
            # STEP 2b: Pattern blocking — strong continuation patterns block
            # counter-trend entries (e.g. ascending triangle → no shorts)
            # Exception: RSI extreme overrides pattern block — at RSI <25 or >75
            # the bearish/bullish move is already exhausted; patterns are stale.
            # ------------------------------------------------------------------
            rsi_extreme_long  = rsi_15m is not None and rsi_15m < 25
            rsi_extreme_short = rsi_15m is not None and rsi_15m > 75
            if new_direction == "short" and pat_strong_bull and not rsi_extreme_short:
                print(f"  [PATTERN BLOCK] {self.asset} short blocked — bullish pattern active: {pat_bull_names}", flush=True)
                new_direction = None
            elif new_direction == "long" and pat_strong_bear and not rsi_extreme_long:
                print(f"  [PATTERN BLOCK] {self.asset} long blocked — bearish pattern active: {pat_bear_names}", flush=True)
                new_direction = None
            elif new_direction and (rsi_extreme_long or rsi_extreme_short):
                if pat_strong_bear and new_direction == "long":
                    print(f"  [PATTERN OVERRIDE] {self.asset} RSI {rsi_15m:.0f} extreme — ignoring bearish patterns {pat_bear_names}", flush=True)
                elif pat_strong_bull and new_direction == "short":
                    print(f"  [PATTERN OVERRIDE] {self.asset} RSI {rsi_15m:.0f} extreme — ignoring bullish patterns {pat_bull_names}", flush=True)

            # ------------------------------------------------------------------
            # NEWS caution layer — adjusts MTF requirement based on news sentiment
            # News is a tiebreaker, not a hard gate:
            #   Headwind (bearish news + long, or bullish news + short) → +1 HTF required
            #   Tailwind (bullish news + long, or bearish news + short) → -1 HTF required
            # Exception: market has already priced the news if RSI is at extremes
            # AND BB is expanding — in that case news adjustment is ignored.
            # Session window also overrides (move already happened at open).
            # ------------------------------------------------------------------
            if new_direction and pair_news:
                news_label    = pair_news.get("label", "no_data")
                rsi_extreme   = (new_direction == "long"  and rsi_15m < 32) or \
                                (new_direction == "short" and rsi_15m > 68)
                market_priced = rsi_extreme and bb["expanding"]

                if not market_priced and not in_session_window and news_label != "no_data":
                    is_headwind = (new_direction == "long"  and news_label == "bearish") or \
                                  (new_direction == "short" and news_label == "bullish")
                    is_tailwind = (new_direction == "long"  and news_label == "bullish") or \
                                  (new_direction == "short" and news_label == "bearish")

                    if is_headwind:
                        mtf_require = mtf_require + 1
                        print(f"  [NEWS] {self.asset} {new_direction.upper()} — "
                              f"{news_label} news headwind, MTF raised to {mtf_require}",
                              flush=True)
                    elif is_tailwind:
                        mtf_require = max(0, mtf_require - 1)
                        print(f"  [NEWS] {self.asset} {new_direction.upper()} — "
                              f"{news_label} news tailwind, MTF lowered to {mtf_require}",
                              flush=True)

            # ------------------------------------------------------------------
            # MTF soft gate — require ≥N HTF confirmations before entering
            # Exceptions: manual entries, liquidity grabs, session breakouts,
            # and sideways range entries (already gated by position-in-range +
            # candle direction — adding MTF creates a catch-22 in flat markets)
            # ------------------------------------------------------------------
            if new_direction and pair_regime == "sideways":
                if mtf_require > 0:
                    print(f"  [MTF SIDEWAYS] {self.asset} MTF dropped 0 (range entry self-confirming)", flush=True)
                mtf_require = 0

            if new_direction and mtf_require > 0:
                is_manual  = lq_note == "manual_entry"
                is_lq      = lq_note and "lq_grab" in lq_note
                is_session = in_session_window   # BB expansion + volume spike already confirmed
                if not is_manual and not is_lq and not is_session:
                    if len(htf_reasons) < mtf_require:
                        print(
                            f"  [MTF GATE] {self.asset} {new_direction.upper()} blocked — "
                            f"only {len(htf_reasons)}/{mtf_require} HTF signals "
                            f"({', '.join(htf_reasons) or 'none'})",
                            flush=True,
                        )
                        new_direction = None   # block entry

            # Shadow trade: if a valid setup existed but was blocked by a gate,
            # log what would have been entered so we can track its forward outcome.
            if not new_direction and _pre_gate_direction and _dyn_levels and _dyn_levels.get("valid"):
                try:
                    record_shadow_trade(
                        asset       = self.asset,
                        direction   = _pre_gate_direction,
                        entry_price = current_price,
                        sl_price    = _dyn_levels["sl_price"],
                        tp_price    = _dyn_levels["tp_price"],
                        signal      = lq_note or "signal",
                    )
                except Exception:
                    pass

            if new_direction:
                # Recalculate capital based on conviction score
                _active_patterns = pat_bull_names if new_direction == "long" else pat_bear_names
                usdt_to_deploy, _conv_score = self._conviction_capital(
                    strategy, htf_reasons, lq_note, rsi_15m, _active_patterns, new_direction
                )
                qty = usdt_to_deploy * entry_leverage / current_price if not is_live() else 0.0

                if is_live():
                    notional = usdt_to_deploy * entry_leverage
                    order = (open_long if new_direction == "long" else open_short)(self.asset, notional)
                    qty   = float(order.get("filled") or order.get("amount") or (notional / current_price))

                self.open_position = {
                    "asset":              self.asset,
                    "direction":          new_direction,
                    "entry_price":        current_price,
                    "entry_time":         int(time.time()),
                    "position_size_r":    position_size_r,
                    "usdt_deployed":      usdt_to_deploy,
                    "leverage":           entry_leverage,
                    "notional_usdt":      round(usdt_to_deploy * entry_leverage, 2),
                    "qty":                qty,
                    # Dynamic SL/TP (structural levels)
                    "sl_price":           _dyn_levels["sl_price"]  if _dyn_levels else None,
                    "tp_price":           _dyn_levels["tp_price"]  if _dyn_levels else None,
                    "sl_pct":             _dyn_levels["sl_pct"]    if _dyn_levels else stop_loss_pct * 100,
                    "tp_pct":             _dyn_levels["tp_pct"]    if _dyn_levels else take_profit_pct * 100,
                    "rr_ratio":           _dyn_levels["rr_ratio"]  if _dyn_levels else None,
                    "sl_method":          _dyn_levels["sl_method"] if _dyn_levels else "pct_fallback",
                    "tp_method":          _dyn_levels["tp_method"] if _dyn_levels else "pct_fallback",
                    # Legacy pct fields kept for compatibility
                    "stop_loss_pct":      (_dyn_levels["sl_pct"]   if _dyn_levels else stop_loss_pct * 100),
                    "take_profit_pct":    (_dyn_levels["tp_pct"]   if _dyn_levels else take_profit_pct * 100),
                    "rsi_at_entry":       round(rsi_15m, 2),
                    "trend_at_entry":     trend,
                    "regime_at_entry":    regime,
                    "pair_regime":        pair_regime,
                    "is_sideways":        is_sideways,
                    "range_pos_at_entry": round(rng_pos, 4) if rng_pos is not None else None,
                    "range_high":         round(rng_high, 6) if rng_high else None,
                    "range_low":          round(rng_low,  6) if rng_low  else None,
                    "htf_signals":        htf_reasons,
                    "lq_grab":            lq_note or None,
                    "session_breakout":   active_session["name"] if in_session_window else None,
                    "news_label":         pair_news.get("label", "no_data") if pair_news else "no_data",
                    "news_sentiment":     pair_news.get("sentiment", 0.0)   if pair_news else 0.0,
                    "active_features":    active_feature_labels(),
                    "strategy_version":   strategy.get("version", "unknown"),
                    "mode":               "live" if is_live() else "paper",
                    "conviction_score":   _conv_score,
                }
                self._save_position()

                # Shadow log — what fixed $50 would have deployed vs dynamic
                try:
                    _fixed_capital = float(strategy.get("conviction_sizing", {}).get("medium", 50))
                    _shadow_entry = {
                        "ts":               int(time.time()),
                        "asset":            self.asset,
                        "direction":        new_direction,
                        "entry_price":      current_price,
                        "conviction_score": _conv_score,
                        "dynamic_usdt":     usdt_to_deploy,
                        "fixed_usdt":       _fixed_capital,
                        "sl_price":         _dyn_levels["sl_price"] if _dyn_levels else None,
                        "tp_price":         _dyn_levels["tp_price"] if _dyn_levels else None,
                    }
                    _cap_shadow_file = STATE_DIR / "capital_shadow.jsonl"
                    with open(_cap_shadow_file, "a") as _f:
                        _f.write(json.dumps(_shadow_entry) + "\n")
                except Exception:
                    pass

                send_entry_notification({
                    **self.open_position,
                    "conviction_score": _conv_score,
                    "mtf_signals":      htf_reasons,
                })
                signals_str = ", ".join(htf_reasons + ([lq_note] if lq_note else []))
                lvl_str = ""
                if _dyn_levels:
                    lvl_str = (f" | SL={_dyn_levels['sl_price']:.6g}({_dyn_levels['sl_pct']:.2f}%,"
                               f"{_dyn_levels['sl_method']}) "
                               f"TP={_dyn_levels['tp_price']:.6g}({_dyn_levels['tp_pct']:.2f}%,"
                               f"{_dyn_levels['tp_method']}) "
                               f"R:R={_dyn_levels['rr_ratio']:.2f}")
                print(f"  → ENTRY {new_direction.upper()} {self.asset} @ {current_price:.4f} "
                      f"lev={entry_leverage}x notional=${usdt_to_deploy*entry_leverage:.0f} "
                      f"rsi={rsi_15m:.1f}{lvl_str} | {signals_str}", flush=True)

        # Heartbeat
        self.write_heartbeat("ok", {
            "rsi_15m":       round(rsi_15m, 2),
            "price":         current_price,
            "trend":         trend,
            "ma50":          round(ma50, 6) if ma50 else None,
            "regime":        regime,
            "pair_regime":   pair_regime,
            "is_sideways":   is_sideways,
            "rng_high":      round(rng_high, 6) if rng_high else None,
            "rng_low":       round(rng_low,  6) if rng_low  else None,
            "rng_pos":       round(rng_pos,  4) if rng_pos is not None else None,
            "pdh":           range_lvls["pdh"] if range_lvls else None,
            "pdl":           range_lvls["pdl"] if range_lvls else None,
            "or_high":       or_lvls["or_high"] if or_lvls else None,
            "or_low":        or_lvls["or_low"]  if or_lvls else None,
            "lq_bullish":    lq["bullish"],
            "lq_bearish":    lq["bearish"],
            "bo_breakout":   bo["breakout"],
            "bo_breakdown":  bo["breakdown"],
            "bo_false_breakout":  bo["false_breakout"],
            "bo_false_breakdown": bo["false_breakdown"],
            "rsi_div_bull":  rsi_div_15m["bullish"],
            "rsi_div_bear":  rsi_div_15m["bearish"],
            "cs_bullish":    cs["bullish_signals"],
            "cs_bearish":    cs["bearish_signals"],
            "patterns_bull": pat_bull_names,
            "patterns_bear": pat_bear_names,
            "bb_squeeze":    bb["squeeze"],
            "bb_expanding":  bb["expanding"],
            "bb_dir":        bb["expansion_dir"],
            "bb_bandwidth":  bb["bb"]["bandwidth"] if bb["bb"] else None,
            "bb_pct_b":      bb["bb"]["pct_b"]     if bb["bb"] else None,
            # MACD 15m (informational — not a hard gate, displayed on dashboard)
            "macd_15m":           round(macd_15m["macd"],      6) if macd_15m and macd_15m.get("macd") is not None else None,
            "macd_signal_15m":    round(macd_15m["signal"],    6) if macd_15m and macd_15m.get("signal") is not None else None,
            "macd_hist_15m":      round(macd_15m["histogram"], 6) if macd_15m and macd_15m.get("histogram") is not None else None,
            "macd_bull_15m":      macd_15m.get("crossover_bullish",  False) if macd_15m else False,
            "macd_bear_15m":      macd_15m.get("crossover_bearish",  False) if macd_15m else False,
            # News (CryptoPanic sentiment, 15m cached)
            "news_label":         pair_news.get("label",            "no_data"),
            "news_sentiment":     pair_news.get("sentiment",        0.0),
            "news_posts":         pair_news.get("post_count",       0),
            "news_headlines":        pair_news.get("headlines",        []),
            "news_headline_urls":    pair_news.get("headline_urls",    []),
            "news_headline_sources": pair_news.get("headline_sources", []),
            # Session breakout
            "session_name":       active_session["name"]              if active_session else None,
            "session_emoji":      active_session["emoji"]             if active_session else None,
            "session_mins_left":  active_session["minutes_remaining"] if active_session else None,
            "session_vol_mul":    session_vol_mul,
            "session_active":     in_session_window,
            # VWAP (Step 3 — intraday institutional anchor)
            "vwap":          vwap_15m["vwap"]          if vwap_15m else None,
            "vwap_upper_1":  vwap_15m["upper_1"]       if vwap_15m else None,
            "vwap_lower_1":  vwap_15m["lower_1"]       if vwap_15m else None,
            "vwap_upper_2":  vwap_15m["upper_2"]       if vwap_15m else None,
            "vwap_lower_2":  vwap_15m["lower_2"]       if vwap_15m else None,
            "vwap_above":    vwap_15m["price_above"]   if vwap_15m else None,
            "vwap_pct_dev":  vwap_15m["pct_from_vwap"] if vwap_15m else None,
            "vwap_at_upper": vwap_15m.get("at_upper_1") or vwap_15m.get("at_upper_2") if vwap_15m else None,
            "vwap_at_lower": vwap_15m.get("at_lower_1") or vwap_15m.get("at_lower_2") if vwap_15m else None,
            # Active position levels (for dashboard display)
            "pos_sl_price":  self.open_position.get("sl_price")  if self.open_position else None,
            "pos_tp_price":  self.open_position.get("tp_price")  if self.open_position else None,
            "pos_rr_ratio":  self.open_position.get("rr_ratio")  if self.open_position else None,
            "pos_sl_method": self.open_position.get("sl_method") if self.open_position else None,
            "pos_tp_method": self.open_position.get("tp_method") if self.open_position else None,
            # Total2 / Total3 macro layer
            "total2_bias":       total2_bias,
            "total3_bias":       total3_bias,
            "alt_season":        alt_season,
            "btc_dom_rising":    btc_dom_rising,
            "macro_sentiment":   macro_sentiment,
            "eth_vs_btc":        eth_vs_btc,
            "all_stop":           all_stop,
            "entry_leverage":     entry_leverage,
            "regime_params":      regime_params,
            "in_cooldown":        in_cooldown,
            "cooldown_remaining_m": round(cooldown_remaining / 60, 1) if in_cooldown else 0,
            "daily_pnl":          DailyLossGuard.summary(),
        })
        self.consecutive_failures = 0
        mode_tag   = "[LIVE]" if is_live() else "[paper]"
        pos_tag    = self.open_position["direction"] if self.open_position else "none"
        rng_tag    = f" rng={rng_pos:.0%}" if (rng_pos is not None) else ""
        stop_tag     = " ⛔STOP" if all_stop else ""
        cooldown_tag = f" 🕐{cooldown_remaining/60:.0f}m" if in_cooldown else ""
        vwap_tag = ""
        if vwap_15m:
            vwap_side = "↑VWAP" if vwap_15m["price_above"] else "↓VWAP"
            vwap_tag  = f" {vwap_side}({vwap_15m['pct_from_vwap']:+.2f}%)"
        macro_tag = f" T2={total2_bias[0].upper()} T3={total3_bias[0].upper()}"
        if alt_season:
            macro_tag += " 🌙ALT"
        elif btc_dom_rising:
            macro_tag += " 🟠BTC↑"
        print(
            f"{mode_tag} [{self.asset}] price={current_price:.4f} rsi={rsi_15m:.1f} "
            f"pair={pair_regime} macro={regime}{macro_tag} lev={entry_leverage}x{rng_tag}{vwap_tag} "
            f"lq={'B' if lq['bullish'] else 'S' if lq['bearish'] else '-'} "
            f"dd={self._pair_drawdown():.1%} day=${DailyLossGuard._daily_pnl:+.2f} "
            f"pos={pos_tag}{stop_tag}{cooldown_tag}",
            flush=True,
        )

        # Shadow resolution + downtime trigger (self-throttled, non-blocking)
        self._tick_maintenance(current_price, _load_real_trades())

    # ------------------------------------------------------------------
    # Downtime + shadow trade maintenance (called each tick, self-throttled)
    # ------------------------------------------------------------------

    def _tick_maintenance(self, current_price: float, all_trades: list[dict]):
        """Lightweight per-tick work: resolve shadow trades, trigger downtime check."""
        # Resolve any shadow trades that hit TP/SL
        try:
            resolve_shadow_trades({self.asset: current_price})
        except Exception:
            pass

        # Downtime check: fires at most once per IDLE_THRESHOLD_HOURS, async
        try:
            if should_run_downtime(all_trades):
                asyncio.create_task(_run_downtime_check(all_trades))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def run(self, market_queue: asyncio.Queue = None):
        mode_tag = "[LIVE]" if is_live() else "[paper]"
        print(f"Booting {mode_tag} [{self.asset}] capital={self.capital_usdt} USDT", flush=True)
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self._load_position()   # restore any open position that survived a restart

        # Register active code-level features at boot so trades are tagged correctly
        try:
            strategy = load_strategy()
            sb_cfg   = strategy.get("session_breakout", {})
            if sb_cfg.get("enabled", True):
                register_feature("session_breakout", "code",
                                 "Relaxed RSI+MTF gate during Asia/London/US open windows",
                                 rsi_relax=sb_cfg.get("rsi_relax_pts", 5))
            register_feature("news_caution", "code",
                             "News sentiment adjusts MTF requirement per trade direction")
            register_feature("macd_15m_scoring", "code",
                             "MACD 15m crossover direction-aware scanner conviction bonus")
            register_feature("per_pair_rsi_tuning", "code",
                             "RSI thresholds auto-tuned per pair via reflection cycle")
            register_feature("downtime_engine", "code",
                             "Idle diagnosis, OOS backtest, shadow trading during quiet markets")
        except Exception as e:
            print(f"[features] boot registration failed: {e}", flush=True)

        if market_queue is not None:
            while True:
                market_data = await market_queue.get()
                try:
                    await self.tick(market_data=market_data)
                except Exception as e:
                    self._handle_error(e)
                finally:
                    market_queue.task_done()
        else:
            while True:
                try:
                    await self.tick()
                except Exception as e:
                    self._handle_error(e)
                await asyncio.sleep(LOOP_INTERVAL)

    def _handle_error(self, e: Exception):
        self.consecutive_failures += 1
        print(f"[ERROR] [{self.asset}] tick failed ({self.consecutive_failures}): {e}", flush=True)
        self.write_heartbeat("error", {"error": str(e)})
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print(f"[CIRCUIT BREAKER] [{self.asset}] halting.", flush=True)
            raise SystemExit(1)
