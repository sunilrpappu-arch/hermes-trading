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
    prev_day_levels,
    opening_range as compute_opening_range,
    swing_levels as compute_swing_levels,
    range_position,
    classify_pair_regime,
)
from hermes_trading.adapters.candles import closes as get_closes, highs as get_highs, lows as get_lows
from hermes_trading.notify import send_trade_email, send_reflection_notification

STATE_DIR      = Path(os.getenv("STATE_DIR", Path(__file__).parent.parent / "state"))
TRADES_FILE    = STATE_DIR / "trades.jsonl"
STRATEGY_FILE  = STATE_DIR / "strategy.yaml"
DD_FILE        = STATE_DIR / "drawdown.json"   # portfolio-level drawdown state
CONTROLS_FILE  = STATE_DIR / "controls.json"   # manual override commands


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

CAPITAL_PER_PAIR_USDT = float(os.getenv("CAPITAL_PER_PAIR_USDT", "200"))

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
        Uses total_capital as the base so early losses aren't inflated."""
        if cls._peak_usdt > 0:
            # Normal case: peak exists, measure from peak
            dd_from_peak = (cls._peak_usdt - cls._current_usdt) / cls._peak_usdt
        else:
            # No profitable trades yet — measure loss directly against total capital
            dd_from_peak = abs(min(cls._current_usdt, 0)) / max(cls._total_capital, 1)
        return max(0.0, dd_from_peak)

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
    "version": "14",
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
    # MTF is informational — logged on entry but no longer a hard gate
    "mtf": {
        "enabled":         True,
        "require_signals": 0,
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

    summary = (
        f"🔄 <b>Reflection #{n // 5}</b> — last {len(recent)} trades\n\n"
        f"W/L: {wins_recent}W / {losses_recent}L   "
        f"PnL: {'+' if pnl_recent >= 0 else ''}${pnl_recent:.4f}\n"
        f"Best:  {best.get('asset','?')} {best_pct:+.2f}%\n"
        f"Worst: {worst.get('asset','?')} {worst_pct:+.2f}%\n"
        f"Reasons: {reasons_str}\n"
        f"Regimes: {regimes_str}\n\n"
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

    # ------------------------------------------------------------------
    # HTF signal evaluation
    # ------------------------------------------------------------------

    def _htf_signals_long(self, candles: dict,
                          patterns_4h: dict = None, patterns_1h: dict = None) -> tuple[int, list[str]]:
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
        return len(confirmed), confirmed

    def _htf_signals_short(self, candles: dict,
                           patterns_4h: dict = None, patterns_1h: dict = None) -> tuple[int, list[str]]:
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
        return len(confirmed), confirmed

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    async def tick(self, market_data: dict = None):
        strategy        = load_strategy()
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
        regime  = (market_data or {}).get("regime", "normal")

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
            # 3. Normal stop/TP
            elif price_pct <= -stop_trigger:
                should_close = True
                close_reason = "stop_loss"
            elif price_pct >= tp_trigger:
                should_close = True
                close_reason = "take_profit"

            if should_close:
                if is_live():
                    qty = self.open_position.get("qty", 0)
                    if qty > 0:
                        (close_long if pos_direction == "long" else close_short)(self.asset, qty)

                self._record_pnl(pnl_pct, usdt_deployed, close_reason)

                trade = {
                    **self.open_position,
                    "exit_price":         current_price,
                    "exit_time":          int(time.time()),
                    "pnl_pct":            round(pnl_pct, 6),
                    "pnl_usdt":           round(pnl_pct * usdt_deployed, 4),
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
                all_trades = []
                try:
                    for line in TRADES_FILE.read_text().splitlines():
                        if line.strip():
                            import json as _json
                            all_trades.append(_json.loads(line))
                except Exception:
                    pass
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

                # Reflection cycle — every 5 closed trades
                reflection_every = 5
                if len(all_trades) % reflection_every == 0:
                    _run_reflection(all_trades, stats)

                print(f"  ← CLOSE {pos_direction.upper()} {self.asset} @ {current_price:.4f} "
                      f"pnl={pnl_pct:+.3%} (lev={pos_leverage}x) [{close_reason}] "
                      f"pair_dd={self._pair_drawdown():.2%} port_dd={PortfolioDrawdown.drawdown():.2%}",
                      flush=True)

        # ------------------------------------------------------------------
        # STEP 2: Trend recognition — per-pair regime + chart patterns
        # ------------------------------------------------------------------
        candles_4h_raw = candles.get("4h", [])
        pair_regime    = classify_pair_regime(candles_4h_raw) if len(candles_4h_raw) >= 30 else "neutral"

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
                    # Range extreme short: RSI overbought + bearish candle + not breaking out
                    cs_bear_confirms = bool(cs["bearish_signals"])   # shooting star, bear engulf, bear marubozu, doji
                    cs_bull_blocks   = cs["bull_marubozu"]           # strong bull candle = don't short
                    if (rng_pos >= (1.0 - sw_entry_pct)
                            and rsi_15m > range_short_rsi
                            and not bo["breakout"]
                            and not cs_bull_blocks):
                        cs_note = f"+cs({','.join(cs['bearish_signals'])})" if cs_bear_confirms else ""
                        lq_note             = f"range_extreme_short({rng_pos:.0%} rsi={rsi_15m:.0f}){cs_note}"
                        _, htf_reasons      = self._htf_signals_short(candles, patterns_4h, patterns_1h)
                        new_direction       = "short"
                        range_override_fired = True

                    # Range extreme long: RSI oversold + bullish candle + not breaking down
                    cs_bull_confirms = bool(cs["bullish_signals"])   # hammer, bull engulf, bull marubozu, doji
                    cs_bear_blocks   = cs["bear_marubozu"]           # strong bear candle = don't long
                    elif (rng_pos <= sw_entry_pct
                            and rsi_15m < range_long_rsi
                            and not bo["breakdown"]
                            and not cs_bear_blocks):
                        cs_note = f"+cs({','.join(cs['bullish_signals'])})" if cs_bull_confirms else ""
                        lq_note             = f"range_extreme_long({rng_pos:.0%} rsi={rsi_15m:.0f}){cs_note}"
                        _, htf_reasons      = self._htf_signals_long(candles, patterns_4h, patterns_1h)
                        new_direction       = "long"
                        range_override_fired = True

                if not range_override_fired:
                    # Candlestick conviction helpers (reused across regimes)
                    cs_bull_ok = bool(cs["bullish_signals"]) and not cs["bear_marubozu"]
                    cs_bear_ok = bool(cs["bearish_signals"]) and not cs["bull_marubozu"]
                    cs_bull_str = f"+cs({','.join(cs['bullish_signals'])})" if cs["bullish_signals"] else ""
                    cs_bear_str = f"+cs({','.join(cs['bearish_signals'])})" if cs["bearish_signals"] else ""

                    if pair_regime == "bull":
                        bull_long_thr = bull_cfg.get("long_threshold", 35)
                        # 1. Breakout long — confirmed by bull marubozu or engulfing
                        if bo["breakout"] and trend == "uptrend" and not cs["bear_marubozu"]:
                            lq_note        = f"breakout(res={bo['resistance']:.6g}){cs_bull_str}"
                            _, htf_reasons = self._htf_signals_long(candles, patterns_4h, patterns_1h)
                            new_direction  = "long"
                        # 2. RSI dip buy — require bullish candle confirmation (hammer/engulf)
                        elif (rsi_15m < bull_long_thr and trend == "uptrend"
                                and (cs_bull_ok or lq_bull)):
                            lq_note        = cs_bull_str
                            _, htf_reasons = self._htf_signals_long(candles, patterns_4h, patterns_1h)
                            new_direction  = "long"
                        # 3. Short signals — only fire when not breaking out + bearish candle
                        elif not bo["breakout"]:
                            if bo["false_breakout"] and cs_bear_ok:
                                lq_note        = f"false_breakout(res={bo['resistance']:.6g}){cs_bear_str}"
                                _, htf_reasons = self._htf_signals_short(candles, patterns_4h, patterns_1h)
                                new_direction  = "short"
                            elif rsi_div_15m["bearish"] and cs_bear_ok:
                                lq_note        = f"rsi_div_bearish_15m{cs_bear_str}"
                                _, htf_reasons = self._htf_signals_short(candles, patterns_4h, patterns_1h)
                                new_direction  = "short"
                            elif bull_cfg.get("short_on_lq_only", True) and lq_bear and cs_bear_ok:
                                lq_note        = f"lq_grab_bear(wick={lq['wick_pct']:.3%}){cs_bear_str}"
                                _, htf_reasons = self._htf_signals_short(candles, patterns_4h, patterns_1h)
                                new_direction  = "short"

                    elif pair_regime == "bear":
                        bear_short_thr = bear_cfg.get("short_threshold", 60)
                        # 1. Breakdown short — confirmed by bear marubozu or engulfing
                        if bo["breakdown"] and trend == "downtrend" and not cs["bull_marubozu"]:
                            lq_note        = f"breakdown(sup={bo['support']:.6g}){cs_bear_str}"
                            _, htf_reasons = self._htf_signals_short(candles, patterns_4h, patterns_1h)
                            new_direction  = "short"
                        # 2. RSI bounce short — require bearish candle confirmation
                        elif (rsi_15m > bear_short_thr and trend == "downtrend"
                                and (cs_bear_ok or lq_bear)):
                            lq_note        = cs_bear_str
                            _, htf_reasons = self._htf_signals_short(candles, patterns_4h, patterns_1h)
                            new_direction  = "short"
                        # 3. Long signals — only fire when not breaking down + bullish candle
                        elif not bo["breakdown"]:
                            if bo["false_breakdown"] and cs_bull_ok:
                                lq_note        = f"false_breakdown(sup={bo['support']:.6g}){cs_bull_str}"
                                _, htf_reasons = self._htf_signals_long(candles, patterns_4h, patterns_1h)
                                new_direction  = "long"
                            elif rsi_div_15m["bullish"] and cs_bull_ok:
                                lq_note        = f"rsi_div_bullish_15m{cs_bull_str}"
                                _, htf_reasons = self._htf_signals_long(candles, patterns_4h, patterns_1h)
                                new_direction  = "long"
                            elif bear_cfg.get("long_on_lq_only", True) and lq_bull and cs_bull_ok:
                                lq_note        = f"lq_grab_bull(wick={lq['wick_pct']:.3%}){cs_bull_str}"
                                _, htf_reasons = self._htf_signals_long(candles, patterns_4h, patterns_1h)
                                new_direction  = "long"

                    elif pair_regime == "sideways":
                        if rng_pos is not None and rng_high and rng_low:
                            rng_total = rng_high - rng_low
                            if rng_total > 0:
                                # Range bottom long — need bullish candle, no strong bear momentum
                                if rng_pos <= sw_entry_pct and (cs_bull_ok or lq_bull):
                                    lq_note        = f"range_bottom({rng_pos:.1%}){cs_bull_str}"
                                    _, htf_reasons = self._htf_signals_long(candles, patterns_4h, patterns_1h)
                                    new_direction  = "long"
                                # Range top short — need bearish candle, no strong bull momentum
                                elif rng_pos >= (1.0 - sw_entry_pct) and (cs_bear_ok or lq_bear):
                                    lq_note        = f"range_top({rng_pos:.1%}){cs_bear_str}"
                                    _, htf_reasons = self._htf_signals_short(candles, patterns_4h, patterns_1h)
                                    new_direction  = "short"

                    else:
                        # neutral / warming up fallback
                        if (rsi_div_15m["bullish"] or (rsi_15m < 30 and (trend == "uptrend" or lq_bull))) and cs_bull_ok:
                            if rsi_div_15m["bullish"]:
                                lq_note = f"rsi_div_bullish_15m{cs_bull_str}"
                            elif lq_bull and trend != "uptrend":
                                lq_note = f"lq_grab(wick={lq['wick_pct']:.3%}){cs_bull_str}"
                            _, htf_reasons = self._htf_signals_long(candles, patterns_4h, patterns_1h)
                            new_direction  = "long"
                        elif (rsi_div_15m["bearish"] or (rsi_15m > 70 and (trend == "downtrend" or lq_bear))) and cs_bear_ok:
                            if rsi_div_15m["bearish"]:
                                lq_note = f"rsi_div_bearish_15m{cs_bear_str}"
                            elif lq_bear and trend != "downtrend":
                                lq_note = f"lq_grab(wick={lq['wick_pct']:.3%}){cs_bear_str}"
                            _, htf_reasons = self._htf_signals_short(candles, patterns_4h, patterns_1h)
                            new_direction  = "short"

            # ------------------------------------------------------------------
            # STEP 2b: Pattern blocking — strong continuation patterns block
            # counter-trend entries (e.g. ascending triangle → no shorts)
            # ------------------------------------------------------------------
            if new_direction == "short" and pat_strong_bull:
                print(f"  [PATTERN BLOCK] {self.asset} short blocked — bullish pattern active: {pat_bull_names}", flush=True)
                new_direction = None
            elif new_direction == "long" and pat_strong_bear:
                print(f"  [PATTERN BLOCK] {self.asset} long blocked — bearish pattern active: {pat_bear_names}", flush=True)
                new_direction = None

            # ------------------------------------------------------------------
            # MTF soft gate — require ≥N HTF confirmations before entering
            # Exceptions: manual entries and liquidity grabs bypass the gate
            # (high-conviction price-action signals are self-confirming)
            # ------------------------------------------------------------------
            if new_direction and mtf_require > 0:
                is_manual  = lq_note == "manual_entry"
                is_lq      = lq_note and "lq_grab" in lq_note
                if not is_manual and not is_lq:
                    if len(htf_reasons) < mtf_require:
                        print(
                            f"  [MTF GATE] {self.asset} {new_direction.upper()} blocked — "
                            f"only {len(htf_reasons)}/{mtf_require} HTF signals "
                            f"({', '.join(htf_reasons) or 'none'})",
                            flush=True,
                        )
                        new_direction = None   # block entry

            if new_direction:
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
                    "stop_loss_pct":      stop_loss_pct * 100,
                    "take_profit_pct":    take_profit_pct * 100,
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
                    "strategy_version":   strategy.get("version", "unknown"),
                    "mode":               "live" if is_live() else "paper",
                }
                self._save_position()
                signals_str = ", ".join(htf_reasons + ([lq_note] if lq_note else []))
                print(f"  → ENTRY {new_direction.upper()} {self.asset} @ {current_price:.4f} "
                      f"lev={entry_leverage}x notional=${usdt_to_deploy*entry_leverage:.0f} "
                      f"rsi={rsi_15m:.1f} | {signals_str}", flush=True)

        # Heartbeat
        self.write_heartbeat("ok", {
            "rsi_15m":       round(rsi_15m, 2),
            "price":         current_price,
            "trend":         trend,
            "ma50":          round(ma50, 6) if ma50 else None,
            "regime":        regime,
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
        print(
            f"{mode_tag} [{self.asset}] price={current_price:.4f} rsi={rsi_15m:.1f} "
            f"pair={pair_regime} macro={regime} lev={entry_leverage}x{rng_tag} "
            f"lq={'B' if lq['bullish'] else 'S' if lq['bearish'] else '-'} "
            f"dd={self._pair_drawdown():.1%} day=${DailyLossGuard._daily_pnl:+.2f} "
            f"pos={pos_tag}{stop_tag}{cooldown_tag}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def run(self, market_queue: asyncio.Queue = None):
        mode_tag = "[LIVE]" if is_live() else "[paper]"
        print(f"Booting {mode_tag} [{self.asset}] capital={self.capital_usdt} USDT", flush=True)
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self._load_position()   # restore any open position that survived a restart

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
