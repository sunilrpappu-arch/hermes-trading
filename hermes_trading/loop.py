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
    prev_day_levels,
    opening_range as compute_opening_range,
    swing_levels as compute_swing_levels,
    range_position,
)
from hermes_trading.adapters.candles import closes as get_closes, highs as get_highs, lows as get_lows
from hermes_trading.notify import send_trade_email

STATE_DIR      = Path(os.getenv("STATE_DIR", Path(__file__).parent.parent / "state"))
TRADES_FILE    = STATE_DIR / "trades.jsonl"
STRATEGY_FILE  = STATE_DIR / "strategy.yaml"
DD_FILE        = STATE_DIR / "drawdown.json"   # portfolio-level drawdown state

LOOP_INTERVAL          = int(os.getenv("LOOP_INTERVAL_SECONDS", "60"))
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
    _peak_usdt:    float = 0.0
    _current_usdt: float = 0.0
    _halted:       bool  = False

    @classmethod
    def record_trade(cls, pnl_usdt: float, total_capital: float):
        cls._current_usdt += pnl_usdt
        if cls._current_usdt > cls._peak_usdt:
            cls._peak_usdt = cls._current_usdt
        cls._save(total_capital)

    @classmethod
    def drawdown(cls) -> float:
        """Current drawdown as fraction of total capital (0.05 = 5%)."""
        if cls._peak_usdt <= 0:
            return abs(min(cls._current_usdt, 0)) / max(cls._peak_usdt, 1)
        return max(0.0, (cls._peak_usdt - cls._current_usdt) / cls._peak_usdt)

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


async def fetch_with_retry(fn, *args, **kwargs):
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            await asyncio.sleep(2 ** attempt)


DEFAULT_STRATEGY = {
    "version": "08",
    "entry": {
        "indicator":       "rsi",
        "direction":       "both",
        "long_threshold":  25,
        "short_threshold": 65,
    },
    "trend_filter": {"enabled": True, "ma_period": 50},
    "liquidity_grab": {
        "enabled":    True,
        "wick_ratio": 2.0,     # lower wick must be >= 2x candle body
        "sweep_pct":  0.002,   # must sweep prior swing by at least 0.2%
        "overrides_trend_filter": True,  # a confirmed grab bypasses trend filter
    },
    "mtf": {
        "enabled":         True,
        "require_signals": 1,
    },
    "drawdown": {
        "per_pair_cap":   0.10,   # pause a pair if it loses 10% of its allocated capital
        "portfolio_cap":  0.08,   # halt all entries if portfolio drops 8%
    },
    "take_profit_pct":  3.0,
    "stop_loss_pct":    1.8,
    "position_size_r":  0.05,
    "sideways": {
        "adx_threshold":   20.0,  # market is ranging below this ADX value
        "range_entry_pct": 0.20,  # enter within bottom / top 20% of PDH-PDL range
        "or_bars":         4,     # opening range = first 4 × 15m candles of UTC day
        "swing_lookback":  20,    # bars to look back for swing high / low
    },
}


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

    # ------------------------------------------------------------------
    # Drawdown helpers
    # ------------------------------------------------------------------

    def _record_pnl(self, pnl_pct: float, usdt_deployed: float):
        pnl_usdt = pnl_pct * usdt_deployed
        self._realised_pnl_usdt += pnl_usdt
        if self._realised_pnl_usdt > self._pair_peak_usdt:
            self._pair_peak_usdt = self._realised_pnl_usdt
        PortfolioDrawdown.record_trade(pnl_usdt, self.capital_usdt)

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

    def _htf_signals_long(self, candles: dict) -> tuple[int, list[str]]:
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
        return len(confirmed), confirmed

    def _htf_signals_short(self, candles: dict) -> tuple[int, list[str]]:
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
        return len(confirmed), confirmed

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    async def tick(self, market_data: dict = None):
        strategy        = load_strategy()
        entry_cfg       = strategy.get("entry", {})
        trend_cfg       = strategy.get("trend_filter", {})
        mtf_cfg         = strategy.get("mtf", {})
        lq_cfg          = strategy.get("liquidity_grab", {})
        dd_cfg          = strategy.get("drawdown", {})

        direction_mode    = entry_cfg.get("direction", "both")
        long_threshold    = entry_cfg.get("long_threshold", 25)
        short_threshold   = entry_cfg.get("short_threshold", 65)
        trend_enabled     = trend_cfg.get("enabled", True)
        ma_period         = trend_cfg.get("ma_period", 50)
        mtf_enabled       = mtf_cfg.get("enabled", True)
        require_signals   = mtf_cfg.get("require_signals", 1)
        lq_enabled        = lq_cfg.get("enabled", True)
        lq_overrides_trend = lq_cfg.get("overrides_trend_filter", True)
        pair_dd_cap       = dd_cfg.get("per_pair_cap", 0.10)
        portfolio_dd_cap  = dd_cfg.get("portfolio_cap", 0.08)

        sw_cfg            = strategy.get("sideways", {})
        sw_entry_pct      = sw_cfg.get("range_entry_pct", 0.20)
        sw_or_bars        = sw_cfg.get("or_bars",         4)
        sw_swing_lbk      = sw_cfg.get("swing_lookback",  20)

        regime_params     = (market_data or {}).get("regime_params", {})
        is_sideways       = (market_data or {}).get("is_sideways", False)
        stop_loss_pct     = regime_params.get("stop_loss_pct",   strategy.get("stop_loss_pct",   1.8)) / 100
        take_profit_pct   = regime_params.get("take_profit_pct", strategy.get("take_profit_pct", 3.0)) / 100
        position_size_r   = strategy.get("position_size_r", 0.05)

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

        # Sideways / range level computation
        # These are computed every tick but only used when is_sideways is True
        candles_1h_raw  = candles.get("1h", [])
        candles_15m_raw = candles.get("15m", [])
        range_lvls   = prev_day_levels(candles_1h_raw)      # {pdh, pdl, pdo, pdc, pd_range}
        or_lvls      = compute_opening_range(candles_15m_raw, sw_or_bars)  # {or_high, or_low, or_mid}
        sw_lvls      = compute_swing_levels(candles_1h_raw, sw_swing_lbk)  # {swing_high, swing_low}

        # Composite range: use PDH/PDL as primary, tighten with swing levels if closer
        rng_high = rng_low = None
        if range_lvls:
            rng_high = range_lvls["pdh"]
            rng_low  = range_lvls["pdl"]
            # If swing high is lower than PDH — stronger resistance (use it)
            if sw_lvls and sw_lvls["swing_high"] < rng_high:
                rng_high = sw_lvls["swing_high"]
            # If swing low is higher than PDL — stronger support (use it)
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
            pnl_pct = ((current_price - entry_price) / entry_price) * (
                1 if pos_direction == "long" else -1
            )

            should_close = False
            close_reason = ""
            if pnl_pct <= -stop_loss_pct:
                should_close = True
                close_reason = "stop_loss"
            elif pnl_pct >= take_profit_pct:
                should_close = True
                close_reason = "take_profit"

            if should_close:
                if is_live():
                    qty = self.open_position.get("qty", 0)
                    if qty > 0:
                        (close_long if pos_direction == "long" else close_short)(self.asset, qty)

                self._record_pnl(pnl_pct, usdt_deployed)

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
                send_trade_email(trade, {
                    "total_trades":   len(all_trades),
                    "wins":           wins,
                    "losses":         losses,
                    "win_rate":       round(wins / len(all_trades) * 100, 1) if all_trades else 0,
                    "total_pnl_usdt": round(sum(t.get("pnl_usdt", 0) for t in all_trades), 4),
                })

                print(f"  ← CLOSE {pos_direction.upper()} {self.asset} @ {current_price:.4f} "
                      f"pnl={pnl_pct:+.3%} [{close_reason}] "
                      f"pair_dd={self._pair_drawdown():.2%} port_dd={PortfolioDrawdown.drawdown():.2%}",
                      flush=True)

        # ------------------------------------------------------------------
        # Entry logic
        # ------------------------------------------------------------------
        if (not self.open_position
                and trend != "warming_up"
                and not self._pair_halted(pair_dd_cap)
                and not PortfolioDrawdown.is_halted(portfolio_dd_cap)):

            usdt_to_deploy = self._deploy_usdt(position_size_r, regime_params)
            qty            = 0.0
            new_direction  = None
            htf_reasons    = []
            lq_note        = ""

            # --- Long ---
            if direction_mode in ("long", "both"):
                rsi_ok   = rsi_15m < long_threshold
                trend_ok = (not trend_enabled) or trend == "uptrend"
                lq_ok    = lq_enabled and lq["bullish"] and lq_overrides_trend

                if rsi_ok and (trend_ok or lq_ok):
                    if lq_ok and not trend_ok:
                        lq_note = f"lq_grab(wick={lq['wick_pct']:.3%})"

                    if mtf_enabled:
                        htf_count, htf_reasons = self._htf_signals_long(candles)
                        if htf_count >= require_signals:
                            new_direction = "long"
                    else:
                        new_direction  = "long"
                        htf_reasons    = ["mtf_disabled"]

            # --- Short ---
            if new_direction is None and direction_mode in ("short", "both"):
                rsi_ok   = rsi_15m > short_threshold
                trend_ok = (not trend_enabled) or trend == "downtrend"
                lq_ok    = lq_enabled and lq["bearish"] and lq_overrides_trend

                if rsi_ok and (trend_ok or lq_ok):
                    if lq_ok and not trend_ok:
                        lq_note = f"lq_grab(wick={lq['wick_pct']:.3%})"

                    if mtf_enabled:
                        htf_count, htf_reasons = self._htf_signals_short(candles)
                        if htf_count >= require_signals:
                            new_direction = "short"
                    else:
                        new_direction  = "short"
                        htf_reasons    = ["mtf_disabled"]

            # --- Sideways / range mean-reversion ---
            # Only enters when regime is sideways AND price is at range extreme.
            # PDH-PDL defines the range; bottom/top 20% are the entry zones.
            # Still requires MTF MACD confirmation to filter false bounces.
            if new_direction is None and is_sideways and rng_pos is not None and rng_high and rng_low:
                rng_total = rng_high - rng_low

                if rng_total > 0:
                    # Long: price in bottom 20% of range (near support)
                    if direction_mode in ("long", "both") and rng_pos <= sw_entry_pct:
                        range_tag = f"range_bottom({rng_pos:.1%} of {rng_total:.4f})"
                        if mtf_enabled:
                            htf_count, htf_reasons = self._htf_signals_long(candles)
                            if htf_count >= require_signals:
                                new_direction = "long"
                                lq_note       = range_tag
                        else:
                            new_direction = "long"
                            htf_reasons   = ["mtf_disabled"]
                            lq_note       = range_tag

                    # Short: price in top 20% of range (near resistance)
                    elif direction_mode in ("short", "both") and rng_pos >= (1.0 - sw_entry_pct):
                        range_tag = f"range_top({rng_pos:.1%} of {rng_total:.4f})"
                        if mtf_enabled:
                            htf_count, htf_reasons = self._htf_signals_short(candles)
                            if htf_count >= require_signals:
                                new_direction = "short"
                                lq_note       = range_tag
                        else:
                            new_direction = "short"
                            htf_reasons   = ["mtf_disabled"]
                            lq_note       = range_tag

            if new_direction:
                if is_live():
                    order = (open_long if new_direction == "long" else open_short)(self.asset, usdt_to_deploy)
                    qty   = float(order.get("filled") or order.get("amount") or (usdt_to_deploy / current_price))

                self.open_position = {
                    "asset":            self.asset,
                    "direction":        new_direction,
                    "entry_price":      current_price,
                    "entry_time":       int(time.time()),
                    "position_size_r":  position_size_r,
                    "usdt_deployed":    usdt_to_deploy,
                    "qty":              qty,
                    "rsi_at_entry":     round(rsi_15m, 2),
                    "trend_at_entry":   trend,
                    "regime_at_entry":  regime,
                    "is_sideways":      is_sideways,
                    "range_pos_at_entry": round(rng_pos, 4) if rng_pos is not None else None,
                    "range_high":       round(rng_high, 6) if rng_high else None,
                    "range_low":        round(rng_low,  6) if rng_low  else None,
                    "htf_signals":      htf_reasons,
                    "lq_grab":          lq_note or None,
                    "strategy_version": strategy.get("version", "unknown"),
                    "mode":             "live" if is_live() else "paper",
                }
                signals_str = ", ".join(htf_reasons + ([lq_note] if lq_note else []))
                print(f"  → ENTRY {new_direction.upper()} {self.asset} @ {current_price:.4f} "
                      f"rsi={rsi_15m:.1f} | {signals_str}", flush=True)

        # Heartbeat
        self.write_heartbeat("ok", {
            "rsi_15m":     round(rsi_15m, 2),
            "price":       current_price,
            "trend":       trend,
            "ma50":        round(ma50, 6) if ma50 else None,
            "regime":      regime,
            "is_sideways": is_sideways,
            "rng_high":    round(rng_high, 6) if rng_high else None,
            "rng_low":     round(rng_low,  6) if rng_low  else None,
            "rng_pos":     round(rng_pos,  4) if rng_pos is not None else None,
            "pdh":         range_lvls["pdh"] if range_lvls else None,
            "pdl":         range_lvls["pdl"] if range_lvls else None,
            "or_high":     or_lvls["or_high"] if or_lvls else None,
            "or_low":      or_lvls["or_low"]  if or_lvls else None,
            "lq_bullish":  lq["bullish"],
            "lq_bearish":  lq["bearish"],
        })
        self.consecutive_failures = 0
        mode_tag   = "[LIVE]" if is_live() else "[paper]"
        pos_tag    = self.open_position["direction"] if self.open_position else "none"
        sw_tag     = f" rng={rng_pos:.0%}" if (is_sideways and rng_pos is not None) else ""
        print(
            f"{mode_tag} [{self.asset}] price={current_price:.4f} rsi={rsi_15m:.1f} "
            f"trend={trend} regime={regime}{sw_tag} "
            f"lq={'B' if lq['bullish'] else 'S' if lq['bearish'] else '-'} "
            f"dd={self._pair_drawdown():.1%} pos={pos_tag}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def run(self, market_queue: asyncio.Queue = None):
        mode_tag = "[LIVE]" if is_live() else "[paper]"
        print(f"Booting {mode_tag} [{self.asset}] capital={self.capital_usdt} USDT", flush=True)
        STATE_DIR.mkdir(parents=True, exist_ok=True)

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
