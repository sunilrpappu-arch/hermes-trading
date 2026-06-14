"""
Coordinator — single entry point for the Hermes worker.

Every LOOP_INTERVAL seconds:
  1. Fetch prices for all active pairs (one batched request)
  2. Fetch candles for all active pairs (cached, 3 timeframes)
  3. Push MarketData to each loop's queue

Every SCAN_INTERVAL seconds:
  1. Detect volatility regime from BTC 1H candles
  2. Run pair scanner → pick top N from universe
  3. Spin up new loops / retire stale ones
"""
import argparse
import asyncio
import shutil
import yaml
from datetime import datetime, timezone
from pathlib import Path
import os
import json as _json_mod  # noqa — used for alternative.me cache

STATE_DIR    = Path(os.getenv("STATE_DIR", Path(__file__).parent.parent / "state"))
DEFAULTS_DIR = Path(__file__).parent.parent / "state_defaults"

LOOP_INTERVAL = int(os.getenv("LOOP_INTERVAL_SECONDS", "60"))
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL_SECONDS", str(30 * 60)))  # 30 min

DEFAULT_UNIVERSE = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "HYPE/USDT"]


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_state():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "history").mkdir(parents=True, exist_ok=True)

    # goal.yaml and strategy.yaml are config — always overwrite from defaults
    _ALWAYS_OVERWRITE = {"goal.yaml", "strategy.yaml"}
    if DEFAULTS_DIR.exists():
        for src in DEFAULTS_DIR.glob("*.yaml"):
            dst = STATE_DIR / src.name
            if not dst.exists() or src.name in _ALWAYS_OVERWRITE:
                shutil.copy(src, dst)
                print(f"[bootstrap] copied {src.name} → {dst}", flush=True)
    else:
        print(f"[bootstrap] state_defaults not found — using inline defaults", flush=True)

    for fname in ("trades.jsonl", "hypotheses.jsonl"):
        f = STATE_DIR / fname
        if not f.exists():
            f.touch()

    # Always delete drawdown.json on startup — it gets recalculated from live trades.
    # Stale/corrupt values (e.g. from the early divide-by-1 bug) cause false DD halts.
    import json as _json
    dd_file = STATE_DIR / "drawdown.json"
    if dd_file.exists():
        try:
            dd_file.unlink()
            print(f"[bootstrap] cleared drawdown.json — will recalculate from trades", flush=True)
        except Exception:
            pass


def load_goal() -> dict:
    goal_file = STATE_DIR / "goal.yaml"
    if goal_file.exists():
        with open(goal_file) as f:
            return yaml.safe_load(f) or {}
    return {}


def load_universe() -> list[str]:
    goal = load_goal()
    return goal.get("universe") or goal.get("pairs") or DEFAULT_UNIVERSE


def load_total_capital() -> float:
    env_val = os.getenv("TOTAL_CAPITAL_USDT")
    if env_val:
        return float(env_val)
    return float(load_goal().get("total_capital_usdt", 1000.0))


# ---------------------------------------------------------------------------
# Market data bundle
# ---------------------------------------------------------------------------

async def build_market_data(pairs: list[str], regime_info: dict) -> dict[str, dict]:
    """
    Fetch prices + candles for all pairs in one coordinated sweep.
    Returns {asset: MarketData} where MarketData is passed to each loop tick.
    """
    from hermes_trading.adapters.price import fetch_all as fetch_prices_all
    from hermes_trading.adapters.candles import fetch_all_multi

    # Parallel fetch: prices + candles
    prices_task  = fetch_prices_all(pairs)
    candles_task = fetch_all_multi(pairs, intervals=["15m", "1h", "4h"], limit=100)
    prices, all_candles = await asyncio.gather(prices_task, candles_task)

    market_data = {}
    for pair in pairs:
        pd = prices.get(pair, {})
        market_data[pair] = {
            "asset":           pair,
            "price":           pd.get("price", 0.0),
            "timestamp":       pd.get("timestamp", 0),
            "candles":         all_candles.get(pair, {}),
            "regime":          regime_info.get("regime", "normal"),
            "is_sideways":     regime_info.get("is_sideways", False),
            "adx":             regime_info.get("adx"),
            "regime_params":   regime_info,
            # Total2 / Total3 macro signals
            "total2_bias":     regime_info.get("total2_bias", "neutral"),
            "total3_bias":     regime_info.get("total3_bias", "neutral"),
            "alt_season":      regime_info.get("alt_season", False),
            "btc_dom_rising":  regime_info.get("btc_dom_rising", False),
            "macro_sentiment": regime_info.get("macro_sentiment", "neutral"),
            "eth_vs_btc":      regime_info.get("eth_vs_btc", "neutral"),
        }
    return market_data


# ---------------------------------------------------------------------------
# Main coordinator
# ---------------------------------------------------------------------------

async def run_all(universe: list[str], total_capital: float, force_pairs: list[str] = None):
    from hermes_trading.loop import TradingLoop
    from hermes_trading.volatility import detect as detect_regime
    from hermes_trading.scanner import scan as scan_pairs, fetch_universe
    from hermes_trading.adapters.candles import fetch as fetch_candles

    loops:  dict[str, TradingLoop]   = {}
    queues: dict[str, asyncio.Queue] = {}
    tasks:  dict[str, asyncio.Task]  = {}
    regime_info: dict = {"regime": "normal", "max_pairs": 5, "capital_per_pair": 200.0,
                         "position_size_r": 0.05, "stop_loss_pct": 1.8, "take_profit_pct": 3.0}
    active_pairs: list[str] = []

    async def rescan():
        nonlocal regime_info, active_pairs

        # 1. Detect volatility regime — BTC (Layer 1) + ETH (Total2) + alt basket (Total3)
        _macro_pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT"]
        _macro_candles = await asyncio.gather(
            *[fetch_candles(p, "1h", 100) for p in _macro_pairs],
            return_exceptions=True,
        )
        btc_candles_1h  = _macro_candles[0] if not isinstance(_macro_candles[0], Exception) else []
        eth_candles_1h  = _macro_candles[1] if not isinstance(_macro_candles[1], Exception) else []
        _total3_symbols = ["SOL/USDT", "BNB/USDT", "ADA/USDT"]
        alt_candles_1h  = {
            sym: _macro_candles[i + 2]
            for i, sym in enumerate(_total3_symbols)
            if not isinstance(_macro_candles[i + 2], Exception)
        }
        regime_info = await detect_regime(btc_candles_1h, eth_candles_1h, alt_candles_1h)

        # 2. Select pairs — fetch live universe from Binance, fall back to goal.yaml list
        if force_pairs:
            selected = force_pairs[:regime_info["max_pairs"]]
        else:
            goal      = load_goal()
            live_universe = await fetch_universe(filters=goal.get("universe_filters", {}))
            selected  = await scan_pairs(live_universe, regime_info["max_pairs"], regime_info.get("vol", 0.02), regime_info)

        # 3. Retire loops no longer selected — but NEVER retire a pair with an open position
        for pair in list(loops.keys()):
            if pair not in selected:
                loop = loops[pair]
                if loop.open_position:
                    # Keep the loop alive until position closes
                    print(f"[coordinator] keeping {pair} — open position, will retire after close", flush=True)
                    selected.append(pair)   # add back so market data keeps flowing
                    continue
                print(f"[coordinator] retiring {pair}", flush=True)
                if tasks.get(pair) and not tasks[pair].done():
                    tasks[pair].cancel()
                # Clean up heartbeat file so it doesn't linger on dashboard
                safe = pair.replace("/", "_")
                hb_file = STATE_DIR / f"heartbeat_{safe}.json"
                try:
                    hb_file.unlink(missing_ok=True)
                except Exception:
                    pass
                del loops[pair]
                del queues[pair]
                del tasks[pair]

        # 4. Spin up new loops
        capital_per = regime_info.get("capital_per_pair", total_capital / max(len(selected), 1))
        for pair in selected:
            if pair not in loops:
                print(f"[coordinator] activating {pair} @ {capital_per:.0f} USDT", flush=True)
                loops[pair]  = TradingLoop(asset=pair, capital_usdt=capital_per)
                queues[pair] = asyncio.Queue(maxsize=1)
                tasks[pair]  = asyncio.create_task(loops[pair].run(market_queue=queues[pair]))

        active_pairs = list(loops.keys())
        print(f"[coordinator] active pairs: {active_pairs} | regime: {regime_info.get('label','?')}", flush=True)

    # ── alternative.me Fear & Greed (real index, 10-min cache) ─────────────
    _altme_cache: dict = {}

    async def _fetch_real_fear_greed() -> dict | None:
        """Fetch the industry-standard Fear & Greed index from alternative.me (free, no key)."""
        import time as _t
        if _altme_cache.get("expires", 0) > _t.time():
            return _altme_cache.get("data")
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    "https://api.alternative.me/fng/?limit=1",
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as r:
                    raw = await r.json(content_type=None)
            entry = raw["data"][0]
            score = int(entry["value"])
            label = entry["value_classification"]
            emoji = ("😱" if score < 20 else "😰" if score < 35
                     else "😐" if score < 50 else "😊" if score < 65
                     else "🤑" if score < 80 else "🚀")
            result = {"score": score, "label": label, "emoji": emoji}
            _altme_cache["data"]    = result
            _altme_cache["expires"] = _t.time() + 600   # 10-min cache
            return result
        except Exception:
            return _altme_cache.get("data")   # serve stale on error

    async def coordinator_loop():
        import json as _json
        from hermes_trading import black_swan as bs

        last_scan    = 0.0
        _prev_prices: dict[str, float] = {}   # {asset: price} from previous tick
        _bs_alerted  = False                   # avoid spam-alerting on sustained critical

        def _read_heartbeats_dict() -> list[dict]:
            """Quick read of all fresh heartbeat files → list of dicts for sentiment calc."""
            import time as _time
            now   = _time.time()
            items = []
            for hf in STATE_DIR.glob("heartbeat_*.json"):
                try:
                    d   = _json.loads(hf.read_text())
                    ts  = d.get("timestamp")
                    if ts:
                        age = now - datetime.fromisoformat(
                            ts.replace("Z", "+00:00")).timestamp()
                        if age > 600:   # >10 min stale → skip
                            continue
                    # Build heartbeat dict expected by black_swan.py
                    hb = {
                        "asset":      d.get("asset", ""),
                        "price":      d.get("price", 0),
                        "rsi_15m":    d.get("rsi_15m"),
                        "trend":      d.get("trend"),
                        "rng_pos":    d.get("rng_pos"),
                        "vwap_above": d.get("vwap_above"),
                    }
                    items.append(hb)
                except Exception:
                    pass
            return items

        def _write_sentiment(fg: dict, swan: dict, real_fg: dict | None = None):
            sf = STATE_DIR / "sentiment.json"
            try:
                sf.write_text(_json.dumps({
                    "fear_greed": fg,
                    "real_fear_greed": real_fg,
                    "black_swan": {
                        "level":   swan["level"],
                        "events":  swan["events"],
                        "action":  swan["action"],
                        "all_stop": swan["all_stop"],
                    },
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }))
            except Exception:
                pass

        while True:
            now = asyncio.get_event_loop().time()

            # Rescan on first run and every SCAN_INTERVAL
            if now - last_scan >= SCAN_INTERVAL:
                try:
                    await rescan()
                    last_scan = now
                except Exception as e:
                    print(f"[coordinator] rescan failed: {e}", flush=True)

            if active_pairs:
                try:
                    market_data = await build_market_data(active_pairs, regime_info)
                    for pair, data in market_data.items():
                        if pair in queues:
                            try:
                                queues[pair].put_nowait(data)
                            except asyncio.QueueFull:
                                pass

                    # ── Fear/Greed + Black Swan ──────────────────────────────
                    try:
                        heartbeats = _read_heartbeats_dict()
                        btc_vol    = regime_info.get("vol", None)
                        fg_score   = bs.fear_greed_score(heartbeats, btc_vol)

                        # Build prev_prices for flash crash detection
                        swan = bs.check(
                            heartbeats  = heartbeats,
                            regime_info = regime_info,
                            prev_prices = _prev_prices,
                            fg_score    = fg_score,
                        )

                        # Update prev_prices for next tick
                        for pair, md in market_data.items():
                            asset = pair.split("/")[0]
                            if md.get("price"):
                                _prev_prices[asset] = md["price"]
                                _prev_prices[pair]  = md["price"]

                        real_fg = await _fetch_real_fear_greed()
                        _write_sentiment(fg_score, swan, real_fg)

                        score_lbl = f"{fg_score['emoji']} {fg_score['label']} {fg_score['score']}/100"
                        print(f"[sentinel] {score_lbl} | {swan['level'].upper()}: {swan['action']}", flush=True)

                        # Trigger all_stop on critical events
                        if swan["all_stop"] and not _bs_alerted:
                            controls_file = STATE_DIR / "controls.json"
                            try:
                                ctrl = _json.loads(controls_file.read_text()) if controls_file.exists() else {}
                            except Exception:
                                ctrl = {}
                            if not ctrl.get("all_stop"):
                                ctrl["all_stop"] = True
                                controls_file.write_text(_json.dumps(ctrl, indent=2))
                                print(f"[sentinel] 🚨 CRITICAL — all_stop set", flush=True)
                                try:
                                    from hermes_trading.notify import _send_telegram
                                    _send_telegram(swan["message"])
                                except Exception:
                                    pass
                            _bs_alerted = True
                        elif not swan["all_stop"]:
                            _bs_alerted = False   # reset once conditions normalise

                    except Exception as e:
                        print(f"[sentinel] sentiment error: {e}", flush=True)

                except Exception as e:
                    print(f"[coordinator] market data fetch failed: {e}", flush=True)

            await asyncio.sleep(LOOP_INTERVAL)

    await coordinator_loop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hermes Trading Worker")
    parser.add_argument("--pairs", nargs="+", help="Force specific pairs (skip scanner)")
    args = parser.parse_args()

    bootstrap_state()

    universe      = load_universe()
    total_capital = load_total_capital()

    print(f"[hermes] Universe: {universe}", flush=True)
    print(f"[hermes] Total capital: {total_capital} USDT", flush=True)
    print(f"[hermes] Scan interval: {SCAN_INTERVAL}s", flush=True)

    asyncio.run(_run_with_dashboard(universe, total_capital, force_pairs=args.pairs))


async def _run_with_dashboard(universe, total_capital, force_pairs=None):
    """Run trading coordinator and dashboard server concurrently."""
    from hermes_trading.dashboard import start as start_dashboard

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _on_sigterm():
        print("[hermes] SIGTERM received — initiating graceful shutdown", flush=True)
        shutdown_event.set()

    import signal
    loop.add_signal_handler(signal.SIGTERM, _on_sigterm)
    loop.add_signal_handler(signal.SIGINT,  _on_sigterm)

    main_task      = asyncio.create_task(run_all(universe, total_capital, force_pairs=force_pairs))
    dashboard_task = asyncio.create_task(start_dashboard())

    # Wait until shutdown signal or main task finishes
    shutdown_waiter = asyncio.create_task(shutdown_event.wait())
    done, _ = await asyncio.wait(
        [main_task, dashboard_task, shutdown_waiter],
        return_when=asyncio.FIRST_COMPLETED,
    )

    if shutdown_event.is_set():
        await _graceful_shutdown()

    for t in [main_task, dashboard_task]:
        if not t.done():
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass


async def _graceful_shutdown():
    """
    Called on SIGTERM — mark-to-market all open positions, log them to
    trades.jsonl, send Telegram notification, then exit cleanly.
    Railway sends SIGTERM ~10s before SIGKILL, so keep this fast.
    """
    import time, json as _json
    from hermes_trading.loop import TradingLoop, TRADES_FILE
    from hermes_trading.notify import send_trade_email, _send_telegram

    position_files = list(STATE_DIR.glob("position_*.json"))
    if not position_files:
        print("[shutdown] no open positions — clean exit", flush=True)
        return

    closed = []
    for pf in position_files:
        try:
            pos = _json.loads(pf.read_text())
        except Exception:
            continue

        asset     = pos.get("asset", "?")
        direction = pos.get("direction", "long")
        entry     = pos.get("entry_price", 0)
        deployed  = pos.get("usdt_deployed", 0)
        leverage  = pos.get("leverage", 1)

        # Fetch current price for mark-to-market
        try:
            from hermes_trading.adapters.exchange import fetch_ticker
            ticker  = await fetch_ticker(asset)
            current = float(ticker.get("last") or ticker.get("close") or entry)
        except Exception:
            current = entry   # fallback: flat PnL

        pnl_pct = (current - entry) / entry if direction == "long" else (entry - current) / entry
        comm    = round(deployed * leverage * 0.001 * 2, 6)
        gross   = round(pnl_pct * deployed, 4)
        net     = round(gross - comm, 4)

        trade = {
            **pos,
            "exit_price":      current,
            "exit_time":       int(time.time()),
            "pnl_pct":         round(pnl_pct, 6),
            "pnl_usdt":        gross,
            "commission_usdt": comm,
            "slippage_usdt":   0.0,
            "net_pnl_usdt":    net,
            "close_reason":    "shutdown",
        }

        # Append to trades.jsonl
        try:
            with open(TRADES_FILE, "a") as f:
                f.write(_json.dumps(trade) + "\n")
            print(f"[shutdown] logged {asset} {direction} pnl={pnl_pct:+.3%} → trades.jsonl", flush=True)
        except Exception as e:
            print(f"[shutdown] failed to log {asset}: {e}", flush=True)

        # Remove position file so it doesn't re-open on restart
        pf.unlink(missing_ok=True)
        closed.append(f"{asset} {direction.upper()} {pnl_pct:+.2%}")

    if closed:
        try:
            _send_telegram(
                f"⚡ <b>Hermes Shutdown</b>\n\n"
                f"Redeploying — {len(closed)} position(s) closed at market:\n"
                + "\n".join(f"  · {c}" for c in closed)
            )
        except Exception:
            pass

    print(f"[shutdown] graceful shutdown complete — {len(closed)} positions logged", flush=True)


if __name__ == "__main__":
    main()
