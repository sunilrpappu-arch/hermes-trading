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
from pathlib import Path
import os

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
            "asset":         pair,
            "price":         pd.get("price", 0.0),
            "timestamp":     pd.get("timestamp", 0),
            "candles":       all_candles.get(pair, {}),
            "regime":        regime_info.get("regime", "normal"),
            "is_sideways":   regime_info.get("is_sideways", False),
            "adx":           regime_info.get("adx"),
            "regime_params": regime_info,
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

        # 1. Detect volatility regime from BTC 1H candles
        btc_candles_1h = await fetch_candles("BTC/USDT", "1h", 100)
        regime_info = await detect_regime(btc_candles_1h)

        # 2. Select pairs — fetch live universe from Binance, fall back to goal.yaml list
        if force_pairs:
            selected = force_pairs[:regime_info["max_pairs"]]
        else:
            goal      = load_goal()
            live_universe = await fetch_universe(filters=goal.get("universe_filters", {}))
            selected  = await scan_pairs(live_universe, regime_info["max_pairs"], regime_info.get("vol", 0.02))

        # 3. Retire loops no longer selected
        for pair in list(loops.keys()):
            if pair not in selected:
                print(f"[coordinator] retiring {pair}", flush=True)
                if tasks.get(pair) and not tasks[pair].done():
                    tasks[pair].cancel()
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

    async def coordinator_loop():
        last_scan = 0.0
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
    await asyncio.gather(
        run_all(universe, total_capital, force_pairs=force_pairs),
        start_dashboard(),
    )


if __name__ == "__main__":
    main()
