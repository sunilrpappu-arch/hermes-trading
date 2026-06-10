"""
Exchange adapter — paper mode uses in-memory simulation, live mode uses ccxt/Binance Futures.

Set HERMES_TRADING_MODE=live and BINANCE_API_KEY / BINANCE_API_SECRET to go live.
Set HERMES_TRADING_TESTNET=true to use Binance Futures testnet instead of mainnet.

Uses USDT-M perpetual futures (defaultType=future). Leverage is fixed at 1x by default —
override via HERMES_LEVERAGE env var. At 1x there is no liquidation risk beyond your margin.
"""
import os
import logging

logger = logging.getLogger(__name__)

MODE = os.getenv("HERMES_TRADING_MODE", "paper").lower()
LEVERAGE = int(os.getenv("HERMES_LEVERAGE", "1"))
_exchange = None


def _get_exchange():
    global _exchange
    if _exchange is not None:
        return _exchange

    import ccxt

    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    testnet = os.getenv("HERMES_TRADING_TESTNET", "false").lower() == "true"

    if not api_key or not api_secret:
        raise RuntimeError(
            "BINANCE_API_KEY and BINANCE_API_SECRET must be set for live trading"
        )

    _exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},   # USDT-M perpetual futures
    })

    if testnet:
        _exchange.set_sandbox_mode(True)
        logger.info("Binance Futures TESTNET mode enabled")

    return _exchange


def is_live() -> bool:
    return MODE == "live" and os.getenv("HERMES_TRADING_I_ACCEPT_RISK", "").lower() == "true"


def _set_leverage(symbol: str, leverage: int):
    """Set leverage for a symbol. Called once before opening a position."""
    try:
        ex = _get_exchange()
        ex.set_leverage(leverage, symbol)
        logger.info("Leverage set to %dx for %s", leverage, symbol)
    except Exception as e:
        logger.warning("Could not set leverage for %s: %s", symbol, e)


def open_long(symbol: str, usdt_amount: float) -> dict:
    """Open a long (buy) position with `usdt_amount` USDT of notional at current leverage."""
    if not is_live():
        return {"paper": True, "side": "buy", "direction": "long", "symbol": symbol, "usdt_amount": usdt_amount}

    ex = _get_exchange()
    _set_leverage(symbol, LEVERAGE)
    ticker = ex.fetch_ticker(symbol)
    price = ticker["last"]
    market = ex.market(symbol)
    qty = (usdt_amount * LEVERAGE) / price
    qty = _round_down(qty, market["precision"]["amount"])
    order = ex.create_market_buy_order(symbol, qty, params={"reduceOnly": False})
    logger.info("OPEN LONG %s qty=%.6f price~=%.4f order_id=%s", symbol, qty, price, order["id"])
    return order


def open_short(symbol: str, usdt_amount: float) -> dict:
    """Open a short (sell) position with `usdt_amount` USDT of notional at current leverage."""
    if not is_live():
        return {"paper": True, "side": "sell", "direction": "short", "symbol": symbol, "usdt_amount": usdt_amount}

    ex = _get_exchange()
    _set_leverage(symbol, LEVERAGE)
    ticker = ex.fetch_ticker(symbol)
    price = ticker["last"]
    market = ex.market(symbol)
    qty = (usdt_amount * LEVERAGE) / price
    qty = _round_down(qty, market["precision"]["amount"])
    order = ex.create_market_sell_order(symbol, qty, params={"reduceOnly": False})
    logger.info("OPEN SHORT %s qty=%.6f price~=%.4f order_id=%s", symbol, qty, price, order["id"])
    return order


def close_long(symbol: str, qty: float) -> dict:
    """Close a long position by selling `qty`."""
    if not is_live():
        return {"paper": True, "side": "sell", "direction": "close_long", "symbol": symbol, "qty": qty}

    ex = _get_exchange()
    market = ex.market(symbol)
    qty = _round_down(qty, market["precision"]["amount"])
    order = ex.create_market_sell_order(symbol, qty, params={"reduceOnly": True})
    logger.info("CLOSE LONG %s qty=%.6f order_id=%s", symbol, qty, order["id"])
    return order


def close_short(symbol: str, qty: float) -> dict:
    """Close a short position by buying back `qty`."""
    if not is_live():
        return {"paper": True, "side": "buy", "direction": "close_short", "symbol": symbol, "qty": qty}

    ex = _get_exchange()
    market = ex.market(symbol)
    qty = _round_down(qty, market["precision"]["amount"])
    order = ex.create_market_buy_order(symbol, qty, params={"reduceOnly": True})
    logger.info("CLOSE SHORT %s qty=%.6f order_id=%s", symbol, qty, order["id"])
    return order


def fetch_balance_usdt() -> float:
    """Return available USDT margin balance. Returns 0.0 in paper mode."""
    if not is_live():
        return 0.0
    ex = _get_exchange()
    bal = ex.fetch_balance(params={"type": "future"})
    return float(bal["free"].get("USDT", 0.0))


# Keep backward-compat aliases for any code still calling the old spot functions
def place_market_buy(symbol: str, usdt_amount: float) -> dict:
    return open_long(symbol, usdt_amount)


def place_market_sell(symbol: str, qty: float) -> dict:
    return close_long(symbol, qty)


def _round_down(value: float, precision) -> float:
    """Round down to exchange precision (int decimal places or float step size)."""
    import math
    if isinstance(precision, int):
        factor = 10 ** precision
        return math.floor(value * factor) / factor
    step = float(precision)
    return math.floor(value / step) * step
