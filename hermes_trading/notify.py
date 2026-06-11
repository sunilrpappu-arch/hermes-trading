"""
Notification helpers — Telegram bot (HTTPS, works on Railway free tier).

SMTP is blocked by Railway networking. Telegram uses HTTPS only.

Set these Railway env vars:
  TELEGRAM_BOT_TOKEN  — from @BotFather
  TELEGRAM_CHAT_ID    — your personal chat ID (message the bot once, then check getUpdates)
"""
from __future__ import annotations
import os
import urllib.request
import urllib.parse
import json


def _send_telegram(message: str) -> bool:
    """Send a Telegram message via HTTPS. Returns True on success."""
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print(f"[notify] skipping — TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set", flush=True)
        return False
    try:
        payload = urllib.parse.urlencode({
            "chat_id":    chat_id,
            "text":       message,
            "parse_mode": "HTML",
        }).encode()
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        req = urllib.request.Request(url, data=payload, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("ok"):
                print(f"[notify] Telegram sent ✓", flush=True)
                return True
    except Exception as e:
        print(f"[notify] Telegram failed: {e}", flush=True)
    return False


def send_trade_email(trade: dict, stats: dict):
    """Send a trade-close notification via Telegram."""
    asset     = trade.get("asset", "?")
    direction = trade.get("direction", "?").upper()
    pnl_pct   = (trade.get("pnl_pct", 0) or 0) * 100
    pnl_usd   = trade.get("pnl_usdt", 0) or 0
    entry     = trade.get("entry_price", 0)
    exit_p    = trade.get("exit_price", 0)
    reason    = trade.get("close_reason", "?")
    regime    = trade.get("regime_at_entry", "?")
    version   = trade.get("strategy_version", "?")
    mode      = trade.get("mode", "paper")

    sign    = "+" if pnl_pct >= 0 else ""
    outcome = "✅ WIN" if pnl_pct >= 0 else "❌ LOSS"

    msg = (
        f"⚡ <b>Hermes [{mode.upper()}] Trade Closed</b>\n\n"
        f"{outcome}  {sign}{pnl_pct:.2f}%  ({sign}${pnl_usd:.4f})\n\n"
        f"<b>Pair:</b> {asset}\n"
        f"<b>Dir:</b> {direction}\n"
        f"<b>Entry:</b> ${entry:.6f}\n"
        f"<b>Exit:</b> ${exit_p:.6f}\n"
        f"<b>Reason:</b> {reason}\n"
        f"<b>Regime:</b> {regime}\n"
        f"<b>Strategy:</b> v{version}\n\n"
        f"<b>Portfolio</b>\n"
        f"Trades: {stats.get('total_trades','?')}  "
        f"({stats.get('wins','?')}W / {stats.get('losses','?')}L)  "
        f"WR: {stats.get('win_rate',0):.0f}%\n"
        f"Total PnL: {'+' if stats.get('total_pnl_usdt',0)>=0 else ''}${stats.get('total_pnl_usdt',0):.4f}"
    )
    _send_telegram(msg)


def send_reflection_notification(summary: str):
    """Send a reflection-cycle summary via Telegram."""
    _send_telegram(f"⚡ <b>Hermes Reflection</b>\n\n{summary}")
