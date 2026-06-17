"""
Notification helpers — Telegram bot (HTTPS) + Resend email (HTTPS).

Set these Railway env vars:
  TELEGRAM_BOT_TOKEN  — from @BotFather
  TELEGRAM_CHAT_ID    — your personal chat ID
  RESEND_API_KEY      — from resend.com dashboard
  GMAIL_TO            — recipient address for email alerts
"""
from __future__ import annotations
import os
import urllib.request
import urllib.parse
import json


_NOTIFY_LOG = None  # set lazily from STATE_DIR


def _send_email(subject: str, body: str) -> bool:
    """Send alert email via Resend HTTPS API."""
    api_key = os.getenv("RESEND_API_KEY", "")
    to      = os.getenv("GMAIL_TO") or os.getenv("ALERT_EMAIL_TO", "")
    if not api_key or not to:
        print("[notify] email skipped — RESEND_API_KEY or GMAIL_TO not set", flush=True)
        return False
    try:
        payload = json.dumps({
            "from":    "Hermes Crypto <hermes@resend.dev>",
            "to":      [to],
            "subject": subject,
            "text":    body,
        }).encode()
        req = urllib.request.Request(
            "https://api.resend.com/emails",
            data=payload,
            headers={
                "Authorization":  f"Bearer {api_key}",
                "Content-Type":   "application/json",
                "User-Agent":     "HermesTrading/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("id"):
                print(f"[notify] email sent ✓ → {to} (id={result['id']})", flush=True)
                return True
    except Exception as e:
        print(f"[notify] email failed: {e}", flush=True)
    return False

def _log_notification(message: str, delivered: bool):
    """Persist every notification to state/notifications.jsonl regardless of Telegram status."""
    try:
        from pathlib import Path as _Path
        from datetime import datetime as _dt, timezone as _tz
        log = _Path(__file__).parent.parent / "state" / "notifications.jsonl"
        entry = json.dumps({
            "ts":        _dt.now(_tz.utc).isoformat(),
            "message":   message,
            "delivered": delivered,
        })
        with open(log, "a") as f:
            f.write(entry + "\n")
    except Exception:
        pass


def _send_telegram(message: str, email_subject: str = "Hermes Alert") -> bool:
    """Send via Telegram + Gmail. Always logs locally regardless of delivery."""
    # Email runs in parallel (best-effort, non-blocking to Telegram result)
    plain = message.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "")
    _send_email(email_subject, plain)

    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print(f"[notify] Telegram skipped — token/chat_id not set", flush=True)
        _log_notification(message, delivered=False)
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
                _log_notification(message, delivered=True)
                return True
    except Exception as e:
        print(f"[notify] Telegram failed: {e}", flush=True)
    _log_notification(message, delivered=False)
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
    _send_telegram(msg, email_subject=f"Hermes {outcome} {asset} {sign}{pnl_pct:.1f}%")


def send_entry_notification(trade: dict):
    """Send a trade-open notification via Telegram."""
    asset     = trade.get("asset", "?")
    direction = trade.get("direction", "?").upper()
    entry     = trade.get("entry_price", 0)
    sl        = trade.get("sl_price", 0)
    tp        = trade.get("tp_price", 0)
    rr        = trade.get("rr_ratio", 0)
    regime    = trade.get("regime_at_entry", "?")
    tp_method = trade.get("tp_method", "?")
    mode      = trade.get("mode", "paper")
    lev       = trade.get("leverage", 1)
    rsi       = trade.get("rsi_at_entry", None)
    mtf       = trade.get("mtf_signals", [])
    mtf_str   = " · ".join(mtf[:3]) if mtf else "—"

    arrow = "↑" if direction == "LONG" else "↓"
    msg = (
        f"⚡ <b>Hermes [{mode.upper()}] Entry</b> {arrow}\n\n"
        f"<b>{asset}</b>  {direction}  ·  {regime}  ·  {lev}x\n\n"
        f"<b>Entry:</b> ${entry:.6f}\n"
        f"<b>SL:</b> ${sl:.6f}\n"
        f"<b>TP:</b> ${tp:.6f}  ({tp_method})\n"
        f"<b>R:R:</b> {rr:.2f}x\n"
    )
    if rsi is not None:
        msg += f"<b>RSI:</b> {rsi:.1f}\n"
    if mtf_str != "—":
        msg += f"<b>MTF:</b> {mtf_str}\n"
    _send_telegram(msg, email_subject=f"Hermes Entry {arrow} {asset} {direction}")


def send_reflection_notification(summary: str):
    """Send a reflection-cycle summary via Telegram + email."""
    _send_telegram(f"⚡ <b>Hermes Reflection</b>\n\n{summary}", email_subject="Hermes Reflection Report")
