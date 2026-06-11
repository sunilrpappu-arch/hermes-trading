"""
Notification helpers — email via Gmail SMTP.

Credentials are read from environment variables (set in Railway):
  GMAIL_APP_PASSWORD  — 16-char Google App Password
  GMAIL_TO            — recipient address
  GMAIL_FROM          — sender address (defaults to GMAIL_TO)
"""
from __future__ import annotations
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_trade_email(trade: dict, stats: dict):
    """
    Send an email notification when a trade closes.
    Reads credentials fresh from env vars each call (Railway sets them at runtime).
    Silently skips if credentials are not configured.
    """
    gmail_to       = os.getenv("GMAIL_TO", "")
    gmail_from     = os.getenv("GMAIL_FROM", gmail_to)
    gmail_password = os.getenv("GMAIL_APP_PASSWORD", "")

    if not gmail_to or not gmail_password:
        print(f"[notify] skipping email — GMAIL_TO or GMAIL_APP_PASSWORD not set", flush=True)
        return

    try:
        asset     = trade.get("asset", "?")
        direction = trade.get("direction", "?").upper()
        pnl_pct   = trade.get("pnl_pct", 0) * 100
        pnl_usd   = trade.get("pnl_usdt", 0)
        entry     = trade.get("entry_price", 0)
        exit_p    = trade.get("exit_price", 0)
        reason    = trade.get("close_reason", "?")
        regime    = trade.get("regime_at_entry", "?")
        lq_grab   = trade.get("lq_grab", "")
        version   = trade.get("strategy_version", "?")
        mode      = trade.get("mode", "paper")

        sign      = "+" if pnl_pct >= 0 else ""
        outcome   = "WIN ✅" if pnl_pct >= 0 else "LOSS ❌"

        subject = (
            f"⚡ Hermes [{mode.upper()}]: {direction} {asset} "
            f"{sign}{pnl_pct:.2f}% ({sign}${pnl_usd:.4f}) {outcome}"
        )

        regime_line = regime
        if lq_grab:
            regime_line += f"  ·  {lq_grab}"

        body = f"""
Trade closed on Hermes [{mode.upper()}]

Pair:       {asset}
Direction:  {direction}
Entry:      ${entry:.6f}
Exit:       ${exit_p:.6f}
PnL:        {sign}{pnl_pct:.3f}%  |  {sign}${pnl_usd:.4f} USDT
Reason:     {reason}
Regime:     {regime_line}
Strategy:   v{version}

────────────────────────────────
Portfolio Summary
────────────────────────────────
Total trades:  {stats.get('total_trades', '?')}
Win / Loss:    {stats.get('wins', '?')}W / {stats.get('losses', '?')}L
Win rate:      {stats.get('win_rate', 0):.1f}%
Cumulative PnL: {'+' if stats.get('total_pnl_usdt', 0) >= 0 else ''}${stats.get('total_pnl_usdt', 0):.4f} USDT

────────────────────────────────
Dashboard: https://hermes-trading-production-bcda.up.railway.app
        """.strip()

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = gmail_from
        msg["To"]      = gmail_to
        msg.attach(MIMEText(body, "plain"))

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(gmail_from, gmail_password)
            server.sendmail(gmail_from, gmail_to, msg.as_string())

        print(f"[notify] email sent → {gmail_to} ({subject})", flush=True)

    except Exception as e:
        print(f"[notify] email failed: {e}", flush=True)
