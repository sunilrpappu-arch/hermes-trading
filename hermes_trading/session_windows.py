"""
Session window detector — returns whether we are currently inside a major
market-open volatility window and which session it is.

Crypto is 24/7 but institutional flow and retail activity cluster around
traditional market opens, producing reliable volatility spikes:

  Asia open   00:00–01:30 UTC  (Tokyo/Singapore/HK retail + futures)
  London open 07:00–08:30 UTC  (institutional FX + crypto desks)
  US open     13:30–15:00 UTC  (equities open, options flow, ETF arb)

These windows are wider than the literal open bell to capture the
pre-open positioning and the first directional impulse.
"""
from __future__ import annotations
from datetime import datetime, timezone, time as dtime

SESSION_WINDOWS: list[dict] = [
    {
        "name":       "Asia",
        "emoji":      "🌏",
        "open_utc":   dtime(0,  0),
        "close_utc":  dtime(1, 30),
    },
    {
        "name":       "London",
        "emoji":      "🇬🇧",
        "open_utc":   dtime(7,  0),
        "close_utc":  dtime(8, 30),
    },
    {
        "name":       "US",
        "emoji":      "🇺🇸",
        "open_utc":   dtime(13, 30),
        "close_utc":  dtime(15,  0),
    },
]


def current_session(now_utc: datetime | None = None) -> dict | None:
    """
    Return the active session dict (with 'name', 'emoji', 'minutes_remaining')
    or None if no session window is active right now.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    t = now_utc.time().replace(second=0, microsecond=0)
    for s in SESSION_WINDOWS:
        if s["open_utc"] <= t < s["close_utc"]:
            # Minutes remaining in window
            end_mins   = s["close_utc"].hour * 60 + s["close_utc"].minute
            now_mins   = t.hour * 60 + t.minute
            remaining  = end_mins - now_mins
            return {**s, "minutes_remaining": remaining}
    return None


def next_session(now_utc: datetime | None = None) -> dict:
    """
    Return the next upcoming session and how many minutes until it opens.
    Always returns a value (wraps around midnight if needed).
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    t        = now_utc.time().replace(second=0, microsecond=0)
    now_mins = t.hour * 60 + t.minute

    for s in SESSION_WINDOWS:
        open_mins = s["open_utc"].hour * 60 + s["open_utc"].minute
        if open_mins > now_mins:
            return {**s, "minutes_until": open_mins - now_mins}

    # All sessions passed today — next is Asia tomorrow
    asia = SESSION_WINDOWS[0]
    return {**asia, "minutes_until": (24 * 60 - now_mins)}


def session_volume_multiplier(candles_15m: list[dict]) -> float:
    """
    Return the volume ratio of the last 2 bars vs the 24h average bar volume.
    A ratio >= 1.5 indicates a session-driven volume spike.
    Returns 1.0 if insufficient candles.
    """
    if len(candles_15m) < 20:
        return 1.0
    avg_vol  = sum(c.get("volume", 0) for c in candles_15m[-96:]) / min(len(candles_15m), 96)
    recent   = sum(c.get("volume", 0) for c in candles_15m[-2:]) / 2
    if avg_vol <= 0:
        return 1.0
    return round(recent / avg_vol, 2)
