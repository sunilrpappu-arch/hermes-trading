"""
Black Swan detection + Fear/Greed meter.

Two responsibilities:
  1. fear_greed_score(heartbeats, btc_vol) → 0-100 composite sentiment score
  2. check(heartbeats, regime_info)        → detect extreme/anomalous conditions

Fear/Greed score (0 = Extreme Fear, 100 = Extreme Greed):
  30pts  Average RSI across all active pairs
  25pts  % pairs above their 50MA
  20pts  % pairs above VWAP
  15pts  BTC volatility vs baseline (inverted — high vol = fear)
  10pts  24H price momentum across pairs

Black Swan levels:
  normal   → nothing, all clear
  warning  → unusual but not critical; Telegram alert, no halt
  critical → halt all new entries, close all longs, Telegram emergency alert
"""
from __future__ import annotations
import time
from datetime import datetime, timezone

# Fear/Greed labels
LABELS = [
    (0,  20, "Extreme Fear",  "🔴"),
    (20, 35, "Fear",          "🟠"),
    (35, 50, "Caution",       "🟡"),
    (50, 65, "Neutral",       "⚪"),
    (65, 80, "Greed",         "🟢"),
    (80, 101,"Extreme Greed", "🤑"),
]

# Flash crash threshold: single candle move > this % = black swan
FLASH_CRASH_PCT      = 0.08   # 8%
# Cascade threshold: N pairs all moving > this % in same direction
CASCADE_MOVE_PCT     = 0.05   # 5%
CASCADE_MIN_PAIRS    = 3      # at least 3 pairs
# Price feed anomaly: new price deviates > this from last known
FEED_ANOMALY_PCT     = 0.20   # 20% deviation = bad data
# Extreme sentiment thresholds for warning/critical
FEAR_WARNING_SCORE   = 18
FEAR_CRITICAL_SCORE  = 10
GREED_WARNING_SCORE  = 85
GREED_CRITICAL_SCORE = 93


# ---------------------------------------------------------------------------
# Fear / Greed score
# ---------------------------------------------------------------------------

def fear_greed_score(heartbeats: list[dict], btc_vol: float = None) -> dict:
    """
    Compute a 0-100 fear/greed composite from live heartbeat data.

    Parameters
    ----------
    heartbeats : list of heartbeat dicts (one per active pair)
    btc_vol    : BTC realized vol (from volatility.detect)

    Returns
    -------
    {
      score        int     0–100
      label        str     "Extreme Fear" … "Extreme Greed"
      emoji        str     colour emoji
      components   dict    breakdown of each signal
      signals      list    human-readable signal descriptions
    }
    """
    if not heartbeats:
        return _neutral_score()

    # ── Signal 1: Average RSI (30pts) ─────────────────────────────────────
    rsi_vals = [h["rsi_15m"] for h in heartbeats if h.get("rsi_15m") is not None]
    if rsi_vals:
        avg_rsi  = sum(rsi_vals) / len(rsi_vals)
        # RSI 50 = neutral. Map 0–100 RSI → 0–30 pts score
        # RSI 25 → 0pts (extreme fear), RSI 75 → 30pts (extreme greed)
        rsi_score = max(0, min(30, (avg_rsi - 25) / 50 * 30))
    else:
        avg_rsi   = 50
        rsi_score = 15.0

    # ── Signal 2: % pairs above 50MA (25pts) ─────────────────────────────
    above_ma  = [h for h in heartbeats if h.get("trend") == "uptrend"]
    below_ma  = [h for h in heartbeats if h.get("trend") == "downtrend"]
    total_ma  = len(above_ma) + len(below_ma)
    if total_ma > 0:
        pct_above_ma = len(above_ma) / total_ma
        ma_score     = pct_above_ma * 25
    else:
        pct_above_ma = 0.5
        ma_score     = 12.5

    # ── Signal 3: % pairs above VWAP (20pts) ─────────────────────────────
    vwap_above  = [h for h in heartbeats if h.get("vwap_above") is True]
    vwap_below  = [h for h in heartbeats if h.get("vwap_above") is False]
    total_vwap  = len(vwap_above) + len(vwap_below)
    if total_vwap > 0:
        pct_above_vwap = len(vwap_above) / total_vwap
        vwap_score     = pct_above_vwap * 20
    else:
        pct_above_vwap = 0.5
        vwap_score     = 10.0

    # ── Signal 4: BTC volatility (15pts, inverted) ────────────────────────
    # High vol = fear, low vol = greed
    # Baseline calm vol ~1%, extreme vol ~5%+
    if btc_vol is not None and btc_vol > 0:
        # vol 0.01 (1%) = calm = greedy → high score
        # vol 0.05 (5%) = extreme → low score
        vol_score = max(0, min(15, (1 - min(btc_vol / 0.04, 1)) * 15))
    else:
        vol_score = 7.5

    # ── Signal 5: 24H momentum (10pts) ───────────────────────────────────
    # Use rng_pos as a proxy: pairs near range top = greed, near bottom = fear
    rng_vals = [h["rng_pos"] for h in heartbeats if h.get("rng_pos") is not None]
    if rng_vals:
        avg_rng    = sum(rng_vals) / len(rng_vals)
        mom_score  = avg_rng * 10    # rng_pos 0–1 → 0–10pts
    else:
        avg_rng   = 0.5
        mom_score = 5.0

    total = rsi_score + ma_score + vwap_score + vol_score + mom_score
    score = max(0, min(100, round(total)))

    label, emoji = _label(score)

    # Human-readable signal descriptions
    signals = []
    if avg_rsi < 35:
        signals.append(f"RSI oversold ({avg_rsi:.0f}) — market in fear")
    elif avg_rsi > 65:
        signals.append(f"RSI overbought ({avg_rsi:.0f}) — market greedy")
    else:
        signals.append(f"RSI neutral ({avg_rsi:.0f})")

    signals.append(f"{pct_above_ma*100:.0f}% pairs above 50MA")
    signals.append(f"{pct_above_vwap*100:.0f}% pairs above VWAP")

    if btc_vol is not None:
        signals.append(f"BTC vol {btc_vol*100:.2f}% ({'high — fear' if btc_vol > 0.03 else 'low — calm'})")

    return {
        "score":  score,
        "label":  label,
        "emoji":  emoji,
        "components": {
            "rsi_score":   round(rsi_score,  2),
            "ma_score":    round(ma_score,   2),
            "vwap_score":  round(vwap_score, 2),
            "vol_score":   round(vol_score,  2),
            "mom_score":   round(mom_score,  2),
            "avg_rsi":     round(avg_rsi,    1),
            "pct_above_ma":   round(pct_above_ma   * 100, 1),
            "pct_above_vwap": round(pct_above_vwap * 100, 1),
            "btc_vol_pct":    round((btc_vol or 0) * 100, 3),
            "avg_rng_pos":    round(avg_rng, 3),
        },
        "signals": signals,
    }


def _label(score: int) -> tuple[str, str]:
    for lo, hi, label, emoji in LABELS:
        if lo <= score < hi:
            return label, emoji
    return "Extreme Greed", "🤑"


def _neutral_score() -> dict:
    return {"score": 50, "label": "Neutral", "emoji": "⚪",
            "components": {}, "signals": ["No data yet"]}


# ---------------------------------------------------------------------------
# Black swan detector
# ---------------------------------------------------------------------------

def check(
    heartbeats:    list[dict],
    regime_info:   dict,
    prev_prices:   dict[str, float] | None = None,   # {asset: price} from last tick
    fg_score:      dict | None = None,
) -> dict:
    """
    Check for black swan / extreme conditions.

    Returns
    -------
    {
      level       str    "normal" | "warning" | "critical"
      events      list   detected events with descriptions
      action      str    what Hermes should do
      all_stop    bool   True if trading should halt immediately
      close_longs bool   True if open longs should be closed
      message     str    Telegram-ready alert message
    }
    """
    events     = []
    level      = "normal"
    all_stop   = False
    close_longs = False
    prev        = prev_prices or {}

    # ── 1. Flash crash: single tick price shock ───────────────────────────
    for hb in heartbeats:
        asset = hb.get("asset", "")
        price = hb.get("price", 0)
        last  = prev.get(asset, 0)
        if last and last > 0 and price > 0:
            move = (price - last) / last
            if move <= -FLASH_CRASH_PCT:
                events.append({
                    "type":    "flash_crash",
                    "asset":   asset,
                    "move_pct": round(move * 100, 2),
                    "severity": "critical",
                })
                level      = "critical"
                all_stop   = True
                close_longs = True
            elif move <= -FLASH_CRASH_PCT * 0.5:   # 4% single tick = warning
                events.append({
                    "type":    "price_shock",
                    "asset":   asset,
                    "move_pct": round(move * 100, 2),
                    "severity": "warning",
                })
                if level == "normal":
                    level = "warning"

    # ── 2. Cascade crash: multiple pairs moving sharply same direction ────
    prices     = {hb["asset"]: hb.get("price", 0) for hb in heartbeats if hb.get("price")}
    dump_pairs = []
    pump_pairs = []
    for asset, price in prices.items():
        last = prev.get(asset, 0)
        if last and last > 0:
            move = (price - last) / last
            if move <= -CASCADE_MOVE_PCT:
                dump_pairs.append((asset, move))
            elif move >= CASCADE_MOVE_PCT:
                pump_pairs.append((asset, move))

    if len(dump_pairs) >= CASCADE_MIN_PAIRS:
        events.append({
            "type":     "cascade_crash",
            "pairs":    [a for a, _ in dump_pairs],
            "avg_move": round(sum(m for _, m in dump_pairs) / len(dump_pairs) * 100, 2),
            "severity": "critical",
        })
        level       = "critical"
        all_stop    = True
        close_longs = True

    if len(pump_pairs) >= CASCADE_MIN_PAIRS:
        events.append({
            "type":     "cascade_pump",
            "pairs":    [a for a, _ in pump_pairs],
            "avg_move": round(sum(m for _, m in pump_pairs) / len(pump_pairs) * 100, 2),
            "severity": "warning",   # pump is warning, not critical
        })
        if level == "normal":
            level = "warning"

    # ── 3. Price feed anomaly: data integrity check ───────────────────────
    for hb in heartbeats:
        asset = hb.get("asset", "")
        price = hb.get("price", 0)
        last  = prev.get(asset, 0)
        if price <= 0:
            events.append({
                "type":     "feed_anomaly",
                "asset":    asset,
                "detail":   "price=0 or missing",
                "severity": "warning",
            })
            if level == "normal":
                level = "warning"
        elif last and last > 0:
            deviation = abs(price - last) / last
            if deviation > FEED_ANOMALY_PCT:
                events.append({
                    "type":     "feed_anomaly",
                    "asset":    asset,
                    "detail":   f"{deviation*100:.1f}% deviation from last tick",
                    "severity": "warning",
                })
                if level == "normal":
                    level = "warning"

    # ── 4. Extreme Fear/Greed from sentiment score ────────────────────────
    if fg_score:
        score = fg_score.get("score", 50)
        if score <= FEAR_CRITICAL_SCORE:
            events.append({
                "type":     "extreme_fear",
                "score":    score,
                "label":    fg_score.get("label"),
                "severity": "critical",
            })
            level       = "critical"
            all_stop    = True
            close_longs = True
        elif score <= FEAR_WARNING_SCORE:
            events.append({
                "type":     "fear_warning",
                "score":    score,
                "label":    fg_score.get("label"),
                "severity": "warning",
            })
            if level == "normal":
                level = "warning"
        elif score >= GREED_CRITICAL_SCORE:
            events.append({
                "type":     "extreme_greed",
                "score":    score,
                "label":    fg_score.get("label"),
                "severity": "warning",   # greed = warning (don't halt, but be careful)
            })
            if level == "normal":
                level = "warning"

    # ── 5. Volatile / extreme macro regime ───────────────────────────────
    regime = regime_info.get("regime", "normal")
    vol    = regime_info.get("vol", 0)
    if regime == "extreme":
        events.append({
            "type":     "macro_extreme",
            "vol":      round(vol * 100, 2),
            "severity": "warning",
        })
        if level == "normal":
            level = "warning"

    # Build action description
    if level == "critical":
        action = "HALT all entries + close all longs"
    elif level == "warning":
        action = "Reduce size, no new longs, monitor closely"
    else:
        action = "All clear"

    message = _format_alert(level, events, fg_score)

    return {
        "level":       level,
        "events":      events,
        "action":      action,
        "all_stop":    all_stop,
        "close_longs": close_longs,
        "message":     message,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }


def _format_alert(level: str, events: list[dict], fg_score: dict | None) -> str:
    if level == "normal":
        return ""
    icon = "🚨" if level == "critical" else "⚠️"
    lines = [f"{icon} <b>Black Swan Alert — {level.upper()}</b>\n"]
    for e in events:
        t = e["type"]
        if t == "flash_crash":
            lines.append(f"💥 Flash crash: {e['asset']} {e['move_pct']:+.1f}% in one tick")
        elif t == "price_shock":
            lines.append(f"📉 Price shock: {e['asset']} {e['move_pct']:+.1f}%")
        elif t == "cascade_crash":
            lines.append(f"🌊 Cascade crash: {', '.join(e['pairs'])} avg {e['avg_move']:.1f}%")
        elif t == "cascade_pump":
            lines.append(f"🚀 Cascade pump: {', '.join(e['pairs'])} avg +{e['avg_move']:.1f}%")
        elif t == "feed_anomaly":
            lines.append(f"📡 Feed anomaly: {e['asset']} — {e['detail']}")
        elif t == "extreme_fear":
            lines.append(f"😱 Extreme Fear score={e['score']}/100")
        elif t == "fear_warning":
            lines.append(f"😟 Fear warning score={e['score']}/100")
        elif t == "extreme_greed":
            lines.append(f"🤑 Extreme Greed score={e['score']}/100 — risk of reversal")
        elif t == "macro_extreme":
            lines.append(f"🌋 Macro extreme vol={e['vol']:.1f}%")

    if level == "critical":
        lines.append("\n⛔ <b>All entries halted. Open longs closing.</b>")
    else:
        lines.append("\n⚠️ Position sizes reduced. No new longs.")

    return "\n".join(lines)
