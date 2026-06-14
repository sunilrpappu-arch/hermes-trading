"""
Hermes Dashboard — FastAPI web server that runs alongside the trading loop.

Endpoints:
  GET /           → self-contained HTML dashboard (auto-refreshes every 30s)
  GET /api/state  → full JSON state snapshot
  GET /api/trades → list of all closed trades
  GET /api/pairs  → per-pair heartbeat data

Port defaults to 8080 (set PORT env var to override).
"""
from __future__ import annotations
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

STATE_DIR     = Path(os.getenv("STATE_DIR", Path(__file__).parent.parent / "state"))
CONTROLS_FILE = STATE_DIR / "controls.json"
PORT          = int(os.getenv("PORT", "8080"))

app = FastAPI(title="Hermes Trading Dashboard", docs_url=None, redoc_url=None)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _read_trades() -> list[dict]:
    tf = STATE_DIR / "trades.jsonl"
    if not tf.exists():
        return []
    trades = []
    for line in tf.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                trades.append(json.loads(line))
            except Exception:
                pass
    return trades


_HEARTBEAT_MAX_AGE = 45 * 60   # 45 minutes — pairs not updated in this window are considered retired


def _read_heartbeats() -> dict[str, dict]:
    """Return {asset: heartbeat_dict} for all recently-active pairs (updated within 45 min)."""
    hbs  = {}
    stale = []
    now  = time.time()
    for hf in STATE_DIR.glob("heartbeat_*.json"):
        try:
            data  = json.loads(hf.read_text())
            asset = data.get("asset") or hf.stem.replace("heartbeat_", "").replace("_", "/", 1)
            ts_str = data.get("timestamp")
            if ts_str:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                age = now - ts
                if age > _HEARTBEAT_MAX_AGE:
                    stale.append(hf)
                    continue
            hbs[asset] = data
        except Exception:
            pass
    # Clean up stale heartbeat files (pairs that were retired long ago)
    for hf in stale:
        try:
            hf.unlink()
        except Exception:
            pass
    return hbs


def _read_drawdown() -> dict:
    df = STATE_DIR / "drawdown.json"
    if not df.exists():
        return {}
    try:
        return json.loads(df.read_text())
    except Exception:
        return {}


def _read_active_features() -> dict:
    af = STATE_DIR / "active_features.json"
    try:
        return json.loads(af.read_text()) if af.exists() else {}
    except Exception:
        return {}


def _read_strategy_notes() -> list:
    nf = STATE_DIR / "strategy_notes.json"
    try:
        return json.loads(nf.read_text()) if nf.exists() else []
    except Exception:
        return []


def _read_strategy() -> dict:
    sf = STATE_DIR / "strategy.yaml"
    if not sf.exists():
        return {}
    try:
        import yaml
        with open(sf) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _trade_pnl_usdt(t: dict) -> float:
    """
    Return pnl in USDT for a trade.
    Falls back to pnl_pct × usdt_deployed for old trades that predate the field.
    """
    v = t.get("pnl_usdt")
    if v is not None:
        return float(v)
    pnl_pct     = t.get("pnl_pct", 0) or 0
    deployed    = t.get("usdt_deployed") or t.get("position_size_r", 0.05) * 200
    return pnl_pct * float(deployed)


def _portfolio_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pnl_usdt": 0.0, "total_pnl_pct": 0.0,
                "total_commission_usdt": 0.0, "total_slippage_usdt": 0.0,
                "total_net_pnl_usdt": 0.0, "cost_drag_pct": 0.0,
                "best_trade": None, "worst_trade": None}

    wins   = [t for t in trades if (t.get("pnl_pct") or 0) > 0]
    losses = [t for t in trades if (t.get("pnl_pct") or 0) <= 0]
    total_pnl_usdt = sum(_trade_pnl_usdt(t) for t in trades)
    total_pnl_pct  = sum(t.get("pnl_pct", 0) or 0 for t in trades)

    total_commission = sum(t.get("commission_usdt", 0) or 0 for t in trades)
    total_slippage   = sum(t.get("slippage_usdt",   0) or 0 for t in trades)
    total_net_pnl    = sum(
        t.get("net_pnl_usdt", _trade_pnl_usdt(t)) for t in trades
    )
    # Cost drag: what % of gross PnL was eaten by fees + slippage (0 if no gross PnL)
    cost_drag = ((total_commission + total_slippage) / abs(total_pnl_usdt) * 100
                 if total_pnl_usdt != 0 else 0.0)

    best  = max(trades, key=lambda t: t.get("pnl_pct", 0))
    worst = min(trades, key=lambda t: t.get("pnl_pct", 0))

    return {
        "total_trades":          len(trades),
        "wins":                  len(wins),
        "losses":                len(losses),
        "win_rate":              round(len(wins) / len(trades) * 100, 1),
        "total_pnl_usdt":        round(total_pnl_usdt, 4),
        "total_pnl_pct":         round(total_pnl_pct * 100, 3),
        "total_commission_usdt": round(total_commission, 4),
        "total_slippage_usdt":   round(total_slippage, 4),
        "total_net_pnl_usdt":    round(total_net_pnl, 4),
        "cost_drag_pct":         round(cost_drag, 1),
        "best_trade":            best,
        "worst_trade":           worst,
    }


def _cumulative_pnl(trades: list[dict]) -> list[dict]:
    """Sorted by exit_time, cumulative PnL in USDT."""
    sorted_trades = sorted(trades, key=lambda t: t.get("exit_time", 0))
    running = 0.0
    points  = []
    for t in sorted_trades:
        running += _trade_pnl_usdt(t)
        points.append({
            "time":  datetime.fromtimestamp(t["exit_time"], tz=timezone.utc).strftime("%m/%d %H:%M")
                     if t.get("exit_time") else "?",
            "pnl":   round(running, 4),
            "asset": t.get("asset", ""),
        })
    return points


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/state")
async def api_state():
    trades    = _read_trades()
    heartbeats = _read_heartbeats()
    drawdown  = _read_drawdown()
    strategy  = _read_strategy()
    stats     = _portfolio_stats(trades)
    cum_pnl   = _cumulative_pnl(trades)
    sentiment = _read_sentiment()

    # Detect current regime from any active heartbeat
    regime     = "unknown"
    is_sideways = False
    for hb in heartbeats.values():
        if hb.get("regime"):
            regime      = hb["regime"]
            is_sideways = hb.get("is_sideways", False)
            break

    total_capital = float(os.getenv("TOTAL_CAPITAL_USDT", "1000"))

    return JSONResponse({
        "generated_at":       datetime.now(timezone.utc).isoformat(),
        "total_capital_usdt": total_capital,
        "strategy":           strategy,
        "strategy_version": strategy.get("version", "?"),
        "regime":          regime,
        "is_sideways":     is_sideways,
        "portfolio":     stats,
        "drawdown":      drawdown,
        "cum_pnl":       cum_pnl,
        "heartbeats":    heartbeats,
        "recent_trades": list(reversed(trades)),  # all trades, newest first — paginated client-side
        "sentiment":        sentiment,
        "active_features":  _read_active_features(),
        "strategy_notes":   _read_strategy_notes(),
    })


@app.get("/api/trades")
async def api_trades():
    return JSONResponse({"trades": list(reversed(_read_trades()))})


@app.post("/api/reset-stats")
async def api_reset_stats():
    """
    Archive and wipe all historical state so the dashboard starts fresh.
    Safe in paper mode. In live mode, open positions are preserved.
    """
    import time as _time
    archived = []
    errors   = []

    # 1. Archive + clear trades
    tf = STATE_DIR / "trades.jsonl"
    if tf.exists() and tf.stat().st_size > 0:
        ts  = int(_time.time())
        dst = STATE_DIR / "history" / f"trades_archive_{ts}.jsonl"
        (STATE_DIR / "history").mkdir(parents=True, exist_ok=True)
        try:
            dst.write_bytes(tf.read_bytes())
            archived.append(str(dst.name))
        except Exception as e:
            errors.append(f"archive: {e}")
    try:
        tf.write_text("")
        archived.append("trades.jsonl cleared")
    except Exception as e:
        errors.append(f"clear trades: {e}")

    # 2. Clear drawdown
    dd = STATE_DIR / "drawdown.json"
    try:
        dd.unlink(missing_ok=True)
        archived.append("drawdown.json deleted")
    except Exception as e:
        errors.append(f"drawdown: {e}")

    # 3. Clear hypotheses
    hyp = STATE_DIR / "hypotheses.jsonl"
    try:
        hyp.write_text("")
        archived.append("hypotheses.jsonl cleared")
    except Exception as e:
        errors.append(f"hypotheses: {e}")

    # 4. Clear paper positions (skip in live mode to avoid losing track of open trades)
    from hermes_trading.adapters.exchange import is_live
    if not is_live():
        for pf in STATE_DIR.glob("position_*.json"):
            try:
                pf.unlink()
                archived.append(f"{pf.name} deleted")
            except Exception as e:
                errors.append(f"{pf.name}: {e}")

    status = "ok" if not errors else "partial"
    print(f"[reset-stats] {status}: {', '.join(archived)}", flush=True)
    return JSONResponse({"status": status, "archived": archived, "errors": errors})


@app.get("/api/pairs")
async def api_pairs():
    return JSONResponse({"pairs": _read_heartbeats()})


def _read_sentiment() -> dict:
    sf = STATE_DIR / "sentiment.json"
    if not sf.exists():
        return {}
    try:
        return json.loads(sf.read_text())
    except Exception:
        return {}


@app.get("/api/sentiment")
async def api_sentiment():
    return JSONResponse(_read_sentiment())


def _read_controls() -> dict:
    try:
        if CONTROLS_FILE.exists():
            return json.loads(CONTROLS_FILE.read_text())
    except Exception:
        pass
    return {"all_stop": False, "manual_exits": [], "pending_entries": [], "leverage_overrides": {}}


def _write_controls(ctrl: dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROLS_FILE.write_text(json.dumps(ctrl, indent=2))


@app.get("/api/controls")
async def api_controls():
    return JSONResponse(_read_controls())


@app.post("/api/control")
async def api_control(request: Request):
    """
    Manual override endpoint — called from dashboard buttons.

    Actions:
      all_stop          → close all positions, halt new entries
      resume            → clear all_stop flag, resume normal trading
      exit              → force-close a specific pair's open position
      enter             → queue a manual entry for a specific pair
      set_leverage      → override leverage for a specific pair (persists until changed)
      clear_leverage    → remove leverage override for a pair (reverts to regime default)
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid JSON"}, status_code=400)

    action = body.get("action")
    ctrl   = _read_controls()

    if action == "all_stop":
        ctrl["all_stop"] = True
        _write_controls(ctrl)
        return JSONResponse({"ok": True, "all_stop": True})

    elif action == "resume":
        ctrl["all_stop"] = False
        _write_controls(ctrl)
        return JSONResponse({"ok": True, "all_stop": False})

    elif action == "exit":
        asset = body.get("asset")
        if not asset:
            return JSONResponse({"ok": False, "error": "asset required"}, status_code=400)
        exits = ctrl.get("manual_exits", [])
        if asset not in exits:
            exits.append(asset)
        ctrl["manual_exits"] = exits
        _write_controls(ctrl)
        return JSONResponse({"ok": True, "queued_exit": asset})

    elif action == "enter":
        asset     = body.get("asset")
        direction = body.get("direction", "long")
        leverage  = body.get("leverage")
        if not asset:
            return JSONResponse({"ok": False, "error": "asset required"}, status_code=400)
        entries = [e for e in ctrl.get("pending_entries", []) if e.get("asset") != asset]
        entry = {"asset": asset, "direction": direction}
        if leverage:
            entry["leverage"] = float(leverage)
        entries.append(entry)
        ctrl["pending_entries"] = entries
        _write_controls(ctrl)
        return JSONResponse({"ok": True, "queued_entry": entry})

    elif action == "set_leverage":
        asset    = body.get("asset")
        leverage = body.get("leverage")
        if not asset or leverage is None:
            return JSONResponse({"ok": False, "error": "asset and leverage required"}, status_code=400)
        overrides = ctrl.get("leverage_overrides", {})
        overrides[asset] = float(leverage)
        ctrl["leverage_overrides"] = overrides
        _write_controls(ctrl)
        return JSONResponse({"ok": True, "leverage_override": {asset: float(leverage)}})

    elif action == "clear_leverage":
        asset = body.get("asset")
        if not asset:
            return JSONResponse({"ok": False, "error": "asset required"}, status_code=400)
        overrides = ctrl.get("leverage_overrides", {})
        overrides.pop(asset, None)
        ctrl["leverage_overrides"] = overrides
        _write_controls(ctrl)
        return JSONResponse({"ok": True, "cleared_leverage": asset})

    else:
        return JSONResponse({"ok": False, "error": f"unknown action: {action}"}, status_code=400)


# ---------------------------------------------------------------------------
# Telegram test endpoint
# ---------------------------------------------------------------------------

@app.get("/api/test-telegram")
async def api_test_telegram():
    """Send a test Telegram message to verify bot config."""
    from hermes_trading.notify import _send_telegram
    ok = _send_telegram(
        "✅ <b>Hermes Telegram test</b>\n\nNotifications are working correctly!"
    )
    if ok:
        return JSONResponse({"ok": True, "message": "Test message sent — check Telegram!"})
    return JSONResponse(
        {"ok": False, "message": "Failed — check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in Railway env vars"},
        status_code=500,
    )


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=_HTML, status_code=200)


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Hermes Trading Dashboard</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚡</text></svg>"/>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body { background:#0f172a; color:#e2e8f0; font-family:'Inter',system-ui,sans-serif; }
  .card { background:#1e293b; border:1px solid #334155; border-radius:12px; padding:20px; }
  .badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.75rem; font-weight:600; }
  .pos-long  { color:#34d399; }
  .pos-short { color:#f87171; }
  .pos-none  { color:#94a3b8; }
  .pnl-pos   { color:#34d399; }
  .pnl-neg   { color:#f87171; }
  table { width:100%; border-collapse:collapse; }
  th { color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:.05em;
       text-align:left; padding:8px 12px; border-bottom:1px solid #334155; }
  td { padding:10px 12px; border-bottom:1px solid #1e293b; font-size:0.875rem; }
  tr:hover td { background:#1e293b; }
  .regime-sideways  { background:#4338ca22; color:#818cf8; border:1px solid #4338ca55; }
  .regime-calm      { background:#05966922; color:#34d399; border:1px solid #05966955; }
  .regime-normal    { background:#ca8a0422; color:#fbbf24; border:1px solid #ca8a0455; }
  .regime-volatile  { background:#ea580c22; color:#fb923c; border:1px solid #ea580c55; }
  .regime-extreme   { background:#dc262622; color:#f87171; border:1px solid #dc262655; }
  #pnl-chart-wrap   { position:relative; height:220px; }
  .spinner { border:2px solid #334155; border-top:2px solid #818cf8; border-radius:50%;
             width:16px; height:16px; animation:spin .8s linear infinite; display:inline-block; }
  @keyframes spin { to { transform:rotate(360deg); } }
</style>
</head>
<body class="min-h-screen p-4 md:p-8">

<!-- Header -->
<div class="flex items-center justify-between mb-8">
  <div>
    <h1 class="text-2xl font-bold text-white">⚡ Hermes</h1>
    <p class="text-slate-400 text-sm mt-1">Self-improving trading agent</p>
  </div>
  <div class="flex items-center gap-3">
    <span id="mode-badge" class="badge bg-slate-700 text-slate-300">…</span>
    <span id="regime-badge" class="badge">…</span>
    <span id="strategy-ver" class="text-slate-500 text-xs">v?</span>
    <button onclick="document.getElementById('strategy-modal').classList.remove('hidden')"
      class="px-3 py-1 rounded-lg text-xs bg-slate-700 border border-slate-600 text-slate-300 hover:text-white hover:bg-slate-600 transition">
      📋 Strategy
    </button>
    <span id="refresh-spinner" class="spinner"></span>
  </div>
</div>

<!-- Strategy Quick Reference Modal -->
<div id="strategy-modal" class="hidden fixed inset-0 z-50 flex items-start justify-center pt-10 px-4"
  style="background:rgba(0,0,0,0.75)" onclick="if(event.target===this)this.classList.add('hidden')">
  <div class="relative w-full max-w-3xl rounded-2xl border border-slate-700 bg-slate-900 shadow-2xl overflow-y-auto max-h-[85vh] p-6">
    <button onclick="document.getElementById('strategy-modal').classList.add('hidden')"
      class="absolute top-4 right-4 text-slate-500 hover:text-white text-xl leading-none">✕</button>
    <h2 class="text-white font-bold text-lg mb-4">⚡ Hermes Strategy · Quick Reference</h2>
    <div class="space-y-4 text-sm text-slate-300">

      <div class="rounded-lg bg-slate-800 p-4">
        <p class="text-slate-400 text-xs font-semibold uppercase mb-2">Regime Detection (4H)</p>
        <div class="space-y-1">
          <div><span class="text-green-400 font-semibold">Bull</span> — Price &gt; 50MA &amp; ADX ≥ 20 · Longs preferred · Shorts only at liquidity grabs</div>
          <div><span class="text-red-400 font-semibold">Bear</span> — Price &lt; 50MA &amp; ADX ≥ 20 · Shorts preferred · Longs only at liquidity grabs</div>
          <div><span class="text-yellow-400 font-semibold">Sideways</span> — ADX &lt; 20 · Mean-reversion at range extremes (bottom 20% long, top 20% short) · MTF = 0</div>
        </div>
      </div>

      <div class="rounded-lg bg-slate-800 p-4">
        <p class="text-slate-400 text-xs font-semibold uppercase mb-2">Entry Gates (all must pass)</p>
        <div class="space-y-1">
          <div>1. <span class="text-white">Scanner score</span> — Bull long ≥ 35 · Bear short ≥ 60</div>
          <div>2. <span class="text-white">MTF gate</span> — ≥ 1 higher-TF signal (4H MACD, 4H MA, 1H RSI div, 1H MACD) · waived sideways / liq-grabs</div>
          <div>3. <span class="text-white">Pattern gate</span> — bearish pattern blocks longs, bullish blocks shorts · bypassed if RSI &lt; 25 (long) or &gt; 75 (short)</div>
          <div>4. <span class="text-white">Senti-meter</span> — ≤ 10 entries halted · ≥ 93 no new longs</div>
          <div>5. <span class="text-white">Drawdown / cooldown</span> — 10% per-pair cap · 8% portfolio cap · 30m cooldown after stop-loss</div>
          <div>6. <span class="text-white">Daily loss</span> — halted if down &gt; 3% on the day</div>
        </div>
      </div>

      <div class="rounded-lg bg-slate-800 p-4">
        <p class="text-slate-400 text-xs font-semibold uppercase mb-2">TP / SL Logic</p>
        <div class="space-y-1">
          <div><span class="text-white">Default</span> — TP 3% · SL 1.8% · min R:R 1.0×</div>
          <div><span class="text-white">Pattern TP</span> — measured move from neckline/breakout · min R:R relaxed to 0.75× if confidence ≥ 70%</div>
          <div><span class="text-white">Dynamic levels</span> — swing high/low · Fibonacci · VWAP bands override defaults when R:R improves</div>
          <div><span class="text-white">Session breakout</span> — Asia/London/US open windows · requires BB expansion + volume ≥ 1.5× avg</div>
        </div>
      </div>

      <div class="rounded-lg bg-slate-800 p-4">
        <p class="text-slate-400 text-xs font-semibold uppercase mb-2">Leverage (dynamic)</p>
        <div class="space-y-1">
          <div>Sideways / Calm → 2× · Normal → 1.5× · Volatile / Extreme → 1× · Hard cap 3×</div>
        </div>
      </div>

      <div class="rounded-lg bg-slate-800 p-4">
        <p class="text-slate-400 text-xs font-semibold uppercase mb-2">Senti-meter Signals → Action</p>
        <div class="space-y-1">
          <div>≤ 10 → <b>OVERSOLD</b> — entries halted, open longs closing</div>
          <div>≤ 18 → <b>BEARISH</b> — reduce sizes, wait for stabilisation</div>
          <div>19–35 → <b>BEARISH</b> — caution, favour shorts</div>
          <div>36–64 → <b>SIDEWAYS</b> — normal conditions</div>
          <div>65–84 → <b>BULLISH</b> — normal conditions, favour longs</div>
          <div>85–92 → <b>OVERBOUGHT</b> — tighten stops, avoid chasing entries</div>
          <div>≥ 93 → <b>OVERBOUGHT</b> — reversal likely, no new longs</div>
        </div>
      </div>

      <div class="rounded-lg bg-slate-800 p-4">
        <p class="text-slate-400 text-xs font-semibold uppercase mb-2">Active Modules</p>
        <div id="active-features" class="flex flex-wrap gap-2"></div>
      </div>

      <div id="strategy-notes-section" class="rounded-lg bg-slate-800 p-4 hidden">
        <p class="text-slate-400 text-xs font-semibold uppercase mb-2">🤖 Hermes Notes <span id="strategy-notes-ts" class="text-slate-600 font-normal normal-case ml-1"></span></p>
        <div id="strategy-notes-list" class="space-y-1"></div>
      </div>

      <div class="rounded-lg bg-slate-800 p-4">
        <p class="text-slate-400 text-xs font-semibold uppercase mb-2">Self-Improvement Cycle</p>
        <div class="space-y-1">
          <div><span class="text-white">Every 5 trades</span> — reflection: WR, avg R:R, regime breakdown, pattern performance</div>
          <div><span class="text-white">After 8h silence</span> — downtime: idle diagnosis, OOS backtest, R:R bleed check, shadow trade review</div>
          <div><span class="text-white">R:R bleed check</span> — if pattern TP WR lags structural TP by &gt; 10pp (≥ 5 samples) → flags relaxed 0.75× R:R</div>
          <div><span class="text-white">Shadow trading</span> — blocked setups tracked 48h forward to validate gates</div>
        </div>
      </div>

    </div>
  </div>
</div>

<!-- Summary cards -->
<div class="grid grid-cols-2 md:grid-cols-6 gap-4 mb-6">
  <div class="card">
    <p class="text-slate-400 text-xs mb-1">TOTAL CAPITAL</p>
    <p id="stat-capital" class="text-2xl font-bold text-white">$0</p>
    <p id="stat-capital-deployed" class="text-slate-500 text-xs mt-1">$0 deployed</p>
  </div>
  <div class="card">
    <p class="text-slate-400 text-xs mb-1">GROSS PnL</p>
    <p id="stat-pnl" class="text-2xl font-bold">$0.00</p>
    <p id="stat-net-pnl" class="text-slate-500 text-xs mt-1">net $0.00</p>
  </div>
  <div class="card">
    <p class="text-slate-400 text-xs mb-1">WIN RATE</p>
    <p id="stat-winrate" class="text-2xl font-bold">—</p>
  </div>
  <div class="card">
    <p class="text-slate-400 text-xs mb-1">TOTAL TRADES</p>
    <p id="stat-trades" class="text-2xl font-bold">0</p>
  </div>
  <div class="card">
    <p class="text-slate-400 text-xs mb-1">PORTFOLIO DD</p>
    <p id="stat-dd" class="text-2xl font-bold">0%</p>
  </div>
  <div class="card">
    <p class="text-slate-400 text-xs mb-1">FEES + SLIP</p>
    <p id="stat-costs" class="text-2xl font-bold text-orange-400">$0.00</p>
    <p id="stat-cost-drag" class="text-slate-500 text-xs mt-1">0% of gross PnL</p>
  </div>
</div>

<!-- Black Swan Alert Banner (hidden when normal) -->
<div id="swan-banner" class="mb-6 rounded-xl px-5 py-4 hidden"
     style="background:#1a0808;border:1px solid #7f1d1d;">
  <div class="flex items-start gap-3">
    <span id="swan-icon" class="text-2xl">🚨</span>
    <div class="flex-1">
      <p id="swan-title" class="text-red-300 font-bold text-sm mb-1">BLACK SWAN ALERT</p>
      <div id="swan-events" class="text-red-400 text-xs space-y-0.5"></div>
    </div>
    <span id="swan-action" class="text-red-300 text-xs font-semibold bg-red-950 px-2 py-1 rounded"></span>
  </div>
</div>

<!-- Session Window Bar (hidden when no active session) -->
<div id="session-bar" class="mb-4 rounded-lg px-4 py-2 hidden"
     style="background:#0f172a;border:1px solid #334155;">
  <div class="flex items-center justify-between">
    <div class="flex items-center gap-2">
      <span id="session-emoji" class="text-lg"></span>
      <span id="session-label" class="text-slate-200 text-sm font-semibold"></span>
      <span class="text-slate-500 text-xs">· Session open window active</span>
    </div>
    <div class="flex items-center gap-4">
      <span class="text-slate-400 text-xs">Vol spike: <span id="session-vol" class="text-indigo-300 font-mono"></span>×</span>
      <span class="text-slate-400 text-xs"><span id="session-mins" class="text-indigo-300 font-mono"></span> min remaining</span>
      <span id="session-mode" class="text-xs px-2 py-0.5 rounded font-semibold"></span>
    </div>
  </div>
</div>

<!-- Fear/Greed + Macro row -->
<div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">

  <!-- Left column: Real F&G + Hermes Senti-meter stacked -->
  <div class="col-span-1 flex flex-col gap-4">

    <!-- Real Fear & Greed (alternative.me) -->
    <div class="card">
      <p class="text-slate-400 text-xs font-semibold mb-2">FEAR &amp; GREED INDEX <span class="text-slate-600 font-normal">· alternative.me</span></p>
      <div class="flex flex-col items-center">
        <div style="position:relative;width:160px;height:88px;overflow:hidden;">
          <svg width="160" height="88" viewBox="0 0 180 100">
            <path d="M10,90 A80,80 0 0,1 170,90" fill="none" stroke="#1e293b" stroke-width="16" stroke-linecap="round"/>
            <path d="M10,90 A80,80 0 0,1 42,26"  fill="none" stroke="#ef4444" stroke-width="16" stroke-linecap="butt" opacity="0.5"/>
            <path d="M42,26 A80,80 0 0,1 90,10"  fill="none" stroke="#f97316" stroke-width="16" stroke-linecap="butt" opacity="0.5"/>
            <path d="M90,10 A80,80 0 0,1 138,26" fill="none" stroke="#64748b" stroke-width="16" stroke-linecap="butt" opacity="0.5"/>
            <path d="M138,26 A80,80 0 0,1 170,90" fill="none" stroke="#10b981" stroke-width="16" stroke-linecap="butt" opacity="0.5"/>
            <line id="rfg-needle" x1="90" y1="90" x2="90" y2="18"
              stroke="#e2e8f0" stroke-width="2.5" stroke-linecap="round"
              transform="rotate(-90, 90, 90)"/>
            <circle cx="90" cy="90" r="5" fill="#e2e8f0"/>
          </svg>
          <div style="position:absolute;bottom:2px;width:100%;text-align:center;">
            <span id="rfg-score-num" class="text-2xl font-bold text-white">—</span>
          </div>
        </div>
        <p id="rfg-label" class="text-xs font-bold mt-1 text-slate-300">Loading…</p>
      </div>
    </div>

    <!-- Hermes Senti-meter (internal composite) -->
    <div class="card">
      <p class="text-slate-400 text-xs font-semibold mb-2">HERMES SENTI-METER <span class="text-slate-600 font-normal">· live pairs</span></p>
      <div class="flex flex-col items-center">
        <div style="position:relative;width:160px;height:88px;overflow:hidden;">
          <svg width="160" height="88" viewBox="0 0 180 100">
            <path d="M10,90 A80,80 0 0,1 170,90" fill="none" stroke="#1e293b" stroke-width="16" stroke-linecap="round"/>
            <path d="M10,90 A80,80 0 0,1 42,26" fill="none" stroke="#ef4444" stroke-width="16" stroke-linecap="butt" opacity="0.5"/>
            <path d="M42,26 A80,80 0 0,1 90,10" fill="none" stroke="#f97316" stroke-width="16" stroke-linecap="butt" opacity="0.5"/>
            <path d="M90,10 A80,80 0 0,1 138,26" fill="none" stroke="#64748b" stroke-width="16" stroke-linecap="butt" opacity="0.5"/>
            <path d="M138,26 A80,80 0 0,1 170,90" fill="none" stroke="#10b981" stroke-width="16" stroke-linecap="butt" opacity="0.5"/>
            <line id="fg-needle" x1="90" y1="90" x2="90" y2="18"
              stroke="#e2e8f0" stroke-width="2.5" stroke-linecap="round"
              transform="rotate(0, 90, 90)"/>
            <circle cx="90" cy="90" r="5" fill="#e2e8f0"/>
          </svg>
          <div style="position:absolute;bottom:2px;width:100%;text-align:center;">
            <span id="fg-score-num" class="text-2xl font-bold text-white">—</span>
          </div>
        </div>
        <p id="fg-label" class="text-xs font-bold mt-1 text-slate-300">Loading…</p>
        <div id="fg-signals" class="mt-1 space-y-0.5 text-center"></div>
      </div>
    </div>

  </div><!-- end left column -->

  <!-- Macro regime signals -->
  <div class="card col-span-2">
    <p class="text-slate-400 text-xs font-semibold mb-3">MACRO SIGNALS</p>
    <div id="macro-signals-grid" class="grid grid-cols-2 gap-3"></div>
  </div>

</div>

<!-- Controls Panel -->
<div id="controls-panel" class="card mb-6">
  <div class="flex items-center justify-between mb-4">
    <p class="text-slate-400 text-xs font-semibold">MANUAL CONTROLS</p>
    <span id="all-stop-badge" class="badge bg-slate-700 text-slate-400">● ACTIVE</span>
  </div>
  <div class="flex flex-wrap gap-3 mb-4">
    <button onclick="doAllStop()"
      class="flex items-center gap-2 px-4 py-2 rounded-lg font-semibold text-sm bg-red-950 hover:bg-red-900 text-red-300 border border-red-800 transition-colors">
      ⛔ ALL STOP
    </button>
    <button onclick="doResume()"
      class="flex items-center gap-2 px-4 py-2 rounded-lg font-semibold text-sm bg-emerald-950 hover:bg-emerald-900 text-emerald-300 border border-emerald-800 transition-colors">
      ▶ RESUME TRADING
    </button>
    <button onclick="doTestTelegram()"
      class="flex items-center gap-2 px-4 py-2 rounded-lg font-semibold text-sm bg-blue-950 hover:bg-blue-900 text-blue-300 border border-blue-800 transition-colors">
      📨 TEST TELEGRAM
    </button>
  </div>

  <!-- Manual entry form -->
  <div class="border-t border-slate-700 pt-4">
    <p class="text-slate-500 text-xs mb-3">MANUAL ENTRY — force open a position on next tick</p>
    <div class="flex flex-wrap gap-2 items-end">
      <div>
        <label class="text-slate-500 text-xs block mb-1">Pair</label>
        <select id="manual-asset" class="bg-slate-800 border border-slate-700 text-slate-200 text-sm rounded px-2 py-1.5">
          <option value="">Select pair…</option>
        </select>
      </div>
      <div>
        <label class="text-slate-500 text-xs block mb-1">Direction</label>
        <select id="manual-dir" class="bg-slate-800 border border-slate-700 text-slate-200 text-sm rounded px-2 py-1.5">
          <option value="long">LONG</option>
          <option value="short">SHORT</option>
        </select>
      </div>
      <div>
        <label class="text-slate-500 text-xs block mb-1">Leverage</label>
        <select id="manual-lev" class="bg-slate-800 border border-slate-700 text-slate-200 text-sm rounded px-2 py-1.5">
          <option value="">Regime default</option>
          <option value="1">1x</option>
          <option value="1.5">1.5x</option>
          <option value="2">2x</option>
          <option value="3">3x</option>
        </select>
      </div>
      <button onclick="doManualEntry()"
        class="px-4 py-1.5 rounded-lg text-sm font-semibold bg-indigo-900 hover:bg-indigo-800 text-indigo-200 border border-indigo-700 transition-colors">
        Queue Entry →
      </button>
    </div>
  </div>
</div>

<!-- PnL chart + Active pairs -->
<div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">

  <!-- PnL curve -->
  <div class="card">
    <p class="text-slate-400 text-xs font-semibold mb-3">CUMULATIVE PnL (USDT)</p>
    <div id="pnl-chart-wrap">
      <canvas id="pnlChart"></canvas>
    </div>
  </div>

  <!-- Active pairs -->
  <div class="card">
    <p class="text-slate-400 text-xs font-semibold mb-3">ACTIVE PAIRS <span class="text-slate-600 font-normal">(click pair to load chart)</span></p>
    <div id="pairs-grid" class="space-y-3"></div>
  </div>

</div>

<!-- TradingView Chart -->
<div class="card mb-6">
  <div class="flex items-center justify-between mb-3">
    <p class="text-slate-400 text-xs font-semibold">CHART — <span id="chart-symbol-label" class="text-white">BTC/USDT</span></p>
    <div class="flex items-center gap-2">
      <select id="chart-interval" onchange="reloadChart()"
        class="bg-slate-800 text-slate-300 text-xs rounded px-2 py-1 border border-slate-700">
        <option value="15">15m</option>
        <option value="60" selected>1H</option>
        <option value="240">4H</option>
        <option value="D">1D</option>
      </select>
      <a id="tv-expand-btn" href="https://www.tradingview.com/chart/?symbol=BINANCE:BTCUSDTPERP"
        target="_blank"
        class="flex items-center gap-1 text-xs text-slate-400 hover:text-white border border-slate-700 rounded px-2 py-1 transition-colors">
        ↗ Open in TradingView
      </a>
    </div>
  </div>
  <div id="tv-chart-container" style="height:480px;"></div>
</div>

<!-- Latest News -->
<div class="card mb-6">
  <div class="flex items-center justify-between mb-3">
    <p class="text-slate-400 text-xs font-semibold">📰 LATEST NEWS <span class="text-slate-600 font-normal">(active pairs · last 24h)</span></p>
    <span id="news-updated" class="text-slate-600 text-xs"></span>
  </div>
  <div id="news-panel" class="space-y-1">
    <p class="text-slate-600 text-xs">Loading...</p>
  </div>
</div>

<!-- Recent trades -->
<div class="card mb-6">
  <div class="flex items-center justify-between mb-3">
    <p class="text-slate-400 text-xs font-semibold">RECENT TRADES</p>
    <div class="flex items-center gap-3">
      <span id="trades-page-info" class="text-slate-500 text-xs"></span>
      <button onclick="tradesPagePrev()" id="btn-trades-prev"
        class="px-2 py-1 rounded text-xs bg-slate-800 border border-slate-700 text-slate-400 hover:text-slate-200 disabled:opacity-30">◀ Prev</button>
      <button onclick="tradesPageNext()" id="btn-trades-next"
        class="px-2 py-1 rounded text-xs bg-slate-800 border border-slate-700 text-slate-400 hover:text-slate-200 disabled:opacity-30">Next ▶</button>
    </div>
  </div>
  <div class="overflow-x-auto">
    <table id="trades-table">
      <thead>
        <tr>
          <th>Pair</th><th>Dir</th><th>Entry</th><th>Exit</th>
          <th>R:R</th><th>SL</th><th>TP</th>
          <th>PnL %</th><th>PnL $</th><th>Net $</th><th>Fees</th><th>Reason</th><th>Regime</th><th>Time</th>
        </tr>
      </thead>
      <tbody id="trades-body">
        <tr><td colspan="14" class="text-center text-slate-500 py-8">No trades yet</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- Last updated -->
<p class="text-center text-slate-600 text-xs pb-4">
  Updated <span id="last-updated">—</span> · auto-refreshes every 30s
</p>

<script>
let pnlChart = null;

function pnlClass(v) {
  return parseFloat(v) >= 0 ? 'pnl-pos' : 'pnl-neg';
}
function fmtPct(v) {
  const n = parseFloat(v);
  return (n >= 0 ? '+' : '') + n.toFixed(3) + '%';
}
function fmtUSD(v) {
  const n = parseFloat(v);
  return (n >= 0 ? '+$' : '-$') + Math.abs(n).toFixed(4);
}
function regimeClass(r) {
  const map = {sideways:'regime-sideways', calm:'regime-calm', normal:'regime-normal',
               volatile:'regime-volatile', extreme:'regime-extreme'};
  return map[r] || 'bg-slate-700 text-slate-300';
}
function regimeLabel(r, sw) {
  const labels = {sideways:'↔ Sideways', calm:'🟢 Calm', normal:'🟡 Normal',
                  volatile:'🟠 Volatile', extreme:'🔴 Extreme'};
  return labels[r] || r;
}

function tvUrl(asset) {
  const sym = asset.replace('/USDT', '');
  // Use perpetual futures chart where available, fall back to spot
  return `https://www.tradingview.com/chart/?symbol=BINANCE:${sym}USDTPERP`;
}

let _savedControls = {};   // latest controls from /api/controls — used to keep UI in sync

function renderPairs(heartbeats) {
  const el = document.getElementById('pairs-grid');
  const entries = Object.entries(heartbeats);
  if (!entries.length) {
    el.innerHTML = '<p class="text-slate-500 text-sm">No active pairs</p>';
    return;
  }
  el.innerHTML = entries.map(([asset, hb]) => {
    const pos      = hb.open_position;
    const posDir   = pos ? pos.direction : null;
    const posClass = posDir === 'long' ? 'pos-long' : posDir === 'short' ? 'pos-short' : 'pos-none';
    const posLabel = posDir ? posDir.toUpperCase() : '—';
    const rsi      = hb.rsi_15m ?? '—';
    const rng      = hb.rng_pos != null ? Math.round(hb.rng_pos * 100) + '%' : null;
    const tsAge    = hb.timestamp ? Math.round((Date.now() - new Date(hb.timestamp).getTime()) / 1000) : null;
    const newsLabel     = hb.news_label || 'no_data';
    const newsHeadlines = hb.news_headlines || [];
    const newsHeadline  = newsHeadlines[0] || null;
    const newsBadge = newsLabel === 'bullish'  ? `<span title="${newsHeadline||''}" style="color:#34d399;font-size:10px">📰▲</span>`
                    : newsLabel === 'bearish'  ? `<span title="${newsHeadline||''}" style="color:#f87171;font-size:10px">📰▼</span>`
                    : newsLabel === 'neutral'  ? `<span title="${newsHeadline||''}" style="color:#64748b;font-size:10px">📰</span>`
                    : '';
    const ageStr   = tsAge != null ? (tsAge < 120 ? tsAge + 's ago' : Math.round(tsAge/60) + 'm ago') : '?';
    const ageColor = tsAge == null ? '#64748b' : tsAge < 90 ? '#34d399' : tsAge < 300 ? '#fbbf24' : '#f87171';

    if (pos && pos.entry_price && hb.price) {
      // ── EXPANDED card for open position ──
      const mult     = pos.direction === 'long' ? 1 : -1;
      const entry    = parseFloat(pos.entry_price);
      const current  = parseFloat(hb.price);
      const pnlPct   = ((current - entry) / entry) * mult * 100;
      const deployed = parseFloat(pos.usdt_deployed || 0);
      const pnlUsd   = pnlPct / 100 * deployed;
      const qty      = pos.qty > 0 ? parseFloat(pos.qty) : (entry > 0 ? deployed / entry : 0);
      // Use structural levels when available, fall back to percentage-derived
      const slPrice  = pos.sl_price ? parseFloat(pos.sl_price)
                     : (pos.direction === 'long' ? entry*(1-(parseFloat(pos.stop_loss_pct||1.8)/100))
                                                 : entry*(1+(parseFloat(pos.stop_loss_pct||1.8)/100)));
      const tpPrice  = pos.tp_price ? parseFloat(pos.tp_price)
                     : (pos.direction === 'long' ? entry*(1+(parseFloat(pos.take_profit_pct||3.0)/100))
                                                 : entry*(1-(parseFloat(pos.take_profit_pct||3.0)/100)));
      const slPct    = pos.sl_pct ? parseFloat(pos.sl_pct) : Math.abs((slPrice - entry)/entry*100);
      const tpPct    = pos.tp_pct ? parseFloat(pos.tp_pct) : Math.abs((tpPrice - entry)/entry*100);
      const rrRatio  = pos.rr_ratio ? parseFloat(pos.rr_ratio) : (tpPct / slPct);
      const slMethod = pos.sl_method || null;
      const tpMethod = pos.tp_method || null;
      const pnlColor = pnlPct >= 0 ? '#34d399' : '#f87171';
      const barPct   = Math.min(Math.abs(pnlPct) / tpPct * 100, 100).toFixed(1);
      const rrColor  = rrRatio >= 2.0 ? '#34d399' : rrRatio >= 1.0 ? '#fbbf24' : '#f87171';
      const signals    = (pos.htf_signals || []).join(', ') || null;
      const regime     = pos.pair_regime || '—';
      // Use saved override if present, else fall back to position's entry leverage
      const lev_override = _savedControls.leverage_overrides && _savedControls.leverage_overrides[asset];
      const pos_leverage = parseFloat(lev_override || pos.leverage || 1.0);
      const borderCol  = pos.direction === 'long' ? '#065f46' : '#7f1d1d';

      return `
      <div class="rounded-lg px-3 py-3" style="background:#0f1a2e;border:1px solid ${borderCol};cursor:pointer"
           onclick="loadChart('${asset}')" title="Click to load chart">
        <div class="flex items-center justify-between mb-2">
          <div class="flex items-center gap-2">
            <a href="${tvUrl(asset)}" target="_blank" class="font-bold text-white text-sm hover:text-indigo-400 transition-colors" title="Open on TradingView">${asset.replace('/USDT','')}/USDT ↗</a>
            <span class="${posClass} text-xs font-bold px-2 py-0.5 rounded-full"
              style="border:1px solid ${borderCol};background:${pos.direction==='long'?'#05966920':'#dc262620'}">${posLabel}</span>
            <span class="text-slate-500 text-xs">${regime}</span>
          </div>
          <div style="color:${ageColor}" class="text-xs">${ageStr}</div>
        </div>

        <div class="grid grid-cols-3 gap-x-3 gap-y-1 text-xs mb-3">
          <div><span class="text-slate-500">Entry </span><span class="text-white font-mono">$${entry.toFixed(6)}</span></div>
          <div><span class="text-slate-500">Now </span><span class="text-white font-mono" data-live-asset="${asset}" data-field="price">$${current.toFixed(6)}</span></div>
          <div><span class="text-slate-500">PnL </span><span class="font-bold font-mono" data-live-asset="${asset}" data-field="pnlpct" style="color:${pnlColor}">${pnlPct>=0?'+':''}${pnlPct.toFixed(3)}%</span></div>
          <div><span class="text-slate-500">Capital </span><span class="text-slate-300">$${deployed.toFixed(2)}</span></div>
          <div><span class="text-slate-500">Qty </span><span class="text-slate-300">${qty.toPrecision(4)} ${asset.replace('/USDT','')}</span></div>
          <div><span class="text-slate-500">PnL$ </span><span class="font-mono" data-live-asset="${asset}" data-field="pnlusd" style="color:${pnlColor}">${pnlUsd>=0?'+':''}$${pnlUsd.toFixed(4)}</span></div>
          <div>
            <span class="text-slate-500">Stop </span>
            <span class="text-red-400 font-mono">$${slPrice.toPrecision(6)}</span>
            ${slMethod ? `<span class="text-slate-600 text-xs block">${slMethod}</span>` : ''}
          </div>
          <div>
            <span class="text-slate-500">Target </span>
            <span class="text-emerald-400 font-mono">$${tpPrice.toPrecision(6)}</span>
            ${tpMethod ? `<span class="text-slate-600 text-xs block">${tpMethod}</span>` : ''}
          </div>
          <div>
            <span class="text-slate-500">R:R </span>
            <span class="font-bold font-mono" style="color:${rrColor}">${rrRatio.toFixed(2)}</span>
            <span class="text-slate-600 text-xs"> (1:${rrRatio.toFixed(2)})</span>
          </div>
          <div><span class="text-slate-500">RSI </span><span class="text-slate-300">${rsi}${rng?' · '+rng:''}</span></div>
        </div>

        <div class="w-full rounded-full h-1.5 mb-1" style="background:#1e293b">
          <div class="h-1.5 rounded-full transition-all duration-500" data-live-asset="${asset}" data-field="bar" style="width:${barPct}%;background:${pnlColor}"></div>
        </div>
        <div class="flex justify-between" style="font-size:0.65rem;color:#475569">
          <span class="text-red-500">SL -${slPct.toFixed(2)}%</span>
          <span data-live-asset="${asset}" data-field="barlabel">${pnlPct>=0?'+':''}${pnlPct.toFixed(2)}% of ${tpPct.toFixed(2)}% target</span>
          <span class="text-emerald-600">TP +${tpPct.toFixed(2)}%</span>
        </div>
        ${signals ? `<div class="mt-1 mb-2" style="font-size:0.65rem;color:#475569">MTF: ${signals}</div>` : ''}

        <!-- Per-position controls -->
        <div class="flex gap-2 mt-2 pt-2 border-t border-slate-800 flex-wrap">
          <button onclick="event.stopPropagation();doForceExit('${asset}')"
            class="px-3 py-1 text-xs font-semibold rounded bg-red-950 hover:bg-red-900 text-red-300 border border-red-800 transition-colors">
            ✕ Force Exit
          </button>
          <div class="flex items-center gap-1">
            <span class="text-slate-500 text-xs">Leverage:</span>
            <select onchange="event.stopPropagation();doSetLeverage('${asset}', this.value)"
              class="bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded px-1 py-0.5">
              <option value="">default</option>
              <option value="1" ${pos_leverage==1?'selected':''}>1x</option>
              <option value="1.5" ${pos_leverage==1.5?'selected':''}>1.5x</option>
              <option value="2" ${pos_leverage==2?'selected':''}>2x</option>
              <option value="3" ${pos_leverage==3?'selected':''}>3x</option>
            </select>
            <span class="text-slate-400 text-xs">(now ${pos_leverage}x · notional $${(pos_leverage * deployed).toFixed(0)})</span>
          </div>
        </div>
      </div>`;
    }

    // ── COMPACT card for flat pairs ──
    return `
    <div class="flex items-center justify-between bg-slate-900 rounded-lg px-3 py-2 cursor-pointer hover:bg-slate-800 transition-colors"
         onclick="loadChart('${asset}')" title="${newsHeadlines.length ? '📰 ' + newsHeadlines.join(' | ') : 'Click to load chart'}">
      <div>
        <span class="font-semibold text-sm text-white">${asset.replace('/USDT','')}/USDT</span>
        ${newsBadge}
        <span class="${posClass} text-xs font-bold ml-1">${posLabel}</span>
      </div>
      <div class="text-right text-xs text-slate-400">
        <div>$${parseFloat(hb.price || 0).toFixed(4)}</div>
        <div>RSI ${rsi}${rng ? ' · rng ' + rng : ''}</div>
        <div style="color:${ageColor}">${ageStr}</div>
      </div>
    </div>`;
  }).join('');
}

// Trades pagination state
let _allTrades   = [];
let _tradesPage  = 0;
const TRADES_PER_PAGE = 20;

function renderTrades(trades) {
  // already newest-first from server — no reverse needed
  _allTrades  = [...trades];
  _tradesPage = 0;
  _renderTradesPage();
}

function _renderTradesPage() {
  const tbody    = document.getElementById('trades-body');
  const info     = document.getElementById('trades-page-info');
  const btnPrev  = document.getElementById('btn-trades-prev');
  const btnNext  = document.getElementById('btn-trades-next');
  const total    = _allTrades.length;
  const pages    = Math.max(1, Math.ceil(total / TRADES_PER_PAGE));
  const start    = _tradesPage * TRADES_PER_PAGE;
  const slice    = _allTrades.slice(start, start + TRADES_PER_PAGE);

  if (info)    info.textContent = total
    ? `${start + 1}–${Math.min(start + TRADES_PER_PAGE, total)} of ${total} trades`
    : '';
  if (btnPrev) btnPrev.disabled = _tradesPage === 0;
  if (btnNext) btnNext.disabled = _tradesPage >= pages - 1;

  if (!total) {
    tbody.innerHTML = '<tr><td colspan="12" class="text-center text-slate-500 py-8">No trades yet</td></tr>';
    return;
  }

  tbody.innerHTML = slice.map(t => {
    const pct  = t.pnl_pct != null ? t.pnl_pct * 100 : null;
    const usd  = t.pnl_usdt != null ? t.pnl_usdt : (t.pnl_pct != null ? t.pnl_pct * (t.usdt_deployed || 10) : null);
    const dt   = t.exit_time ? new Date(t.exit_time * 1000).toLocaleString() : '—';
    const dir  = t.direction === 'long'
      ? '<span class="pos-long font-semibold">LONG</span>'
      : '<span class="pos-short font-semibold">SHORT</span>';
    const pnlPct = pct != null
      ? `<span class="${pnlClass(pct)}">${fmtPct(pct)}</span>` : '—';
    const pnlUsd = usd != null
      ? `<span class="${pnlClass(usd)}">${fmtUSD(usd)}</span>` : '—';
    const reason = t.close_reason || '—';
    const regime = t.pair_regime || t.regime_at_entry || '—';
    const signal = t.lq_grab ? `<br><span class="text-slate-500 text-xs">${t.lq_grab}</span>` : '';

    // R:R — colour-coded: ≥2 green, ≥1 yellow, <1 red
    let rrCell = '—';
    if (t.rr_ratio != null) {
      const rr = parseFloat(t.rr_ratio);
      const rrCls = rr >= 2.0 ? 'pnl-pos' : rr >= 1.0 ? 'text-yellow-400' : 'pnl-neg';
      rrCell = `<span class="${rrCls} font-semibold">${rr.toFixed(2)}</span>`;
    }

    // SL cell: price + % + method
    let slCell = '—';
    if (t.sl_price != null) {
      const slPct  = t.sl_pct    != null ? parseFloat(t.sl_pct).toFixed(2)  + '%' : '';
      const slMeth = t.sl_method ? `<span class="text-slate-500 text-xs block">${t.sl_method}</span>` : '';
      slCell = `<span class="text-red-400">${parseFloat(t.sl_price).toPrecision(6)}</span>
                <span class="text-slate-500 text-xs"> −${slPct}</span>${slMeth}`;
    } else if (t.stop_loss_pct != null) {
      slCell = `<span class="text-slate-400 text-xs">−${parseFloat(t.stop_loss_pct).toFixed(2)}%</span>`;
    }

    // TP cell: price + % + method
    let tpCell = '—';
    if (t.tp_price != null) {
      const tpPct  = t.tp_pct    != null ? parseFloat(t.tp_pct).toFixed(2)  + '%' : '';
      const tpMeth = t.tp_method ? `<span class="text-slate-500 text-xs block">${t.tp_method}</span>` : '';
      tpCell = `<span class="text-emerald-400">${parseFloat(t.tp_price).toPrecision(6)}</span>
                <span class="text-slate-500 text-xs"> +${tpPct}</span>${tpMeth}`;
    } else if (t.take_profit_pct != null) {
      tpCell = `<span class="text-slate-400 text-xs">+${parseFloat(t.take_profit_pct).toFixed(2)}%</span>`;
    }

    return `<tr>
      <td><a href="${tvUrl(t.asset||'')}" target="_blank" class="font-medium text-white hover:text-indigo-400 transition-colors">${(t.asset||'').replace('/USDT','')}/USDT ↗</a></td>
      <td>${dir}</td>
      <td>${parseFloat(t.entry_price||0).toPrecision(6)}</td>
      <td>${parseFloat(t.exit_price||0).toPrecision(6)}</td>
      <td class="text-center">${rrCell}</td>
      <td class="text-xs leading-tight">${slCell}</td>
      <td class="text-xs leading-tight">${tpCell}</td>
      <td>${pnlPct}</td>
      <td>${pnlUsd}</td>
      <td>${t.net_pnl_usdt != null ? (t.net_pnl_usdt >= 0 ? '<span class="pnl-pos">+$' + Math.abs(t.net_pnl_usdt).toFixed(4) + '</span>' : '<span class="pnl-neg">-$' + Math.abs(t.net_pnl_usdt).toFixed(4) + '</span>') : '<span class="text-slate-500">—</span>'}</td>
      <td class="text-xs text-orange-400">${t.commission_usdt != null ? '-$' + ((t.commission_usdt||0)+(t.slippage_usdt||0)).toFixed(4) : '—'}</td>
      <td><span class="text-slate-400">${reason}</span>${signal}</td>
      <td><span class="badge ${regimeClass(regime)} text-xs">${regime}</span></td>
      <td class="text-slate-500 text-xs">${dt}</td>
    </tr>`;
  }).join('');
}

function tradesPagePrev() {
  if (_tradesPage > 0) { _tradesPage--; _renderTradesPage(); }
}
function tradesPageNext() {
  const pages = Math.ceil(_allTrades.length / TRADES_PER_PAGE);
  if (_tradesPage < pages - 1) { _tradesPage++; _renderTradesPage(); }
}

function renderPnlChart(points) {
  const labels = points.map(p => p.time);
  const data   = points.map(p => p.pnl);
  const color  = (data[data.length - 1] ?? 0) >= 0 ? '#34d399' : '#f87171';

  if (pnlChart) {
    pnlChart.data.labels = labels;
    pnlChart.data.datasets[0].data = data;
    pnlChart.data.datasets[0].borderColor = color;
    pnlChart.data.datasets[0].pointBackgroundColor = color;
    pnlChart.update();
    return;
  }

  const ctx = document.getElementById('pnlChart').getContext('2d');
  pnlChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Cumulative PnL (USDT)',
        data,
        borderColor: color,
        backgroundColor: color + '15',
        pointBackgroundColor: color,
        pointRadius: 4,
        tension: 0.3,
        fill: true,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: '#1e293b' } },
        y: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: '#334155' } }
      }
    }
  });
}

// ── Active features chips ────────────────────────────────────────────────────

function renderActiveFeatures(features) {
  const el = document.getElementById('active-features');
  const entries = Object.entries(features || {});
  if (!entries.length) { el.innerHTML = ''; return; }

  const typeColor = { code: '#1e3a5f', auto: '#1a2e1a', pair: '#2d1f3d' };
  const typeText  = { code: '#93c5fd', auto: '#86efac', pair: '#c4b5fd' };

  el.innerHTML = entries.map(([name, f]) => {
    const bg  = typeColor[f.type] || '#1e293b';
    const col = typeText[f.type]  || '#94a3b8';
    const tip = `${f.description || name}\nEnabled: ${f.enabled_at ? new Date(f.enabled_at).toLocaleString() : '?'}`;
    const label = name.length > 20 ? name.slice(0, 18) + '…' : name;
    return `<span title="${tip}"
      style="background:${bg};color:${col};font-size:9px;padding:2px 6px;border-radius:4px;cursor:default;white-space:nowrap">
      ${label}</span>`;
  }).join('');
}

// ── Latest news panel ────────────────────────────────────────────────────────

let _allNewsItems = [];   // persists so filterNewsByChart() can re-render

function renderNews(heartbeats) {
  const hbList = Object.values(heartbeats || {});

  const items = [];
  for (const hb of hbList) {
    const headlines = hb.news_headlines || [];
    const label     = hb.news_label || 'no_data';
    const asset     = hb.asset || '';
    const sym       = asset.replace('/USDT', '');
    const color     = label === 'bullish' ? '#34d399' : label === 'bearish' ? '#f87171' : '#64748b';
    const icon      = label === 'bullish' ? '▲' : label === 'bearish' ? '▼' : '·';
    headlines.forEach(h => items.push({ sym, color, icon, headline: h, label }));
  }

  // Sort: bullish first, then bearish, then neutral
  const order = { bullish: 0, bearish: 1, neutral: 2, no_data: 3 };
  items.sort((a, b) => (order[a.label] ?? 3) - (order[b.label] ?? 3));

  _allNewsItems = items;
  _renderNewsItems(items, null);
}

function filterNewsByChart(sym) {
  // sym = e.g. "XRP" — show that pair's news highlighted, others dimmed
  _renderNewsItems(_allNewsItems, sym);
}

function _renderNewsItems(items, activeSym) {
  const panel = document.getElementById('news-panel');
  if (!items.length) {
    panel.innerHTML = '<p class="text-slate-600 text-xs italic">No news matched for active pairs in the last 24h</p>';
    document.getElementById('news-updated').textContent = 'no data';
    return;
  }

  // When a chart pair is active: show that pair's rows first and full-bright,
  // other pairs dimmed below with a divider
  let rows;
  if (activeSym) {
    const mine   = items.filter(i => i.sym === activeSym);
    const others = items.filter(i => i.sym !== activeSym);
    const makeRow = (item, dim) => `
      <div class="flex items-start gap-2 py-1 border-b border-slate-800 last:border-0"
           style="${dim ? 'opacity:0.35' : ''}">
        <span class="font-bold text-xs font-mono shrink-0 w-12 pt-0.5 cursor-pointer"
              style="color:${item.color}" onclick="loadChart('${item.sym}/USDT')"
              title="Load ${item.sym} chart">${item.sym}</span>
        <span style="color:${item.color};font-size:10px" class="shrink-0 pt-0.5">${item.icon}</span>
        <span class="text-slate-300 text-xs leading-relaxed">${item.headline}</span>
      </div>`;
    const mineHtml   = mine.map(i => makeRow(i, false)).join('');
    const othersHtml = others.map(i => makeRow(i, true)).join('');
    const divider    = others.length && mine.length
      ? `<p class="text-slate-700 text-xs pt-1 pb-0.5">— other pairs —</p>` : '';
    rows = mineHtml + divider + othersHtml;
  } else {
    rows = items.map(item => `
      <div class="flex items-start gap-2 py-1 border-b border-slate-800 last:border-0">
        <span class="font-bold text-xs font-mono shrink-0 w-12 pt-0.5 cursor-pointer"
              style="color:${item.color}" onclick="loadChart('${item.sym}/USDT')"
              title="Load ${item.sym} chart">${item.sym}</span>
        <span style="color:${item.color};font-size:10px" class="shrink-0 pt-0.5">${item.icon}</span>
        <span class="text-slate-300 text-xs leading-relaxed">${item.headline}</span>
      </div>`).join('');
  }

  panel.innerHTML = rows;
  const updated = document.getElementById('news-updated');
  if (updated) updated.textContent = `${items.length} headlines · refreshed ${new Date().toLocaleTimeString()}`;
}

// ── Session window bar ───────────────────────────────────────────────────────

function renderSessionBar(heartbeats) {
  const bar     = document.getElementById('session-bar');
  const hbList  = Object.values(heartbeats || {});
  // Use the first heartbeat that has session data
  const hb      = hbList.find(h => h.session_name != null) || {};
  const active  = hb.session_active;
  const name    = hb.session_name;
  const emoji   = hb.session_emoji  || '';
  const mins    = hb.session_mins_left;
  const vol     = hb.session_vol_mul;

  if (!active || !name) {
    bar.classList.add('hidden');
    return;
  }

  bar.classList.remove('hidden');
  document.getElementById('session-emoji').textContent  = emoji;
  document.getElementById('session-label').textContent  = name + ' Open';
  document.getElementById('session-mins').textContent   = mins  != null ? mins  : '—';
  document.getElementById('session-vol').textContent    = vol   != null ? vol.toFixed(1) : '—';

  const modeEl = document.getElementById('session-mode');
  modeEl.textContent        = '⚡ Breakout Mode';
  modeEl.style.background   = '#1e3a5f';
  modeEl.style.color        = '#93c5fd';
}

// ── Fear/Greed + Black Swan rendering ───────────────────────────────────────

function renderStrategyNotes(notes) {
  const section = document.getElementById('strategy-notes-section');
  const list    = document.getElementById('strategy-notes-list');
  const tsEl    = document.getElementById('strategy-notes-ts');
  if (!notes || !notes.length) { section.classList.add('hidden'); return; }
  section.classList.remove('hidden');
  tsEl.textContent = notes[0]?.ts ? '· ' + notes[0].ts : '';
  list.innerHTML = notes.map(n =>
    `<div class="flex gap-2"><span>${n.icon||'•'}</span><span class="text-slate-300">${n.text}</span></div>`
  ).join('');
}

function renderSentiment(sentiment, data) {
  data = data || {};
  const fg   = sentiment.fear_greed || {};
  const swan = sentiment.black_swan || {};

  // ── Real Fear & Greed gauge (alternative.me) ──────────────────────────────
  const rfg      = sentiment.real_fear_greed || {};
  const rfgScore = rfg.score != null ? parseInt(rfg.score) : null;
  const rfgLabel = rfg.label || 'No data';
  const rfgEmoji = rfg.emoji || '⚪';
  const rfgScoreEl  = document.getElementById('rfg-score-num');
  const rfgLabelEl  = document.getElementById('rfg-label');
  const rfgNeedleEl = document.getElementById('rfg-needle');
  if (rfgScoreEl) rfgScoreEl.textContent = rfgScore != null ? rfgScore : '—';
  if (rfgLabelEl) {
    rfgLabelEl.textContent = rfgEmoji + ' ' + rfgLabel;
    const col = rfgScore == null ? '#64748b' : rfgScore < 20 ? '#ef4444' : rfgScore < 35 ? '#f97316'
              : rfgScore < 50 ? '#eab308' : rfgScore < 65 ? '#64748b' : rfgScore < 80 ? '#10b981' : '#34d399';
    rfgLabelEl.style.color = col;
  }
  if (rfgNeedleEl && rfgScore != null) {
    rfgNeedleEl.setAttribute('transform', `rotate(${(rfgScore / 100) * 180 - 90}, 90, 90)`);
  }

  // ── Hermes Senti-meter gauge ──────────────────────────────────────────────
  const score = fg.score != null ? parseInt(fg.score) : null;
  const label = fg.label || 'No data';
  const emoji = fg.emoji || '⚪';

  const scoreEl  = document.getElementById('fg-score-num');
  const labelEl  = document.getElementById('fg-label');
  const needleEl = document.getElementById('fg-needle');
  const sigEl    = document.getElementById('fg-signals');

  if (scoreEl) scoreEl.textContent = score != null ? score : '—';
  if (labelEl) {
    labelEl.textContent = emoji + ' ' + label;
    // Colour based on zone
    const col = score == null     ? '#64748b'
              : score < 20        ? '#ef4444'
              : score < 35        ? '#f97316'
              : score < 50        ? '#eab308'
              : score < 65        ? '#64748b'
              : score < 80        ? '#10b981'
              :                     '#34d399';
    labelEl.style.color = col;
  }

  // Rotate needle: score 0 → -90deg (left), score 50 → 0deg (top), score 100 → +90deg (right)
  if (needleEl && score != null) {
    const angle = (score / 100) * 180 - 90;
    needleEl.setAttribute('transform', `rotate(${angle}, 90, 90)`);
  }

  if (sigEl && fg.signals) {
    sigEl.innerHTML = fg.signals.map(s =>
      `<p class="text-slate-500 text-xs">${s}</p>`
    ).join('');
  }

  // ── Macro signals grid ────────────────────────────────────────────────────
  // Each signal has an optional chart link — clicking opens TradingView in the chart panel.
  const macroEl = document.getElementById('macro-signals-grid');
  if (macroEl && fg.components) {
    const c = fg.components;
    // Source: all signals are computed from the active pairs' 15m heartbeat data,
    // except BTC Vol which comes from BTC/USDT 1H candles via volatility.py.
    // Also pull regime-level macro data (Total2/Total3/BTC.D) from heartbeat
    const hbs = Object.values(data.heartbeats || {});
    const regimeInfo = hbs.length ? (hbs[0] || {}) : {};
    const total2 = regimeInfo.total2_bias || sentiment.total2_bias || '—';
    const total3 = regimeInfo.total3_bias || sentiment.total3_bias || '—';
    const altSeason = regimeInfo.alt_season || false;
    const btcDomRising = regimeInfo.btc_dom_rising || false;

    const biasColor = b => b === 'bullish' ? '#34d399' : b === 'bearish' ? '#ef4444' : '#64748b';

    const macroItems = [
      { label: 'Avg RSI (15m)',
        value: c.avg_rsi != null ? c.avg_rsi.toFixed(1) : '—',
        note: c.avg_rsi < 35 ? 'Oversold — entry zone' : c.avg_rsi > 65 ? 'Overbought — caution' : 'Neutral',
        tip: '15m RSI averaged across all active pairs. Score: 30pts max. RSI 25→0pts, RSI 75→30pts.',
        color: c.avg_rsi < 35 ? '#ef4444' : c.avg_rsi > 65 ? '#34d399' : '#64748b',
        pair: 'BTC/USDT' },
      { label: '% Pairs above 50MA',
        value: c.pct_above_ma != null ? c.pct_above_ma.toFixed(0) + '%' : '—',
        note: c.pct_above_ma < 30 ? 'Mostly bearish' : c.pct_above_ma > 70 ? 'Mostly bullish' : 'Mixed',
        tip: '% of active pairs whose price is above their 50-period MA on 15m. Score: 25pts max.',
        color: c.pct_above_ma < 30 ? '#ef4444' : c.pct_above_ma > 70 ? '#34d399' : '#64748b',
        pair: 'BTC/USDT' },
      { label: '% Pairs above VWAP',
        value: c.pct_above_vwap != null ? c.pct_above_vwap.toFixed(0) + '%' : '—',
        note: c.pct_above_vwap < 30 ? 'Below VWAP — bearish' : c.pct_above_vwap > 70 ? 'Above VWAP — bullish' : 'Split',
        tip: 'VWAP = Volume Weighted Average Price (intraday anchor). % of pairs trading above it. Score: 20pts max.',
        color: c.pct_above_vwap < 30 ? '#ef4444' : c.pct_above_vwap > 70 ? '#34d399' : '#64748b',
        pair: 'BTC/USDT' },
      { label: 'BTC Realised Vol (1H)',
        value: c.btc_vol_pct != null ? c.btc_vol_pct.toFixed(2) + '%' : '—',
        note: c.btc_vol_pct > 3 ? 'High vol — panic / bearish' : c.btc_vol_pct < 1 ? 'Low vol — calm / bullish' : 'Normal',
        tip: 'BTC 1H realised volatility. High vol = panic (bearish signal). Low vol = calm bull run. Score: 15pts max (inverted).',
        color: c.btc_vol_pct > 3 ? '#ef4444' : c.btc_vol_pct < 1 ? '#34d399' : '#64748b',
        pair: 'BTC/USDT' },
      { label: 'Price Momentum',
        value: c.avg_rng_pos != null ? (c.avg_rng_pos*100).toFixed(0) + '%' : '—',
        note: c.avg_rng_pos < 0.3 ? 'Near 20-bar low — oversold' : c.avg_rng_pos > 0.7 ? 'Near 20-bar high — overbought' : 'Mid range',
        tip: 'Where price sits within its recent 20-bar high/low range across all pairs. 0% = at lows, 100% = at highs. Score: 10pts max.',
        color: c.avg_rng_pos < 0.3 ? '#ef4444' : c.avg_rng_pos > 0.7 ? '#34d399' : '#64748b',
        pair: 'BTC/USDT' },
    ];

    // Compute total score and what it triggers
    const totalScore = fg.score != null ? fg.score : null;
    const scoreColor = totalScore == null ? '#64748b'
      : totalScore <= 10  ? '#ef4444'   // Extreme Fear → CRITICAL halt
      : totalScore <= 18  ? '#f97316'   // Fear → WARNING banner
      : totalScore >= 93  ? '#f59e0b'   // Extreme Greed → WARNING banner
      : totalScore >= 85  ? '#fbbf24'   // Greed warning
      : '#34d399';
    const scoreTrigger = totalScore == null ? ''
      : totalScore <= 10  ? '🚨 CRITICAL — entries halted, open longs closing'
      : totalScore <= 18  ? '⚠️ Oversold — reduce sizes, wait for stabilisation'
      : totalScore >= 93  ? '🔴 Extreme overbought — reversal likely, no new longs'
      : totalScore >= 85  ? '⚠️ Overbought — tighten stops, avoid chasing entries'
      : '✅ Clear — conditions normal';

    // MACD breadth — build per-pair data for clickable chips
    const hbList    = Object.values(data.heartbeats || {});
    const macdBull  = hbList.filter(h => h.macd_bull_15m).length;
    const macdBear  = hbList.filter(h => h.macd_bear_15m).length;
    const histVals  = hbList.map(h => h.macd_hist_15m).filter(v => v != null);
    const avgHist   = histVals.length ? histVals.reduce((a,b)=>a+b,0)/histVals.length : null;
    const bullPairs = hbList.filter(h => h.macd_hist_15m != null && h.macd_hist_15m > 0).length;
    const bearPairs = hbList.filter(h => h.macd_hist_15m != null && h.macd_hist_15m < 0).length;
    const macdColor = bullPairs > bearPairs ? '#34d399' : bearPairs > bullPairs ? '#ef4444' : '#64748b';
    // Per-pair chips: crossovers first, then sorted by histogram magnitude
    const macdPairs = hbList
      .filter(h => h.asset && h.macd_hist_15m != null)
      .sort((a, b) => {
        const aCross = (a.macd_bull_15m || a.macd_bear_15m) ? 1 : 0;
        const bCross = (b.macd_bull_15m || b.macd_bear_15m) ? 1 : 0;
        if (bCross !== aCross) return bCross - aCross;
        return Math.abs(b.macd_hist_15m) - Math.abs(a.macd_hist_15m);
      });
    const pairChips = macdPairs.map(h => {
      const sym   = (h.asset || '').replace('/USDT','');
      const cross = h.macd_bull_15m ? '🟢' : h.macd_bear_15m ? '🔴' : (h.macd_hist_15m > 0 ? '↑' : '↓');
      const col   = h.macd_bull_15m ? '#34d399' : h.macd_bear_15m ? '#ef4444' : (h.macd_hist_15m > 0 ? '#34d399' : '#f87171');
      return `<span onclick="loadChart('${h.asset}')" title="Load ${sym} chart"
        style="cursor:pointer;color:${col};background:#1e293b;border-radius:4px;padding:1px 5px;font-size:10px;margin:1px;display:inline-block">
        ${cross} ${sym}</span>`;
    }).join('');
    // MACD card is rendered separately (not via macroItems) so chips are clickable
    const macdCardHtml = `
      <div class="bg-slate-900 rounded-lg px-3 py-2">
        <p class="text-slate-500 text-xs mb-0.5">MACD Breadth (15m) <span class="text-indigo-500">↗</span></p>
        <p class="font-bold text-sm font-mono mb-1" style="color:${macdColor}">${avgHist != null ? (avgHist >= 0 ? '+' : '') + avgHist.toFixed(5) : '—'}
          <span class="text-slate-500 font-normal text-xs ml-1">avg histogram · ${bullPairs}↑ ${bearPairs}↓ pairs</span>
        </p>
        <div class="flex flex-wrap gap-0.5 mb-1">${pairChips || '<span class="text-slate-600 text-xs">No data</span>'}</div>
        <p class="text-slate-600 text-xs">🟢 bull crossover &nbsp;🔴 bear crossover &nbsp;↑ hist+ &nbsp;↓ hist− &nbsp;· click pair to view chart</p>
      </div>`;
    // Row 4: Total3 + Total2
    macroItems.push({
      label: 'Total3 (alts)',
      value: total3 !== '—' ? total3.toUpperCase() : '—',
      note:  btcDomRising ? 'BTC dominance rising' : 'BTC dom stable/falling',
      tip:   'SOL + BNB + ADA basket vs BTC. Bullish = small caps outperforming. BTC.D rising = capital flowing back to BTC.',
      color: biasColor(total3),
      pair:  'TOTAL3',
    });
    macroItems.push({
      label: 'Total2 (ex-BTC)',
      value: total2 !== '—' ? total2.toUpperCase() : '—',
      note:  altSeason ? '🌟 Alt season active' : 'No alt season',
      tip:   'Total crypto market cap excluding BTC (ETH+alts). Computed from ETH 1H candles vs BTC dominance. Bullish = alts outperforming.',
      color: biasColor(total2),
      pair:  'TOTAL2',
    });

    const breakdown = c.rsi_score != null
      ? `RSI ${c.rsi_score.toFixed(0)}/30 · MA ${c.ma_score.toFixed(0)}/25 · VWAP ${c.vwap_score.toFixed(0)}/20 · Vol ${(c.vol_score||0).toFixed(0)}/15 · Mom ${(c.mom_score||0).toFixed(0)}/10`
      : '—';

    const verdict = totalScore == null ? { label: '—', color: '#64748b' }
      : totalScore <= 10  ? { label: 'OVERSOLD',   color: '#ef4444' }
      : totalScore <= 18  ? { label: 'BEARISH',    color: '#f87171' }
      : totalScore >= 93  ? { label: 'OVERBOUGHT', color: '#ef4444' }
      : totalScore >= 85  ? { label: 'OVERBOUGHT', color: '#f59e0b' }
      : totalScore >= 65  ? { label: 'BULLISH',    color: '#34d399' }
      : totalScore <= 35  ? { label: 'BEARISH',    color: '#f87171' }
      : { label: 'SIDEWAYS', color: '#818cf8' };

    // Insert MACD card after Price Momentum (5th card = index 4), before Total3/Total2
    const macroCardsList = macroItems.filter(Boolean).map(m => `
      <div class="bg-slate-900 rounded-lg px-3 py-2 cursor-default" title="${m.tip}"
           ${m.pair ? `onclick="loadChart('${m.pair}')" style="cursor:pointer"` : ''}>
        <p class="text-slate-500 text-xs mb-0.5">${m.label}${m.pair ? ' <span class="text-indigo-500">↗</span>' : ''}</p>
        <p class="font-bold text-sm font-mono" style="color:${m.color}">${m.value}</p>
        <p class="text-slate-600 text-xs">${m.note}</p>
      </div>`);
    // Inject MACD card after index 4 (Price Momentum)
    macroCardsList.splice(5, 0, macdCardHtml);
    macroEl.innerHTML = macroCardsList.join('') + `
    <div class="col-span-2 bg-slate-900 rounded-lg px-3 py-2 border border-slate-700">
      <div class="flex items-center justify-between mb-1">
        <p class="text-slate-500 text-xs">SCORE BREAKDOWN — what each signal contributed</p>
        <span class="font-bold text-sm px-2 py-0.5 rounded" style="color:${verdict.color};background:${verdict.color}22;border:1px solid ${verdict.color}44">${verdict.label}</span>
      </div>
      <p class="font-mono text-xs text-slate-300 mb-1">${breakdown}</p>
      <div class="flex items-center justify-between">
        <span class="text-slate-500 text-xs">${scoreTrigger}</span>
        <span class="font-bold text-lg font-mono" style="color:${scoreColor}">${totalScore != null ? totalScore + '/100' : '—'}</span>
      </div>
      <p class="text-slate-600 text-xs mt-1">Thresholds: ≤10 entries halted · ≤18 reduce sizes · ≥85 tighten stops · ≥93 no new longs</p>
    </div>`;
  }

  // ── Black Swan banner ─────────────────────────────────────────────────────
  const banner   = document.getElementById('swan-banner');
  const swanIcon = document.getElementById('swan-icon');
  const swanTitle= document.getElementById('swan-title');
  const swanEvts = document.getElementById('swan-events');
  const swanAct  = document.getElementById('swan-action');

  const level = swan.level || 'normal';

  if (!banner) return;

  if (level === 'normal') {
    banner.classList.add('hidden');
    return;
  }

  banner.classList.remove('hidden');

  if (level === 'critical') {
    banner.style.background    = '#1a0808';
    banner.style.borderColor   = '#7f1d1d';
    swanIcon.textContent       = '🚨';
    swanTitle.style.color      = '#f87171';
    swanTitle.textContent      = '🚨 BLACK SWAN — CRITICAL (all entries halted)';
  } else {
    banner.style.background    = '#1a1000';
    banner.style.borderColor   = '#78350f';
    swanIcon.textContent       = '⚠️';
    swanTitle.style.color      = '#fbbf24';
    swanTitle.textContent      = '⚠️ MARKET WARNING — advisory only, trades not blocked';
  }

  const eventTypeLabels = {
    flash_crash:    e => `💥 Flash crash: ${e.asset} ${(e.move_pct||0).toFixed(1)}%`,
    price_shock:    e => `📉 Price shock: ${e.asset} ${(e.move_pct||0).toFixed(1)}%`,
    cascade_crash:  e => `🌊 Cascade crash: ${(e.pairs||[]).join(', ')} avg ${(e.avg_move||0).toFixed(1)}%`,
    cascade_pump:   e => `🚀 Cascade pump: ${(e.pairs||[]).join(', ')} avg +${(e.avg_move||0).toFixed(1)}%`,
    feed_anomaly:   e => `📡 Feed anomaly: ${e.asset} — ${e.detail}`,
    extreme_fear:   e => `😱 Extreme Bearish senti — ${e.score}/100`,
    fear_warning:   e => `😟 Bearish senti warning — ${e.score}/100`,
    extreme_greed:  e => `🚀 Extreme Bullish senti — ${e.score}/100 — reversal risk`,
    macro_extreme:  e => `🌋 Macro extreme vol ${(e.vol||0).toFixed(1)}%`,
  };

  const events = swan.events || [];
  if (swanEvts) {
    swanEvts.innerHTML = events.map(e => {
      const fn = eventTypeLabels[e.type];
      return `<p>${fn ? fn(e) : JSON.stringify(e)}</p>`;
    }).join('') || '<p>Unknown event</p>';
  }

  if (swanAct) swanAct.textContent = swan.action || '';
}

async function refresh() {
  const spinner = document.getElementById('refresh-spinner');
  spinner.style.display = 'inline-block';
  try {
    const res  = await fetch('/api/state');
    const data = await res.json();

    // Header
    const mode = data.strategy?.mode || (data.heartbeats && Object.values(data.heartbeats)[0]?.mode) || 'paper';
    document.getElementById('mode-badge').textContent = mode.toUpperCase();
    document.getElementById('mode-badge').className =
      'badge ' + (mode === 'live' ? 'bg-red-900 text-red-300' : 'bg-slate-700 text-slate-300');

    const regime = data.regime || 'unknown';
    document.getElementById('regime-badge').textContent = regimeLabel(regime, data.is_sideways);
    document.getElementById('regime-badge').className = 'badge ' + regimeClass(regime);

    const ver = data.strategy?.version || data.strategy_version;
    document.getElementById('strategy-ver').textContent = ver ? 'v' + ver : '';

    // Active features chips
    renderActiveFeatures(data.active_features || {});

    // Total capital + deployed
    const totalCap = data.total_capital_usdt || data.drawdown?.total_capital || 1000;
    const deployed = Object.values(data.heartbeats || {})
      .filter(hb => hb.open_position)
      .reduce((s, hb) => s + parseFloat(hb.open_position.usdt_deployed || 0), 0);
    document.getElementById('stat-capital').textContent = '$' + parseFloat(totalCap).toLocaleString(undefined, {minimumFractionDigits:0, maximumFractionDigits:0});
    document.getElementById('stat-capital-deployed').textContent = '$' + deployed.toFixed(2) + ' deployed';

    // Stats
    const p = data.portfolio;
    const pnlEl = document.getElementById('stat-pnl');
    pnlEl.textContent = (p.total_pnl_usdt >= 0 ? '+$' : '-$') + Math.abs(p.total_pnl_usdt).toFixed(4);
    pnlEl.className = 'text-2xl font-bold ' + (p.total_pnl_usdt >= 0 ? 'pnl-pos' : 'pnl-neg');

    // Net PnL (after fees + slippage)
    const netPnl = p.total_net_pnl_usdt ?? p.total_pnl_usdt;
    document.getElementById('stat-net-pnl').textContent =
      'net ' + (netPnl >= 0 ? '+$' : '-$') + Math.abs(netPnl).toFixed(4);

    const wrEl = document.getElementById('stat-winrate');
    wrEl.textContent = p.total_trades > 0 ? p.win_rate + '%' : '—';
    wrEl.className = 'text-2xl font-bold ' + (p.win_rate >= 50 ? 'pnl-pos' : 'pnl-neg');

    document.getElementById('stat-trades').textContent =
      p.wins + 'W / ' + p.losses + 'L';

    const dd = data.drawdown?.drawdown_pct ?? 0;
    const ddEl = document.getElementById('stat-dd');
    ddEl.textContent = dd.toFixed(2) + '%';
    ddEl.className = 'text-2xl font-bold ' + (dd > 5 ? 'pnl-neg' : 'text-white');

    // Fees + slippage card
    const totalCosts = (p.total_commission_usdt || 0) + (p.total_slippage_usdt || 0);
    document.getElementById('stat-costs').textContent = '-$' + totalCosts.toFixed(4);
    const dragPct = p.cost_drag_pct || 0;
    document.getElementById('stat-cost-drag').textContent =
      dragPct.toFixed(1) + '% of gross' +
      (p.total_commission_usdt ? ' · fees $' + (p.total_commission_usdt||0).toFixed(4) : '');

    // Chart
    if (data.cum_pnl?.length) renderPnlChart(data.cum_pnl);
    else renderPnlChart([]);

    // Fetch controls state and update badge
    try {
      const ctrlRes = await fetch('/api/controls');
      const ctrl = await ctrlRes.json();
      _savedControls = ctrl;
      updateControlsBadge(ctrl);
    } catch(e) {}

    // Pairs + populate open positions for live price polling
    const heartbeats = data.heartbeats || {};
    renderPairs(heartbeats);

    // Populate manual-entry asset dropdown
    const sel = document.getElementById('manual-asset');
    if (sel) {
      const existing = Array.from(sel.options).map(o => o.value);
      Object.keys(heartbeats).forEach(asset => {
        if (!existing.includes(asset)) {
          const opt = document.createElement('option');
          opt.value = asset; opt.textContent = asset;
          sel.appendChild(opt);
        }
      });
    }
    _openPositions = {};
    for (const [asset, hb] of Object.entries(heartbeats)) {
      if (hb.open_position && hb.open_position.entry_price) {
        _openPositions[asset] = {
          ...hb.open_position,
          take_profit_pct: hb.open_position.take_profit_pct || hb.regime_params?.take_profit_pct || 3.0,
          stop_loss_pct:   hb.open_position.stop_loss_pct   || hb.regime_params?.stop_loss_pct   || 1.8,
        };
      }
    }

    // Trades
    renderTrades(data.recent_trades || []);

    // Hermes strategy notes
    renderStrategyNotes(data.strategy_notes || []);

    // Fear/Greed + Black Swan
    renderSentiment(data.sentiment || {}, data);

    // Session window bar
    renderSessionBar(data.heartbeats || {});

    // News panel
    renderNews(data.heartbeats || {});

    // Timestamp
    document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();
  } catch(e) {
    console.error('refresh failed', e);
  } finally {
    spinner.style.display = 'none';
  }
}

// Track open positions for live price updates
let _openPositions = {};  // { asset: { entry_price, direction, usdt_deployed, ... } }

// Fast 5-second loop: update prices + PnL for open positions directly from Binance
async function refreshLivePrices() {
  const assets = Object.keys(_openPositions);
  if (!assets.length) return;
  await Promise.all(assets.map(async asset => {
    try {
      const sym = asset.replace('/', '');
      const r   = await fetch(`https://api.binance.com/api/v3/ticker/price?symbol=${sym}`);
      if (!r.ok) return;
      const data  = await r.json();
      const price = parseFloat(data.price);
      if (!price) return;
      const pos   = _openPositions[asset];
      const mult  = pos.direction === 'long' ? 1 : -1;
      const entry = parseFloat(pos.entry_price);
      const pnlPct   = ((price - entry) / entry) * mult * 100;
      const deployed = parseFloat(pos.usdt_deployed || 0);
      const pnlUsd   = pnlPct / 100 * deployed;
      // Use structural levels when available for bar progress
      const tpPrice  = pos.tp_price ? parseFloat(pos.tp_price)
                     : (pos.direction === 'long' ? entry*(1+(parseFloat(pos.take_profit_pct||3.0)/100))
                                                 : entry*(1-(parseFloat(pos.take_profit_pct||3.0)/100)));
      const tpPct    = pos.tp_pct ? parseFloat(pos.tp_pct) : Math.abs((tpPrice-entry)/entry*100);
      const pnlColor = pnlPct >= 0 ? '#34d399' : '#f87171';
      const barPct   = Math.min(Math.abs(pnlPct) / tpPct * 100, 100).toFixed(1);

      // Update all live fields in the expanded card
      const cardEls = document.querySelectorAll(`[data-live-asset="${asset}"]`);
      cardEls.forEach(el => {
        if (el.dataset.field === 'price')   el.textContent = '$' + price.toFixed(6);
        if (el.dataset.field === 'pnlpct')  { el.textContent = (pnlPct>=0?'+':'') + pnlPct.toFixed(3) + '%'; el.style.color = pnlColor; }
        if (el.dataset.field === 'pnlusd')  { el.textContent = (pnlUsd>=0?'+':'') + '$' + pnlUsd.toFixed(4); el.style.color = pnlColor; }
        if (el.dataset.field === 'bar')     { el.style.width = barPct + '%'; el.style.background = pnlColor; }
        if (el.dataset.field === 'barlabel'){ el.textContent = (pnlPct>=0?'+':'') + pnlPct.toFixed(2) + '% of ' + tpPct.toFixed(2) + '% target'; }
      });
    } catch(e) { /* silent */ }
  }));
}

// ── Control helpers ──────────────────────────────────────────────────────────

async function ctrlPost(body) {
  try {
    const r = await fetch('/api/control', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const data = await r.json();
    if (!data.ok) { alert('Error: ' + (data.error || JSON.stringify(data))); return null; }
    return data;
  } catch(e) { alert('Request failed: ' + e); return null; }
}

async function doAllStop() {
  if (!confirm('⛔ ALL STOP — close all open positions and halt new entries?')) return;
  const res = await ctrlPost({action: 'all_stop'});
  if (res) { showToast('⛔ ALL STOP activated — positions will close on next tick'); refresh(); }
}

async function doResume() {
  const res = await ctrlPost({action: 'resume'});
  if (res) { showToast('▶ Trading resumed'); refresh(); }
}

async function doResetStats() {
  const input = prompt('⚠️ This will wipe all trade history, win rate, PnL and drawdown.\\n\\nTrades are archived first but the dashboard resets to zero.\\n\\nType RESET to confirm:');
  if (input !== 'RESET') { alert('Cancelled — you must type RESET exactly.'); return; }
  try {
    const r = await fetch('/api/reset-stats', { method: 'POST' });
    const d = await r.json();
    if (d.status === 'ok' || d.status === 'partial') {
      alert('✅ Stats reset!\\n' + d.archived.join('\\n') + (d.errors.length ? '\\n⚠️ ' + d.errors.join(', ') : ''));
      await refreshData();  // reload dashboard immediately
    } else {
      alert('❌ Reset failed: ' + JSON.stringify(d));
    }
  } catch(e) {
    alert('❌ Error: ' + e);
  }
}

async function doTestTelegram() {
  showToast('📨 Sending test message…');
  try {
    const r = await fetch('/api/test-telegram');
    const j = await r.json();
    showToast(j.ok ? '✅ ' + j.message : '❌ ' + j.message, j.ok ? 'green' : 'red');
  } catch(e) {
    showToast('❌ Request failed: ' + e, 'red');
  }
}

async function doForceExit(asset) {
  if (!confirm(`Force-exit ${asset}?`)) return;
  const res = await ctrlPost({action: 'exit', asset});
  if (res) { showToast(`✕ Exit queued for ${asset} — will close on next tick`); }
}

async function doSetLeverage(asset, leverage) {
  if (!leverage) {
    await ctrlPost({action: 'clear_leverage', asset});
    if (!_savedControls.leverage_overrides) _savedControls.leverage_overrides = {};
    delete _savedControls.leverage_overrides[asset];
    showToast(`${asset} leverage reset to regime default`);
  } else {
    const res = await ctrlPost({action: 'set_leverage', asset, leverage: parseFloat(leverage)});
    if (res) {
      if (!_savedControls.leverage_overrides) _savedControls.leverage_overrides = {};
      _savedControls.leverage_overrides[asset] = parseFloat(leverage);
      showToast(`${asset} leverage override → ${leverage}x (saved)`);
    }
  }
}

async function doManualEntry() {
  const asset     = document.getElementById('manual-asset').value;
  const direction = document.getElementById('manual-dir').value;
  const leverage  = document.getElementById('manual-lev').value;
  if (!asset) { alert('Select a pair first'); return; }
  if (!confirm(`Queue manual ${direction.toUpperCase()} entry for ${asset}${leverage ? ' @ '+leverage+'x' : ''}?`)) return;
  const body = {action: 'enter', asset, direction};
  if (leverage) body.leverage = parseFloat(leverage);
  const res = await ctrlPost(body);
  if (res) showToast(`→ ${direction.toUpperCase()} entry queued for ${asset} — fires on next tick`);
}

function showToast(msg) {
  let t = document.getElementById('toast');
  if (!t) {
    t = document.createElement('div');
    t.id = 'toast';
    t.style.cssText = 'position:fixed;bottom:24px;left:50%;transform:translateX(-50%);' +
      'background:#1e293b;border:1px solid #334155;color:#e2e8f0;padding:10px 20px;' +
      'border-radius:8px;font-size:0.875rem;z-index:9999;transition:opacity .3s;';
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.style.opacity = '1';
  clearTimeout(t._to);
  t._to = setTimeout(() => { t.style.opacity = '0'; }, 3500);
}

function updateControlsBadge(ctrl) {
  const badge = document.getElementById('all-stop-badge');
  if (!badge) return;
  if (ctrl && ctrl.all_stop) {
    badge.textContent = '⛔ ALL STOP';
    badge.className = 'badge bg-red-950 text-red-300 border border-red-700';
    document.getElementById('controls-panel').style.borderColor = '#7f1d1d';
  } else {
    badge.textContent = '● ACTIVE';
    badge.className = 'badge bg-slate-700 text-slate-400';
    document.getElementById('controls-panel').style.borderColor = '';
  }
  // populate pending entries/exits info if any
  const exits = (ctrl && ctrl.manual_exits) || [];
  const pendings = (ctrl && ctrl.pending_entries) || [];
  // could add a small status line here if needed
}

// Initial load + 30s full refresh + 5s live price refresh
refresh();
setInterval(refresh, 30000);
setInterval(refreshLivePrices, 5000);
</script>

<!-- TradingView widget library -->
<script src="https://s3.tradingview.com/tv.js"></script>
<script>
let _currentTvSymbol = 'BINANCE:BTCUSDTPERP';
let _tvWidget = null;

function tvSymbol(asset) {
  // Handle special index symbols (no exchange prefix, no PERP suffix)
  const indexSymbols = ['TOTAL', 'TOTAL2', 'TOTAL3', 'BTC.D', 'OTHERS.D'];
  const clean = (asset || 'BTC/USDT').replace('/USDT', '').replace('USDT', '');
  if (indexSymbols.includes(clean)) return clean;
  return `BINANCE:${clean}USDTPERP`;
}

function loadChart(asset) {
  const sym  = tvSymbol(asset);
  const ivEl = document.getElementById('chart-interval');
  const iv   = ivEl ? ivEl.value : '60';

  _currentTvSymbol = sym;

  // Update label and expand link
  const pairLabel = (asset || 'BTC/USDT').replace('/USDT', '') + '/USDT';
  document.getElementById('chart-symbol-label').textContent = pairLabel;
  document.getElementById('tv-expand-btn').href =
    `https://www.tradingview.com/chart/?symbol=${sym}`;

  // Filter news panel to this pair
  filterNewsByChart(pairLabel.replace('/USDT', ''));

  // Scroll chart into view
  document.getElementById('tv-chart-container')?.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Clear and recreate widget
  const container = document.getElementById('tv-chart-container');
  container.innerHTML = '';
  const div = document.createElement('div');
  div.id = 'tv_widget_inner';
  container.appendChild(div);

  if (typeof TradingView === 'undefined') return;

  _tvWidget = new TradingView.widget({
    container_id:       'tv_widget_inner',
    symbol:             sym,
    interval:           iv,
    timezone:           'Etc/UTC',
    theme:              'dark',
    style:              '1',
    locale:             'en',
    width:              '100%',
    height:             480,
    hide_top_toolbar:   false,
    hide_legend:        false,
    allow_symbol_change: false,
    save_image:         false,
    studies: [
      'RSI@tv-basicstudies',
      'MACD@tv-basicstudies',
    ],
  });
}

function reloadChart() {
  // Re-load with current symbol + new interval
  const sym   = _currentTvSymbol;
  const asset = sym.replace('BINANCE:', '').replace('USDTPERP', '') + '/USDT';
  loadChart(asset);
}

// Load default chart after TV library is ready
window.addEventListener('load', () => {
  setTimeout(() => loadChart('BTC/USDT'), 800);
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Start server (called from run.py as an asyncio task)
# ---------------------------------------------------------------------------

async def start(host: str = "0.0.0.0", port: int = PORT):
    import uvicorn
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    print(f"[dashboard] starting on http://{host}:{port}", flush=True)
    await server.serve()
