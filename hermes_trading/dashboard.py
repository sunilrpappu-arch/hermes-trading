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
                "total_pnl_usdt": 0.0, "total_pnl_pct": 0.0, "best_trade": None, "worst_trade": None}

    wins   = [t for t in trades if (t.get("pnl_pct") or 0) > 0]
    losses = [t for t in trades if (t.get("pnl_pct") or 0) <= 0]
    total_pnl_usdt = sum(_trade_pnl_usdt(t) for t in trades)
    total_pnl_pct  = sum(t.get("pnl_pct", 0) or 0 for t in trades)

    best  = max(trades, key=lambda t: t.get("pnl_pct", 0))
    worst = min(trades, key=lambda t: t.get("pnl_pct", 0))

    return {
        "total_trades":   len(trades),
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate":       round(len(wins) / len(trades) * 100, 1),
        "total_pnl_usdt": round(total_pnl_usdt, 4),
        "total_pnl_pct":  round(total_pnl_pct * 100, 3),
        "best_trade":     best,
        "worst_trade":    worst,
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

    # Detect current regime from any active heartbeat
    regime     = "unknown"
    is_sideways = False
    for hb in heartbeats.values():
        if hb.get("regime"):
            regime      = hb["regime"]
            is_sideways = hb.get("is_sideways", False)
            break

    return JSONResponse({
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "strategy":        strategy,
        "strategy_version": strategy.get("version", "?"),
        "regime":          regime,
        "is_sideways":     is_sideways,
        "portfolio":     stats,
        "drawdown":      drawdown,
        "cum_pnl":       cum_pnl,
        "heartbeats":    heartbeats,
        "recent_trades": list(reversed(trades[-20:])),  # last 20, newest first
    })


@app.get("/api/trades")
async def api_trades():
    return JSONResponse({"trades": list(reversed(_read_trades()))})


@app.get("/api/pairs")
async def api_pairs():
    return JSONResponse({"pairs": _read_heartbeats()})


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
    <span id="refresh-spinner" class="spinner"></span>
  </div>
</div>

<!-- Summary cards -->
<div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
  <div class="card">
    <p class="text-slate-400 text-xs mb-1">TOTAL PnL</p>
    <p id="stat-pnl" class="text-2xl font-bold">$0.00</p>
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

<!-- Recent trades -->
<div class="card mb-6">
  <p class="text-slate-400 text-xs font-semibold mb-3">RECENT TRADES</p>
  <div class="overflow-x-auto">
    <table id="trades-table">
      <thead>
        <tr>
          <th>Pair</th><th>Dir</th><th>Entry</th><th>Exit</th>
          <th>Capital</th><th>Qty</th><th>PnL %</th><th>PnL $</th><th>Reason</th><th>Regime</th><th>Strategy</th><th>Time</th>
        </tr>
      </thead>
      <tbody id="trades-body">
        <tr><td colspan="10" class="text-center text-slate-500 py-8">No trades yet</td></tr>
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
    const posLabel = posDir ? posDir.toUpperCase() : 'FLAT';
    const rsi      = hb.rsi_15m ?? '—';
    const rng      = hb.rng_pos != null ? Math.round(hb.rng_pos * 100) + '%' : null;
    const tsAge    = hb.timestamp ? Math.round((Date.now() - new Date(hb.timestamp).getTime()) / 1000) : null;
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
      const slPct    = parseFloat(pos.stop_loss_pct   || hb.regime_params?.stop_loss_pct   || 1.8);
      const tpPct    = parseFloat(pos.take_profit_pct || hb.regime_params?.take_profit_pct || 3.0);
      const slPrice  = pos.direction === 'long' ? entry*(1-slPct/100) : entry*(1+slPct/100);
      const tpPrice  = pos.direction === 'long' ? entry*(1+tpPct/100) : entry*(1-tpPct/100);
      const pnlColor = pnlPct >= 0 ? '#34d399' : '#f87171';
      const barPct   = Math.min(Math.abs(pnlPct) / tpPct * 100, 100).toFixed(1);
      const signals    = (pos.htf_signals || []).join(', ') || null;
      const regime     = pos.pair_regime || '—';
      const pos_leverage = parseFloat(pos.leverage || 1.0);
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
          <div><span class="text-slate-500">Stop </span><span class="text-red-400 font-mono">$${slPrice.toFixed(6)}</span></div>
          <div><span class="text-slate-500">Target </span><span class="text-emerald-400 font-mono">$${tpPrice.toFixed(6)}</span></div>
          <div><span class="text-slate-500">RSI </span><span class="text-slate-300">${rsi}${rng?' · '+rng:''}</span></div>
        </div>

        <div class="w-full rounded-full h-1.5 mb-1" style="background:#1e293b">
          <div class="h-1.5 rounded-full transition-all duration-500" data-live-asset="${asset}" data-field="bar" style="width:${barPct}%;background:${pnlColor}"></div>
        </div>
        <div class="flex justify-between" style="font-size:0.65rem;color:#475569">
          <span>SL -${slPct}%</span>
          <span data-live-asset="${asset}" data-field="barlabel">${pnlPct>=0?'+':''}${pnlPct.toFixed(2)}% of ${tpPct}% target</span>
          <span>TP +${tpPct}%</span>
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
            <span class="text-slate-400 text-xs">(now ${pos_leverage}x · notional $${(parseFloat(pos.notional_usdt||deployed)).toFixed(0)})</span>
          </div>
        </div>
      </div>`;
    }

    // ── COMPACT card for flat pairs ──
    return `
    <div class="flex items-center justify-between bg-slate-900 rounded-lg px-3 py-2 cursor-pointer hover:bg-slate-800 transition-colors"
         onclick="loadChart('${asset}')" title="Click to load chart">
      <div>
        <span class="font-semibold text-sm text-white">${asset.replace('/USDT','')}/USDT</span>
        <span class="${posClass} text-xs font-bold ml-2">${posLabel}</span>
      </div>
      <div class="text-right text-xs text-slate-400">
        <div>$${parseFloat(hb.price || 0).toFixed(4)}</div>
        <div>RSI ${rsi}${rng ? ' · rng ' + rng : ''}</div>
        <div style="color:${ageColor}">${ageStr}</div>
      </div>
    </div>`;
  }).join('');
}

function renderTrades(trades) {
  const tbody = document.getElementById('trades-body');
  if (!trades.length) return;
  tbody.innerHTML = trades.map(t => {
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
    const regime = t.regime_at_entry || '—';
    const ver    = t.strategy_version ? 'v' + t.strategy_version : '—';
    const capital = t.usdt_deployed != null ? '$' + parseFloat(t.usdt_deployed).toFixed(2) : '—';
    const qty     = t.qty != null && t.qty > 0
      ? parseFloat(t.qty).toPrecision(4)
      : (t.usdt_deployed && t.entry_price
          ? (parseFloat(t.usdt_deployed) / parseFloat(t.entry_price)).toPrecision(4)
          : '—');
    return `<tr>
      <td><a href="${tvUrl(t.asset||'')}" target="_blank" class="font-medium text-white hover:text-indigo-400 transition-colors">${(t.asset||'').replace('/USDT','')}/USDT ↗</a></td>
      <td>${dir}</td>
      <td>${parseFloat(t.entry_price||0).toFixed(4)}</td>
      <td>${parseFloat(t.exit_price||0).toFixed(4)}</td>
      <td class="text-slate-300">${capital}</td>
      <td class="text-slate-400 text-xs">${qty}</td>
      <td>${pnlPct}</td>
      <td>${pnlUsd}</td>
      <td><span class="text-slate-400">${reason}</span></td>
      <td><span class="badge ${regimeClass(regime)} text-xs">${regime}</span></td>
      <td class="text-slate-500">${ver}</td>
      <td class="text-slate-500 text-xs">${dt}</td>
    </tr>`;
  }).join('');
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

    // Stats
    const p = data.portfolio;
    const pnlEl = document.getElementById('stat-pnl');
    pnlEl.textContent = (p.total_pnl_usdt >= 0 ? '+$' : '-$') + Math.abs(p.total_pnl_usdt).toFixed(4);
    pnlEl.className = 'text-2xl font-bold ' + (p.total_pnl_usdt >= 0 ? 'pnl-pos' : 'pnl-neg');

    const wrEl = document.getElementById('stat-winrate');
    wrEl.textContent = p.total_trades > 0 ? p.win_rate + '%' : '—';
    wrEl.className = 'text-2xl font-bold ' + (p.win_rate >= 50 ? 'pnl-pos' : 'pnl-neg');

    document.getElementById('stat-trades').textContent =
      p.wins + 'W / ' + p.losses + 'L';

    const dd = data.drawdown?.drawdown_pct ?? 0;
    const ddEl = document.getElementById('stat-dd');
    ddEl.textContent = dd.toFixed(2) + '%';
    ddEl.className = 'text-2xl font-bold ' + (dd > 5 ? 'pnl-neg' : 'text-white');

    // Chart
    if (data.cum_pnl?.length) renderPnlChart(data.cum_pnl);
    else renderPnlChart([]);

    // Fetch controls state and update badge
    try {
      const ctrlRes = await fetch('/api/controls');
      const ctrl = await ctrlRes.json();
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
      const pnlPct = ((price - entry) / entry) * mult * 100;
      const deployed = parseFloat(pos.usdt_deployed || 0);
      const pnlUsd   = pnlPct / 100 * deployed;
      const tpPct    = parseFloat(pos.take_profit_pct || 3.0);
      const slPct    = parseFloat(pos.stop_loss_pct   || 1.8);
      const pnlColor = pnlPct >= 0 ? '#34d399' : '#f87171';
      const barPct   = Math.min(Math.abs(pnlPct) / tpPct * 100, 100).toFixed(1);

      // Update all live fields in the expanded card
      const cardEls = document.querySelectorAll(`[data-live-asset="${asset}"]`);
      cardEls.forEach(el => {
        if (el.dataset.field === 'price')   el.textContent = '$' + price.toFixed(6);
        if (el.dataset.field === 'pnlpct')  { el.textContent = (pnlPct>=0?'+':'') + pnlPct.toFixed(3) + '%'; el.style.color = pnlColor; }
        if (el.dataset.field === 'pnlusd')  { el.textContent = (pnlUsd>=0?'+':'') + '$' + pnlUsd.toFixed(4); el.style.color = pnlColor; }
        if (el.dataset.field === 'bar')     { el.style.width = barPct + '%'; el.style.background = pnlColor; }
        if (el.dataset.field === 'barlabel'){ el.textContent = (pnlPct>=0?'+':'') + pnlPct.toFixed(2) + '% of ' + tpPct + '% target'; }
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

async function doForceExit(asset) {
  if (!confirm(`Force-exit ${asset}?`)) return;
  const res = await ctrlPost({action: 'exit', asset});
  if (res) { showToast(`✕ Exit queued for ${asset} — will close on next tick`); }
}

async function doSetLeverage(asset, leverage) {
  if (!leverage) {
    await ctrlPost({action: 'clear_leverage', asset});
    showToast(`${asset} leverage reset to regime default`);
  } else {
    const res = await ctrlPost({action: 'set_leverage', asset, leverage: parseFloat(leverage)});
    if (res) showToast(`${asset} leverage override → ${leverage}x`);
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
  const sym = (asset || 'BTC/USDT').replace('/USDT', '');
  return `BINANCE:${sym}USDTPERP`;
}

function loadChart(asset) {
  const sym  = tvSymbol(asset);
  const ivEl = document.getElementById('chart-interval');
  const iv   = ivEl ? ivEl.value : '60';

  _currentTvSymbol = sym;

  // Update label and expand link
  document.getElementById('chart-symbol-label').textContent = (asset || 'BTC/USDT').replace('/USDT', '') + '/USDT';
  document.getElementById('tv-expand-btn').href =
    `https://www.tradingview.com/chart/?symbol=${sym}`;

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
