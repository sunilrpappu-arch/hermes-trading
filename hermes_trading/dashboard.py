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

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

STATE_DIR = Path(os.getenv("STATE_DIR", Path(__file__).parent.parent / "state"))
PORT      = int(os.getenv("PORT", "8080"))

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


def _read_heartbeats() -> dict[str, dict]:
    """Return {asset: heartbeat_dict} for all active pairs."""
    hbs = {}
    for hf in STATE_DIR.glob("heartbeat_*.json"):
        try:
            data = json.loads(hf.read_text())
            asset = data.get("asset") or hf.stem.replace("heartbeat_", "").replace("_", "/", 1)
            hbs[asset] = data
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
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "strategy":      strategy,
        "regime":        regime,
        "is_sideways":   is_sideways,
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
    <p class="text-slate-400 text-xs font-semibold mb-3">ACTIVE PAIRS</p>
    <div id="pairs-grid" class="space-y-3"></div>
  </div>

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

function renderPairs(heartbeats) {
  const el = document.getElementById('pairs-grid');
  const entries = Object.entries(heartbeats);
  if (!entries.length) {
    el.innerHTML = '<p class="text-slate-500 text-sm">No active pairs</p>';
    return;
  }
  el.innerHTML = entries.map(([asset, hb]) => {
    const pos     = hb.open_position;
    const posDir  = pos ? pos.direction : null;
    const posClass = posDir === 'long' ? 'pos-long' : posDir === 'short' ? 'pos-short' : 'pos-none';
    const posLabel = posDir ? posDir.toUpperCase() : 'FLAT';

    const rsi  = hb.rsi_15m ?? '—';
    const rng  = hb.rng_pos != null ? Math.round(hb.rng_pos * 100) + '%' : null;
    const trend = hb.trend ?? '—';
    const tsAge = hb.timestamp ? Math.round((Date.now() - new Date(hb.timestamp).getTime()) / 1000) : null;
    const ageStr = tsAge != null ? (tsAge < 120 ? tsAge + 's ago' : Math.round(tsAge/60) + 'm ago') : '?';
    const ageColor = tsAge == null ? '#64748b' : tsAge < 90 ? '#34d399' : tsAge < 300 ? '#fbbf24' : '#f87171';

    let pnlStr = '';
    let posDetail = '';
    if (pos && pos.entry_price && hb.price) {
      const mult = pos.direction === 'long' ? 1 : -1;
      const pnlPct = ((hb.price - pos.entry_price) / pos.entry_price) * mult * 100;
      const pnlUsd = pnlPct / 100 * (pos.usdt_deployed || 0);
      pnlStr = `<span class="${pnlPct >= 0 ? 'pnl-pos' : 'pnl-neg'} text-xs font-semibold">${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}% (${pnlUsd >= 0 ? '+' : ''}$${pnlUsd.toFixed(3)})</span>`;
      const cap = pos.usdt_deployed ? '$' + parseFloat(pos.usdt_deployed).toFixed(2) : '';
      const qty = pos.qty > 0 ? parseFloat(pos.qty).toPrecision(4) + ' ' + asset.replace('/USDT','') : '';
      posDetail = cap || qty ? `<div class="text-slate-500 text-xs mt-0.5">${[cap, qty].filter(Boolean).join(' · ')}</div>` : '';
    }

    return `
    <div class="flex items-center justify-between bg-slate-900 rounded-lg px-3 py-2">
      <div>
        <span class="font-semibold text-sm text-white">${asset.replace('/USDT','')}</span>
        <span class="text-slate-400 text-xs ml-1">USDT</span>
        <span class="${posClass} text-xs font-bold ml-2">${posLabel}</span>
        ${pnlStr}
        ${posDetail}
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
      <td class="font-medium text-white">${(t.asset||'').replace('/USDT','')}/USDT</td>
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

    const ver = data.strategy?.version;
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

    // Pairs
    renderPairs(data.heartbeats || {});

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

// Initial load + 30s polling
refresh();
setInterval(refresh, 30000);
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
