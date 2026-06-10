import math
from typing import List, Dict, Any


def score(trades: List[Dict[str, Any]], goal: Dict[str, Any]) -> float:
    """Score a list of trades against goal.yaml — returns float in [-1, +1]."""
    if not trades:
        return 0.0

    target_return = goal.get("target_return_30d", 0.05)
    max_drawdown = goal.get("max_drawdown", 0.05)
    min_sharpe = goal.get("min_sharpe", 2.0)
    failure_below = goal.get("failure_below", -0.04)

    returns = [t.get("pnl_pct", 0.0) for t in trades]
    realised_return = sum(returns)

    # Drawdown
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in returns:
        cumulative += r
        if cumulative > peak:
            peak = cumulative
        dd = (peak - cumulative) / (peak + 1e-9)
        if dd > max_dd:
            max_dd = dd

    # Sharpe (annualised, assume 1-min bars → ~525600 per year)
    if len(returns) > 1:
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(variance) if variance > 0 else 1e-9
        sharpe = (mean_r / std_r) * math.sqrt(525600)
    else:
        sharpe = 0.0

    # Component scores [-1, +1]
    return_score = max(-1.0, min(1.0, realised_return / target_return))
    dd_score = max(-1.0, min(1.0, 1.0 - (max_dd / max_drawdown)))
    sharpe_score = max(-1.0, min(1.0, sharpe / min_sharpe))

    composite = (return_score * 0.5) + (dd_score * 0.3) + (sharpe_score * 0.2)
    composite = max(-1.0, min(1.0, composite))

    if realised_return < failure_below:
        composite = min(composite, -0.8)

    return round(composite, 4)
