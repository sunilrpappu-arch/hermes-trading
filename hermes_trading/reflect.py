"""
Reflection engine — two modes:
  --fallback  deterministic rule-based reflection (no Hermes needed)
  --hermes    calls hermes subprocess for AI-driven reflection
"""
import argparse
import json
import shutil
import subprocess
import time
import yaml
from pathlib import Path
import os

STATE_DIR = Path(os.getenv("STATE_DIR", Path(__file__).parent.parent / "state"))
STRATEGY_FILE = STATE_DIR / "strategy.yaml"
TRADES_FILE = STATE_DIR / "trades.jsonl"
HYPOTHESES_FILE = STATE_DIR / "hypotheses.jsonl"
HISTORY_DIR = STATE_DIR / "history"
GOAL_FILE = STATE_DIR / "goal.yaml"

HERMES_WINDOW = 25


def load_strategy() -> dict:
    with open(STRATEGY_FILE) as f:
        return yaml.safe_load(f)


def load_goal() -> dict:
    with open(GOAL_FILE) as f:
        return yaml.safe_load(f)


def load_recent_trades(n: int = HERMES_WINDOW) -> list:
    if not TRADES_FILE.exists():
        return []
    lines = TRADES_FILE.read_text().strip().splitlines()
    return [json.loads(l) for l in lines[-n:] if l.strip()]


def bump_version(current: str) -> str:
    try:
        n = int(current.lstrip("0") or "0")
    except ValueError:
        n = 0
    return str(n + 1).zfill(2)


def save_strategy(strategy: dict, variable_changed: str, reason: str, mode: str):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    old_version = strategy.get("version", "01")
    new_version = bump_version(old_version)

    # Archive current
    archive_path = HISTORY_DIR / f"v{old_version}.yaml"
    shutil.copy(STRATEGY_FILE, archive_path)

    # Write new
    strategy["version"] = new_version
    with open(STRATEGY_FILE, "w") as f:
        yaml.dump(strategy, f, default_flow_style=False)

    # Append hypothesis
    hypothesis = {
        "timestamp": int(time.time()),
        "mode": mode,
        "from_version": old_version,
        "to_version": new_version,
        "variable_changed": variable_changed,
        "reason": reason,
        "strategy_snapshot": strategy,
    }
    with open(HYPOTHESES_FILE, "a") as f:
        f.write(json.dumps(hypothesis) + "\n")

    print(f"Strategy v{old_version} → v{new_version} | changed: {variable_changed}")
    print(f"Reason: {reason}")


def reflect_fallback():
    strategy = load_strategy()
    goal = load_goal()
    trades = load_recent_trades()

    if not trades:
        print("No trades yet — nothing to reflect on.")
        return

    returns = [t.get("pnl_pct", 0.0) for t in trades]
    realised_return = sum(returns)
    target_return = goal.get("target_return_30d", 0.05)
    max_drawdown = goal.get("max_drawdown", 0.05)

    # Drawdown
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in returns:
        cumulative += r
        if cumulative > peak:
            peak = cumulative
        if peak > 0:
            dd = (peak - cumulative) / peak
        else:
            dd = abs(cumulative) if cumulative < 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Exactly ONE variable changed — priority: drawdown > return
    if max_dd > max_drawdown:
        old_sl = strategy.get("stop_loss_pct", 2.0)
        new_sl = round(old_sl - 0.2, 2)
        strategy["stop_loss_pct"] = new_sl
        save_strategy(
            strategy,
            variable_changed="stop_loss_pct",
            reason=f"Drawdown {max_dd:.2%} exceeded max {max_drawdown:.2%}. Tightened stop_loss_pct {old_sl} → {new_sl}.",
            mode="fallback",
        )
    elif realised_return < target_return:
        old_threshold = strategy["entry"]["threshold"]
        new_threshold = old_threshold + 2
        strategy["entry"]["threshold"] = new_threshold
        save_strategy(
            strategy,
            variable_changed="entry.threshold",
            reason=f"Realised return {realised_return:.2%} below target {target_return:.2%}. Loosened entry threshold {old_threshold} → {new_threshold}.",
            mode="fallback",
        )
    else:
        print("Strategy on target — no change needed.")


def reflect_hermes():
    strategy = load_strategy()
    trades = load_recent_trades()
    goal = load_goal()

    prompt = f"""You are the reflection engine for a trading agent. Analyse these {len(trades)} recent trades and propose exactly ONE change to improve the strategy.

Current strategy:
{yaml.dump(strategy)}

Goal:
{yaml.dump(goal)}

Recent trades (last {len(trades)}):
{json.dumps(trades, indent=2)}

Rules:
- Change EXACTLY one variable in the strategy YAML
- Name the variable, the old value, the new value, and the predicted score direction
- Output as JSON: {{"variable": "...", "old_value": ..., "new_value": ..., "reason": "...", "predicted_improvement": "..."}}
"""

    result = subprocess.run(
        ["hermes"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        print(f"Hermes error: {result.stderr}")
        return

    output = result.stdout.strip()
    # Parse JSON from output
    try:
        start = output.index("{")
        end = output.rindex("}") + 1
        hypothesis = json.loads(output[start:end])
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Could not parse Hermes output: {e}\nOutput: {output}")
        return

    variable = hypothesis["variable"]
    new_value = hypothesis["new_value"]
    reason = hypothesis.get("reason", "") + " | " + hypothesis.get("predicted_improvement", "")

    # Apply the change
    keys = variable.split(".")
    target = strategy
    for k in keys[:-1]:
        target = target[k]
    target[keys[-1]] = new_value

    save_strategy(strategy, variable_changed=variable, reason=reason, mode="hermes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument("--hermes", action="store_true")
    args = parser.parse_args()

    if args.fallback:
        reflect_fallback()
    elif args.hermes:
        reflect_hermes()
    else:
        print("Pass --fallback or --hermes")


if __name__ == "__main__":
    main()
