from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from evaluate import full_report


def random_policy(env):
    return env.action_space.sample()


def ema_atr_trend_policy(env, threshold_atr: float = 0.10, sl_idx: int = 1, tp_idx: int = 2):
    """Simple sanity baseline: long above EMA trend, short below EMA trend."""
    row = env.decision_df.iloc[env.i]
    if row["ema20"] > row["ema50"] and row["close_ema20_atr"] > threshold_atr:
        return np.array([1, sl_idx, tp_idx], dtype=int)  # long
    if row["ema20"] < row["ema50"] and row["close_ema20_atr"] < -threshold_atr:
        return np.array([2, sl_idx, tp_idx], dtype=int)  # short
    return np.array([0, sl_idx, tp_idx], dtype=int)


@dataclass(frozen=True)
class TrendHoldPolicyParams:
    threshold_atr: float = 0.7
    sl_idx: int = 2
    tp_idx: int = 3


def trend_hold_policy(env, threshold_atr: float = 0.7, sl_idx: int = 2, tp_idx: int = 3):
    """Position-aware trend policy that holds exposure until TP/SL or a flip."""
    row = env.decision_df.iloc[env.i]
    long_sig = row["ema20"] > row["ema50"] and row["close_ema20_atr"] > threshold_atr
    short_sig = row["ema20"] < row["ema50"] and row["close_ema20_atr"] < -threshold_atr

    if long_sig:
        return np.array([1, sl_idx, tp_idx], dtype=int)
    if short_sig:
        return np.array([2, sl_idx, tp_idx], dtype=int)
    if env.position.direction == 1:
        return np.array([1, sl_idx, tp_idx], dtype=int)
    if env.position.direction == -1:
        return np.array([2, sl_idx, tp_idx], dtype=int)
    return np.array([0, sl_idx, tp_idx], dtype=int)


def make_trend_hold_policy(params: TrendHoldPolicyParams):
    def _policy(env):
        return trend_hold_policy(
            env,
            threshold_atr=params.threshold_atr,
            sl_idx=params.sl_idx,
            tp_idx=params.tp_idx,
        )

    return _policy


def run_policy(env, policy_fn, max_steps: int | None = None):
    obs, info = env.reset()
    done = False
    steps = 0
    while not done:
        action = policy_fn(env)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return env.equity_curve(), env.trade_log()


def evaluate_policy(env, policy_fn, initial_equity: float = 10_000.0,
                    periods_per_year: int = 252 * 24 * 12):
    equity, trades = run_policy(env, policy_fn)
    report = full_report(equity, trades, initial_equity=initial_equity,
                         periods_per_year=periods_per_year)
    return equity, trades, report


def _selection_score(val_report: dict) -> float:
    total_return = float(val_report.get("total_return_pct", np.nan))
    max_drawdown = float(val_report.get("max_drawdown_pct", np.nan))
    profit_factor = float(val_report.get("profit_factor", np.nan))

    if not np.isfinite(total_return):
        total_return = -1e9
    if not np.isfinite(max_drawdown):
        max_drawdown = -100.0
    if not np.isfinite(profit_factor):
        profit_factor = 0.0

    return total_return + 100.0 * (profit_factor - 1.0) + 0.25 * max_drawdown


def optimize_trend_hold_policy(
    train_env_factory,
    threshold_grid,
    sl_idx_grid,
    tp_idx_grid,
    initial_equity: float = 10_000.0,
    val_env_factory=None,
    train_weight: float = 0.4,
    val_weight: float = 0.6,
    periods_per_year: int = 252 * 24 * 12,
    verbose: bool = True,
):
    import sys
    import time

    if val_env_factory is None:
        val_env_factory = train_env_factory

    log = print if verbose else (lambda *args, **kwargs: None)
    combos = list(product(threshold_grid, sl_idx_grid, tp_idx_grid))
    n_total = len(combos)
    rows = []
    best_params = TrendHoldPolicyParams()
    best_score = -np.inf
    t0 = time.perf_counter()

    for done, (threshold_atr, sl_idx, tp_idx) in enumerate(combos):
        params = TrendHoldPolicyParams(
            threshold_atr=float(threshold_atr),
            sl_idx=int(sl_idx),
            tp_idx=int(tp_idx),
        )

        # Progress line: show combo being evaluated + ETA
        pct = done / n_total * 100
        elapsed = time.perf_counter() - t0
        eta = (elapsed / done * (n_total - done)) if done > 0 else 0.0
        log(
            f"  [{done:>{len(str(n_total))}}/{n_total}] {pct:5.1f}%  "
            f"thr={threshold_atr:.2f} sl={sl_idx} tp={tp_idx}  "
            f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
            flush=True,
        )

        _, _, train_report = evaluate_policy(
            train_env_factory(),
            make_trend_hold_policy(params),
            initial_equity=initial_equity,
            periods_per_year=periods_per_year,
        )
        _, _, val_report = evaluate_policy(
            val_env_factory(),
            make_trend_hold_policy(params),
            initial_equity=initial_equity,
            periods_per_year=periods_per_year,
        )
        train_metrics = train_report["value"].to_dict()
        val_metrics = val_report["value"].to_dict()
        score = train_weight * _selection_score(train_metrics) + val_weight * _selection_score(val_metrics)
        row = {
            "threshold_atr": params.threshold_atr,
            "sl_idx": params.sl_idx,
            "tp_idx": params.tp_idx,
            "selection_score": score,
        }
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})

        # Show quick result for this combo so you can watch quality evolve
        tr_ret = train_metrics.get("total_return_pct", float("nan"))
        va_ret = val_metrics.get("total_return_pct", float("nan"))
        tr_pf  = train_metrics.get("profit_factor",   float("nan"))
        va_pf  = val_metrics.get("profit_factor",     float("nan"))
        marker = " ← best" if score > best_score else ""
        log(
            f"         score={score:+.2f}  "
            f"train_ret={tr_ret:+.1f}%  val_ret={va_ret:+.1f}%  "
            f"train_PF={tr_pf:.2f}  val_PF={va_pf:.2f}{marker}",
            flush=True,
        )

        rows.append(row)
        if score > best_score:
            best_score = score
            best_params = params

    total_time = time.perf_counter() - t0
    log(f"  [{n_total}/{n_total}] 100.0%  done in {total_time:.0f}s", flush=True)

    search_df = pd.DataFrame(rows).sort_values(
        ["selection_score", "val_total_return_pct", "val_profit_factor"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return best_params, search_df
