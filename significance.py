"""Bootstrap significance tests for out-of-sample trading results.

All tests are one-sided against "no edge": a p-value is the share of bootstrap
resamples whose statistic is <= 0.  numpy only; deterministic given ``seed``.

    trade_mean_r_test     — IID bootstrap of the mean R-multiple per trade
    block_sharpe_test     — circular moving-block bootstrap of the Sharpe ratio
                            (blocks keep volatility clustering / autocorrelation)
    paired_fold_test      — RL vs baseline on the SAME OOS windows: bootstrap of
                            the mean per-fold difference + exact sign test
    daily_returns         — MTM equity curve → daily returns (sparse bar-level
                            returns make bar Sharpe noisy and slow to resample)
"""
from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd


def _ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def trade_mean_r_test(r_mult, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> dict:
    r = np.asarray(pd.Series(r_mult, dtype=float).dropna())
    if len(r) < 2:
        return {"n": int(len(r)), "mean_r": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = r[rng.integers(0, len(r), len(r))].mean()
    lo, hi = _ci(means, alpha)
    return {"n": int(len(r)), "mean_r": float(r.mean()), "ci_lo": lo, "ci_hi": hi,
            "p_value": float((means <= 0).mean())}


def _sharpe(x: np.ndarray, periods_per_year: int) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan


def block_sharpe_test(returns, periods_per_year: int = 261, block_len: int = 5,
                      n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> dict:
    x = np.asarray(pd.Series(returns, dtype=float).dropna())
    n = len(x)
    if n < 2 * block_len:
        return {"n": int(n), "sharpe": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    offsets = np.arange(block_len)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        stats[b] = _sharpe(x[idx], periods_per_year)
    stats = stats[np.isfinite(stats)]
    lo, hi = _ci(stats, alpha) if len(stats) else (np.nan, np.nan)
    return {"n": int(n), "sharpe": _sharpe(x, periods_per_year), "ci_lo": lo, "ci_hi": hi,
            "p_value": float((stats <= 0).mean()) if len(stats) else np.nan}


def paired_fold_test(a, b, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Is ``a`` (e.g. RL per-fold test return) better than ``b`` (baseline)?"""
    d = (pd.Series(a, dtype=float).reset_index(drop=True)
         - pd.Series(b, dtype=float).reset_index(drop=True)).dropna().to_numpy()
    n = len(d)
    if n < 2:
        return {"n": int(n), "mean_diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "p_value": np.nan, "wins": np.nan, "sign_p_value": np.nan}
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = _ci(means, alpha)
    wins = int((d > 0).sum())
    ties = int((d == 0).sum())
    m = n - ties
    # Exact one-sided binomial sign test: P(X >= wins | p = 0.5), ties dropped.
    sign_p = float(sum(comb(m, k) for k in range(wins, m + 1)) / 2 ** m) if m else np.nan
    return {"n": int(n), "mean_diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
            "p_value": float((means <= 0).mean()), "wins": wins, "sign_p_value": sign_p}


def daily_returns(equity_df: pd.DataFrame) -> pd.Series:
    if equity_df is None or equity_df.empty or "equity" not in equity_df:
        return pd.Series(dtype=float)
    daily = equity_df["equity"].astype(float).resample("1D").last().dropna()
    return daily.pct_change().dropna()
