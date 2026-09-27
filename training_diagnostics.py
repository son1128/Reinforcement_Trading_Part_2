"""
training_diagnostics.py
-----------------------
Run after train_ppo.py to assess:

  1. Learning curve  — is the agent actually learning?
  2. Overfitting     — train / val / test performance gap
  3. Feature quality — stationarity, variance, predictive signal, collinearity

Usage:
    python training_diagnostics.py

Requires: scipy  (pip install scipy)
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from model_artifacts import (
    assert_holdout_is_unseen,
    load_run_info,
    resolve_project_path,
    resolve_sb3_model_path,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Learning curve
# ─────────────────────────────────────────────────────────────────────────────

def _load_eval_logs(log_dir: Path) -> dict | None:
    path = log_dir / "evaluations.npz"
    if not path.exists():
        return None
    d = np.load(str(path))
    return {"timesteps": d["timesteps"], "results": d["results"], "ep_lengths": d["ep_lengths"]}


def _learning_curve_df(logs: dict) -> pd.DataFrame:
    ts = logs["timesteps"]
    r = logs["results"]          # (n_evals, n_episodes_per_eval)
    ep = logs["ep_lengths"]
    return pd.DataFrame({
        "timesteps":       ts,
        "mean_ep_reward":  r.mean(axis=1),
        "std_ep_reward":   r.std(axis=1),
        "mean_ep_length":  ep.mean(axis=1) if ep.ndim == 2 else ep,
    })


def _print_learning_curve(curve: pd.DataFrame) -> None:
    n = len(curve)
    if n == 0:
        print("  (empty)")
        return

    print(curve.to_string(index=False))

    first_avg  = curve["mean_ep_reward"].iloc[:max(n // 2, 1)].mean()
    second_avg = curve["mean_ep_reward"].iloc[n // 2:].mean()
    delta      = second_avg - first_avg
    final      = float(curve["mean_ep_reward"].iloc[-1])
    final_std  = float(curve["std_ep_reward"].iloc[-1])
    tail_std   = float(curve["mean_ep_reward"].tail(min(5, n)).std())

    print(f"\n  Summary:")
    print(f"    Evaluations          : {n}")
    print(f"    First-half avg reward: {first_avg:+.4f}")
    print(f"    Second-half avg reward:{second_avg:+.4f}")
    print(f"    Δ (2nd - 1st half)   : {delta:+.4f}  "
          f"{'✓ learning' if delta > 0 else '✗ no improvement detected'}")
    print(f"    Final mean reward    : {final:+.4f} ± {final_std:.4f}")
    print(f"    Tail stability (std) : {tail_std:.4f}  "
          f"{'✓ stable' if tail_std < abs(final) * 0.3 + 1e-9 else '⚠ unstable'}")
    print(f"    Is learning          : {'YES' if delta > 0 else 'NO'}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Overfitting diagnostics
# ─────────────────────────────────────────────────────────────────────────────

_METRICS_OF_INTEREST = [
    "total_return_pct", "annualized_return_pct",
    "sharpe_like", "sortino_ratio", "calmar_ratio",
    "max_drawdown_pct", "profit_factor", "win_rate_pct",
    "avg_r", "n_trades",
]


def _run_split(model, decision_df, m1_df, feature_cols, vecnorm_path):
    """Evaluate a loaded PPO model on one split and return the report dict."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from train_ppo import _CaptureDoneWrapper, build_env
    from evaluate import full_report
    from config import CFG

    def _env_fn():
        return _CaptureDoneWrapper(
            build_env(
                decision_df,
                m1_df,
                feature_cols,
                randomize_start=False,
                episode_steps=None,
            )
        )

    raw = DummyVecEnv([_env_fn])
    if vecnorm_path and vecnorm_path.exists():
        venv = VecNormalize.load(str(vecnorm_path), raw)
        venv.training = False
        venv.norm_reward = False
    else:
        venv = raw

    obs = venv.reset()
    done = [False]
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = venv.step(action)

    capture = venv.venv.envs[0] if hasattr(venv, "venv") else venv.envs[0]
    eq = capture.saved_equity if capture.saved_equity is not None else pd.DataFrame()
    trades = capture.saved_trades if capture.saved_trades is not None else pd.DataFrame()
    rep = full_report(eq, trades, initial_equity=CFG.initial_equity,
                      periods_per_year=CFG.periods_per_year)
    return rep["value"].to_dict()


def _print_overfitting(results: dict[str, dict]) -> None:
    splits = list(results.keys())
    rows = []
    for m in _METRICS_OF_INTEREST:
        row = {"metric": m}
        for s in splits:
            row[s] = results[s].get(m, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("metric")
    print(df.to_string())

    print("\n  Overfitting flags:")
    train = results.get("train", {})
    val   = results.get("val",   {})
    test  = results.get("test",  {})
    flags = []

    def _gap(a, b):
        if a and b and np.isfinite(a) and np.isfinite(b) and abs(a) > 1e-9:
            return (b - a) / abs(a) * 100
        return np.nan

    tr_ret = train.get("total_return_pct", np.nan)
    va_ret = val.get("total_return_pct",   np.nan)
    te_ret = test.get("total_return_pct",  np.nan)
    ret_gap_vt = _gap(tr_ret, va_ret)
    ret_gap_te = _gap(tr_ret, te_ret)
    if np.isfinite(ret_gap_vt) and ret_gap_vt < -30:
        flags.append(f"  ⚠ Val total_return is {ret_gap_vt:.1f}% of train — possible overfit")
    if np.isfinite(ret_gap_te) and ret_gap_te < -30:
        flags.append(f"  ⚠ Test total_return is {ret_gap_te:.1f}% of train — possible overfit")

    tr_pf = train.get("profit_factor", np.nan)
    va_pf = val.get("profit_factor",   np.nan)
    if np.isfinite(tr_pf) and np.isfinite(va_pf) and tr_pf > 1.3 and va_pf < 1.0:
        flags.append(f"  ⚠ Train PF={tr_pf:.2f} but val PF={va_pf:.2f} < 1.0 — overfit")

    tr_sh = train.get("sharpe_like", np.nan)
    va_sh = val.get("sharpe_like",   np.nan)
    sh_gap = _gap(tr_sh, va_sh)
    if np.isfinite(sh_gap) and sh_gap < -40:
        flags.append(f"  ⚠ Val Sharpe is {sh_gap:.1f}% of train — possible overfit")

    if not flags:
        flags = ["  ✓ No severe overfitting flags triggered."]
    for f in flags:
        print(f)


# ─────────────────────────────────────────────────────────────────────────────
# 2b. Walk-forward validation summary
# ─────────────────────────────────────────────────────────────────────────────

def _print_walk_forward_summary(models_path: Path) -> None:
    """Echo the per-fold walk-forward results written by
    train_ppo.train_walk_forward(), plus an aggregate that is the honest
    out-of-sample generalisation estimate (vs. a single lucky/unlucky split)."""
    summary_csv = models_path / "walk_forward_summary.csv"
    if not summary_csv.exists():
        print("  No walk_forward_summary.csv found.")
        print("  Run walk-forward training first:  python train_ppo.py")
        print("  (or train_ppo.train_walk_forward()).  Single-split metrics are in [3].")
        return

    summary = pd.read_csv(summary_csv)
    print("  Per-fold out-of-sample validation results:")
    print(summary.to_string(index=False))

    metric_cols = [c for c in
                   ["val_return_pct", "val_sharpe", "val_profit_factor",
                    "val_win_rate_pct", "val_max_dd_pct", "val_avg_r"]
                   if c in summary.columns]
    if metric_cols:
        agg = pd.DataFrame({
            "mean": summary[metric_cols].mean(),
            "std":  summary[metric_cols].std(),
            "min":  summary[metric_cols].min(),
            "max":  summary[metric_cols].max(),
        })
        print("\n  Aggregate across folds (this is the generalisation estimate):")
        print(agg.to_string())

    n = len(summary)
    if "val_return_pct" in summary.columns:
        pos = int((summary["val_return_pct"] > 0).sum())
        print(f"\n  Folds with positive OOS return : {pos}/{n}")
    if "val_profit_factor" in summary.columns:
        pf_ok = int((summary["val_profit_factor"] > 1.0).sum())
        print(f"  Folds with OOS profit factor>1 : {pf_ok}/{n}")
    print("  A strategy that only works on some folds has no robust edge —")
    print("  look for consistency across folds, not one strong fold.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature quality
# ─────────────────────────────────────────────────────────────────────────────

def _feature_quality(feat: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    try:
        from scipy import stats as sp_stats
        has_scipy = True
    except ImportError:
        has_scipy = False

    # 1-bar ahead return (the most granular predictive target).
    fut_ret = feat["Close"].pct_change().shift(-1)
    rows = []

    for col in feature_cols:
        s = feat[col].dropna()
        if len(s) < 50:
            continue
        mean_v = float(s.mean())
        std_v  = float(s.std())
        skew_v = float(s.skew())
        kurt_v = float(s.kurtosis())

        # Coefficient of variation as a proxy for "does this feature move?"
        cv = std_v / (abs(mean_v) + 1e-12)

        # Predictive signal: Spearman ρ with 1-bar future return.
        corr_r = corr_p = np.nan
        if has_scipy:
            common = s.index.intersection(fut_ret.dropna().index)
            if len(common) >= 30:
                corr_r, corr_p = sp_stats.spearmanr(s.loc[common], fut_ret.loc[common])

        rows.append({
            "feature":           col,
            "mean":              round(mean_v, 5),
            "std":               round(std_v,  5),
            "skew":              round(skew_v, 3),
            "excess_kurt":       round(kurt_v, 3),
            "cv":                round(cv,     4),
            "spearman_r_1bar":   round(float(corr_r), 5) if np.isfinite(corr_r) else np.nan,
            "spearman_p_1bar":   round(float(corr_p), 5) if np.isfinite(corr_p) else np.nan,
            "signal_p05":        bool(np.isfinite(corr_p) and corr_p < 0.05),
        })

    return pd.DataFrame(rows).set_index("feature")


def _feature_collinearity_flags(feat: pd.DataFrame, feature_cols: list[str],
                                threshold: float = 0.85) -> list[str]:
    corr = feat[feature_cols].corr().abs()
    flags = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            c = corr.iloc[i, j]
            if c >= threshold:
                flags.append(
                    f"  |r|={c:.2f}: {feature_cols[i]}  ↔  {feature_cols[j]}"
                )
    return flags or ["  ✓ No collinear pairs found (|r| < 0.85)."]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostics(models_dir: str = "models", reveal_test: bool = False) -> None:
    from config import CFG
    from data_loader import load_mt_ohlcv_csv, resample_ohlcv, split_train_val_test
    from features import prepare_feature_frame

    models_path = Path(models_dir)
    eval_log_dir = models_path / "eval_logs"

    sep = "=" * 72
    print(sep)
    print("  RL TRADING — TRAINING DIAGNOSTICS")
    print(sep)

    # ── 1. Learning curve ────────────────────────────────────────────────────
    print("\n[1] LEARNING CURVE")
    print("-" * 50)
    logs = _load_eval_logs(eval_log_dir)
    if logs is None:
        print("  No eval logs found. Train the model first (train_ppo.py).")
    else:
        curve = _learning_curve_df(logs)
        _print_learning_curve(curve)

    # ── 2. Data + feature quality ────────────────────────────────────────────
    print("\n[2] FEATURE QUALITY")
    print("-" * 50)
    print("  Loading data …")
    m1 = load_mt_ohlcv_csv(
        CFG.csv_path, time_col=CFG.time_col, source_tz=CFG.source_tz,
        timestamp_is_bar_open=CFG.timestamp_is_bar_open,
        bar_duration=CFG.pandas_execution_tf,
        start_date=CFG.start_date, end_date=CFG.end_date,
        max_days_for_demo=CFG.max_days_for_demo,
    )
    from data_loader import resample_ohlcv
    decision = resample_ohlcv(m1, CFG.pandas_tf)
    feat, feature_cols = prepare_feature_frame(
        decision, warmup_bars=CFG.warmup_bars,
        atr_period=CFG.atr_period, rsi_period=CFG.rsi_period,
    )
    train_feat, val_feat, test_feat = split_train_val_test(
        feat, train_frac=CFG.train_frac, val_frac=CFG.val_frac,
        embargo_bars=CFG.split_embargo_bars,
    )
    print(f"  Total bars (post-warmup): {len(feat):,}")
    print(f"  Train: {len(train_feat):,}  Val: {len(val_feat):,}  Test: {len(test_feat):,}")
    print(f"  Feature count: {len(feature_cols)}")
    print()

    feature_diag_df = pd.concat([train_feat, val_feat]).sort_index()
    print("  Feature quality is computed on train+val only; test remains sealed.")
    fq = _feature_quality(feature_diag_df, feature_cols)
    print(fq.to_string())

    print("\n  Features with 1-bar predictive signal (Spearman p < 0.05):")
    sig = fq[fq["signal_p05"] == True]
    if len(sig):
        print(sig[["spearman_r_1bar", "spearman_p_1bar"]].to_string())
        print("  Note: even weak short-term signal can be exploitable by RL.")
    else:
        print("  None at p=0.05. Expected for efficient intraday markets.")

    print("\n  High-collinearity pairs (|Pearson r| ≥ 0.85):")
    for f in _feature_collinearity_flags(feature_diag_df, feature_cols):
        print(f)

    # ── 3. Overfitting ───────────────────────────────────────────────────────
    print("\n[3] OVERFITTING DIAGNOSTICS")
    print("-" * 50)

    model_path: Path | None = None
    run_info: dict | None = None
    vecnorm_path = models_path / "vec_normalize.pkl"
    try:
        _, run_info = load_run_info(models_path)
        # Prefer the best (eligible) checkpoint + its normalisation snapshot so
        # diagnostics evaluate the SAME model that would be deployed — not the
        # final (typically most-overfit) checkpoint.
        mp_key = "best_model_path" if "best_model_path" in run_info else "model_path"
        vp_key = "best_model_vecnorm_path" if "best_model_vecnorm_path" in run_info else "vecnorm_path"
        model_path = resolve_sb3_model_path(run_info[mp_key], base_dir=models_path.parent)
        vecnorm_path = resolve_project_path(run_info[vp_key], base_dir=models_path.parent)
        if mp_key == "best_model_path":
            print("  Evaluating best (eligible) walk-forward checkpoint.")
    except (FileNotFoundError, KeyError):
        # Fallback for older runs before run_info.json existed.
        best_model_path = models_path / "best_model" / "best_model.zip"
        legacy_model_path = models_path / "ppo_xauusd_seed_42.zip"
        if best_model_path.exists():
            model_path = best_model_path
        elif legacy_model_path.exists():
            model_path = legacy_model_path

    if model_path is None:
        print("  No trained model found. Run train_ppo.py first.")
        print(sep)
        return

    print(f"  Loading model: {model_path}")
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize

        if vecnorm_path.exists():
            print(f"  VecNormalize stats: {vecnorm_path}")
        else:
            print("  ⚠ vec_normalize.pkl not found — evaluation will use unnormalized obs.")

        model = PPO.load(str(model_path))

        def _m1_slice(df):
            return m1.loc[(m1.index > df.index.min()) & (m1.index <= df.index.max())]

        split_frames = [("train", train_feat), ("val", val_feat)]
        if reveal_test:
            assert_holdout_is_unseen(run_info or {}, test_feat.index.min())
            split_frames.append(("test", test_feat))
        else:
            print("  Test split remains sealed. Re-run with reveal_test=True for final holdout diagnostics.")

        results = {}
        for name, df in split_frames:
            print(f"  Evaluating on {name} split …")
            results[name] = _run_split(
                model, df, _m1_slice(df), feature_cols, vecnorm_path
            )

        print()
        _print_overfitting(results)

    except Exception as exc:
        print(f"  Error during model evaluation: {exc}")
        print("  (If the obs shape changed after the feature audit, retrain the "
              "model so it matches the current 25-feature observation.)")
        import traceback
        traceback.print_exc()

    # ── 4. Walk-forward validation ───────────────────────────────────────────
    print("\n[4] WALK-FORWARD VALIDATION")
    print("-" * 50)
    _print_walk_forward_summary(models_path)

    print("\n" + sep)
    print("  DIAGNOSTICS COMPLETE")
    print(sep)


if __name__ == "__main__":
    run_diagnostics()
