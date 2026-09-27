"""
calibrate_dd_penalty.py — choose CFG.dd_penalty from data, without retraining.

dd_penalty only changes WHICH checkpoint the consistency callback keeps; it does
not change training.  So a sliding walk-forward run made with
``CFG.log_checkpoint_test = True`` (every evaluation also logs the fold's TEST
window, record-only) can be re-scored offline for any dd_penalty:

  1. For each candidate value, replay the callback's selection over every
     fold's consistency_evals.csv — same score min(q_train, q_val), same
     eligibility, and the same early-stopping rule as the live config — and
     read the chosen checkpoint's logged test metrics.
  2. Pick the value on the EARLY folds (calibration set) only.
  3. Report how that value does on the LATER folds (holdout set), next to the
     default 1 / (100 · risk_fraction) and 0 — so the choice is checked on
     windows it was not picked on.

Usage
-----
  python calibrate_dd_penalty.py                          # models/**/sliding/fold_*
  python calibrate_dd_penalty.py --models-dir models/multi_seed
  python calibrate_dd_penalty.py --grid 0,1,2,3,4 --metric test_sharpe --calib-frac 0.6
  python calibrate_dd_penalty.py --no-early-stop          # replay without early stopping

Writes <models-dir>/dd_penalty_calibration.csv (per value) and
<models-dir>/dd_penalty_selections.csv (per fold × value).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG
from significance import paired_fold_test

_TEST_COLS = ["test_return_pct", "test_sharpe", "test_profit_factor",
              "test_max_dd_pct", "test_n_trades"]


def find_logs(models_dir: str | Path) -> pd.DataFrame:
    """All sliding fold logs under models_dir as (run, fold, path) rows."""
    rows = []
    for path in sorted(Path(models_dir).glob("**/sliding/fold_*/eval_logs/consistency_evals.csv")):
        fold_dir = path.parent.parent
        m = re.fullmatch(r"fold_(\d+)", fold_dir.name)
        if not m:
            continue
        run = str(fold_dir.parent.parent)          # the dir that holds sliding/
        rows.append({"run": run, "fold": int(m.group(1)), "path": path})
    return pd.DataFrame(rows, columns=["run", "fold", "path"])


def select_checkpoint(log: pd.DataFrame, dd_penalty: float,
                      patience: int | None, min_evals: int) -> tuple[pd.Series, bool, int]:
    """Replay _ConsistencyEvalCallback for one dd_penalty.

    Returns (chosen row, fallback, n_evals_seen).  fallback=True means no
    checkpoint was eligible, so the pipeline would use the final model — taken
    here as the last evaluation reached (approximate: training can run a little
    past the last evaluation)."""
    best_score = -np.inf
    chosen = None
    since_best = 0
    seen = 0
    for _, row in log.iterrows():
        seen += 1
        since_best += 1
        score = min(row["train_eval_r"] - dd_penalty * row["train_dd_pct"],
                    row["val_r"] - dd_penalty * row["val_dd_pct"])
        if bool(row["eligible"]) and score > best_score:
            best_score, chosen, since_best = score, row, 0
        if patience is not None and seen >= min_evals and since_best >= patience:
            break
    if chosen is None:
        return log.iloc[seen - 1], True, seen
    return chosen, False, seen


def calibrate(models_dir: str | Path = "models", grid=(0, 0.5, 1, 1.5, 2, 3, 4, 6),
              metric: str = "test_return_pct", calib_frac: float = 0.5,
              patience: int | None = None, min_evals: int | None = None,
              early_stop: bool = True) -> pd.DataFrame:
    if patience is None:
        patience = CFG.early_stop_patience
    if min_evals is None:
        min_evals = CFG.early_stop_min_evals
    if not early_stop:
        patience = None
    default = CFG.resolved_dd_penalty
    grid = sorted(set(float(g) for g in grid) | {default, 0.0})

    logs = find_logs(models_dir)
    if logs.empty:
        raise FileNotFoundError(
            f"No sliding fold logs under {models_dir}/**/sliding/fold_*/eval_logs/.\n"
            "Run the sliding walk-forward with CFG.log_checkpoint_test = True first."
        )

    selections = []
    skipped = []
    for _, lg in logs.iterrows():
        log = pd.read_csv(lg["path"])
        if metric not in log.columns:
            skipped.append(str(lg["path"]))
            continue
        for lam in grid:
            row, fallback, seen = select_checkpoint(log, lam, patience, min_evals)
            selections.append({
                "run": lg["run"], "fold": lg["fold"], "dd_penalty": lam,
                "timesteps": int(row["timesteps"]), "fallback_final": fallback,
                "evals_seen": seen, "evals_logged": len(log),
                **{c: row.get(c, np.nan) for c in _TEST_COLS},
            })
    if skipped:
        print(f"Skipped {len(skipped)} log(s) without test columns "
              f"(trained with log_checkpoint_test=False), e.g. {skipped[0]}")
    sel = pd.DataFrame(selections)
    if sel.empty:
        raise ValueError(f"No log has the '{metric}' column — enable CFG.log_checkpoint_test and retrain.")

    folds = sorted(sel["fold"].unique())
    n_calib = max(1, min(len(folds) - 1, int(round(len(folds) * calib_frac))))
    calib_folds = set(folds[:n_calib])
    sel["set"] = np.where(sel["fold"].isin(calib_folds), "calib", "holdout")

    def _agg(g):
        return pd.Series({
            "mean": g[metric].mean(),
            "median": g[metric].median(),
            "positive": int((g[metric] > 0).sum()),
            "n": int(g[metric].notna().sum()),
            "fallbacks": int(g["fallback_final"].sum()),
        })

    table = (sel.groupby(["dd_penalty", "set"]).apply(_agg).unstack("set"))
    table.columns = [f"{s}_{m}" for m, s in table.columns]
    table = table.reset_index()
    for c in table.columns:
        if c.endswith(("_positive", "_n", "_fallbacks")):
            table[c] = table[c].astype(int)

    # Choose on calibration folds only.  Ties with the best mean keep the
    # default (no evidence to move); otherwise the smaller penalty wins.
    best_mean = table["calib_mean"].max()
    tied = table[np.isclose(table["calib_mean"], best_mean, rtol=0, atol=1e-9)]
    if np.isclose(tied["dd_penalty"], default).any():
        chosen = default
    else:
        chosen = float(tied["dd_penalty"].min())

    out = Path(models_dir)
    sel.to_csv(out / "dd_penalty_selections.csv", index=False)
    table.to_csv(out / "dd_penalty_calibration.csv", index=False)

    runs = sel["run"].nunique()
    print("=" * 76)
    print(f"  dd_penalty calibration — metric={metric}, {runs} run(s), {len(folds)} folds "
          f"(calib = folds {min(calib_folds)}–{max(calib_folds)}, holdout = the rest)")
    print(f"  early stopping replay: "
          f"{'patience=%s, min_evals=%s' % (patience, min_evals) if patience is not None else 'off'}")
    print("=" * 76)
    cols = ["dd_penalty", "calib_mean", "calib_median", "calib_positive", "calib_n",
            "holdout_mean", "holdout_median", "holdout_positive", "holdout_n",
            "calib_fallbacks", "holdout_fallbacks"]
    print(table[[c for c in cols if c in table]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    hold = sel[sel["set"] == "holdout"]

    def _hold_series(lam):
        return (hold[hold["dd_penalty"] == lam]
                .sort_values(["run", "fold"])[metric].reset_index(drop=True))

    print(f"\n  Chosen on calibration folds : dd_penalty = {chosen:g}")
    print(f"  Default (1/(100·risk))      : dd_penalty = {default:g}")
    for label, lam in (("chosen", chosen), ("default", default), ("zero", 0.0)):
        h = _hold_series(lam)
        print(f"    holdout {label:<8} ({lam:g}): mean {metric} = {h.mean():+.3f}  "
              f"positive {int((h > 0).sum())}/{int(h.notna().sum())}")
    if chosen != default:
        t = paired_fold_test(_hold_series(chosen), _hold_series(default),
                             n_boot=CFG.bootstrap_samples, seed=CFG.bootstrap_seed)
        print(f"    chosen − default on holdout: mean {t['mean_diff']:+.3f} "
              f"[{t['ci_lo']:+.3f}, {t['ci_hi']:+.3f}]  wins {t['wins']}/{t['n']}  "
              f"sign p={t['sign_p_value']:.3f}")
        confirmed = t["mean_diff"] > 0
        print("\n  Recommendation: " + (
            f"set CFG.dd_penalty = {chosen:g} (it also beats the default on the holdout folds)."
            if confirmed else
            f"keep the default ({default:g}): the calibration pick ({chosen:g}) did not beat "
            "it on the holdout folds."))
    else:
        print("\n  Recommendation: keep the default — it is also the calibration pick.")
    print(f"\n  → {out / 'dd_penalty_calibration.csv'}\n  → {out / 'dd_penalty_selections.csv'}")
    print("=" * 76)
    return table


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--grid", default="0,0.5,1,1.5,2,3,4,6",
                   help="comma-separated dd_penalty candidates (default and 0 are always added)")
    p.add_argument("--metric", default="test_return_pct",
                   choices=["test_return_pct", "test_sharpe", "test_profit_factor"])
    p.add_argument("--calib-frac", type=float, default=0.5,
                   help="share of the earliest folds used to choose the value")
    p.add_argument("--no-early-stop", action="store_true",
                   help="replay selection without the early-stopping rule")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    calibrate(a.models_dir, grid=[float(x) for x in a.grid.split(",") if x.strip()],
              metric=a.metric, calib_frac=a.calib_frac, early_stop=not a.no_early_stop)
