from __future__ import annotations

from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from baselines import TrendHoldPolicyParams, evaluate_policy, make_trend_hold_policy
from config import CFG
from evaluate import full_report
from model_artifacts import (
    assert_holdout_is_unseen,
    load_run_info,
    resolve_project_path,
    resolve_sb3_model_path,
)
from train_ppo import _CaptureDoneWrapper, _slice_m1_for_decision_window, build_env, load_datasets


def _scenario_rows(scenario: str, report_df: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {"scenario": scenario, "metric": metric, "value": value}
        for metric, value in report_df["value"].items()
    ]


def _load_selected_policy(path: str | Path = "outputs/selected_policy.csv") -> TrendHoldPolicyParams | None:
    policy_path = Path(path)
    if not policy_path.exists():
        return None
    sp = pd.read_csv(policy_path, index_col=0)
    return TrendHoldPolicyParams(
        threshold_atr=float(sp.loc["threshold_atr", "value"]),
        sl_idx=int(sp.loc["sl_idx", "value"]),
        tp_idx=int(sp.loc["tp_idx", "value"]),
    )


def _run_rl_holdout(
    decision_df: pd.DataFrame,
    m1_df: pd.DataFrame,
    feature_cols: list[str],
    model_path: Path,
    vecnorm_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _make() -> _CaptureDoneWrapper:
        return _CaptureDoneWrapper(
            build_env(
                decision_df,
                m1_df,
                feature_cols,
                randomize_start=False,
                episode_steps=None,
            )
        )

    raw = DummyVecEnv([_make])
    venv = VecNormalize.load(str(vecnorm_path), raw)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(str(model_path), env=venv)
    obs = venv.reset()
    done = [False]
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = venv.step(action)

    capture = venv.venv.envs[0]
    equity = capture.saved_equity if capture.saved_equity is not None else pd.DataFrame()
    trades = capture.saved_trades if capture.saved_trades is not None else pd.DataFrame()
    report = full_report(
        equity,
        trades,
        initial_equity=CFG.initial_equity,
        periods_per_year=CFG.periods_per_year,
    )
    return equity, trades, report


def main(out_dir: str = "outputs/final_holdout", allow_unverified: bool = False) -> None:
    print("Revealing the sealed test split and writing holdout-only artifacts.")

    out_path = Path(out_dir)

    _, run_info = load_run_info("models")
    model_path = resolve_sb3_model_path(run_info["model_path"], ".")
    vecnorm_path = resolve_project_path(run_info["vecnorm_path"], ".")
    best_model_path = Path("models/best_model/best_model.zip")
    if best_model_path.exists():
        print(f"Using best validation checkpoint: {best_model_path}")
        model_path = best_model_path
        # Pair the best checkpoint with the normalisation it was selected under,
        # when train_walk_forward saved a snapshot alongside it.
        if "best_model_vecnorm_path" in run_info:
            vecnorm_path = resolve_project_path(run_info["best_model_vecnorm_path"], ".")
        elif Path("models/best_model/best_model_vecnorm.pkl").exists():
            vecnorm_path = Path("models/best_model/best_model_vecnorm.pkl")
    else:
        print(f"Using final checkpoint from run_info.json: {model_path}")

    m1, feature_cols, _train_feat, _val_feat, test_feat = load_datasets()
    # Refuse before writing anything if the model already saw the test period.
    assert_holdout_is_unseen(run_info, test_feat.index.min(), allow_unverified=allow_unverified)
    out_path.mkdir(parents=True, exist_ok=True)
    test_m1 = _slice_m1_for_decision_window(m1, test_feat)

    rl_eq, rl_trades, rl_report = _run_rl_holdout(
        test_feat,
        test_m1,
        feature_cols,
        model_path=model_path,
        vecnorm_path=vecnorm_path,
    )
    rl_eq.to_csv(out_path / "rl_test_equity_curve.csv")
    rl_trades.to_csv(out_path / "rl_test_trade_log.csv", index=False)

    report_rows = _scenario_rows("rl_test", rl_report)

    selected_policy = _load_selected_policy()
    if selected_policy is not None:
        baseline_eq, baseline_trades, baseline_report = evaluate_policy(
            build_env(test_feat, test_m1, feature_cols, randomize_start=False, episode_steps=None).env,
            make_trend_hold_policy(selected_policy),
            initial_equity=CFG.initial_equity,
            periods_per_year=CFG.periods_per_year,
        )
        baseline_eq.to_csv(out_path / "baseline_test_equity_curve.csv")
        baseline_trades.to_csv(out_path / "baseline_test_trade_log.csv", index=False)
        report_rows.extend(_scenario_rows("baseline_test", baseline_report))

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(out_path / "holdout_report.csv", index=False)

    print(report_df.pivot(index="metric", columns="scenario", values="value"))
    print(f"Saved holdout artifacts to: {out_path.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/final_holdout")
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Evaluate a legacy model whose run_info.json lacks train/val window metadata.",
    )
    args = parser.parse_args()
    main(out_dir=args.out_dir, allow_unverified=args.allow_unverified)
