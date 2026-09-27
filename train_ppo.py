from __future__ import annotations

import json
import shutil
import sys
from copy import deepcopy
from functools import partial
from pathlib import Path

import gymnasium as gym
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import CFG
from data_loader import (
    load_mt_ohlcv_csv,
    make_sliding_folds,
    make_walk_forward_folds,
    resample_ohlcv,
    split_train_val_test,
)
from evaluate import drawdown
from features import prepare_feature_frame
from env_bracket import BracketTradingEnv
from model_artifacts import load_run_info


# ── Pickle-safe factory for SubprocVecEnv ────────────────────────────────────
# Must be at module level (not a lambda) so subprocesses can import it.
def _spawn_train_env(decision_df, m1_df, feature_cols, episode_steps):
    """Called inside each worker process to build one training environment."""
    return build_env(decision_df, m1_df, feature_cols,
                     randomize_start=True, episode_steps=episode_steps)


def _slice_m1_for_decision_window(m1_df, decision_df):
    if decision_df.empty:
        return m1_df.iloc[0:0].copy()
    start = decision_df.index.min()
    end = decision_df.index.max()
    return m1_df.loc[(m1_df.index > start) & (m1_df.index <= end)].copy()


def _load_decision_features():
    """Load M1, resample to the decision timeframe, build the causal feature
    frame.  Shared by load_datasets() (single split) and load_folds() (walk-
    forward) so both modes see byte-identical bars and features."""
    m1 = load_mt_ohlcv_csv(
        CFG.csv_path,
        time_col=CFG.time_col,
        source_tz=CFG.source_tz,
        timestamp_is_bar_open=CFG.timestamp_is_bar_open,
        bar_duration=CFG.pandas_execution_tf,
        start_date=CFG.start_date,
        end_date=CFG.end_date,
        max_days_for_demo=CFG.max_days_for_demo,
    )
    decision = resample_ohlcv(m1, CFG.pandas_tf)
    feat, feature_cols = prepare_feature_frame(
        decision,
        warmup_bars=CFG.warmup_bars,
        atr_period=CFG.atr_period,
        rsi_period=CFG.rsi_period,
    )
    return m1, feat, feature_cols


def load_datasets():
    """Single chronological train/val/test split (legacy mode)."""
    m1, feat, feature_cols = _load_decision_features()
    train_feat, val_feat, test_feat = split_train_val_test(
        feat,
        train_frac=CFG.train_frac,
        val_frac=CFG.val_frac,
        embargo_bars=CFG.split_embargo_bars,
    )
    return m1, feature_cols, train_feat, val_feat, test_feat


def load_folds():
    """Walk-forward variant of load_datasets().

    Returns (m1, feature_cols, folds, test_feat) where ``folds`` is a list of
    ``(train_df, val_df)`` tuples whose validation windows roll forward in time,
    and ``test_feat`` is the sealed final-test segment (identical to the
    single-split test because CFG.test_frac == 1 - train_frac - val_frac)."""
    m1, feat, feature_cols = _load_decision_features()
    folds, test_feat = make_walk_forward_folds(
        feat,
        n_folds=CFG.n_walk_forward_folds,
        test_frac=CFG.test_frac,
        embargo_bars=CFG.split_embargo_bars,
        anchored=CFG.walk_forward_anchored,
    )
    return m1, feature_cols, folds, test_feat


def build_env(decision_df, m1_df, feature_cols, randomize_start: bool = False,
              episode_steps: int | None = None):
    env = BracketTradingEnv(
        decision_df,
        m1_df,
        feature_cols,
        sl_atr_multipliers=CFG.sl_atr_multipliers,
        tp_r_multipliers=CFG.tp_r_multipliers,
        initial_equity=CFG.initial_equity,
        risk_fraction=CFG.risk_fraction,
        spread_price=CFG.spread_price,
        slippage_price=CFG.slippage_price,
        commission_per_trade=CFG.commission_per_trade,
        holding_penalty=CFG.holding_penalty,
        reward_mtm_weight=CFG.reward_mtm_weight,
        randomize_start=randomize_start,
        max_episode_steps=episode_steps,
    )
    return Monitor(env)


class _SyncVecNormCallback(BaseCallback):
    """Copies running obs/ret stats from the train VecNormalize to every eval
    VecNormalize before each checkpoint evaluation."""

    def __init__(self, train_venv: VecNormalize,
                 eval_venvs: list[VecNormalize], eval_freq: int):
        super().__init__()
        self.train_venv  = train_venv
        self.eval_venvs  = eval_venvs
        self.eval_freq   = eval_freq
        self._last_sync  = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_sync >= self.eval_freq:
            for venv in self.eval_venvs:
                venv.obs_rms = deepcopy(self.train_venv.obs_rms)
                venv.ret_rms = deepcopy(self.train_venv.ret_rms)
            self._last_sync = self.num_timesteps
        return True


class _ConsistencyEvalCallback(BaseCallback):
    """Saves the checkpoint that maximises a drawdown-penalised consistency score.

    Standard EvalCallback picks the highest val reward in isolation.  That can
    select a model that got lucky on the val period while being terrible on
    training data — a sign it never really learned.

    This callback evaluates on BOTH a 'train eval' slice (the last ~len(val)
    bars of training data) and the val set.  For each leg it computes a
    risk-adjusted quality = cumulative_reward − dd_penalty · max_drawdown_pct,
    then scores the checkpoint by the *weaker* of the two legs:

        q_leg  = reward_leg − dd_penalty · max_drawdown_pct_leg
        score  = min(q_train, q_val)

    → rewards models that are simultaneously profitable AND low-drawdown on
      both sides of the split boundary
    → anchored by the weaker leg (penalises large train/val gap)

    Do-nothing guard: an idle policy makes no trades → flat equity → ~0 DD,
    which would otherwise look 'perfectly stable' under the penalty.  A
    checkpoint is only eligible to become 'best' when BOTH legs are profitable
    (reward > 0) and place at least `min_trades` trades.

    dd_penalty is in reward-units per 1% of max drawdown.  Calibrate it against
    the reward/dd magnitudes logged in eval_logs/consistency_evals.csv: too
    large and a flat, barely-trading model wins; too small and it has no effect.
    """

    def __init__(
        self,
        train_eval_env: VecNormalize,
        val_env: VecNormalize,
        eval_freq: int,
        best_model_save_path: str,
        log_path: str | None = None,
        dd_penalty: float = 1.0,
        min_trades: int = 5,
        verbose: int = 1,
        train_venv: VecNormalize | None = None,
    ):
        super().__init__(verbose=verbose)
        self.train_eval_env     = train_eval_env
        self.val_env            = val_env
        # The live training VecNormalize — snapshotted next to best_model so the
        # saved checkpoint is later evaluated under the normalisation it was
        # selected with (the obs_rms drifts as training continues).
        self.train_venv         = train_venv
        self.eval_freq          = eval_freq
        self.best_model_save_path = Path(best_model_save_path)
        self.log_path           = Path(log_path) if log_path else None
        self.dd_penalty         = float(dd_penalty)
        self.min_trades         = int(min_trades)
        self.best_score         = -float("inf")
        self._last_eval         = 0
        self._rows: list[dict]  = []

    def _run_one_episode(self, venv: VecNormalize) -> tuple[float, float, int]:
        """Run one deterministic episode.

        Returns (cumulative_reward, max_drawdown_pct, n_trades).  The equity
        curve is recovered from the _CaptureDoneWrapper that wraps the eval env,
        because SB3's evaluate_policy only returns the scalar episode reward.
        """
        from stable_baselines3.common.evaluation import evaluate_policy
        rewards, _ = evaluate_policy(
            self.model, venv,
            n_eval_episodes=1,
            deterministic=True,
            return_episode_rewards=True,
        )
        reward = float(rewards[0])

        # venv = VecNormalize → .venv = DummyVecEnv → .envs[0] = _CaptureDoneWrapper.
        # saved_equity/saved_trades were stashed before DummyVecEnv's auto-reset.
        capture = venv.venv.envs[0]
        eq = capture.saved_equity
        if eq is not None and not eq.empty and "equity" in eq:
            max_dd_pct = float(abs(drawdown(eq["equity"].astype(float)).min()) * 100.0)
            n_trades = len(capture.saved_trades) if capture.saved_trades is not None else 0
        else:
            max_dd_pct = 0.0
            n_trades = 0
        return reward, max_dd_pct, n_trades

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps

        train_r, train_dd, train_n = self._run_one_episode(self.train_eval_env)
        val_r,   val_dd,   val_n   = self._run_one_episode(self.val_env)

        # Risk-adjusted quality per leg, then anchor on the weaker leg.
        q_train = train_r - self.dd_penalty * train_dd
        q_val   = val_r   - self.dd_penalty * val_dd
        score   = min(q_train, q_val)
        gap     = abs(train_r - val_r)

        # Do-nothing guard: both legs must be genuinely profitable and active,
        # otherwise a flat ~0-drawdown idle policy would win on the penalty term.
        eligible = (
            train_r > 0 and val_r > 0
            and train_n >= self.min_trades and val_n >= self.min_trades
        )

        row = dict(timesteps=self.num_timesteps,
                   train_eval_r=round(train_r, 3),
                   val_r=round(val_r, 3),
                   train_dd_pct=round(train_dd, 3),
                   val_dd_pct=round(val_dd, 3),
                   q_train=round(q_train, 3),
                   q_val=round(q_val, 3),
                   score=round(score, 3),
                   gap=round(gap, 3),
                   eligible=eligible)
        self._rows.append(row)

        marker = ""
        if eligible and score > self.best_score:
            self.best_score = score
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.best_model_save_path / "best_model"))
            # Snapshot the obs/reward normalisation stats AS OF this checkpoint
            # (project invariant: VecNormalize travels with every saved model).
            if self.train_venv is not None:
                self.train_venv.save(str(self.best_model_save_path / "best_model_vecnorm.pkl"))
            marker = "  ← BEST"

        if self.verbose >= 1:
            flag = "" if eligible else "  (ineligible)"
            print(
                f"[{self.num_timesteps:>9,}]  "
                f"train={train_r:+7.2f}/dd{train_dd:4.1f}%  "
                f"val={val_r:+7.2f}/dd{val_dd:4.1f}%  "
                f"score={score:+8.2f}{flag}{marker}",
                flush=True,
            )

        # Persist log so training_diagnostics.py can plot it
        if self.log_path:
            import pandas as _pd
            self.log_path.mkdir(parents=True, exist_ok=True)
            _pd.DataFrame(self._rows).to_csv(
                self.log_path / "consistency_evals.csv", index=False
            )

        return True


def _linear_schedule(initial_value: float):
    """SB3 learning-rate schedule: decays linearly from initial_value to 0 as
    training progresses (SB3 passes progress_remaining = 1.0 → 0.0).  Shrinking
    late-training updates keeps an over-fitting tail from overrunning the best
    checkpoint."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


def train(
    total_timesteps: int = 2_000_000,
    # ── timesteps ────────────────────────────────────────────────────────────
    # Normally set by the caller. train_walk_forward() scales this PER FOLD by
    # the fold's train length so every fold makes a similar number of passes
    # over its data — a fixed budget over-trains the small early folds and
    # under-trains the large late ones. 2048-step episodes → total/2048 rollouts
    # × n_epochs minibatch passes each.
    # ────────────────────────────────────────────────────────────────────────
    seed: int = 42,
    out_dir: str = "models",
    train_episode_steps: int = 2048,
    eval_freq: int = 50_000,
    # dd_penalty: reward-units subtracted per 1% of max drawdown in checkpoint
    # selection.  Higher → prefer stabler (lower-DD) models; too high selects a
    # near-idle policy.  Calibrate against eval_logs/consistency_evals.csv.
    dd_penalty: float = 1.0,
    # ── Parallelism & device ─────────────────────────────────────────────────
    # n_envs > 1 uses SubprocVecEnv: near-linear speedup because env simulation
    # (not the network) is the bottleneck.  n_envs=4 on a quad-core CPU gives
    # ~3.5× faster wall-clock.  Requires the __main__ guard at the bottom.
    # device="auto" picks CUDA if available; adds ~15–30% on the network side.
    # With n_envs=4 + GPU, effective batch per update = n_steps*n_envs = 8192
    # → increase batch_size to 512 or 1024 for better GPU utilisation.
    n_envs: int = 1,
    device: str = "auto",
    # ────────────────────────────────────────────────────────────────────────
    reveal_test: bool = False,
    # Inject pre-built splits (used by train_walk_forward to train one fold).
    # When None, load the single chronological split via load_datasets().
    datasets: tuple | None = None,
):
    import torch
    from stable_baselines3 import PPO

    # ── Device selection ─────────────────────────────────────────────────────
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {props.name}  ({props.total_memory / 1e9:.1f} GB VRAM)")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if datasets is None:
        m1, feature_cols, train_feat, val_feat, test_feat = load_datasets()
    else:
        m1, feature_cols, train_feat, val_feat, test_feat = datasets
    train_m1 = _slice_m1_for_decision_window(m1, train_feat)
    val_m1   = _slice_m1_for_decision_window(m1, val_feat)

    # ── Training environments ─────────────────────────────────────────────────
    if n_envs > 1:
        # SubprocVecEnv spawns n_envs worker processes that step in parallel.
        # partial() wraps a module-level function so it is pickle-safe on Windows.
        from stable_baselines3.common.vec_env import SubprocVecEnv
        env_fns = [
            partial(_spawn_train_env, train_feat, train_m1, feature_cols, train_episode_steps)
            for _ in range(n_envs)
        ]
        start_method = "spawn" if sys.platform == "win32" else "forkserver"
        print(f"Training envs : {n_envs} parallel (SubprocVecEnv, {start_method})")
        train_env_raw = SubprocVecEnv(env_fns, start_method=start_method)
    else:
        print("Training envs : 1 (DummyVecEnv — set n_envs=4 for ~4x speedup)")
        train_env_raw = DummyVecEnv([
            lambda: build_env(train_feat, train_m1, feature_cols,
                              randomize_start=True, episode_steps=train_episode_steps)
        ])

    train_env = VecNormalize(train_env_raw, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ── Eval environments (deterministic, no reward norm) ─────────────────────
    # _CaptureDoneWrapper stashes the equity curve before DummyVecEnv's auto-reset
    # wipes it, so the consistency callback can read each episode's max drawdown.
    val_env_raw = DummyVecEnv([
        lambda: _CaptureDoneWrapper(
            build_env(val_feat, val_m1, feature_cols,
                      randomize_start=False, episode_steps=None))
    ])
    val_env = VecNormalize(val_env_raw, norm_obs=True, norm_reward=False,
                           clip_obs=10.0, training=False)

    # Train-eval slice: the last ~len(val) bars of training data.
    # Using the tail of training (regime just before the embargo) makes the
    # train/val comparison as fair as possible — same era, different side of
    # the split boundary.  Its size matches val so rewards are on the same scale.
    train_eval_feat = train_feat.iloc[-len(val_feat):]
    train_eval_m1   = _slice_m1_for_decision_window(m1, train_eval_feat)
    train_eval_env_raw = DummyVecEnv([
        lambda: _CaptureDoneWrapper(
            build_env(train_eval_feat, train_eval_m1, feature_cols,
                      randomize_start=False, episode_steps=None))
    ])
    train_eval_env = VecNormalize(train_eval_env_raw, norm_obs=True, norm_reward=False,
                                  clip_obs=10.0, training=False)

    print(f"Train-eval slice : {len(train_eval_feat):,} bars  "
          f"{train_eval_feat.index.min().date()} to {train_eval_feat.index.max().date()}")
    print(f"Val              : {len(val_feat):,} bars  "
          f"{val_feat.index.min().date()} to {val_feat.index.max().date()}")

    # Sync VecNorm stats to BOTH eval envs before every checkpoint evaluation.
    sync_cb = _SyncVecNormCallback(
        train_env,
        eval_venvs=[train_eval_env, val_env],
        eval_freq=eval_freq,
    )

    # Consistency-based checkpoint selection: saves best model only when the
    # drawdown-penalised score min(q_train, q_val) improves AND both legs are
    # profitable — favours models that are good and stable on both splits.
    eval_cb = _ConsistencyEvalCallback(
        train_eval_env=train_eval_env,
        val_env=val_env,
        eval_freq=eval_freq,
        best_model_save_path=str(Path(out_dir) / "best_model"),
        log_path=str(Path(out_dir) / "eval_logs"),
        dd_penalty=dd_penalty,
        verbose=1,
        train_venv=train_env,
    )

    # batch_size: with n_envs parallel envs each rollout collects
    # n_steps * n_envs transitions.  Use larger batches on GPU.
    effective_buffer = train_episode_steps * n_envs
    batch_size = min(1024 if device == "cuda" else 256, effective_buffer)

    # Generalisation-first PPO regularisation (all knobs in config.py).
    if CFG.ppo_lr_schedule == "linear":
        learning_rate = _linear_schedule(CFG.ppo_learning_rate)
    else:
        learning_rate = CFG.ppo_learning_rate

    model = PPO(
        "MlpPolicy",
        train_env,
        device=device,
        verbose=1,
        seed=seed,
        learning_rate=learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=CFG.ppo_clip_range,
        ent_coef=CFG.ppo_ent_coef,
        vf_coef=0.5,
        n_steps=train_episode_steps,
        batch_size=batch_size,
        n_epochs=CFG.ppo_n_epochs,
        target_kl=CFG.ppo_target_kl,
        # Smaller net than the original [256,128] + weight decay regularise the
        # thin 25-feature signal that the larger net mostly memorised.
        policy_kwargs={
            "net_arch": list(CFG.ppo_net_arch),
            "optimizer_kwargs": {"weight_decay": CFG.ppo_weight_decay},
        },
    )
    print(f"PPO batch_size : {batch_size}  (buffer {effective_buffer} transitions per update)")
    print(f"PPO reg        : lr={CFG.ppo_learning_rate:g}/{CFG.ppo_lr_schedule}  "
          f"ent={CFG.ppo_ent_coef}  epochs={CFG.ppo_n_epochs}  "
          f"target_kl={CFG.ppo_target_kl}  wd={CFG.ppo_weight_decay:g}  "
          f"net={list(CFG.ppo_net_arch)}")

    model.learn(total_timesteps=total_timesteps, callback=CallbackList([sync_cb, eval_cb]))

    # Build a human-readable slug:  ppo_H4_sl1-1.5-2_tp1-1.5-2-3_500k_seed42
    def _fmt(v: float) -> str:
        return str(v).rstrip("0").rstrip(".")
    sl_str = "-".join(_fmt(x) for x in CFG.sl_atr_multipliers)
    tp_str = "-".join(_fmt(x) for x in CFG.tp_r_multipliers)
    slug = (f"ppo_{CFG.decision_timeframe}"
            f"_sl{sl_str}_tp{tp_str}"
            f"_{total_timesteps // 1000}k_seed{seed}")

    model_path = Path(out_dir) / f"{slug}.zip"
    vecnorm_path = Path(out_dir) / f"{slug}_vecnorm.pkl"

    # Keep the explicit `.zip`: the slug contains decimal points, so removing
    # the suffix causes SB3 to save a legacy no-extension file instead.
    model.save(model_path)
    train_env.save(str(vecnorm_path))

    # Metadata file so notebooks/scripts can auto-locate the right pair.
    run_info = {
        "slug": slug,
        "model_path": str(model_path),
        "vecnorm_path": str(vecnorm_path),
        "decision_timeframe": CFG.decision_timeframe,
        "sl_atr_multipliers": list(CFG.sl_atr_multipliers),
        "tp_r_multipliers": list(CFG.tp_r_multipliers),
        "total_timesteps": total_timesteps,
        "train_episode_steps": train_episode_steps,
        "seed": seed,
        "dd_penalty": dd_penalty,
        "risk_fraction": CFG.risk_fraction,
        "spread_price": CFG.spread_price,
        # Data windows this model has SEEN (training + checkpoint selection).
        # final_holdout_eval.py refuses to "reveal" a test split that overlaps them.
        "train_start": str(train_feat.index.min()),
        "train_end": str(train_feat.index.max()),
        "val_start": str(val_feat.index.min()),
        "val_end": str(val_feat.index.max()),
    }
    # If the consistency callback saved a best (eligible) checkpoint, record it +
    # its normalisation snapshot so eval/holdout use the SAME checkpoint we'd ship.
    best_model_zip = Path(out_dir) / "best_model" / "best_model.zip"
    best_model_vecnorm = Path(out_dir) / "best_model" / "best_model_vecnorm.pkl"
    if best_model_zip.exists():
        run_info["best_model_path"] = str(best_model_zip)
        if best_model_vecnorm.exists():
            run_info["best_model_vecnorm_path"] = str(best_model_vecnorm)
    info_path = Path(out_dir) / "run_info.json"
    info_path.write_text(json.dumps(run_info, indent=2))

    print(f"Model      → {model_path}")
    print(f"VecNorm    → {vecnorm_path}")
    print(f"Run info   → {info_path}")

    # ── Post-training equity chart: in-sample → out-of-sample ───────────────
    if reveal_test:
        print("\nRunning post-training evaluation (train / val / test) …")
        eval_splits = {"Train": train_feat, "Val": val_feat, "Test": test_feat}
    else:
        print("\nRunning post-training evaluation (train / val only; test remains sealed) …")
        eval_splits = {"Train": train_feat, "Val": val_feat}
    _post_training_eval(
        model=model,
        vecnorm_path=str(vecnorm_path),
        m1=m1,
        feature_cols=feature_cols,
        splits=eval_splits,
        out_dir=out_dir,
        slug=slug,
    )
    if not reveal_test:
        print("Test split remains sealed. Use final_holdout_eval.py when the model is frozen.")
    return model, train_env


class _CaptureDoneWrapper(gym.Wrapper):
    """Saves equity_curve and trade_log the instant the episode ends.

    DummyVecEnv calls env.reset() automatically when step() returns done=True,
    which wipes BracketTradingEnv.trades before we can read it.  This wrapper
    sits between DummyVecEnv and Monitor, intercepts the done signal inside
    step(), and stashes the results before the auto-reset fires.
    """

    def __init__(self, env):
        super().__init__(env)
        self.saved_equity: pd.DataFrame | None = None
        self.saved_trades: pd.DataFrame | None = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            # Dig down to BracketTradingEnv to capture data before reset.
            inner = self.env
            while hasattr(inner, "env"):
                inner = inner.env
            self.saved_equity = inner.equity_curve()
            self.saved_trades = inner.trade_log()
        return obs, reward, terminated, truncated, info


def _post_training_eval(
    model,
    vecnorm_path: str,
    m1,
    feature_cols: list,
    splits: dict,          # {"Train": df, "Val": df, "Test": df}
    out_dir: str,
    slug: str,
) -> Path:
    """Run the trained model deterministically on every split, chain the equity
    curves into a single in-sample → out-of-sample plot, and save it to disk."""
    from visualize import plot_insample_oos_equity, save_html

    equities: dict = {}

    for name, decision_df in splits.items():
        if decision_df is None or decision_df.empty:
            continue
        m1_sl = _slice_m1_for_decision_window(m1, decision_df)

        # _CaptureDoneWrapper must sit OUTSIDE Monitor so DummyVecEnv's
        # auto-reset (env.reset()) fires on the wrapper, not on the raw env.
        # Stack: DummyVecEnv → _CaptureDoneWrapper → Monitor → BracketTradingEnv
        def _make(d=decision_df, m=m1_sl):
            return _CaptureDoneWrapper(
                build_env(d, m, feature_cols, randomize_start=False, episode_steps=None)
            )

        raw  = DummyVecEnv([_make])
        venv = VecNormalize.load(vecnorm_path, raw)
        venv.training    = False
        venv.norm_reward = False

        obs  = venv.reset()
        done = [False]
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = venv.step(action)

        # Read from the wrapper (captured before DummyVecEnv reset fired).
        capture: _CaptureDoneWrapper = venv.venv.envs[0]
        eq     = capture.saved_equity if capture.saved_equity is not None else pd.DataFrame()
        trades = capture.saved_trades if capture.saved_trades is not None else pd.DataFrame()

        equities[name] = eq

        final    = float(eq["equity"].iloc[-1]) if not eq.empty else CFG.initial_equity
        n_trades = len(trades)
        ret      = (final / CFG.initial_equity - 1) * 100
        print(f"  {name:<6}  trades={n_trades:>4}  "
              f"final=${final:>9,.0f}  return={ret:+.1f}%")

    fig = plot_insample_oos_equity(
        equities,
        initial_equity=CFG.initial_equity,
        title=(f"Equity — in-sample vs out-of-sample  [{slug}]"),
    )
    out_path = Path(out_dir) / f"{slug}_equity_insample_oos.html"
    save_html(fig, out_path)
    print(f"\nEquity chart → {out_path}")
    return out_path


def _rollout_on_split(model, vecnorm_path, m1, feature_cols, decision_df):
    """Deterministic rollout of a model over one decision split.

    Returns (equity_df, trades_df, report_dict).  VecNormalize stats are loaded
    from disk and frozen (training=False, norm_reward=False) so evaluation never
    updates them."""
    from evaluate import full_report

    m1_sl = _slice_m1_for_decision_window(m1, decision_df)

    def _make():
        return _CaptureDoneWrapper(
            build_env(decision_df, m1_sl, feature_cols,
                      randomize_start=False, episode_steps=None)
        )

    raw = DummyVecEnv([_make])
    if vecnorm_path is not None and Path(vecnorm_path).exists():
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
    return eq, trades, rep["value"].to_dict()


def evaluate_on_split(model, vecnorm_path, m1, feature_cols, decision_df) -> dict:
    """full_report() metrics dict for a deterministic rollout over one split.
    Thin wrapper over _rollout_on_split(); shared with training_diagnostics.py."""
    _eq, _trades, rep = _rollout_on_split(model, vecnorm_path, m1, feature_cols, decision_df)
    return rep


def _passes_consistency_gate(
    summary: pd.DataFrame,
    ret_col: str = "val_return_pct",
    pf_col: str = "val_profit_factor",
    sharpe_col: str = "val_sharpe",
) -> tuple[bool, list[str]]:
    """Decide whether the walk-forward folds are consistent enough to deploy.

    A strategy that only works on some folds has no robust edge, so the gate
    requires breadth (enough folds genuinely profitable), a floor (no single
    fold catastrophic), and a positive mean risk-adjusted result.  Thresholds
    live in config.py.  Pass ret_col/pf_col/sharpe_col="test_*" to gate on the
    true out-of-sample test windows (sliding walk-forward) instead of val.
    """
    n = len(summary)
    ret = summary[ret_col]
    pf = summary[pf_col]
    sharpe = summary[sharpe_col]

    good = int(((ret > 0) & (pf > CFG.gate_min_profit_factor)).sum())
    worst_pf = float(pf.min())          # NaN folds (no trades) skipped by .min()
    mean_sharpe = float(sharpe.mean())

    c_count = good >= CFG.min_consistent_folds
    c_worst = worst_pf >= CFG.gate_worst_fold_min_pf
    c_sharpe = (mean_sharpe > 0) if CFG.gate_require_mean_sharpe_positive else True
    passed = bool(c_count and c_worst and c_sharpe)

    ok = lambda b: "OK  " if b else "FAIL"
    detail = [
        f"[{ok(c_count)}] folds with return>0 & PF>{CFG.gate_min_profit_factor:g}: "
        f"{good}/{n}  (need >= {CFG.min_consistent_folds})",
        f"[{ok(c_worst)}] worst-fold val PF: {worst_pf:.2f}  "
        f"(need >= {CFG.gate_worst_fold_min_pf:g})",
    ]
    if CFG.gate_require_mean_sharpe_positive:
        detail.append(f"[{ok(c_sharpe)}] mean val Sharpe: {mean_sharpe:+.2f}  (need > 0)")
    return passed, detail


def _promote_fold_to_production(out_dir: str, fold_k: int, subdir: str = "walk_forward",
                                gate_passed: bool = True) -> None:
    """Copy a walk-forward fold's artifacts up to the production slot (out_dir)
    and write a production run_info.json that points at the best (eligible)
    checkpoint + its normalisation snapshot when available.  ``gate_passed`` is
    recorded in run_info so the saved model is always available even on a failed
    gate, while downstream can still tell it was not approved."""
    src = Path(out_dir) / subdir / f"fold_{fold_k}"
    dst = Path(out_dir)
    _, fr = load_run_info(src)
    slug = fr["slug"]

    for name in (f"{slug}.zip", f"{slug}_vecnorm.pkl",
                 f"{slug}_equity_insample_oos.html"):
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)

    if (src / "best_model").exists():
        shutil.copytree(src / "best_model", dst / "best_model", dirs_exist_ok=True)

    # Mirror the fold's consistency log to the conventional production location.
    src_log = src / "eval_logs" / "consistency_evals.csv"
    if src_log.exists():
        (dst / "eval_logs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_log, dst / "eval_logs" / "consistency_evals.csv")

    prod = dict(fr)
    prod["model_path"] = str(dst / f"{slug}.zip")
    prod["vecnorm_path"] = str(dst / f"{slug}_vecnorm.pkl")
    bm = dst / "best_model" / "best_model.zip"
    bmv = dst / "best_model" / "best_model_vecnorm.pkl"
    if bm.exists():
        prod["best_model_path"] = str(bm)
        if bmv.exists():
            prod["best_model_vecnorm_path"] = str(bmv)
    prod["promoted_from_fold"] = fold_k
    prod["validation_scheme"] = subdir
    prod["gate_passed"] = bool(gate_passed)
    (dst / "run_info.json").write_text(json.dumps(prod, indent=2))


def _finalize_deployment(out_dir: str, fold_k: int, subdir: str, passed: bool) -> None:
    """Always save the final fold's best model into the production slot; the gate
    only controls the approval flag and the NO_DEPLOY marker (so the model is
    kept for inspection / manual use even when the folds were inconsistent)."""
    _promote_fold_to_production(out_dir, fold_k, subdir=subdir, gate_passed=passed)
    marker = Path(out_dir) / "NO_DEPLOY.txt"
    if passed:
        marker.unlink(missing_ok=True)   # clear any stale marker from a prior run
        print(f"\n  ✓ GATE PASSED — fold {fold_k} promoted to production at {out_dir}/")
        print("    run_info.json (gate_passed=true) → best_model/best_model.zip + its vecnorm")
    else:
        marker.write_text(
            f"Walk-forward consistency gate FAILED.\n"
            f"The best model of the final fold ({fold_k}) IS still saved to "
            f"{out_dir}/best_model/ and referenced by run_info.json "
            f"(gate_passed=false) — but it is NOT gate-approved for live "
            f"deployment. Inspect the per-fold summary before any use.\n"
        )
        print(f"\n  ✗ GATE FAILED — best model still saved to {out_dir}/best_model/ "
              f"(gate_passed=false, NOT approved for deployment).")
        print("    Out-of-sample folds are inconsistent; inspect the summary before using it.")


def train_walk_forward(
    total_timesteps: int = 5_000_000,        # budget for the LARGEST fold
    min_timesteps_per_fold: int = 2000_000,   # floor so the smallest fold isn't starved
    target_evals_per_fold: int = 20,         # checkpoints scored per fold (caps eval overhead)
    seed: int = 42,
    out_dir: str = "models",
    train_episode_steps: int = 2048,
    dd_penalty: float = 1.0,
    n_envs: int = 4,                         # SubprocVecEnv workers (CPU). Each worker copies its
                                             # fold's M1 slice under Windows 'spawn'; drop to 2 if RAM-tight.
    device: str = "auto",
):
    """Walk-forward validation + gated deployment.

    Trains an independent PPO model on each rolling fold and scores it on that
    fold's out-of-sample validation window.  The honest generalisation estimate
    is the *aggregate* of the per-fold metrics — not any single split.

    Every fold (including the last) trains into ``out_dir/walk_forward/fold_k``.
    After the loop, a consistency gate (see _passes_consistency_gate / config)
    decides whether to PROMOTE the final fold — the most history, most-recent
    val window — into the production slot ``out_dir``.  On failure nothing is
    promoted, so final_holdout_eval.py / training_diagnostics.py find no
    production model rather than an overfit one.  The sealed test is untouched.

    Per-fold timesteps are scaled by the fold's train length (capped at
    ``total_timesteps`` for the largest fold, floored at ``min_timesteps_per_fold``)
    so every fold makes a similar number of passes over its data — otherwise the
    small early folds over-train and the large late ones under-train.  eval_freq
    is derived per fold to hit ~``target_evals_per_fold`` checkpoints.
    """
    m1, feature_cols, folds, test_feat = load_folds()
    n_folds = len(folds)

    # Scale each fold's budget by its train size (equalise passes over the data).
    train_lens = [len(tr) for tr, _ in folds]
    max_len = max(train_lens)
    fold_timesteps = [
        max(min_timesteps_per_fold, int(round(total_timesteps * L / max_len)))
        for L in train_lens
    ]

    print("=" * 72)
    print(f"  WALK-FORWARD VALIDATION — {n_folds} folds"
          f"  ({'anchored / expanding' if CFG.walk_forward_anchored else 'rolling fixed-window'} train)")
    print(f"  budget <= {total_timesteps:,} steps/fold (scaled by train size, floor "
          f"{min_timesteps_per_fold:,}); total ~= {sum(fold_timesteps):,} steps over {n_folds} folds")
    print(f"  envs: {n_envs}   sealed test: {len(test_feat):,} bars"
          + (f"  ({test_feat.index.min().date()}→{test_feat.index.max().date()})" if len(test_feat) else ""))
    print("=" * 72)

    summary_rows: list[dict] = []
    for k, (tr, va) in enumerate(folds, start=1):
        fold_dir = str(Path(out_dir) / "walk_forward" / f"fold_{k}")
        fold_ts = fold_timesteps[k - 1]
        eval_freq_k = max(25_000, fold_ts // target_evals_per_fold)
        passes = fold_ts / max(len(tr), 1)
        print(f"\n── Fold {k}/{n_folds}"
              f"  | train {len(tr):,} bars ({tr.index.min().date()}→{tr.index.max().date()})"
              f"  | val {len(va):,} bars ({va.index.min().date()}→{va.index.max().date()})"
              f"\n   budget {fold_ts:,} steps (~{passes:.0f} passes), eval every "
              f"{eval_freq_k:,}  → {fold_dir}")

        model, _ = train(
            total_timesteps=fold_ts,
            seed=seed,
            out_dir=fold_dir,
            train_episode_steps=train_episode_steps,
            eval_freq=eval_freq_k,
            dd_penalty=dd_penalty,
            n_envs=n_envs,
            device=device,
            reveal_test=False,
            datasets=(m1, feature_cols, tr, va, test_feat),
        )

        # Re-evaluate the just-trained fold model on its own (OOS) val window,
        # preferring the best (eligible) checkpoint + its normalisation snapshot.
        _, run_info = load_run_info(fold_dir)
        if "best_model_vecnorm_path" in run_info:
            vecnorm_path = Path(fold_dir) / "best_model" / "best_model_vecnorm.pkl"
        else:
            vecnorm_path = Path(fold_dir) / Path(run_info["vecnorm_path"]).name
        rep = evaluate_on_split(model, vecnorm_path, m1, feature_cols, va)
        summary_rows.append({
            "fold": k,
            "train_bars": len(tr),
            "val_bars": len(va),
            "val_start": va.index.min().date(),
            "val_end": va.index.max().date(),
            "val_return_pct": rep.get("total_return_pct"),
            "val_sharpe": rep.get("sharpe_like"),
            "val_profit_factor": rep.get("profit_factor"),
            "val_win_rate_pct": rep.get("win_rate_pct"),
            "val_max_dd_pct": rep.get("max_drawdown_pct"),
            "val_avg_r": rep.get("avg_r"),
            "val_n_trades": rep.get("n_trades"),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = Path(out_dir) / "walk_forward_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 72)
    print("  WALK-FORWARD VALIDATION SUMMARY (out-of-sample, per fold)")
    print("=" * 72)
    print(summary.to_string(index=False))

    metric_cols = ["val_return_pct", "val_sharpe", "val_profit_factor",
                   "val_win_rate_pct", "val_max_dd_pct", "val_avg_r"]
    agg = pd.DataFrame({
        "mean": summary[metric_cols].mean(),
        "std":  summary[metric_cols].std(),
        "min":  summary[metric_cols].min(),
        "max":  summary[metric_cols].max(),
    })
    print("\n  Aggregate across folds:")
    print(agg.to_string())

    pos = int((summary["val_return_pct"] > 0).sum())
    pf_ok = int((summary["val_profit_factor"] > 1.0).sum())
    print(f"\n  Folds with positive OOS return : {pos}/{n_folds}")
    print(f"  Folds with OOS profit factor>1 : {pf_ok}/{n_folds}")

    # ── Deployment gate ──────────────────────────────────────────────────────
    passed, detail = _passes_consistency_gate(summary)
    print("\n  Consistency gate:")
    for line in detail:
        print(f"    {line}")
    # The final fold's best model is always saved to out_dir/best_model/; the
    # gate only sets gate_passed + the NO_DEPLOY marker.
    _finalize_deployment(out_dir, n_folds, "walk_forward", passed)
    if not passed:
        print("    Fix upstream (fewer steps / more regularisation / signal) and re-run.")

    print(f"\n  Per-fold summary → {summary_path}")
    print("  Sealed test remains untouched — reveal once via final_holdout_eval.py.")
    print("=" * 72)
    return summary


def _load_fold_model(fold_dir: str):
    """Load a fold's deployable model + its vecnorm path, preferring the best
    (eligible) checkpoint and its normalisation snapshot when present."""
    from stable_baselines3 import PPO
    _, ri = load_run_info(fold_dir)
    if "best_model_path" in ri and Path(ri["best_model_path"]).exists():
        model = PPO.load(ri["best_model_path"])
        vp = ri.get("best_model_vecnorm_path", ri["vecnorm_path"])
    else:
        model = PPO.load(ri["model_path"])
        vp = ri["vecnorm_path"]
    return model, Path(vp)


def train_sliding_walk_forward(
    total_timesteps: int = 3_000_000,
    seed: int = 42,
    out_dir: str = "models",
    train_episode_steps: int = 2048,
    target_evals_per_fold: int = 20,
    dd_penalty: float = 0.8,
    n_envs: int = 4,
    device: str = "auto",
):
    """Sliding-window walk-forward that simulates periodic retraining.

    Each fold trains on ``CFG.sliding_train_years`` of data, selects its
    checkpoint on the next ``CFG.sliding_val_months`` (val), and is judged
    TRUE out-of-sample on the following ``CFG.sliding_test_months`` (test) — a
    window the model never saw during training OR checkpoint selection.  The
    whole window then slides ``CFG.sliding_step_months`` forward and repeats.

    Every fold's test window is stitched into one continuous out-of-sample
    equity curve (``sliding_oos_equity.csv``) — the realistic backtest of
    "retrain every step_months, trade the next test_months live".  The gate and
    aggregate are computed on the TEST metrics (the honest ones), and the final
    (most recent) fold is the deployable model.

    Per-fold timesteps are fixed because every train window is the same calendar
    length, so equal timesteps already means equal passes over the data.
    """
    from evaluate import full_report

    m1, feat, feature_cols = _load_decision_features()
    folds = make_sliding_folds(
        feat,
        train_years=CFG.sliding_train_years,
        val_months=CFG.sliding_val_months,
        test_months=CFG.sliding_test_months,
        step_months=CFG.sliding_step_months,
        embargo_bars=CFG.split_embargo_bars,
    )
    n_folds = len(folds)
    if n_folds == 0:
        raise ValueError("No sliding folds produced — not enough data for the "
                         "chosen train/val/test window. Check CFG.sliding_* / dataset.")

    print("=" * 72)
    print(f"  SLIDING WALK-FORWARD — {n_folds} folds  "
          f"(train {CFG.sliding_train_years:g}y → val {CFG.sliding_val_months}m → "
          f"test {CFG.sliding_test_months}m, slide {CFG.sliding_step_months}m)")
    print(f"  {total_timesteps:,} steps/fold × {n_folds} folds = "
          f"~{total_timesteps * n_folds:,} env-steps total. envs={n_envs}.")
    print("  Schedule (each row = retrain + 'live' test window):")
    for k, (tr, va, te) in enumerate(folds, start=1):
        print(f"    fold {k:>2}: train {tr.index.min().date()}→{tr.index.max().date()}"
              f" | val {va.index.min().date()}→{va.index.max().date()}"
              f" | TEST {te.index.min().date()}→{te.index.max().date()} ({len(te):,} bars)")
    print("=" * 72)

    summary_rows: list[dict] = []
    test_equities: list[pd.DataFrame] = []
    test_trade_logs: list[pd.DataFrame] = []

    for k, (tr, va, te) in enumerate(folds, start=1):
        fold_dir = str(Path(out_dir) / "sliding" / f"fold_{k}")
        eval_freq_k = max(25_000, total_timesteps // target_evals_per_fold)
        print(f"\n── Fold {k}/{n_folds}"
              f"  | train {len(tr):,}  val {len(va):,}  test {len(te):,} bars"
              f"  | TEST {te.index.min().date()}→{te.index.max().date()}  → {fold_dir}")

        train(
            total_timesteps=total_timesteps,
            seed=seed,
            out_dir=fold_dir,
            train_episode_steps=train_episode_steps,
            eval_freq=eval_freq_k,
            dd_penalty=dd_penalty,
            n_envs=n_envs,
            device=device,
            reveal_test=False,
            datasets=(m1, feature_cols, tr, va, te),
        )

        # Evaluate the deployable (best) checkpoint on val (reference) and on the
        # untouched TEST window (the honest out-of-sample result).
        model, vp = _load_fold_model(fold_dir)
        val_rep = evaluate_on_split(model, vp, m1, feature_cols, va)
        test_eq, test_trades, test_rep = _rollout_on_split(model, vp, m1, feature_cols, te)
        test_equities.append(test_eq)
        if test_trades is not None and not test_trades.empty:
            test_trade_logs.append(test_trades)

        summary_rows.append({
            "fold": k,
            "train_start": tr.index.min().date(), "train_end": tr.index.max().date(),
            "test_start": te.index.min().date(), "test_end": te.index.max().date(),
            "val_return_pct": val_rep.get("total_return_pct"),
            "val_profit_factor": val_rep.get("profit_factor"),
            "test_return_pct": test_rep.get("total_return_pct"),
            "test_sharpe": test_rep.get("sharpe_like"),
            "test_profit_factor": test_rep.get("profit_factor"),
            "test_win_rate_pct": test_rep.get("win_rate_pct"),
            "test_max_dd_pct": test_rep.get("max_drawdown_pct"),
            "test_avg_r": test_rep.get("avg_r"),
            "test_n_trades": test_rep.get("n_trades"),
        })
        print(f"   fold {k} TEST: return={test_rep.get('total_return_pct'):+.1f}%  "
              f"PF={test_rep.get('profit_factor'):.2f}  "
              f"Sharpe={test_rep.get('sharpe_like'):+.2f}  "
              f"trades={test_rep.get('n_trades')}")

    summary = pd.DataFrame(summary_rows)
    summary_path = Path(out_dir) / "sliding_walk_forward_summary.csv"
    summary.to_csv(summary_path, index=False)

    # ── Stitch every fold's test window into one continuous OOS equity curve ──
    running = CFG.initial_equity
    parts = []
    for eq in test_equities:
        if eq is None or eq.empty or "equity" not in eq:
            continue
        s = eq["equity"].astype(float)
        scaled = s / CFG.initial_equity * running          # chain (compound) folds
        parts.append(scaled)
        running = float(scaled.iloc[-1])
    stitched = pd.concat(parts) if parts else pd.Series(dtype=float)
    stitched_df = stitched.to_frame("equity")
    stitched_path = Path(out_dir) / "sliding_oos_equity.csv"
    stitched_df.to_csv(stitched_path)

    all_trades = pd.concat(test_trade_logs, ignore_index=True) if test_trade_logs else pd.DataFrame()
    oos = full_report(stitched_df, all_trades, initial_equity=CFG.initial_equity,
                      periods_per_year=CFG.periods_per_year)["value"].to_dict()

    print("\n" + "=" * 72)
    print("  SLIDING WALK-FORWARD — PER-FOLD TEST (true out-of-sample)")
    print("=" * 72)
    show = ["fold", "test_start", "test_end", "test_return_pct", "test_sharpe",
            "test_profit_factor", "test_win_rate_pct", "test_max_dd_pct", "test_n_trades"]
    print(summary[show].to_string(index=False))

    pos = int((summary["test_return_pct"] > 0).sum())
    pf_ok = int((summary["test_profit_factor"] > 1.0).sum())
    print(f"\n  Test folds positive return : {pos}/{n_folds}")
    print(f"  Test folds profit factor>1 : {pf_ok}/{n_folds}")
    print("\n  STITCHED out-of-sample track record (compounded across all test windows):")
    print(f"    total return : {oos.get('total_return_pct'):+.1f}%")
    print(f"    Sharpe-like  : {oos.get('sharpe_like'):+.2f}")
    print(f"    max drawdown : {oos.get('max_drawdown_pct'):+.1f}%")
    print(f"    profit factor: {oos.get('profit_factor'):.2f}   trades: {oos.get('n_trades')}")
    print(f"    curve → {stitched_path}")

    # ── Deployment gate on the TEST windows ──────────────────────────────────
    passed, detail = _passes_consistency_gate(
        summary, ret_col="test_return_pct",
        pf_col="test_profit_factor", sharpe_col="test_sharpe")
    print("\n  Consistency gate (on test windows):")
    for line in detail:
        print(f"    {line}")
    # The final fold's best model is always saved to out_dir/best_model/ (even on
    # a failed gate); the gate only sets gate_passed + the NO_DEPLOY marker.
    _finalize_deployment(out_dir, n_folds, "sliding", passed)

    print(f"\n  Per-fold summary → {summary_path}")
    print("=" * 72)
    return summary


if __name__ == "__main__":
    # Default entry point: sliding-window walk-forward (realistic retrain
    # simulation; stitches a continuous out-of-sample test track record).
    # For the block-fold scheme call train_walk_forward(); for a single
    # chronological split call train() directly.
    train_sliding_walk_forward()
