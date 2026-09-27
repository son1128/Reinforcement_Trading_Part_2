# XAUUSD M1 → RL-ready feature + bracket-trading pipeline

This project is built for MT4/MT5-style XAUUSD minute data such as:

```csv
Time (EET),Open,High,Low,Close,Volume
2020.01.09 01:00:00,1557.152,1557.452,1555.202,1555.302,0.045
```

It gives you a complete research pipeline:

1. Load and validate raw M1 data.
2. Parse broker/EET time safely.
3. Resample M1 execution data into M5/M15 decision bars.
4. Build causal stationary features normalized by ATR/price.
5. Visualize candles, indicators, volatility, sessions, feature correlations, trades, equity, and drawdown.
6. Tune a simple trend-following baseline on train/validation splits while keeping the test split sealed.
7. Optionally train PPO with a `MultiDiscrete([direction, SL bucket, TP/R bucket])` action space.
8. Validate with **walk-forward** cross-validation (rolling out-of-sample windows) rather than a single train/val split.
9. Reveal the test split only once via `final_holdout_eval.py` after the model is frozen.

## Quick start

```bash
pip install -r requirements.txt
```

Put your full file here (the long Bid history is the primary dataset; the short
Ask file is kept as a fast smoke-test fallback — switch between them in `config.py`):

```text
data/XAUUSD_1 Min_Bid_2003.05.05_2026.05.31.csv   # primary (~23 years)
data/XAUUSD_1 Min_Ask_2020.01.09_2026.01.15.csv   # smoke-test fallback (~6 years)
```

Then run:

```bash
python run_pipeline.py
```

This keeps the test split sealed and writes validation-only artifacts, including
a `walk_forward_baseline.csv` showing baseline stability across the rolling folds.

Train the RL agent with walk-forward validation (one model per fold, aggregate
out-of-sample report; the final fold is saved as the production model):

```bash
python train_ppo.py               # == train_ppo.train_sliding_walk_forward()
python train_ppo.py --multi-seed  # repeat it for CFG.multi_seeds → models/multi_seed/
```

When you are ready for the one-time holdout reveal, run:

```bash
python final_holdout_eval.py
```

Or open:

```text
notebooks/XAUUSD_RL_Pipeline_Demo.ipynb
```

## Important defaults

- `SOURCE_TZ = Europe/Helsinki`, because many brokers label server time as EET/EEST.
- MT4/MT5 M1 timestamps are treated as candle-open timestamps and shifted to candle-close timestamps before resampling/backtesting.
- `DECISION_TIMEFRAME = H1`, while M1 is retained for intrabar TP/SL simulation.
- Features are causal and mostly stationary: ATR-normalized distances, ratios, session flags, candle shape ratios.
- The observation set is **25 features**. Four redundant ones were dropped in the 2026-06-02 collinearity audit (`rsi_centered`, `roc5_atr`, `body_atr`, `ema50_slope5_atr`); each was |r| ≥ 0.96 with a retained feature (two were exact duplicates). The underlying `ema20/50/200` columns are still computed for the baselines and charts.
- **Sliding-window walk-forward** (`train_ppo.train_sliding_walk_forward`, the default `python train_ppo.py` entry point) is the realistic validator: each fold trains on `sliding_train_years` (5y), selects its checkpoint on the next `sliding_val_months` (6m), and is judged out-of-sample on the following `sliding_test_months` (6m); the window then slides `sliding_step_months` (6m) and repeats. This simulates *retraining every 6 months and trading the next 6 months live*. Every fold's test window is stitched into one continuous out-of-sample equity curve (`models/sliding_oos_equity.csv`); the gate runs on the **test** metrics. Because every train window is the same length, equal timesteps per fold = equal passes. ~35 folds over the 23y dataset (raise `sliding_step_months` to reduce).
- **Block walk-forward** (`train_ppo.train_walk_forward`; `config.py`: `n_walk_forward_folds`, `walk_forward_anchored`, `test_frac`) is the alternative: it seals the last `test_frac` of bars and rolls `n_folds` train→val windows over the rest. `test_frac == 1 - train_frac - val_frac`, so the sealed holdout matches the single-split test exactly.
- When TP and SL are both inside the same M1 candle, the simulator assumes SL first. This is deliberately pessimistic.
- If an M1 bar **opens beyond the SL** (weekend/news gap), the stop fills at that bar's open, not at the SL price (`gap_fill=True` in the trade log). A gap through the TP still fills at the TP price (no price improvement assumed).
- **Holdout guard:** `train_ppo.train()` records the train/val windows in `run_info.json`, and `final_holdout_eval.py` (and `training_diagnostics.py` with `reveal_test=True`) refuse to evaluate a test split the model has already seen. A sliding walk-forward promotes its last fold, whose train/val windows overlap the single-split test, so for that scheme use the per-fold TEST results and `sliding_oos_equity.csv`. Legacy `run_info.json` files without window metadata are refused unless `--allow-unverified` is passed.
- **Execution model:** `price_basis` (`bid`/`ask`/`mid`, inferred from the file name by default) says which quote the CSV holds; buys fill on the ask and sells on the bid, so a short's SL/TP trigger on the ask. SL = stop order (fills at the level, or the gapped open, minus slippage); TP = limit order (fills at the level, no slippage). `spread_mode="relative"` charges `max(min_spread_price, price × spread_bps/1e4)` instead of a fixed dollar spread across 23 years; the default stays `"fixed"` — calibrate either to your broker.
- **Equity is mark-to-market:** `equity` = realized + unrealized PnL at the end of each decision interval (`realized_equity` keeps the old series), so max drawdown includes open-trade losses. Evaluation episodes liquidate any open position at the end (`exit_reason="episode_end"`); training episodes (random fixed-length windows) do not. The MTM reward-shaping term also uses the interval-end close instead of the stale decision close.
- **Over-training controls:** `max_passes_per_fold` caps each fold's timesteps at N passes over its train bars (off by default), and `early_stop_patience` stops a run after that many evaluations without a new best eligible checkpoint (never before `early_stop_min_evals`).
- **Baseline + significance:** the sliding walk-forward also runs the trend baseline on every fold's test window (params tuned on that fold's own train/val when `baseline_tune_per_fold=True`), writes `sliding_oos_equity_baseline.csv`, and bootstrap-tests the results (`sliding_significance.csv`, `significance.py`): mean trade R > 0, daily Sharpe > 0 (moving-block bootstrap), and RL − baseline per fold (bootstrap + exact sign test).
- **Multi-seed:** `train_ppo.train_multi_seed()` repeats the sliding walk-forward for `multi_seeds`, sharing data and the baseline, and reports the per-fold and stitched-OOS spread across seeds.
- Position size is fixed-fractional risk-based; the RL agent controls direction and bracket shape, not size.
- `run_pipeline.py` now performs a temporal train/validation/test split, tunes on train/val, and keeps test sealed by default.
- `training_diagnostics.py` and `train_ppo.py` also keep the test split sealed by default.
- RL rewards are normalized by risk budget rather than raw cash PnL, which is materially more stable for PPO.

## Files

- `config.py`: all main parameters.
- `data_loader.py`: CSV parsing, timezone handling, validation, resampling.
- `features.py`: causal indicators and stationary feature matrix.
- `leakage_checks.py`: simple future-append stability test.
- `env_bracket.py`: Gymnasium-compatible bracket trading environment.
- `baselines.py`: random and EMA/ATR rule policies.
- `evaluate.py`: metrics, trade-log summary, drawdown.
- `significance.py`: bootstrap / sign tests for out-of-sample results.
- `visualize.py`: Plotly visualization functions.
- `train_ppo.py`: optional PPO training scaffold.
- `run_pipeline.py`: one-command pre-test pipeline with validation-only outputs.
- `final_holdout_eval.py`: explicit one-time holdout evaluation entry point.
- `notebooks/XAUUSD_RL_Pipeline_Demo.ipynb`: guided notebook.
