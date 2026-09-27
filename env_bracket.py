from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover
    raise ImportError("Install gymnasium first: pip install gymnasium") from exc


@dataclass
class Position:
    direction: int = 0  # +1 long, -1 short, 0 flat
    entry_time: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    units: float = 0.0
    risk_cash: float = 0.0
    sl_distance: float = 0.0
    tp_r: float = 0.0          # planned TP R-multiple (bracket choice)
    sl_atr_mult: float = 0.0   # planned SL ATR multiplier (bracket choice)
    bars_in_trade: int = 0


class BracketTradingEnv(gym.Env):
    """Gymnasium bracket-order trading environment.

    One step = one decision bar (e.g. M5). TP/SL detection is simulated inside
    the next interval using M1 execution bars.

    Action: MultiDiscrete([3, n_sl_buckets, n_tp_buckets])
        direction: 0 flat/close, 1 long, 2 short
        sl bucket: ATR multiplier index
        tp bucket: R-multiple index
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        decision_df: pd.DataFrame,
        m1_df: pd.DataFrame,
        feature_cols: List[str],
        sl_atr_multipliers=(1.0, 1.5, 2.0),
        tp_r_multipliers=(1.0, 1.5, 2.0, 3.0),
        initial_equity: float = 10_000.0,
        risk_fraction: float = 0.005,
        spread_price: float = 0.20,
        slippage_price: float = 0.02,
        commission_per_trade: float = 0.0,
        holding_penalty: float = 0.00002,
        reward_mtm_weight: float = 0.01,
        max_episode_steps: Optional[int] = None,
        randomize_start: bool = False,
    ):
        super().__init__()
        self.decision_df = decision_df.dropna(subset=feature_cols + ["atr"]).copy()
        self.m1_df = m1_df.copy()
        self.feature_cols = list(feature_cols)
        self.sl_atr_multipliers = tuple(sl_atr_multipliers)
        self.tp_r_multipliers = tuple(tp_r_multipliers)
        self.initial_equity = float(initial_equity)
        self.risk_fraction = float(risk_fraction)
        self.spread_price = float(spread_price)
        self.slippage_price = float(slippage_price)
        self.commission_per_trade = float(commission_per_trade)
        self.holding_penalty = float(holding_penalty)
        self.reward_mtm_weight = float(reward_mtm_weight)
        self.max_episode_steps = max_episode_steps or (len(self.decision_df) - 2)

        self.randomize_start = randomize_start

        # Pre-extract M1 high/low as contiguous numpy arrays and cache the
        # sorted DatetimeIndex for O(log n) searchsorted lookups.
        # This replaces the O(n) boolean-mask scan on every step — with 1.5 M
        # M1 bars the speedup is ~100-200x per step.
        self._m1_open  = self.m1_df["Open"].to_numpy(dtype=np.float64)
        self._m1_high  = self.m1_df["High"].to_numpy(dtype=np.float64)
        self._m1_low   = self.m1_df["Low"].to_numpy(dtype=np.float64)
        self._m1_index = self.m1_df.index  # DatetimeIndex (sorted, tz-aware)

        self.action_space = spaces.MultiDiscrete([3, len(self.sl_atr_multipliers), len(self.tp_r_multipliers)])

        # Market features + position state.
        self.n_pos_features = 6
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_cols) + self.n_pos_features,),
            dtype=np.float32,
        )
        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.randomize_start:
            # Random start so each episode covers a different segment of history.
            # Reserve enough room for at least max_episode_steps remaining bars.
            headroom = max(self.max_episode_steps + 2, 2)
            max_start = max(len(self.decision_df) - headroom, 1)
            self.i = int(self.np_random.integers(0, max_start))
        else:
            self.i = 0
        self.steps = 0
        self.equity = self.initial_equity
        self.realized_pnl = 0.0
        self.position = Position()
        self.trades: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        obs = self._observation()
        return obs, {}

    def _current_row(self):
        return self.decision_df.iloc[self.i]

    def _current_time(self):
        return self.decision_df.index[self.i]

    def _next_time(self):
        return self.decision_df.index[min(self.i + 1, len(self.decision_df) - 1)]

    def _position_state_features(self) -> np.ndarray:
        row = self._current_row()
        close = float(row["Close"])
        atr = max(float(row["atr"]), 1e-12)
        p = self.position
        if p.direction == 0:
            return np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

        unrealized = (close - p.entry_price) * p.units * p.direction
        unrealized_r = unrealized / max(p.risk_cash, 1e-12)
        dist_tp_atr = ((p.tp - close) * p.direction) / atr
        dist_sl_atr = ((close - p.sl) * p.direction) / atr
        return np.array([
            p.direction,
            unrealized_r,
            min(p.bars_in_trade / 100.0, 10.0),
            dist_tp_atr,
            dist_sl_atr,
            p.tp_r,
        ], dtype=np.float32)

    def _observation(self):
        row = self._current_row()
        market = row[self.feature_cols].astype(float).to_numpy(dtype=np.float32)
        pos = self._position_state_features()
        obs = np.concatenate([market, pos]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs

    def _entry_price(self, close: float, direction: int) -> float:
        return close + direction * (self.spread_price / 2.0 + self.slippage_price)

    def _exit_price(self, price: float, direction: int) -> float:
        # Long exits at bid below mid; short exits at ask above mid.
        return price - direction * (self.spread_price / 2.0 + self.slippage_price)

    def _open_position(self, direction: int, sl_idx: int, tp_idx: int):
        row = self._current_row()
        close = float(row["Close"])
        atr = max(float(row["atr"]), 1e-12)
        sl_atr_mult = self.sl_atr_multipliers[sl_idx]
        sl_dist = max(sl_atr_mult * atr, 1e-8)
        tp_r = self.tp_r_multipliers[tp_idx]
        entry = self._entry_price(close, direction)
        sl = entry - direction * sl_dist
        tp = entry + direction * tp_r * sl_dist
        risk_cash = max(self.equity * self.risk_fraction, 1e-8)
        units = risk_cash / sl_dist

        self.position = Position(
            direction=direction,
            entry_time=self._current_time(),
            entry_price=entry,
            sl=sl,
            tp=tp,
            units=units,
            risk_cash=risk_cash,
            sl_distance=sl_dist,
            tp_r=tp_r,
            sl_atr_mult=sl_atr_mult,
            bars_in_trade=0,
        )

    def _close_position(self, exit_price_raw: float, exit_time: pd.Timestamp, reason: str,
                        gap_fill: bool = False) -> float:
        p = self.position
        if p.direction == 0:
            return 0.0
        exit_price = self._exit_price(exit_price_raw, p.direction)
        pnl = (exit_price - p.entry_price) * p.units * p.direction - self.commission_per_trade
        self.equity += pnl
        self.realized_pnl += pnl
        r_mult = pnl / max(p.risk_cash, 1e-12)
        self.trades.append({
            "entry_time": p.entry_time,
            "exit_time": exit_time,
            "direction": p.direction,
            "entry_price": p.entry_price,
            "exit_price": exit_price,
            "sl": p.sl,
            "tp": p.tp,
            "sl_atr_mult": p.sl_atr_mult,   # bracket choice: SL ATR multiplier
            "tp_r_bracket": p.tp_r,          # bracket choice: planned TP R-multiple
            "units": p.units,
            "pnl": pnl,
            "r_mult": r_mult,                # realized R-multiple (actual outcome)
            "bars_in_trade": p.bars_in_trade,
            "exit_reason": reason,
            "gap_fill": gap_fill,            # SL filled at a gapped-through open
        })
        self.position = Position()
        return pnl

    def _simulate_m1_until_next_decision(self) -> float:
        """Advance within the next decision interval and close if TP/SL is touched.

        Uses DatetimeIndex.searchsorted (O log n) + pre-cached numpy arrays to
        avoid a full pandas boolean-mask scan on each step (~100-200x faster on
        a multi-year M1 dataset).
        """
        p = self.position
        if p.direction == 0:
            return 0.0

        start = self._current_time()
        end   = self._next_time()

        # Half-open interval (start, end]: "right" excludes start, includes end.
        lo = int(self._m1_index.searchsorted(start, side="right"))
        hi = int(self._m1_index.searchsorted(end,   side="right"))

        realized = 0.0
        for idx in range(lo, hi):
            open_ = self._m1_open[idx]
            high = self._m1_high[idx]
            low  = self._m1_low[idx]
            p = self.position
            if p.direction == 0:
                break

            # Gap-through stop: if the bar OPENS beyond the SL (weekend / news
            # gap), a stop order fills at the open, not at the SL price.
            # A TP gap is left at the TP price (no price improvement assumed).
            if (open_ - p.sl) * p.direction <= 0:
                realized += self._close_position(open_, self._m1_index[idx], "SL", gap_fill=True)
                break

            if p.direction == 1:
                sl_hit = low  <= p.sl
                tp_hit = high >= p.tp
            else:
                sl_hit = high >= p.sl
                tp_hit = low  <= p.tp

            # Pessimistic intrabar rule: if both touched, assume SL first.
            if sl_hit:
                realized += self._close_position(p.sl, self._m1_index[idx], "SL")
                break
            if tp_hit:
                realized += self._close_position(p.tp, self._m1_index[idx], "TP")
                break
        return realized

    def step(self, action):
        action = np.asarray(action, dtype=int)
        direction_raw, sl_idx, tp_idx = int(action[0]), int(action[1]), int(action[2])
        desired_direction = {0: 0, 1: 1, 2: -1}[direction_raw]

        prev_equity = self.equity
        reward_risk_unit = max(prev_equity * self.risk_fraction, 1e-12)
        row = self._current_row()
        close = float(row["Close"])

        # Handle explicit close or flip at current decision close.
        if self.position.direction != 0:
            current_dir = self.position.direction
            if desired_direction == 0:
                self._close_position(close, self._current_time(), "manual_close")
            elif desired_direction != current_dir:
                self._close_position(close, self._current_time(), "flip_close")
                self._open_position(desired_direction, sl_idx, tp_idx)

        # Fresh entry if flat and action wants exposure.
        if self.position.direction == 0 and desired_direction != 0:
            self._open_position(desired_direction, sl_idx, tp_idx)

        # Simulate TP/SL using the M1 candles inside this decision interval.
        self._simulate_m1_until_next_decision()

        # Dense mark-to-market reward after interval.
        # Uses the CURRENT bar's close (known at decision time) to avoid lookahead.
        # The primary reward is realized PnL from TP/SL hits above; this is a small
        # shaping term (weight 0.01) that gives the agent a directional signal while
        # the trade is open, without leaking any future price information.
        if self.position.direction != 0:
            self.position.bars_in_trade += 1
            current_close = float(self.decision_df.iloc[self.i]["Close"])
            unrealized = (current_close - self.position.entry_price) * self.position.units * self.position.direction
        else:
            unrealized = 0.0

        reward = (self.equity - prev_equity) / reward_risk_unit
        if self.position.direction != 0:
            reward += (unrealized / max(self.position.risk_cash, 1e-12)) * self.reward_mtm_weight
            reward -= self.holding_penalty

        self.history.append({
            "time": self._current_time(),
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "position": self.position.direction,
            "close": close,
            "reward": reward,
        })

        self.i += 1
        self.steps += 1
        terminated = self.i >= len(self.decision_df) - 2
        truncated = self.steps >= self.max_episode_steps
        obs = self._observation() if not (terminated or truncated) else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {"equity": self.equity, "n_trades": len(self.trades)}
        return obs, float(reward), terminated, truncated, info

    def equity_curve(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame(columns=["time", "equity", "realized_pnl", "position", "close", "reward"])
        return pd.DataFrame(self.history).set_index("time")

    def trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)
