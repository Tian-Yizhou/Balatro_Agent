"""Gymnasium wrapper that records per-episode summary statistics to Parquet.

Produces a single ``.parquet`` file with one row per completed episode.
Rows are flushed in batches for efficiency.  The file is lightweight,
columnar, and directly loadable with ``pandas.read_parquet()`` for
statistical analysis.

Typical usage::

    from balatro_gym.wrappers import EpisodeStatsRecorder

    env = gym.make("Balatro-Easy-v0")
    env = EpisodeStatsRecorder(env, save_path="data/results/run_001.parquet")

    for ep in range(100):
        obs, info = env.reset()
        done = False
        while not done:
            action = ...
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

    env.close()  # flushes remaining rows
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False


# Schema for the parquet file — fixed columns, all non-nullable.
_SCHEMA = None  # lazily built so import succeeds even without pyarrow


def _get_schema() -> "pa.Schema":
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = pa.schema(
            [
                ("episode_id", pa.int32()),
                ("episode_seed_id", pa.string()),
                ("timestamp", pa.float64()),
                ("seed", pa.int64()),
                ("won", pa.bool_()),
                ("antes_beaten", pa.int32()),
                ("blinds_beaten", pa.int32()),
                ("total_steps", pa.int32()),
                ("total_hands_played", pa.int32()),
                ("total_reward", pa.float32()),
                ("max_score", pa.int64()),
                ("final_money", pa.int32()),
                ("final_ante", pa.int32()),
                ("num_jokers_final", pa.int32()),
                ("num_consumables_final", pa.int32()),
                ("max_ante_reached", pa.int32()),
                ("max_blinds_beaten", pa.int32()),
                ("play_actions", pa.int32()),
                ("discard_actions", pa.int32()),
                ("buy_actions", pa.int32()),
                ("sell_actions", pa.int32()),
                ("reroll_actions", pa.int32()),
                ("skip_actions", pa.int32()),
            ]
        )
    return _SCHEMA


# Action range boundaries (imported lazily to avoid circular deps)
_ACTION_BOUNDS: dict[str, tuple[int, int]] | None = None


def _get_action_bounds() -> dict[str, tuple[int, int]]:
    global _ACTION_BOUNDS
    if _ACTION_BOUNDS is None:
        from balatro_gym.envs.balatro_env import (
            PLAY_OFFSET,
            DISCARD_OFFSET,
            BUY_OFFSET,
            SELL_OFFSET,
            REROLL_ACTION,
            SKIP_ACTION,
            NUM_PLAY_ACTIONS,
            NUM_DISCARD_ACTIONS,
            NUM_BUY_ACTIONS,
            NUM_SELL_ACTIONS,
        )

        _ACTION_BOUNDS = {
            "play": (PLAY_OFFSET, PLAY_OFFSET + NUM_PLAY_ACTIONS),
            "discard": (DISCARD_OFFSET, DISCARD_OFFSET + NUM_DISCARD_ACTIONS),
            "buy": (BUY_OFFSET, BUY_OFFSET + NUM_BUY_ACTIONS),
            "sell": (SELL_OFFSET, SELL_OFFSET + NUM_SELL_ACTIONS),
            "reroll": (REROLL_ACTION, REROLL_ACTION + 1),
            "skip": (SKIP_ACTION, SKIP_ACTION + 1),
        }
    return _ACTION_BOUNDS


class EpisodeStatsRecorder(gym.Wrapper):
    """Records per-episode summary statistics to a Parquet file.

    Each completed episode appends one row.  Rows are buffered and flushed
    every *flush_every* episodes (default 10) or on ``close()``.

    Columns recorded:

    ========================= ==============================================
    Column                    Description
    ========================= ==============================================
    episode_id                Sequential episode counter (0-based)
    timestamp                 Unix timestamp when episode ended
    seed                      Seed passed to ``reset()``, -1 if None
    won                       True if all antes were beaten
    antes_beaten              Number of full antes beaten (blinds_beaten // 3)
    blinds_beaten             Total blinds beaten
    total_steps               Steps taken in the episode
    total_hands_played        Poker hands played (not discards)
    total_reward              Sum of rewards across the episode
    max_score                 Highest cumulative score seen in any blind
    final_money               Money at episode end
    final_ante                Ante number at episode end
    num_jokers_final          Jokers held at episode end
    num_consumables_final     Consumables held at episode end
    max_ante_reached          Highest ante entered
    max_blinds_beaten         Same as blinds_beaten (kept for clarity)
    play_actions              Count of play actions taken
    discard_actions           Count of discard actions taken
    buy_actions               Count of buy actions taken
    sell_actions              Count of sell actions taken
    reroll_actions            Count of reroll actions taken
    skip_actions              Count of skip actions taken
    ========================= ==============================================

    Args:
        env: The Gymnasium environment to wrap.
        save_path: Path for the ``.parquet`` output file.
        flush_every: Flush buffered rows to disk every N episodes.

    Raises:
        ImportError: If ``pyarrow`` is not installed.
    """

    def __init__(
        self,
        env: gym.Env,
        save_path: str | Path = "data/results/episode_stats.parquet",
        flush_every: int = 10,
    ):
        if not _HAS_PYARROW:
            raise ImportError(
                "EpisodeStatsRecorder requires pyarrow. "
                "Install it with: pip install pyarrow"
            )
        super().__init__(env)
        self._save_path = Path(save_path)
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_every = max(1, flush_every)

        self._episode_id: int = 0
        self._writer: pq.ParquetWriter | None = None
        self._row_buf: list[dict[str, Any]] = []

        # Per-episode accumulators (reset each episode)
        self._ep_seed: int = -1
        self._ep_steps: int = 0
        self._ep_reward: float = 0.0
        self._ep_max_score: int = 0
        self._ep_max_ante: int = 0
        self._action_counts: dict[str, int] = {}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset per-episode accumulators
        self._ep_seed_id = info.get("episode_seed_id", "")
        self._ep_seed = seed if seed is not None else -1
        self._ep_steps = 0
        self._ep_reward = 0.0
        self._ep_max_score = 0
        self._ep_max_ante = info.get("ante", 1)
        self._action_counts = {
            "play": 0,
            "discard": 0,
            "buy": 0,
            "sell": 0,
            "reroll": 0,
            "skip": 0,
        }

        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._ep_steps += 1
        self._ep_reward += reward
        self._ep_max_score = max(self._ep_max_score, info.get("score", 0))
        self._ep_max_ante = max(self._ep_max_ante, info.get("ante", 0))

        # Classify action
        bounds = _get_action_bounds()
        for name, (lo, hi) in bounds.items():
            if lo <= action < hi:
                self._action_counts[name] += 1
                break

        if terminated or truncated:
            self._record_episode(info)

        return obs, reward, terminated, truncated, info

    def _record_episode(self, final_info: dict[str, Any]) -> None:
        """Collect final stats and buffer the row."""
        won = final_info.get("phase", "") == "game_won"
        blinds_beaten = final_info.get("blinds_beaten", 0)

        row = {
            "episode_id": self._episode_id,
            "episode_seed_id": self._ep_seed_id,
            "timestamp": time.time(),
            "seed": self._ep_seed,
            "won": won,
            "antes_beaten": blinds_beaten // 3,
            "blinds_beaten": blinds_beaten,
            "total_steps": self._ep_steps,
            "total_hands_played": self._action_counts["play"],
            "total_reward": float(self._ep_reward),
            "max_score": self._ep_max_score,
            "final_money": final_info.get("money", 0),
            "final_ante": final_info.get("ante", 0),
            "num_jokers_final": final_info.get("num_jokers", 0),
            "num_consumables_final": final_info.get("num_consumables", 0),
            "max_ante_reached": self._ep_max_ante,
            "max_blinds_beaten": blinds_beaten,
            "play_actions": self._action_counts["play"],
            "discard_actions": self._action_counts["discard"],
            "buy_actions": self._action_counts["buy"],
            "sell_actions": self._action_counts["sell"],
            "reroll_actions": self._action_counts["reroll"],
            "skip_actions": self._action_counts["skip"],
        }
        self._row_buf.append(row)
        self._episode_id += 1

        if len(self._row_buf) >= self._flush_every:
            self._flush()

    def _flush(self) -> None:
        """Write buffered rows to the parquet file."""
        if not self._row_buf:
            return

        schema = _get_schema()
        # Build columnar arrays from row dicts
        columns = {col.name: [] for col in schema}
        for row in self._row_buf:
            for key in columns:
                columns[key].append(row[key])

        table = pa.table(columns, schema=schema)

        if self._writer is None:
            self._writer = pq.ParquetWriter(
                str(self._save_path), schema, compression="snappy"
            )
        self._writer.write_table(table)
        self._row_buf.clear()

    def close(self) -> None:
        """Flush remaining rows and close the parquet writer."""
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        super().close()

    @staticmethod
    def load(path: str | Path) -> "pa.Table":
        """Load the parquet file as a PyArrow Table.

        For pandas: ``EpisodeStatsRecorder.load("path").to_pandas()``
        """
        return pq.read_table(str(path))
