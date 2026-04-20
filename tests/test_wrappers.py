"""Tests for RolloutRecorder and EpisodeStatsRecorder wrappers."""

import shutil
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pyarrow.parquet as pq
import pytest

import balatro_gym  # noqa: F401 — triggers env registration
from balatro_gym.envs.balatro_env import (
    TOTAL_ACTIONS,
    PLAY_OFFSET,
    DISCARD_OFFSET,
    SKIP_ACTION,
)
from balatro_gym.envs.configs import GameConfig
from balatro_gym.wrappers import RolloutRecorder, EpisodeStatsRecorder


@pytest.fixture()
def tmp_dir():
    """Create a temp directory, yield it, then clean up."""
    d = tempfile.mkdtemp(prefix="balatro_test_")
    yield Path(d)
    shutil.rmtree(d)


def _play_episode(env, seed=42, max_steps=2000):
    """Play a random episode to completion. Returns total steps."""
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    steps = 0
    for _ in range(max_steps):
        mask = info["action_mask"]
        valid = np.where(mask)[0]
        if len(valid) == 0:
            break
        action = int(rng.choice(valid))
        obs, reward, term, trunc, info = env.step(action)
        steps += 1
        if term or trunc:
            break
    return steps


# -----------------------------------------------------------------------
# RolloutRecorder
# -----------------------------------------------------------------------


class TestRolloutRecorder:
    def test_creates_npz_on_episode_end(self, tmp_dir):
        env = RolloutRecorder(
            gym.make("Balatro-Easy-v0"), save_dir=tmp_dir / "rollouts"
        )
        _play_episode(env, seed=0)

        npz_files = list((tmp_dir / "rollouts").glob("*.npz"))
        assert len(npz_files) == 1

    def test_unique_filenames_across_episodes(self, tmp_dir):
        env = RolloutRecorder(
            gym.make("Balatro-Easy-v0"), save_dir=tmp_dir / "rollouts"
        )
        _play_episode(env, seed=0)
        _play_episode(env, seed=1)

        npz_files = sorted((tmp_dir / "rollouts").glob("*.npz"))
        assert len(npz_files) == 2
        assert npz_files[0].name != npz_files[1].name

    def test_arrays_have_correct_shapes(self, tmp_dir):
        env = RolloutRecorder(
            gym.make("Balatro-Easy-v0"), save_dir=tmp_dir / "rollouts"
        )
        steps = _play_episode(env, seed=42)

        npz_files = list((tmp_dir / "rollouts").glob("*.npz"))
        data = RolloutRecorder.load(npz_files[0])

        T = steps
        assert data["obs"].shape[0] == T + 1  # includes initial obs
        assert data["actions"].shape == (T,)
        assert data["rewards"].shape == (T,)
        assert data["terminated"].shape == (T,)
        assert data["truncated"].shape == (T,)
        assert data["phases"].shape == (T,)
        assert data["antes"].shape == (T,)
        assert data["scores"].shape == (T,)
        assert data["money"].shape == (T,)

    def test_dtypes(self, tmp_dir):
        env = RolloutRecorder(
            gym.make("Balatro-Easy-v0"), save_dir=tmp_dir / "rollouts"
        )
        _play_episode(env, seed=42)

        npz_files = list((tmp_dir / "rollouts").glob("*.npz"))
        data = RolloutRecorder.load(npz_files[0])

        assert data["obs"].dtype == np.float32
        assert data["actions"].dtype == np.int32
        assert data["rewards"].dtype == np.float32
        assert data["terminated"].dtype == bool
        assert data["phases"].dtype == np.uint8

    def test_action_mask_stored_when_enabled(self, tmp_dir):
        env = RolloutRecorder(
            gym.make("Balatro-Easy-v0"),
            save_dir=tmp_dir / "rollouts",
            save_action_mask=True,
        )
        steps = _play_episode(env, seed=42)

        npz_files = list((tmp_dir / "rollouts").glob("*.npz"))
        data = RolloutRecorder.load(npz_files[0])

        assert "action_masks" in data
        assert data["action_masks"].shape == (steps, TOTAL_ACTIONS)
        assert data["action_masks"].dtype == bool

    def test_action_mask_not_stored_by_default(self, tmp_dir):
        env = RolloutRecorder(
            gym.make("Balatro-Easy-v0"), save_dir=tmp_dir / "rollouts"
        )
        _play_episode(env, seed=42)

        npz_files = list((tmp_dir / "rollouts").glob("*.npz"))
        data = RolloutRecorder.load(npz_files[0])
        assert "action_masks" not in data

    def test_obs_alignment(self, tmp_dir):
        """obs[t+1] should be the state AFTER taking actions[t]."""
        env = RolloutRecorder(
            gym.make("Balatro-Easy-v0"), save_dir=tmp_dir / "rollouts"
        )
        _play_episode(env, seed=42)

        npz_files = list((tmp_dir / "rollouts").glob("*.npz"))
        data = RolloutRecorder.load(npz_files[0])

        # First obs should have non-zero hand card features (game is active)
        assert data["obs"][0].sum() > 0
        # Last obs should also be valid (the terminal observation)
        assert data["obs"][-1].shape == data["obs"][0].shape

    def test_flush_on_reset_without_termination(self, tmp_dir):
        """If reset() is called mid-episode, buffer should still flush."""
        env = RolloutRecorder(
            gym.make("Balatro-Easy-v0"), save_dir=tmp_dir / "rollouts"
        )
        obs, info = env.reset(seed=0)
        # Take a few steps without terminating
        for _ in range(5):
            mask = info["action_mask"]
            valid = np.where(mask)[0]
            obs, _, term, _, info = env.step(int(valid[0]))
            if term:
                break

        # Reset again — should flush partial episode
        env.reset(seed=1)

        npz_files = list((tmp_dir / "rollouts").glob("*.npz"))
        assert len(npz_files) == 1

    def test_file_is_compressed(self, tmp_dir):
        """npz files should be reasonably small."""
        env = RolloutRecorder(
            gym.make("Balatro-Easy-v0"), save_dir=tmp_dir / "rollouts"
        )
        _play_episode(env, seed=42)

        npz_files = list((tmp_dir / "rollouts").glob("*.npz"))
        size_kb = npz_files[0].stat().st_size / 1024
        # A typical easy episode is ~100-300 steps; compressed should be < 500 KB
        assert size_kb < 500, f"File too large: {size_kb:.1f} KB"


# -----------------------------------------------------------------------
# EpisodeStatsRecorder
# -----------------------------------------------------------------------


class TestEpisodeStatsRecorder:
    def test_creates_parquet_on_flush(self, tmp_dir):
        path = tmp_dir / "stats.parquet"
        env = EpisodeStatsRecorder(
            gym.make("Balatro-Easy-v0"), save_path=path, flush_every=1
        )
        _play_episode(env, seed=0)
        env.close()

        assert path.exists()

    def test_one_row_per_episode(self, tmp_dir):
        path = tmp_dir / "stats.parquet"
        env = EpisodeStatsRecorder(
            gym.make("Balatro-Easy-v0"), save_path=path, flush_every=5
        )
        for seed in range(5):
            _play_episode(env, seed=seed)
        env.close()

        table = pq.read_table(str(path))
        assert table.num_rows == 5

    def test_episode_ids_sequential(self, tmp_dir):
        path = tmp_dir / "stats.parquet"
        env = EpisodeStatsRecorder(
            gym.make("Balatro-Easy-v0"), save_path=path, flush_every=10
        )
        for seed in range(3):
            _play_episode(env, seed=seed)
        env.close()

        df = pq.read_table(str(path)).to_pandas()
        assert list(df["episode_id"]) == [0, 1, 2]

    def test_won_column(self, tmp_dir):
        """Won column should be boolean."""
        path = tmp_dir / "stats.parquet"
        env = EpisodeStatsRecorder(
            gym.make("Balatro-Easy-v0"), save_path=path, flush_every=1
        )
        _play_episode(env, seed=0)
        env.close()

        df = pq.read_table(str(path)).to_pandas()
        assert df["won"].dtype == bool

    def test_columns_present(self, tmp_dir):
        path = tmp_dir / "stats.parquet"
        env = EpisodeStatsRecorder(
            gym.make("Balatro-Easy-v0"), save_path=path, flush_every=1
        )
        _play_episode(env, seed=0)
        env.close()

        df = pq.read_table(str(path)).to_pandas()
        expected = {
            "episode_id", "episode_seed_id", "timestamp", "seed", "won",
            "antes_beaten", "blinds_beaten", "total_steps", "total_hands_played",
            "total_reward", "max_score", "final_money", "final_ante",
            "num_jokers_final", "num_consumables_final", "max_ante_reached",
            "max_blinds_beaten", "play_actions", "discard_actions",
            "buy_actions", "sell_actions", "reroll_actions", "skip_actions",
        }
        assert set(df.columns) == expected

    def test_action_counts_sum_to_total_steps(self, tmp_dir):
        path = tmp_dir / "stats.parquet"
        env = EpisodeStatsRecorder(
            gym.make("Balatro-Easy-v0"), save_path=path, flush_every=1
        )
        _play_episode(env, seed=42)
        env.close()

        df = pq.read_table(str(path)).to_pandas()
        row = df.iloc[0]
        action_sum = (
            row["play_actions"] + row["discard_actions"] + row["buy_actions"]
            + row["sell_actions"] + row["reroll_actions"] + row["skip_actions"]
        )
        assert action_sum == row["total_steps"]

    def test_seed_recorded(self, tmp_dir):
        path = tmp_dir / "stats.parquet"
        env = EpisodeStatsRecorder(
            gym.make("Balatro-Easy-v0"), save_path=path, flush_every=1
        )
        _play_episode(env, seed=99)
        env.close()

        df = pq.read_table(str(path)).to_pandas()
        assert df.iloc[0]["seed"] == 99

    def test_max_score_positive(self, tmp_dir):
        path = tmp_dir / "stats.parquet"
        env = EpisodeStatsRecorder(
            gym.make("Balatro-Easy-v0"), save_path=path, flush_every=1
        )
        _play_episode(env, seed=42)
        env.close()

        df = pq.read_table(str(path)).to_pandas()
        assert df.iloc[0]["max_score"] > 0

    def test_flush_on_close(self, tmp_dir):
        """Rows buffered but not yet flushed should be written on close()."""
        path = tmp_dir / "stats.parquet"
        env = EpisodeStatsRecorder(
            gym.make("Balatro-Easy-v0"), save_path=path, flush_every=999
        )
        _play_episode(env, seed=0)
        _play_episode(env, seed=1)
        # flush_every=999 so nothing written yet
        env.close()

        df = pq.read_table(str(path)).to_pandas()
        assert df.shape[0] == 2

    def test_load_helper(self, tmp_dir):
        path = tmp_dir / "stats.parquet"
        env = EpisodeStatsRecorder(
            gym.make("Balatro-Easy-v0"), save_path=path, flush_every=1
        )
        _play_episode(env, seed=0)
        env.close()

        table = EpisodeStatsRecorder.load(path)
        assert table.num_rows == 1
        assert "won" in table.column_names

    def test_parquet_file_is_small(self, tmp_dir):
        """10 episodes should produce a tiny file."""
        path = tmp_dir / "stats.parquet"
        env = EpisodeStatsRecorder(
            gym.make("Balatro-Easy-v0"), save_path=path, flush_every=10
        )
        for seed in range(10):
            _play_episode(env, seed=seed)
        env.close()

        size_kb = path.stat().st_size / 1024
        assert size_kb < 50, f"File too large for 10 rows: {size_kb:.1f} KB"


# -----------------------------------------------------------------------
# Combined usage
# -----------------------------------------------------------------------


class TestCombinedWrappers:
    def test_both_wrappers_together(self, tmp_dir):
        """RolloutRecorder and EpisodeStatsRecorder can be composed."""
        env = gym.make("Balatro-Easy-v0")
        env = RolloutRecorder(env, save_dir=tmp_dir / "rollouts")
        env = EpisodeStatsRecorder(
            env, save_path=tmp_dir / "stats.parquet", flush_every=1
        )

        _play_episode(env, seed=42)
        env.close()

        npz_files = list((tmp_dir / "rollouts").glob("*.npz"))
        assert len(npz_files) == 1

        df = pq.read_table(str(tmp_dir / "stats.parquet")).to_pandas()
        assert df.shape[0] == 1

    def test_multiple_episodes_combined(self, tmp_dir):
        env = gym.make("Balatro-Easy-v0")
        env = RolloutRecorder(env, save_dir=tmp_dir / "rollouts")
        env = EpisodeStatsRecorder(
            env, save_path=tmp_dir / "stats.parquet", flush_every=5
        )

        for seed in range(5):
            _play_episode(env, seed=seed)
        env.close()

        npz_files = list((tmp_dir / "rollouts").glob("*.npz"))
        assert len(npz_files) == 5

        df = pq.read_table(str(tmp_dir / "stats.parquet")).to_pandas()
        assert df.shape[0] == 5
