"""Tests for public user-facing environment construction and configuration."""

from __future__ import annotations

import numpy as np
import pytest

import balatro_gym
from balatro_gym.envs.configs import GameConfig
from balatro_gym.envs.rewards import SparseReward


class TestMake:
    def test_make_uses_difficulty_preset(self):
        env = balatro_gym.make("easy")
        assert env.config.num_antes == 4

    def test_make_custom_config_takes_priority_over_preset(self):
        config = GameConfig(num_antes=2, starting_money=13)
        env = balatro_gym.make("hard", config=config)
        assert env.config.num_antes == 2
        assert env.config.starting_money == 13

    def test_make_passes_custom_reward_function(self):
        reward_fn = SparseReward()
        env = balatro_gym.make("easy", reward_fn=reward_fn)
        assert env.reward_fn is reward_fn

    def test_make_rejects_unknown_preset(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            balatro_gym.make("unknown")

    def test_make_seed_produces_repeatable_initial_state(self):
        env1 = balatro_gym.make("easy", seed=23)
        env2 = balatro_gym.make("easy", seed=23)
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        np.testing.assert_array_equal(obs1, obs2)

    def test_explicit_reset_seed_overrides_make_seed(self):
        env = balatro_gym.make("easy", seed=23)
        obs1, _ = env.reset(seed=91)
        comparison = balatro_gym.make("easy")
        obs2, _ = comparison.reset(seed=91)
        np.testing.assert_array_equal(obs1, obs2)


class TestGameConfigFiles:
    def test_partial_yaml_is_merged_over_declared_base(self, tmp_path):
        config_file = tmp_path / "custom.yaml"
        config_file.write_text("base: easy\nnum_antes: 2\nstarting_money: 19\n")

        config = GameConfig.from_file(config_file)

        assert config.num_antes == 2
        assert config.starting_money == 19
        assert config.hands_per_round == GameConfig.easy().hands_per_round

    def test_yaml_roundtrip_preserves_configuration(self, tmp_path):
        path = tmp_path / "game.yaml"
        original = GameConfig.easy(seed=42)
        original.to_yaml(path)

        restored = GameConfig.from_file(path)

        assert restored.to_dict() == original.to_dict()

    def test_make_reads_yaml_config_and_seed_override(self, tmp_path):
        config_file = tmp_path / "custom.yaml"
        config_file.write_text("base: hard\nnum_antes: 1\n")
        env = balatro_gym.make("easy", config_path=str(config_file), seed=8)

        assert env.config.num_antes == 1
        assert env.config.hands_per_round == GameConfig.hard().hands_per_round
        assert env.config.seed == 8


class TestMakeVec:
    def test_sync_vector_environment_steps_legal_actions(self):
        env = balatro_gym.make_vec("easy", num_envs=2, seed=9, vectorization_mode="sync")
        obs, info = env.reset()
        actions = np.array([int(mask.nonzero()[0][0]) for mask in info["action_mask"]])

        next_obs, rewards, terminated, truncated, next_info = env.step(actions)

        assert obs.shape[0] == 2
        assert next_obs.shape == obs.shape
        assert rewards.shape == (2,)
        assert not np.any(terminated)
        assert not np.any(truncated)
        env.close()

    def test_async_vector_environment_resets_and_steps_legal_actions(self):
        env = balatro_gym.make_vec("easy", num_envs=2, seed=9, vectorization_mode="async")
        obs, info = env.reset()
        actions = np.array([int(mask.nonzero()[0][0]) for mask in info["action_mask"]])

        next_obs, rewards, terminated, truncated, next_info = env.step(actions)

        assert obs.shape[0] == 2
        assert next_obs.shape == obs.shape
        assert rewards.shape == (2,)
        env.close()

    def test_rejects_unknown_vectorization_mode(self):
        with pytest.raises(ValueError, match="Unknown vectorization_mode"):
            balatro_gym.make_vec("easy", vectorization_mode="incorrect")
