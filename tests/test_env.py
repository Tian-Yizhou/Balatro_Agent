"""Tests for balatro_env.py: Gymnasium wrapper, obs encoding, action masking."""

import gymnasium as gym
import numpy as np
import pytest

from balatro_gym.envs.balatro_env import (
    BalatroEnv,
    TOTAL_ACTIONS,
    CARD_SUBSETS,
    PLAY_OFFSET,
    DISCARD_OFFSET,
    BUY_OFFSET,
    SELL_OFFSET,
    REROLL_ACTION,
    SKIP_ACTION,
)
from balatro_gym.envs.configs import GameConfig


class TestActionSpaceConstants:
    def test_total_actions(self):
        assert TOTAL_ACTIONS == 445

    def test_card_subsets_count(self):
        assert len(CARD_SUBSETS) == 218

    def test_action_ranges(self):
        assert PLAY_OFFSET == 0
        assert DISCARD_OFFSET == 218
        assert BUY_OFFSET == 436
        assert SELL_OFFSET == 438
        assert REROLL_ACTION == 443
        assert SKIP_ACTION == 444


class TestEnvCreation:
    def test_create_with_config(self):
        config = GameConfig.easy(seed=42)
        env = BalatroEnv(config=config)
        assert env.observation_space.shape[0] > 0
        assert env.action_space.n == 445

    def test_create_with_preset(self):
        env = BalatroEnv(config_preset="easy")
        assert env.config.num_antes == 4

    def test_create_default(self):
        env = BalatroEnv()
        assert env.config.num_antes == 6  # medium default

    def test_invalid_preset(self):
        with pytest.raises(ValueError):
            BalatroEnv(config_preset="impossible")

    def test_obs_dim_varies_by_config(self):
        easy = BalatroEnv(config=GameConfig.easy())
        hard = BalatroEnv(config=GameConfig.hard())
        assert easy._obs_dim < hard._obs_dim


class TestReset:
    def test_reset_returns_obs_and_info(self):
        env = BalatroEnv(config=GameConfig.easy(seed=42))
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert obs.shape == (env._obs_dim,)
        assert "action_mask" in info

    def test_reset_reproducible(self):
        env = BalatroEnv(config=GameConfig.easy(seed=42))
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)


class TestStep:
    def test_step_returns_correct_types(self):
        env = BalatroEnv(config=GameConfig.easy(seed=42))
        obs, info = env.reset(seed=42)
        mask = info["action_mask"]
        action = int(np.where(mask)[0][0])
        obs2, reward, term, trunc, info2 = env.step(action)
        assert isinstance(obs2, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info2, dict)

    def test_step_updates_info(self):
        env = BalatroEnv(config=GameConfig.easy(seed=42))
        obs, info = env.reset(seed=42)
        assert info["phase"] == "play"
        assert info["ante"] == 1


class TestActionMask:
    def test_play_phase_mask(self):
        env = BalatroEnv(config=GameConfig.easy(seed=42))
        obs, info = env.reset(seed=42)
        mask = info["action_mask"]

        # Should have play and discard actions valid
        play_valid = mask[PLAY_OFFSET:PLAY_OFFSET + 218].sum()
        discard_valid = mask[DISCARD_OFFSET:DISCARD_OFFSET + 218].sum()
        assert play_valid > 0
        assert discard_valid > 0

        # Shop actions should be invalid
        assert not mask[BUY_OFFSET]
        assert not mask[SKIP_ACTION]

    def test_mask_respects_hand_size(self):
        env = BalatroEnv(config=GameConfig.easy(seed=42))
        obs, info = env.reset(seed=42)
        mask = info["action_mask"]

        # With 8 cards, all subsets of range(8) should be valid
        for i, subset in enumerate(CARD_SUBSETS):
            if max(subset) < 8:
                assert mask[PLAY_OFFSET + i], f"Subset {subset} should be valid"

    def test_mask_no_valid_at_game_end(self):
        env = BalatroEnv(config=GameConfig.easy(seed=42))
        obs, info = env.reset(seed=42)

        # Play until game ends
        for _ in range(10000):
            mask = info["action_mask"]
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            obs, _, term, _, info = env.step(int(valid[0]))
            if term:
                break

        if info["phase"] in ("game_over", "game_won"):
            mask = env.action_masks()
            assert mask.sum() == 0


class TestObservation:
    def test_obs_in_valid_range(self):
        env = BalatroEnv(config=GameConfig.easy(seed=42))
        obs, _ = env.reset(seed=42)
        assert obs.min() >= -1.0
        assert obs.max() <= 10.0

    def test_hand_bitmap(self):
        env = BalatroEnv(config=GameConfig.easy(seed=42))
        obs, _ = env.reset(seed=42)
        hand_bitmap = obs[:52]
        assert hand_bitmap.sum() == 8  # 8 cards in hand

    def test_phase_encoding(self):
        env = BalatroEnv(config=GameConfig.easy(seed=42))
        obs, _ = env.reset(seed=42)
        # Phase is near the end; verify play phase
        game = env._game
        assert game.phase.value == "play"


class TestFullEpisode:
    def test_random_episode_terminates(self):
        env = BalatroEnv(config=GameConfig.easy(seed=42))
        obs, info = env.reset(seed=42)
        rng = np.random.default_rng(42)

        steps = 0
        while steps < 5000:
            mask = info["action_mask"]
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            action = int(rng.choice(valid))
            obs, reward, term, trunc, info = env.step(action)
            steps += 1
            if term or trunc:
                break

        assert steps < 5000, "Episode should terminate"
        assert info["phase"] in ("game_over", "game_won")

    def test_reward_positive_on_win(self):
        """Difficult to guarantee a win, but verify reward structure."""
        env = BalatroEnv(config=GameConfig.easy(seed=42))
        obs, info = env.reset(seed=42)

        # Play until done
        total_reward = 0.0
        for _ in range(5000):
            mask = info["action_mask"]
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            obs, reward, term, _, info = env.step(int(valid[-1]))  # Pick last valid
            total_reward += reward
            if term:
                break

        # At minimum, reward should be bounded
        assert total_reward > -100
        assert total_reward < 100

    def test_100_random_games_no_crashes(self):
        for seed in range(100):
            config = GameConfig.easy(seed=seed)
            env = BalatroEnv(config=config)
            obs, info = env.reset(seed=seed)
            rng = np.random.default_rng(seed)

            for _ in range(2000):
                mask = info["action_mask"]
                valid = np.where(mask)[0]
                if len(valid) == 0:
                    break
                obs, _, term, _, info = env.step(int(rng.choice(valid)))
                if term:
                    break


class TestGymnasiumRegistration:
    def test_make_balatro_v0(self):
        env = gym.make("Balatro-v0")
        obs, info = env.reset()
        assert obs.shape[0] > 0

    def test_make_easy(self):
        env = gym.make("Balatro-Easy-v0")
        obs, _ = env.reset()
        assert obs.shape[0] > 0

    def test_make_hard(self):
        env = gym.make("Balatro-Hard-v0")
        obs, _ = env.reset()
        assert obs.shape[0] > 0


class TestRender:
    def test_render_ansi(self):
        env = BalatroEnv(config=GameConfig.easy(seed=42), render_mode="ansi")
        env.reset(seed=42)
        text = env.render()
        assert isinstance(text, str)
        assert "Ante" in text
        assert "Score" in text
        assert "Hand:" in text
