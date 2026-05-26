"""Tests for reward functions and reward delivery through environment steps."""

from __future__ import annotations

import pytest

from balatro_gym.envs.balatro_env import BalatroEnv, SKIP_ACTION
from balatro_gym.envs.configs import GameConfig
from balatro_gym.envs.rewards import DefaultReward, RewardContext, SparseReward


def _context(**overrides) -> RewardContext:
    values = {
        "phase": "play",
        "action_valid": True,
        "prev_score": 0,
        "new_score": 0,
        "score_target": 100,
        "prev_blinds_beaten": 0,
        "new_blinds_beaten": 0,
        "total_blinds": 12,
        "blind_just_beaten": False,
        "won": False,
        "lost": False,
        "ante": 1,
        "money": 4,
        "hands_remaining": 4,
        "discards_remaining": 3,
    }
    values.update(overrides)
    return RewardContext(**values)


class TestDefaultReward:
    def test_invalid_action_is_penalized(self):
        reward = DefaultReward()(_context(action_valid=False))
        assert reward == pytest.approx(-0.01)

    def test_play_score_progress_is_rewarded_and_clipped(self):
        reward = DefaultReward()(_context(new_score=1000))
        assert reward == pytest.approx(0.01)

    def test_shop_score_change_does_not_receive_play_progress_reward(self):
        reward = DefaultReward()(_context(phase="shop", new_score=100))
        assert reward == 0.0

    def test_blind_and_win_rewards_are_accumulated(self):
        reward = DefaultReward()(
            _context(
                phase="game_won",
                new_blinds_beaten=12,
                blind_just_beaten=True,
                won=True,
            )
        )
        assert reward == pytest.approx(12.0)

    def test_loss_penalty_is_emitted(self):
        assert DefaultReward()(_context(phase="game_over", lost=True)) == -1.0


class TestSparseReward:
    def test_ignores_non_terminal_actions(self):
        assert SparseReward()(_context(action_valid=False, new_score=100)) == 0.0

    def test_emits_terminal_values(self):
        reward_fn = SparseReward(win_reward=3.0, lose_penalty=-2.0)
        assert reward_fn(_context(won=True)) == 3.0
        assert reward_fn(_context(lost=True)) == -2.0


class TestRewardEnvironmentBoundary:
    def test_custom_reward_receives_invalid_user_action_context(self):
        contexts: list[RewardContext] = []

        def reward_fn(ctx: RewardContext) -> float:
            contexts.append(ctx)
            return 7.0 if not ctx.action_valid else 0.0

        env = BalatroEnv(config=GameConfig.easy(seed=1), reward_fn=reward_fn)
        _, info = env.reset(seed=1)
        assert not info["action_mask"][SKIP_ACTION]

        _, reward, _, _, _ = env.step(SKIP_ACTION)

        assert reward == 7.0
        assert contexts[-1].action_valid is False
        assert contexts[-1].phase == "play"

    def test_sparse_reward_integrates_with_non_terminal_environment_action(self):
        env = BalatroEnv(config=GameConfig.medium(seed=3), reward_fn=SparseReward())
        _, info = env.reset(seed=3)
        action = int(info["action_mask"].nonzero()[0][0])

        _, reward, terminated, _, _ = env.step(action)

        assert not terminated
        assert reward == 0.0
