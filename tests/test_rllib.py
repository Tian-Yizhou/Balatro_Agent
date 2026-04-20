"""Tests for the Ray RLlib integration (env wrapper, action masking, config)."""

from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

import balatro_gym  # noqa: F401 — triggers env registration
from balatro_gym.envs.balatro_env import BalatroEnv, TOTAL_ACTIONS
from balatro_gym.envs.configs import GameConfig
from balatro_gym.rllib.env_wrapper import BalatroRLlibEnv, make_balatro_env


# -----------------------------------------------------------------------
# BalatroRLlibEnv wrapper
# -----------------------------------------------------------------------


class TestBalatroRLlibEnv:
    def _make_env(self, difficulty="easy", seed=42):
        config = getattr(GameConfig, difficulty)(seed=seed)
        inner = BalatroEnv(config=config)
        return BalatroRLlibEnv(inner)

    def test_observation_space_is_dict(self):
        env = self._make_env()
        assert isinstance(env.observation_space, spaces.Dict)
        assert "action_mask" in env.observation_space.spaces
        assert "observations" in env.observation_space.spaces

    def test_action_mask_shape(self):
        env = self._make_env()
        mask_space = env.observation_space["action_mask"]
        assert mask_space.shape == (TOTAL_ACTIONS,)
        assert mask_space.dtype == np.float32

    def test_observations_space_matches_inner(self):
        config = GameConfig.easy(seed=42)
        inner = BalatroEnv(config=config)
        env = BalatroRLlibEnv(inner)
        assert env.observation_space["observations"].shape == inner.observation_space.shape

    def test_action_space_unchanged(self):
        env = self._make_env()
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == TOTAL_ACTIONS

    def test_reset_returns_dict_obs(self):
        env = self._make_env()
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert "action_mask" in obs
        assert "observations" in obs
        assert obs["action_mask"].dtype == np.float32
        assert obs["observations"].dtype == np.float32

    def test_reset_mask_has_valid_actions(self):
        env = self._make_env()
        obs, info = env.reset(seed=42)
        mask = obs["action_mask"]
        assert mask.sum() > 0, "At least one action should be valid after reset"

    def test_step_returns_dict_obs(self):
        env = self._make_env()
        obs, info = env.reset(seed=42)
        mask = obs["action_mask"]
        valid = np.where(mask > 0.5)[0]
        assert len(valid) > 0

        obs2, reward, term, trunc, info2 = env.step(int(valid[0]))
        assert isinstance(obs2, dict)
        assert "action_mask" in obs2
        assert "observations" in obs2

    def test_obs_in_space(self):
        env = self._make_env()
        obs, info = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_mask_values_binary(self):
        """Action mask values should be 0.0 or 1.0."""
        env = self._make_env()
        obs, _ = env.reset(seed=42)
        mask = obs["action_mask"]
        assert np.all((mask == 0.0) | (mask == 1.0))

    def test_play_full_episode(self):
        """Play a full episode using only valid actions from the mask."""
        env = self._make_env(seed=0)
        obs, info = env.reset(seed=0)
        rng = np.random.default_rng(0)

        for _ in range(2000):
            mask = obs["action_mask"]
            valid = np.where(mask > 0.5)[0]
            if len(valid) == 0:
                break
            action = int(rng.choice(valid))
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break
        # Should have terminated (won or lost)
        assert term or trunc or len(valid) == 0


# -----------------------------------------------------------------------
# make_balatro_env factory
# -----------------------------------------------------------------------


class TestMakeBalatroEnv:
    def test_default_config(self):
        env = make_balatro_env({})
        assert isinstance(env, BalatroRLlibEnv)
        obs, _ = env.reset()
        assert "action_mask" in obs

    def test_easy_difficulty(self):
        env = make_balatro_env({"difficulty": "easy"})
        obs, _ = env.reset(seed=42)
        assert obs["observations"].shape[0] > 0

    def test_medium_difficulty(self):
        env = make_balatro_env({"difficulty": "medium"})
        obs, _ = env.reset(seed=42)
        assert obs["observations"].shape[0] > 0

    def test_hard_difficulty(self):
        env = make_balatro_env({"difficulty": "hard"})
        obs, _ = env.reset(seed=42)
        assert obs["observations"].shape[0] > 0

    def test_invalid_difficulty_raises(self):
        with pytest.raises(ValueError, match="Unknown difficulty"):
            make_balatro_env({"difficulty": "impossible"})

    def test_custom_game_config(self):
        gc = GameConfig.easy(seed=99)
        env = make_balatro_env({"game_config": gc})
        obs, info = env.reset(seed=99)
        assert "action_mask" in obs

    def test_seed_passed_through(self):
        env = make_balatro_env({"difficulty": "easy", "seed": 123})
        obs, _ = env.reset(seed=123)
        assert obs["observations"].shape[0] > 0


# -----------------------------------------------------------------------
# ActionMaskingTorchRLModule (import + instantiation)
# -----------------------------------------------------------------------


class TestActionMaskingModule:
    def test_import(self):
        """Module class should be importable."""
        from balatro_gym.rllib.action_mask_model import ActionMaskingTorchRLModule
        assert ActionMaskingTorchRLModule is not None

    def test_requires_dict_observation_space(self):
        """Should raise ValueError if given a non-Dict observation space."""
        from balatro_gym.rllib.action_mask_model import _ActionMaskingBase

        with pytest.raises(ValueError, match="Dict"):
            _ActionMaskingBase(
                observation_space=spaces.Box(0, 1, shape=(10,)),
                action_space=spaces.Discrete(5),
            )


# -----------------------------------------------------------------------
# train.py config builder
# -----------------------------------------------------------------------


class TestTrainConfig:
    def test_parse_args_defaults(self):
        from balatro_gym.rllib.train import parse_args
        args = parse_args([])
        assert args.difficulty == "easy"
        assert args.num_env_runners == 2
        assert args.num_gpus_per_learner == 0
        assert args.lr == 3e-4
        assert args.fcnet_hiddens == [256, 256]

    def test_parse_args_custom(self):
        from balatro_gym.rllib.train import parse_args
        args = parse_args([
            "--difficulty", "hard",
            "--num-env-runners", "8",
            "--num-gpus-per-learner", "1",
            "--lr", "1e-4",
            "--fcnet-hiddens", "512", "256",
            "--num-iterations", "100",
        ])
        assert args.difficulty == "hard"
        assert args.num_env_runners == 8
        assert args.num_gpus_per_learner == 1.0
        assert args.lr == 1e-4
        assert args.fcnet_hiddens == [512, 256]
        assert args.num_iterations == 100

    def test_build_config(self):
        from balatro_gym.rllib.train import parse_args, build_config
        args = parse_args(["--num-env-runners", "0"])
        config = build_config(args)
        # Should be a valid PPOConfig
        from ray.rllib.algorithms.ppo import PPOConfig
        assert isinstance(config, PPOConfig)


# -----------------------------------------------------------------------
# evaluate.py arg parser
# -----------------------------------------------------------------------


class TestEvaluateConfig:
    def test_parse_args(self):
        from balatro_gym.rllib.evaluate import parse_args
        args = parse_args([
            "--checkpoint", "/tmp/test_ckpt",
            "--num-episodes", "50",
            "--difficulty", "medium",
        ])
        assert args.checkpoint == "/tmp/test_ckpt"
        assert args.num_episodes == 50
        assert args.difficulty == "medium"


# -----------------------------------------------------------------------
# Smoke test: RLlib can build an algorithm with this env
# -----------------------------------------------------------------------


class TestRLlibSmoke:
    @pytest.mark.slow
    def test_build_algo_and_sample(self):
        """Build a PPO algorithm and run one training step (smoke test).

        Marked slow because it starts Ray and builds a full algorithm.
        Skip in CI with: pytest -m 'not slow'
        """
        import ray
        from ray.tune.registry import register_env
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        from balatro_gym.rllib.action_mask_model import ActionMaskingTorchRLModule

        ray.init(num_cpus=2, num_gpus=0, ignore_reinit_error=True)
        try:
            register_env("BalatroTest", make_balatro_env)

            config = (
                PPOConfig()
                .environment(
                    env="BalatroTest",
                    env_config={"difficulty": "easy"},
                )
                .env_runners(
                    num_env_runners=0,  # local only for testing
                )
                .learners(
                    num_gpus_per_learner=0,
                )
                .training(
                    train_batch_size_per_learner=200,
                    minibatch_size=64,
                    num_epochs=1,
                )
                .rl_module(
                    rl_module_spec=RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        model_config={
                            "head_fcnet_hiddens": [32, 32],
                            "head_fcnet_activation": "relu",
                        },
                    ),
                )
            )

            algo = config.build()
            result = algo.train()

            # Should have sampled some steps
            assert result["env_runners"]["num_env_steps_sampled_lifetime"] > 0
            algo.stop()
        finally:
            ray.shutdown()
