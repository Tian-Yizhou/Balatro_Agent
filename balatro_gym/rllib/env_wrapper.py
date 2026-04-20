"""RLlib-compatible environment wrapper for Balatro.

Wraps :class:`BalatroEnv` so that the observation is a ``gymnasium.spaces.Dict``
containing ``"observations"`` (the flat vector) and ``"action_mask"`` (boolean
mask over the discrete action space).  This is the format expected by the
:class:`ActionMaskingTorchRLModule` shipped alongside this package.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from balatro_gym.envs.balatro_env import BalatroEnv, TOTAL_ACTIONS
from balatro_gym.envs.configs import GameConfig


class BalatroRLlibEnv(gym.Wrapper):
    """Gymnasium wrapper that re-packages observations for RLlib action masking.

    The wrapped observation space becomes::

        Dict({
            "action_mask": Box(0.0, 1.0, shape=(TOTAL_ACTIONS,), dtype=float32),
            "observations": Box(low, high, shape=(obs_dim,), dtype=float32),
        })

    This wrapper can be registered directly with RLlib via ``register_env`` or
    by passing the class to ``PPOConfig().environment(env=...)``.

    Args:
        env: A :class:`BalatroEnv` instance (or any ``gym.Env`` with an
            ``action_masks()`` method).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        inner_obs_space = self.env.observation_space
        self.observation_space = spaces.Dict(
            {
                "action_mask": spaces.Box(
                    0.0, 1.0, shape=(TOTAL_ACTIONS,), dtype=np.float32
                ),
                "observations": inner_obs_space,
            }
        )
        # Action space is unchanged (Discrete(446))

    # ------------------------------------------------------------------
    # Gymnasium API overrides
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        mask = info.get("action_mask", self.env.action_masks())
        wrapped_obs = {
            "action_mask": mask.astype(np.float32),
            "observations": obs,
        }
        return wrapped_obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        mask = info.get("action_mask", self.env.action_masks())
        wrapped_obs = {
            "action_mask": mask.astype(np.float32),
            "observations": obs,
        }
        return wrapped_obs, reward, terminated, truncated, info


# ------------------------------------------------------------------
# Factory helpers for RLlib ``register_env``
# ------------------------------------------------------------------


def make_balatro_env(config: dict[str, Any]) -> BalatroRLlibEnv:
    """Environment creator function for :func:`ray.tune.register_env`.

    ``config`` is the dict passed via ``PPOConfig().environment(env_config=...)``.

    Supported keys:

    * ``difficulty`` (str): ``"easy"``, ``"medium"``, or ``"hard"``.
      Default ``"medium"``.
    * ``game_config`` (:class:`GameConfig`): A pre-built config object.
      Takes priority over ``difficulty`` if provided.
    * ``seed`` (int | None): Game seed.  Default ``None``.

    Example::

        from ray.tune.registry import register_env
        from balatro_gym.rllib import make_balatro_env

        register_env("Balatro", make_balatro_env)
    """
    game_config = config.get("game_config")
    if game_config is None:
        difficulty = config.get("difficulty", "medium")
        seed = config.get("seed")
        factory = {
            "easy": GameConfig.easy,
            "medium": GameConfig.medium,
            "hard": GameConfig.hard,
        }
        if difficulty not in factory:
            raise ValueError(
                f"Unknown difficulty {difficulty!r}. "
                f"Choose from: {list(factory)}"
            )
        game_config = factory[difficulty](seed=seed)

    inner_env = BalatroEnv(config=game_config)
    return BalatroRLlibEnv(inner_env)
