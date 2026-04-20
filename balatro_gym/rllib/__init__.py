"""Ray RLlib integration for the Balatro gymnasium environment.

Key exports:

* :class:`BalatroRLlibEnv` — Wrapper that packages observations as a
  ``Dict(action_mask=..., observations=...)`` for action masking.
* :func:`make_balatro_env` — Factory for :func:`ray.tune.register_env`.
* :class:`ActionMaskingTorchRLModule` — PPO RL module with action masking.
* :func:`build_config` — Build a :class:`PPOConfig` from parsed CLI args.
"""

from balatro_gym.rllib.env_wrapper import BalatroRLlibEnv, make_balatro_env
from balatro_gym.rllib.action_mask_model import ActionMaskingTorchRLModule
from balatro_gym.rllib.train import build_config
