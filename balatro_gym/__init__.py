"""Balatro Gym: A Gymnasium-compatible card game environment inspired by Balatro.

Quick start::

    import balatro_gym

    # Single environment
    env = balatro_gym.make("easy")
    obs, info = env.reset()

    # Parallel environments for data collection
    vec_env = balatro_gym.make_vec("easy", num_envs=8)
    obs, infos = vec_env.reset()
"""

from __future__ import annotations

__version__ = "0.1.0"

# Register Gymnasium environments on import
import balatro_gym.envs  # noqa: F401

from balatro_gym.envs.balatro_env import BalatroEnv
from balatro_gym.envs.configs import GameConfig
from balatro_gym.envs.rewards import (  # noqa: F401
    DefaultReward,
    RewardContext,
    RewardFunction,
    SparseReward,
)


def make(
    preset: str = "medium",
    *,
    config: GameConfig | None = None,
    config_path: str | None = None,
    seed: int | None = None,
    render_mode: str | None = None,
    reward_fn: RewardFunction | None = None,
) -> BalatroEnv:
    """Create a single Balatro environment.

    This is the recommended entry point. Accepts a difficulty preset name,
    a :class:`GameConfig` object, or a path to a YAML config file.

    Priority: *config* > *config_path* > *preset*.

    Args:
        preset: Difficulty preset (``"easy"``, ``"medium"``, ``"hard"``).
        config: A pre-built :class:`GameConfig`. Takes priority over
            *config_path* and *preset*.
        config_path: Path to a YAML config file (merged over *preset*).
        seed: Game seed. Overrides seed in config if provided.
        render_mode: Gymnasium render mode (``"human"`` or ``"ansi"``).
        reward_fn: Custom reward function. Any callable with signature
            ``(RewardContext) -> float``. Defaults to :class:`DefaultReward`.

    Returns:
        A :class:`BalatroEnv` instance ready for ``reset()`` / ``step()``.

    Examples::

        # Preset
        env = balatro_gym.make("easy")

        # From YAML
        env = balatro_gym.make(config_path="configs/example_custom.yaml")

        # Sparse reward
        from balatro_gym.envs.rewards import SparseReward
        env = balatro_gym.make("easy", reward_fn=SparseReward())

        # Custom lambda
        env = balatro_gym.make("easy", reward_fn=lambda ctx: ctx.won * 100)
    """
    if config is not None:
        if seed is not None:
            config = GameConfig(**{**config.to_dict(), "seed": seed})
    elif config_path is not None:
        config = GameConfig.from_file(config_path, base=preset)
        if seed is not None:
            config = GameConfig(**{**config.to_dict(), "seed": seed})
    else:
        factory = {"easy": GameConfig.easy, "medium": GameConfig.medium, "hard": GameConfig.hard}
        if preset not in factory:
            raise ValueError(
                f"Unknown preset {preset!r}. Choose from: {list(factory)}"
            )
        config = factory[preset](seed=seed)

    return BalatroEnv(config=config, render_mode=render_mode, reward_fn=reward_fn)


def make_vec(
    preset: str = "medium",
    *,
    num_envs: int = 4,
    config: GameConfig | None = None,
    config_path: str | None = None,
    seed: int | None = None,
    vectorization_mode: str = "async",
    reward_fn: RewardFunction | None = None,
) -> "gymnasium.vector.VectorEnv":
    """Create a vectorized Balatro environment for parallel data collection.

    Wraps multiple :class:`BalatroEnv` instances using Gymnasium's
    vector API for efficient parallel stepping.

    Args:
        preset: Difficulty preset (``"easy"``, ``"medium"``, ``"hard"``).
        num_envs: Number of parallel environments.
        config: A pre-built :class:`GameConfig`.
        config_path: Path to a YAML config file.
        seed: Base seed. Each sub-env gets ``seed + i`` for reproducibility.
            If ``None``, sub-envs are randomly seeded.
        vectorization_mode: ``"async"`` (default, multiprocess) or ``"sync"``
            (single process, useful for debugging).
        reward_fn: Custom reward function applied to each sub-env.

    Returns:
        A :class:`gymnasium.vector.VectorEnv` with *num_envs* sub-environments.

    Examples::

        # 8 parallel easy envs
        vec_env = balatro_gym.make_vec("easy", num_envs=8)
        obs, infos = vec_env.reset()
        # obs is a batch: shape (8, obs_dim)

        # Reproducible seeds
        vec_env = balatro_gym.make_vec("medium", num_envs=4, seed=42)

        # Sync mode for debugging
        vec_env = balatro_gym.make_vec("easy", num_envs=2, vectorization_mode="sync")
    """
    import gymnasium

    def _make_env(env_seed: int | None = None):
        def _thunk():
            return make(preset, config=config, config_path=config_path,
                       seed=env_seed, reward_fn=reward_fn)
        return _thunk

    env_fns = []
    for i in range(num_envs):
        env_seed = (seed + i) if seed is not None else None
        env_fns.append(_make_env(env_seed))

    if vectorization_mode == "async":
        return gymnasium.vector.AsyncVectorEnv(env_fns)
    elif vectorization_mode == "sync":
        return gymnasium.vector.SyncVectorEnv(env_fns)
    else:
        raise ValueError(
            f"Unknown vectorization_mode {vectorization_mode!r}. "
            f"Choose 'async' or 'sync'."
        )
