"""Base Agent protocol for Balatro environments.

Any agent that interacts with a :class:`BalatroEnv` should implement this
protocol. This ensures all agents — random, heuristic, RL, LLM-based —
share a consistent interface for evaluation and comparison.

Example::

    from agent.base import Agent
    import balatro_gym

    class MyAgent:
        def act(self, obs, info):
            mask = info["action_mask"]
            valid = mask.nonzero()[0]
            return int(valid[0])  # just pick the first valid action

        def reset(self):
            pass

    env = balatro_gym.make("easy")
    agent = MyAgent()
    agent.reset()
    obs, info = env.reset()
    action = agent.act(obs, info)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Agent(Protocol):
    """Protocol for agents that interact with a Balatro environment.

    Any object with ``act(obs, info) -> int`` and ``reset() -> None``
    satisfies this protocol.
    """

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        """Choose an action given the current observation and info dict.

        Args:
            obs: The observation array from the environment.
            info: The info dict from the environment, containing at least
                ``"action_mask"`` (bool array of valid actions).

        Returns:
            An integer action index (0 to TOTAL_ACTIONS-1).
        """
        ...

    def reset(self) -> None:
        """Reset any internal state at the start of a new episode."""
        ...


def run_episode(
    env,
    agent: Agent,
    *,
    seed: int | None = None,
    max_steps: int = 5000,
) -> dict[str, Any]:
    """Run a single episode with the given agent.

    Args:
        env: A Gymnasium environment (BalatroEnv or wrapped).
        agent: An agent implementing the :class:`Agent` protocol.
        seed: Optional seed for env.reset().
        max_steps: Safety limit on episode length.

    Returns:
        A dict with episode statistics::

            {
                "total_reward": float,
                "steps": int,
                "won": bool,
                "ante": int,
                "blinds_beaten": int,
                "money": int,
                "score": int,
                "phase": str,
            }
    """
    agent.reset()
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0

    for _ in range(max_steps):
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "steps": steps,
        "won": info.get("phase") == "game_won",
        "ante": info.get("ante", 0),
        "blinds_beaten": info.get("blinds_beaten", 0),
        "money": info.get("money", 0),
        "score": info.get("score", 0),
        "phase": info.get("phase", "unknown"),
    }


def evaluate_agent(
    env,
    agent: Agent,
    *,
    num_episodes: int = 100,
    seed: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Evaluate an agent over multiple episodes.

    Args:
        env: A Gymnasium environment.
        agent: An agent implementing the :class:`Agent` protocol.
        num_episodes: Number of episodes to run.
        seed: Base seed (each episode gets seed+i). None = random.
        verbose: Print per-episode results.

    Returns:
        Aggregate statistics dict::

            {
                "num_episodes": int,
                "win_rate": float,
                "mean_reward": float,
                "std_reward": float,
                "mean_steps": float,
                "mean_ante": float,
                "mean_blinds_beaten": float,
                "mean_money": float,
                "max_score": int,
                "episodes": list[dict],  # per-episode results
            }
    """
    episodes = []

    for i in range(num_episodes):
        ep_seed = (seed + i) if seed is not None else None
        result = run_episode(env, agent, seed=ep_seed)
        episodes.append(result)

        if verbose:
            status = "WON" if result["won"] else "LOST"
            print(
                f"Episode {i + 1:4d}: {status} | "
                f"reward={result['total_reward']:7.2f} | "
                f"steps={result['steps']:4d} | "
                f"ante={result['ante']} | "
                f"blinds={result['blinds_beaten']}"
            )

    rewards = [e["total_reward"] for e in episodes]
    return {
        "num_episodes": num_episodes,
        "win_rate": sum(1 for e in episodes if e["won"]) / num_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean([e["steps"] for e in episodes])),
        "mean_ante": float(np.mean([e["ante"] for e in episodes])),
        "mean_blinds_beaten": float(np.mean([e["blinds_beaten"] for e in episodes])),
        "mean_money": float(np.mean([e["money"] for e in episodes])),
        "max_score": max(e["score"] for e in episodes),
        "episodes": episodes,
    }
