"""Random agent baseline: picks uniformly from valid actions."""

from __future__ import annotations

import numpy as np

from balatro_gym.envs.balatro_env import BalatroEnv, TOTAL_ACTIONS


class RandomAgent:
    """Picks a random valid action each step."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, info: dict) -> int:
        mask = info["action_mask"]
        valid = np.where(mask)[0]
        if len(valid) == 0:
            return 0
        return int(self.rng.choice(valid))

    def run_episode(self, env: BalatroEnv) -> dict:
        """Run a full episode and return stats."""
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            action = self.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        return {
            "total_reward": total_reward,
            "steps": steps,
            "blinds_beaten": info["blinds_beaten"],
            "phase": info["phase"],
            "won": info["phase"] == "game_won",
        }
