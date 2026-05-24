"""Custom RLlib callbacks for Balatro game-specific metrics.

Captures game-specific statistics (antes beaten, money, score, etc.)
at episode end and reports them as custom metrics in the training results.
"""

from __future__ import annotations

from typing import Any

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.single_agent_episode import SingleAgentEpisode


class BalatroMetricsCallback(DefaultCallbacks):
    """Captures Balatro-specific metrics at episode end.

    Reported metrics (accessible in result["env_runners"]["metrics"]):
        - game_won: 1.0 if won, 0.0 if lost
        - blinds_beaten: number of blinds beaten
        - ante_reached: highest ante reached
        - final_money: money at end of game
        - final_score: score at end of last blind attempt
        - episode_length: number of steps in the episode
    """

    def on_episode_end(
        self,
        *,
        episode: SingleAgentEpisode,
        env_runner=None,
        metrics_logger=None,
        env=None,
        rl_module=None,
        env_index: int = 0,
        **kwargs: Any,
    ) -> None:
        """Extract game-specific metrics from the final info dict."""
        # Get the last info from the episode
        info = episode.get_infos(-1)
        if info is None:
            return

        # Extract game metrics
        won = 1.0 if info.get("phase") == "game_won" else 0.0
        blinds_beaten = info.get("blinds_beaten", 0)
        ante = info.get("ante", 0)
        money = info.get("money", 0)
        score = info.get("score", 0)

        # Log via metrics_logger (RLlib new API stack)
        if metrics_logger is not None:
            metrics_logger.log_value("game_won", won)
            metrics_logger.log_value("blinds_beaten", blinds_beaten)
            metrics_logger.log_value("ante_reached", ante)
            metrics_logger.log_value("final_money", money)
            metrics_logger.log_value("final_score", score)
            metrics_logger.log_value("episode_length", episode.env_steps())
