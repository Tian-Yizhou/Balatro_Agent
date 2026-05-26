"""Tests for public agent helpers and structured user/LLM state rendering."""

from __future__ import annotations

import json

import numpy as np

import balatro_gym
from agent.base import Agent, evaluate_agent, run_episode
from agent.baselines import HeuristicAgent, RandomAgent
from agent.llm.renderer import GameStateRenderer
from balatro_gym.envs.configs import GameConfig


class FirstValidAgent:
    def __init__(self):
        self.resets = 0
        self.actions = 0

    def reset(self) -> None:
        self.resets += 1

    def act(self, obs: np.ndarray, info: dict) -> int:
        self.actions += 1
        return int(info["action_mask"].nonzero()[0][0])


class TestAgentProtocolAndHelpers:
    def test_baseline_agent_satisfies_public_protocol(self):
        assert isinstance(RandomAgent(seed=1), Agent)

    def test_run_episode_resets_agent_and_returns_environment_stats(self):
        env = balatro_gym.make(config=GameConfig(num_antes=1, hands_per_round=1), seed=4)
        agent = FirstValidAgent()

        result = run_episode(env, agent, seed=4)

        assert agent.resets == 1
        assert result["steps"] >= 1
        assert result["phase"] in ("game_over", "game_won")
        assert isinstance(result["total_reward"], float)

    def test_evaluate_agent_runs_requested_seeded_episodes(self):
        env = balatro_gym.make(config=GameConfig(num_antes=1, hands_per_round=1))
        agent = FirstValidAgent()

        result = evaluate_agent(env, agent, num_episodes=3, seed=10)

        assert agent.resets == 3
        assert result["num_episodes"] == 3
        assert len(result["episodes"]) == 3
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_random_baseline_run_episode_returns_terminal_statistics(self):
        env = balatro_gym.make(config=GameConfig(num_antes=1, hands_per_round=1), seed=2)

        result = RandomAgent(seed=2).run_episode(env)

        assert result["phase"] in ("game_over", "game_won")
        assert result["steps"] >= 1

    def test_heuristic_baseline_run_episode_returns_terminal_statistics(self):
        env = balatro_gym.make(config=GameConfig(num_antes=1, hands_per_round=1), seed=2)

        result = HeuristicAgent(seed=2).run_episode(env)

        assert result["phase"] in ("game_over", "game_won")
        assert result["steps"] >= 1


class TestGameStateRenderer:
    def test_uninitialized_environment_reports_error(self):
        renderer = GameStateRenderer(balatro_gym.make("easy"))
        assert "error" in renderer.render()

    def test_render_json_contains_user_visible_state_and_actions(self):
        env = balatro_gym.make("easy", seed=42)
        _, info = env.reset(seed=42)
        rendered = json.loads(GameStateRenderer(env).render_json(info))

        assert rendered["game_progress"]["phase"] == "play"
        assert len(rendered["hand"]) == 8
        assert rendered["jokers"][0]["id"] == "joker_basic"
        assert {"play", "discard"} <= {
            action["type"] for action in rendered["valid_actions"]
        }
