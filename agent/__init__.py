"""Balatro agent code: baselines, RLlib training, and LLM agents.

This package is independent from ``balatro_gym`` (the environment).
It imports from ``balatro_gym`` but ``balatro_gym`` never imports from here.

Subpackages:

* ``agent.baselines`` — Random and heuristic agents
* ``agent.rllib`` — Ray RLlib PPO training with action masking
* ``agent.llm`` — LLM-based agent with game state renderer

Key exports:

* :class:`Agent` — Protocol for all agents
* :func:`run_episode` — Run one episode with any agent
* :func:`evaluate_agent` — Evaluate agent over multiple episodes
"""

from agent.base import Agent, run_episode, evaluate_agent
