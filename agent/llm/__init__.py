"""LLM-based agent for Balatro.

Provides a text-based interface that converts game state to structured
JSON (or natural language) for LLM-based decision making.

Key components:

* :class:`GameStateRenderer` — Converts BalatroEnv internal state to a
  structured JSON dict suitable for LLM prompting.
* :class:`LLMAgent` — Agent that uses a model backend to play via text.
  (Backend implementations are TODO — see ``backends.py``.)
"""

from agent.llm.renderer import GameStateRenderer
