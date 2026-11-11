"""Versions recorded in every snapshot and replay."""

__all__ = ["GAME_VERSION", "RULES_VERSION", "STATE_SCHEMA_VERSION"]

#: The version of the original game whose rules the engine reproduces.
GAME_VERSION = "1.0.1o"

#: Bumped whenever the engine's game behavior changes. A replay recorded under a
#: different rules version may no longer re-simulate to the same states.
RULES_VERSION = 3

#: Bumped whenever the layout of the serialized state changes.
STATE_SCHEMA_VERSION = 3
