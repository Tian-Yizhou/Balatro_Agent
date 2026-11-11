"""``Game``: a convenient object wrapper around a run state and the engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import engine
from .actions import Action
from .events import Event
from .serialize import clone, restore, snapshot
from .state import RunConfig, RunState

__all__ = ["Game"]


class Game:
    """One run of the game.

    ``step`` changes the game in place. Use ``clone`` to branch off an exact copy,
    or ``determinize`` to branch off a copy whose hidden information is
    re-randomized.
    """

    __slots__ = ("state",)

    def __init__(self, state: RunState) -> None:
        self.state = state

    @classmethod
    def new(cls, seed: str | int, config: RunConfig | None = None) -> Game:
        """Start a new run."""
        return cls(engine.new_run(seed, config))

    @classmethod
    def from_snapshot(cls, data: Mapping[str, Any]) -> Game:
        """Resume a run from a snapshot made by :meth:`snapshot`."""
        return cls(restore(data))

    def snapshot(self) -> dict[str, Any]:
        """A JSON-compatible snapshot that restores this exact game."""
        return snapshot(self.state)

    def step(self, action: Action) -> list[Event]:
        """Apply an atomic action and return the events it caused."""
        return engine.step(self.state, action)

    def legal_actions(self) -> list[Action]:
        return engine.legal_actions(self.state)

    def is_legal(self, action: Action) -> bool:
        return engine.is_legal(self.state, action)

    def clone(self) -> Game:
        """An exact copy that shares nothing with this game."""
        return Game(clone(self.state))

    def determinize(self, seed: str | int) -> Game:
        """A copy with the same visible state and a re-randomized hidden future."""
        return Game(engine.determinize(self.state, seed))

    @property
    def is_over(self) -> bool:
        return self.state.is_over
