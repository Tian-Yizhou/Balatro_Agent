"""The rules engine: a pure, deterministic simulation of Balatro.

Typical use::

    from balatro_gym.sim import Game, macros

    game = Game.new(seed="ABC123")
    game.step(game.legal_actions()[0])  # select the small blind
    for action in macros.play_cards(game.state, [0, 1]):
        game.step(action)
"""

from . import actions, events, macros
from .actions import Action, parse_action
from .engine import IllegalActionError, determinize, is_legal, legal_actions, new_run, step
from .events import Event
from .game import Game
from .serialize import SnapshotError, clone, restore, snapshot
from .state import Phase, RunConfig, RunState
from .version import GAME_VERSION, RULES_VERSION, STATE_SCHEMA_VERSION

__all__ = [
    "GAME_VERSION",
    "RULES_VERSION",
    "STATE_SCHEMA_VERSION",
    "Action",
    "Event",
    "Game",
    "IllegalActionError",
    "Phase",
    "RunConfig",
    "RunState",
    "SnapshotError",
    "actions",
    "clone",
    "determinize",
    "events",
    "is_legal",
    "legal_actions",
    "macros",
    "new_run",
    "parse_action",
    "restore",
    "snapshot",
    "step",
]
