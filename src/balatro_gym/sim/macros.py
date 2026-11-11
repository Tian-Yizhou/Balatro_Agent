"""Macro helpers: expand a high-level intent into atomic actions.

Each helper returns the atomic actions that carry out the intent from the given
state, always in the same way, so scripted agents, tests and LLM tools can work
at a higher level while replays still record only atomic actions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .actions import Action, Discard, MoveCard, Play, ToggleCard
from .engine import SELECTION_LIMIT
from .state import RunState

__all__ = ["arrange_hand", "discard_cards", "play_cards", "select_cards"]


def select_cards(state: RunState, indices: Iterable[int]) -> list[Action]:
    """Toggles that leave exactly the hand cards at ``indices`` selected.

    Cards that should not be selected are deselected first, then the others are
    selected from left to right.
    """
    wanted = sorted(set(indices))
    if any(not 0 <= i < len(state.hand) for i in wanted):
        raise ValueError(f"indices {wanted} out of range for a hand of {len(state.hand)}")
    if len(wanted) > SELECTION_LIMIT:
        raise ValueError(f"at most {SELECTION_LIMIT} cards can be selected")
    wanted_set = set(wanted)
    actions: list[Action] = [
        ToggleCard(i) for i, card in enumerate(state.hand) if card.selected and i not in wanted_set
    ]
    actions.extend(ToggleCard(i) for i in wanted if not state.hand[i].selected)
    return actions


def play_cards(state: RunState, indices: Iterable[int]) -> list[Action]:
    """Select the hand cards at ``indices`` and play them."""
    return [*select_cards(state, indices), Play()]


def discard_cards(state: RunState, indices: Iterable[int]) -> list[Action]:
    """Select the hand cards at ``indices`` and discard them."""
    return [*select_cards(state, indices), Discard()]


def arrange_hand(state: RunState, order: Sequence[int]) -> list[Action]:
    """Moves that reorder the hand so that new position ``k`` holds old card ``order[k]``."""
    size = len(state.hand)
    if sorted(order) != list(range(size)):
        raise ValueError(f"{list(order)} is not a permutation of the {size} hand positions")
    current = list(range(size))
    actions: list[Action] = []
    for target, wanted in enumerate(order):
        source = current.index(wanted)
        if source != target:
            current.insert(target, current.pop(source))
            actions.append(MoveCard(source, target))
    return actions
