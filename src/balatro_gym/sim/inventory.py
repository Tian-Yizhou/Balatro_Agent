"""Gaining and losing jokers: the effects that last while a joker is owned.

Source: card.lua ``Card:add_to_deck`` and ``Card:remove_from_deck``; cardarea.lua
``CardArea:change_size``; functions/common_events.lua ``calculate_reroll_cost``.
"""

from __future__ import annotations

from .cards import sort_key
from .effects import value
from .events import CardsDrawn
from .scope import Scope
from .state import HandSort, Item, Phase

__all__ = ["add_to_deck", "calculate_reroll_cost", "change_hand_size", "remove_from_deck"]


def add_to_deck(scope: Scope, item: Item) -> None:
    """Apply the lasting effects of a joker the player has just gained."""
    _apply(scope, item, 1)


def remove_from_deck(scope: Scope, item: Item) -> None:
    """Undo the lasting effects of a joker the player is losing."""
    _apply(scope, item, -1)


def _apply(scope: Scope, item: Item, sign: int) -> None:
    state = scope.state
    hand_size = int(value(item, "h_size"))
    if hand_size:
        change_hand_size(scope, sign * hand_size)
    discards = int(value(item, "d_size"))
    if discards > 0:
        state.discards_per_round += sign * discards
        state.discards_left += sign * discards
    if item.key == "j_credit_card":
        state.bankrupt_at -= sign * int(value(item, "extra"))
    if item.key == "j_chaos":
        state.free_rerolls += sign
        calculate_reroll_cost(scope, skip_increment=True)


def change_hand_size(scope: Scope, delta: int) -> None:
    """Change the hand size; a bigger hand draws at once while a round is played."""
    state = scope.state
    state.hand_size += delta
    if delta > 0 and state.phase is Phase.SELECTING_HAND and state.hand:
        drawn = [state.deck.pop() for _ in range(min(delta, len(state.deck)))]
        if drawn:
            state.hand.extend(drawn)
            mode = "suit" if state.hand_sort is HandSort.SUIT else "rank"
            state.hand.sort(key=lambda card: sort_key(card, mode), reverse=True)
            scope.emit(CardsDrawn(tuple(card.uid for card in drawn)))


def calculate_reroll_cost(scope: Scope, *, skip_increment: bool = False) -> None:
    state = scope.state
    state.free_rerolls = max(state.free_rerolls, 0)
    if state.free_rerolls > 0:
        state.reroll_cost = 0
        return
    if not skip_increment:
        state.reroll_cost_increase += 1
    state.reroll_cost = state.base_reroll_cost + state.reroll_cost_increase
