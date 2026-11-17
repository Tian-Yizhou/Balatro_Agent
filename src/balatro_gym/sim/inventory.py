"""Gaining and losing cards: the effects that last while a joker is owned, and
the jokers' reactions to new playing cards.

Source: card.lua ``Card:add_to_deck`` and ``Card:remove_from_deck``; cardarea.lua
``CardArea:change_size``; functions/common_events.lua ``calculate_reroll_cost``;
functions/misc_functions.lua ``playing_card_joker_effects``.
"""

from __future__ import annotations

from collections.abc import Sequence

from .cards import sort_key
from .effects import Context, ContextKind, calculate_joker, value
from .events import CardsDrawn, JokerDestroyed, PlayingCardCreated
from .scope import Scope
from .state import HandSort, Item, Phase, PlayingCard

__all__ = [
    "add_playing_cards",
    "add_to_deck",
    "calculate_reroll_cost",
    "change_hand_size",
    "destroy_jokers",
    "draw_cards",
    "remove_from_deck",
]


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


def destroy_jokers(scope: Scope, jokers: Sequence[Item], reason: str) -> None:
    """Remove jokers at once, undoing their lasting effects (``Card:start_dissolve``)."""
    state = scope.state
    for joker in jokers:
        if joker in state.jokers:
            state.jokers.remove(joker)
            remove_from_deck(scope, joker)
            scope.emit(JokerDestroyed(joker.uid, joker.key, reason))


def add_playing_cards(scope: Scope, cards: Sequence[PlayingCard], area: str) -> None:
    """Announce new playing cards, already placed in ``area``, and let jokers react."""
    for card in cards:
        scope.emit(PlayingCardCreated(card.uid, card.label, area))
    for joker in list(scope.state.jokers):
        calculate_joker(scope, joker, Context(ContextKind.PLAYING_CARD_ADDED, cards=cards))


def change_hand_size(scope: Scope, delta: int) -> None:
    """Change the hand size; a bigger hand draws at once while a round is played."""
    state = scope.state
    state.hand_size += delta
    if delta > 0 and state.phase is Phase.SELECTING_HAND and state.hand:
        draw_cards(scope, delta)


def draw_cards(scope: Scope, count: int) -> None:
    """Draw up to ``count`` cards from the top of the deck and sort the hand."""
    state = scope.state
    drawn = [state.deck.pop() for _ in range(max(0, min(count, len(state.deck))))]
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
