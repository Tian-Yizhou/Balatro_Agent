"""Gaining and losing cards: the effects that last while a joker is owned (and
not debuffed), the jokers' reactions to new playing cards, drawing cards, debuffs,
and disabling the boss blind, which Chicot does as soon as it is gained.

Source: card.lua ``Card:add_to_deck``, ``Card:remove_from_deck`` and
``Card:set_debuff``; cardarea.lua ``CardArea:change_size``; blind.lua
``Blind:disable``; functions/common_events.lua ``calculate_reroll_cost``;
functions/misc_functions.lua ``playing_card_joker_effects``.
"""

from __future__ import annotations

from collections.abc import Sequence

from .blinds import BLINDS, debuffs_card, keeps_joker_debuffs, stays_face_down
from .cards import sort_key
from .effects import Context, ContextKind, calculate_joker, value
from .events import BlindDisabled, CardsDrawn, JokerDebuffed, JokerDestroyed, PlayingCardCreated
from .scope import Scope
from .state import HandSort, Item, Phase, PlayingCard, RunState

__all__ = [
    "add_playing_cards",
    "add_to_deck",
    "calculate_reroll_cost",
    "change_hand_size",
    "destroy_jokers",
    "disable_blind",
    "draw_cards",
    "refresh_card_debuffs",
    "refresh_debuffs",
    "remove_from_deck",
    "set_joker_debuff",
]

# Bosses that deal cards face down; disabling one turns the hand face up.
_FACE_DOWN_BOSSES = frozenset({"bl_wheel", "bl_house", "bl_mark", "bl_fish"})


def add_to_deck(scope: Scope, item: Item) -> None:
    """Apply the lasting effects of a joker the player has just gained.

    Chicot gained while a boss is played disables it at once.
    """
    _apply(scope, item, 1)
    if item.key == "j_chicot" and _boss_can_be_disabled(scope.state):
        disable_blind(scope)


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
    """Draw up to ``count`` cards from the top of the deck and sort the hand. Some
    bosses keep drawn cards face down."""
    state = scope.state
    drawn = [state.deck.pop() for _ in range(max(0, min(count, len(state.deck))))]
    for card in drawn:
        card.face_down = stays_face_down(scope, card)
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


def set_joker_debuff(scope: Scope, joker: Item, debuffed: bool) -> None:
    """Debuff a joker or lift its debuff. A debuffed joker loses its lasting
    effects until the debuff is lifted; a perishable joker that has worn out stays
    debuffed. Source: card.lua ``Card:set_debuff``."""
    state = scope.state
    owned = any(joker is owned for owned in state.jokers)
    if joker.perishable and joker.perish_tally <= 0:
        debuffed = True
    if debuffed == joker.debuffed:
        return
    joker.debuffed = debuffed
    if owned:
        if debuffed:
            remove_from_deck(scope, joker)
        else:
            add_to_deck(scope, joker)
    scope.emit(JokerDebuffed(joker.uid, joker.key, debuffed))


def refresh_card_debuffs(state: RunState, cards: Sequence[PlayingCard]) -> None:
    """Re-check the blind's debuff on cards that just changed (their rank, suit or
    enhancement), as the game does when it sets a card's base or ability."""
    for card in cards:
        card.debuffed = debuffs_card(state, card)


def refresh_debuffs(scope: Scope) -> None:
    """Re-check every playing card and joker against the blind being played.

    Source: blind.lua ``Blind:set_blind`` and ``Blind:disable`` (the debuff loops).
    """
    state = scope.state
    refresh_card_debuffs(state, state.playing_cards())
    if not keeps_joker_debuffs(state):
        for joker in list(state.jokers):
            set_joker_debuff(scope, joker, False)


def _boss_can_be_disabled(state: RunState) -> bool:
    blind = state.blind
    return blind is not None and BLINDS[blind.key].boss and not blind.disabled


def disable_blind(scope: Scope) -> None:
    """Turn off the boss's effect for the rest of the round.

    What the boss took away comes back, cards and jokers turn face up, debuffs
    are re-checked, and The Wall and Violet Vessel lower their score target. The
    caller ends the round if the score now beats the target.
    Source: blind.lua ``Blind:disable``.
    """
    state = scope.state
    blind = state.blind
    assert blind is not None
    blind.disabled = True
    for joker in state.jokers:
        joker.face_down = False
    key = blind.key
    if key == "bl_water":
        state.discards_left += blind.discards_taken
    if key in _FACE_DOWN_BOSSES:
        for card in state.hand:
            card.face_down = False
    if key == "bl_needle":
        state.hands_left += blind.hands_taken
    if key == "bl_wall":
        blind.chips /= 2
    if key == "bl_final_bell":
        for card in state.playing_cards():
            card.forced_selection = False
    if key == "bl_manacle":
        change_hand_size(scope, 1)
        draw_cards(scope, 1)
    if key == "bl_final_vessel":
        blind.chips /= 3
    refresh_debuffs(scope)
    scope.emit(BlindDisabled(key))
