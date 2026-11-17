"""Boss blind effects over a round: when the blind is set, when a hand is played,
after each draw, and when the boss is beaten.

The rules for single cards (debuffs, cards dealt face down) live in ``blinds``;
disabling a boss is in ``inventory``, since Chicot does it as soon as it is
gained; and the effects on a played hand (hands that don't score, The Arm, The Ox
and The Flint) are part of scoring, in ``scoring``.

Source: blind.lua ``Blind:set_blind``, ``Blind:press_play``,
``Blind:drawn_to_hand`` and ``Blind:defeat``; functions/state_events.lua
``G.FUNCS.draw_from_deck_to_hand`` (The Serpent) and ``end_round`` (the most
played hand).
"""

from __future__ import annotations

from .blinds import BLINDS
from .events import BlindTriggered, CardToggled
from .inventory import change_hand_size, refresh_debuffs, set_joker_debuff
from .scope import Scope
from .state import HandType, PlayingCard, RunState

__all__ = [
    "defeat_blind",
    "draw_count",
    "drawn_to_hand",
    "most_played_hand",
    "press_play",
    "start_blind",
]


def _active_boss_key(state: RunState) -> str | None:
    blind = state.blind
    if blind is None or blind.disabled or not BLINDS[blind.key].boss:
        return None
    return blind.key


def start_blind(scope: Scope) -> None:
    """Apply the blind's effects as it is set, then re-check every debuff.

    The Water takes every discard, The Needle every hand but one, The Manacle one
    hand size, and Amber Acorn turns the jokers face down and shuffles them.
    Source: blind.lua ``Blind:set_blind``.
    """
    state = scope.state
    blind = state.blind
    assert blind is not None
    key = blind.key
    blind.prepped = key != "bl_fish"
    if key == "bl_water":
        blind.discards_taken = state.discards_left
        state.discards_left = 0
    if key == "bl_needle":
        blind.hands_taken = state.hands_per_round - 1
        state.hands_left -= blind.hands_taken
    if key == "bl_manacle":
        change_hand_size(scope, -1)
    if key == "bl_final_acorn" and state.jokers:
        for joker in state.jokers:
            joker.face_down = True
        if len(state.jokers) > 1:
            # The shuffles run in the game's event queue, after the jokers react
            # to the blind.
            scope.defer(lambda: scope.defer(lambda: _shuffle_jokers(scope)))
    refresh_debuffs(scope)


def _shuffle_jokers(scope: Scope) -> None:
    jokers = scope.state.jokers
    for _ in range(3):
        jokers.sort(key=lambda joker: joker.uid)
        scope.rng.shuffle("aajk", jokers)


def press_play(scope: Scope, played: list[PlayingCard]) -> list[PlayingCard]:
    """React to a hand being played, before it scores. Returns the cards The Hook
    discards from the hand, in hand order.

    The Tooth costs $1 per card played; The Fish and Crimson Heart get ready to
    act on the next draw. Source: blind.lua ``Blind:press_play``.
    """
    state = scope.state
    key = _active_boss_key(state)
    blind = state.blind
    if key is None or blind is None:
        return []
    if key == "bl_hook":
        scope.emit(BlindTriggered(key))
        candidates = sorted(state.hand, key=lambda card: card.uid)
        chosen: list[PlayingCard] = []
        for _ in range(min(2, len(state.hand))):
            card = scope.rng.choice("hook", candidates)
            candidates.remove(card)
            chosen.append(card)
        return [card for card in state.hand if card in chosen]
    if key == "bl_final_heart" and state.jokers:
        blind.prepped = True
    if key == "bl_fish":
        blind.prepped = True
    if key == "bl_tooth":
        scope.emit(BlindTriggered(key))
        scope.change_money(-len(played), "tooth")
    return []


def draw_count(state: RunState) -> int:
    """How many cards to draw: up to the hand size, or always 3 with The Serpent
    after the first hand or discard. Source: ``G.FUNCS.draw_from_deck_to_hand``."""
    played_or_discarded = state.hands_played_this_round or state.discards_used_this_round
    if _active_boss_key(state) == "bl_serpent" and played_or_discarded:
        return min(len(state.deck), 3)
    return state.hand_size - len(state.hand)


def drawn_to_hand(scope: Scope) -> None:
    """Act once the hand is drawn: Cerulean Bell selects a random card for good,
    and Crimson Heart moves its debuff to another random joker after each hand.

    Source: blind.lua ``Blind:drawn_to_hand``.
    """
    state = scope.state
    key = _active_boss_key(state)
    blind = state.blind
    forced_already = any(card.forced_selection for card in state.hand)
    if key == "bl_final_bell" and state.hand and not forced_already:
        for card in state.hand:
            card.selected = False
        forced = scope.rng.choice("cerulean_bell", sorted(state.hand, key=lambda c: c.uid))
        forced.forced_selection = True
        forced.selected = True
        scope.emit(CardToggled(forced.uid, True))
    if key == "bl_final_heart" and blind is not None and blind.prepped and state.jokers:
        jokers = state.jokers
        candidates = [joker for joker in jokers if not joker.debuffed or len(jokers) < 2]
        for joker in list(jokers):
            set_joker_debuff(scope, joker, False)
        if candidates:
            chosen = scope.rng.choice("crimson_heart", sorted(candidates, key=lambda j: j.uid))
            set_joker_debuff(scope, chosen, True)
            scope.emit(BlindTriggered(key))
    if blind is not None:
        blind.prepped = False


def defeat_blind(scope: Scope) -> None:
    """The blind is beaten: The Manacle gives back its hand size, the jokers turn
    face up, and the debuffs end. Source: blind.lua ``Blind:defeat``."""
    state = scope.state
    blind = state.blind
    assert blind is not None
    for joker in state.jokers:
        joker.face_down = False
    if blind.key == "bl_manacle" and not blind.disabled:
        change_hand_size(scope, 1)
    state.blind = None
    refresh_debuffs(scope)


def most_played_hand(state: RunState) -> HandType:
    """The hand played most often in the run, for The Ox.

    On a tie the game's pick depends on the order Lua iterates a table in, which
    varies; the engine takes the tied hand that comes first in the game's hand
    order, as the comparison in the game's code intends.
    Source: functions/state_events.lua ``end_round``.
    """
    # ``max`` keeps the first of equal hands, and HandType is in the game's order.
    return max(HandType, key=lambda hand: state.hands[hand].played)
