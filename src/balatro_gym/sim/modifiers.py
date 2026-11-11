"""Enhancements, editions and seals on playing cards, and what they do in scoring.

Each function mirrors one ``Card`` method. A debuffed card has no effects.

Source: card.lua ``Card:get_chip_bonus``, ``get_chip_mult``, ``get_chip_x_mult``,
``get_chip_h_mult``, ``get_chip_h_x_mult``, ``get_edition``, ``get_p_dollars``,
``get_end_of_round_effect`` and ``calculate_seal``.
"""

from __future__ import annotations

from .cards import STONE, nominal
from .content import CENTERS, config_value
from .items import new_item
from .scope import Scope
from .state import Item, PlayingCard

__all__ = [
    "LUCKY",
    "chip_bonus",
    "chip_h_mult",
    "chip_h_x_mult",
    "chip_mult",
    "chip_x_mult",
    "edition_effect",
    "end_of_round_dollars",
    "has_room_for_consumable",
    "p_dollars",
    "red_seal_repetitions",
]

LUCKY = "m_lucky"


def _config(card: PlayingCard, name: str) -> float:
    if card.enhancement is None:
        return 0
    result: float = config_value(card.enhancement, name, 0)
    return result


def chip_bonus(card: PlayingCard) -> float:
    """Chips the card adds when it scores: its rank, plus any bonus."""
    if card.debuffed:
        return 0
    bonus = _config(card, "bonus") + card.perma_bonus
    if card.enhancement == STONE:
        return bonus
    return nominal(card) + bonus


def chip_mult(scope: Scope, card: PlayingCard) -> tuple[float, bool]:
    """Mult the card adds when it scores, and whether a Lucky Card triggered."""
    if card.debuffed:
        return 0, False
    mult = _config(card, "mult")
    if card.enhancement == LUCKY:
        if scope.chance("lucky_mult", 5):
            return mult, True
        return 0, False
    return mult, False


def chip_x_mult(card: PlayingCard) -> float:
    """The card's x-mult when it scores, or 0 if it has none."""
    if card.debuffed:
        return 0
    x_mult = _config(card, "Xmult") or 1
    return x_mult if x_mult > 1 else 0


def chip_h_mult(card: PlayingCard) -> float:
    return 0 if card.debuffed else _config(card, "h_mult")


def chip_h_x_mult(card: PlayingCard) -> float:
    return 0 if card.debuffed else _config(card, "h_x_mult")


def p_dollars(scope: Scope, card: PlayingCard) -> tuple[int, bool]:
    """Money the card pays when it scores, and whether a Lucky Card triggered."""
    if card.debuffed:
        return 0, False
    dollars = 3 if card.seal == "Gold" else 0
    lucky = False
    pay = int(_config(card, "p_dollars"))
    if pay > 0:
        if card.enhancement == LUCKY:
            if scope.chance("lucky_money", 15):
                dollars += pay
                lucky = True
        else:
            dollars += pay
    return dollars, lucky


def edition_effect(edition: str | None, debuffed: bool) -> tuple[float, float, float]:
    """The ``(chips, mult, x_mult)`` an edition adds; neutral values if none."""
    if debuffed or edition is None:
        return 0, 0, 1
    amount: float = CENTERS[edition].config.get("extra", 0)
    if edition == "e_foil":
        return amount, 0, 1
    if edition == "e_holo":
        return 0, amount, 1
    if edition == "e_polychrome":
        return 0, 0, amount
    return 0, 0, 1


def red_seal_repetitions(card: PlayingCard) -> int:
    return 1 if card.seal == "Red" and not card.debuffed else 0


def has_room_for_consumable(scope: Scope) -> bool:
    state = scope.state
    negatives = sum(item.edition == "e_negative" for item in state.consumables)
    return len(state.consumables) < state.consumable_slots + negatives


def end_of_round_dollars(scope: Scope, card: PlayingCard) -> tuple[int, bool]:
    """Money a card held at the end of the round pays, and whether it did anything.

    A Blue Seal also creates the Planet card for the last hand played, if there
    is room for it.
    """
    if card.debuffed:
        return 0, False
    dollars = int(_config(card, "h_dollars"))
    acted = dollars > 0
    if card.seal == "Blue" and has_room_for_consumable(scope):
        acted = True
        hand = scope.state.last_hand_played
        if hand is not None:
            planet = next(
                c.key
                for c in CENTERS.values()
                if c.set == "Planet" and c.config.get("hand_type") == hand.value
            )
            scope.defer(lambda: _add_consumable(scope, new_item(scope.state, planet)))
    return dollars, acted


def _add_consumable(scope: Scope, item: Item) -> None:
    scope.state.consumables.append(item)
