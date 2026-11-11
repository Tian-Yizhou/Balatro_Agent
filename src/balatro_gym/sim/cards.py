"""Rank, suit and ordering rules for playing cards.

Source: card.lua ``Card:set_base``, ``Card:get_id``, ``Card:get_nominal``,
``Card:get_chip_bonus`` and ``Card:is_suit``.
"""

from __future__ import annotations

from typing import Literal

from .state import PlayingCard, Suit

__all__ = [
    "STONE",
    "SUITS_IN_FLUSH_ORDER",
    "WILD",
    "card_id",
    "chip_bonus",
    "is_suit_for_flush",
    "nominal",
    "sort_key",
]

STONE = "m_stone"
WILD = "m_wild"

#: The order in which ``get_flush`` tries suits.
SUITS_IN_FLUSH_ORDER = (Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS)

# Card:set_base. Ranks 11-14 are Jack, Queen, King and Ace.
_NOMINAL = {r: r for r in range(2, 11)} | {11: 10, 12: 10, 13: 10, 14: 11}
_FACE_NOMINAL = {11: 0.1, 12: 0.2, 13: 0.3, 14: 0.4}
_SUIT_NOMINAL = {Suit.DIAMONDS: 0.01, Suit.CLUBS: 0.02, Suit.HEARTS: 0.03, Suit.SPADES: 0.04}
_SUIT_NOMINAL_ORIGINAL = {
    Suit.DIAMONDS: 0.001,
    Suit.CLUBS: 0.002,
    Suit.HEARTS: 0.003,
    Suit.SPADES: 0.004,
}
# Card:init derives ``unique_val`` from the card's node ID with this divisor.
_UNIQUE_DIVISOR = 1603301


def nominal(card: PlayingCard) -> int:
    """The chips a card's rank is worth: 2-10 at face value, J/Q/K 10, Ace 11."""
    return _NOMINAL[card.rank]


def card_id(card: PlayingCard) -> int | None:
    """The rank used for matching (``Card:get_id``), or ``None`` for Stone cards.

    The game gives a Stone card a random negative ID on every call, so it never
    matches another card's rank and is never part of a straight.
    """
    if card.enhancement == STONE:
        return None
    return card.rank


def chip_bonus(card: PlayingCard) -> int:
    """Chips a card adds when it scores (``Card:get_chip_bonus``)."""
    if card.debuffed:
        return 0
    return _NOMINAL[card.rank] + card.perma_bonus


def is_suit_for_flush(card: PlayingCard, suit: Suit, *, smeared: bool = False) -> bool:
    """Whether a card counts as ``suit`` when looking for a flush.

    Wild cards count as every suit unless debuffed; Stone cards count as none.
    With Smeared Joker, Hearts and Diamonds count as one suit, as do Spades and
    Clubs.
    """
    if card.enhancement == STONE:
        return False
    if card.enhancement == WILD and not card.debuffed:
        return True
    if smeared and _is_red(card.suit) == _is_red(suit):
        return True
    return card.suit is suit


def sort_key(card: PlayingCard, mode: Literal["rank", "suit"] = "rank") -> float:
    """The value the game sorts and compares cards by (``Card:get_nominal``).

    Higher means further left when the hand is sorted. The arithmetic follows the
    original expression term by term, so ties and float rounding match it.
    """
    mult = 1000 if mode == "suit" else 1
    if card.enhancement == STONE:
        mult = -1000
    unique_val = 1 - card.uid / _UNIQUE_DIVISOR
    return (
        _NOMINAL[card.rank]
        + _SUIT_NOMINAL[card.suit] * mult
        + _SUIT_NOMINAL_ORIGINAL[card.suit] * 0.0001 * mult
        + _FACE_NOMINAL.get(card.rank, 0)
        + 0.000001 * unique_val
    )


def _is_red(suit: Suit) -> bool:
    return suit is Suit.HEARTS or suit is Suit.DIAMONDS
