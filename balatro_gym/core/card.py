"""Card and Deck primitives for the Balatro game environment."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


class Suit(enum.IntEnum):
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3


class Rank(enum.IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


SUIT_SYMBOLS: dict[Suit, str] = {
    Suit.HEARTS: "\u2665",
    Suit.DIAMONDS: "\u2666",
    Suit.CLUBS: "\u2663",
    Suit.SPADES: "\u2660",
}

RANK_SYMBOLS: dict[Rank, str] = {
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "10",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
}


@dataclass
class Card:
    rank: Rank
    suit: Suit
    face_down: bool = False

    @property
    def chip_value(self) -> int:
        """Base chip value of this card (nominal). Ace=11, face cards=10, others=rank.

        Returns 0 if the card is debuffed (face-down).
        """
        if self.face_down:
            return 0
        if self.rank == Rank.ACE:
            return 11
        elif self.rank >= Rank.JACK:
            return 10
        else:
            return int(self.rank)

    @property
    def nominal(self) -> int:
        """Raw chip value ignoring debuff. Used for joker calculations like Raised Fist."""
        if self.rank == Rank.ACE:
            return 11
        elif self.rank >= Rank.JACK:
            return 10
        else:
            return int(self.rank)

    @property
    def id(self) -> int:
        """Card ID matching Lua's card.base.id (2-14, Ace=14)."""
        return int(self.rank)

    @property
    def is_face_card(self) -> bool:
        return self.rank in (Rank.JACK, Rank.QUEEN, Rank.KING)

    def __repr__(self) -> str:
        return f"{RANK_SYMBOLS[self.rank]}{SUIT_SYMBOLS[self.suit]}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))


def make_standard_deck() -> list[Card]:
    """Create a standard 52-card deck."""
    return [Card(rank=r, suit=s) for s in Suit for r in Rank]


class Deck:
    """A deck of cards with draw pile and discard pile."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        self._full_deck: list[Card] = make_standard_deck()
        self.draw_pile: list[Card] = []
        self.discard_pile: list[Card] = []

    @property
    def cards_remaining(self) -> int:
        return len(self.draw_pile)

    def reset(self) -> None:
        """Reset deck to full 52 cards and shuffle."""
        # Create fresh cards (so face_down state is clean)
        self.draw_pile = make_standard_deck()
        self.discard_pile = []
        self.shuffle()

    def shuffle(self) -> None:
        """Shuffle the draw pile in-place."""
        self.rng.shuffle(self.draw_pile)

    def draw(self, n: int) -> list[Card]:
        """Draw n cards from the top of the draw pile.

        If fewer than n cards remain, shuffle discard pile back in first.
        If still not enough, draw whatever is available.
        """
        if n <= 0:
            return []

        if len(self.draw_pile) < n:
            # Shuffle discard pile back into draw pile
            self.draw_pile.extend(self.discard_pile)
            self.discard_pile = []
            self.rng.shuffle(self.draw_pile)

        actual_n = min(n, len(self.draw_pile))
        drawn = self.draw_pile[:actual_n]
        self.draw_pile = self.draw_pile[actual_n:]
        return drawn

    def return_cards(self, cards: Sequence[Card]) -> None:
        """Return cards to the discard pile."""
        self.discard_pile.extend(cards)
