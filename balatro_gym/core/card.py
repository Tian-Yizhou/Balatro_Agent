"""Card and Deck primitives for the Balatro game environment.

Supports enhancements, editions, seals, unique card IDs, and mutable decks.
Reference: Balatro Lua source — card.lua (get_chip_bonus, get_chip_mult, etc.)
"""

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


# ---------------------------------------------------------------------------
# Card property enums
# ---------------------------------------------------------------------------

class Enhancement(enum.Enum):
    """Card enhancements (Lua: G.P_CENTERS m_* entries)."""
    BONUS = "m_bonus"       # +30 chips
    MULT = "m_mult"         # +4 mult
    WILD = "m_wild"         # counts as any suit
    GLASS = "m_glass"       # X2 mult, 1/4 chance to shatter
    STEEL = "m_steel"       # X1.5 mult when held in hand
    STONE = "m_stone"       # +50 chips, no nominal value
    GOLD = "m_gold"         # $3 at end of round when held
    LUCKY = "m_lucky"       # 1/5 chance +20 mult, 1/15 chance $20


class Edition(enum.Enum):
    """Card editions (Lua: G.P_CENTERS e_* entries)."""
    FOIL = "e_foil"             # +50 chips
    HOLO = "e_holo"             # +10 mult
    POLYCHROME = "e_polychrome" # X1.5 mult


class Seal(enum.Enum):
    """Card seals."""
    GOLD = "Gold"       # +$3 when played
    RED = "Red"         # retrigger (card scores twice)
    BLUE = "Blue"       # create planet card at end of round
    PURPLE = "Purple"   # create tarot card when discarded


# Enhancement config lookup (matching Lua game.lua lines 648-655)
ENHANCEMENT_CONFIG: dict[Enhancement, dict] = {
    Enhancement.BONUS:  {"bonus": 30},
    Enhancement.MULT:   {"mult": 4},
    Enhancement.WILD:   {},
    Enhancement.GLASS:  {"Xmult": 2, "extra": 4},      # 1 in extra chance to shatter
    Enhancement.STEEL:  {"h_x_mult": 1.5},
    Enhancement.STONE:  {"bonus": 50},
    Enhancement.GOLD:   {"h_dollars": 3},
    Enhancement.LUCKY:  {"mult": 20, "p_dollars": 20},  # 1/5 mult, 1/15 dollars
}

# Edition config lookup (matching Lua game.lua lines 658-662)
EDITION_CONFIG: dict[Edition, dict] = {
    Edition.FOIL:       {"chips": 50},
    Edition.HOLO:       {"mult": 10},
    Edition.POLYCHROME: {"x_mult": 1.5},
}


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

_SUIT_NAME: dict[str, Suit] = {
    "Hearts": Suit.HEARTS,
    "Diamonds": Suit.DIAMONDS,
    "Clubs": Suit.CLUBS,
    "Spades": Suit.SPADES,
}


# ---------------------------------------------------------------------------
# Card UID counter
# ---------------------------------------------------------------------------
_next_uid: int = 0


def _make_uid() -> int:
    global _next_uid
    _next_uid += 1
    return _next_uid


def reset_uid_counter() -> None:
    """Reset the UID counter (useful for tests)."""
    global _next_uid
    _next_uid = 0


def _get_next_uid() -> int:
    """Return current UID counter value (for serialization)."""
    return _next_uid


def _set_next_uid(value: int) -> None:
    """Set the UID counter (for deserialization)."""
    global _next_uid
    _next_uid = value


# ---------------------------------------------------------------------------
# Card dataclass
# ---------------------------------------------------------------------------

@dataclass
class Card:
    rank: Rank
    suit: Suit
    face_down: bool = False
    enhancement: Enhancement | None = None
    edition: Edition | None = None
    seal: Seal | None = None
    uid: int = field(default_factory=_make_uid)

    # ------------------------------------------------------------------
    # Chip value (matching Lua card.lua get_chip_bonus, lines 976-982)
    # ------------------------------------------------------------------

    @property
    def chip_value(self) -> int:
        """Chip bonus of this card including enhancement.

        Matches Lua: Card:get_chip_bonus().
        Stone Card returns only its bonus (no nominal).
        Bonus Card adds +30 to nominal.
        Returns 0 if debuffed.
        """
        if self.face_down:
            return 0
        if self.enhancement == Enhancement.STONE:
            return ENHANCEMENT_CONFIG[Enhancement.STONE]["bonus"]
        base = self.nominal
        if self.enhancement == Enhancement.BONUS:
            base += ENHANCEMENT_CONFIG[Enhancement.BONUS]["bonus"]
        return base

    @property
    def nominal(self) -> int:
        """Raw chip value ignoring debuff and enhancement.

        Ace=11, face cards=10, others=rank value.
        """
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

    @property
    def is_wild(self) -> bool:
        """True if this card has the Wild Card enhancement (counts as any suit)."""
        return self.enhancement == Enhancement.WILD

    # ------------------------------------------------------------------
    # Enhancement scoring methods (matching Lua card.lua lines 984-1014)
    # ------------------------------------------------------------------

    def get_chip_mult(self, rng: np.random.Generator | None = None) -> int:
        """Mult bonus from enhancement.

        Lua: Card:get_chip_mult()
        - Mult Card: +4
        - Lucky Card: 1/5 (20%) chance for +20
        - Others: 0
        """
        if self.face_down:
            return 0
        if self.enhancement == Enhancement.MULT:
            return ENHANCEMENT_CONFIG[Enhancement.MULT]["mult"]
        if self.enhancement == Enhancement.LUCKY:
            if rng is not None and rng.random() < 1 / 5:
                return ENHANCEMENT_CONFIG[Enhancement.LUCKY]["mult"]
            return 0
        return 0

    def get_chip_x_mult(self) -> float:
        """X-mult from enhancement.

        Lua: Card:get_chip_x_mult()
        - Glass Card: X2
        - Others: 0
        """
        if self.face_down:
            return 0.0
        if self.enhancement == Enhancement.GLASS:
            return float(ENHANCEMENT_CONFIG[Enhancement.GLASS]["Xmult"])
        return 0.0

    def get_held_x_mult(self) -> float:
        """X-mult from enhancement when held in hand (not played).

        Lua: Card:get_chip_h_x_mult()
        - Steel Card: X1.5
        - Others: 0
        """
        if self.face_down:
            return 0.0
        if self.enhancement == Enhancement.STEEL:
            return ENHANCEMENT_CONFIG[Enhancement.STEEL]["h_x_mult"]
        return 0.0

    def get_held_dollars(self) -> int:
        """Dollars earned at end of round when held.

        - Gold Card: $3
        """
        if self.face_down:
            return 0
        if self.enhancement == Enhancement.GOLD:
            return ENHANCEMENT_CONFIG[Enhancement.GOLD]["h_dollars"]
        return 0

    def get_played_dollars(self) -> int:
        """Dollars earned when this card is played (from seals).

        - Gold Seal: $3
        """
        if self.seal == Seal.GOLD and not self.face_down:
            return 3
        return 0

    # ------------------------------------------------------------------
    # Edition scoring (matching Lua card.lua get_edition, lines 1016-1030)
    # ------------------------------------------------------------------

    def get_edition_bonus(self) -> tuple[int, int, float]:
        """Return (chips, mult, x_mult) from this card's edition.

        Foil: +50 chips
        Holo: +10 mult
        Polychrome: X1.5 mult
        """
        if self.face_down or self.edition is None:
            return (0, 0, 0.0)
        cfg = EDITION_CONFIG[self.edition]
        return (
            cfg.get("chips", 0),
            cfg.get("mult", 0),
            cfg.get("x_mult", 0.0),
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def same_card(self, other: Card) -> bool:
        """True if same rank and suit (ignoring uid, enhancement, etc.)."""
        return self.rank == other.rank and self.suit == other.suit

    def copy(self) -> Card:
        """Create a copy of this card with a new uid."""
        return Card(
            rank=self.rank,
            suit=self.suit,
            face_down=self.face_down,
            enhancement=self.enhancement,
            edition=self.edition,
            seal=self.seal,
        )

    def __repr__(self) -> str:
        parts = [f"{RANK_SYMBOLS[self.rank]}{SUIT_SYMBOLS[self.suit]}"]
        if self.enhancement:
            parts.append(f"[{self.enhancement.name}]")
        if self.edition:
            parts.append(f"({self.edition.name})")
        if self.seal:
            parts.append(f"<{self.seal.name}>")
        return "".join(parts)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize this card to a JSON-compatible dict."""
        return {
            "rank": int(self.rank),
            "suit": int(self.suit),
            "uid": self.uid,
            "enhancement": self.enhancement.value if self.enhancement else None,
            "edition": self.edition.value if self.edition else None,
            "seal": self.seal.value if self.seal else None,
            "face_down": self.face_down,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Card:
        """Deserialize a card from a dict (as produced by ``to_dict``)."""
        enh = Enhancement(d["enhancement"]) if d.get("enhancement") else None
        ed = Edition(d["edition"]) if d.get("edition") else None
        sl = Seal(d["seal"]) if d.get("seal") else None
        card = cls.__new__(cls)
        card.rank = Rank(d["rank"])
        card.suit = Suit(d["suit"])
        card.uid = d["uid"]
        card.enhancement = enh
        card.edition = ed
        card.seal = sl
        card.face_down = d.get("face_down", False)
        return card

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.uid == other.uid

    def __hash__(self) -> int:
        return hash(self.uid)


def make_standard_deck() -> list[Card]:
    """Create a standard 52-card deck."""
    return [Card(rank=r, suit=s) for s in Suit for r in Rank]


class Deck:
    """A deck of cards with draw pile and discard pile.

    Supports mutable operations: cards can be added or removed at any time,
    allowing the deck to grow beyond or shrink below 52 cards.
    """

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        self.draw_pile: list[Card] = []
        self.discard_pile: list[Card] = []

    @property
    def cards_remaining(self) -> int:
        """Number of cards in the draw pile."""
        return len(self.draw_pile)

    @property
    def card_count(self) -> int:
        """Total number of cards in the deck (draw + discard)."""
        return len(self.draw_pile) + len(self.discard_pile)

    @property
    def all_cards(self) -> list[Card]:
        """All cards in the deck (draw pile + discard pile)."""
        return self.draw_pile + self.discard_pile

    def reset(self, cards: list[Card] | None = None) -> None:
        """Reset deck and shuffle.

        Args:
            cards: If provided, use these cards instead of a standard 52-card deck.
                   The cards are copied (new uids) to avoid sharing state.
        """
        if cards is not None:
            self.draw_pile = [c.copy() for c in cards]
        else:
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

    def add_card(self, card: Card) -> None:
        """Add a new card to the draw pile (deck grows)."""
        self.draw_pile.append(card)

    def remove_card(self, card: Card) -> bool:
        """Remove a card by uid from the deck (draw or discard pile).

        Returns True if the card was found and removed.
        """
        for pile in (self.draw_pile, self.discard_pile):
            for i, c in enumerate(pile):
                if c.uid == card.uid:
                    pile.pop(i)
                    return True
        return False
