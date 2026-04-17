"""Blind progression, score targets, and boss blind effects.

Reference: Balatro Lua source — blind.lua, game.lua, misc_functions.lua
"""

from __future__ import annotations

import enum
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from balatro_gym.core.card import Card, Suit, Rank

if TYPE_CHECKING:
    from balatro_gym.core.game_state import GameState


class BlindType(enum.Enum):
    SMALL = "small"
    BIG = "big"
    BOSS = "boss"


# ---------------------------------------------------------------------------
# Base amounts per ante (Lua: get_blind_amount, scaling=1)
# ---------------------------------------------------------------------------
_BASE_AMOUNTS: list[int] = [
    300, 800, 2_000, 5_000, 11_000, 20_000, 35_000, 50_000
]


def get_blind_amount(ante: int) -> int:
    """Get the base blind amount for a given ante.

    Matches Lua's get_blind_amount() with scaling=1.
    For ante > 8, uses the Lua extrapolation formula.
    """
    if ante < 1:
        return 100
    if ante <= 8:
        return _BASE_AMOUNTS[ante - 1]

    # Extrapolation for ante > 8 (Lua formula)
    k = 0.75
    a = _BASE_AMOUNTS[7]  # 50000
    b = 1.6
    c = ante - 8
    d = 1 + 0.2 * c
    amount = int(math.floor(a * (b + (k * c) ** d) ** c))
    amount = amount - amount % (10 ** int(math.floor(math.log10(amount) - 1)))
    return amount


# ---------------------------------------------------------------------------
# Blind definitions (Lua: G.P_BLINDS)
# ---------------------------------------------------------------------------
# Each blind has: dollars (reward for beating), mult (score multiplier)

@dataclass
class BlindDef:
    name: str
    dollars: int
    mult: float
    is_boss: bool = False
    debuff_suit: str | None = None
    debuff_face: bool = False


SMALL_BLIND = BlindDef(name="Small Blind", dollars=3, mult=1.0)
BIG_BLIND = BlindDef(name="Big Blind", dollars=4, mult=1.5)

# Boss blinds (Lua: G.P_BLINDS)
BOSS_BLINDS: list[BlindDef] = [
    BlindDef(name="The Hook", dollars=5, mult=2.0, is_boss=True),
    BlindDef(name="The Club", dollars=5, mult=2.0, is_boss=True, debuff_suit="Clubs"),
    BlindDef(name="The Wall", dollars=5, mult=4.0, is_boss=True),
    BlindDef(name="The Goad", dollars=5, mult=2.0, is_boss=True, debuff_suit="Spades"),
    BlindDef(name="The Window", dollars=5, mult=2.0, is_boss=True, debuff_suit="Diamonds"),
    BlindDef(name="The Head", dollars=5, mult=2.0, is_boss=True, debuff_suit="Hearts"),
    BlindDef(name="The Plant", dollars=5, mult=2.0, is_boss=True, debuff_face=True),
    BlindDef(name="The Needle", dollars=5, mult=1.0, is_boss=True),  # Only 1 hand
    BlindDef(name="The Flint", dollars=5, mult=2.0, is_boss=True),   # Halves base chips and mult
]

# Map suit name strings to Suit enum
_SUIT_MAP: dict[str, Suit] = {
    "Hearts": Suit.HEARTS,
    "Diamonds": Suit.DIAMONDS,
    "Clubs": Suit.CLUBS,
    "Spades": Suit.SPADES,
}


# ---------------------------------------------------------------------------
# Boss effects
# ---------------------------------------------------------------------------

class BossEffect(ABC):
    """Base class for boss blind debuff effects."""

    name: str
    description: str

    @abstractmethod
    def apply(self, game_state: GameState) -> None:
        """Apply the debuff at the start of the boss blind."""
        ...

    @abstractmethod
    def remove(self, game_state: GameState) -> None:
        """Remove the debuff after the boss blind is beaten."""
        ...

    def modify_hand(self, mult: int, hand_chips: int) -> tuple[int, int, bool]:
        """Optionally modify base mult and chips at scoring time.

        Returns (mult, hand_chips, was_modified).
        """
        return mult, hand_chips, False

    def press_play(self, game_state: GameState) -> None:
        """Called when the player presses play (before scoring)."""
        pass

    def debuff_card(self, card: Card) -> bool:
        """Check if a card should be debuffed. Returns True if debuffed."""
        return False

    def __repr__(self) -> str:
        return f"{self.name}: {self.description}"


class DebuffSuit(BossEffect):
    """Debuffs all cards of a specific suit."""

    def __init__(self, suit: Suit, name: str, description: str):
        self._suit = suit
        self.name = name
        self.description = description

    def apply(self, game_state: GameState) -> None:
        for card in game_state.deck.draw_pile + game_state.deck.discard_pile + game_state.hand:
            if card.suit == self._suit:
                card.face_down = True

    def remove(self, game_state: GameState) -> None:
        for card in game_state.deck.draw_pile + game_state.deck.discard_pile + game_state.hand:
            if card.suit == self._suit:
                card.face_down = False

    def debuff_card(self, card: Card) -> bool:
        return card.suit == self._suit


class DebuffFaceCards(BossEffect):
    """All face cards (J, Q, K) are debuffed."""
    name = "The Plant"
    description = "All face cards are debuffed"

    def apply(self, game_state: GameState) -> None:
        for card in game_state.deck.draw_pile + game_state.deck.discard_pile + game_state.hand:
            if card.is_face_card:
                card.face_down = True

    def remove(self, game_state: GameState) -> None:
        for card in game_state.deck.draw_pile + game_state.deck.discard_pile + game_state.hand:
            if card.is_face_card:
                card.face_down = False

    def debuff_card(self, card: Card) -> bool:
        return card.is_face_card


class TheNeedle(BossEffect):
    """Only 1 hand allowed."""
    name = "The Needle"
    description = "Only 1 hand allowed"

    def apply(self, game_state: GameState) -> None:
        game_state.hands_remaining = 1

    def remove(self, game_state: GameState) -> None:
        pass  # Hands are reset at next blind start


class TheWall(BossEffect):
    """Extra large blind (4x mult), no effect on cards."""
    name = "The Wall"
    description = "Extra large blind"

    def apply(self, game_state: GameState) -> None:
        # Wall's effect is built into its 4x mult, no card debuffs
        pass

    def remove(self, game_state: GameState) -> None:
        pass


class TheFlint(BossEffect):
    """Halves base chips and base mult (rounded)."""
    name = "The Flint"
    description = "Base Chips and Mult are halved"

    def apply(self, game_state: GameState) -> None:
        pass

    def remove(self, game_state: GameState) -> None:
        pass

    def modify_hand(self, mult: int, hand_chips: int) -> tuple[int, int, bool]:
        new_mult = max(int(math.floor(mult * 0.5 + 0.5)), 1)
        new_chips = max(int(math.floor(hand_chips * 0.5 + 0.5)), 0)
        return new_mult, new_chips, True


class TheHook(BossEffect):
    """Discards 2 random cards from hand each hand played."""
    name = "The Hook"
    description = "Discards 2 random cards per hand played"

    def apply(self, game_state: GameState) -> None:
        pass

    def remove(self, game_state: GameState) -> None:
        pass

    def press_play(self, game_state: GameState) -> None:
        # Discard up to 2 random cards from hand
        if len(game_state.hand) > 0:
            n_discard = min(2, len(game_state.hand))
            indices = game_state.rng.choice(
                len(game_state.hand), size=n_discard, replace=False
            )
            discarded = [game_state.hand[i] for i in sorted(indices, reverse=True)]
            for i in sorted(indices, reverse=True):
                game_state.hand.pop(i)
            game_state.deck.return_cards(discarded)


def _make_boss_effect(blind_def: BlindDef) -> BossEffect:
    """Create a BossEffect from a BlindDef."""
    if blind_def.debuff_suit:
        suit = _SUIT_MAP[blind_def.debuff_suit]
        return DebuffSuit(suit, blind_def.name, f"All {blind_def.debuff_suit} are debuffed")
    if blind_def.debuff_face:
        return DebuffFaceCards()
    if blind_def.name == "The Needle":
        return TheNeedle()
    if blind_def.name == "The Wall":
        return TheWall()
    if blind_def.name == "The Flint":
        return TheFlint()
    if blind_def.name == "The Hook":
        return TheHook()
    # Default: no special effect
    return TheWall()  # Placeholder with no card effects


# ---------------------------------------------------------------------------
# Blind manager
# ---------------------------------------------------------------------------

class BlindManager:
    """Manages blind progression through antes."""

    BLINDS_PER_ANTE = [BlindType.SMALL, BlindType.BIG, BlindType.BOSS]

    def __init__(self, num_antes: int = 8):
        self.num_antes = num_antes

    def get_blind_def(self, blind_type: BlindType, boss_def: BlindDef | None = None) -> BlindDef:
        """Get the blind definition for a blind type."""
        if blind_type == BlindType.SMALL:
            return SMALL_BLIND
        elif blind_type == BlindType.BIG:
            return BIG_BLIND
        else:
            return boss_def or BOSS_BLINDS[0]

    def get_score_target(self, ante: int, blind_def: BlindDef) -> int:
        """Calculate score target: get_blind_amount(ante) * blind.mult.

        Matches Lua: self.chips = get_blind_amount(ante) * self.mult * ante_scaling
        We use ante_scaling=1 (default).
        """
        return int(get_blind_amount(ante) * blind_def.mult)

    def choose_boss(self, ante: int, rng: np.random.Generator) -> BlindDef:
        """Select a random boss blind for this ante.

        In Lua, boss selection is filtered by min/max ante.
        We simplify: any boss can appear at any ante.
        """
        idx = rng.integers(0, len(BOSS_BLINDS))
        return BOSS_BLINDS[idx]

    def get_boss_effect(self, boss_def: BlindDef) -> BossEffect:
        """Create the BossEffect for a given boss blind definition."""
        return _make_boss_effect(boss_def)

    def get_blind_sequence(self) -> list[tuple[int, BlindType]]:
        """Return the full sequence of (ante, blind_type) pairs for a game."""
        sequence = []
        for ante in range(1, self.num_antes + 1):
            for blind_type in self.BLINDS_PER_ANTE:
                sequence.append((ante, blind_type))
        return sequence

    @property
    def total_blinds(self) -> int:
        return self.num_antes * len(self.BLINDS_PER_ANTE)
