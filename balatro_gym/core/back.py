"""Deck Backs (deck variants).

Reference: Balatro Lua source — back.lua. Each back modifies the starting
state and/or per-round rules of a run.

Architecture
------------
Most deck effects are static deltas to the starting GameConfig values
(e.g. Red Deck = +1 discard per round). These are declared via the
``MODIFIERS`` class variable using the :class:`BackModifiers` dataclass.

A handful of backs need runtime hooks (e.g. Green Deck replaces interest
with per-unused-hand/discard bonuses). Subclasses override the hook
methods for these cases. Defaults preserve vanilla behavior.

Future expansion points already in place (no-op by default):
- ``apply_starting_items(state)`` for decks that grant starting vouchers,
  jokers, or consumables (Magic, Nebula, Ghost, Zodiac).
- ``modify_deck(deck)`` for decks that alter the 52-card composition
  (Abandoned: no face cards; Checkered: suit conversion; Erratic: random).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from balatro_gym.core.card import Deck
    from balatro_gym.core.game_state import GameState


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BackInfo:
    id: str           # e.g. "b_red"
    name: str         # e.g. "Red Deck"
    description: str


@dataclass(frozen=True)
class BackModifiers:
    """Static config-level deltas applied once at game start.

    All fields are signed deltas vs the GameConfig defaults. A back that
    declares ``BackModifiers(discards_per_round_delta=1)`` adds +1 discard
    per round to whatever the GameConfig provides.
    """
    starting_money_delta: int = 0
    hands_per_round_delta: int = 0
    discards_per_round_delta: int = 0
    hand_size_delta: int = 0
    max_jokers_delta: int = 0
    consumable_slots_delta: int = 0
    reroll_base_cost_delta: int = 0   # negative = cheaper rerolls


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseBack:
    """Base class for deck backs.

    Subclasses declare static modifiers via ``MODIFIERS`` and override
    hook methods for any runtime effects.
    """

    INFO: ClassVar[BackInfo]
    MODIFIERS: ClassVar[BackModifiers] = BackModifiers()

    # -- Runtime hooks. Defaults preserve vanilla. ------------------------

    def disables_interest(self) -> bool:
        """If True, end-of-round interest is suppressed."""
        return False

    def money_per_unused_hand(self) -> int:
        """Dollars per unused hand at round end. Vanilla = $1."""
        return 1

    def money_per_unused_discard(self) -> int:
        """Dollars per unused discard at round end. Vanilla = $0."""
        return 0

    # -- Future expansion points (no-op for now). -------------------------
    # Implemented in later phases (Magic/Nebula/Ghost/Zodiac decks need
    # starting items; Abandoned/Checkered/Erratic need deck modification).

    def apply_starting_items(self, state: GameState) -> None:
        """Add starting jokers/vouchers/consumables. Default: no-op."""
        pass

    def modify_deck(self, deck: Deck) -> None:
        """Mutate the initial 52-card deck. Default: no-op."""
        pass

    def __repr__(self) -> str:
        return f"{self.INFO.name} ({self.INFO.id})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BACK_REGISTRY: dict[str, type[BaseBack]] = {}


def register_back(cls: type[BaseBack]) -> type[BaseBack]:
    """Class decorator to register a deck back in the global registry."""
    _BACK_REGISTRY[cls.INFO.id] = cls
    return cls


def get_back_class(back_id: str) -> type[BaseBack]:
    """Get back class by ID. Raises KeyError if not found."""
    return _BACK_REGISTRY[back_id]


def get_all_back_ids() -> list[str]:
    """Return all registered back IDs."""
    return list(_BACK_REGISTRY.keys())


def create_back(back_id: str) -> BaseBack:
    """Create a back instance by ID."""
    return _BACK_REGISTRY[back_id]()


# ===========================================================================
# Starter decks (5 examples — exercises every code path in the scaffolding)
# ===========================================================================

@register_back
class RedDeck(BaseBack):
    """Lua: b_red, config = {discards = 1}."""
    INFO = BackInfo(
        id="b_red",
        name="Red Deck",
        description="+1 discard per round",
    )
    MODIFIERS = BackModifiers(discards_per_round_delta=1)


@register_back
class BlueDeck(BaseBack):
    """Lua: b_blue, config = {hands = 1}."""
    INFO = BackInfo(
        id="b_blue",
        name="Blue Deck",
        description="+1 hand per round",
    )
    MODIFIERS = BackModifiers(hands_per_round_delta=1)


@register_back
class YellowDeck(BaseBack):
    """Lua: b_yellow, config = {dollars = 10}."""
    INFO = BackInfo(
        id="b_yellow",
        name="Yellow Deck",
        description="+$10 starting money",
    )
    MODIFIERS = BackModifiers(starting_money_delta=10)


@register_back
class BlackDeck(BaseBack):
    """Lua: b_black, config = {joker_slot = 1, hands = -1}."""
    INFO = BackInfo(
        id="b_black",
        name="Black Deck",
        description="+1 joker slot, -1 hand per round",
    )
    MODIFIERS = BackModifiers(max_jokers_delta=1, hands_per_round_delta=-1)


@register_back
class GreenDeck(BaseBack):
    """Lua: b_green, config = {extra_hand_bonus = 2, extra_discard_bonus = 1, no_interest = true}.

    Replaces end-of-round interest with per-unused-resource bonuses.
    """
    INFO = BackInfo(
        id="b_green",
        name="Green Deck",
        description=(
            "At end of round: $2 per unused Hand, $1 per unused Discard, "
            "no interest earned"
        ),
    )
    # No static modifiers — the effect is all runtime.

    def disables_interest(self) -> bool:
        return True

    def money_per_unused_hand(self) -> int:
        return 2

    def money_per_unused_discard(self) -> int:
        return 1
