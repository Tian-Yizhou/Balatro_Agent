"""Stakes — global difficulty tiers applied on top of a run.

Reference: Balatro Lua source — game.lua P_CENTERS.stake_* and the
``start_run`` block (game.lua:2050-2059) which applies cumulative
modifiers per stake level::

    if stake >= 2 then no_blind_reward.Small = true            -- Red
    if stake >= 3 then modifiers.scaling = 2                   -- Green
    if stake >= 4 then enable_eternals_in_shop = true          -- Black
    if stake >= 5 then starting_discards -= 1                  -- Blue
    if stake >= 6 then modifiers.scaling = 3                   -- Purple
    if stake >= 7 then enable_perishables_in_shop = true       -- Orange
    if stake >= 8 then enable_rentals_in_shop = true           -- Gold

Architecture mirrors :mod:`balatro_gym.core.back` — declarative
:class:`StakeModifiers` dataclass for the common case, plus future
runtime hooks via subclass methods if needed.

Each stake declares its **full effective** modifiers (cumulative). e.g.
``BlueStake.MODIFIERS`` already includes Red + Green's effects. This is
verbose but explicit; for 5 starters it's tractable. When all 8 stakes
land, we can switch to a "declare only new effects + combine at lookup"
model if churn becomes a problem.

Starter stakes shipped (5 of 8)
-------------------------------
- White (L1)  — no-op baseline.
- Red (L2)    — Small Blind gives no reward money.
- Green (L3)  — Score scaling tier 2 (faster blind targets).
- Blue (L5)   — -1 starting discard (cumulative).
- Purple (L6) — Score scaling tier 3 (cumulative).

Deferred to a later phase
-------------------------
- Black (L4)  — Eternal jokers in shop. Needs joker eternal flag.
- Orange (L7) — Perishable jokers in shop. Needs joker perishable flag.
- Gold (L8)   — Rental jokers in shop. Needs joker rental flag + $3/round drain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StakeInfo:
    id: str
    name: str
    level: int        # 1-8 (ordering / cumulative threshold)
    description: str


@dataclass(frozen=True)
class StakeModifiers:
    """Effective (cumulative) modifiers for a stake level.

    Every stake's MODIFIERS reflects the **total** effect at that level,
    including all lower-stake effects rolled in.
    """
    # Economy
    no_small_blind_money: bool = False        # Red stake

    # Blind scoring
    score_scaling_tier: int = 1               # Green: 2, Purple: 3

    # Starting resources
    starting_discards_delta: int = 0          # Blue stake: -1

    # Joker stake-stickers in shop (flags only — joker side lands later).
    eternal_jokers_in_shop: bool = False      # Black
    perishable_jokers_in_shop: bool = False   # Orange
    rental_jokers_in_shop: bool = False       # Gold


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseStake:
    """Base class for stakes. Subclasses declare MODIFIERS."""

    INFO: ClassVar[StakeInfo]
    MODIFIERS: ClassVar[StakeModifiers] = StakeModifiers()

    def __repr__(self) -> str:
        return f"{self.INFO.name} (L{self.INFO.level})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_STAKE_REGISTRY: dict[str, type[BaseStake]] = {}


def register_stake(cls: type[BaseStake]) -> type[BaseStake]:
    """Class decorator to register a stake in the global registry."""
    _STAKE_REGISTRY[cls.INFO.id] = cls
    return cls


def get_stake_class(stake_id: str) -> type[BaseStake]:
    """Get stake class by ID. Raises KeyError if not found."""
    return _STAKE_REGISTRY[stake_id]


def get_all_stake_ids() -> list[str]:
    """Return all registered stake IDs."""
    return list(_STAKE_REGISTRY.keys())


def create_stake(stake_id: str) -> BaseStake:
    """Create a stake instance by ID."""
    return _STAKE_REGISTRY[stake_id]()


# ===========================================================================
# Starter stakes
# ===========================================================================

@register_stake
class WhiteStake(BaseStake):
    """Level 1 — base difficulty. No modifiers."""
    INFO = StakeInfo(
        id="stake_white",
        name="White Stake",
        level=1,
        description="Base Difficulty",
    )
    # MODIFIERS = default (all zeros / False) → no effect.


@register_stake
class RedStake(BaseStake):
    """Level 2 — Small Blind gives no reward money."""
    INFO = StakeInfo(
        id="stake_red",
        name="Red Stake",
        level=2,
        description="Small Blind gives no reward money",
    )
    MODIFIERS = StakeModifiers(
        no_small_blind_money=True,
    )


@register_stake
class GreenStake(BaseStake):
    """Level 3 — Required score scales faster for each Ante (tier 2)."""
    INFO = StakeInfo(
        id="stake_green",
        name="Green Stake",
        level=3,
        description="Required score scales faster (tier 2)",
    )
    MODIFIERS = StakeModifiers(
        no_small_blind_money=True,            # from Red
        score_scaling_tier=2,                 # new
    )


@register_stake
class BlueStake(BaseStake):
    """Level 5 — -1 starting discard, cumulative.

    Skips Black (L4 — Eternal jokers) for now; the eternal-joker mechanic
    needs joker-sticker support not yet built.
    """
    INFO = StakeInfo(
        id="stake_blue",
        name="Blue Stake",
        level=5,
        description="-1 Discard (applies all previous Stakes)",
    )
    MODIFIERS = StakeModifiers(
        no_small_blind_money=True,            # from Red
        score_scaling_tier=2,                 # from Green
        starting_discards_delta=-1,           # new
    )


@register_stake
class PurpleStake(BaseStake):
    """Level 6 — Score scaling tier 3, cumulative."""
    INFO = StakeInfo(
        id="stake_purple",
        name="Purple Stake",
        level=6,
        description="Required score scales even faster (tier 3, applies all previous Stakes)",
    )
    MODIFIERS = StakeModifiers(
        no_small_blind_money=True,            # from Red
        score_scaling_tier=3,                 # new (overrides Green's 2)
        starting_discards_delta=-1,           # from Blue
    )
