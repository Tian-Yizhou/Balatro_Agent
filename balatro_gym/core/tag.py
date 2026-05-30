"""Tags — rewards earned by skipping Small or Big blinds.

Reference: Balatro Lua source — game.lua P_CENTERS.tag_* and tag.lua.

Mechanic
--------
At a Small or Big blind, the player may choose to skip it. Skipping yields
no money/score but awards a tag drawn from the unredeemed pool. Tags
trigger at various points: immediately, on entering the next shop, at the
start of the next round, or after beating the next boss.

Boss blinds cannot be skipped.

Architecture
------------
Mirrors :mod:`balatro_gym.core.voucher` / :mod:`stake` — class-per-tag
with declarative INFO + override hooks. The four hooks correspond to the
four timing points; each returns ``True`` if the tag is fully consumed
(should be removed from the active list).

Starter tags shipped (5 of 24)
------------------------------
- Investment Tag  — after beating next boss: +$25.
- Handy Tag       — immediately: +$1 per hand played so far in the run.
- Foil Tag        — next shop: first joker offering becomes Foil edition.
- Voucher Tag     — next shop: an extra voucher offering is added.
- Juggle Tag      — at next round start: +3 hand size for that round only.

Each starter exercises a distinct hook timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from balatro_gym.core.card import Edition

if TYPE_CHECKING:
    from balatro_gym.core.blind import BlindType
    from balatro_gym.core.game_state import GameState


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TagInfo:
    id: str
    name: str
    description: str


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseTag:
    """Base class for tags.

    Each hook returns ``True`` if the tag is fully consumed (and should be
    removed from ``state.active_tags``). Returning ``False`` keeps the tag
    around for future hooks.

    Default implementations do nothing and return ``False`` — subclasses
    override only the hook(s) relevant to their effect.
    """

    INFO: ClassVar[TagInfo]

    def on_award(self, state: "GameState") -> bool:
        """Called when the player claims this tag by skipping a blind.

        Use for ``immediate``-type tags (Handy, Garbage, Top-up, ...).
        Default: not consumed; the tag is parked in ``state.active_tags``.
        """
        return False

    def on_shop_enter(self, state: "GameState") -> bool:
        """Called once when the player enters the next shop.

        Use for ``store_*`` and ``voucher_add`` tags (Foil, Voucher, ...).
        """
        return False

    def on_round_start(self, state: "GameState") -> bool:
        """Called at ``_start_blind`` for the next round after award.

        Use for ``round_start_bonus`` tags (Juggle).
        """
        return False

    def on_blind_beaten(self, state: "GameState",
                        blind_type: "BlindType") -> bool:
        """Called inside ``_end_blind`` for each beaten blind.

        Use for ``eval`` tags whose payout is tied to a specific blind type
        (Investment fires on Boss).
        """
        return False

    def __repr__(self) -> str:
        return f"{self.INFO.name} ({self.INFO.id})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TAG_REGISTRY: dict[str, type[BaseTag]] = {}


def register_tag(cls: type[BaseTag]) -> type[BaseTag]:
    """Class decorator to register a tag in the global registry."""
    _TAG_REGISTRY[cls.INFO.id] = cls
    return cls


def get_tag_class(tag_id: str) -> type[BaseTag]:
    """Get tag class by ID. Raises KeyError if not found."""
    return _TAG_REGISTRY[tag_id]


def get_all_tag_ids() -> list[str]:
    """Return all registered tag IDs."""
    return list(_TAG_REGISTRY.keys())


def create_tag(tag_id: str) -> BaseTag:
    """Create a tag instance by ID."""
    return _TAG_REGISTRY[tag_id]()


# ===========================================================================
# Starter tags
# ===========================================================================

@register_tag
class InvestmentTag(BaseTag):
    """Lua: tag_investment, config = {dollars = 25, type = 'eval'}.

    After beating the next boss blind, gain $25.
    """
    INFO = TagInfo(
        id="tag_investment",
        name="Investment Tag",
        description="After beating the next boss blind, gain $25",
    )

    def on_blind_beaten(self, state, blind_type) -> bool:
        # Local import to avoid a top-level cycle.
        from balatro_gym.core.blind import BlindType
        if blind_type == BlindType.BOSS:
            state.money += 25
            return True
        return False


@register_tag
class HandyTag(BaseTag):
    """Lua: tag_handy, config = {dollars_per_hand = 1, type = 'immediate'}.

    Immediately gain $1 per hand played in this run.
    """
    INFO = TagInfo(
        id="tag_handy",
        name="Handy Tag",
        description="Immediately gain $1 per hand played in this run",
    )

    def on_award(self, state) -> bool:
        state.money += state.total_hands_played
        return True   # immediate, no further hooks


@register_tag
class FoilTag(BaseTag):
    """Lua: tag_foil, config = {edition = 'foil', type = 'store_joker_modify'}.

    Next shop: the first joker offering has Foil edition.
    """
    INFO = TagInfo(
        id="tag_foil",
        name="Foil Tag",
        description="Next shop: first Joker offering is Foil",
    )

    def on_shop_enter(self, state) -> bool:
        # Find the first joker offering and apply Foil edition + cost bump.
        for offering in state.shop.offerings:
            if offering.item_type == "joker" and offering.joker is not None:
                offering.joker.edition = Edition.FOIL
                offering.cost = state.shop._apply_cost_mult(
                    offering.joker.cost_with_edition
                )
                return True
        # No joker offering in this shop — silently expire.
        return True


@register_tag
class VoucherTag(BaseTag):
    """Lua: tag_voucher, config = {type = 'voucher_add'}.

    Next shop: an extra voucher offering is added.
    """
    INFO = TagInfo(
        id="tag_voucher",
        name="Voucher Tag",
        description="Next shop: an extra Voucher is added",
    )

    def on_shop_enter(self, state) -> bool:
        bonus = state.shop._spawn_voucher_offering()
        if bonus is not None:
            state.shop.offerings.append(bonus)
        return True


@register_tag
class JuggleTag(BaseTag):
    """Lua: tag_juggle, config = {h_size = 3, type = 'round_start_bonus'}.

    For the next round only: +3 hand size.
    """
    INFO = TagInfo(
        id="tag_juggle",
        name="Juggle Tag",
        description="+3 hand size for the next round",
    )

    def on_round_start(self, state) -> bool:
        state.current_round_hand_size_bonus += 3
        return True   # one-shot
