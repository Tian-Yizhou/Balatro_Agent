"""Vouchers — persistent run-wide upgrades purchased from the shop.

Reference: Balatro Lua source — game.lua P_CENTERS.v_* entries.

Architecture
------------
Mirrors :mod:`balatro_gym.core.back` / :mod:`balatro_gym.core.stake`:
declarative :class:`VoucherEffects` dataclass for static modifiers, plus
``on_redeem`` hook for one-time effects at purchase time.

Vouchers are **permanent** — once redeemed, the effect persists for the
rest of the run and the voucher cannot be sold or re-rolled. Each shop
visit shows one voucher slot drawn from the unredeemed pool.

Starter vouchers shipped (5 of 32)
----------------------------------
- Overstock (+1 shop joker slot)         — shop_slots_delta
- Clearance Sale (25% discount)          — shop_cost_multiplier
- Hone (foil/holo/poly 2× more often)    — edition_rate_multiplier
- Reroll Surplus (-$2 reroll cost)       — reroll_base_cost_delta
- Grabber (+1 hand per round)            — hands_per_round_delta (one-time)

Each exercises a distinct effect kind to validate the scaffolding.

Deferred to bulk-fill
---------------------
The 27 remaining vouchers (Crystal Ball, Telescope, Seed Money, etc.) follow
the same pattern — most are one-line VoucherEffects, a few need new fields
(tarot_rate, spectral_rate, playing_card_rate, ...). The 16 tier-2 vouchers
(Overstock Plus etc.) also need a prereq-check on top of the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from balatro_gym.core.game_state import GameState


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VoucherInfo:
    id: str               # e.g. "v_overstock_norm"
    name: str             # e.g. "Overstock"
    description: str
    cost: int = 10        # All base vouchers cost $10 in Lua.


@dataclass(frozen=True)
class VoucherEffects:
    """Static effects of redeeming this voucher.

    Effects fall into two categories:

    1. **Passive shop modifiers** — applied to ``state.shop`` at redeem
       time and persist through future shop visits (Overstock, Clearance
       Sale, Hone, Reroll Surplus).
    2. **One-time game-state deltas** — applied immediately to GameState
       fields like ``hands_per_round`` (Grabber).

    The redemption logic in :meth:`GameState.redeem_voucher` reads these
    fields and mutates the relevant targets.
    """
    # Passive shop modifiers
    shop_slots_delta: int = 0
    shop_cost_multiplier: float = 1.0       # 0.75 = 25% off
    edition_rate_multiplier: float = 1.0    # 1.0 = unchanged
    reroll_base_cost_delta: int = 0         # negative = cheaper

    # One-time game-state deltas (applied once at redeem)
    hands_per_round_delta: int = 0
    discards_per_round_delta: int = 0
    hand_size_delta: int = 0
    consumable_slots_delta: int = 0
    starting_money_delta: int = 0


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseVoucher:
    """Base class for vouchers. Subclasses declare EFFECTS."""

    INFO: ClassVar[VoucherInfo]
    EFFECTS: ClassVar[VoucherEffects] = VoucherEffects()

    def on_redeem(self, state: "GameState") -> None:
        """Hook for additional one-time effects beyond static EFFECTS.

        Default: no-op. Override for vouchers whose effect doesn't fit a
        simple static delta (e.g., future Magic Trick which unlocks
        playing-card shop rolls).
        """
        pass

    def __repr__(self) -> str:
        return f"{self.INFO.name} ({self.INFO.id})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_VOUCHER_REGISTRY: dict[str, type[BaseVoucher]] = {}


def register_voucher(cls: type[BaseVoucher]) -> type[BaseVoucher]:
    """Class decorator to register a voucher in the global registry."""
    _VOUCHER_REGISTRY[cls.INFO.id] = cls
    return cls


def get_voucher_class(voucher_id: str) -> type[BaseVoucher]:
    """Get voucher class by ID. Raises KeyError if not found."""
    return _VOUCHER_REGISTRY[voucher_id]


def get_all_voucher_ids() -> list[str]:
    """Return all registered voucher IDs."""
    return list(_VOUCHER_REGISTRY.keys())


def create_voucher(voucher_id: str) -> BaseVoucher:
    """Create a voucher instance by ID."""
    return _VOUCHER_REGISTRY[voucher_id]()


# ===========================================================================
# Starter vouchers
# ===========================================================================

@register_voucher
class Overstock(BaseVoucher):
    """Lua: v_overstock_norm. +1 card slot available in shop."""
    INFO = VoucherInfo(
        id="v_overstock_norm",
        name="Overstock",
        description="+1 card slot available in shop",
    )
    EFFECTS = VoucherEffects(shop_slots_delta=1)


@register_voucher
class ClearanceSale(BaseVoucher):
    """Lua: v_clearance_sale, config = {extra = 25}. 25% off all shop prices."""
    INFO = VoucherInfo(
        id="v_clearance_sale",
        name="Clearance Sale",
        description="All cards in shop are 25% off",
    )
    EFFECTS = VoucherEffects(shop_cost_multiplier=0.75)


@register_voucher
class Hone(BaseVoucher):
    """Lua: v_hone, config = {extra = 2}. Foil/Holo/Polychrome appear 2× more often."""
    INFO = VoucherInfo(
        id="v_hone",
        name="Hone",
        description="Foil, Holographic, and Polychrome cards appear 2× more often",
    )
    EFFECTS = VoucherEffects(edition_rate_multiplier=2.0)


@register_voucher
class RerollSurplus(BaseVoucher):
    """Lua: v_reroll_surplus, config = {extra = 2}. Rerolls cost $2 less."""
    INFO = VoucherInfo(
        id="v_reroll_surplus",
        name="Reroll Surplus",
        description="Rerolls cost $2 less",
    )
    EFFECTS = VoucherEffects(reroll_base_cost_delta=-2)


@register_voucher
class Grabber(BaseVoucher):
    """Lua: v_grabber, config = {extra = 1}. +1 hand per round."""
    INFO = VoucherInfo(
        id="v_grabber",
        name="Grabber",
        description="+1 hand per round",
    )
    EFFECTS = VoucherEffects(hands_per_round_delta=1)
