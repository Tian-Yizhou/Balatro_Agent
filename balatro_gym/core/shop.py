"""Shop mechanics: offerings, buying, selling, and rerolling.

Supports both joker and consumable offerings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from balatro_gym.core.card import Edition
from balatro_gym.core.joker import BaseJoker, create_joker, get_joker_class
from balatro_gym.core.consumable import (
    BaseConsumable, ConsumableType, create_consumable, get_consumables_by_type,
)
from balatro_gym.core.voucher import BaseVoucher, create_voucher


# Lua: functions/common_events.lua poll_edition (default _mod=1, no_neg=False)
# Probability table — read cumulatively from 1.0 downward:
#   roll > 0.997  → Negative   (0.3%, only if Negative allowed)
#   roll > 0.994  → Polychrome (0.3%)
#   roll > 0.98   → Holographic (1.4%)
#   roll > 0.96   → Foil       (2%)
#   else          → no edition (96%)
def poll_edition(
    rng: np.random.Generator,
    edition_rate: float = 1.0,
    allow_negative: bool = True,
) -> Edition | None:
    """Roll a random edition matching Lua's poll_edition probabilities.

    Args:
        rng: numpy Generator for the roll.
        edition_rate: multiplier on each edition's probability band (default 1).
        allow_negative: if False, Negative is skipped (used for consumable rolls
            in the base shop).

    Returns ``None`` ~96% of the time at default rates.
    """
    roll = float(rng.random())
    # Negative is rolled with a flat 0.3% (NOT scaled by edition_rate in Lua).
    if allow_negative and roll > 1 - 0.003:
        return Edition.NEGATIVE
    if roll > 1 - 0.006 * edition_rate:
        return Edition.POLYCHROME
    if roll > 1 - 0.02 * edition_rate:
        return Edition.HOLO
    if roll > 1 - 0.04 * edition_rate:
        return Edition.FOIL
    return None


@dataclass
class ShopOffering:
    """A single item available in the shop.

    Holds a joker, consumable, or voucher; indicated by item_type.
    """
    joker: BaseJoker | None = None
    consumable: BaseConsumable | None = None
    voucher: BaseVoucher | None = None
    cost: int = 0
    sold: bool = False
    item_type: str = "joker"  # "joker" | "consumable" | "voucher"

    @property
    def name(self) -> str:
        if self.item_type == "joker" and self.joker is not None:
            return self.joker.INFO.name
        if self.item_type == "consumable" and self.consumable is not None:
            return self.consumable.INFO.name
        if self.item_type == "voucher" and self.voucher is not None:
            return self.voucher.INFO.name
        return "???"


class Shop:
    """The between-rounds shop where players buy/sell jokers and consumables."""

    def __init__(
        self,
        joker_pool: list[str],
        rng: np.random.Generator,
        num_slots: int = 2,
        reroll_base_cost: int = 5,
        consumable_pool: list[str] | None = None,
        num_consumable_slots: int = 1,
        edition_rate: float = 1.0,
        voucher_pool: list[str] | None = None,
        num_voucher_slots: int = 0,
        cost_multiplier: float = 1.0,
    ):
        self.joker_pool = joker_pool
        self.consumable_pool = consumable_pool or []
        self.voucher_pool = voucher_pool or []
        self.rng = rng
        self.num_slots = num_slots
        self.num_consumable_slots = num_consumable_slots
        self.num_voucher_slots = num_voucher_slots
        self.reroll_cost = reroll_base_cost
        self.reroll_base_cost = reroll_base_cost
        self.edition_rate = edition_rate
        self.cost_multiplier = cost_multiplier  # Clearance Sale etc.
        self.offerings: list[ShopOffering] = []

    def _apply_cost_mult(self, base_cost: int) -> int:
        """Apply the shop's global cost multiplier (Clearance Sale, etc.).

        Lua: ``math.max(1, math.floor((base_cost + extra + 0.5) * mult))``
        — minimum cost is 1.
        """
        return max(1, int(base_cost * self.cost_multiplier + 0.5))

    def _spawn_joker_offering(self) -> ShopOffering:
        """Spawn one joker offering with a possible edition roll.

        Edition probabilities follow Lua's ``poll_edition`` (~96% no edition).
        Cost includes the edition bump (Foil +$2, Holo +$3, Polychrome/Negative +$5)
        and the shop's global cost multiplier.
        """
        joker_id = self.joker_pool[
            int(self.rng.integers(0, len(self.joker_pool)))
        ]
        edition = poll_edition(self.rng, edition_rate=self.edition_rate, allow_negative=True)
        joker = create_joker(joker_id, edition=edition)
        return ShopOffering(
            joker=joker,
            cost=self._apply_cost_mult(joker.cost_with_edition),
            item_type="joker",
        )

    def _spawn_consumable_offering(self) -> ShopOffering:
        """Spawn one consumable offering. Base shop never rolls editions on
        consumables (Lua passes ``_no_neg=True`` and the other branches require
        special tags); we mirror that by always returning a base-edition consumable.
        """
        cid = self.consumable_pool[
            int(self.rng.integers(0, len(self.consumable_pool)))
        ]
        consumable = create_consumable(cid)
        return ShopOffering(
            consumable=consumable,
            cost=self._apply_cost_mult(consumable.INFO.cost),
            item_type="consumable",
        )

    def _spawn_voucher_offering(self) -> ShopOffering | None:
        """Spawn one voucher offering, or None if the pool is exhausted.

        Vouchers are unique per run — the pool here should already exclude
        already-redeemed vouchers (handled by the caller).
        """
        if not self.voucher_pool:
            return None
        vid = self.voucher_pool[
            int(self.rng.integers(0, len(self.voucher_pool)))
        ]
        voucher = create_voucher(vid)
        return ShopOffering(
            voucher=voucher,
            cost=self._apply_cost_mult(voucher.INFO.cost),
            item_type="voucher",
        )

    def generate_offerings(self) -> None:
        """Populate the shop with random jokers, vouchers, and consumables."""
        self.offerings = []
        if self.joker_pool:
            for _ in range(self.num_slots):
                self.offerings.append(self._spawn_joker_offering())
        for _ in range(self.num_voucher_slots):
            v = self._spawn_voucher_offering()
            if v is not None:
                self.offerings.append(v)
        if self.consumable_pool:
            for _ in range(self.num_consumable_slots):
                self.offerings.append(self._spawn_consumable_offering())
        # Reset reroll cost each time we enter shop
        self.reroll_cost = self.reroll_base_cost

    def get_available_offerings(self) -> list[tuple[int, ShopOffering]]:
        """Return (index, offering) pairs for items not yet sold."""
        return [(i, o) for i, o in enumerate(self.offerings) if not o.sold]

    def buy_joker(
        self,
        slot_index: int,
        player_money: int,
        player_jokers: list[BaseJoker],
        max_jokers: int,
    ) -> tuple[BaseJoker | None, int]:
        """Attempt to buy a joker from the shop.

        Returns:
            (joker, remaining_money) if purchase succeeds.
            (None, player_money) if purchase fails.
        """
        if slot_index < 0 or slot_index >= len(self.offerings):
            return None, player_money

        offering = self.offerings[slot_index]
        if offering.sold or offering.item_type != "joker":
            return None, player_money

        if player_money < offering.cost:
            return None, player_money

        if len(player_jokers) >= max_jokers:
            return None, player_money

        offering.sold = True
        return offering.joker, player_money - offering.cost

    def buy_consumable(
        self,
        slot_index: int,
        player_money: int,
        player_consumables: list[BaseConsumable],
        max_consumables: int,
    ) -> tuple[BaseConsumable | None, int]:
        """Attempt to buy a consumable from the shop.

        Returns:
            (consumable, remaining_money) if purchase succeeds.
            (None, player_money) if purchase fails.
        """
        if slot_index < 0 or slot_index >= len(self.offerings):
            return None, player_money

        offering = self.offerings[slot_index]
        if offering.sold or offering.item_type != "consumable":
            return None, player_money

        if player_money < offering.cost:
            return None, player_money

        if len(player_consumables) >= max_consumables:
            return None, player_money

        offering.sold = True
        return offering.consumable, player_money - offering.cost

    def buy_voucher(
        self,
        slot_index: int,
        player_money: int,
    ) -> tuple[BaseVoucher | None, int]:
        """Attempt to buy a voucher. Vouchers have no slot limit (always one).

        Returns:
            (voucher, remaining_money) if purchase succeeds.
            (None, player_money) if purchase fails.
        """
        if slot_index < 0 or slot_index >= len(self.offerings):
            return None, player_money
        offering = self.offerings[slot_index]
        if offering.sold or offering.item_type != "voucher":
            return None, player_money
        if player_money < offering.cost:
            return None, player_money
        offering.sold = True
        return offering.voucher, player_money - offering.cost

    def buy_item(
        self,
        slot_index: int,
        player_money: int,
        player_jokers: list[BaseJoker],
        max_jokers: int,
        player_consumables: list[BaseConsumable] | None = None,
        max_consumables: int = 2,
    ) -> tuple[BaseJoker | BaseConsumable | BaseVoucher | None, int]:
        """Buy any item by slot index, routing to the correct buy method.

        Returns:
            (item, remaining_money) or (None, player_money).
        """
        if slot_index < 0 or slot_index >= len(self.offerings):
            return None, player_money

        offering = self.offerings[slot_index]
        if offering.item_type == "joker":
            return self.buy_joker(slot_index, player_money,
                                  player_jokers, max_jokers)
        elif offering.item_type == "consumable":
            return self.buy_consumable(
                slot_index, player_money,
                player_consumables or [], max_consumables,
            )
        elif offering.item_type == "voucher":
            return self.buy_voucher(slot_index, player_money)
        return None, player_money

    def sell_value(self, joker: BaseJoker) -> int:
        """Calculate the sell value of a joker (half its cost, minimum 1)."""
        return max(1, joker.INFO.cost // 2)

    def reroll(self, player_money: int) -> tuple[bool, int]:
        """Reroll the shop offerings.

        Returns:
            (success, remaining_money).
        """
        if player_money < self.reroll_cost:
            return False, player_money

        remaining = player_money - self.reroll_cost
        self.reroll_cost += 1  # Rerolls get more expensive

        # Re-generate offerings (same edition rolls + cost adjustments).
        # Note: rerolling does NOT re-spawn the voucher slot — in Balatro
        # the voucher persists across rerolls within a shop visit.
        existing_vouchers = [
            o for o in self.offerings if o.item_type == "voucher"
        ]
        self.offerings = []
        if self.joker_pool:
            for _ in range(self.num_slots):
                self.offerings.append(self._spawn_joker_offering())
        self.offerings.extend(existing_vouchers)
        if self.consumable_pool:
            for _ in range(self.num_consumable_slots):
                self.offerings.append(self._spawn_consumable_offering())

        return True, remaining
