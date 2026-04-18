"""Shop mechanics: offerings, buying, selling, and rerolling.

Supports both joker and consumable offerings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from balatro_gym.core.joker import BaseJoker, create_joker, get_joker_class
from balatro_gym.core.consumable import (
    BaseConsumable, ConsumableType, create_consumable, get_consumables_by_type,
)


@dataclass
class ShopOffering:
    """A single item available in the shop.

    Holds either a joker or consumable, indicated by item_type.
    The `joker` field is kept for backward compatibility; for consumable
    offerings use the `consumable` field.
    """
    joker: BaseJoker | None = None
    consumable: BaseConsumable | None = None
    cost: int = 0
    sold: bool = False
    item_type: str = "joker"  # "joker" or "consumable"

    @property
    def name(self) -> str:
        if self.item_type == "joker" and self.joker is not None:
            return self.joker.INFO.name
        if self.item_type == "consumable" and self.consumable is not None:
            return self.consumable.INFO.name
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
    ):
        self.joker_pool = joker_pool
        self.consumable_pool = consumable_pool or []
        self.rng = rng
        self.num_slots = num_slots
        self.num_consumable_slots = num_consumable_slots
        self.reroll_cost = reroll_base_cost
        self.reroll_base_cost = reroll_base_cost
        self.offerings: list[ShopOffering] = []

    def generate_offerings(self) -> None:
        """Populate the shop with random jokers and consumables from pools."""
        self.offerings = []

        # Joker slots
        if self.joker_pool:
            for _ in range(self.num_slots):
                joker_id = self.joker_pool[
                    int(self.rng.integers(0, len(self.joker_pool)))
                ]
                joker = create_joker(joker_id)
                self.offerings.append(ShopOffering(
                    joker=joker, cost=joker.INFO.cost, item_type="joker",
                ))

        # Consumable slots
        if self.consumable_pool:
            for _ in range(self.num_consumable_slots):
                cid = self.consumable_pool[
                    int(self.rng.integers(0, len(self.consumable_pool)))
                ]
                consumable = create_consumable(cid)
                self.offerings.append(ShopOffering(
                    consumable=consumable, cost=consumable.INFO.cost,
                    item_type="consumable",
                ))

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

    def buy_item(
        self,
        slot_index: int,
        player_money: int,
        player_jokers: list[BaseJoker],
        max_jokers: int,
        player_consumables: list[BaseConsumable] | None = None,
        max_consumables: int = 2,
    ) -> tuple[BaseJoker | BaseConsumable | None, int]:
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

        # Re-generate offerings
        self.offerings = []
        if self.joker_pool:
            for _ in range(self.num_slots):
                joker_id = self.joker_pool[
                    int(self.rng.integers(0, len(self.joker_pool)))
                ]
                joker = create_joker(joker_id)
                self.offerings.append(ShopOffering(
                    joker=joker, cost=joker.INFO.cost, item_type="joker",
                ))

        if self.consumable_pool:
            for _ in range(self.num_consumable_slots):
                cid = self.consumable_pool[
                    int(self.rng.integers(0, len(self.consumable_pool)))
                ]
                consumable = create_consumable(cid)
                self.offerings.append(ShopOffering(
                    consumable=consumable, cost=consumable.INFO.cost,
                    item_type="consumable",
                ))

        return True, remaining
