"""Shop mechanics: offerings, buying, selling, and rerolling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from balatro_gym.core.joker import BaseJoker, create_joker, get_joker_class


@dataclass
class ShopOffering:
    """A single item available in the shop."""
    joker: BaseJoker
    cost: int
    sold: bool = False


class Shop:
    """The between-rounds shop where players buy/sell jokers."""

    def __init__(
        self,
        joker_pool: list[str],
        rng: np.random.Generator,
        num_slots: int = 2,
        reroll_base_cost: int = 5,
    ):
        self.joker_pool = joker_pool
        self.rng = rng
        self.num_slots = num_slots
        self.reroll_cost = reroll_base_cost
        self.reroll_base_cost = reroll_base_cost
        self.offerings: list[ShopOffering] = []

    def generate_offerings(self) -> None:
        """Populate the shop with random jokers from the pool."""
        self.offerings = []
        if not self.joker_pool:
            return

        for _ in range(self.num_slots):
            joker_id = self.joker_pool[self.rng.integers(0, len(self.joker_pool))]
            joker = create_joker(joker_id)
            self.offerings.append(ShopOffering(joker=joker, cost=joker.INFO.cost))

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
            (None, player_money) if purchase fails (can't afford, slot empty, full).
        """
        if slot_index < 0 or slot_index >= len(self.offerings):
            return None, player_money

        offering = self.offerings[slot_index]
        if offering.sold:
            return None, player_money

        if player_money < offering.cost:
            return None, player_money

        if len(player_jokers) >= max_jokers:
            return None, player_money

        offering.sold = True
        return offering.joker, player_money - offering.cost

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
        for _ in range(self.num_slots):
            joker_id = self.joker_pool[self.rng.integers(0, len(self.joker_pool))]
            joker = create_joker(joker_id)
            self.offerings.append(ShopOffering(joker=joker, cost=joker.INFO.cost))

        return True, remaining
