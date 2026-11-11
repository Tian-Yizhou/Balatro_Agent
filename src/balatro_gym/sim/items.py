"""Creating jokers and consumables, and their prices.

Source: card.lua ``Card:set_ability`` and ``Card:set_cost``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .content import CENTERS
from .state import Item, RunState, Scalar

__all__ = ["EDITION_COST", "initial_values", "new_item", "set_cost"]

#: Extra cost of an edition. Source: card.lua ``Card:set_cost``.
EDITION_COST = {"e_foil": 2, "e_holo": 3, "e_polychrome": 5, "e_negative": 5}


def initial_values(key: str) -> dict[str, Scalar]:
    """An item's starting values: its config, with nested tables flattened.

    ``{"extra": {"odds": 6}}`` becomes ``{"extra.odds": 6}``, and the game's
    ``Xmult`` config is stored as ``x_mult``, the name its ability table uses.
    """
    values: dict[str, Scalar] = {}
    for name, item in CENTERS[key].config.items():
        target = "x_mult" if name == "Xmult" else name
        if isinstance(item, Mapping):
            nested: Mapping[str, Any] = item  # pyright: ignore[reportUnknownVariableType]
            for sub, sub_value in nested.items():
                values[f"{target}.{sub}"] = sub_value
        else:
            values[target] = item
    return values


def new_item(state: RunState, key: str, *, edition: str | None = None) -> Item:
    """A new joker or consumable with a fresh uid, priced for the current run."""
    item = Item(uid=state.new_uid(), key=key, edition=edition, values=initial_values(key))
    set_cost(state, item)
    return item


def set_cost(state: RunState, item: Item) -> None:
    """Set an item's buying price and selling value from the run's economy."""
    extra = state.inflation + EDITION_COST.get(item.edition or "", 0)
    base = CENTERS[item.key].cost
    item.cost = max(1, math.floor((base + extra + 0.5) * (100 - state.discount_percent) / 100))
    if item.rental:
        item.cost = 1
    extra_value = item.values.get("extra_value", 0)
    bonus = extra_value if isinstance(extra_value, int) else 0
    item.sell_value = max(1, math.floor(item.cost / 2)) + bonus
