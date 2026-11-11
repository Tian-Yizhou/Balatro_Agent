"""Creating cards: jokers, consumables and playing cards, and their prices.

Source: card.lua ``Card:set_ability`` and ``Card:set_cost``;
functions/common_events.lua ``create_playing_card``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .content import CENTERS
from .state import Item, PlayingCard, RunState, Scalar, Suit

__all__ = [
    "EDITION_COST",
    "copy_item",
    "copy_playing_card",
    "initial_values",
    "new_item",
    "new_playing_card",
    "set_cost",
]

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
    # Counters that Card:set_ability starts beyond the item's config.
    if key == "j_caino":
        values["caino_xmult"] = 1
    elif key == "j_yorick":
        values["yorick_discards"] = values["extra.discards"]
    return values


def new_item(state: RunState, key: str, *, edition: str | None = None) -> Item:
    """A new joker or consumable with a fresh uid, priced for the current run."""
    item = Item(uid=state.new_uid(), key=key, edition=edition, values=initial_values(key))
    set_cost(state, item)
    return item


def new_playing_card(
    state: RunState, rank: int, suit: Suit, enhancement: str | None = None
) -> PlayingCard:
    """A new playing card with a fresh uid; the caller places it in an area."""
    return PlayingCard(uid=state.new_uid(), rank=rank, suit=suit, enhancement=enhancement)


def copy_item(state: RunState, item: Item, *, strip_edition: bool = False) -> Item:
    """A copy of a joker or consumable with a fresh uid, as ``copy_card`` makes.

    The copy keeps the original's values and stickers.
    """
    copy = Item(
        uid=state.new_uid(),
        key=item.key,
        edition=None if strip_edition else item.edition,
        eternal=item.eternal,
        perishable=item.perishable,
        perish_tally=item.perish_tally,
        rental=item.rental,
        debuffed=item.debuffed,
        values=dict(item.values),
    )
    set_cost(state, copy)
    return copy


def copy_playing_card(state: RunState, card: PlayingCard) -> PlayingCard:
    """A copy of a playing card with a fresh uid, as ``copy_card`` makes."""
    return PlayingCard(
        uid=state.new_uid(),
        rank=card.rank,
        suit=card.suit,
        enhancement=card.enhancement,
        edition=card.edition,
        seal=card.seal,
        perma_bonus=card.perma_bonus,
        debuffed=card.debuffed,
        played_this_ante=card.played_this_ante,
    )


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
