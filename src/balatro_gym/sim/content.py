"""Item data extracted from the game: jokers, consumables, vouchers, packs and more.

The numbers come from ``data/content.json``, written by
``tools/extract_content/extract.py`` from the game's item definitions. This module
gives typed, read-only access to them.

Source: game.lua ``Game:init_item_prototypes``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType
from typing import Any

__all__ = [
    "CENTERS",
    "Center",
    "center",
    "config_value",
    "pool",
]


@dataclass(frozen=True, slots=True)
class Center:
    """The static definition of one item (the game calls these "centers")."""

    key: str
    set: str
    name: str
    order: int
    cost: int
    rarity: int
    config: Mapping[str, Any]
    unlocked: bool
    discovered: bool
    blueprint_compat: bool
    eternal_compat: bool
    perishable_compat: bool
    effect: str
    raw: Mapping[str, Any]

    @property
    def is_joker(self) -> bool:
        return self.set == "Joker"

    @property
    def is_consumable(self) -> bool:
        return bool(self.raw.get("consumeable"))


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        mapping: dict[str, Any] = value  # pyright: ignore[reportUnknownVariableType]
        return MappingProxyType({str(k): _freeze(v) for k, v in mapping.items()})
    if isinstance(value, list):
        sequence: list[Any] = value  # pyright: ignore[reportUnknownVariableType]
        return tuple(_freeze(v) for v in sequence)
    return value


def _load() -> dict[str, Any]:
    text = resources.files("balatro_gym.sim").joinpath("data", "content.json").read_text("utf-8")
    data: dict[str, Any] = json.loads(text)
    return data


_DATA = _load()


def _center(key: str, raw: Mapping[str, Any]) -> Center:
    return Center(
        key=key,
        set=str(raw.get("set", "")),
        name=str(raw.get("name", key)),
        order=int(raw.get("order", 0)),
        cost=int(raw.get("cost", 0)),
        rarity=int(raw.get("rarity", 0)),
        config=_freeze(raw.get("config", {})),
        unlocked=raw.get("unlocked", True) is not False,
        discovered=bool(raw.get("discovered", False)),
        blueprint_compat=bool(raw.get("blueprint_compat", False)),
        eternal_compat=bool(raw.get("eternal_compat", False)),
        perishable_compat=bool(raw.get("perishable_compat", False)),
        effect=str(raw.get("effect", "")),
        raw=_freeze(raw),
    )


#: Every item by key, for example ``CENTERS["j_blueprint"]``.
CENTERS: Mapping[str, Center] = MappingProxyType(
    {key: _center(key, raw) for key, raw in _DATA["centers"].items()}
)


def center(key: str) -> Center:
    """The definition of item ``key``; raises ``KeyError`` for unknown keys."""
    return CENTERS[key]


def config_value(key: str, name: str, default: Any = None) -> Any:
    """One value from an item's config, such as ``config_value("m_glass", "Xmult")``."""
    return CENTERS[key].config.get(name, default)


def pool(set_name: str, rarity: int | None = None) -> tuple[Center, ...]:
    """The items of one set in the game's pool order (by ``order``).

    For jokers, ``rarity`` selects one rarity pool (1 common to 4 legendary).
    """
    members = [
        c
        for c in CENTERS.values()
        if c.set == set_name
        and not c.raw.get("wip")
        and not c.raw.get("omit")
        and not c.raw.get("skip_pool")
        and (rarity is None or c.rarity == rarity)
    ]
    return tuple(sorted(members, key=lambda c: c.order))
