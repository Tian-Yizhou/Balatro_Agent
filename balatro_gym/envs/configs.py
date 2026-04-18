"""Game configuration and difficulty presets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from balatro_gym.core.joker import get_all_joker_ids, get_jokers_by_rarity
from balatro_gym.core.consumable import (
    get_all_consumable_ids, get_consumables_by_type, ConsumableType,
)


# Joker pools by difficulty tier
_PRIORITY_1_JOKERS: list[str] = [
    "joker_basic", "greedy_joker", "lusty_joker", "wrathful_joker",
    "gluttonous_joker", "jolly_joker", "zany_joker", "banner",
    "mystic_summit", "ice_cream",
]

_PRIORITY_2_JOKERS: list[str] = [
    "raised_fist", "fibonacci", "even_steven", "odd_todd", "scholar",
    "business_card", "stencil", "half_joker", "blueprint", "dna",
]

_PRIORITY_3_JOKERS: list[str] = [
    "abstract_joker", "blackboard", "the_duo", "the_trio", "the_family",
    "loyalty_card", "ceremonial_dagger", "ride_the_bus", "runner", "supernova",
]

# Consumable pools by difficulty tier
_ALL_PLANETS: list[str] = get_consumables_by_type(ConsumableType.PLANET)

# Simple tarots (enhancements + suit conversion — no card destruction/creation)
_SIMPLE_TAROTS: list[str] = [
    "c_magician", "c_empress", "c_hierophant", "c_lovers", "c_chariot",
    "c_justice", "c_devil", "c_tower",
    "c_star", "c_moon", "c_sun", "c_world",
    "c_strength", "c_hermit",
]

_ALL_TAROTS: list[str] = get_consumables_by_type(ConsumableType.TAROT)

_SIMPLE_SPECTRALS: list[str] = [
    "c_talisman", "c_deja_vu", "c_trance", "c_medium",
    "c_aura", "c_cryptid",
]

_ALL_SPECTRALS: list[str] = get_consumables_by_type(ConsumableType.SPECTRAL)


@dataclass
class GameConfig:
    """Configuration for a Balatro game environment.

    Controls game parameters, joker pool, consumable pool, and difficulty settings.
    """
    num_antes: int = 8
    hands_per_round: int = 4
    discards_per_round: int = 3
    hand_size: int = 8
    max_jokers: int = 5
    starting_money: int = 4
    shop_slots: int = 2
    reroll_base_cost: int = 5
    consumable_slots: int = 2
    joker_pool: list[str] = field(default_factory=list)
    starting_joker_ids: list[str] = field(default_factory=list)
    consumable_pool: list[str] = field(default_factory=list)
    seed: int | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate that all joker/consumable IDs exist in registries."""
        all_joker_ids = set(get_all_joker_ids())
        for jid in self.joker_pool:
            if jid not in all_joker_ids:
                raise ValueError(f"Unknown joker ID in pool: {jid!r}")
        for jid in self.starting_joker_ids:
            if jid not in all_joker_ids:
                raise ValueError(f"Unknown starting joker ID: {jid!r}")
        all_consumable_ids = set(get_all_consumable_ids())
        for cid in self.consumable_pool:
            if cid not in all_consumable_ids:
                raise ValueError(f"Unknown consumable ID in pool: {cid!r}")

    @classmethod
    def easy(cls, seed: int | None = None) -> GameConfig:
        """Easy difficulty: 4 antes, extra hands/discards, simple jokers + planets."""
        return cls(
            num_antes=4,
            hands_per_round=5,
            discards_per_round=4,
            hand_size=8,
            max_jokers=5,
            starting_money=6,
            shop_slots=2,
            reroll_base_cost=5,
            consumable_slots=2,
            joker_pool=list(_PRIORITY_1_JOKERS),
            starting_joker_ids=["joker_basic"],
            consumable_pool=list(_ALL_PLANETS + _SIMPLE_TAROTS),
            seed=seed,
        )

    @classmethod
    def medium(cls, seed: int | None = None) -> GameConfig:
        """Medium difficulty: 6 antes, standard params, expanded pools."""
        return cls(
            num_antes=6,
            hands_per_round=4,
            discards_per_round=3,
            hand_size=8,
            max_jokers=5,
            starting_money=4,
            shop_slots=2,
            reroll_base_cost=5,
            consumable_slots=2,
            joker_pool=_PRIORITY_1_JOKERS + _PRIORITY_2_JOKERS,
            starting_joker_ids=[],
            consumable_pool=list(_ALL_PLANETS + _ALL_TAROTS + _SIMPLE_SPECTRALS),
            seed=seed,
        )

    @classmethod
    def hard(cls, seed: int | None = None) -> GameConfig:
        """Hard difficulty: 8 antes, all jokers/consumables, no starting advantage."""
        return cls(
            num_antes=8,
            hands_per_round=4,
            discards_per_round=3,
            hand_size=8,
            max_jokers=5,
            starting_money=4,
            shop_slots=2,
            reroll_base_cost=5,
            consumable_slots=2,
            joker_pool=_PRIORITY_1_JOKERS + _PRIORITY_2_JOKERS + _PRIORITY_3_JOKERS,
            starting_joker_ids=[],
            consumable_pool=list(_ALL_PLANETS + _ALL_TAROTS + _ALL_SPECTRALS),
            seed=seed,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> GameConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary."""
        return {
            "num_antes": self.num_antes,
            "hands_per_round": self.hands_per_round,
            "discards_per_round": self.discards_per_round,
            "hand_size": self.hand_size,
            "max_jokers": self.max_jokers,
            "starting_money": self.starting_money,
            "shop_slots": self.shop_slots,
            "reroll_base_cost": self.reroll_base_cost,
            "consumable_slots": self.consumable_slots,
            "joker_pool": list(self.joker_pool),
            "starting_joker_ids": list(self.starting_joker_ids),
            "consumable_pool": list(self.consumable_pool),
            "seed": self.seed,
        }
