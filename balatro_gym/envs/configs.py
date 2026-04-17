"""Game configuration and difficulty presets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from balatro_gym.core.joker import get_all_joker_ids, get_jokers_by_rarity


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


@dataclass
class GameConfig:
    """Configuration for a Balatro game environment.

    Controls game parameters, joker pool, and difficulty settings.
    """
    num_antes: int = 8
    hands_per_round: int = 4
    discards_per_round: int = 3
    hand_size: int = 8
    max_jokers: int = 5
    starting_money: int = 4
    shop_slots: int = 2
    reroll_base_cost: int = 5
    joker_pool: list[str] = field(default_factory=list)
    starting_joker_ids: list[str] = field(default_factory=list)
    seed: int | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate that all joker IDs exist in the registry."""
        all_ids = set(get_all_joker_ids())
        for jid in self.joker_pool:
            if jid not in all_ids:
                raise ValueError(f"Unknown joker ID in pool: {jid!r}")
        for jid in self.starting_joker_ids:
            if jid not in all_ids:
                raise ValueError(f"Unknown starting joker ID: {jid!r}")

    @classmethod
    def easy(cls, seed: int | None = None) -> GameConfig:
        """Easy difficulty: 4 antes, extra hands/discards, simple jokers."""
        return cls(
            num_antes=4,
            hands_per_round=5,
            discards_per_round=4,
            hand_size=8,
            max_jokers=5,
            starting_money=6,
            shop_slots=2,
            reroll_base_cost=5,
            joker_pool=list(_PRIORITY_1_JOKERS),
            starting_joker_ids=["joker_basic"],
            seed=seed,
        )

    @classmethod
    def medium(cls, seed: int | None = None) -> GameConfig:
        """Medium difficulty: 6 antes, standard parameters, expanded joker pool."""
        return cls(
            num_antes=6,
            hands_per_round=4,
            discards_per_round=3,
            hand_size=8,
            max_jokers=5,
            starting_money=4,
            shop_slots=2,
            reroll_base_cost=5,
            joker_pool=_PRIORITY_1_JOKERS + _PRIORITY_2_JOKERS,
            starting_joker_ids=[],
            seed=seed,
        )

    @classmethod
    def hard(cls, seed: int | None = None) -> GameConfig:
        """Hard difficulty: 8 antes, all jokers, no starting advantage."""
        return cls(
            num_antes=8,
            hands_per_round=4,
            discards_per_round=3,
            hand_size=8,
            max_jokers=5,
            starting_money=4,
            shop_slots=2,
            reroll_base_cost=5,
            joker_pool=_PRIORITY_1_JOKERS + _PRIORITY_2_JOKERS + _PRIORITY_3_JOKERS,
            starting_joker_ids=[],
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
            "joker_pool": list(self.joker_pool),
            "starting_joker_ids": list(self.starting_joker_ids),
            "seed": self.seed,
        }
