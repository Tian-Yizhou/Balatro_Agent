"""Game configuration dataclass and YAML loader.

Difficulty *content* (joker pools, consumable pools, antes) lives in
:mod:`balatro_gym.difficulty` — one file per difficulty, plug-in style.
This module only defines the :class:`GameConfig` dataclass plus a
backward-compatible ``GameConfig.easy()/.medium()/.hard()`` shim that
delegates to the new registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from balatro_gym.core.back import get_all_back_ids
from balatro_gym.core.consumable import get_all_consumable_ids
from balatro_gym.core.joker import get_all_joker_ids
from balatro_gym.core.stake import get_all_stake_ids
from balatro_gym.core.tag import get_all_tag_ids
from balatro_gym.core.voucher import get_all_voucher_ids


@dataclass
class GameConfig:
    """Configuration for a Balatro game environment.

    Controls game parameters, joker pool, consumable pool, and difficulty
    settings. Construct directly, via :meth:`from_file` to load from YAML,
    or via the plug-in registry::

        from balatro_gym.difficulty import get_difficulty
        cfg = get_difficulty("easy", seed=42)
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
    voucher_pool: list[str] = field(default_factory=list)
    tag_pool: list[str] = field(default_factory=list)
    deck_back: str | None = None     # ID from balatro_gym.core.back; None = no modifier
    stake: str = "stake_white"       # ID from balatro_gym.core.stake; default = no modifier
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
        all_voucher_ids = set(get_all_voucher_ids())
        for vid in self.voucher_pool:
            if vid not in all_voucher_ids:
                raise ValueError(f"Unknown voucher ID in pool: {vid!r}")
        all_tag_ids = set(get_all_tag_ids())
        for tid in self.tag_pool:
            if tid not in all_tag_ids:
                raise ValueError(f"Unknown tag ID in pool: {tid!r}")
        if self.deck_back is not None and self.deck_back not in set(get_all_back_ids()):
            raise ValueError(
                f"Unknown deck_back: {self.deck_back!r}. "
                f"Available: {get_all_back_ids()}"
            )
        if self.stake not in set(get_all_stake_ids()):
            raise ValueError(
                f"Unknown stake: {self.stake!r}. "
                f"Available: {get_all_stake_ids()}"
            )

    # ------------------------------------------------------------------
    # Backward-compatible preset shims. The actual content lives in
    # balatro_gym/difficulty/<name>.py. Imports are lazy to avoid a
    # circular dependency at module load time.
    # ------------------------------------------------------------------

    @classmethod
    def easy(cls, seed: int | None = None) -> GameConfig:
        """Easy preset — see :mod:`balatro_gym.difficulty.easy`."""
        from balatro_gym.difficulty import get_difficulty
        return get_difficulty("easy", seed=seed)

    @classmethod
    def medium(cls, seed: int | None = None) -> GameConfig:
        """Medium preset — see :mod:`balatro_gym.difficulty.medium`."""
        from balatro_gym.difficulty import get_difficulty
        return get_difficulty("medium", seed=seed)

    @classmethod
    def hard(cls, seed: int | None = None) -> GameConfig:
        """Hard preset — see :mod:`balatro_gym.difficulty.hard`."""
        from balatro_gym.difficulty import get_difficulty
        return get_difficulty("hard", seed=seed)

    @classmethod
    def from_file(cls, path: str | Path, base: str = "medium") -> GameConfig:
        """Load config from a YAML file, merging over a base preset.

        Any field specified in the YAML overrides the base preset value.
        Fields not in the file keep their base preset defaults.

        The special key ``base`` in the YAML selects which preset to use
        as the starting point. If ``base`` is present in the file, the
        *base* parameter is ignored. The base can be any difficulty name
        registered under :mod:`balatro_gym.difficulty`, including
        user-added files.

        Args:
            path: Path to a YAML config file.
            base: Default base preset name. Overridden by the ``base``
                  key in the YAML file if present.

        Example YAML file::

            # Start from easy preset, override two fields
            base: easy
            num_antes: 2
            hands_per_round: 6
        """
        from balatro_gym.difficulty import get_difficulty, list_difficulties

        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        preset_name = data.pop("base", base)
        available = list_difficulties()
        if preset_name not in available:
            raise ValueError(
                f"Unknown base preset {preset_name!r}. "
                f"Choose from: {available}"
            )

        base_config = get_difficulty(preset_name)
        merged = base_config.to_dict()
        merged.update(data)
        return cls(**merged)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dictionary."""
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
            "voucher_pool": list(self.voucher_pool),
            "tag_pool": list(self.tag_pool),
            "deck_back": self.deck_back,
            "stake": self.stake,
            "seed": self.seed,
        }

    def to_yaml(self, path: str | Path) -> None:
        """Write config to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
