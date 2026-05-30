"""Customized difficulty: a worked example for users.

This is a template. Copy it (``cp customized.py my_setup.py``) and edit the
new file to define your own difficulty. Once the file exists, it's usable
immediately — no registration step::

    env = balatro_gym.make("my_setup")
    # or
    env = gymnasium.make("Balatro-v0", config_preset="my_setup")

Configurable fields (see :class:`balatro_gym.envs.configs.GameConfig`)
---------------------------------------------------------------------
- ``num_antes``: how many antes (each ante = 3 blinds: small, big, boss).
- ``hands_per_round`` / ``discards_per_round``: resources per blind.
- ``hand_size``: cards drawn to hand.
- ``max_jokers``: joker slot count (1–5).
- ``starting_money``: opening cash.
- ``shop_slots``: number of joker slots offered in the shop.
- ``reroll_base_cost``: cost to reroll the shop (scales up per-blind).
- ``consumable_slots``: how many tarots/planets/spectrals you can hold.
- ``joker_pool``: which jokers can appear in the shop.
- ``starting_joker_ids``: jokers the player owns at episode start.
- ``consumable_pool``: which consumables can appear in the shop.
- ``seed``: episode seed. Leave ``None`` here and pass per-episode via
  ``env.reset(seed=...)``.

Helpers (imported below)
------------------------
- ``get_all_joker_ids()``            — all 30 registered joker IDs.
- ``get_jokers_by_rarity(rarity)``   — filter by rarity (1=common, 2=uncommon, 3=rare).
- ``get_consumables_by_type(t)``     — filter by ConsumableType (PLANET, TAROT, SPECTRAL).
- ``get_all_consumable_ids()``       — all 40 consumable IDs.
"""

from __future__ import annotations

from balatro_gym.core.consumable import (
    ConsumableType,
    get_all_consumable_ids,
    get_consumables_by_type,
)
from balatro_gym.core.joker import get_all_joker_ids, get_jokers_by_rarity
from balatro_gym.envs.configs import GameConfig


def build_config(seed: int | None = None) -> GameConfig:
    # ----- Pick jokers --------------------------------------------------
    # Option A: explicit IDs (most control). Useful when you want a
    # narrow themed run, e.g. suit-mult jokers only.
    #
    #   joker_pool = [
    #       "greedy_joker", "lusty_joker",
    #       "wrathful_joker", "gluttonous_joker",
    #   ]
    #
    # Option B: filter by rarity. Rarities: 1=common, 2=uncommon, 3=rare.
    #
    #   joker_pool = get_jokers_by_rarity(1) + get_jokers_by_rarity(2)
    #
    # Option C: use everything.
    #
    #   joker_pool = get_all_joker_ids()
    #
    # Default for this example: all jokers, sorted for determinism.
    joker_pool = sorted(get_all_joker_ids())

    # ----- Pick consumables ---------------------------------------------
    # Planets level up hand types (e.g. boost flush base score).
    # Tarots can enhance, destroy, or convert cards (deck mutates).
    # Spectrals are powerful one-off effects.
    #
    # Examples:
    #   consumable_pool = get_consumables_by_type(ConsumableType.PLANET)
    #   consumable_pool = get_all_consumable_ids()
    #
    # Default for this example: planets + tarots, no spectrals.
    consumable_pool = (
        get_consumables_by_type(ConsumableType.PLANET)
        + get_consumables_by_type(ConsumableType.TAROT)
    )

    # ----- Build the config ---------------------------------------------
    return GameConfig(
        num_antes=5,
        hands_per_round=4,
        discards_per_round=3,
        hand_size=8,
        max_jokers=5,
        starting_money=10,          # generous opening so the shop is usable early
        shop_slots=3,               # 3 joker offers per shop visit
        reroll_base_cost=4,
        consumable_slots=2,
        joker_pool=joker_pool,
        starting_joker_ids=["joker_basic"],
        consumable_pool=consumable_pool,
        seed=seed,
    )
