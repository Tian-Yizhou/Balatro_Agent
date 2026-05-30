"""Hard difficulty: full Balatro feature set.

Per the project plan, hard should eventually mirror the actual game:

- 8 antes (24 blinds, including the final boss).
- All jokers, planets, tarots, spectrals.
- All boss blind debuff effects.
- Vouchers (shop upgrades that persist across rounds).
- Deck variants (Red, Blue, Yellow, Black, Plasma, ...).
- Stake levels (White → Gold), each adding global modifiers.
- No starting joker, minimal starting cash.

What's currently implemented
----------------------------
- All 30 jokers, 40 consumables.
- 9 boss blind effects (subset of the full game).
- 8-ante progression.

TODO
----
- Vouchers (the registry exists in name only).
- Deck variants.
- Stake levels.
- Remaining boss blind effects.

The reference for unimplemented mechanics is the Lua source under
``Balatro/`` (the original game's code, included for reading only).
"""

from __future__ import annotations

from balatro_gym.core.consumable import ConsumableType, get_consumables_by_type
from balatro_gym.core.joker import get_all_joker_ids
from balatro_gym.envs.configs import GameConfig


def build_config(seed: int | None = None) -> GameConfig:
    return GameConfig(
        num_antes=8,
        hands_per_round=4,
        discards_per_round=3,
        hand_size=8,
        max_jokers=5,
        starting_money=4,
        shop_slots=2,
        reroll_base_cost=5,
        consumable_slots=2,
        joker_pool=get_all_joker_ids(),
        starting_joker_ids=[],
        consumable_pool=(
            get_consumables_by_type(ConsumableType.PLANET)
            + get_consumables_by_type(ConsumableType.TAROT)
            + get_consumables_by_type(ConsumableType.SPECTRAL)
        ),
        seed=seed,
    )
