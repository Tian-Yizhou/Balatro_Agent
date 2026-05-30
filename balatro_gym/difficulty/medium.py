"""Medium difficulty: full joker + tarot pools.

Per the project plan, medium adds:

- All 30 jokers (including state-tracking ones like Runner, Ride the Bus,
  Loyalty Card, Ceremonial Dagger).
- All tarots — some of these mutate cards (enhancements, suit conversion,
  destruction, creation), so the deck is no longer the fixed 52.
- Some spectrals.
- 6 antes (18 blinds).
- No starting joker.

TODO
----
- Decide on a deck-modification budget per run (rate-limit destruction
  effects so the deck doesn't shrink to nothing).
- Tune joker pool size for sane shop offerings.
- Validate that all consumables work end-to-end during training.
- Confirm boss blind balance with the larger joker pool.
"""

from __future__ import annotations

from balatro_gym.core.consumable import ConsumableType, get_consumables_by_type
from balatro_gym.core.joker import get_all_joker_ids
from balatro_gym.envs.configs import GameConfig


# Spectral subset chosen for medium — only effects that don't depend on
# game-state features not yet fully implemented.
SAFE_SPECTRALS: list[str] = [
    "c_talisman", "c_deja_vu", "c_trance", "c_medium", "c_aura", "c_cryptid",
]


def build_config(seed: int | None = None) -> GameConfig:
    return GameConfig(
        num_antes=6,
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
            + list(SAFE_SPECTRALS)
        ),
        seed=seed,
    )
