"""Easy difficulty: a beginner-friendly starting point.

Restrictions
------------
- Standard 52-card deck. Consumable pool is Planets-only, which level up
  hand types but never modify the deck — the 52 cards stay fixed across
  the whole episode.
- Only entry-level jokers (10 Priority-1 jokers). All have simple,
  unconditional effects (per-suit mult, hand-type mult, flat chips).
  None of them mutate cards or track state across rounds.
- 4 antes (= 12 blinds). A short game suitable for fast iteration and
  early RL experiments.
- Player starts with the basic +4 Mult joker so the first blind is
  reachable without lucky hands.
"""

from __future__ import annotations

from balatro_gym.core.consumable import ConsumableType, get_consumables_by_type
from balatro_gym.envs.configs import GameConfig


# Entry-level jokers (Priority-1). Simple, unconditional effects only —
# nothing that tracks state, modifies cards, or has complex triggers.
ENTRY_LEVEL_JOKERS: list[str] = [
    "joker_basic",        # +4 Mult
    "greedy_joker",       # +3 Mult per played Diamond
    "lusty_joker",        # +3 Mult per played Heart
    "wrathful_joker",     # +3 Mult per played Spade
    "gluttonous_joker",   # +3 Mult per played Club
    "jolly_joker",        # +8 Mult if hand contains Pair
    "zany_joker",         # +12 Mult if hand contains Three of a Kind
    "banner",             # +30 Chips per remaining discard
    "mystic_summit",      # +15 Mult when 0 discards remaining
    "ice_cream",          # +100 Chips, -5 each hand played
]


def build_config(seed: int | None = None) -> GameConfig:
    return GameConfig(
        num_antes=4,
        hands_per_round=5,
        discards_per_round=4,
        hand_size=8,
        max_jokers=5,
        starting_money=6,
        shop_slots=2,
        reroll_base_cost=5,
        consumable_slots=2,
        joker_pool=list(ENTRY_LEVEL_JOKERS),
        starting_joker_ids=["joker_basic"],
        consumable_pool=get_consumables_by_type(ConsumableType.PLANET),
        seed=seed,
    )
