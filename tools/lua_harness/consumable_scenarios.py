"""Random situations for using Planet and Tarot cards, described as plain data.

Played through the original game (``rules.use_consumable``) and through the engine
(``tests/scenarios.py``); the results must match.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from typing import Any

from scoring_scenarios import EDITIONS, HANDS, _random_card

__all__ = ["random_consumable_scenarios", "random_pack_scenarios", "random_shop_scenarios"]


def random_consumable_scenarios(
    seed: int, count: int, consumable_keys: Sequence[str], joker_keys: Sequence[str]
) -> Iterator[dict[str, Any]]:
    """Yield ``count`` random uses of consumables from ``consumable_keys``."""
    randomizer = random.Random(seed)
    for _ in range(count):
        uid = iter(range(1, 400))
        held = [
            _random_card(randomizer, next(uid), False, ("Red", "Gold", "Blue", "Purple"))
            for _ in range(randomizer.randint(1, 8))
        ]
        selected = sorted(
            randomizer.sample(range(len(held)), randomizer.randint(0, min(4, len(held))))
        )
        jokers = []
        for key in randomizer.sample(list(joker_keys), randomizer.randint(0, 5)):
            joker: dict[str, Any] = {"uid": next(uid), "key": key}
            if randomizer.random() < 0.3:
                joker["edition"] = randomizer.choice(EDITIONS)
            jokers.append(joker)
        yield {
            "held": held,
            "selected": selected,
            "jokers": jokers,
            "consumable": randomizer.choice(consumable_keys),
            "consumables": randomizer.sample(list(consumable_keys), randomizer.randint(0, 1)),
            "last_tarot_planet": randomizer.choice([None, *consumable_keys]),
            "levels": {hand: randomizer.randint(1, 3) for hand in randomizer.sample(HANDS, 2)},
            "dollars": randomizer.randint(0, 40),
            "probability": randomizer.choice((0, 1000)),
            "phase": randomizer.choice(("selecting_hand", "selecting_hand", "shop", "pack_open")),
        }


def random_shop_scenarios(
    seed: int, count: int, joker_keys: Sequence[str], consumable_keys: Sequence[str]
) -> Iterator[dict[str, Any]]:
    """Yield ``count`` random shop situations (see ``rules.shop``)."""
    randomizer = random.Random(seed)
    lasting = ("j_juggler", "j_drunkard", "j_credit_card", "j_chaos", "j_joker")
    flags = ("gros_michel_extinct",)
    for _ in range(count):
        uid = iter(range(1, 400))
        jokers = []
        for key in randomizer.sample(list(joker_keys), randomizer.randint(0, 5)):
            joker: dict[str, Any] = {"uid": next(uid), "key": key}
            if randomizer.random() < 0.3:
                joker["edition"] = randomizer.choice((*EDITIONS, "e_negative"))
            if randomizer.random() < 0.15:
                joker["rental"] = True
            if key == "j_egg":
                joker["values"] = {"extra_value": randomizer.randint(0, 9)}
            jokers.append(joker)
        yield {
            "jokers": jokers,
            "consumables": randomizer.sample(list(consumable_keys), randomizer.randint(0, 2)),
            "dollars": randomizer.randint(0, 50),
            "inflation": randomizer.choice((0, 0, 1, 3)),
            "discount": randomizer.choice((0, 0, 25, 50)),
            "played_hands": randomizer.sample(
                ("Flush House", "Five of a Kind", "Flush Five"), randomizer.randint(0, 2)
            ),
            "pool_flags": [flag for flag in flags if randomizer.random() < 0.4],
            "gained": randomizer.choice(lasting),
            "rerolls": randomizer.randint(0, 3),
            "sell": randomizer.randrange(len(jokers))
            if jokers and randomizer.random() < 0.8
            else None,
        }


def random_pack_scenarios(
    seed: int, count: int, pack_keys: Sequence[str], joker_keys: Sequence[str]
) -> Iterator[dict[str, Any]]:
    """Yield ``count`` random booster pack openings in the shop (see ``rules.open_pack``)."""
    randomizer = random.Random(seed)
    for number in range(count):
        uid = iter(range(1, 400))
        jokers = []
        for key in randomizer.sample(list(joker_keys), randomizer.choice((0, 2, 5, 5))):
            joker: dict[str, Any] = {"key": key}
            if randomizer.random() < 0.2:
                joker["edition"] = "e_negative"
            jokers.append(joker)
        yield {
            "pack": pack_keys[number % len(pack_keys)],
            "seed": f"pack{seed}-{number}",
            "ante": randomizer.randint(1, 8),
            # Enough to buy any pack.
            "dollars": randomizer.randint(8, 30),
            "jokers": jokers,
            "held": [
                _random_card(randomizer, next(uid), False, ("Red", "Gold", "Blue", "Purple"))
                for _ in range(8)
            ],
        }
