"""Random situations for vouchers, described as plain data.

Played through the original game (``rules.redeem_voucher``, ``rules.reroll_boss``
and ``rules.shop_playing_cards``) and through the engine (``tests/scenarios.py``);
the results must match.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from scoring_scenarios import EDITIONS

__all__ = [
    "random_boss_reroll_scenarios",
    "random_redeem_scenarios",
    "random_shop_playing_card_scenarios",
]

_BOSSES = (
    "bl_hook",
    "bl_club",
    "bl_psychic",
    "bl_goad",
    "bl_window",
    "bl_manacle",
    "bl_pillar",
    "bl_head",
    "bl_wall",
    "bl_wheel",
)


def _with_requirements(keys: Sequence[str], requires: Mapping[str, Sequence[str]]) -> list[str]:
    """``keys`` with the vouchers each one upgrades, those first."""
    result: list[str] = []
    for key in keys:
        for voucher in (*requires.get(key, ()), key):
            if voucher not in result:
                result.append(voucher)
    return result


def random_redeem_scenarios(
    seed: int,
    count: int,
    requires: Mapping[str, Sequence[str]],
    joker_keys: Sequence[str],
    consumable_keys: Sequence[str],
    pack_keys: Sequence[str],
) -> Iterator[dict[str, Any]]:
    """Yield ``count`` random voucher redemptions in the shop, cycling through every
    voucher. ``requires`` maps each voucher to the vouchers it upgrades."""
    randomizer = random.Random(seed)
    vouchers = sorted(requires)
    for number in range(count):
        voucher = vouchers[number % len(vouchers)]
        others = [key for key in vouchers if key != voucher and voucher not in requires[key]]
        used = _with_requirements(
            [*randomizer.sample(others, randomizer.randint(0, 6)), *requires[voucher]], requires
        )
        jokers = []
        for key in randomizer.sample(list(joker_keys), randomizer.randint(0, 4)):
            joker: dict[str, Any] = {"key": key}
            if randomizer.random() < 0.3:
                joker["edition"] = randomizer.choice((*EDITIONS, "e_negative"))
            if randomizer.random() < 0.15:
                joker["rental"] = True
            if key == "j_egg":
                joker["values"] = {"extra_value": randomizer.randint(0, 9)}
            jokers.append(joker)
        shop_slots = 2 + sum(key in used for key in ("v_overstock_norm", "v_overstock_plus"))
        yield {
            "voucher": voucher,
            "used": used,
            "dollars": randomizer.randint(0, 30),
            "hands_left": randomizer.randint(1, 4),
            "discards_left": randomizer.randint(0, 3),
            "ante": randomizer.randint(1, 8),
            "reroll_increase": randomizer.randint(0, 3),
            "jokers": jokers,
            "consumables": randomizer.sample(list(consumable_keys), randomizer.randint(0, 2)),
            "shop_slots": shop_slots,
            "shop": randomizer.sample(
                [*joker_keys, *consumable_keys], randomizer.randint(0, shop_slots)
            ),
            "pack": randomizer.choice(pack_keys),
        }


def random_boss_reroll_scenarios(seed: int, count: int) -> Iterator[dict[str, Any]]:
    """Yield ``count`` random presses of the boss reroll button at blind selection."""
    randomizer = random.Random(seed)
    for _ in range(count):
        yield {
            "used": randomizer.choice(
                ([], ["v_directors_cut"], ["v_directors_cut", "v_retcon"], ["v_retcon"])
            ),
            "rerolled": randomizer.random() < 0.4,
            "dollars": randomizer.randint(0, 25),
            "bankrupt_at": randomizer.choice((0, 0, -20)),
            "ante": randomizer.randint(1, 16),
            "bosses_used": {
                key: randomizer.randint(0, 2)
                for key in randomizer.sample(_BOSSES, randomizer.randint(0, len(_BOSSES)))
            },
        }


def random_shop_playing_card_scenarios(
    seed: int, count: int, joker_keys: Sequence[str]
) -> Iterator[dict[str, Any]]:
    """Yield ``count`` random shops that may sell playing cards. Half of them sell
    only playing cards, so every kind of card shows up."""
    randomizer = random.Random(seed)
    for number in range(count):
        used = randomizer.choice(([], ["v_magic_trick"], ["v_magic_trick", "v_illusion"]))
        playing_card_rate = 4 if used else 0
        # Without Magic Trick the shop sells no playing cards at all.
        rates = [20, 4, 4, playing_card_rate, 0] if number % 2 or not used else [0, 0, 0, 4, 0]
        yield {
            "used": used,
            "rates": rates,
            "discount": randomizer.choice((0, 25, 50)),
            "dollars": randomizer.randint(0, 10),
            "ante": randomizer.randint(1, 8),
            "seed": f"shop-cards{seed}-{number}",
            "jokers": randomizer.sample(list(joker_keys), randomizer.randint(0, 3)),
            "count": 300,
        }
