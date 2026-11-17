"""Random situations for boss blinds, described as plain data.

Played through the original game (``rules.boss_round``, and ``rules.score`` with a
``blind``) and through the engine (``tests/scenarios.py``); the results must match.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from typing import Any

from scoring_scenarios import EDITIONS, HANDS, _random_card, random_scoring_scenarios

__all__ = ["random_boss_round_scenarios", "random_boss_scoring_scenarios"]

#: Bosses whose effect acts on a played hand as it scores.
SCORING_BOSSES = ("bl_psychic", "bl_eye", "bl_mouth", "bl_arm", "bl_ox", "bl_flint")


def random_boss_round_scenarios(
    seed: int, count: int, boss_keys: Sequence[str], joker_keys: Sequence[str]
) -> Iterator[dict[str, Any]]:
    """Yield ``count`` random rounds against each boss in turn (see ``rules.boss_round``)."""
    randomizer = random.Random(seed)
    for number in range(count):
        uid = iter(range(1, 400))
        narrow = randomizer.random() < 0.5
        hand = [_random_card(randomizer, next(uid), narrow, ("Red",)) for _ in range(8)]
        deck = [
            _random_card(randomizer, next(uid), narrow, ("Red",))
            for _ in range(randomizer.randint(2, 10))
        ]
        for card in [*hand, *deck]:
            card["played_this_ante"] = randomizer.random() < 0.3
        keys = randomizer.sample(list(joker_keys), randomizer.randint(0, 4))
        if randomizer.random() < 0.25:
            keys.append("j_chicot")
        jokers = []
        for key in keys:
            joker: dict[str, Any] = {"uid": next(uid), "key": key}
            if randomizer.random() < 0.2:
                joker["edition"] = randomizer.choice(EDITIONS)
            jokers.append(joker)
        yield {
            "blind": boss_keys[number % len(boss_keys)],
            "seed": f"boss{seed}-{number}",
            "ante": randomizer.randint(1, 8),
            "dollars": randomizer.randint(0, 20),
            # 0 or 7 makes The Wheel's 1 in 7 chance certain either way.
            "probability": randomizer.choice((0, 7)),
            "hands_played": randomizer.choice((0, 0, 1)),
            "discards_used": randomizer.choice((0, 0, 1)),
            "jokers": jokers,
            "hand": hand,
            "deck": deck,
            "played": sorted(randomizer.sample(range(8), randomizer.randint(1, 5))),
            "disable": randomizer.random() < 0.4,
        }


def random_boss_scoring_scenarios(
    seed: int, count: int, joker_keys: Sequence[str], seals: Sequence[str]
) -> Iterator[dict[str, Any]]:
    """Yield ``count`` random scoring situations against the bosses that act on a
    played hand, with what they remember from earlier hands this round."""
    randomizer = random.Random(seed)
    scenarios = random_scoring_scenarios(seed, count, joker_keys, seals=seals)
    for number, scenario in enumerate(scenarios):
        blind: dict[str, Any] = {"key": SCORING_BOSSES[number % len(SCORING_BOSSES)]}
        blind["hands_used"] = randomizer.sample(HANDS, randomizer.randint(0, 4))
        if randomizer.random() < 0.6:
            blind["only_hand"] = randomizer.choice(HANDS)
        blind["most_played"] = randomizer.choice(HANDS)
        blind["disabled"] = randomizer.random() < 0.15
        yield {**scenario, "blind": blind}
