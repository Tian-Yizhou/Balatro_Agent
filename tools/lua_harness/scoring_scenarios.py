"""Random scoring situations, described as plain data.

A scenario lists the played cards, the cards held in hand, the jokers and the
numbers of the round. The same description is played through the original game
(``rules.LuaRules.score``) and through the engine (``tests/scenarios.py``), and
the results must match.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from typing import Any

__all__ = ["random_discard_scenarios", "random_round_end_scenarios", "random_scoring_scenarios"]

Scenario = dict[str, Any]

ENHANCEMENTS = ("m_bonus", "m_mult", "m_wild", "m_glass", "m_steel", "m_stone", "m_gold", "m_lucky")
EDITIONS = ("e_foil", "e_holo", "e_polychrome")
SEALS = ("Red", "Gold", "Blue", "Purple")
HANDS = (
    "High Card",
    "Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
)

#: Joker values worth randomizing, with the range to draw from.
VALUE_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "j_ride_the_bus": {"mult": (0, 12)},
    "j_green_joker": {"mult": (0, 12)},
    "j_ice_cream": {"extra.chips": (5, 100)},
    "j_popcorn": {"mult": (4, 20)},
    "j_flash": {"mult": (0, 20)},
    "j_campfire": {"x_mult": (1, 3)},
    "j_constellation": {"x_mult": (1, 2)},
}


def random_scoring_scenarios(
    seed: int, count: int, joker_keys: Sequence[str], seals: Sequence[str] = SEALS
) -> Iterator[Scenario]:
    """Yield ``count`` random scenarios using jokers from ``joker_keys``."""
    randomizer = random.Random(seed)
    for _ in range(count):
        uid = iter(range(1, 1000))
        size = randomizer.choice((1, 2, 3, 4, 5, 5, 5))
        narrow = randomizer.random() < 0.5
        played = [_random_card(randomizer, next(uid), narrow, seals) for _ in range(size)]
        held = [
            _random_card(randomizer, next(uid), narrow, seals)
            for _ in range(randomizer.randint(0, 6))
        ]
        jokers = []
        for key in randomizer.sample(list(joker_keys), randomizer.randint(0, 5)):
            joker: dict[str, Any] = {"uid": next(uid), "key": key}
            if randomizer.random() < 0.25:
                joker["edition"] = randomizer.choice(EDITIONS)
            values: dict[str, float] = {}
            for name, (low, high) in VALUE_RANGES.get(key, {}).items():
                if name == "x_mult":
                    values[name] = round(randomizer.uniform(low, high), 2)
                else:
                    values[name] = randomizer.randint(int(low), int(high))
            if values:
                joker["values"] = values
            jokers.append(joker)
        yield {
            "played": played,
            "held": held,
            "jokers": jokers,
            "levels": {hand: randomizer.randint(1, 4) for hand in randomizer.sample(HANDS, 3)},
            "dollars": randomizer.randint(0, 40),
            "hands_left": randomizer.randint(0, 3),
            "discards_left": randomizer.randint(0, 3),
            "probability": randomizer.choice((0, 1000)),
            "tarots_used": randomizer.choice((0, 0, 1, 4)),
        }


def _random_card(
    randomizer: random.Random, uid: int, narrow: bool, seals: Sequence[str]
) -> dict[str, Any]:
    """A card; ``narrow`` draws from few ranks and suits, so hands form more often."""
    if narrow:
        rank = randomizer.choice((2, 3, 4, 5, 10, 11, 12, 13, 14))
        suit = randomizer.choice("SH")
    else:
        rank = randomizer.randint(2, 14)
        suit = randomizer.choice("SHCD")
    card: dict[str, Any] = {"uid": uid, "rank": rank, "suit": suit}
    if randomizer.random() < 0.3:
        card["enhancement"] = randomizer.choice(ENHANCEMENTS)
    if randomizer.random() < 0.15:
        card["edition"] = randomizer.choice(EDITIONS)
    if randomizer.random() < 0.15:
        card["seal"] = randomizer.choice(seals)
    if randomizer.random() < 0.08:
        card["debuffed"] = True
    return card


def random_round_end_scenarios(
    seed: int, count: int, joker_keys: Sequence[str], seals: Sequence[str] = SEALS
) -> Iterator[Scenario]:
    """Yield ``count`` random end-of-round situations."""
    randomizer = random.Random(seed)
    for _ in range(count):
        uid = iter(range(1, 1000))
        held = [
            _random_card(randomizer, next(uid), False, seals)
            for _ in range(randomizer.randint(0, 8))
        ]
        for card in held:
            if randomizer.random() < 0.3:
                card["enhancement"] = "m_gold"
        jokers = []
        for key in randomizer.sample(list(joker_keys), randomizer.randint(0, 5)):
            joker: dict[str, Any] = {"uid": next(uid), "key": key}
            if randomizer.random() < 0.25:
                joker["edition"] = randomizer.choice(EDITIONS)
            values: dict[str, float] = {}
            for name, (low, high) in VALUE_RANGES.get(key, {}).items():
                if name == "x_mult":
                    values[name] = round(randomizer.uniform(low, high), 2)
                else:
                    values[name] = randomizer.randint(int(low), int(high))
            if values:
                joker["values"] = values
            jokers.append(joker)
        yield {
            "held": held,
            "jokers": jokers,
            "dollars": randomizer.randint(0, 60),
            "hands_left": randomizer.randint(0, 3),
            "discards_left": randomizer.randint(0, 3),
            "probability": randomizer.choice((0, 1000)),
            "beaten": randomizer.random() < 0.8,
            "boss": randomizer.random() < 0.3,
            "last_hand": randomizer.choice(HANDS),
        }


def random_discard_scenarios(
    seed: int, count: int, joker_keys: Sequence[str], seals: Sequence[str] = SEALS
) -> Iterator[Scenario]:
    """Yield ``count`` random discards: a hand, the positions discarded, and jokers."""
    randomizer = random.Random(seed)
    for _ in range(count):
        uid = iter(range(1, 1000))
        held = [
            _random_card(randomizer, next(uid), False, seals)
            for _ in range(randomizer.randint(1, 8))
        ]
        selected = sorted(
            randomizer.sample(range(len(held)), randomizer.randint(1, min(5, len(held))))
        )
        jokers = []
        for key in randomizer.sample(list(joker_keys), randomizer.randint(0, 4)):
            joker: dict[str, Any] = {"uid": next(uid), "key": key}
            if key == "j_green_joker":
                joker["values"] = {"mult": randomizer.randint(0, 3)}
            jokers.append(joker)
        yield {
            "held": held,
            "selected": selected,
            "jokers": jokers,
            "dollars": randomizer.randint(0, 20),
            "discards_left": randomizer.randint(1, 3),
            "probability": randomizer.choice((0, 1000)),
        }
