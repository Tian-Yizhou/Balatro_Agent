from __future__ import annotations

import json
from pathlib import Path

import pytest

from balatro_gym.sim.hands import HAND_BASE
from balatro_gym.sim.state import HandType
from scenarios import describe

VECTORS = Path(__file__).parent / "data" / "lua_hands.json"


@pytest.mark.parametrize(
    ("text", "flags", "top"),
    [
        ("AS 2H 3C 4D 5S", "", HandType.STRAIGHT),
        ("JS QH KC AD 2S", "", HandType.HIGH_CARD),
        ("2S 3H 4C 5D", "", HandType.HIGH_CARD),
        ("2S 3H 4C 5D", "f", HandType.STRAIGHT),
        ("2S 3H 5C 6D 7S", "s", HandType.STRAIGHT),
        ("AH KD QH JD 9H", "m", HandType.FLUSH),
        ("AS AH AC AD AS", "", HandType.FIVE_OF_A_KIND),
        ("AS AS AS AS AS", "", HandType.FLUSH_FIVE),
        ("KS KS KS 2S 2S", "", HandType.FLUSH_HOUSE),
        ("KS KH KC 2S 2D", "", HandType.FULL_HOUSE),
        ("KS KH 2C 2S 5D", "", HandType.TWO_PAIR),
        ("KHw 2S 7S 9S JS", "", HandType.FLUSH),
        ("KHw! 2S 7S 9S JS", "", HandType.HIGH_CARD),
        ("KSs KH", "", HandType.HIGH_CARD),
    ],
)
def test_top_hand(text: str, flags: str, top: HandType) -> None:
    assert describe(text, flags)["top"] == top.value


def test_full_house_contains_smaller_hands() -> None:
    result = describe("KS KH KC 2S 2D")
    assert result["groups"]["Three of a Kind"] == [[0, 1, 2]]
    assert result["groups"]["Two Pair"] == [[3, 4, 0, 1, 2]]
    # The game reports the three-of-a-kind group as the pair.
    assert result["groups"]["Pair"] == [[0, 1, 2]]


def test_royal_flush_is_displayed_by_name() -> None:
    assert describe("TS JS QS KS AS")["display"] == "Royal Flush"
    assert describe("9S TS JS QS KS")["display"] == "Straight Flush"


def test_no_cards() -> None:
    result = describe("")
    assert result == {"top": None, "display": "", "scoring": [], "groups": {}}


def test_hand_base_values() -> None:
    assert HAND_BASE[HandType.PAIR].chips == 10
    assert HAND_BASE[HandType.FLUSH_FIVE].mult == 16
    assert [base.order for base in HAND_BASE.values()] == list(range(1, 13))


def test_matches_original_game_vectors() -> None:
    cases = json.loads(VECTORS.read_text())["cases"]
    assert len(cases) > 1000
    mismatches = [
        (case["cards"], case["flags"])
        for case in cases
        if describe(case["cards"], case["flags"])
        != {key: case[key] for key in ("top", "display", "scoring", "groups")}
    ]
    assert mismatches == []
