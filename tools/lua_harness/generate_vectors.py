"""Write test vectors computed by the original game's Lua code.

The vectors hold only inputs and the outputs the game produces for them, no game
code, so they are committed and let CI check the engine without the Lua source.

Usage::

    BALATRO_LUA_DIR=/path/to/game/source uv run python tools/lua_harness/generate_vectors.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from consumable_scenarios import random_consumable_scenarios, random_shop_scenarios
from harness import (
    CardSpec,
    LuaGame,
    format_cards,
    lua_dir_from_env,
    parse_cards,
    random_hand_cases,
)
from rules import LuaRules, discard, end_of_round, score, shop, use_consumable
from scoring_scenarios import (
    random_discard_scenarios,
    random_round_end_scenarios,
    random_scoring_scenarios,
)

from balatro_gym.sim.consumables import IMPLEMENTED as CONSUMABLES
from balatro_gym.sim.effects import JOKERS

#: Jokers compared against the game. Misprint's mult is random, so it is checked
#: separately, by range.
VERIFIED_JOKERS = sorted(key for key in JOKERS if key != "j_misprint")
#: Seals whose effects the engine implements.
VERIFIED_SEALS = ("Red", "Gold", "Blue", "Purple")

OUT_DIR = Path(__file__).resolve().parents[2] / "tests" / "sim" / "data"

# Hands chosen by hand to cover each rule; the expected results still come from Lua.
CRAFTED = [
    ("AS 2H 3C 4D 5S", ""),  # the wheel: Ace plays low
    ("TS JH QC KD AS", ""),
    ("JS QH KC AD 2S", ""),  # no wrap-around straight
    ("2S 3H 4C 5D", "f"),  # Four Fingers straight
    ("2S 3H 5C 6D 7S", "s"),  # Shortcut skips one rank
    ("2S 4H 6C 8D TS", "s"),
    ("AS 3H 5C 7D 9S", "s"),
    ("2H 3H 4H 5H 7S", "f"),  # four-card straight flush
    ("AH KD QH JD 9H", "m"),  # Smeared Joker flush
    ("AS AH AC AD AS", ""),  # five of a kind
    ("AS AS AS AS AS", ""),  # flush five
    ("KS KS KS 2S 2S", ""),  # flush house
    ("KS KH KC 2S 2D", ""),
    ("KS KH 2C 2S 5D", ""),
    ("KS KH KC KD 5D", ""),
    ("KSw 2S 7S 9S JS", ""),  # Wild card in a flush
    ("KSw! 2S 7S 9S JH", ""),  # a debuffed Wild card is not wild
    ("KSs 2S 7S 9S JS", ""),  # a Stone card breaks a flush
    ("KSs KHs", ""),
    ("5D", ""),
    ("AS AH KS KH QS QH", ""),  # more than five cards
]


def main() -> int:
    lua_dir = lua_dir_from_env()
    if lua_dir is None:
        print("Set BALATRO_LUA_DIR to the directory that holds the game's Lua source.")
        return 1
    game = LuaGame(lua_dir)

    inputs: list[tuple[list[CardSpec], str]] = [
        (parse_cards(text), flags) for text, flags in CRAFTED
    ]
    inputs.extend(random_hand_cases(seed=20261110, count=1000))
    cases = []
    for cards, flags in inputs:
        result = game.evaluate(cards, flags)
        cases.append({"cards": format_cards(cards), "flags": flags, **result})
    _write("lua_hands.json", {"cases": cases})

    amounts = {
        str(scaling): [game.blind_amount(ante, scaling) for ante in range(-1, 61)]
        for scaling in (1, 2, 3)
    }
    _write("lua_blind_amounts.json", {"first_ante": -1, "amounts": amounts})

    rules = LuaRules(lua_dir)
    runs = (
        ("scoring", score, random_scoring_scenarios, 200),
        ("round_end", end_of_round, random_round_end_scenarios, 120),
        ("discard", discard, random_discard_scenarios, 120),
    )
    payload: dict[str, object] = {}
    for name, run, scenarios, count in runs:
        payload[name] = [
            {"scenario": scenario, "expected": run(rules, scenario)}
            for scenario in scenarios(20261111, count, VERIFIED_JOKERS, seals=VERIFIED_SEALS)
        ]
    consumables = sorted(CONSUMABLES)
    payload["consumable"] = [
        {"scenario": scenario, "expected": use_consumable(rules, scenario)}
        for scenario in random_consumable_scenarios(20261111, 200, consumables, VERIFIED_JOKERS)
    ]
    payload["shop"] = [
        {"scenario": scenario, "expected": shop(rules, scenario)}
        for scenario in random_shop_scenarios(20261111, 80, sorted(JOKERS), consumables)
    ]
    _write("lua_rules.json.gz", payload)
    return 0


def _write(name: str, payload: dict[str, object]) -> None:
    """Write ``payload`` as JSON, with each element of a list on its own line so
    regenerated vectors give readable diffs."""
    header = {"generated_by": "tools/lua_harness/generate_vectors.py", "game_version": "1.0.1o"}
    parts = []
    for key, value in {**header, **payload}.items():
        if isinstance(value, list):
            rows = ",\n".join(json.dumps(row, separators=(",", ":")) for row in value)
            parts.append(f"{json.dumps(key)}:[\n{rows}\n]")
        else:
            parts.append(f"{json.dumps(key)}:{json.dumps(value, separators=(',', ':'))}")
    path = OUT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "{" + ",\n".join(parts) + "}\n"
    if path.suffix == ".gz":
        # The situations with their results are large; they are stored compressed.
        path.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
    else:
        path.write_text(text, encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    sys.exit(main())
