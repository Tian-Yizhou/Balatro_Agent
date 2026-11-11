"""The engine against the game's own rule code: scoring, the round end and discards.

``lua_rules.json`` holds situations with the results the original game computed for
them, written by ``tools/lua_harness/generate_vectors.py``; CI checks the engine
against them. Where ``BALATRO_LUA_DIR`` points to the game's source, the live
tests also compare thousands of fresh random situations directly.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from balatro_gym.sim.effects import JOKERS
from scenarios import (
    compare_results,
    engine_discard,
    engine_end_of_round,
    engine_score,
)

DATA = Path(__file__).parent / "data" / "lua_rules.json"
HARNESS = Path(__file__).resolve().parents[2] / "tools" / "lua_harness"

ENGINE_RUNNERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "scoring": engine_score,
    "round_end": engine_end_of_round,
    "discard": engine_discard,
}


@pytest.mark.parametrize("kind", sorted(ENGINE_RUNNERS))
def test_matches_original_game_vectors(kind: str) -> None:
    cases = json.loads(DATA.read_text())[kind]
    assert len(cases) >= 100
    mismatches = []
    for number, case in enumerate(cases):
        scenario = case["scenario"]
        ours = ENGINE_RUNNERS[kind](scenario)
        problems = compare_results(ours, case["expected"], scenario["jokers"])
        if problems:
            mismatches.append((number, problems))
    assert mismatches == []


def _load(name: str) -> ModuleType:
    path = HARNESS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"lua_harness_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = _load("harness")
LUA_DIR = harness.lua_dir_from_env()
live = pytest.mark.skipif(
    LUA_DIR is None, reason="BALATRO_LUA_DIR does not point to the game's Lua source"
)
VERIFIED_JOKERS = sorted(key for key in JOKERS if key != "j_misprint")
VERIFIED_SEALS = ("Red", "Gold", "Blue")


@pytest.fixture(scope="module")
def lua() -> Any:
    pytest.importorskip("lupa")
    return _load("rules").LuaRules(LUA_DIR)


@live
@pytest.mark.parametrize(
    ("kind", "generator", "lua_runner", "count"),
    [
        ("scoring", "random_scoring_scenarios", "score", 2000),
        ("round_end", "random_round_end_scenarios", "end_of_round", 800),
        ("discard", "random_discard_scenarios", "discard", 800),
    ],
)
def test_random_situations_match(
    lua: Any, kind: str, generator: str, lua_runner: str, count: int
) -> None:
    scenarios = _load("scoring_scenarios")
    run_in_game = getattr(_load("rules"), lua_runner)
    mismatches = []
    for number, scenario in enumerate(
        getattr(scenarios, generator)(1, count, VERIFIED_JOKERS, seals=VERIFIED_SEALS)
    ):
        game = run_in_game(lua, scenario)
        problems = compare_results(ENGINE_RUNNERS[kind](scenario), game, scenario["jokers"])
        if problems:
            mismatches.append((number, problems))
    assert mismatches == []


@live
def test_misprint_range_matches(lua: Any) -> None:
    """Misprint adds a random mult; the game's and the engine's ranges must agree."""
    score = _load("rules").score

    def scenario(seed: int) -> dict[str, Any]:
        return {
            "played": [{"uid": 1, "rank": 14, "suit": "S"}],
            "held": [],
            "jokers": [{"uid": 2, "key": "j_misprint"}],
            "levels": {},
            "dollars": 0,
            "hands_left": 1,
            "discards_left": 1,
            "probability": 1,
            "seed": seed,
        }

    in_game = {int(score(lua, scenario(seed))["score"]) // 16 - 1 for seed in range(400)}
    in_engine = set()
    for seed in range(400):
        case = scenario(seed)
        in_engine.add(int(engine_score(case)["score"]) // 16 - 1)
    assert in_game == set(range(24))
    assert in_engine == set(range(24))
