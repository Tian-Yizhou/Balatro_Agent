"""The engine against the game's own rule code: scoring, the round end and discards.

``lua_rules.json.gz`` holds situations with the results the original game computed for
them, written by ``tools/lua_harness/generate_vectors.py``; CI checks the engine
against them. Where ``BALATRO_LUA_DIR`` points to the game's source, the live
tests also compare thousands of fresh random situations directly.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from balatro_gym.sim.consumables import IMPLEMENTED as CONSUMABLES
from balatro_gym.sim.content import pool
from balatro_gym.sim.effects import JOKERS
from scenarios import (
    compare_consumable_use,
    compare_open_pack,
    compare_results,
    compare_shop,
    engine_discard,
    engine_end_of_round,
    engine_open_pack,
    engine_score,
    engine_shop,
    engine_use_consumable,
)

DATA = Path(__file__).parent / "data" / "lua_rules.json.gz"
HARNESS = Path(__file__).resolve().parents[2] / "tools" / "lua_harness"

ENGINE_RUNNERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "scoring": engine_score,
    "round_end": engine_end_of_round,
    "discard": engine_discard,
}


@pytest.mark.parametrize("kind", sorted(ENGINE_RUNNERS))
def test_matches_original_game_vectors(kind: str) -> None:
    cases = _vectors()[kind]
    assert len(cases) >= 100
    mismatches = []
    for number, case in enumerate(cases):
        scenario = case["scenario"]
        ours = ENGINE_RUNNERS[kind](scenario)
        problems = compare_results(ours, _json_keys_to_int(case["expected"]), scenario["jokers"])
        if problems:
            mismatches.append((number, problems))
    assert mismatches == []


def _vectors() -> dict[str, Any]:
    return json.loads(gzip.decompress(DATA.read_bytes()))


def _load(name: str) -> ModuleType:
    """Import a harness module by path; its own imports of sibling modules need the
    harness directory on the import path."""
    if str(HARNESS) not in sys.path:
        sys.path.append(str(HARNESS))
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
VERIFIED_SEALS = ("Red", "Gold", "Blue", "Purple")
IMPLEMENTED = set(JOKERS) | set(CONSUMABLES)
PACKS = [center.key for center in pool("Booster")]


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


def _json_keys_to_int(result: dict[str, Any]) -> dict[str, Any]:
    """JSON turns the uid keys of ``jokers`` into strings; turn them back."""
    jokers = result.get("jokers")
    if isinstance(jokers, dict):
        return result | {"jokers": {int(uid): values for uid, values in jokers.items()}}
    return result


def test_consumables_match_original_game_vectors() -> None:
    cases = _vectors()["consumable"]
    assert len(cases) >= 100
    mismatches = []
    for number, case in enumerate(cases):
        scenario = case["scenario"]
        problems = compare_consumable_use(
            scenario, engine_use_consumable(scenario), case["expected"]
        )
        if problems:
            mismatches.append((number, problems))
    assert mismatches == []


def test_shop_matches_original_game_vectors() -> None:
    cases = _vectors()["shop"]
    assert len(cases) >= 50
    mismatches = []
    for number, case in enumerate(cases):
        expected = _json_keys_to_int(case["expected"])
        problems = compare_shop(engine_shop(case["scenario"]), expected, IMPLEMENTED)
        if problems:
            mismatches.append((number, problems))
    assert mismatches == []


def test_packs_match_original_game_vectors() -> None:
    cases = _vectors()["pack"]
    assert len(cases) >= len(PACKS)
    mismatches = []
    for number, case in enumerate(cases):
        expected = case["expected"]
        ours = engine_open_pack(case["scenario"], expected["cards"])
        problems = compare_open_pack(ours, expected)
        if problems:
            mismatches.append((number, problems))
    assert mismatches == []


@live
def test_random_pack_openings_match(lua: Any) -> None:
    scenarios = _load("consumable_scenarios")
    open_in_game = _load("rules").open_pack
    mismatches = []
    for number, scenario in enumerate(
        scenarios.random_pack_scenarios(1, 640, PACKS, VERIFIED_JOKERS)
    ):
        game = open_in_game(lua, scenario)
        problems = compare_open_pack(engine_open_pack(scenario, game["cards"]), game)
        if problems:
            mismatches.append((number, problems))
    assert mismatches == []


@live
def test_random_consumable_uses_match(lua: Any) -> None:
    scenarios = _load("consumable_scenarios")
    use_in_game = _load("rules").use_consumable
    mismatches = []
    for number, scenario in enumerate(
        scenarios.random_consumable_scenarios(1, 2000, sorted(CONSUMABLES), VERIFIED_JOKERS)
    ):
        problems = compare_consumable_use(
            scenario, engine_use_consumable(scenario), use_in_game(lua, scenario)
        )
        if problems:
            mismatches.append((number, problems))
    assert mismatches == []


@live
def test_random_shops_match(lua: Any) -> None:
    scenarios = _load("consumable_scenarios")
    shop_in_game = _load("rules").shop
    mismatches = []
    for number, scenario in enumerate(
        scenarios.random_shop_scenarios(1, 800, sorted(JOKERS), sorted(CONSUMABLES))
    ):
        problems = compare_shop(engine_shop(scenario), shop_in_game(lua, scenario), IMPLEMENTED)
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
