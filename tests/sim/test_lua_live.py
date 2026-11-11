"""Compare the engine with the original game's Lua code, run live under LuaJIT.

These tests run only where the game's source is available locally, named by the
``BALATRO_LUA_DIR`` environment variable. CI checks the committed vectors instead.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from balatro_gym.sim.blinds import get_blind_amount
from scenarios import describe

HARNESS = Path(__file__).resolve().parents[2] / "tools" / "lua_harness" / "harness.py"


def _load_harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lua_harness", HARNESS)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = _load_harness()
LUA_DIR = harness.lua_dir_from_env()

pytestmark = pytest.mark.skipif(
    LUA_DIR is None, reason="BALATRO_LUA_DIR does not point to the game's Lua source"
)


@pytest.fixture(scope="module")
def lua() -> Any:
    pytest.importorskip("lupa")
    return harness.LuaGame(LUA_DIR)


def test_random_hands_match(lua: Any) -> None:
    mismatches = []
    for specs, flags in harness.random_hand_cases(seed=1, count=10_000):
        text = harness.format_cards(specs)
        expected = lua.evaluate(specs, flags)
        if describe(text, flags) != expected:
            mismatches.append((text, flags))
    assert mismatches == []


def test_blind_amounts_match(lua: Any) -> None:
    for scaling in (1, 2, 3):
        for ante in range(-2, 120):
            expected = lua.blind_amount(ante, scaling)
            actual = get_blind_amount(ante, scaling)
            assert actual == expected or (math.isnan(actual) and math.isnan(expected))
