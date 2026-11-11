from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest

from balatro_gym.sim.blinds import BLINDS, choose_boss, get_blind_amount, lua_floor, lua_mod
from balatro_gym.sim.rng import Rng

VECTORS = Path(__file__).parent / "data" / "lua_blind_amounts.json"


def test_matches_original_game_vectors() -> None:
    data = json.loads(VECTORS.read_text())
    first = data["first_ante"]
    for scaling, amounts in data["amounts"].items():
        for offset, expected in enumerate(amounts):
            actual = get_blind_amount(first + offset, int(scaling))
            assert actual == expected or (math.isnan(actual) and math.isnan(expected)), (
                scaling,
                first + offset,
            )


@pytest.mark.parametrize(
    ("ante", "scaling", "amount"),
    [(0, 1, 100), (1, 1, 300), (8, 1, 50000), (9, 1, 110000), (2, 2, 900), (8, 3, 200000)],
)
def test_known_amounts(ante: int, scaling: int, amount: float) -> None:
    assert get_blind_amount(ante, scaling) == amount


def test_deep_endless_targets_overflow_like_the_game() -> None:
    assert math.isnan(get_blind_amount(60))


def test_lua_number_helpers() -> None:
    assert lua_floor(-1.5) == -2.0
    assert lua_floor(math.inf) == math.inf
    assert lua_mod(7.0, 3.0) == 1.0
    assert lua_mod(-7.0, 3.0) == 2.0


def test_blind_table() -> None:
    bosses = [info for info in BLINDS.values() if info.boss]
    assert len(bosses) == 28
    assert sum(info.showdown for info in bosses) == 5
    assert BLINDS["bl_big"].mult == 1.5
    assert BLINDS["bl_wall"].mult == 4


def test_boss_respects_minimum_ante() -> None:
    for seed in range(200):
        boss = choose_boss(Rng(str(seed)), ante=1, win_ante=8, bosses_used={})
        info = BLINDS[boss]
        assert info.boss_min <= 1
        assert not info.showdown


def test_showdown_boss_in_final_ante() -> None:
    for ante in (8, 16):
        boss = choose_boss(Rng("s"), ante=ante, win_ante=8, bosses_used={})
        assert BLINDS[boss].showdown


def test_bosses_do_not_repeat_until_all_were_used() -> None:
    eligible = [i.key for i in BLINDS.values() if i.boss and not i.showdown and i.boss_min <= 3]
    rng = Rng("s")
    used: dict[str, int] = {}
    seen = [choose_boss(rng, ante=3, win_ante=8, bosses_used=used) for _ in eligible]
    assert Counter(seen) == Counter(eligible)


def test_banned_bosses_are_never_chosen() -> None:
    eligible = {i.key for i in BLINDS.values() if i.boss and not i.showdown and i.boss_min <= 1}
    banned = frozenset(eligible - {"bl_hook"})
    boss = choose_boss(Rng("s"), ante=1, win_ante=8, bosses_used={}, banned=banned)
    assert boss == "bl_hook"
