"""Blind definitions, score targets and boss selection.

Score targets use IEEE doubles exactly as the game does, so in very long endless
runs they grow to infinity and then become NaN, as they do in the original.

Source: game.lua ``P_BLINDS``; functions/misc_functions.lua ``get_blind_amount``;
functions/common_events.lua ``get_new_boss``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .rng import Rng

__all__ = [
    "BLINDS",
    "BlindInfo",
    "choose_boss",
    "get_blind_amount",
    "lua_floor",
    "lua_mod",
]


@dataclass(frozen=True, slots=True)
class BlindInfo:
    """Static data of one blind. ``boss_min`` is the first ante a boss can appear in."""

    key: str
    name: str
    mult: float
    dollars: int
    boss: bool = False
    showdown: bool = False
    boss_min: int = 0


def _boss(key: str, name: str, mult: float = 2, minimum: int = 1) -> BlindInfo:
    return BlindInfo(key, name, mult, 5, boss=True, boss_min=minimum)


def _showdown(key: str, name: str, mult: float = 2) -> BlindInfo:
    return BlindInfo(key, name, mult, 8, boss=True, showdown=True, boss_min=10)


#: Source: game.lua ``P_BLINDS``. Boss effects are implemented separately.
BLINDS: Mapping[str, BlindInfo] = {
    info.key: info
    for info in (
        BlindInfo("bl_small", "Small Blind", 1, 3),
        BlindInfo("bl_big", "Big Blind", 1.5, 4),
        _boss("bl_ox", "The Ox", minimum=6),
        _boss("bl_hook", "The Hook"),
        _boss("bl_mouth", "The Mouth", minimum=2),
        _boss("bl_fish", "The Fish", minimum=2),
        _boss("bl_club", "The Club"),
        _boss("bl_manacle", "The Manacle"),
        _boss("bl_tooth", "The Tooth", minimum=3),
        _boss("bl_wall", "The Wall", mult=4, minimum=2),
        _boss("bl_house", "The House", minimum=2),
        _boss("bl_mark", "The Mark", minimum=2),
        _boss("bl_wheel", "The Wheel", minimum=2),
        _boss("bl_arm", "The Arm", minimum=2),
        _boss("bl_psychic", "The Psychic"),
        _boss("bl_goad", "The Goad"),
        _boss("bl_water", "The Water", minimum=2),
        _boss("bl_eye", "The Eye", minimum=3),
        _boss("bl_plant", "The Plant", minimum=4),
        _boss("bl_needle", "The Needle", mult=1, minimum=2),
        _boss("bl_head", "The Head"),
        _boss("bl_window", "The Window"),
        _boss("bl_serpent", "The Serpent", minimum=5),
        _boss("bl_pillar", "The Pillar"),
        _boss("bl_flint", "The Flint", minimum=2),
        _showdown("bl_final_bell", "Cerulean Bell"),
        _showdown("bl_final_leaf", "Verdant Leaf"),
        _showdown("bl_final_vessel", "Violet Vessel", mult=6),
        _showdown("bl_final_acorn", "Amber Acorn"),
        _showdown("bl_final_heart", "Crimson Heart"),
    )
}

# Base score targets for antes 1-8, by ``G.GAME.modifiers.scaling`` (raised by stakes).
_BASE_AMOUNTS = {
    1: (300, 800, 2000, 5000, 11000, 20000, 35000, 50000),
    2: (300, 900, 2600, 8000, 20000, 36000, 60000, 100000),
    3: (300, 1000, 3200, 9000, 25000, 60000, 110000, 200000),
}


def lua_floor(x: float) -> float:
    """``math.floor`` on a Lua number: infinities and NaN pass through unchanged."""
    if math.isfinite(x):
        return float(math.floor(x))
    return x


def lua_mod(a: float, b: float) -> float:
    """Lua's ``a % b`` on numbers: ``a - floor(a / b) * b``."""
    return a - lua_floor(a / b) * b


def get_blind_amount(ante: int, scaling: int = 1) -> float:
    """The base score target of an ante, before the blind's own multiplier.

    Beyond ante 8 the target grows super-exponentially and is rounded down to two
    significant digits.
    """
    amounts = _BASE_AMOUNTS[scaling]
    if ante < 1:
        return 100.0
    if ante <= 8:
        return float(amounts[ante - 1])
    k = 0.75
    a, b, c, d = float(amounts[7]), 1.6, float(ante - 8), 1 + 0.2 * (ante - 8)
    amount = lua_floor(a * _lua_pow(b + _lua_pow(k * c, d), c))
    magnitude = _lua_log10(amount) - 1
    return amount - lua_mod(amount, _lua_pow(10.0, lua_floor(magnitude)))


def choose_boss(
    rng: Rng,
    ante: int,
    win_ante: int,
    bosses_used: dict[str, int],
    banned: frozenset[str] = frozenset(),
) -> str:
    """Pick the boss blind for ``ante`` and count it in ``bosses_used``.

    Showdown bosses appear only in antes that are multiples of ``win_ante``.
    Among the eligible bosses, only those used the fewest times so far can be
    picked, so bosses rarely repeat before every one has been seen.
    """
    effective_ante = max(1, ante)
    eligible: dict[str, int] = {}
    for info in BLINDS.values():
        if not info.boss or info.key in banned:
            continue
        if info.showdown:
            ok = ante % win_ante == 0 and ante >= 2
        else:
            ok = info.boss_min <= effective_ante and (effective_ante % win_ante != 0 or ante < 2)
        if ok:
            eligible[info.key] = bosses_used.get(info.key, 0)
    fewest = min(eligible.values())
    candidates = sorted(key for key, uses in eligible.items() if uses <= fewest)
    boss = rng.choice("boss", candidates)
    bosses_used[boss] = bosses_used.get(boss, 0) + 1
    return boss


def _lua_log10(x: float) -> float:
    if math.isnan(x):
        return x
    if x == math.inf:
        return math.inf
    return math.log10(x)


def _lua_pow(base: float, exponent: float) -> float:
    try:
        return base**exponent
    except OverflowError:
        return math.inf
