"""Blind definitions, score targets, boss selection, and the boss rules that
decide which cards are debuffed or dealt face down.

Score targets use IEEE doubles exactly as the game does, so in very long endless
runs they grow to infinity and then become NaN, as they do in the original.

Source: game.lua ``P_BLINDS``; functions/misc_functions.lua ``get_blind_amount``;
functions/common_events.lua ``get_new_boss``; blind.lua ``Blind:debuff_card``
and ``Blind:stay_flipped``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .content import BLIND_DATA
from .effects import SUIT_NAMES, is_face, is_suit
from .rng import Rng
from .scope import Scope
from .state import PlayingCard, RunState

__all__ = [
    "BLINDS",
    "BlindInfo",
    "boss_candidates",
    "choose_boss",
    "debuffs_card",
    "get_blind_amount",
    "keeps_joker_debuffs",
    "lua_floor",
    "lua_mod",
    "stays_face_down",
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

    @property
    def debuff(self) -> Mapping[str, object]:
        """The boss's debuff rule, such as ``{"suit": "Clubs"}`` or ``{"h_size_ge": 5}``."""
        rule: Mapping[str, object] = BLIND_DATA[self.key].get("debuff", {})
        return rule


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


def lua_floor(number: float) -> float:
    """``math.floor`` on a Lua number: infinities and NaN pass through unchanged."""
    if math.isfinite(number):
        return float(math.floor(number))
    return number


def lua_mod(dividend: float, divisor: float) -> float:
    """Lua's ``dividend % divisor`` on numbers: ``a - floor(a / b) * b``."""
    return dividend - lua_floor(dividend / divisor) * divisor


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
    # The game's formula: a * (b + (k * c) ^ d) ^ c, with c the antes past 8.
    ante_8_amount = float(amounts[7])
    antes_past_8 = float(ante - 8)
    exponent = 1 + 0.2 * (ante - 8)
    growth = 1.6 + _lua_pow(0.75 * antes_past_8, exponent)
    amount = lua_floor(ante_8_amount * _lua_pow(growth, antes_past_8))
    magnitude = _lua_log10(amount) - 1
    return amount - lua_mod(amount, _lua_pow(10.0, lua_floor(magnitude)))


def choose_boss(
    rng: Rng,
    ante: int,
    win_ante: int,
    bosses_used: dict[str, int],
    banned: frozenset[str] = frozenset(),
) -> str:
    """Pick the boss blind for ``ante`` and count it in ``bosses_used``."""
    boss = rng.choice("boss", boss_candidates(ante, win_ante, bosses_used, banned))
    bosses_used[boss] = bosses_used.get(boss, 0) + 1
    return boss


def boss_candidates(
    ante: int,
    win_ante: int,
    bosses_used: Mapping[str, int],
    banned: frozenset[str] = frozenset(),
) -> list[str]:
    """The bosses that can be picked for ``ante``, sorted by key.

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
            is_eligible = ante % win_ante == 0 and ante >= 2
        else:
            is_eligible = info.boss_min <= effective_ante and (
                effective_ante % win_ante != 0 or ante < 2
            )
        if is_eligible:
            eligible[info.key] = bosses_used.get(info.key, 0)
    fewest = min(eligible.values())
    return sorted(key for key, uses in eligible.items() if uses <= fewest)


def _active_boss(state: RunState) -> BlindInfo | None:
    """The boss being played, unless it is disabled."""
    blind = state.blind
    if blind is None or blind.disabled or not BLINDS[blind.key].boss:
        return None
    return BLINDS[blind.key]


def debuffs_card(state: RunState, card: PlayingCard) -> bool:
    """Whether the blind being played debuffs a playing card.

    Source: blind.lua ``Blind:debuff_card``.
    """
    info = _active_boss(state)
    if info is None:
        return False
    rule = info.debuff
    suit = rule.get("suit")
    if suit is not None and is_suit(state, card, SUIT_NAMES[str(suit)], bypass_debuff=True):
        return True
    if rule.get("is_face") == "face" and is_face(state, card, from_boss=True):
        return True
    if info.key == "bl_pillar" and card.played_this_ante:
        return True
    return info.key == "bl_final_leaf"


def keeps_joker_debuffs(state: RunState) -> bool:
    """Whether jokers keep their debuffs when debuffs are re-checked: Crimson Heart
    decides them itself. Source: blind.lua ``Blind:debuff_card``."""
    info = _active_boss(state)
    return info is not None and info.key == "bl_final_heart"


def stays_face_down(scope: Scope, card: PlayingCard) -> bool:
    """Whether a card drawn into the hand stays face down.

    The Wheel turns 1 in 7 cards down, The House the whole first hand, The Mark
    every face card, and The Fish the cards drawn after a hand is played.
    Source: blind.lua ``Blind:stay_flipped``.
    """
    state = scope.state
    info = _active_boss(state)
    if info is None:
        return False
    if info.key == "bl_wheel" and scope.chance("wheel", 7):
        return True
    if info.key == "bl_house" and not (
        state.hands_played_this_round or state.discards_used_this_round
    ):
        return True
    if info.key == "bl_mark" and is_face(state, card, from_boss=True):
        return True
    assert state.blind is not None
    return info.key == "bl_fish" and state.blind.prepped


def _lua_log10(number: float) -> float:
    if math.isnan(number):
        return number
    if number == math.inf:
        return math.inf
    return math.log10(number)


def _lua_pow(base: float, exponent: float) -> float:
    try:
        return base**exponent
    except OverflowError:
        return math.inf
