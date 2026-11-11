"""Run functions from the original game's Lua source under LuaJIT.

The game's source is never part of this repository. The harness reads the
maintainer's local copy from the directory named by ``BALATRO_LUA_DIR`` and runs
self-contained functions with simple stand-ins for the parts of the game they
touch (the global ``G`` table, ``find_joker``, sprites).

LuaJIT comes from the ``lupa`` package, the same Lua implementation the game
runs on.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "CardSpec",
    "LuaGame",
    "format_cards",
    "lua_dir_from_env",
    "parse_cards",
    "random_hand_cases",
]

ENV_VAR = "BALATRO_LUA_DIR"

_PRELUDE = r"""
Moveable = {}
function Moveable:extend()
  local cls = {}
  cls.__index = cls
  return setmetatable(cls, {__index = self})
end
G = {GAME = {modifiers = {}}, P_CARDS = {}, C = {SUITS = {}}, FUNCS = {}}
"""

_HELPERS = r"""
HARNESS_JOKERS = {}
function find_joker(name)
  if HARNESS_JOKERS[name] then return {true} end
  return {}
end
function localize(text) return text end
function Card:set_sprites() end

function harness_card(id, value, suit, enhancement, debuffed)
  local card = setmetatable({ID = id, config = {}, debuff = debuffed}, Card)
  card.unique_val = 1 - id / 1603301
  card.ability = {effect = enhancement or '', name = enhancement or ''}
  card:set_base({value = value, suit = suit, name = ''}, true)
  return card
end

local function index_groups(index_of, groups)
  local out = {}
  for gi, group in ipairs(groups) do
    local indices = {}
    for ci, card in ipairs(group) do indices[ci] = index_of[card] end
    out[gi] = indices
  end
  return out
end

function harness_evaluate(cards, four_fingers, shortcut, smeared)
  HARNESS_JOKERS = {
    ['Four Fingers'] = four_fingers,
    ['Shortcut'] = shortcut,
    ['Smeared Joker'] = smeared,
  }
  local index_of = {}
  for i, card in ipairs(cards) do index_of[card] = i - 1 end
  local text, _, hands, scoring, display = G.FUNCS.get_poker_hand_info(cards)
  local groups = {}
  for name, value in pairs(hands) do
    if name ~= 'top' then groups[name] = index_groups(index_of, value) end
  end
  local scoring_indices = {}
  for i, card in ipairs(scoring) do scoring_indices[i] = index_of[card] end
  return text, display, scoring_indices, groups
end

function harness_blind_amount(ante, scaling)
  G.GAME.modifiers.scaling = scaling
  return get_blind_amount(ante)
end
"""

_SOURCES = ("functions/misc_functions.lua", "card.lua", "functions/state_events.lua")

_VALUES = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "Jack",
    12: "Queen",
    13: "King",
    14: "Ace",
}
_SUITS = {"S": "Spades", "H": "Hearts", "C": "Clubs", "D": "Diamonds"}
_ENHANCEMENTS = {None: None, "m_wild": "Wild Card", "m_stone": "Stone Card"}
_ENHANCEMENT_CODES = {"w": "m_wild", "s": "m_stone"}
_RANK_CHARS = "23456789TJQKA"


@dataclass(frozen=True)
class CardSpec:
    """A playing card as the harness sees it. ``uid`` must be positive."""

    uid: int
    rank: int
    suit: str
    enhancement: str | None = None
    debuffed: bool = False


def format_cards(cards: Sequence[CardSpec]) -> str:
    """Compact text form: ``"AS KHw 2Cs 5D!"``, where ``w`` is Wild, ``s`` is Stone
    and ``!`` is debuffed. Cards get uids 1, 2, 3, ... in order when parsed back.
    """
    codes = {value: code for code, value in _ENHANCEMENT_CODES.items()}
    parts = []
    for card in cards:
        text = _RANK_CHARS[card.rank - 2] + card.suit
        if card.enhancement:
            text += codes[card.enhancement]
        if card.debuffed:
            text += "!"
        parts.append(text)
    return " ".join(parts)


def parse_cards(text: str) -> list[CardSpec]:
    """Inverse of :func:`format_cards`."""
    cards = []
    for uid, token in enumerate(text.split(), start=1):
        debuffed = token.endswith("!")
        token = token.rstrip("!")
        enhancement = _ENHANCEMENT_CODES.get(token[2:]) if len(token) > 2 else None
        cards.append(
            CardSpec(uid, _RANK_CHARS.index(token[0]) + 2, token[1], enhancement, debuffed)
        )
    return cards


def random_hand_cases(seed: int, count: int) -> Iterator[tuple[list[CardSpec], str]]:
    """Random hands biased towards straights, flushes and repeated ranks.

    Yields ``(cards, flags)``, where ``flags`` holds ``f`` (Four Fingers), ``s``
    (Shortcut) and ``m`` (Smeared Joker).
    """
    rnd = random.Random(seed)
    for _ in range(count):
        size = rnd.choice((1, 2, 3, 4, 4, 5, 5, 5, 5, 5, 5, 6, 7))
        style = rnd.random()
        suits = "SH" if rnd.random() < 0.4 else "SHCD"
        cards = []
        for uid in range(1, size + 1):
            if style < 0.3:
                rank = rnd.randint(2, 14)
            elif style < 0.6:
                rank = rnd.choice((2, 3, 4, 5, 6, 10, 11, 12, 13, 14))
            else:
                rank = rnd.choice((7, 7, 8, 8, 9))
            enhancement = rnd.choices((None, "m_wild", "m_stone"), (0.85, 0.1, 0.05))[0]
            cards.append(CardSpec(uid, rank, rnd.choice(suits), enhancement, rnd.random() < 0.07))
        flags = "".join(flag for flag in "fsm" if rnd.random() < 0.25)
        yield cards, flags


def lua_dir_from_env() -> Path | None:
    """The local Lua source directory, or ``None`` if it isn't configured."""
    value = os.environ.get(ENV_VAR)
    if not value:
        return None
    path = Path(value)
    return path if (path / "card.lua").is_file() else None


class LuaGame:
    """A LuaJIT runtime with the game's hand evaluation and blind amounts loaded."""

    def __init__(self, lua_dir: Path) -> None:
        from lupa import luajit21  # type: ignore[import-untyped]

        self._lua: Any = luajit21.LuaRuntime(unpack_returned_tuples=True)
        self._lua.execute(_PRELUDE)
        for relative in _SOURCES:
            self._lua.execute((lua_dir / relative).read_text(encoding="utf-8"))
        self._lua.execute(_HELPERS)
        self._globals: Any = self._lua.globals()

    def evaluate(self, cards: Sequence[CardSpec], flags: str = "") -> dict[str, Any]:
        """Evaluate ``cards`` with the game's own code.

        ``flags`` holds ``f``, ``s`` and ``m`` for Four Fingers, Shortcut and
        Smeared Joker. Returns the hand name, display name, scoring card indices, and the groups
        of every contained hand, all as plain Python values.
        """
        make = self._globals.harness_card
        lua_cards = self._lua.table(
            *(
                make(
                    card.uid,
                    _VALUES[card.rank],
                    _SUITS[card.suit],
                    _ENHANCEMENTS[card.enhancement],
                    card.debuffed,
                )
                for card in cards
            )
        )
        text, display, scoring, groups = self._globals.harness_evaluate(
            lua_cards, "f" in flags, "s" in flags, "m" in flags
        )
        return {
            "top": None if text == "NULL" else text,
            "display": "" if display == "NULL" else display,
            "scoring": _to_list(scoring),
            "groups": {
                name: [_to_list(group) for group in _values(value)]
                for name, value in groups.items()
                if len(value) > 0
            },
        }

    def blind_amount(self, ante: int, scaling: int) -> float:
        return float(self._globals.harness_blind_amount(ante, scaling))


def _values(table: Any) -> list[Any]:
    return [table[i] for i in range(1, len(table) + 1)]


def _to_list(table: Any) -> list[int]:
    return [int(value) for value in _values(table)]
