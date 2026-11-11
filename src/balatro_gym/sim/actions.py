"""Atomic actions: the canonical interface for agents and replays.

Each action mirrors one click in the game's interface. Indices are positions
(0-based, left to right) in the area the action refers to. Replays store actions
in their list form, for example ``["toggle_card", 3]`` or ``["play"]``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

__all__ = [
    "ACTION_TYPES",
    "Action",
    "Buy",
    "BuyAndUse",
    "BuyPack",
    "CashOut",
    "ContinueEndless",
    "Discard",
    "EndRun",
    "EndShop",
    "MoveCard",
    "MoveJoker",
    "Pick",
    "Play",
    "RedeemVoucher",
    "Reroll",
    "SelectBlind",
    "SellConsumable",
    "SellJoker",
    "SkipBlind",
    "SkipPack",
    "SortByRank",
    "SortBySuit",
    "ToggleCard",
    "UseConsumable",
    "parse_action",
]


@dataclass(frozen=True, slots=True)
class Action:
    """Base class of all actions. ``name`` is the action's serialized name."""

    name: ClassVar[str] = "action"

    def to_list(self) -> list[str | int]:
        """The serialized form: the name followed by the action's arguments."""
        return [self.name, *(getattr(self, f.name) for f in dataclasses.fields(self))]

    def __str__(self) -> str:
        args = ", ".join(str(getattr(self, f.name)) for f in dataclasses.fields(self))
        return f"{self.name}({args})"


# Card and item management, wherever the game allows it.


@dataclass(frozen=True, slots=True)
class ToggleCard(Action):
    """Select or deselect the card at ``index`` in hand."""

    name: ClassVar[str] = "toggle_card"
    index: int


@dataclass(frozen=True, slots=True)
class MoveCard(Action):
    """Take the hand card at ``source`` and insert it at position ``target``."""

    name: ClassVar[str] = "move_card"
    source: int
    target: int


@dataclass(frozen=True, slots=True)
class MoveJoker(Action):
    """Take the joker at ``source`` and insert it at position ``target``."""

    name: ClassVar[str] = "move_joker"
    source: int
    target: int


@dataclass(frozen=True, slots=True)
class SellJoker(Action):
    name: ClassVar[str] = "sell_joker"
    index: int


@dataclass(frozen=True, slots=True)
class SellConsumable(Action):
    name: ClassVar[str] = "sell_consumable"
    index: int


@dataclass(frozen=True, slots=True)
class UseConsumable(Action):
    """Use the consumable at ``index``; the selected hand cards are its targets."""

    name: ClassVar[str] = "use_consumable"
    index: int


# Blind selection.


@dataclass(frozen=True, slots=True)
class SelectBlind(Action):
    name: ClassVar[str] = "select_blind"


@dataclass(frozen=True, slots=True)
class SkipBlind(Action):
    name: ClassVar[str] = "skip_blind"


# Playing a round.


@dataclass(frozen=True, slots=True)
class Play(Action):
    """Play the selected cards. They score from left to right."""

    name: ClassVar[str] = "play"


@dataclass(frozen=True, slots=True)
class Discard(Action):
    name: ClassVar[str] = "discard"


@dataclass(frozen=True, slots=True)
class SortByRank(Action):
    name: ClassVar[str] = "sort_by_rank"


@dataclass(frozen=True, slots=True)
class SortBySuit(Action):
    name: ClassVar[str] = "sort_by_suit"


# Round evaluation.


@dataclass(frozen=True, slots=True)
class CashOut(Action):
    name: ClassVar[str] = "cash_out"


# Shop.


@dataclass(frozen=True, slots=True)
class Buy(Action):
    name: ClassVar[str] = "buy"
    index: int


@dataclass(frozen=True, slots=True)
class BuyAndUse(Action):
    name: ClassVar[str] = "buy_and_use"
    index: int


@dataclass(frozen=True, slots=True)
class RedeemVoucher(Action):
    name: ClassVar[str] = "redeem_voucher"
    index: int


@dataclass(frozen=True, slots=True)
class BuyPack(Action):
    name: ClassVar[str] = "buy_pack"
    index: int


@dataclass(frozen=True, slots=True)
class Reroll(Action):
    name: ClassVar[str] = "reroll"


@dataclass(frozen=True, slots=True)
class EndShop(Action):
    name: ClassVar[str] = "end_shop"


# Booster packs.


@dataclass(frozen=True, slots=True)
class Pick(Action):
    name: ClassVar[str] = "pick"
    index: int


@dataclass(frozen=True, slots=True)
class SkipPack(Action):
    name: ClassVar[str] = "skip_pack"


# After winning.


@dataclass(frozen=True, slots=True)
class ContinueEndless(Action):
    name: ClassVar[str] = "continue_endless"


@dataclass(frozen=True, slots=True)
class EndRun(Action):
    name: ClassVar[str] = "end_run"


#: Every action type, by serialized name.
ACTION_TYPES: dict[str, type[Action]] = {
    cls.name: cls
    for cls in (
        ToggleCard,
        MoveCard,
        MoveJoker,
        SellJoker,
        SellConsumable,
        UseConsumable,
        SelectBlind,
        SkipBlind,
        Play,
        Discard,
        SortByRank,
        SortBySuit,
        CashOut,
        Buy,
        BuyAndUse,
        RedeemVoucher,
        BuyPack,
        Reroll,
        EndShop,
        Pick,
        SkipPack,
        ContinueEndless,
        EndRun,
    )
}


def parse_action(data: Sequence[str | int]) -> Action:
    """Rebuild an action from its list form, as produced by ``Action.to_list``."""
    if not data or not isinstance(data[0], str):
        raise ValueError(f"malformed action {data!r}")
    cls = ACTION_TYPES.get(data[0])
    if cls is None:
        raise ValueError(f"unknown action {data[0]!r}")
    args = data[1:]
    fields = dataclasses.fields(cls)
    if len(args) != len(fields) or not all(type(arg) is int for arg in args):
        raise ValueError(f"malformed arguments for {data[0]!r}: {list(args)!r}")
    return cls(*args)
