"""Events: what happened during a step, for viewers, rewards and debugging.

Events are derived data. A state can always be restored without them, and they
hold plain values (card uids, keys, numbers) rather than references to state
objects, so they stay valid after the state changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from .serialize import to_data
from .state import BlindSlot, HandSort, HandType, Phase

__all__ = [
    "EVENT_TYPES",
    "AnteChanged",
    "BlindSelected",
    "BossChosen",
    "CardMoved",
    "CardScored",
    "CardToggled",
    "CardsDiscarded",
    "CardsDrawn",
    "Event",
    "HandPlayed",
    "HandScored",
    "HandSorted",
    "JokerMoved",
    "MoneyChanged",
    "PhaseChanged",
    "RoundEnded",
    "RunEnded",
    "event_to_data",
]


@dataclass(frozen=True, slots=True)
class Event:
    """Base class of all events. ``kind`` is the event's name in serialized form."""

    kind: ClassVar[str] = "event"


@dataclass(frozen=True, slots=True)
class PhaseChanged(Event):
    kind: ClassVar[str] = "phase_changed"
    old: Phase
    new: Phase


@dataclass(frozen=True, slots=True)
class BossChosen(Event):
    kind: ClassVar[str] = "boss_chosen"
    ante: int
    boss: str


@dataclass(frozen=True, slots=True)
class BlindSelected(Event):
    kind: ClassVar[str] = "blind_selected"
    slot: BlindSlot
    blind: str
    target: float


@dataclass(frozen=True, slots=True)
class CardsDrawn(Event):
    kind: ClassVar[str] = "cards_drawn"
    uids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CardToggled(Event):
    kind: ClassVar[str] = "card_toggled"
    uid: int
    selected: bool


@dataclass(frozen=True, slots=True)
class CardMoved(Event):
    kind: ClassVar[str] = "card_moved"
    uid: int
    source: int
    target: int


@dataclass(frozen=True, slots=True)
class JokerMoved(Event):
    kind: ClassVar[str] = "joker_moved"
    uid: int
    source: int
    target: int


@dataclass(frozen=True, slots=True)
class HandSorted(Event):
    kind: ClassVar[str] = "hand_sorted"
    mode: HandSort


@dataclass(frozen=True, slots=True)
class HandPlayed(Event):
    kind: ClassVar[str] = "hand_played"
    uids: tuple[int, ...]
    hand: HandType
    scoring: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CardScored(Event):
    """A scoring card added ``chips`` to the hand's chips."""

    kind: ClassVar[str] = "card_scored"
    uid: int
    chips: float


@dataclass(frozen=True, slots=True)
class HandScored(Event):
    """The final chips and mult of a played hand, and the score it added."""

    kind: ClassVar[str] = "hand_scored"
    hand: HandType
    chips: float
    mult: float
    score: float
    total: float


@dataclass(frozen=True, slots=True)
class CardsDiscarded(Event):
    kind: ClassVar[str] = "cards_discarded"
    uids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RoundEnded(Event):
    kind: ClassVar[str] = "round_ended"
    blind: str
    won: bool
    score: float
    target: float


@dataclass(frozen=True, slots=True)
class AnteChanged(Event):
    kind: ClassVar[str] = "ante_changed"
    ante: int


@dataclass(frozen=True, slots=True)
class MoneyChanged(Event):
    kind: ClassVar[str] = "money_changed"
    delta: int
    balance: int
    reason: str


@dataclass(frozen=True, slots=True)
class RunEnded(Event):
    kind: ClassVar[str] = "run_ended"
    won: bool
    ante: int
    round: int


#: Every concrete event type, by ``kind``.
EVENT_TYPES: dict[str, type[Event]] = {
    cls.kind: cls
    for cls in (
        PhaseChanged,
        BossChosen,
        BlindSelected,
        CardsDrawn,
        CardToggled,
        CardMoved,
        JokerMoved,
        HandSorted,
        HandPlayed,
        CardScored,
        HandScored,
        CardsDiscarded,
        RoundEnded,
        AnteChanged,
        MoneyChanged,
        RunEnded,
    )
}


def event_to_data(event: Event) -> dict[str, Any]:
    """Serialize an event as ``{"kind": ..., <fields>}``."""
    data: dict[str, Any] = {"kind": event.kind}
    data.update(to_data(event))
    return data
