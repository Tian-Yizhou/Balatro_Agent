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
    "CardDestroyed",
    "CardMoved",
    "CardRetriggered",
    "CardToggled",
    "CardsDiscarded",
    "CardsDrawn",
    "ConsumableUsed",
    "EffectApplied",
    "Event",
    "HandLevelChanged",
    "HandPlayed",
    "HandScored",
    "HandSorted",
    "ItemBought",
    "ItemCreated",
    "ItemSold",
    "JokerDestroyed",
    "JokerMoved",
    "MoneyChanged",
    "PhaseChanged",
    "RoundEnded",
    "RunEnded",
    "ShopRerolled",
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
class EffectApplied(Event):
    """Something changed the score of the hand being played, or paid money.

    ``source`` is the uid of the card or joker the effect comes from, and ``key``
    names what it is: a joker or enhancement key, an edition, ``"card"`` for a
    playing card's own chips, or ``"seal"``. ``target`` is the playing card the
    effect is about, if any. ``via`` is the uid of a Blueprint or Brainstorm
    that copied the effect. Fields an effect doesn't use keep their neutral value.
    """

    kind: ClassVar[str] = "effect_applied"
    source: int
    key: str
    target: int | None = None
    chips: float = 0.0
    mult: float = 0.0
    x_mult: float = 1.0
    dollars: int = 0
    via: int | None = None


@dataclass(frozen=True, slots=True)
class CardRetriggered(Event):
    """A playing card is evaluated again, because of ``source`` (a seal or joker)."""

    kind: ClassVar[str] = "card_retriggered"
    uid: int
    source: int
    key: str


@dataclass(frozen=True, slots=True)
class CardDestroyed(Event):
    kind: ClassVar[str] = "card_destroyed"
    uid: int
    reason: str


@dataclass(frozen=True, slots=True)
class JokerDestroyed(Event):
    kind: ClassVar[str] = "joker_destroyed"
    uid: int
    key: str
    reason: str


@dataclass(frozen=True, slots=True)
class HandLevelChanged(Event):
    kind: ClassVar[str] = "hand_level_changed"
    hand: HandType
    level: int


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
    """The run is over. ``reason`` is ``"lost"``, ``"won"`` or ``"max_ante"``."""

    kind: ClassVar[str] = "run_ended"
    won: bool
    ante: int
    round: int
    reason: str


@dataclass(frozen=True, slots=True)
class ItemBought(Event):
    kind: ClassVar[str] = "item_bought"
    uid: int
    key: str
    cost: int


@dataclass(frozen=True, slots=True)
class ItemSold(Event):
    kind: ClassVar[str] = "item_sold"
    uid: int
    key: str
    value: int


@dataclass(frozen=True, slots=True)
class ItemCreated(Event):
    """A joker or consumable appeared: in the shop or among the player's items."""

    kind: ClassVar[str] = "item_created"
    uid: int
    key: str
    area: str


@dataclass(frozen=True, slots=True)
class ConsumableUsed(Event):
    kind: ClassVar[str] = "consumable_used"
    uid: int
    key: str
    targets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ShopRerolled(Event):
    kind: ClassVar[str] = "shop_rerolled"
    cost: int


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
        EffectApplied,
        CardRetriggered,
        CardDestroyed,
        JokerDestroyed,
        HandLevelChanged,
        HandScored,
        CardsDiscarded,
        RoundEnded,
        AnteChanged,
        MoneyChanged,
        RunEnded,
        ItemBought,
        ItemSold,
        ItemCreated,
        ConsumableUsed,
        ShopRerolled,
    )
}


def event_to_data(event: Event) -> dict[str, Any]:
    """Serialize an event as ``{"kind": ..., <fields>}``."""
    data: dict[str, Any] = {"kind": event.kind}
    data.update(to_data(event))
    return data
