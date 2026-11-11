"""Poker hand detection and the base value of each hand.

A faithful port of the game's hand evaluation, including the quirks that decide
which cards each contained hand reports. Jokers such as Four Fingers, Shortcut
and Smeared Joker change the rules and are passed in as flags.

Card groups are reported as indices into the sequence of cards evaluated, so a
caller can map them back to cards and tests can compare them with the original
code directly.

Source: functions/misc_functions.lua ``evaluate_poker_hand``, ``get_flush``,
``get_straight``, ``get_X_same`` and ``get_highest``; functions/state_events.lua
``G.FUNCS.get_poker_hand_info``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from .cards import SUITS_IN_FLUSH_ORDER, card_id, is_suit_for_flush, sort_key
from .state import HandLevel, HandType, PlayingCard

__all__ = [
    "HAND_BASE",
    "HandBase",
    "PokerHands",
    "display_name",
    "evaluate_poker_hand",
    "initial_hand_levels",
]

#: Groups of card positions, each forming one instance of a poker hand.
CardGroups: TypeAlias = list[list[int]]


@dataclass(frozen=True, slots=True)
class HandBase:
    """Static values of a poker hand: chips and mult at level 1, and per level."""

    order: int
    chips: int
    mult: int
    level_chips: int
    level_mult: int
    visible: bool


#: Source: game.lua ``Game:init_game_object`` (``hands``).
HAND_BASE: dict[HandType, HandBase] = {
    HandType.FLUSH_FIVE: HandBase(1, 160, 16, 50, 3, visible=False),
    HandType.FLUSH_HOUSE: HandBase(2, 140, 14, 40, 4, visible=False),
    HandType.FIVE_OF_A_KIND: HandBase(3, 120, 12, 35, 3, visible=False),
    HandType.STRAIGHT_FLUSH: HandBase(4, 100, 8, 40, 4, visible=True),
    HandType.FOUR_OF_A_KIND: HandBase(5, 60, 7, 30, 3, visible=True),
    HandType.FULL_HOUSE: HandBase(6, 40, 4, 25, 2, visible=True),
    HandType.FLUSH: HandBase(7, 35, 4, 15, 2, visible=True),
    HandType.STRAIGHT: HandBase(8, 30, 4, 30, 3, visible=True),
    HandType.THREE_OF_A_KIND: HandBase(9, 30, 3, 20, 2, visible=True),
    HandType.TWO_PAIR: HandBase(10, 20, 2, 20, 1, visible=True),
    HandType.PAIR: HandBase(11, 10, 2, 15, 1, visible=True),
    HandType.HIGH_CARD: HandBase(12, 5, 1, 10, 1, visible=True),
}


def initial_hand_levels() -> dict[HandType, HandLevel]:
    """Every poker hand at level 1, as at the start of a run."""
    return {
        hand: HandLevel(level=1, chips=base.chips, mult=base.mult, visible=base.visible)
        for hand, base in HAND_BASE.items()
    }


@dataclass(frozen=True, slots=True)
class PokerHands:
    """The result of evaluating a set of cards.

    Attributes:
        top: The best hand the cards make, or ``None`` if there are no cards.
        groups: For each hand type, the card groups that form it. Each group is a
            tuple of indices into the evaluated cards. A hand type the cards don't
            contain has no groups.
        scoring: The indices of the cards that score for ``top``, in the order
            the game lists them.
    """

    top: HandType | None
    groups: dict[HandType, tuple[tuple[int, ...], ...]]
    scoring: tuple[int, ...]

    def contains(self, hand: HandType) -> bool:
        return bool(self.groups[hand])


def evaluate_poker_hand(
    cards: Sequence[PlayingCard],
    *,
    four_fingers: bool = False,
    shortcut: bool = False,
    smeared: bool = False,
) -> PokerHands:
    """Find every poker hand that ``cards`` contain, and the best one."""
    ranks = [card_id(card) for card in cards]
    fives = _groups_of_equal_rank(5, ranks)
    fours = _groups_of_equal_rank(4, ranks)
    threes = _groups_of_equal_rank(3, ranks)
    pairs = _groups_of_equal_rank(2, ranks)
    flush = _flush(cards, four_fingers=four_fingers, smeared=smeared)
    straight = _straight(ranks, four_fingers=four_fingers, shortcut=shortcut)
    highest = _highest(cards)

    results: dict[HandType, CardGroups] = {hand: [] for hand in HandType}
    top: HandType | None = None

    def found(hand: HandType, groups: CardGroups) -> None:
        nonlocal top
        results[hand] = groups
        if top is None:
            top = hand

    if fives and flush:
        found(HandType.FLUSH_FIVE, fives)
    if threes and pairs and flush:
        found(HandType.FLUSH_HOUSE, [threes[0] + pairs[0]])
    if fives:
        found(HandType.FIVE_OF_A_KIND, fives)
    if flush and straight:
        in_flush = set(flush[0])
        found(
            HandType.STRAIGHT_FLUSH,
            [flush[0] + [position for position in straight[0] if position not in in_flush]],
        )
    if fours:
        found(HandType.FOUR_OF_A_KIND, fours)
    if threes and pairs:
        found(HandType.FULL_HOUSE, [threes[0] + pairs[0]])
    if flush:
        found(HandType.FLUSH, flush)
    if straight:
        found(HandType.STRAIGHT, straight)
    if threes:
        found(HandType.THREE_OF_A_KIND, threes)
    if len(pairs) == 2 or (len(threes) == 1 and len(pairs) == 1):
        second = pairs[1] if len(pairs) > 1 else threes[0]
        found(HandType.TWO_PAIR, [pairs[0] + second])
    if pairs:
        found(HandType.PAIR, pairs)
    if highest:
        found(HandType.HIGH_CARD, highest)

    # A bigger set of equal ranks also counts as each smaller one. The game keeps
    # at most the first four, three or two *groups* here, not cards.
    if results[HandType.FIVE_OF_A_KIND]:
        results[HandType.FOUR_OF_A_KIND] = results[HandType.FIVE_OF_A_KIND][:4]
    if results[HandType.FOUR_OF_A_KIND]:
        results[HandType.THREE_OF_A_KIND] = results[HandType.FOUR_OF_A_KIND][:3]
    if results[HandType.THREE_OF_A_KIND]:
        results[HandType.PAIR] = results[HandType.THREE_OF_A_KIND][:2]

    groups = {hand: tuple(tuple(group) for group in results[hand]) for hand in HandType}
    scoring = groups[top][0] if top is not None else ()
    return PokerHands(top=top, groups=groups, scoring=scoring)


def display_name(cards: Sequence[PlayingCard], hands: PokerHands) -> str:
    """The name the game shows, which calls a Straight Flush of 10 to Ace a Royal Flush."""
    if hands.top is None:
        return ""
    if hands.top is HandType.STRAIGHT_FLUSH:
        ranks = [card_id(cards[position]) for position in hands.scoring]
        lowest = min(-1 if rank is None else rank for rank in ranks)
        if lowest >= 10:
            return "Royal Flush"
    return hands.top.value


def _groups_of_equal_rank(size: int, ranks: Sequence[int | None]) -> CardGroups:
    """Groups of exactly ``size`` cards of equal rank, highest rank first.

    Source: functions/misc_functions.lua ``get_X_same``.
    """
    by_rank: dict[int, list[int]] = {}
    for i in range(len(ranks) - 1, -1, -1):
        rank = ranks[i]
        if rank is None:
            continue
        group = [i] + [other for other in range(len(ranks)) if other != i and ranks[other] == rank]
        if len(group) == size:
            by_rank[rank] = group
    return [by_rank[rank] for rank in range(14, 0, -1) if rank in by_rank]


def _flush(cards: Sequence[PlayingCard], *, four_fingers: bool, smeared: bool) -> CardGroups:
    needed = 4 if four_fingers else 5
    if len(cards) > 5 or len(cards) < needed:
        return []
    for suit in SUITS_IN_FLUSH_ORDER:
        members = [
            position
            for position, card in enumerate(cards)
            if is_suit_for_flush(card, suit, smeared=smeared)
        ]
        if len(members) >= needed:
            return [members]
    return []


def _straight(ranks: Sequence[int | None], *, four_fingers: bool, shortcut: bool) -> CardGroups:
    needed = 4 if four_fingers else 5
    if len(ranks) > 5 or len(ranks) < needed:
        return []
    by_rank: dict[int, list[int]] = {}
    for position, rank in enumerate(ranks):
        if rank is not None and 1 < rank < 15:
            by_rank.setdefault(rank, []).append(position)

    members: list[int] = []
    length = 0
    is_straight = False
    skipped = False
    # The Ace is visited twice: first as the low card, then as the high card.
    for step in range(1, 15):
        rank = 14 if step == 1 else step
        if rank in by_rank:
            length += 1
            skipped = False
            members.extend(by_rank[rank])
        elif shortcut and not skipped and step != 14:
            skipped = True
        else:
            length = 0
            skipped = False
            if is_straight:
                break
            members = []
        if length >= needed:
            is_straight = True
    return [members] if is_straight else []


def _highest(cards: Sequence[PlayingCard]) -> CardGroups:
    best: int | None = None
    for position, card in enumerate(cards):
        if best is None or sort_key(card) > sort_key(cards[best]):
            best = position
    return [[best]] if best is not None else []
