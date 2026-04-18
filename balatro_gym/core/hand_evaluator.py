"""Poker hand detection and base scoring for the Balatro game environment."""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass

from balatro_gym.core.card import Card, Rank


class HandType(enum.IntEnum):
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    FIVE_OF_A_KIND = 9
    FLUSH_HOUSE = 10       # Full house + flush
    FLUSH_FIVE = 11        # Five of a kind + flush


# (base_chips, base_mult) for each hand type
HAND_BASE_SCORES: dict[HandType, tuple[int, int]] = {
    HandType.HIGH_CARD:       (5, 1),
    HandType.PAIR:            (10, 2),
    HandType.TWO_PAIR:        (20, 2),
    HandType.THREE_OF_A_KIND: (30, 3),
    HandType.STRAIGHT:        (30, 4),
    HandType.FLUSH:           (35, 4),
    HandType.FULL_HOUSE:      (40, 4),
    HandType.FOUR_OF_A_KIND:  (60, 7),
    HandType.STRAIGHT_FLUSH:  (100, 8),
    HandType.FIVE_OF_A_KIND:  (120, 12),
    HandType.FLUSH_HOUSE:     (140, 14),
    HandType.FLUSH_FIVE:      (160, 16),
}


@dataclass
class HandResult:
    """Result of evaluating a played hand."""
    hand_type: HandType
    scoring_cards: list[Card]   # Cards that form the poker hand
    held_cards: list[Card]      # Cards remaining in hand (not played)
    base_chips: int             # Hand base chips + scoring card chip values
    base_mult: int              # Hand base mult


def _is_flush(cards: list[Card]) -> bool:
    """Check if all cards share the same suit (need 5 cards).

    Wild Card enhancement makes a card count as any suit, so it can
    fill in for any suit in a flush.
    """
    if len(cards) < 5:
        return False
    non_wild = [c for c in cards if not c.is_wild]
    if not non_wild:
        # All wild — it's a flush of any suit
        return True
    # All non-wild cards must share one suit
    return len({c.suit for c in non_wild}) == 1


def _is_straight(cards: list[Card]) -> tuple[bool, list[Card]]:
    """Check if cards form a straight. Returns (is_straight, ordered_cards).

    Handles ace-low (A-2-3-4-5) and ace-high (10-J-Q-K-A) straights.
    Requires exactly 5 cards.
    """
    if len(cards) != 5:
        return False, []

    ranks = sorted({c.rank for c in cards})
    if len(ranks) < 5:
        return False, []

    # Normal straight check: consecutive ranks
    if ranks[-1] - ranks[0] == 4:
        ordered = sorted(cards, key=lambda c: c.rank)
        return True, ordered

    # Ace-low straight: A-2-3-4-5
    if set(ranks) == {Rank.ACE, Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE}:
        # Order with ace first (low)
        ordered = sorted(cards, key=lambda c: (c.rank % 14))  # Ace wraps to position 0
        return True, ordered

    return False, []


def _get_rank_groups(cards: list[Card]) -> dict[Rank, list[Card]]:
    """Group cards by rank."""
    groups: dict[Rank, list[Card]] = {}
    for card in cards:
        groups.setdefault(card.rank, []).append(card)
    return groups


def evaluate_hand(played_cards: list[Card], held_cards: list[Card] | None = None) -> HandResult:
    """Evaluate the best poker hand from played cards.

    Args:
        played_cards: 1-5 cards selected to play.
        held_cards: Remaining cards in hand (for jokers that care about held cards).

    Returns:
        HandResult with detected hand type, scoring cards, and base score.
    """
    if held_cards is None:
        held_cards = []

    if not played_cards:
        raise ValueError("Must play at least one card")
    if len(played_cards) > 5:
        raise ValueError("Cannot play more than 5 cards")

    n = len(played_cards)
    rank_groups = _get_rank_groups(played_cards)
    group_sizes = sorted(Counter(c.rank for c in played_cards).values(), reverse=True)

    is_flush = _is_flush(played_cards)
    is_straight, straight_cards = _is_straight(played_cards)

    hand_type: HandType
    scoring_cards: list[Card]

    # Check from highest to lowest hand type

    # --- Five of a kind (5 cards, same rank — possible with wild/copy effects) ---
    if n == 5 and group_sizes == [5]:
        if is_flush:
            hand_type = HandType.FLUSH_FIVE
        else:
            hand_type = HandType.FIVE_OF_A_KIND
        scoring_cards = list(played_cards)

    # --- Flush house (full house + flush) ---
    elif n == 5 and group_sizes == [3, 2] and is_flush:
        hand_type = HandType.FLUSH_HOUSE
        scoring_cards = list(played_cards)

    # --- Straight flush ---
    elif n == 5 and is_straight and is_flush:
        hand_type = HandType.STRAIGHT_FLUSH
        scoring_cards = straight_cards

    # --- Four of a kind ---
    elif group_sizes[0] >= 4:
        hand_type = HandType.FOUR_OF_A_KIND
        scoring_cards = []
        for rank, cards in rank_groups.items():
            if len(cards) >= 4:
                scoring_cards = cards[:4]
                break

    # --- Full house ---
    elif n >= 5 and group_sizes[:2] == [3, 2]:
        hand_type = HandType.FULL_HOUSE
        scoring_cards = list(played_cards)

    # --- Flush ---
    elif is_flush:
        hand_type = HandType.FLUSH
        scoring_cards = list(played_cards)

    # --- Straight ---
    elif is_straight:
        hand_type = HandType.STRAIGHT
        scoring_cards = straight_cards

    # --- Three of a kind ---
    elif group_sizes[0] >= 3:
        hand_type = HandType.THREE_OF_A_KIND
        scoring_cards = []
        for rank, cards in rank_groups.items():
            if len(cards) >= 3:
                scoring_cards = cards[:3]
                break

    # --- Two pair ---
    elif len([s for s in group_sizes if s >= 2]) >= 2:
        hand_type = HandType.TWO_PAIR
        scoring_cards = []
        for rank in sorted(rank_groups.keys(), reverse=True):
            if len(rank_groups[rank]) >= 2:
                scoring_cards.extend(rank_groups[rank][:2])
            if len(scoring_cards) >= 4:
                break

    # --- Pair ---
    elif group_sizes[0] >= 2:
        hand_type = HandType.PAIR
        scoring_cards = []
        for rank, cards in rank_groups.items():
            if len(cards) >= 2:
                scoring_cards = cards[:2]
                break

    # --- High card ---
    else:
        hand_type = HandType.HIGH_CARD
        # Highest card is the scoring card
        scoring_cards = [max(played_cards, key=lambda c: c.rank)]

    # Calculate base score
    base_chips, base_mult = HAND_BASE_SCORES[hand_type]

    # Add chip values of scoring cards to base chips
    total_chips = base_chips + sum(c.chip_value for c in scoring_cards)

    return HandResult(
        hand_type=hand_type,
        scoring_cards=scoring_cards,
        held_cards=list(held_cards),
        base_chips=total_chips,
        base_mult=base_mult,
    )
