"""Tests for hand_evaluator.py: poker hand detection and scoring."""

import pytest

from balatro_gym.core.card import Card, Rank, Suit
from balatro_gym.core.hand_evaluator import (
    HandType,
    HAND_BASE_SCORES,
    evaluate_hand,
)


def _cards(*specs: str) -> list[Card]:
    """Helper: parse cards like '2H', 'AS', 'KD', '10C'."""
    rank_map = {
        "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE,
        "6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE,
        "10": Rank.TEN, "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE,
    }
    suit_map = {"H": Suit.HEARTS, "D": Suit.DIAMONDS, "C": Suit.CLUBS, "S": Suit.SPADES}
    result = []
    for s in specs:
        suit_char = s[-1]
        rank_str = s[:-1]
        result.append(Card(rank_map[rank_str], suit_map[suit_char]))
    return result


class TestHandDetection:
    def test_high_card(self):
        cards = _cards("2H", "5D", "7C", "9S", "KH")
        result = evaluate_hand(cards)
        assert result.hand_type == HandType.HIGH_CARD
        assert len(result.scoring_cards) == 1
        assert result.scoring_cards[0].rank == Rank.KING

    def test_pair(self):
        cards = _cards("5H", "5D", "7C", "9S", "KH")
        result = evaluate_hand(cards)
        assert result.hand_type == HandType.PAIR
        assert len(result.scoring_cards) == 2
        assert all(c.rank == Rank.FIVE for c in result.scoring_cards)

    def test_two_pair(self):
        cards = _cards("5H", "5D", "9C", "9S", "KH")
        result = evaluate_hand(cards)
        assert result.hand_type == HandType.TWO_PAIR
        assert len(result.scoring_cards) == 4

    def test_three_of_a_kind(self):
        cards = _cards("7H", "7D", "7C", "2S", "KH")
        result = evaluate_hand(cards)
        assert result.hand_type == HandType.THREE_OF_A_KIND
        assert len(result.scoring_cards) == 3

    def test_straight(self):
        cards = _cards("5H", "6D", "7C", "8S", "9H")
        result = evaluate_hand(cards)
        assert result.hand_type == HandType.STRAIGHT
        assert len(result.scoring_cards) == 5

    def test_straight_ace_low(self):
        cards = _cards("AH", "2D", "3C", "4S", "5H")
        result = evaluate_hand(cards)
        assert result.hand_type == HandType.STRAIGHT

    def test_straight_ace_high(self):
        cards = _cards("10H", "JD", "QC", "KS", "AH")
        result = evaluate_hand(cards)
        assert result.hand_type == HandType.STRAIGHT

    def test_flush(self):
        cards = _cards("2H", "5H", "7H", "9H", "KH")
        result = evaluate_hand(cards)
        assert result.hand_type == HandType.FLUSH
        assert len(result.scoring_cards) == 5

    def test_full_house(self):
        cards = _cards("7H", "7D", "7C", "KS", "KH")
        result = evaluate_hand(cards)
        assert result.hand_type == HandType.FULL_HOUSE
        assert len(result.scoring_cards) == 5

    def test_four_of_a_kind(self):
        cards = _cards("9H", "9D", "9C", "9S", "KH")
        result = evaluate_hand(cards)
        assert result.hand_type == HandType.FOUR_OF_A_KIND
        assert len(result.scoring_cards) == 4

    def test_straight_flush(self):
        cards = _cards("5H", "6H", "7H", "8H", "9H")
        result = evaluate_hand(cards)
        assert result.hand_type == HandType.STRAIGHT_FLUSH

    def test_single_card(self):
        cards = _cards("AH")
        result = evaluate_hand(cards)
        assert result.hand_type == HandType.HIGH_CARD
        assert result.scoring_cards[0].rank == Rank.ACE

    def test_two_cards_pair(self):
        cards = _cards("KH", "KD")
        result = evaluate_hand(cards)
        assert result.hand_type == HandType.PAIR

    def test_three_cards_no_hand(self):
        cards = _cards("2H", "5D", "KS")
        result = evaluate_hand(cards)
        assert result.hand_type == HandType.HIGH_CARD


class TestBaseScores:
    """Verify base scores match Lua exactly."""

    def test_high_card(self):
        assert HAND_BASE_SCORES[HandType.HIGH_CARD] == (5, 1)

    def test_pair(self):
        assert HAND_BASE_SCORES[HandType.PAIR] == (10, 2)

    def test_two_pair(self):
        assert HAND_BASE_SCORES[HandType.TWO_PAIR] == (20, 2)

    def test_three_of_a_kind(self):
        assert HAND_BASE_SCORES[HandType.THREE_OF_A_KIND] == (30, 3)

    def test_straight(self):
        assert HAND_BASE_SCORES[HandType.STRAIGHT] == (30, 4)

    def test_flush(self):
        assert HAND_BASE_SCORES[HandType.FLUSH] == (35, 4)

    def test_full_house(self):
        assert HAND_BASE_SCORES[HandType.FULL_HOUSE] == (40, 4)

    def test_four_of_a_kind(self):
        assert HAND_BASE_SCORES[HandType.FOUR_OF_A_KIND] == (60, 7)

    def test_straight_flush(self):
        assert HAND_BASE_SCORES[HandType.STRAIGHT_FLUSH] == (100, 8)

    def test_five_of_a_kind(self):
        assert HAND_BASE_SCORES[HandType.FIVE_OF_A_KIND] == (120, 12)

    def test_flush_house(self):
        assert HAND_BASE_SCORES[HandType.FLUSH_HOUSE] == (140, 14)

    def test_flush_five(self):
        assert HAND_BASE_SCORES[HandType.FLUSH_FIVE] == (160, 16)


class TestChipCalculation:
    def test_pair_chips_include_card_values(self):
        cards = _cards("KH", "KD", "2C", "3S", "5H")
        result = evaluate_hand(cards)
        # Pair base = 10, scoring cards = 2 Kings = 10+10 = 20
        assert result.base_chips == 10 + 10 + 10
        assert result.base_mult == 2

    def test_high_card_ace(self):
        cards = _cards("AH")
        result = evaluate_hand(cards)
        assert result.base_chips == 5 + 11  # base + ace
        assert result.base_mult == 1

    def test_debuffed_cards_zero_chip_contribution(self):
        cards = _cards("KH", "KD")
        cards[0].face_down = True
        result = evaluate_hand(cards)
        # King face-down = 0 chips, King face-up = 10
        assert result.base_chips == 10 + 0 + 10

    def test_held_cards_passed_through(self):
        played = _cards("AH", "AD")
        held = _cards("2C", "3D", "5S")
        result = evaluate_hand(played, held)
        assert len(result.held_cards) == 3


class TestEdgeCases:
    def test_empty_hand_raises(self):
        with pytest.raises(ValueError):
            evaluate_hand([])

    def test_too_many_cards_raises(self):
        cards = _cards("2H", "3H", "4H", "5H", "6H", "7H")
        with pytest.raises(ValueError):
            evaluate_hand(cards)

    def test_not_a_straight(self):
        cards = _cards("2H", "3D", "4C", "5S", "7H")
        result = evaluate_hand(cards)
        assert result.hand_type != HandType.STRAIGHT

    def test_four_card_not_flush(self):
        cards = _cards("2H", "5H", "7H", "KH")
        result = evaluate_hand(cards)
        assert result.hand_type != HandType.FLUSH
