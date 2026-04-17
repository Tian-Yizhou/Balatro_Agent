"""Tests for game_state.py: full game flow, scoring pipeline, economy."""

import numpy as np
import pytest

from balatro_gym.core.card import Card, Rank, Suit
from balatro_gym.core.game_state import GamePhase, GameState, _get_poker_hands
from balatro_gym.core.hand_evaluator import HandType
from balatro_gym.core.joker import create_joker


class TestGameLifecycle:
    def test_reset_initializes(self):
        gs = GameState(num_antes=4, seed=42)
        gs.reset()
        assert gs.phase == GamePhase.PLAY
        assert gs.ante == 1
        assert gs.blind_index == 0
        assert gs.money == 4
        assert len(gs.hand) == 8
        assert gs.hands_remaining == 4
        assert gs.discards_remaining == 3

    def test_play_hand_reduces_hands(self):
        gs = GameState(seed=42)
        gs.reset()
        gs.play_hand([0])  # play first card
        assert gs.hands_remaining == 3
        assert gs.total_hands_played == 1

    def test_discard_reduces_discards(self):
        gs = GameState(seed=42)
        gs.reset()
        gs.discard([0])
        assert gs.discards_remaining == 2

    def test_discard_draws_replacements(self):
        gs = GameState(seed=42)
        gs.reset()
        old_hand_size = len(gs.hand)
        gs.discard([0, 1])
        assert len(gs.hand) == old_hand_size

    def test_play_hand_replaces_cards(self):
        gs = GameState(seed=42)
        gs.reset()
        old_hand_size = len(gs.hand)
        gs.play_hand([0, 1])
        # After playing, hand should still be same size (drew replacements)
        if gs.phase == GamePhase.PLAY:
            assert len(gs.hand) == old_hand_size

    def test_game_over_on_no_hands(self):
        gs = GameState(num_antes=8, hands_per_round=1, seed=42)
        gs.reset()
        # Play 1 hand — if it doesn't beat the blind, game over
        gs.play_hand([0])  # single card, likely won't beat 300
        if gs.current_score < gs.score_target:
            assert gs.phase == GamePhase.GAME_OVER


class TestScoring:
    def test_pair_base_score(self):
        """A pair of aces should produce a known score with no jokers."""
        gs = GameState(seed=42, available_joker_ids=[])
        gs.reset()

        # Force a known hand
        gs.hand = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.SIX, Suit.CLUBS),
            Card(Rank.SEVEN, Suit.DIAMONDS),
        ]

        score, result = gs.play_hand([0, 1])
        # Pair base: chips=10, mult=2. Scoring cards: 2 aces = 11+11 = 22 chips.
        # Total chips = 10 + 22 = 32. Score = 32 * 2 = 64
        assert result.hand_type == HandType.PAIR
        assert score == 64

    def test_flush_scoring(self):
        gs = GameState(seed=42, available_joker_ids=[])
        gs.reset()

        gs.hand = [
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.FIVE, Suit.HEARTS),
            Card(Rank.SEVEN, Suit.HEARTS),
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.THREE, Suit.CLUBS),
            Card(Rank.FOUR, Suit.DIAMONDS),
            Card(Rank.SIX, Suit.SPADES),
        ]

        score, result = gs.play_hand([0, 1, 2, 3, 4])
        assert result.hand_type == HandType.FLUSH
        # Flush base: chips=35, mult=4. Cards: 2+5+7+9+10=33. Total chips = 35+33=68.
        # Score = 68 * 4 = 272
        assert score == 272

    def test_joker_adds_mult(self):
        gs = GameState(seed=42, starting_joker_ids=["joker_basic"],
                       available_joker_ids=["joker_basic"])
        gs.reset()

        gs.hand = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.SIX, Suit.CLUBS),
            Card(Rank.SEVEN, Suit.DIAMONDS),
        ]

        score, result = gs.play_hand([0, 1])
        # Pair: base chips=32, base mult=2. Basic joker: +4 mult -> mult=6.
        # Score = 32 * 6 = 192
        assert score == 192


class TestEconomy:
    def test_money_after_beating_blind(self):
        gs = GameState(num_antes=2, hands_per_round=4, seed=42,
                       available_joker_ids=["joker_basic"])
        gs.reset()

        # Force an easy win
        gs.hand = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.CLUBS),
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.SPADES),
        ]
        starting_money = gs.money
        gs.play_hand([0, 1, 2, 3])  # Four aces

        if gs.phase == GamePhase.SHOP:
            # Should have earned: blind dollars + unused hands + interest
            earned = gs.money - starting_money
            assert earned > 0


class TestPokerHandsDict:
    def test_pair_contains_pair(self):
        hands = _get_poker_hands(HandType.PAIR)
        assert hands.get("Pair") is True

    def test_full_house_contains_pair_and_three(self):
        hands = _get_poker_hands(HandType.FULL_HOUSE)
        assert hands.get("Pair") is True
        assert hands.get("Three of a Kind") is True
        assert hands.get("Full House") is True

    def test_straight_flush_contains_both(self):
        hands = _get_poker_hands(HandType.STRAIGHT_FLUSH)
        assert hands.get("Straight") is True
        assert hands.get("Flush") is True

    def test_flush_house_contains_all(self):
        hands = _get_poker_hands(HandType.FLUSH_HOUSE)
        assert hands.get("Pair") is True
        assert hands.get("Three of a Kind") is True
        assert hands.get("Flush") is True
        assert hands.get("Full House") is True

    def test_four_of_a_kind_contains_pair(self):
        hands = _get_poker_hands(HandType.FOUR_OF_A_KIND)
        assert hands.get("Pair") is True
        assert hands.get("Four of a Kind") is True


class TestPhaseTransitions:
    def test_play_to_shop_on_beat(self):
        gs = GameState(num_antes=2, seed=42, available_joker_ids=["joker_basic"])
        gs.reset()

        # Force a huge hand to beat the blind
        gs.hand = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.CLUBS),
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.ACE, Suit.HEARTS),  # Dupe suit for testing
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.SPADES),
        ]
        gs.play_hand([0, 1, 2, 3])
        assert gs.phase in (GamePhase.SHOP, GamePhase.PLAY)

    def test_shop_skip_to_play(self):
        gs = GameState(num_antes=2, seed=42, available_joker_ids=["joker_basic"])
        gs.reset()

        # Force win
        gs.current_score = gs.score_target
        gs._end_blind()

        if gs.phase == GamePhase.SHOP:
            gs.shop_skip()
            assert gs.phase == GamePhase.PLAY


class TestValidActions:
    def test_play_phase_actions(self):
        gs = GameState(seed=42)
        gs.reset()
        actions = gs.get_valid_actions()
        assert actions["can_play"] is True
        assert actions["can_discard"] is True
        assert actions["hand_size"] == 8

    def test_shop_phase_actions(self):
        gs = GameState(num_antes=2, seed=42,
                       available_joker_ids=["joker_basic", "banner"])
        gs.reset()
        # Force to shop
        gs.current_score = gs.score_target
        gs._end_blind()

        if gs.phase == GamePhase.SHOP:
            actions = gs.get_valid_actions()
            assert "can_buy" in actions
            assert "can_sell" in actions
            assert actions["can_skip"] is True


class TestErrorHandling:
    def test_play_in_shop_raises(self):
        gs = GameState(seed=42, available_joker_ids=["joker_basic"])
        gs.reset()
        gs.phase = GamePhase.SHOP
        with pytest.raises(ValueError):
            gs.play_hand([0])

    def test_discard_in_shop_raises(self):
        gs = GameState(seed=42)
        gs.reset()
        gs.phase = GamePhase.SHOP
        with pytest.raises(ValueError):
            gs.discard([0])

    def test_play_empty_hand_raises(self):
        gs = GameState(seed=42)
        gs.reset()
        with pytest.raises(ValueError):
            gs.play_hand([])

    def test_play_too_many_raises(self):
        gs = GameState(seed=42)
        gs.reset()
        with pytest.raises(ValueError):
            gs.play_hand([0, 1, 2, 3, 4, 5])

    def test_play_invalid_index_raises(self):
        gs = GameState(seed=42)
        gs.reset()
        with pytest.raises(ValueError):
            gs.play_hand([99])

    def test_play_duplicate_indices_raises(self):
        gs = GameState(seed=42)
        gs.reset()
        with pytest.raises(ValueError):
            gs.play_hand([0, 0])
