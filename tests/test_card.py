"""Tests for card.py: Card, Deck primitives."""

import numpy as np
import pytest

from balatro_gym.core.card import Card, Deck, Rank, Suit, make_standard_deck


class TestCard:
    def test_chip_values(self):
        assert Card(Rank.ACE, Suit.SPADES).chip_value == 11
        assert Card(Rank.KING, Suit.HEARTS).chip_value == 10
        assert Card(Rank.QUEEN, Suit.DIAMONDS).chip_value == 10
        assert Card(Rank.JACK, Suit.CLUBS).chip_value == 10
        assert Card(Rank.TEN, Suit.SPADES).chip_value == 10
        assert Card(Rank.FIVE, Suit.HEARTS).chip_value == 5
        assert Card(Rank.TWO, Suit.CLUBS).chip_value == 2

    def test_debuffed_card_zero_chips(self):
        card = Card(Rank.ACE, Suit.SPADES, face_down=True)
        assert card.chip_value == 0

    def test_nominal_ignores_debuff(self):
        card = Card(Rank.ACE, Suit.SPADES, face_down=True)
        assert card.nominal == 11
        assert card.chip_value == 0

    def test_card_id(self):
        assert Card(Rank.TWO, Suit.HEARTS).id == 2
        assert Card(Rank.TEN, Suit.DIAMONDS).id == 10
        assert Card(Rank.ACE, Suit.SPADES).id == 14

    def test_is_face_card(self):
        assert Card(Rank.JACK, Suit.HEARTS).is_face_card
        assert Card(Rank.QUEEN, Suit.HEARTS).is_face_card
        assert Card(Rank.KING, Suit.HEARTS).is_face_card
        assert not Card(Rank.TEN, Suit.HEARTS).is_face_card
        assert not Card(Rank.ACE, Suit.HEARTS).is_face_card

    def test_equality_and_hash(self):
        c1 = Card(Rank.ACE, Suit.SPADES)
        c2 = Card(Rank.ACE, Suit.SPADES)
        c3 = Card(Rank.ACE, Suit.HEARTS)
        assert c1 == c2
        assert c1 != c3
        assert hash(c1) == hash(c2)
        assert len({c1, c2, c3}) == 2

    def test_repr(self):
        card = Card(Rank.ACE, Suit.SPADES)
        assert "A" in repr(card)


class TestDeck:
    def test_standard_deck_52_cards(self):
        deck = make_standard_deck()
        assert len(deck) == 52
        assert len(set(deck)) == 52

    def test_deck_reset_and_draw(self):
        deck = Deck(np.random.default_rng(42))
        deck.reset()
        assert deck.cards_remaining == 52
        drawn = deck.draw(8)
        assert len(drawn) == 8
        assert deck.cards_remaining == 44

    def test_deck_draw_all(self):
        deck = Deck(np.random.default_rng(42))
        deck.reset()
        drawn = deck.draw(52)
        assert len(drawn) == 52
        assert deck.cards_remaining == 0

    def test_deck_reshuffle_on_empty(self):
        deck = Deck(np.random.default_rng(42))
        deck.reset()
        drawn = deck.draw(50)
        deck.return_cards(drawn)
        assert deck.cards_remaining == 2
        assert len(deck.discard_pile) == 50
        # Drawing more than available triggers reshuffle
        more = deck.draw(10)
        assert len(more) == 10

    def test_deck_seeded_reproducibility(self):
        d1 = Deck(np.random.default_rng(42))
        d1.reset()
        hand1 = d1.draw(8)

        d2 = Deck(np.random.default_rng(42))
        d2.reset()
        hand2 = d2.draw(8)

        assert hand1 == hand2

    def test_return_cards(self):
        deck = Deck(np.random.default_rng(42))
        deck.reset()
        drawn = deck.draw(5)
        deck.return_cards(drawn)
        assert len(deck.discard_pile) == 5
