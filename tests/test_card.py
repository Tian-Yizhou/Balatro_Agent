"""Tests for card.py: Card, Deck, enhancements, editions, seals."""

import numpy as np
import pytest

from balatro_gym.core.card import (
    Card, Deck, Rank, Suit, Enhancement, Edition, Seal,
    make_standard_deck, reset_uid_counter,
)


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

    def test_uid_unique(self):
        c1 = Card(Rank.ACE, Suit.SPADES)
        c2 = Card(Rank.ACE, Suit.SPADES)
        assert c1.uid != c2.uid
        assert c1 != c2  # different uid

    def test_same_card(self):
        c1 = Card(Rank.ACE, Suit.SPADES)
        c2 = Card(Rank.ACE, Suit.SPADES)
        c3 = Card(Rank.ACE, Suit.HEARTS)
        assert c1.same_card(c2)
        assert not c1.same_card(c3)

    def test_uid_based_hashing(self):
        c1 = Card(Rank.ACE, Suit.SPADES)
        c2 = Card(Rank.ACE, Suit.SPADES)
        # Different uids = different hash, both in set
        assert len({c1, c2}) == 2
        # Same object = same hash
        assert len({c1, c1}) == 1

    def test_copy(self):
        c = Card(Rank.ACE, Suit.SPADES, enhancement=Enhancement.BONUS, seal=Seal.GOLD)
        c2 = c.copy()
        assert c2.rank == c.rank
        assert c2.suit == c.suit
        assert c2.enhancement == c.enhancement
        assert c2.seal == c.seal
        assert c2.uid != c.uid  # new uid

    def test_repr(self):
        card = Card(Rank.ACE, Suit.SPADES)
        assert "A" in repr(card)

    def test_repr_with_properties(self):
        card = Card(Rank.KING, Suit.HEARTS, enhancement=Enhancement.BONUS,
                    edition=Edition.FOIL, seal=Seal.RED)
        r = repr(card)
        assert "BONUS" in r
        assert "FOIL" in r
        assert "RED" in r


class TestEnhancements:
    def test_bonus_card_adds_30_chips(self):
        card = Card(Rank.FIVE, Suit.HEARTS, enhancement=Enhancement.BONUS)
        assert card.chip_value == 5 + 30

    def test_stone_card_50_chips_no_nominal(self):
        card = Card(Rank.ACE, Suit.SPADES, enhancement=Enhancement.STONE)
        # Stone Card: 50 chips, no nominal (Ace nominal ignored)
        assert card.chip_value == 50

    def test_mult_card(self):
        card = Card(Rank.FIVE, Suit.HEARTS, enhancement=Enhancement.MULT)
        assert card.get_chip_mult() == 4
        assert card.chip_value == 5  # no chip bonus

    def test_glass_card_x_mult(self):
        card = Card(Rank.FIVE, Suit.HEARTS, enhancement=Enhancement.GLASS)
        assert card.get_chip_x_mult() == 2.0

    def test_steel_card_held_x_mult(self):
        card = Card(Rank.FIVE, Suit.HEARTS, enhancement=Enhancement.STEEL)
        assert card.get_held_x_mult() == 1.5
        assert card.get_chip_x_mult() == 0.0  # no x_mult when played

    def test_gold_card_held_dollars(self):
        card = Card(Rank.FIVE, Suit.HEARTS, enhancement=Enhancement.GOLD)
        assert card.get_held_dollars() == 3

    def test_lucky_card_probabilistic_mult(self):
        card = Card(Rank.FIVE, Suit.HEARTS, enhancement=Enhancement.LUCKY)
        rng = np.random.default_rng(42)
        # Run many times to verify probabilistic behavior
        hits = sum(card.get_chip_mult(rng) == 20 for _ in range(1000))
        assert 100 < hits < 350  # ~20% expected

    def test_wild_card(self):
        card = Card(Rank.FIVE, Suit.HEARTS, enhancement=Enhancement.WILD)
        assert card.is_wild
        assert card.chip_value == 5  # no chip bonus

    def test_no_enhancement(self):
        card = Card(Rank.FIVE, Suit.HEARTS)
        assert not card.is_wild
        assert card.get_chip_mult() == 0
        assert card.get_chip_x_mult() == 0.0
        assert card.get_held_x_mult() == 0.0
        assert card.get_held_dollars() == 0

    def test_debuffed_enhancement_returns_zero(self):
        card = Card(Rank.FIVE, Suit.HEARTS, enhancement=Enhancement.MULT, face_down=True)
        assert card.get_chip_mult() == 0
        assert card.chip_value == 0


class TestEditions:
    def test_foil_50_chips(self):
        card = Card(Rank.FIVE, Suit.HEARTS, edition=Edition.FOIL)
        chips, mult, x_mult = card.get_edition_bonus()
        assert chips == 50
        assert mult == 0
        assert x_mult == 0.0

    def test_holo_10_mult(self):
        card = Card(Rank.FIVE, Suit.HEARTS, edition=Edition.HOLO)
        chips, mult, x_mult = card.get_edition_bonus()
        assert chips == 0
        assert mult == 10
        assert x_mult == 0.0

    def test_polychrome_x15(self):
        card = Card(Rank.FIVE, Suit.HEARTS, edition=Edition.POLYCHROME)
        chips, mult, x_mult = card.get_edition_bonus()
        assert chips == 0
        assert mult == 0
        assert x_mult == 1.5

    def test_no_edition(self):
        card = Card(Rank.FIVE, Suit.HEARTS)
        assert card.get_edition_bonus() == (0, 0, 0.0)

    def test_debuffed_edition(self):
        card = Card(Rank.FIVE, Suit.HEARTS, edition=Edition.FOIL, face_down=True)
        assert card.get_edition_bonus() == (0, 0, 0.0)


class TestSeals:
    def test_gold_seal_played_dollars(self):
        card = Card(Rank.FIVE, Suit.HEARTS, seal=Seal.GOLD)
        assert card.get_played_dollars() == 3

    def test_no_seal_no_dollars(self):
        card = Card(Rank.FIVE, Suit.HEARTS)
        assert card.get_played_dollars() == 0

    def test_debuffed_gold_seal_no_dollars(self):
        card = Card(Rank.FIVE, Suit.HEARTS, seal=Seal.GOLD, face_down=True)
        assert card.get_played_dollars() == 0


class TestDeck:
    def test_standard_deck_52_cards(self):
        deck = make_standard_deck()
        assert len(deck) == 52
        assert len(set(deck)) == 52  # all unique uids

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
        more = deck.draw(10)
        assert len(more) == 10

    def test_deck_seeded_reproducibility(self):
        d1 = Deck(np.random.default_rng(42))
        d1.reset()
        hand1 = d1.draw(8)

        d2 = Deck(np.random.default_rng(42))
        d2.reset()
        hand2 = d2.draw(8)

        # Same rank/suit (seeded), but different uids
        for c1, c2 in zip(hand1, hand2):
            assert c1.same_card(c2)

    def test_return_cards(self):
        deck = Deck(np.random.default_rng(42))
        deck.reset()
        drawn = deck.draw(5)
        deck.return_cards(drawn)
        assert len(deck.discard_pile) == 5

    def test_add_card(self):
        deck = Deck(np.random.default_rng(42))
        deck.reset()
        assert deck.card_count == 52
        new_card = Card(Rank.ACE, Suit.SPADES)
        deck.add_card(new_card)
        assert deck.card_count == 53
        assert deck.cards_remaining == 53

    def test_remove_card(self):
        deck = Deck(np.random.default_rng(42))
        deck.reset()
        card = deck.draw_pile[0]
        assert deck.remove_card(card)
        assert deck.card_count == 51

    def test_remove_card_from_discard(self):
        deck = Deck(np.random.default_rng(42))
        deck.reset()
        drawn = deck.draw(5)
        deck.return_cards(drawn)
        card = deck.discard_pile[0]
        assert deck.remove_card(card)
        assert len(deck.discard_pile) == 4

    def test_remove_nonexistent_card(self):
        deck = Deck(np.random.default_rng(42))
        deck.reset()
        fake = Card(Rank.ACE, Suit.SPADES)  # not in deck
        assert not deck.remove_card(fake)

    def test_all_cards(self):
        deck = Deck(np.random.default_rng(42))
        deck.reset()
        drawn = deck.draw(10)
        deck.return_cards(drawn[:5])
        # 42 in draw pile + 5 in discard = 47 (5 drawn cards not returned are gone)
        assert len(deck.all_cards) == 47

    def test_reset_with_custom_cards(self):
        deck = Deck(np.random.default_rng(42))
        custom = [Card(Rank.ACE, Suit.SPADES) for _ in range(5)]
        deck.reset(cards=custom)
        assert deck.card_count == 5
        # Cards should be copies (different uids)
        assert all(c.rank == Rank.ACE for c in deck.draw_pile)
        assert all(c.uid != custom[0].uid for c in deck.draw_pile)

    def test_card_count(self):
        deck = Deck(np.random.default_rng(42))
        deck.reset()
        assert deck.card_count == 52
        deck.draw(10)
        assert deck.card_count == 42
