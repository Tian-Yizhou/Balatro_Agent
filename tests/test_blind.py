"""Tests for blind.py: blind targets, boss effects."""

import numpy as np
import pytest

from balatro_gym.core.blind import (
    BlindType,
    BlindDef,
    BlindManager,
    SMALL_BLIND,
    BIG_BLIND,
    BOSS_BLINDS,
    get_blind_amount,
    DebuffSuit,
    DebuffFaceCards,
    TheNeedle,
    TheWall,
    TheFlint,
    TheHook,
)
from balatro_gym.core.card import Card, Rank, Suit


class TestBlindAmounts:
    """Verify get_blind_amount matches Lua's base amounts."""

    def test_ante_1(self):
        assert get_blind_amount(1) == 300

    def test_ante_2(self):
        assert get_blind_amount(2) == 800

    def test_ante_3(self):
        assert get_blind_amount(3) == 2000

    def test_ante_4(self):
        assert get_blind_amount(4) == 5000

    def test_ante_5(self):
        assert get_blind_amount(5) == 11000

    def test_ante_6(self):
        assert get_blind_amount(6) == 20000

    def test_ante_7(self):
        assert get_blind_amount(7) == 35000

    def test_ante_8(self):
        assert get_blind_amount(8) == 50000

    def test_ante_0(self):
        assert get_blind_amount(0) == 100

    def test_ante_9_extrapolation(self):
        # Should return some value > 50000
        val = get_blind_amount(9)
        assert val > 50000


class TestBlindDefs:
    def test_small_blind(self):
        assert SMALL_BLIND.dollars == 3
        assert SMALL_BLIND.mult == 1.0

    def test_big_blind(self):
        assert BIG_BLIND.dollars == 4
        assert BIG_BLIND.mult == 1.5

    def test_boss_blinds_exist(self):
        assert len(BOSS_BLINDS) >= 7
        for bd in BOSS_BLINDS:
            assert bd.is_boss
            assert bd.dollars == 5

    def test_wall_mult(self):
        wall = next(b for b in BOSS_BLINDS if b.name == "The Wall")
        assert wall.mult == 4.0

    def test_needle_mult(self):
        needle = next(b for b in BOSS_BLINDS if b.name == "The Needle")
        assert needle.mult == 1.0


class TestScoreTargets:
    def test_small_blind_ante_1(self):
        bm = BlindManager(8)
        target = bm.get_score_target(1, SMALL_BLIND)
        # 300 * 1.0 = 300
        assert target == 300

    def test_big_blind_ante_1(self):
        bm = BlindManager(8)
        target = bm.get_score_target(1, BIG_BLIND)
        # 300 * 1.5 = 450
        assert target == 450

    def test_boss_wall_ante_1(self):
        bm = BlindManager(8)
        wall = next(b for b in BOSS_BLINDS if b.name == "The Wall")
        target = bm.get_score_target(1, wall)
        # 300 * 4.0 = 1200
        assert target == 1200

    def test_boss_ante_8(self):
        bm = BlindManager(8)
        boss_2x = BlindDef(name="Test", dollars=5, mult=2.0, is_boss=True)
        target = bm.get_score_target(8, boss_2x)
        # 50000 * 2.0 = 100000
        assert target == 100000


class TestBossEffects:
    def test_debuff_suit_hearts(self):
        from balatro_gym.core.card import Card, Rank, Suit

        effect = DebuffSuit(Suit.HEARTS, "The Head", "Hearts debuffed")
        assert effect.debuff_card(Card(Rank.ACE, Suit.HEARTS)) is True
        assert effect.debuff_card(Card(Rank.ACE, Suit.SPADES)) is False

    def test_debuff_face_cards(self):
        effect = DebuffFaceCards()
        assert effect.debuff_card(Card(Rank.JACK, Suit.HEARTS)) is True
        assert effect.debuff_card(Card(Rank.QUEEN, Suit.DIAMONDS)) is True
        assert effect.debuff_card(Card(Rank.KING, Suit.CLUBS)) is True
        assert effect.debuff_card(Card(Rank.TEN, Suit.SPADES)) is False
        assert effect.debuff_card(Card(Rank.ACE, Suit.HEARTS)) is False

    def test_the_flint_halves(self):
        effect = TheFlint()
        # Halves with rounding: 7*0.5+0.5 = 4, 45*0.5+0.5 = 23
        mult, chips, modified = effect.modify_hand(7, 45)
        assert modified
        assert mult == 4   # floor(7*0.5+0.5) = floor(4.0) = 4
        assert chips == 23  # floor(45*0.5+0.5) = floor(23.0) = 23

    def test_the_flint_odd_values(self):
        effect = TheFlint()
        # 3*0.5+0.5 = 2.0
        mult, chips, _ = effect.modify_hand(3, 11)
        assert mult == 2
        assert chips == 6   # floor(11*0.5+0.5) = floor(6.0) = 6


class TestBlindManager:
    def test_blind_sequence(self):
        bm = BlindManager(2)
        seq = bm.get_blind_sequence()
        assert len(seq) == 6  # 2 antes * 3 blinds
        assert seq[0] == (1, BlindType.SMALL)
        assert seq[1] == (1, BlindType.BIG)
        assert seq[2] == (1, BlindType.BOSS)
        assert seq[3] == (2, BlindType.SMALL)

    def test_total_blinds(self):
        bm = BlindManager(8)
        assert bm.total_blinds == 24

    def test_choose_boss(self):
        bm = BlindManager(8)
        rng = np.random.default_rng(42)
        boss = bm.choose_boss(1, rng)
        assert boss.is_boss
        assert boss in BOSS_BLINDS
