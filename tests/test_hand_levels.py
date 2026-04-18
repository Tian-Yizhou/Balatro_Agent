"""Tests for hand_levels.py: HandLevelData, HandLevelManager."""

import pytest

from balatro_gym.core.hand_evaluator import HandType
from balatro_gym.core.hand_levels import HandLevelData, HandLevelManager


class TestHandLevelData:
    def test_level_1_returns_starting_values(self):
        data = HandLevelData(level=1, s_chips=10, s_mult=2, l_chips=15, l_mult=1)
        assert data.chips == 10
        assert data.mult == 2

    def test_level_2_adds_increment(self):
        data = HandLevelData(level=2, s_chips=10, s_mult=2, l_chips=15, l_mult=1)
        assert data.chips == 25   # 10 + 15*(2-1)
        assert data.mult == 3     # 2 + 1*(2-1)

    def test_level_5(self):
        data = HandLevelData(level=5, s_chips=30, s_mult=4, l_chips=30, l_mult=3)
        assert data.chips == 150  # 30 + 30*4
        assert data.mult == 16    # 4 + 3*4

    def test_chips_floor_zero(self):
        data = HandLevelData(level=1, s_chips=0, s_mult=1, l_chips=0, l_mult=0)
        assert data.chips == 0

    def test_mult_floor_one(self):
        data = HandLevelData(level=1, s_chips=0, s_mult=0, l_chips=0, l_mult=0)
        assert data.mult == 1  # floor of 1


class TestHandLevelManager:
    def test_all_hand_types_present(self):
        mgr = HandLevelManager()
        for ht in HandType:
            assert mgr.get_level(ht) is not None

    def test_initial_levels_are_one(self):
        mgr = HandLevelManager()
        for ht in HandType:
            assert mgr.get_level(ht).level == 1

    def test_pair_level_1_score(self):
        mgr = HandLevelManager()
        chips, mult = mgr.get_score(HandType.PAIR)
        assert chips == 10
        assert mult == 2

    def test_high_card_level_1_score(self):
        mgr = HandLevelManager()
        chips, mult = mgr.get_score(HandType.HIGH_CARD)
        assert chips == 5
        assert mult == 1

    def test_flush_level_1_score(self):
        mgr = HandLevelManager()
        chips, mult = mgr.get_score(HandType.FLUSH)
        assert chips == 35
        assert mult == 4

    def test_straight_flush_level_1_score(self):
        mgr = HandLevelManager()
        chips, mult = mgr.get_score(HandType.STRAIGHT_FLUSH)
        assert chips == 100
        assert mult == 8

    def test_level_up_pair(self):
        mgr = HandLevelManager()
        mgr.level_up(HandType.PAIR)
        data = mgr.get_level(HandType.PAIR)
        assert data.level == 2
        chips, mult = mgr.get_score(HandType.PAIR)
        assert chips == 25   # 10 + 15
        assert mult == 3     # 2 + 1

    def test_level_up_multiple(self):
        mgr = HandLevelManager()
        mgr.level_up(HandType.PAIR, amount=3)
        data = mgr.get_level(HandType.PAIR)
        assert data.level == 4
        chips, mult = mgr.get_score(HandType.PAIR)
        assert chips == 55   # 10 + 15*3
        assert mult == 5     # 2 + 1*3

    def test_level_up_does_not_affect_others(self):
        mgr = HandLevelManager()
        mgr.level_up(HandType.PAIR, amount=5)
        # High Card should still be level 1
        chips, mult = mgr.get_score(HandType.HIGH_CARD)
        assert chips == 5
        assert mult == 1

    def test_reset(self):
        mgr = HandLevelManager()
        mgr.level_up(HandType.PAIR, amount=10)
        mgr.reset()
        assert mgr.get_level(HandType.PAIR).level == 1

    def test_get_all_levels(self):
        mgr = HandLevelManager()
        all_lvls = mgr.get_all_levels()
        assert len(all_lvls) == len(HandType)
        # Returned dict is a new dict (adding keys doesn't affect manager)
        del all_lvls[HandType.PAIR]
        assert mgr.get_level(HandType.PAIR) is not None

    def test_all_default_scores_match_lua(self):
        """Verify all level-1 scores against Lua game.lua defaults."""
        mgr = HandLevelManager()
        expected = {
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
        for ht, (exp_chips, exp_mult) in expected.items():
            chips, mult = mgr.get_score(ht)
            assert chips == exp_chips, f"{ht.name}: chips {chips} != {exp_chips}"
            assert mult == exp_mult, f"{ht.name}: mult {mult} != {exp_mult}"

    def test_get_all_levels_is_shallow_copy(self):
        """get_all_levels returns a dict copy but shares HandLevelData refs."""
        mgr = HandLevelManager()
        all1 = mgr.get_all_levels()
        all2 = mgr.get_all_levels()
        assert all1 is not all2
