"""Tests for the edition system: Negative edition, shop rolling, joker scoring."""

import numpy as np
import pytest

import balatro_gym
from balatro_gym.core.card import (
    Edition, EDITION_COST_BONUS, PLAYING_CARD_EDITIONS,
    edition_scoring_bonus,
)
from balatro_gym.core.joker import BaseJoker, create_joker, get_joker_class
from balatro_gym.core.shop import Shop, poll_edition
from balatro_gym.envs.configs import GameConfig


# ---------------------------------------------------------------------------
# Enum and helpers
# ---------------------------------------------------------------------------

class TestEditionEnum:
    def test_negative_is_in_enum(self):
        assert Edition.NEGATIVE.value == "e_negative"
        assert Edition.NEGATIVE in Edition

    def test_playing_card_editions_excludes_negative(self):
        assert Edition.NEGATIVE not in PLAYING_CARD_EDITIONS
        assert len(PLAYING_CARD_EDITIONS) == 3
        assert set(PLAYING_CARD_EDITIONS) == {
            Edition.FOIL, Edition.HOLO, Edition.POLYCHROME,
        }

    def test_edition_cost_bonus(self):
        assert EDITION_COST_BONUS[Edition.FOIL] == 2
        assert EDITION_COST_BONUS[Edition.HOLO] == 3
        assert EDITION_COST_BONUS[Edition.POLYCHROME] == 5
        assert EDITION_COST_BONUS[Edition.NEGATIVE] == 5

    def test_scoring_bonus_helper(self):
        assert edition_scoring_bonus(None) == (0, 0, 0.0)
        assert edition_scoring_bonus(Edition.FOIL) == (50, 0, 0.0)
        assert edition_scoring_bonus(Edition.HOLO) == (0, 10, 0.0)
        assert edition_scoring_bonus(Edition.POLYCHROME) == (0, 0, 1.5)
        # Negative has no scoring effect; its bonus is the slot.
        assert edition_scoring_bonus(Edition.NEGATIVE) == (0, 0, 0.0)


# ---------------------------------------------------------------------------
# Joker carries edition
# ---------------------------------------------------------------------------

class TestJokerEdition:
    def test_default_edition_none(self):
        j = create_joker("joker_basic")
        assert j.edition is None
        assert j.cost_with_edition == j.INFO.cost
        assert j.edition_scoring_bonus() == (0, 0, 0.0)

    def test_create_joker_with_edition(self):
        j = create_joker("joker_basic", edition=Edition.FOIL)
        assert j.edition == Edition.FOIL
        assert j.cost_with_edition == j.INFO.cost + 2

    def test_cost_bumps_per_edition(self):
        base = get_joker_class("joker_basic").INFO.cost
        for ed, bump in [
            (Edition.FOIL, 2), (Edition.HOLO, 3),
            (Edition.POLYCHROME, 5), (Edition.NEGATIVE, 5),
        ]:
            j = create_joker("joker_basic", edition=ed)
            assert j.cost_with_edition == base + bump


# ---------------------------------------------------------------------------
# Shop edition rolling
# ---------------------------------------------------------------------------

class TestPollEdition:
    def test_default_mostly_no_edition(self):
        rng = np.random.default_rng(0)
        counts = {None: 0, Edition.FOIL: 0, Edition.HOLO: 0,
                  Edition.POLYCHROME: 0, Edition.NEGATIVE: 0}
        n = 20_000
        for _ in range(n):
            counts[poll_edition(rng)] += 1
        # Expected: ~96% None, ~2% Foil, ~1.4% Holo, ~0.3% Poly, ~0.3% Neg.
        # Allow generous tolerance because of finite sample.
        assert 0.94 < counts[None] / n < 0.98
        assert 0.005 < counts[Edition.FOIL] / n < 0.035
        assert 0.005 < counts[Edition.HOLO] / n < 0.025

    def test_no_negative_when_disallowed(self):
        rng = np.random.default_rng(0)
        for _ in range(5_000):
            assert poll_edition(rng, allow_negative=False) != Edition.NEGATIVE

    def test_edition_rate_amplifies(self):
        # With a very high rate, "no edition" should become rare.
        rng = np.random.default_rng(0)
        editions = 0
        n = 1_000
        for _ in range(n):
            if poll_edition(rng, edition_rate=25.0) is not None:
                editions += 1
        # 25× rate roughly saturates the edition tiers (~all non-Negative bands)
        assert editions / n > 0.5


class TestShopEditionRolling:
    def test_shop_assigns_editions(self):
        rng = np.random.default_rng(42)
        shop = Shop(
            joker_pool=["joker_basic"], rng=rng,
            num_slots=2, consumable_pool=[],
            edition_rate=50.0,        # force editions for testing
        )
        # Generate many shops and confirm at least some joker offerings carry editions.
        seen_edition = False
        for _ in range(50):
            shop.generate_offerings()
            for offering in shop.offerings:
                if offering.item_type == "joker" and offering.joker.edition is not None:
                    seen_edition = True
                    # Cost should reflect the edition bump.
                    expected = offering.joker.INFO.cost + EDITION_COST_BONUS[offering.joker.edition]
                    assert offering.cost == expected
        assert seen_edition, "edition_rate=50 should produce at least one editioned joker"

    def test_consumables_never_get_editions_from_shop(self):
        rng = np.random.default_rng(0)
        shop = Shop(
            joker_pool=[], rng=rng,
            num_slots=0, consumable_pool=["c_pluto"],
            num_consumable_slots=1, edition_rate=50.0,
        )
        for _ in range(50):
            shop.generate_offerings()
            for offering in shop.offerings:
                if offering.item_type == "consumable":
                    # Base shop never rolls editions on consumables.
                    assert getattr(offering.consumable, "edition", None) is None


# ---------------------------------------------------------------------------
# Joker scoring picks up edition bonuses
# ---------------------------------------------------------------------------

class TestJokerScoringWithEdition:
    """Confirm that Foil/Holo/Polychrome jokers actually contribute during
    the main scoring loop. We use the simplest setup possible: a single
    joker_basic (+4 mult) with one of each edition, scoring a hand of all-
    same-card so the result is deterministic."""

    def _eval_with_edition(self, edition):
        from balatro_gym.core.game_state import GameState
        from balatro_gym.core.card import Card, Suit, Rank

        gs = GameState(
            num_antes=4, hands_per_round=4, discards_per_round=3,
            hand_size=8, max_jokers=5, starting_money=4,
            available_joker_ids=["joker_basic"],
            starting_joker_ids=[], consumable_slots=2,
            shop_slots=2, available_consumable_ids=[], seed=42,
        )
        gs.reset()
        # Inject one joker with the test edition
        j = create_joker("joker_basic", edition=edition)
        gs.jokers = [j]
        # Force a known hand: five Aces of Hearts (Five of a Kind).
        gs.hand = [Card(rank=Rank.ACE, suit=Suit.HEARTS) for _ in range(5)]
        score, _ = gs.play_hand([0, 1, 2, 3, 4])
        return score

    def test_foil_adds_50_chips(self):
        base = self._eval_with_edition(None)
        foil = self._eval_with_edition(Edition.FOIL)
        assert foil > base   # Foil only adds chips, so total score grows.

    def test_holo_adds_10_mult(self):
        base = self._eval_with_edition(None)
        holo = self._eval_with_edition(Edition.HOLO)
        assert holo > base

    def test_polychrome_multiplies(self):
        base = self._eval_with_edition(None)
        poly = self._eval_with_edition(Edition.POLYCHROME)
        assert poly > base

    def test_negative_no_scoring_effect(self):
        base = self._eval_with_edition(None)
        neg = self._eval_with_edition(Edition.NEGATIVE)
        # Negative grants a slot, never chips/mult — exact same score.
        assert neg == base


# ---------------------------------------------------------------------------
# Negative slot bypass
# ---------------------------------------------------------------------------

class TestNegativeSlotBypass:
    def _make_state(self):
        cfg = GameConfig(
            num_antes=4, hands_per_round=4, discards_per_round=3,
            hand_size=8, max_jokers=5, starting_money=4,
            shop_slots=2, reroll_base_cost=5, consumable_slots=2,
            joker_pool=["joker_basic"], starting_joker_ids=[],
            consumable_pool=[], seed=42,
        )
        env = balatro_gym.make(config=cfg)
        env.reset(seed=42)
        return env._game

    def test_effective_max_equals_base_with_no_negatives(self):
        gs = self._make_state()
        gs.jokers = [create_joker("joker_basic") for _ in range(3)]
        assert gs.effective_max_jokers == gs.max_jokers

    def test_each_negative_grants_a_slot(self):
        gs = self._make_state()
        gs.jokers = [
            create_joker("joker_basic"),
            create_joker("joker_basic", edition=Edition.NEGATIVE),
            create_joker("joker_basic", edition=Edition.NEGATIVE),
        ]
        # Two negatives → +2 effective slots above max_jokers.
        assert gs.effective_max_jokers == gs.max_jokers + 2

    def test_other_editions_dont_grant_slots(self):
        gs = self._make_state()
        gs.jokers = [
            create_joker("joker_basic", edition=Edition.FOIL),
            create_joker("joker_basic", edition=Edition.POLYCHROME),
        ]
        assert gs.effective_max_jokers == gs.max_jokers


# ---------------------------------------------------------------------------
# Wheel of Fortune migration
# ---------------------------------------------------------------------------

class TestWheelOfFortuneMigration:
    def test_wheel_sets_joker_edition_field(self):
        from balatro_gym.core.consumable import create_consumable
        from balatro_gym.core.game_state import GameState

        gs = GameState(
            num_antes=4, hands_per_round=4, discards_per_round=3,
            hand_size=8, max_jokers=5, starting_money=4,
            available_joker_ids=["joker_basic"],
            starting_joker_ids=["joker_basic"],
            consumable_slots=2, shop_slots=2,
            available_consumable_ids=["c_wheel_of_fortune"],
            seed=0,
        )
        gs.reset()
        wheel = create_consumable("c_wheel_of_fortune")
        # Force the 1/4 success with a high enough roll count.
        for seed in range(200):
            gs.rng = np.random.default_rng(seed)
            gs.jokers[0].edition = None
            wheel.use(gs, highlighted_indices=[])
            if gs.jokers[0].edition is not None:
                # Must NOT use _internal_state any more.
                assert "edition" not in gs.jokers[0]._internal_state
                assert gs.jokers[0].edition in {
                    Edition.FOIL, Edition.HOLO, Edition.POLYCHROME,
                }
                # Wheel never rolls Negative.
                assert gs.jokers[0].edition != Edition.NEGATIVE
                return
        pytest.fail("Wheel of Fortune never succeeded in 200 attempts.")
