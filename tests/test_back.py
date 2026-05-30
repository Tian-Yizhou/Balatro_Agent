"""Tests for the deck-back system (balatro_gym/core/back.py)."""

import pytest

import balatro_gym
from balatro_gym.core.back import (
    BackInfo, BackModifiers, BaseBack,
    create_back, get_all_back_ids, get_back_class,
    register_back,
)
from balatro_gym.envs.configs import GameConfig


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestBackRegistry:
    def test_starter_decks_registered(self):
        ids = set(get_all_back_ids())
        assert {"b_red", "b_blue", "b_yellow", "b_black", "b_green"}.issubset(ids)

    def test_get_back_class_returns_correct_class(self):
        cls = get_back_class("b_red")
        assert cls.INFO.id == "b_red"
        assert cls.INFO.name == "Red Deck"

    def test_create_back_returns_instance(self):
        back = create_back("b_blue")
        assert isinstance(back, BaseBack)
        assert back.INFO.id == "b_blue"

    def test_get_unknown_back_raises(self):
        with pytest.raises(KeyError):
            get_back_class("b_nonexistent")


# ---------------------------------------------------------------------------
# Per-deck modifiers
# ---------------------------------------------------------------------------

class TestStarterDeckModifiers:
    def test_red_deck_adds_discard(self):
        mods = get_back_class("b_red").MODIFIERS
        assert mods.discards_per_round_delta == 1
        assert mods.hands_per_round_delta == 0

    def test_blue_deck_adds_hand(self):
        mods = get_back_class("b_blue").MODIFIERS
        assert mods.hands_per_round_delta == 1

    def test_yellow_deck_adds_money(self):
        mods = get_back_class("b_yellow").MODIFIERS
        assert mods.starting_money_delta == 10

    def test_black_deck_trades_hand_for_joker_slot(self):
        mods = get_back_class("b_black").MODIFIERS
        assert mods.max_jokers_delta == 1
        assert mods.hands_per_round_delta == -1

    def test_green_deck_has_no_static_modifiers(self):
        # Green Deck's effect is entirely runtime; static modifiers are zero.
        mods = get_back_class("b_green").MODIFIERS
        assert mods == BackModifiers()


# ---------------------------------------------------------------------------
# Runtime hooks
# ---------------------------------------------------------------------------

class TestRuntimeHooks:
    def test_default_hooks(self):
        red = create_back("b_red")
        assert red.disables_interest() is False
        assert red.money_per_unused_hand() == 1
        assert red.money_per_unused_discard() == 0

    def test_green_deck_hooks(self):
        green = create_back("b_green")
        assert green.disables_interest() is True
        assert green.money_per_unused_hand() == 2
        assert green.money_per_unused_discard() == 1


# ---------------------------------------------------------------------------
# GameConfig integration
# ---------------------------------------------------------------------------

class TestGameConfigIntegration:
    def test_no_back_by_default(self):
        cfg = GameConfig.easy()
        assert cfg.deck_back is None

    def test_invalid_back_id_raises(self):
        with pytest.raises(ValueError, match="Unknown deck_back"):
            GameConfig(deck_back="b_nope")

    def test_valid_back_id_accepted(self):
        cfg = GameConfig(deck_back="b_red")
        assert cfg.deck_back == "b_red"

    def test_to_dict_roundtrip_includes_back(self):
        cfg = GameConfig(deck_back="b_blue", joker_pool=["joker_basic"])
        d = cfg.to_dict()
        assert d["deck_back"] == "b_blue"
        # roundtrip
        cfg2 = GameConfig(**d)
        assert cfg2.deck_back == "b_blue"


# ---------------------------------------------------------------------------
# Env integration — static modifiers apply correctly through balatro_gym.make
# ---------------------------------------------------------------------------

class TestEnvIntegration:
    def _make_env(self, back_id):
        # Build an easy config and override the back. Use joker_basic only
        # to keep the obs small and deterministic.
        cfg = GameConfig(
            num_antes=4,
            hands_per_round=4,
            discards_per_round=3,
            hand_size=8,
            max_jokers=5,
            starting_money=4,
            shop_slots=2,
            reroll_base_cost=5,
            consumable_slots=2,
            joker_pool=["joker_basic"],
            starting_joker_ids=[],
            consumable_pool=[],
            deck_back=back_id,
            seed=42,
        )
        env = balatro_gym.make(config=cfg)
        env.reset(seed=42)
        return env

    def test_red_deck_adds_one_discard(self):
        env = self._make_env("b_red")
        # base discards_per_round = 3, red deck adds +1 = 4
        assert env._game.discards_per_round == 4
        assert env._game.discards_remaining == 4

    def test_blue_deck_adds_one_hand(self):
        env = self._make_env("b_blue")
        assert env._game.hands_per_round == 5
        assert env._game.hands_remaining == 5

    def test_yellow_deck_adds_ten_dollars(self):
        env = self._make_env("b_yellow")
        # base money = 4, +10 = 14
        assert env._game.money == 14

    def test_black_deck_swaps_hand_for_joker_slot(self):
        env = self._make_env("b_black")
        assert env._game.max_jokers == 6
        assert env._game.hands_per_round == 3

    def test_no_back_means_no_modification(self):
        env = self._make_env(None)
        assert env._game.discards_per_round == 3
        assert env._game.hands_per_round == 4
        assert env._game.money == 4
        assert env._game.max_jokers == 5


# ---------------------------------------------------------------------------
# Green Deck economy: replaces interest with per-resource bonuses
# ---------------------------------------------------------------------------

class TestGreenDeckEconomy:
    def _state_with_back(self, back_id):
        cfg = GameConfig(
            num_antes=4,
            hands_per_round=4,
            discards_per_round=3,
            hand_size=8,
            max_jokers=5,
            starting_money=4,
            shop_slots=2,
            reroll_base_cost=5,
            consumable_slots=2,
            joker_pool=["joker_basic"],
            starting_joker_ids=[],
            consumable_pool=[],
            deck_back=back_id,
            seed=42,
        )
        env = balatro_gym.make(config=cfg)
        env.reset(seed=42)
        return env._game

    def test_vanilla_economy(self):
        # No back: $1/hand, $0/discard, +interest.
        state = self._state_with_back(None)
        state.money = 25                  # would earn $5 interest (5*5=25)
        state.hands_remaining = 2
        state.discards_remaining = 2
        # Blind reward (small=3) + 2 hands * $1 + 2 discards * $0 + 0 jokers + $5 interest
        assert state._calculate_economy() == 3 + 2 + 0 + 5

    def test_green_deck_economy(self):
        # Green: $2/hand, $1/discard, no interest.
        state = self._state_with_back("b_green")
        state.money = 25                  # would normally earn $5 interest, but Green disables
        state.hands_remaining = 2
        state.discards_remaining = 2
        # Blind reward (small=3) + 2 hands * $2 + 2 discards * $1 + 0 jokers + 0 interest
        assert state._calculate_economy() == 3 + 4 + 2

    def test_red_deck_uses_vanilla_economy(self):
        # Red deck shouldn't change economy — only adds a discard.
        state = self._state_with_back("b_red")
        state.money = 10                  # $2 interest
        state.hands_remaining = 1
        state.discards_remaining = 3
        # blind=3 + 1 hand * $1 + 3 discards * $0 + 0 jokers + $2 interest
        assert state._calculate_economy() == 3 + 1 + 0 + 2


# ---------------------------------------------------------------------------
# Custom backs are pluggable
# ---------------------------------------------------------------------------

class TestCustomBackPlugin:
    def test_user_can_register_custom_back(self):
        @register_back
        class _TestDeck(BaseBack):
            INFO = BackInfo(id="b_test_only", name="Test Deck", description="test")
            MODIFIERS = BackModifiers(starting_money_delta=99)

        try:
            assert "b_test_only" in get_all_back_ids()
            back = create_back("b_test_only")
            assert back.MODIFIERS.starting_money_delta == 99
        finally:
            # Clean up so other tests aren't polluted
            from balatro_gym.core.back import _BACK_REGISTRY
            _BACK_REGISTRY.pop("b_test_only", None)
