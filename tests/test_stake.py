"""Tests for the stake system (balatro_gym/core/stake.py)."""

import pytest

import balatro_gym
from balatro_gym.core.blind import (
    BlindManager, BlindType, SMALL_BLIND, BIG_BLIND, get_blind_amount,
)
from balatro_gym.core.stake import (
    BaseStake, StakeInfo, StakeModifiers,
    create_stake, get_all_stake_ids, get_stake_class, register_stake,
)
from balatro_gym.envs.configs import GameConfig


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestStakeRegistry:
    def test_starter_stakes_registered(self):
        ids = set(get_all_stake_ids())
        assert {
            "stake_white", "stake_red", "stake_green",
            "stake_blue", "stake_purple",
        }.issubset(ids)

    def test_get_stake_class(self):
        cls = get_stake_class("stake_red")
        assert cls.INFO.id == "stake_red"
        assert cls.INFO.level == 2

    def test_create_stake_returns_instance(self):
        stake = create_stake("stake_white")
        assert isinstance(stake, BaseStake)
        assert stake.INFO.level == 1

    def test_unknown_stake_raises(self):
        with pytest.raises(KeyError):
            get_stake_class("stake_nonexistent")


# ---------------------------------------------------------------------------
# Modifiers
# ---------------------------------------------------------------------------

class TestStakeModifiers:
    def test_white_is_noop(self):
        mods = get_stake_class("stake_white").MODIFIERS
        assert mods == StakeModifiers()  # all defaults

    def test_red_disables_small_blind_money(self):
        mods = get_stake_class("stake_red").MODIFIERS
        assert mods.no_small_blind_money is True
        assert mods.score_scaling_tier == 1

    def test_green_bumps_scaling_to_tier_2(self):
        mods = get_stake_class("stake_green").MODIFIERS
        # cumulative: still no small blind money
        assert mods.no_small_blind_money is True
        assert mods.score_scaling_tier == 2

    def test_blue_subtracts_discard(self):
        mods = get_stake_class("stake_blue").MODIFIERS
        # cumulative chain: Red + Green + Blue
        assert mods.no_small_blind_money is True
        assert mods.score_scaling_tier == 2
        assert mods.starting_discards_delta == -1

    def test_purple_bumps_to_tier_3(self):
        mods = get_stake_class("stake_purple").MODIFIERS
        # cumulative: Red + Green's scaling now overridden by Purple's 3, Blue's -1 carried
        assert mods.no_small_blind_money is True
        assert mods.score_scaling_tier == 3
        assert mods.starting_discards_delta == -1


# ---------------------------------------------------------------------------
# Score scaling tiers (get_blind_amount + BlindManager)
# ---------------------------------------------------------------------------

class TestScoreScaling:
    def test_tier_1_default(self):
        # Lua: 300, 800, 2000, 5000, 11000, 20000, 35000, 50000
        expected = [300, 800, 2_000, 5_000, 11_000, 20_000, 35_000, 50_000]
        for ante, want in enumerate(expected, start=1):
            assert get_blind_amount(ante) == want

    def test_tier_2_green(self):
        # Lua: 300, 900, 2600, 8000, 20000, 36000, 60000, 100000
        expected = [300, 900, 2_600, 8_000, 20_000, 36_000, 60_000, 100_000]
        for ante, want in enumerate(expected, start=1):
            assert get_blind_amount(ante, scaling_tier=2) == want

    def test_tier_3_purple(self):
        # Lua: 300, 1000, 3200, 9000, 25000, 60000, 110000, 200000
        expected = [300, 1_000, 3_200, 9_000, 25_000, 60_000, 110_000, 200_000]
        for ante, want in enumerate(expected, start=1):
            assert get_blind_amount(ante, scaling_tier=3) == want

    def test_unknown_tier_raises(self):
        with pytest.raises(ValueError, match="Unknown score scaling tier"):
            get_blind_amount(1, scaling_tier=99)

    def test_blind_manager_passes_tier(self):
        bm = BlindManager(num_antes=8, score_scaling_tier=2)
        # Ante 5 boss (mult 2.0) on tier 2 = 20000 * 2.0 = 40000
        target = bm.get_score_target(5, SMALL_BLIND)  # mult=1.0, so just amount
        assert target == 20_000

    def test_tier_3_beats_tier_1_at_ante_5(self):
        # Ante 5: tier 1 = 11000, tier 3 = 25000. Confirms separation.
        assert get_blind_amount(5, scaling_tier=1) < get_blind_amount(5, scaling_tier=3)


# ---------------------------------------------------------------------------
# GameConfig integration
# ---------------------------------------------------------------------------

class TestGameConfigIntegration:
    def test_default_stake_is_white(self):
        cfg = GameConfig.easy()
        assert cfg.stake == "stake_white"

    def test_unknown_stake_raises(self):
        with pytest.raises(ValueError, match="Unknown stake"):
            GameConfig(stake="stake_nope")

    def test_valid_stake_accepted(self):
        cfg = GameConfig(stake="stake_blue")
        assert cfg.stake == "stake_blue"

    def test_to_dict_includes_stake(self):
        cfg = GameConfig(stake="stake_green", joker_pool=["joker_basic"])
        d = cfg.to_dict()
        assert d["stake"] == "stake_green"
        cfg2 = GameConfig(**d)
        assert cfg2.stake == "stake_green"


# ---------------------------------------------------------------------------
# Env integration
# ---------------------------------------------------------------------------

def _make_env(stake_id, back_id=None):
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
        stake=stake_id,
        seed=42,
    )
    env = balatro_gym.make(config=cfg)
    env.reset(seed=42)
    return env


class TestEnvIntegration:
    def test_white_stake_no_op(self):
        env = _make_env("stake_white")
        # No changes vs vanilla.
        assert env._game.discards_per_round == 3
        # Default tier 1 → ante 1 target = 300
        assert env._game.blind_manager.get_score_target(1, SMALL_BLIND) == 300

    def test_blue_stake_removes_a_discard(self):
        env = _make_env("stake_blue")
        # base 3 + stake -1 = 2
        assert env._game.discards_per_round == 2
        assert env._game.discards_remaining == 2

    def test_green_stake_uses_tier_2_scaling(self):
        env = _make_env("stake_green")
        # Ante 5 small blind on tier 2 = 20000 * 1.0
        assert env._game.blind_manager.get_score_target(5, SMALL_BLIND) == 20_000

    def test_purple_stake_uses_tier_3(self):
        env = _make_env("stake_purple")
        # Ante 5 small on tier 3 = 25000
        assert env._game.blind_manager.get_score_target(5, SMALL_BLIND) == 25_000
        # Purple inherits Blue's -1 discard.
        assert env._game.discards_per_round == 2

    def test_stake_and_back_stack_additively(self):
        # Red Deck +1 discard, Blue Stake -1 discard → net 0 (base 3)
        env = _make_env(stake_id="stake_blue", back_id="b_red")
        assert env._game.discards_per_round == 3


# ---------------------------------------------------------------------------
# Economy: small-blind money rule
# ---------------------------------------------------------------------------

class TestSmallBlindMoney:
    def test_white_stake_small_blind_pays(self):
        env = _make_env("stake_white")
        state = env._game
        # We're on the small blind from reset.
        assert state.current_blind_type == BlindType.SMALL
        state.money = 0
        state.hands_remaining = 0
        state.discards_remaining = 0
        # Just the $3 small blind reward, no other bonuses.
        assert state._calculate_economy() == 3

    def test_red_stake_small_blind_pays_nothing(self):
        env = _make_env("stake_red")
        state = env._game
        assert state.current_blind_type == BlindType.SMALL
        state.money = 0
        state.hands_remaining = 0
        state.discards_remaining = 0
        # Red Stake: no small blind reward → $0
        assert state._calculate_economy() == 0

    def test_red_stake_big_blind_still_pays(self):
        env = _make_env("stake_red")
        state = env._game
        # Force big blind context.
        state.blind_index = 1
        state.current_blind_def = BIG_BLIND
        state.money = 0
        state.hands_remaining = 0
        state.discards_remaining = 0
        # Big Blind still pays its $4 — Red Stake only kills Small.
        assert state._calculate_economy() == 4


# ---------------------------------------------------------------------------
# Plugin extensibility
# ---------------------------------------------------------------------------

class TestCustomStakePlugin:
    def test_user_can_register_custom_stake(self):
        @register_stake
        class _TestStake(BaseStake):
            INFO = StakeInfo(
                id="stake_test_only", name="Test Stake",
                level=99, description="test",
            )
            MODIFIERS = StakeModifiers(starting_discards_delta=-3)

        try:
            assert "stake_test_only" in get_all_stake_ids()
            stake = create_stake("stake_test_only")
            assert stake.MODIFIERS.starting_discards_delta == -3
        finally:
            from balatro_gym.core.stake import _STAKE_REGISTRY
            _STAKE_REGISTRY.pop("stake_test_only", None)
