"""Tests for the voucher system (balatro_gym/core/voucher.py + shop integration)."""

import numpy as np
import pytest

import balatro_gym
from balatro_gym.core.voucher import (
    BaseVoucher, VoucherInfo, VoucherEffects,
    create_voucher, get_all_voucher_ids, get_voucher_class, register_voucher,
)
from balatro_gym.envs.balatro_env import (
    BUY_OFFSET, NUM_BUY_ACTIONS, TOTAL_ACTIONS,
)
from balatro_gym.envs.configs import GameConfig


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestVoucherRegistry:
    def test_starter_vouchers_registered(self):
        ids = set(get_all_voucher_ids())
        assert {
            "v_overstock_norm", "v_clearance_sale", "v_hone",
            "v_reroll_surplus", "v_grabber",
        }.issubset(ids)

    def test_get_voucher_class(self):
        cls = get_voucher_class("v_overstock_norm")
        assert cls.INFO.id == "v_overstock_norm"
        assert cls.INFO.cost == 10

    def test_unknown_voucher_raises(self):
        with pytest.raises(KeyError):
            get_voucher_class("v_nonexistent")


# ---------------------------------------------------------------------------
# Static effects per starter
# ---------------------------------------------------------------------------

class TestVoucherEffects:
    def test_overstock_grants_shop_slot(self):
        assert get_voucher_class("v_overstock_norm").EFFECTS.shop_slots_delta == 1

    def test_clearance_sale_discount(self):
        assert get_voucher_class("v_clearance_sale").EFFECTS.shop_cost_multiplier == 0.75

    def test_hone_doubles_edition_rate(self):
        assert get_voucher_class("v_hone").EFFECTS.edition_rate_multiplier == 2.0

    def test_reroll_surplus_discounts_reroll(self):
        assert get_voucher_class("v_reroll_surplus").EFFECTS.reroll_base_cost_delta == -2

    def test_grabber_adds_hand(self):
        assert get_voucher_class("v_grabber").EFFECTS.hands_per_round_delta == 1


# ---------------------------------------------------------------------------
# Action space bumped
# ---------------------------------------------------------------------------

class TestActionSpace:
    def test_buy_actions_extended_to_four(self):
        assert NUM_BUY_ACTIONS == 4

    def test_total_actions_is_447(self):
        # Phase 1d landed 447. Phase 1e added SKIP_BLIND → 448.
        # Keeping the test name for git-blame continuity; assert current total.
        assert TOTAL_ACTIONS == 448


# ---------------------------------------------------------------------------
# GameConfig integration
# ---------------------------------------------------------------------------

class TestGameConfigIntegration:
    def test_default_voucher_pool_empty(self):
        cfg = GameConfig.easy()
        assert cfg.voucher_pool == []

    def test_invalid_voucher_id_raises(self):
        with pytest.raises(ValueError, match="Unknown voucher ID"):
            GameConfig(voucher_pool=["v_nope"])

    def test_valid_voucher_pool_accepted(self):
        cfg = GameConfig(voucher_pool=["v_overstock_norm", "v_hone"])
        assert cfg.voucher_pool == ["v_overstock_norm", "v_hone"]

    def test_to_dict_roundtrip(self):
        cfg = GameConfig(
            voucher_pool=["v_grabber"], joker_pool=["joker_basic"],
        )
        d = cfg.to_dict()
        assert d["voucher_pool"] == ["v_grabber"]
        cfg2 = GameConfig(**d)
        assert cfg2.voucher_pool == ["v_grabber"]


# ---------------------------------------------------------------------------
# Env integration / redemption effects
# ---------------------------------------------------------------------------

def _make_env(voucher_ids, joker_pool=None):
    cfg = GameConfig(
        num_antes=4, hands_per_round=4, discards_per_round=3,
        hand_size=8, max_jokers=5, starting_money=100,  # plenty to buy
        shop_slots=2, reroll_base_cost=5, consumable_slots=2,
        joker_pool=joker_pool or ["joker_basic"],
        starting_joker_ids=[],
        consumable_pool=[],
        voucher_pool=list(voucher_ids),
        seed=42,
    )
    env = balatro_gym.make(config=cfg)
    env.reset(seed=42)
    return env


def _force_voucher_in_shop(state, voucher_id: str) -> int:
    """Replace the first voucher slot in the shop with a specific voucher.

    Useful for deterministic tests of redemption effects. Returns the slot
    index of the voucher offering.
    """
    from balatro_gym.core.shop import ShopOffering
    voucher = create_voucher(voucher_id)
    for i, o in enumerate(state.shop.offerings):
        if o.item_type == "voucher":
            state.shop.offerings[i] = ShopOffering(
                voucher=voucher, cost=voucher.INFO.cost, item_type="voucher",
            )
            return i
    # No voucher offering exists — append one.
    state.shop.offerings.append(ShopOffering(
        voucher=voucher, cost=voucher.INFO.cost, item_type="voucher",
    ))
    return len(state.shop.offerings) - 1


class TestEnvIntegration:
    def test_shop_includes_voucher_slot_when_pool_set(self):
        env = _make_env(["v_overstock_norm"])
        # Move to shop phase by ending the round artificially:
        # easier — just inspect the shop directly after generating.
        env._game.shop.generate_offerings()
        types = [o.item_type for o in env._game.shop.offerings]
        assert "voucher" in types

    def test_no_voucher_slot_when_pool_empty(self):
        env = _make_env([])
        env._game.shop.generate_offerings()
        types = [o.item_type for o in env._game.shop.offerings]
        assert "voucher" not in types

    def test_overstock_redeem_grows_shop(self):
        env = _make_env(["v_overstock_norm"])
        state = env._game
        state.phase = type(state.phase).SHOP
        state.shop.generate_offerings()
        slot = _force_voucher_in_shop(state, "v_overstock_norm")
        before_slots = state.shop.num_slots
        ok = state.shop_buy(slot)
        assert ok
        # Overstock raised num_slots by 1.
        assert state.shop.num_slots == before_slots + 1
        # And it's logged in redeemed_vouchers.
        assert "v_overstock_norm" in state.redeemed_vouchers
        # Pool no longer contains it (can't re-spawn).
        assert "v_overstock_norm" not in state.shop.voucher_pool

    def test_clearance_sale_discount_applies_next_shop(self):
        env = _make_env(["v_clearance_sale"])
        state = env._game
        state.phase = type(state.phase).SHOP
        state.shop.generate_offerings()
        slot = _force_voucher_in_shop(state, "v_clearance_sale")
        state.shop_buy(slot)
        # Now re-generate — the new offerings should be 25% off.
        state.shop.generate_offerings()
        for offering in state.shop.offerings:
            if offering.item_type == "joker":
                expected = max(1, int(offering.joker.cost_with_edition * 0.75 + 0.5))
                assert offering.cost == expected

    def test_hone_doubles_shop_edition_rate(self):
        env = _make_env(["v_hone"])
        state = env._game
        state.phase = type(state.phase).SHOP
        state.shop.generate_offerings()
        slot = _force_voucher_in_shop(state, "v_hone")
        before = state.shop.edition_rate
        state.shop_buy(slot)
        assert state.shop.edition_rate == before * 2.0

    def test_reroll_surplus_drops_reroll_cost(self):
        env = _make_env(["v_reroll_surplus"])
        state = env._game
        state.phase = type(state.phase).SHOP
        state.shop.generate_offerings()
        slot = _force_voucher_in_shop(state, "v_reroll_surplus")
        before_base = state.shop.reroll_base_cost
        state.shop_buy(slot)
        assert state.shop.reroll_base_cost == before_base - 2

    def test_grabber_adds_one_hand_per_round(self):
        env = _make_env(["v_grabber"])
        state = env._game
        state.phase = type(state.phase).SHOP
        state.shop.generate_offerings()
        slot = _force_voucher_in_shop(state, "v_grabber")
        before = state.hands_per_round
        state.shop_buy(slot)
        assert state.hands_per_round == before + 1


# ---------------------------------------------------------------------------
# Pool exclusion after redemption
# ---------------------------------------------------------------------------

class TestPoolExclusion:
    def test_redeemed_voucher_not_re_spawned(self):
        env = _make_env(["v_overstock_norm"])
        state = env._game
        state.phase = type(state.phase).SHOP
        state.shop.generate_offerings()
        slot = _force_voucher_in_shop(state, "v_overstock_norm")
        state.shop_buy(slot)
        # Pool is now empty. Generating offerings shouldn't create a voucher.
        state.shop.generate_offerings()
        types = [o.item_type for o in state.shop.offerings]
        assert "voucher" not in types


# ---------------------------------------------------------------------------
# Plugin extensibility
# ---------------------------------------------------------------------------

class TestCustomVoucherPlugin:
    def test_register_custom_voucher(self):
        @register_voucher
        class _TestVoucher(BaseVoucher):
            INFO = VoucherInfo(
                id="v_test_only", name="Test", description="test", cost=7,
            )
            EFFECTS = VoucherEffects(hand_size_delta=2)

        try:
            assert "v_test_only" in get_all_voucher_ids()
            v = create_voucher("v_test_only")
            assert v.EFFECTS.hand_size_delta == 2
        finally:
            from balatro_gym.core.voucher import _VOUCHER_REGISTRY
            _VOUCHER_REGISTRY.pop("v_test_only", None)
