"""Tests for the tag system + skip-blind mechanic."""

import numpy as np
import pytest

import balatro_gym
from balatro_gym.core.blind import BlindType, SMALL_BLIND, BIG_BLIND
from balatro_gym.core.card import Edition
from balatro_gym.core.game_state import GamePhase
from balatro_gym.core.tag import (
    BaseTag, TagInfo,
    create_tag, get_all_tag_ids, get_tag_class, register_tag,
)
from balatro_gym.envs.balatro_env import (
    SKIP_BLIND_ACTION, TOTAL_ACTIONS,
)
from balatro_gym.envs.configs import GameConfig


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestTagRegistry:
    def test_starter_tags_registered(self):
        ids = set(get_all_tag_ids())
        assert {
            "tag_investment", "tag_handy", "tag_foil",
            "tag_voucher", "tag_juggle",
        }.issubset(ids)

    def test_unknown_tag_raises(self):
        with pytest.raises(KeyError):
            get_tag_class("tag_nonexistent")


# ---------------------------------------------------------------------------
# Action layout
# ---------------------------------------------------------------------------

class TestActionLayout:
    def test_skip_blind_action_index(self):
        assert SKIP_BLIND_ACTION == 447

    def test_total_actions(self):
        assert TOTAL_ACTIONS == 448


# ---------------------------------------------------------------------------
# GameConfig
# ---------------------------------------------------------------------------

class TestGameConfigIntegration:
    def test_default_tag_pool_empty(self):
        cfg = GameConfig.easy()
        assert cfg.tag_pool == []

    def test_invalid_tag_id_raises(self):
        with pytest.raises(ValueError, match="Unknown tag ID"):
            GameConfig(tag_pool=["tag_nope"])

    def test_to_dict_roundtrip(self):
        cfg = GameConfig(
            tag_pool=["tag_handy"], joker_pool=["joker_basic"],
        )
        d = cfg.to_dict()
        assert d["tag_pool"] == ["tag_handy"]
        assert GameConfig(**d).tag_pool == ["tag_handy"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(tag_ids, voucher_pool=None, starting_money=20):
    cfg = GameConfig(
        num_antes=4, hands_per_round=4, discards_per_round=3,
        hand_size=8, max_jokers=5, starting_money=starting_money,
        shop_slots=2, reroll_base_cost=5, consumable_slots=2,
        joker_pool=["joker_basic"],
        starting_joker_ids=[],
        consumable_pool=[],
        voucher_pool=voucher_pool or [],
        tag_pool=list(tag_ids),
        seed=42,
    )
    env = balatro_gym.make(config=cfg)
    env.reset(seed=42)
    return env


def _force_active_tag(state, tag_id: str):
    """Bypass the RNG and directly award a specific tag."""
    tag = create_tag(tag_id)
    consumed = tag.on_award(state)
    if not consumed:
        state.active_tags.append(tag)
    return tag


# ---------------------------------------------------------------------------
# Skip-blind mechanic
# ---------------------------------------------------------------------------

class TestSkipBlindMechanic:
    def test_skip_small_advances_to_big(self):
        env = _make_env(["tag_juggle"])
        state = env._game
        assert state.current_blind_type == BlindType.SMALL
        result = state.skip_blind()
        assert result is not None
        assert state.current_blind_type == BlindType.BIG
        # Skip doesn't increment blinds_beaten.
        assert state.blinds_beaten == 0

    def test_cannot_skip_boss(self):
        env = _make_env(["tag_juggle"])
        state = env._game
        # Advance to boss.
        state.skip_blind()              # skip Small → Big
        state.skip_blind()              # skip Big → Boss
        assert state.current_blind_type == BlindType.BOSS
        result = state.skip_blind()
        assert result is None
        # Still on boss.
        assert state.current_blind_type == BlindType.BOSS

    def test_skip_with_empty_pool_returns_none(self):
        env = _make_env([])
        state = env._game
        assert state.skip_blind() is None


# ---------------------------------------------------------------------------
# Action mask
# ---------------------------------------------------------------------------

class TestActionMask:
    def test_skip_blind_masked_when_pool_empty(self):
        env = _make_env([])
        mask = env.action_masks()
        assert not mask[SKIP_BLIND_ACTION]

    def test_skip_blind_unmasked_on_small_with_pool(self):
        env = _make_env(["tag_juggle"])
        mask = env.action_masks()
        assert mask[SKIP_BLIND_ACTION]

    def test_skip_blind_masked_on_boss(self):
        env = _make_env(["tag_juggle"])
        state = env._game
        state.skip_blind()                                 # Small → Big
        state.skip_blind()                                 # Big → Boss
        mask = env.action_masks()
        assert not mask[SKIP_BLIND_ACTION]


# ---------------------------------------------------------------------------
# Per-tag effects
# ---------------------------------------------------------------------------

class TestInvestmentTag:
    def test_pays_25_on_next_boss(self):
        env = _make_env(["tag_investment"])
        state = env._game
        _force_active_tag(state, "tag_investment")
        before = state.money
        # Fire on_blind_beaten for a Small — should NOT pay yet.
        state._fire_tag_hooks("on_blind_beaten", BlindType.SMALL)
        assert state.money == before
        assert len(state.active_tags) == 1
        # Fire for Boss — pays $25 and consumes.
        state._fire_tag_hooks("on_blind_beaten", BlindType.BOSS)
        assert state.money == before + 25
        assert len(state.active_tags) == 0


class TestHandyTag:
    def test_pays_one_per_hand_played_immediately(self):
        env = _make_env(["tag_handy"])
        state = env._game
        state.total_hands_played = 7
        before = state.money
        tag = create_tag("tag_handy")
        consumed = tag.on_award(state)
        assert consumed is True
        assert state.money == before + 7


class TestFoilTag:
    def test_first_joker_offering_becomes_foil(self):
        env = _make_env(["tag_foil"])
        state = env._game
        state.phase = GamePhase.SHOP
        state.shop.generate_offerings()
        _force_active_tag(state, "tag_foil")
        state._fire_tag_hooks("on_shop_enter")
        # Find the first joker offering; it must now be Foil.
        for offering in state.shop.offerings:
            if offering.item_type == "joker":
                assert offering.joker.edition == Edition.FOIL
                # Cost includes Foil bump (+$2).
                assert offering.cost == state.shop._apply_cost_mult(
                    offering.joker.cost_with_edition
                )
                break
        else:
            pytest.fail("no joker offering in shop")
        assert len(state.active_tags) == 0


class TestVoucherTag:
    def test_adds_extra_voucher_offering(self):
        env = _make_env(["tag_voucher"], voucher_pool=["v_overstock_norm"])
        state = env._game
        state.phase = GamePhase.SHOP
        state.shop.generate_offerings()
        voucher_count_before = sum(
            1 for o in state.shop.offerings if o.item_type == "voucher"
        )
        _force_active_tag(state, "tag_voucher")
        state._fire_tag_hooks("on_shop_enter")
        voucher_count_after = sum(
            1 for o in state.shop.offerings if o.item_type == "voucher"
        )
        assert voucher_count_after == voucher_count_before + 1


class TestJuggleTag:
    def test_bumps_hand_size_for_next_round_only(self):
        env = _make_env(["tag_juggle"])
        state = env._game
        base = state.effective_hand_size
        _force_active_tag(state, "tag_juggle")
        # Hook fires inside _start_blind, so simulate next round start.
        state._fire_tag_hooks("on_round_start")
        assert state.effective_hand_size == base + 3
        assert len(state.active_tags) == 0  # consumed
        # End of round resets the bonus.
        state.current_round_hand_size_bonus = 0
        assert state.effective_hand_size == base


# ---------------------------------------------------------------------------
# Plugin extensibility
# ---------------------------------------------------------------------------

class TestCustomTagPlugin:
    def test_register_custom_tag(self):
        @register_tag
        class _TestTag(BaseTag):
            INFO = TagInfo(
                id="tag_test_only", name="Test", description="test",
            )
        try:
            assert "tag_test_only" in get_all_tag_ids()
            t = create_tag("tag_test_only")
            # All four hooks are no-ops by default.
            assert t.on_award(object()) is False
            assert t.on_shop_enter(object()) is False
            assert t.on_round_start(object()) is False
            assert t.on_blind_beaten(object(), BlindType.SMALL) is False
        finally:
            from balatro_gym.core.tag import _TAG_REGISTRY
            _TAG_REGISTRY.pop("tag_test_only", None)
