"""Tests for shop.py: offerings, buying, selling, rerolling."""

import numpy as np
import pytest

from balatro_gym.core.joker import create_joker
from balatro_gym.core.shop import Shop, ShopOffering


@pytest.fixture
def shop():
    pool = ["joker_basic", "greedy_joker", "lusty_joker", "banner"]
    return Shop(joker_pool=pool, rng=np.random.default_rng(42), num_slots=2)


class TestShopGeneration:
    def test_generate_offerings(self, shop):
        shop.generate_offerings()
        assert len(shop.offerings) == 2
        for o in shop.offerings:
            assert not o.sold
            assert o.cost > 0

    def test_empty_pool_no_offerings(self):
        s = Shop(joker_pool=[], rng=np.random.default_rng(42))
        s.generate_offerings()
        assert len(s.offerings) == 0


class TestBuying:
    def test_buy_success(self, shop):
        shop.generate_offerings()
        cost = shop.offerings[0].cost
        joker, remaining = shop.buy_joker(0, cost + 10, [], 5)
        assert joker is not None
        assert remaining == cost + 10 - cost
        assert shop.offerings[0].sold

    def test_buy_cant_afford(self, shop):
        shop.generate_offerings()
        joker, remaining = shop.buy_joker(0, 0, [], 5)
        assert joker is None
        assert remaining == 0

    def test_buy_full_joker_slots(self, shop):
        shop.generate_offerings()
        existing = [create_joker("joker_basic")] * 5
        joker, remaining = shop.buy_joker(0, 100, existing, 5)
        assert joker is None

    def test_buy_already_sold(self, shop):
        shop.generate_offerings()
        shop.offerings[0].sold = True
        joker, remaining = shop.buy_joker(0, 100, [], 5)
        assert joker is None

    def test_buy_invalid_slot(self, shop):
        shop.generate_offerings()
        joker, remaining = shop.buy_joker(99, 100, [], 5)
        assert joker is None


class TestSelling:
    def test_sell_value(self, shop):
        joker = create_joker("joker_basic")  # cost = 2
        assert shop.sell_value(joker) == max(1, 2 // 2)  # = 1

    def test_sell_value_expensive(self, shop):
        joker = create_joker("blueprint")  # cost = 10
        assert shop.sell_value(joker) == 5

    def test_sell_value_minimum_1(self, shop):
        joker = create_joker("joker_basic")  # cost = 2, 2//2=1
        assert shop.sell_value(joker) >= 1


class TestRerolling:
    def test_reroll_success(self, shop):
        shop.generate_offerings()
        old = [o.joker.INFO.id for o in shop.offerings]
        success, remaining = shop.reroll(10)
        assert success
        assert remaining == 10 - 5  # base cost = 5

    def test_reroll_cant_afford(self, shop):
        shop.generate_offerings()
        success, remaining = shop.reroll(2)  # base cost = 5
        assert not success
        assert remaining == 2

    def test_reroll_cost_increases(self, shop):
        shop.generate_offerings()
        shop.reroll(100)
        assert shop.reroll_cost == 6  # 5 + 1
        shop.reroll(100)
        assert shop.reroll_cost == 7

    def test_reroll_resets_on_generate(self, shop):
        shop.generate_offerings()
        shop.reroll(100)
        shop.reroll(100)
        assert shop.reroll_cost == 7
        shop.generate_offerings()
        assert shop.reroll_cost == 5  # reset

    def test_available_offerings(self, shop):
        shop.generate_offerings()
        available = shop.get_available_offerings()
        assert len(available) == 2
        shop.offerings[0].sold = True
        available = shop.get_available_offerings()
        assert len(available) == 1
