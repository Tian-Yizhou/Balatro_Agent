"""The shop, consumables, stickers and run options."""

from __future__ import annotations

from collections import Counter

import pytest

from balatro_gym.sim import Game, macros
from balatro_gym.sim.actions import (
    Buy,
    BuyAndUse,
    CashOut,
    EndRun,
    EndShop,
    Reroll,
    SelectBlind,
    SellJoker,
    ToggleCard,
    UseConsumable,
)
from balatro_gym.sim.consumables import IMPLEMENTED
from balatro_gym.sim.content import CENTERS
from balatro_gym.sim.effects import JOKERS
from balatro_gym.sim.events import Event, ItemBought, RunEnded
from balatro_gym.sim.items import new_item
from balatro_gym.sim.scope import Scope
from balatro_gym.sim.shop import create_card_for_shop
from balatro_gym.sim.state import BlindSlot, HandType, Item, Phase, RunConfig, RunState, Suit
from scenarios import set_hand, started_round


def in_shop(seed: str = "shop", config: RunConfig | None = None) -> Game:
    """A game standing in the shop after winning the small blind."""
    game = Game.new(seed, config)
    game.step(SelectBlind())
    state = game.state
    assert state.blind is not None
    state.blind.chips = 1
    for action in macros.play_cards(state, [0]):
        game.step(action)
    game.step(CashOut())
    assert state.phase is Phase.SHOP
    return game


def stock(state: RunState, *keys: str) -> list[Item]:
    """Replace the shop's cards with new items ``keys``."""
    state.shop_cards = [new_item(state, key) for key in keys]
    return state.shop_cards


def give(state: RunState, key: str) -> Item:
    item = new_item(state, key)
    (state.jokers if CENTERS[key].is_joker else state.consumables).append(item)
    return item


class TestShop:
    def test_cash_out_fills_two_slots(self) -> None:
        game = in_shop()
        assert len(game.state.shop_cards) == 2
        assert all(card.cost > 0 for card in game.state.shop_cards)

    def test_shop_offers_only_implemented_items(self) -> None:
        state = in_shop().state
        scope = Scope(state)
        for _ in range(500):
            item = create_card_for_shop(scope)
            assert item.key in JOKERS or item.key in IMPLEMENTED

    def test_card_type_and_rarity_odds(self) -> None:
        state = in_shop("odds").state
        scope = Scope(state)
        sets: Counter[str] = Counter()
        rarities: Counter[int] = Counter()
        for _ in range(6000):
            center = CENTERS[create_card_for_shop(scope).key]
            sets[center.set] += 1
            if center.is_joker:
                rarities[center.rarity] += 1
        total = sum(sets.values())
        assert abs(sets["Joker"] / total - 20 / 28) < 0.03
        assert abs(sets["Tarot"] / total - 4 / 28) < 0.02
        assert abs(sets["Planet"] / total - 4 / 28) < 0.02
        jokers = sum(rarities.values())
        assert abs(rarities[1] / jokers - 0.70) < 0.03
        assert abs(rarities[2] / jokers - 0.25) < 0.03
        assert abs(rarities[3] / jokers - 0.05) < 0.015

    def test_no_duplicates_without_showman(self) -> None:
        state = in_shop().state
        scope = Scope(state)
        for _ in range(200):
            state.shop_cards = [create_card_for_shop(scope)]
            state.shop_cards.append(create_card_for_shop(scope))
            first, second = state.shop_cards
            assert first.key != second.key

    def test_buy(self) -> None:
        game = in_shop()
        state = game.state
        joker = stock(state, "j_joker")[0]
        money = state.money
        events = game.step(Buy(0))
        assert state.jokers == [joker]
        assert state.money == money - joker.cost
        assert any(isinstance(event, ItemBought) for event in events)

    def test_cannot_buy_without_money_or_room(self) -> None:
        game = in_shop()
        state = game.state
        stock(state, "j_joker")
        state.money = 0
        assert not game.is_legal(Buy(0))
        state.money = 100
        for _ in range(5):
            give(state, "j_joker")
        assert not game.is_legal(Buy(0))
        state.shop_cards[0].edition = "e_negative"
        assert game.is_legal(Buy(0))

    def test_sell(self) -> None:
        game = in_shop()
        state = game.state
        joker = give(state, "j_joker")
        campfire = give(state, "j_campfire")
        money = state.money
        game.step(SellJoker(0))
        assert state.money == money + joker.sell_value
        assert state.jokers == [campfire]
        assert campfire.values["x_mult"] == 1.25

    def test_eternal_jokers_cannot_be_sold(self) -> None:
        game = in_shop()
        give(game.state, "j_joker").eternal = True
        assert not game.is_legal(SellJoker(0))

    def test_reroll_cost_rises_and_resets(self) -> None:
        game = in_shop()
        state = game.state
        state.money = 100
        costs = []
        for _ in range(3):
            costs.append(state.reroll_cost)
            game.step(Reroll())
        assert costs == [5, 6, 7]
        assert state.money == 100 - 18
        game.step(EndShop())
        game.step(SelectBlind())
        assert state.reroll_cost == 5

    def test_reroll_replaces_the_cards(self) -> None:
        game = in_shop()
        state = game.state
        before = [card.uid for card in state.shop_cards]
        game.step(Reroll())
        assert len(state.shop_cards) == 2
        assert not {card.uid for card in state.shop_cards} & set(before)

    def test_chaos_the_clown_and_flash_card(self) -> None:
        game = Game.new("chaos")
        state = game.state
        chaos = new_item(state, "j_chaos")
        flash = give(state, "j_flash")
        state.jokers.insert(0, chaos)
        state.money = 50
        game.step(SelectBlind())
        assert state.blind is not None
        state.blind.chips = 1
        for action in macros.play_cards(state, [0]):
            game.step(action)
        game.step(CashOut())
        assert state.reroll_cost == 0
        game.step(Reroll())
        assert state.reroll_cost == 5
        assert flash.values["mult"] == 2

    def test_rentals_cost_money_each_round(self) -> None:
        game = Game.new("rent")
        state = game.state
        give(state, "j_joker").rental = True
        game.step(SelectBlind())
        assert state.blind is not None
        state.blind.chips = 1
        for action in macros.play_cards(state, [0]):
            game.step(action)
        assert state.money == 4 - 3

    def test_perishable_jokers_wear_out(self) -> None:
        game = Game.new("perish")
        state = game.state
        joker = give(state, "j_joker")
        joker.perishable = True
        joker.perish_tally = 1
        game.step(SelectBlind())
        assert state.blind is not None
        state.blind.chips = 1
        for action in macros.play_cards(state, [0]):
            game.step(action)
        assert joker.debuffed

    @pytest.mark.parametrize("stake", [4, 7, 8])
    def test_stakes_add_stickers(self, stake: int) -> None:
        state = in_shop("stickers", RunConfig(stake=stake)).state
        scope = Scope(state)
        jokers = [
            item
            for item in (create_card_for_shop(scope) for _ in range(600))
            if CENTERS[item.key].is_joker
        ]
        if stake >= 4:
            assert any(joker.eternal for joker in jokers)
        if stake >= 7:
            assert any(joker.perishable for joker in jokers)
        if stake >= 8:
            assert any(joker.rental and joker.cost == 1 for joker in jokers)
        assert not any(joker.eternal and joker.perishable for joker in jokers)

    def test_joker_slots_and_hand_size_effects(self) -> None:
        game = in_shop()
        state = game.state
        state.money = 100
        stock(state, "j_juggler", "j_drunkard")
        game.step(Buy(0))
        game.step(Buy(0))
        assert (state.hand_size, state.discards_per_round) == (9, 5)
        stock(state, "j_credit_card")
        game.step(Buy(0))
        assert state.bankrupt_at == -20
        game.step(SellJoker(0))
        assert state.hand_size == 8

    def test_showman_allows_owned_jokers_again(self) -> None:
        state = in_shop().state
        give(state, "j_joker")
        scope = Scope(state)

        def offered() -> set[str]:
            keys: set[str] = set()
            for _ in range(1500):
                state.shop_cards = []
                keys.add(create_card_for_shop(scope).key)
            return keys

        assert "j_joker" not in offered()
        give(state, "j_ring_master")
        assert "j_joker" in offered()


class TestPlanets:
    @pytest.mark.parametrize(
        ("key", "hand"),
        [
            ("c_pluto", HandType.HIGH_CARD),
            ("c_mercury", HandType.PAIR),
            ("c_uranus", HandType.TWO_PAIR),
            ("c_venus", HandType.THREE_OF_A_KIND),
            ("c_saturn", HandType.STRAIGHT),
            ("c_jupiter", HandType.FLUSH),
            ("c_earth", HandType.FULL_HOUSE),
            ("c_mars", HandType.FOUR_OF_A_KIND),
            ("c_neptune", HandType.STRAIGHT_FLUSH),
            ("c_planet_x", HandType.FIVE_OF_A_KIND),
            ("c_ceres", HandType.FLUSH_HOUSE),
            ("c_eris", HandType.FLUSH_FIVE),
        ],
    )
    def test_level_up(self, key: str, hand: HandType) -> None:
        game = Game.new("planet")
        state = game.state
        give(state, key)
        base = state.hands[hand]
        chips, mult = base.chips, base.mult
        game.step(UseConsumable(0))
        assert base.level == 2
        assert (base.chips, base.mult) == (
            chips + CENTERS_LEVEL[hand][0],
            mult + CENTERS_LEVEL[hand][1],
        )
        assert state.last_tarot_planet == key

    def test_secret_planets_need_their_hand_played(self) -> None:
        state = in_shop().state
        scope = Scope(state)
        keys = {create_card_for_shop(scope).key for _ in range(1500)}
        assert not keys & {"c_ceres", "c_eris", "c_planet_x"}

    def test_buy_and_use(self) -> None:
        game = in_shop()
        state = game.state
        stock(state, "c_pluto")
        game.step(BuyAndUse(0))
        assert state.hands[HandType.HIGH_CARD].level == 2
        assert state.consumables == []

    def test_constellation(self) -> None:
        game = Game.new("const")
        state = game.state
        constellation = give(state, "j_constellation")
        give(state, "c_pluto")
        game.step(UseConsumable(0))
        assert constellation.values["x_mult"] == pytest.approx(1.1)


CENTERS_LEVEL = {
    HandType.HIGH_CARD: (10, 1),
    HandType.PAIR: (15, 1),
    HandType.TWO_PAIR: (20, 1),
    HandType.THREE_OF_A_KIND: (20, 2),
    HandType.STRAIGHT: (30, 3),
    HandType.FLUSH: (15, 2),
    HandType.FULL_HOUSE: (25, 2),
    HandType.FOUR_OF_A_KIND: (30, 3),
    HandType.STRAIGHT_FLUSH: (40, 4),
    HandType.FIVE_OF_A_KIND: (35, 3),
    HandType.FLUSH_HOUSE: (40, 4),
    HandType.FLUSH_FIVE: (50, 3),
}


class TestTarots:
    def use_on(self, key: str, labels: str, targets: list[int]) -> Game:
        game = started_round("tarot")
        give(game.state, key)
        set_hand(game.state, labels)
        for action in macros.select_cards(game.state, targets):
            game.step(action)
        game.step(UseConsumable(0))
        return game

    @pytest.mark.parametrize(
        ("key", "enhancement"),
        [
            ("c_magician", "m_lucky"),
            ("c_empress", "m_mult"),
            ("c_heirophant", "m_bonus"),
            ("c_lovers", "m_wild"),
            ("c_chariot", "m_steel"),
            ("c_justice", "m_glass"),
            ("c_devil", "m_gold"),
            ("c_tower", "m_stone"),
        ],
    )
    def test_enhance(self, key: str, enhancement: str) -> None:
        game = self.use_on(key, "AS KS QS", [0])
        assert game.state.hand[0].enhancement == enhancement
        assert game.state.hand[1].enhancement is None
        assert not any(card.selected for card in game.state.hand)

    @pytest.mark.parametrize(
        ("key", "suit"),
        [
            ("c_star", Suit.DIAMONDS),
            ("c_moon", Suit.CLUBS),
            ("c_sun", Suit.HEARTS),
            ("c_world", Suit.SPADES),
        ],
    )
    def test_suit_change(self, key: str, suit: Suit) -> None:
        game = self.use_on(key, "AH KC QD JS", [0, 1, 2])
        assert [card.suit for card in game.state.hand[:3]] == [suit] * 3

    def test_strength(self) -> None:
        game = self.use_on("c_strength", "AS 9H", [0, 1])
        assert [card.rank for card in game.state.hand] == [2, 10]

    def test_death_copies_the_rightmost_card(self) -> None:
        game = self.use_on("c_death", "2C AS", [0, 1])
        assert [card.label for card in game.state.hand] == ["AS", "AS"]
        assert len({card.uid for card in game.state.hand}) == 2

    def test_hanged_man_destroys(self) -> None:
        game = self.use_on("c_hanged_man", "2C 3D 4H", [0, 1])
        assert [card.label for card in game.state.hand] == ["4H"]
        assert len(game.state.playing_cards()) == 50

    def test_targeted_tarots_need_a_selection(self) -> None:
        game = started_round("tarot")
        give(game.state, "c_magician")
        give(game.state, "c_death")
        assert not game.is_legal(UseConsumable(0))
        game.step(ToggleCard(0))
        assert game.is_legal(UseConsumable(0))
        assert not game.is_legal(UseConsumable(1))
        for i in (1, 2):
            game.step(ToggleCard(i))
        assert not game.is_legal(UseConsumable(0))

    def test_targeted_tarots_cannot_be_used_in_the_shop(self) -> None:
        game = in_shop()
        give(game.state, "c_magician")
        assert not game.is_legal(UseConsumable(0))

    def test_hermit(self) -> None:
        game = Game.new("hermit")
        give(game.state, "c_hermit")
        game.state.money = 30
        game.step(UseConsumable(0))
        assert game.state.money == 50

    def test_temperance(self) -> None:
        game = Game.new("temperance")
        state = game.state
        give(state, "j_joker")
        give(state, "j_duo")
        give(state, "c_temperance")
        game.step(UseConsumable(0))
        assert state.money == 4 + sum(joker.sell_value for joker in state.jokers)

    @pytest.mark.parametrize(
        ("key", "set_name"), [("c_emperor", "Tarot"), ("c_high_priestess", "Planet")]
    )
    def test_create_consumables(self, key: str, set_name: str) -> None:
        game = Game.new("create")
        give(game.state, key)
        game.step(UseConsumable(0))
        assert [CENTERS[card.key].set for card in game.state.consumables] == [set_name] * 2

    def test_judgement(self) -> None:
        game = Game.new("judgement")
        give(game.state, "c_judgement")
        game.step(UseConsumable(0))
        assert len(game.state.jokers) == 1
        assert game.state.jokers[0].key in JOKERS

    def test_wheel_of_fortune(self) -> None:
        game = Game.new("wheel")
        state = game.state
        joker = give(state, "j_joker")
        give(state, "c_wheel_of_fortune")
        state.probabilities_normal = 4
        game.step(UseConsumable(0))
        assert joker.edition in ("e_foil", "e_holo", "e_polychrome")

    def test_the_fool_copies_the_last_card_used(self) -> None:
        game = Game.new("fool")
        state = game.state
        give(state, "c_fool")
        assert not game.is_legal(UseConsumable(0))
        give(state, "c_pluto")
        game.step(UseConsumable(1))
        game.step(UseConsumable(0))
        assert [card.key for card in state.consumables] == ["c_pluto"]

    def test_fortune_teller_counts_tarots(self) -> None:
        game = Game.new("teller")
        state = game.state
        give(state, "j_fortune_teller")
        give(state, "c_hermit")
        game.step(UseConsumable(0))
        assert state.consumable_usage == {"c_hermit": 1}

    def test_purple_seal_creates_a_tarot(self) -> None:
        game = started_round("purple")
        cards = set_hand(game.state, "2C 3D")
        cards[0].seal = "Purple"
        for action in macros.discard_cards(game.state, [0]):
            game.step(action)
        assert [CENTERS[card.key].set for card in game.state.consumables] == ["Tarot"]


def test_every_tarot_and_planet_is_implemented() -> None:
    expected = {key for key, center in CENTERS.items() if center.set in ("Tarot", "Planet")}
    assert expected == IMPLEMENTED
    assert len(IMPLEMENTED) == 34


def test_egg_grows_its_sell_value() -> None:
    game = Game.new("egg")
    state = game.state
    egg = give(state, "j_egg")
    before = egg.sell_value
    game.step(SelectBlind())
    assert state.blind is not None
    state.blind.chips = 1
    for action in macros.play_cards(state, [0]):
        game.step(action)
    assert egg.sell_value == before + 3


def test_gros_michel_extinction_unlocks_cavendish() -> None:
    state = in_shop().state
    scope = Scope(state)
    keys = {create_card_for_shop(scope).key for _ in range(3000)}
    assert "j_gros_michel" in keys
    assert "j_cavendish" not in keys
    state.pool_flags["gros_michel_extinct"] = True
    keys = {create_card_for_shop(scope).key for _ in range(3000)}
    assert "j_cavendish" in keys
    assert "j_gros_michel" not in keys


class TestRunOptions:
    def win_final_boss(self, config: RunConfig) -> Game:
        game = Game.new("options", config)
        state = game.state
        state.ante = 8
        state.blind_on_deck = BlindSlot.BOSS
        state.hands[HandType.HIGH_CARD].mult = 10**9
        game.step(SelectBlind())
        for action in macros.play_cards(state, [0]):
            game.step(action)
        return game

    def test_endless_off(self) -> None:
        game = self.win_final_boss(RunConfig(endless=False))
        assert game.legal_actions() == [EndRun()]

    def test_max_ante_ends_the_run(self) -> None:
        game = Game.new("max", RunConfig(max_ante=1))
        state = game.state
        state.blind_on_deck = BlindSlot.BOSS
        state.hands[HandType.HIGH_CARD].mult = 10**9
        game.step(SelectBlind())
        events: list[Event] = []
        for action in macros.play_cards(state, [0]):
            events += game.step(action)
        assert game.is_over
        ended = events[-1]
        assert isinstance(ended, RunEnded)
        assert (ended.won, ended.reason) == (False, "max_ante")
