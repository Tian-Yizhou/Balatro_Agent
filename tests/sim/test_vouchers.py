"""Vouchers: the shop's voucher slot, redeeming, each voucher's effect, and the
boss reroll (ADR 0004)."""

from __future__ import annotations

from collections import Counter

import pytest

from balatro_gym.replay.render import render_events, render_state
from balatro_gym.sim import Game, macros
from balatro_gym.sim.actions import (
    Buy,
    BuyPack,
    CashOut,
    EndShop,
    RedeemVoucher,
    RerollBoss,
    SelectBlind,
)
from balatro_gym.sim.blinds import boss_candidates
from balatro_gym.sim.content import CENTERS, pool
from balatro_gym.sim.engine import new_run
from balatro_gym.sim.events import BossRerolled, PlayingCardCreated, VoucherRedeemed
from balatro_gym.sim.items import new_item
from balatro_gym.sim.scope import Scope
from balatro_gym.sim.shop import create_card_for_shop, get_next_voucher_key, is_available
from balatro_gym.sim.state import BlindSlot, HandType, Phase, PlayingCard, RunState
from balatro_gym.sim.vouchers import BOSS_REROLL_COST, IMPLEMENTED
from scenarios import give, in_shop, play_hand


def sell_voucher(game: Game, key: str, money: int = 50) -> None:
    """Put voucher ``key`` in the shop as the only one for sale."""
    state = game.state
    state.money = max(state.money, money)
    state.shop_vouchers = [new_item(state, key)]
    state.shop_voucher = key


def redeem(game: Game, key: str) -> list[object]:
    sell_voucher(game, key)
    return list(game.step(RedeemVoucher(0)))


def beat_blind(game: Game) -> None:
    """Select the blind on deck and win it with one card."""
    state = game.state
    game.step(SelectBlind())
    assert state.blind is not None
    state.blind.chips = 1
    for action in macros.play_cards(state, [0]):
        game.step(action)


def available_vouchers(state: RunState) -> set[str]:
    return {center.key for center in pool("Voucher") if is_available(state, center)}


def test_every_voucher_is_implemented() -> None:
    assert {center.key for center in pool("Voucher")} == IMPLEMENTED
    assert len(IMPLEMENTED) == 32


class TestVoucherSlot:
    def test_a_run_starts_with_a_voucher_without_an_upgrade_requirement(self) -> None:
        for seed in range(20):
            key = new_run(seed).shop_voucher
            assert key is not None
            assert not CENTERS[key].raw.get("requires")

    def test_the_shop_offers_the_antes_voucher_until_it_is_redeemed(self) -> None:
        game = in_shop("voucher")
        state = game.state
        key = state.shop_voucher
        assert [item.key for item in state.shop_vouchers] == [key]
        assert state.shop_vouchers[0].cost == 10
        game.step(EndShop())
        assert state.shop_vouchers == []
        # Still offered after the big blind, in the same ante.
        beat_blind(game)
        game.step(CashOut())
        assert [item.key for item in state.shop_vouchers] == [key]
        state.money = 50
        events = game.step(RedeemVoucher(0))
        assert any(isinstance(event, VoucherRedeemed) and event.key == key for event in events)
        assert state.used_vouchers == [key]
        assert state.shop_vouchers == []
        assert state.shop_voucher is None

    def test_beating_the_boss_brings_a_new_voucher(self) -> None:
        game = in_shop("next-voucher")
        state = game.state
        state.shop_voucher = None
        game.step(EndShop())
        beat_blind(game)
        game.step(CashOut())
        assert state.shop_vouchers == []
        game.step(EndShop())
        beat_blind(game)
        assert state.ante == 2
        assert state.shop_voucher is not None
        game.step(CashOut())
        assert [item.key for item in state.shop_vouchers] == [state.shop_voucher]

    def test_redeeming_needs_the_money(self) -> None:
        game = in_shop("voucher-money")
        state = game.state
        sell_voucher(game, "v_blank")
        state.money = 9
        assert not game.is_legal(RedeemVoucher(0))
        assert RedeemVoucher(0) not in game.legal_actions()
        give(state, "j_credit_card")
        state.bankrupt_at = -20
        assert game.is_legal(RedeemVoucher(0))
        game.step(RedeemVoucher(0))
        assert state.money == -1

    def test_upgrades_need_their_base_voucher(self) -> None:
        state = new_run("pool")
        assert "v_overstock_plus" not in available_vouchers(state)
        assert "v_overstock_norm" in available_vouchers(state)
        state.used_vouchers.append("v_overstock_norm")
        assert "v_overstock_norm" not in available_vouchers(state)
        assert "v_overstock_plus" in available_vouchers(state)

    def test_a_voucher_for_sale_is_not_drawn_again(self) -> None:
        state = new_run("pool")
        state.shop_vouchers = [new_item(state, "v_hone")]
        give(state, "j_ring_master")
        assert "v_hone" not in available_vouchers(state)

    def test_with_every_voucher_redeemed_the_shop_offers_blank(self) -> None:
        state = new_run("pool")
        state.used_vouchers = sorted(IMPLEMENTED)
        assert get_next_voucher_key(Scope(state)) == "v_blank"

    def test_a_voucher_tag_draws_on_its_own_stream(self) -> None:
        state = new_run("tag")
        get_next_voucher_key(Scope(state), from_tag=True)
        assert "Voucher_fromtag" in state.rng.counters


class TestVoucherEffects:
    @pytest.mark.parametrize(
        ("key", "field", "after"),
        [
            ("v_hone", "edition_rate", 2),
            ("v_glow_up", "edition_rate", 4),
            ("v_tarot_merchant", "tarot_rate", 9.6),
            ("v_tarot_tycoon", "tarot_rate", 32),
            ("v_planet_merchant", "planet_rate", 9.6),
            ("v_planet_tycoon", "planet_rate", 32),
            ("v_magic_trick", "playing_card_rate", 4),
            ("v_seed_money", "interest_cap", 50),
            ("v_money_tree", "interest_cap", 100),
            ("v_crystal_ball", "consumable_slots", 3),
            ("v_antimatter", "joker_slots", 6),
            ("v_paint_brush", "hand_size", 9),
            ("v_clearance_sale", "discount_percent", 25),
            ("v_liquidation", "discount_percent", 50),
        ],
    )
    def test_a_number_of_the_run_changes(self, key: str, field: str, after: float) -> None:
        game = in_shop("effects")
        redeem(game, key)
        assert getattr(game.state, field) == after

    def test_grabber_and_wasteful_add_hands_and_discards_now_and_each_round(self) -> None:
        game = in_shop("grabber")
        state = game.state
        hands, discards = state.hands_per_round, state.discards_per_round
        redeem(game, "v_grabber")
        redeem(game, "v_wasteful")
        assert (state.hands_per_round, state.discards_per_round) == (hands + 1, discards + 1)
        game.step(EndShop())
        game.step(SelectBlind())
        assert (state.hands_left, state.discards_left) == (hands + 1, discards + 1)

    def test_hieroglyph_and_petroglyph_go_back_an_ante(self) -> None:
        game = in_shop("glyph")
        state = game.state
        state.ante = 3
        hands, discards = state.hands_per_round, state.discards_per_round
        redeem(game, "v_hieroglyph")
        assert (state.ante, state.hands_per_round) == (2, hands - 1)
        redeem(game, "v_petroglyph")
        assert (state.ante, state.discards_per_round) == (1, discards - 1)

    def test_rerolls_get_cheaper(self) -> None:
        game = in_shop("reroll")
        state = game.state
        redeem(game, "v_reroll_surplus")
        assert (state.base_reroll_cost, state.reroll_cost) == (3, 3)
        redeem(game, "v_reroll_glut")
        assert (state.base_reroll_cost, state.reroll_cost) == (1, 1)

    def test_a_discount_reprices_everything_including_sell_values(self) -> None:
        game = in_shop("discount")
        state = game.state
        joker = give(state, "j_blueprint")
        assert (joker.cost, joker.sell_value) == (10, 5)
        pack = state.shop_boosters[0]
        before = pack.cost
        redeem(game, "v_liquidation")
        assert (joker.cost, joker.sell_value) == (5, 2)
        assert pack.cost < before

    def test_overstock_fills_the_new_slot_at_once(self) -> None:
        game = in_shop("overstock")
        state = game.state
        state.money = 50
        state.shop_cards.pop(0)
        assert len(state.shop_cards) == 1
        redeem(game, "v_overstock_norm")
        assert state.shop_joker_max == 3
        assert len(state.shop_cards) == 3


class TestShopPlayingCards:
    def test_magic_trick_and_illusion_make_playing_cards(self) -> None:
        state = new_run("magic")
        state.phase = Phase.SHOP
        state.joker_rate = state.tarot_rate = state.planet_rate = 0
        state.playing_card_rate = 4
        scope = Scope(state)
        plain = [create_card_for_shop(scope) for _ in range(200)]
        assert all(item.playing_card is not None for item in plain)
        assert {item.key for item in plain} == {"c_base"}
        assert all(item.playing_card and item.playing_card.edition is None for item in plain)
        state.used_vouchers = ["v_magic_trick", "v_illusion"]
        illusion = [create_card_for_shop(scope) for _ in range(400)]
        keys = Counter(item.key for item in illusion)
        assert keys["c_base"] > 0
        assert len(keys) > 2
        editions = {item.playing_card.edition for item in illusion if item.playing_card}
        assert editions == {None, "e_foil", "e_holo", "e_polychrome"}

    def test_a_bought_playing_card_goes_under_the_deck(self) -> None:
        game = in_shop("buy-card")
        state = game.state
        redeem(game, "v_magic_trick")
        state.shop_cards = []
        state.joker_rate = state.tarot_rate = state.planet_rate = 0
        state.shop_cards.append(create_card_for_shop(Scope(state)))
        offer = state.shop_cards[0]
        card = offer.playing_card
        assert card is not None
        assert offer.uid == card.uid
        size = len(state.playing_cards())
        money = state.money
        events = game.step(Buy(0))
        assert state.deck[0] is card
        assert len(state.playing_cards()) == size + 1
        assert state.money == money - offer.cost
        assert PlayingCardCreated(card.uid, card.label, "deck") in events
        assert render_events(events, state)[0].startswith(f"Bought {card.label}")

    def test_playing_cards_never_need_a_free_slot(self) -> None:
        game = in_shop("full")
        state = game.state
        state.joker_rate = state.tarot_rate = state.planet_rate = 0
        state.playing_card_rate = 4
        state.shop_cards = [create_card_for_shop(Scope(state))]
        state.jokers = [new_item(state, "j_joker") for _ in range(state.joker_slots)]
        state.money = 10
        assert game.is_legal(Buy(0))


class TestPackVouchers:
    def test_telescope_puts_the_most_played_hands_planet_first(self) -> None:
        game = in_shop("telescope")
        state = game.state
        redeem(game, "v_telescope")
        state.hands[HandType.FLUSH].played = 3
        state.hands[HandType.PAIR].played = 3
        state.hands[HandType.HIGH_CARD].played = 9
        state.hands[HandType.HIGH_CARD].visible = False
        state.shop_boosters = [new_item(state, "p_celestial_normal_1")]
        game.step(BuyPack(0))
        assert state.pack is not None
        # Flush comes before Pair in the game's hand order.
        assert state.pack.items[0].key == "c_jupiter"

    def test_omen_globe_mixes_spectral_cards_into_arcana_packs(self) -> None:
        sets: Counter[str] = Counter()
        for seed in range(60):
            game = in_shop(f"omen{seed}")
            state = game.state
            state.used_vouchers.append("v_omen_globe")
            state.money = 50
            state.shop_boosters = [new_item(state, "p_arcana_mega_1")]
            game.step(BuyPack(0))
            assert state.pack is not None
            sets.update(CENTERS[item.key].set for item in state.pack.items)
        assert sets["Spectral"] > 0
        assert 0.1 < sets["Spectral"] / sets.total() < 0.3


@pytest.mark.parametrize(("vouchers", "mult"), [((), 1), (("v_observatory",), 1.5)])
def test_observatory_lets_a_held_planet_multiply_its_hands_mult(
    vouchers: tuple[str, ...], mult: float
) -> None:
    def setup(state: RunState, cards: list[PlayingCard]) -> None:
        state.used_vouchers.extend(vouchers)
        # A single card is a High Card: Pluto applies, Mercury (Pair) does not.
        state.consumables = [new_item(state, "c_mercury"), new_item(state, "c_pluto")]

    _, _, scored = play_hand("AS", setup=setup)
    assert scored.mult == mult


class TestBossReroll:
    def at_blind_select(self, *vouchers: str) -> Game:
        game = Game.new("reroll-boss")
        game.state.used_vouchers.extend(vouchers)
        game.state.money = 30
        return game

    def test_the_reroll_needs_a_voucher(self) -> None:
        game = self.at_blind_select()
        assert not game.is_legal(RerollBoss())
        assert RerollBoss() not in game.legal_actions()

    def test_directors_cut_rerolls_once_per_ante(self) -> None:
        game = self.at_blind_select("v_directors_cut")
        state = game.state
        assert RerollBoss() in game.legal_actions()
        candidates = boss_candidates(state.ante, state.win_ante, state.bosses_used)
        events = game.step(RerollBoss())
        assert state.money == 30 - BOSS_REROLL_COST
        assert state.blind_choices[BlindSlot.BOSS] in candidates
        assert any(isinstance(event, BossRerolled) for event in events)
        assert not game.is_legal(RerollBoss())
        # Beating the boss starts a new ante, with a new reroll.
        while state.ante == 1:
            beat_blind(game)
            game.step(CashOut())
            game.step(EndShop())
        state.money = 30
        assert game.is_legal(RerollBoss())

    def test_retcon_rerolls_any_number_of_times(self) -> None:
        game = self.at_blind_select("v_directors_cut", "v_retcon")
        for _ in range(3):
            game.step(RerollBoss())
        assert game.state.money == 0
        assert not game.is_legal(RerollBoss())

    def test_the_reroll_is_only_at_blind_selection(self) -> None:
        game = self.at_blind_select("v_retcon")
        game.step(SelectBlind())
        assert not game.is_legal(RerollBoss())


def test_the_shop_view_shows_vouchers() -> None:
    game = in_shop("render")
    state = game.state
    sell_voucher(game, "v_hone")
    state.used_vouchers.append("v_blank")
    text = render_state(state)
    assert "Voucher v_hone $10" in text
    assert "Owned   v_blank" in text
