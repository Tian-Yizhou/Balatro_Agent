"""Booster packs, the pack phase and Spectral cards."""

from __future__ import annotations

import math
from collections import Counter

import pytest

from balatro_gym.sim import Game, macros
from balatro_gym.sim.actions import (
    BuyPack,
    EndShop,
    Pick,
    Reroll,
    SkipPack,
    ToggleCard,
    UseConsumable,
)
from balatro_gym.sim.content import CENTERS, pool
from balatro_gym.sim.engine import new_run
from balatro_gym.sim.events import (
    CardDestroyed,
    Event,
    PackOpened,
    PackPicked,
    PackSkipped,
    PlayingCardCreated,
)
from balatro_gym.sim.items import new_item
from balatro_gym.sim.packs import _standard_card, get_pack
from balatro_gym.sim.scope import Scope
from balatro_gym.sim.shop import create_card
from balatro_gym.sim.state import HandType, Phase
from scenarios import give, in_shop, set_hand, started_round


def open_pack(game: Game, key: str, *contents: str) -> list[Event]:
    """Buy and open pack ``key``; ``contents`` replace the items it rolled."""
    state = game.state
    state.money = max(state.money, 50)
    state.shop_boosters = [new_item(state, key)]
    events = game.step(BuyPack(0))
    assert state.pack is not None
    if contents:
        state.pack.items = [new_item(state, item) for item in contents]
    return events


def within_odds(count: int, total: int, probability: float) -> bool:
    """Whether ``count`` hits in ``total`` draws fit ``probability`` (5 standard errors)."""
    expected = total * probability
    return abs(count - expected) <= 5 * math.sqrt(expected * (1 - probability)) + 1


class TestShopPacks:
    def test_the_first_shop_offers_a_buffoon_pack(self) -> None:
        state = in_shop("first").state
        keys = [pack.key for pack in state.shop_boosters]
        assert len(keys) == 2
        assert keys[0] in ("p_buffoon_normal_1", "p_buffoon_normal_2")
        assert state.first_shop_buffoon

    def test_rerolling_keeps_the_packs(self) -> None:
        game = in_shop("reroll")
        state = game.state
        state.money = 50
        before = [pack.uid for pack in state.shop_boosters]
        game.step(Reroll())
        assert [pack.uid for pack in state.shop_boosters] == before

    def test_leaving_the_shop_removes_the_packs(self) -> None:
        game = in_shop("leave")
        game.step(EndShop())
        assert game.state.shop_boosters == []

    def test_buying_a_pack_needs_the_money(self) -> None:
        game = in_shop("money")
        state = game.state
        state.shop_boosters = [new_item(state, "p_arcana_normal_1")]
        state.money = 3
        assert not game.is_legal(BuyPack(0))
        give(state, "j_credit_card")
        state.bankrupt_at = -20
        assert game.is_legal(BuyPack(0))

    def test_pack_weights_match_the_game(self) -> None:
        state = new_run("weights")
        state.first_shop_buffoon = True
        scope = Scope(state)
        draws = 40_000
        counts = Counter(get_pack(scope, "shop_pack") for _ in range(draws))
        weights = {center.key: float(center.raw.get("weight", 1)) for center in pool("Booster")}
        total = sum(weights.values())
        for key, weight in weights.items():
            assert within_odds(counts[key], draws, weight / total), key


class TestOpening:
    @pytest.mark.parametrize(
        ("key", "set_name"),
        [
            ("p_arcana_normal_1", "Tarot"),
            ("p_celestial_jumbo_1", "Planet"),
            ("p_spectral_mega_1", "Spectral"),
            ("p_buffoon_normal_1", "Joker"),
        ],
    )
    def test_a_pack_holds_cards_of_its_kind(self, key: str, set_name: str) -> None:
        game = in_shop("contents")
        state = game.state
        state.money = 20
        state.shop_boosters = [new_item(state, key)]
        cost = state.shop_boosters[0].cost
        events = game.step(BuyPack(0))
        pack = state.pack
        assert pack is not None
        assert state.phase is Phase.PACK_OPEN
        assert state.money == 20 - cost
        config = CENTERS[key].config
        assert len(pack.items) == config["extra"]
        assert pack.choices_left == config["choose"]
        rare = ("c_soul", "c_black_hole")
        assert all(CENTERS[item.key].set == set_name or item.key in rare for item in pack.items)
        opened = next(event for event in events if isinstance(event, PackOpened))
        assert opened.cards == tuple(item.uid for item in pack.items)

    def test_arcana_packs_draw_a_hand_that_goes_back_under_the_deck(self) -> None:
        game = in_shop("hand")
        state = game.state
        top = {card.uid for card in state.deck[-state.hand_size :]}
        open_pack(game, "p_arcana_normal_1")
        assert {card.uid for card in state.hand} == top
        hand = list(state.hand)
        game.step(ToggleCard(0))
        game.step(SkipPack())
        assert state.phase is Phase.SHOP
        assert state.hand == []
        assert state.deck[: len(hand)] == list(reversed(hand))
        assert not any(card.selected for card in state.deck)
        assert len(state.deck) == 52

    @pytest.mark.parametrize("key", ["p_celestial_normal_1", "p_buffoon_normal_1"])
    def test_other_packs_draw_no_hand(self, key: str) -> None:
        game = in_shop("no-hand")
        open_pack(game, key)
        assert game.state.hand == []
        assert not game.is_legal(ToggleCard(0))

    def test_standard_pack_cards_follow_the_game_odds(self) -> None:
        state = new_run("standard")
        scope = Scope(state)
        draws = 20_000
        cards = [_standard_card(scope) for _ in range(draws)]
        assert within_odds(sum(card.enhancement is not None for card in cards), draws, 0.4)
        editions = Counter(card.edition for card in cards)
        assert within_odds(editions["e_foil"], draws, 0.04)
        assert within_odds(editions["e_holo"], draws, 0.028)
        assert within_odds(editions["e_polychrome"], draws, 0.012)
        assert editions["e_negative"] == 0
        seals = Counter(card.seal for card in cards)
        for seal in ("Red", "Blue", "Gold", "Purple"):
            assert within_odds(seals[seal], draws, 0.2 * 0.25), seal

    def test_the_soul_and_black_hole_rarely_replace_pack_cards(self) -> None:
        state = new_run("soul")
        scope = Scope(state)
        draws = 60_000
        keys = Counter(
            create_card(scope, "Spectral", "pack", key_append="spe", soulable=True).key
            for _ in range(draws)
        )
        assert within_odds(keys["c_soul"], draws, 0.003 * 0.997)
        assert within_odds(keys["c_black_hole"], draws, 0.003)
        plain = Counter(create_card(scope, "Spectral", "shop").key for _ in range(5000))
        assert plain["c_soul"] == plain["c_black_hole"] == 0

    def test_an_owned_soul_is_not_offered_again_without_showman(self) -> None:
        state = new_run("owned")
        scope = Scope(state)
        give(state, "c_soul")
        keys = {
            create_card(scope, "Tarot", "pack", key_append="ar1", soulable=True).key
            for _ in range(5000)
        }
        assert "c_soul" not in keys
        give(state, "j_ring_master")
        keys = {
            create_card(scope, "Tarot", "pack", key_append="ar1", soulable=True).key
            for _ in range(5000)
        }
        assert "c_soul" in keys


class TestPicking:
    def test_a_planet_levels_up_its_hand(self) -> None:
        game = in_shop("planet")
        open_pack(game, "p_celestial_normal_1", "c_pluto", "c_mars", "c_venus")
        events = game.step(Pick(0))
        assert game.state.hands[HandType.HIGH_CARD].level == 2
        assert PackPicked(game.state.next_uid - 3, "c_pluto") in events
        assert game.state.phase is Phase.SHOP
        assert game.state.pack is None

    def test_a_tarot_uses_the_selected_cards(self) -> None:
        game = in_shop("tarot")
        state = game.state
        open_pack(game, "p_arcana_normal_1", "c_empress", "c_fool", "c_hermit")
        assert not game.is_legal(Pick(0))
        targets = [state.hand[0], state.hand[1]]
        for action in macros.select_cards(state, [0, 1]):
            game.step(action)
        game.step(Pick(0))
        assert [card.enhancement for card in targets] == ["m_mult", "m_mult"]
        assert all(card in state.deck for card in targets)

    def test_a_joker_needs_a_free_slot_unless_negative(self) -> None:
        game = in_shop("joker")
        state = game.state
        for _ in range(5):
            give(state, "j_joker")
        open_pack(game, "p_buffoon_normal_1", "j_jolly", "j_zany")
        assert state.pack is not None
        state.pack.items[1].edition = "e_negative"
        assert not game.is_legal(Pick(0))
        game.step(Pick(1))
        assert [joker.key for joker in state.jokers][-1] == "j_zany"
        assert len(state.jokers) == 6

    def test_a_standard_card_goes_under_the_deck(self) -> None:
        game = in_shop("standard")
        state = game.state
        open_pack(game, "p_standard_normal_1")
        assert state.pack is not None
        card = state.pack.playing_cards[2]
        events = game.step(Pick(2))
        assert state.deck[0] is card
        assert len(state.playing_cards()) == 53
        assert PlayingCardCreated(card.uid, card.label, "deck") in events

    def test_mega_packs_allow_two_picks(self) -> None:
        game = in_shop("mega")
        open_pack(game, "p_celestial_mega_1")
        game.step(Pick(0))
        assert game.state.phase is Phase.PACK_OPEN
        assert game.state.pack is not None
        assert game.state.pack.choices_left == 1
        game.step(Pick(0))
        assert game.state.phase is Phase.SHOP

    def test_skipping_closes_the_pack(self) -> None:
        game = in_shop("skip")
        open_pack(game, "p_buffoon_normal_1")
        assert PackSkipped() in game.step(SkipPack())
        assert game.state.phase is Phase.SHOP
        assert game.state.pack is None

    def test_consumables_can_be_used_from_their_slots_during_a_pack(self) -> None:
        game = in_shop("slots")
        state = game.state
        give(state, "c_strength")
        open_pack(game, "p_spectral_normal_1")
        game.step(ToggleCard(0))
        card = state.hand[0]
        rank = card.rank
        game.step(UseConsumable(0))
        assert card.rank == (2 if rank == 14 else rank + 1)
        assert state.phase is Phase.PACK_OPEN


def use_spectral(
    key: str, hand: str = "2C 5D 8H JS KC AD 3S 9H", selected: tuple[int, ...] = (), **setup: int
) -> tuple[Game, list[Event]]:
    """Use Spectral card ``key`` while selecting a hand, on the cards at ``selected``."""
    game = started_round(f"spectral-{key}")
    state = game.state
    set_hand(state, hand)
    state.money = setup.get("money", state.money)
    item = give(state, key)
    for action in macros.select_cards(state, selected):
        game.step(action)
    events = game.step(UseConsumable(state.consumables.index(item)))
    return game, events


class TestSpectralCards:
    @pytest.mark.parametrize(
        ("key", "count", "ranks"),
        [
            ("c_familiar", 3, {11, 12, 13}),
            ("c_grim", 2, {14}),
            ("c_incantation", 4, set(range(2, 11))),
        ],
    )
    def test_card_creators(self, key: str, count: int, ranks: set[int]) -> None:
        game, events = use_spectral(key)
        state = game.state
        destroyed = [event for event in events if isinstance(event, CardDestroyed)]
        created = [event.uid for event in events if isinstance(event, PlayingCardCreated)]
        assert len(destroyed) == 1
        assert len(created) == count
        new = [card for card in state.hand if card.uid in created]
        assert all(card.rank in ranks for card in new)
        assert all(card.enhancement not in (None, "m_stone") for card in new)
        assert len(state.hand) == 8 - 1 + count
        assert len(state.playing_cards()) == 52 - 1 + count

    @pytest.mark.parametrize(
        ("key", "seal"),
        [
            ("c_talisman", "Gold"),
            ("c_deja_vu", "Red"),
            ("c_trance", "Blue"),
            ("c_medium", "Purple"),
        ],
    )
    def test_seals(self, key: str, seal: str) -> None:
        game, _ = use_spectral(key, selected=(2,))
        assert [card.seal for card in game.state.hand] == [None, None, seal, *[None] * 5]

    def test_aura_adds_an_edition(self) -> None:
        game, _ = use_spectral("c_aura", selected=(0,))
        assert game.state.hand[0].edition in ("e_foil", "e_holo", "e_polychrome")

    def test_aura_needs_one_card_without_an_edition(self) -> None:
        game = started_round("aura")
        state = game.state
        give(state, "c_aura")
        game.step(ToggleCard(0))
        game.step(ToggleCard(1))
        assert not game.is_legal(UseConsumable(0))
        game.step(ToggleCard(1))
        state.hand[0].edition = "e_foil"
        assert not game.is_legal(UseConsumable(0))

    def test_wraith_creates_a_rare_joker_and_takes_all_money(self) -> None:
        game, _ = use_spectral("c_wraith", money=30)
        state = game.state
        assert state.money == 0
        assert [CENTERS[joker.key].rarity for joker in state.jokers] == [3]

    def test_the_soul_creates_a_legendary_joker(self) -> None:
        game, _ = use_spectral("c_soul")
        assert [CENTERS[joker.key].rarity for joker in game.state.jokers] == [4]

    def test_sigil_and_ouija(self) -> None:
        game, _ = use_spectral("c_sigil")
        assert len({card.suit for card in game.state.hand}) == 1
        game, _ = use_spectral("c_ouija")
        assert len({card.rank for card in game.state.hand}) == 1
        assert game.state.hand_size == 7

    def test_ectoplasm_shrinks_the_hand_more_each_time(self) -> None:
        game = started_round("ectoplasm")
        state = game.state
        for key in ("j_joker", "j_jolly"):
            give(state, key)
        give(state, "c_ectoplasm")
        game.step(UseConsumable(0))
        assert state.hand_size == 7
        give(state, "c_ectoplasm")
        game.step(UseConsumable(0))
        assert state.hand_size == 5
        assert [joker.edition for joker in state.jokers] == ["e_negative", "e_negative"]
        give(state, "c_ectoplasm")
        assert not game.is_legal(UseConsumable(0))

    def test_immolate(self) -> None:
        game, events = use_spectral("c_immolate", money=4)
        state = game.state
        assert sum(isinstance(event, CardDestroyed) for event in events) == 5
        assert len(state.hand) == 3
        assert state.money == 24

    def test_ankh_copies_a_joker_and_keeps_eternal_ones(self) -> None:
        game = started_round("ankh")
        state = game.state
        for key in ("j_joker", "j_jolly", "j_zany"):
            give(state, key)
        state.jokers[2].eternal = True
        give(state, "c_ankh")
        game.step(UseConsumable(0))
        keys = Counter(joker.key for joker in state.jokers)
        assert keys["j_zany"] >= 1
        assert len(state.jokers) in (2, 3)
        assert max(keys.values()) == 2

    def test_ankh_needs_a_free_joker_slot(self) -> None:
        game = started_round("ankh-full")
        state = game.state
        for _ in range(5):
            give(state, "j_joker")
        give(state, "c_ankh")
        assert not game.is_legal(UseConsumable(0))

    def test_hex_makes_one_joker_polychrome_and_destroys_the_rest(self) -> None:
        game = started_round("hex")
        state = game.state
        for key in ("j_joker", "j_jolly", "j_zany"):
            give(state, key)
        state.jokers[0].eternal = True
        give(state, "c_hex")
        game.step(UseConsumable(0))
        assert [joker.edition for joker in state.jokers].count("e_polychrome") == 1
        assert state.jokers[0].key == "j_joker"
        assert len(state.jokers) in (1, 2)

    def test_cryptid_copies_the_selected_card(self) -> None:
        game, events = use_spectral("c_cryptid", selected=(3,))
        state = game.state
        original = state.hand[3]
        copies = state.hand[-2:]
        assert all((card.rank, card.suit) == (original.rank, original.suit) for card in copies)
        assert len({card.uid for card in state.hand}) == 10
        assert sum(isinstance(event, PlayingCardCreated) for event in events) == 2

    def test_black_hole_levels_up_every_hand(self) -> None:
        game, _ = use_spectral("c_black_hole")
        assert {level.level for level in game.state.hands.values()} == {2}
