"""Boss blind effects over a round, Chicot, and hidden information."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from balatro_gym.replay.render import render_state
from balatro_gym.sim import Game, macros
from balatro_gym.sim.actions import (
    CashOut,
    Discard,
    Play,
    SelectBlind,
    SellJoker,
    ToggleCard,
    UseConsumable,
)
from balatro_gym.sim.bosses import most_played_hand
from balatro_gym.sim.engine import determinize
from balatro_gym.sim.events import BlindDisabled, BlindTriggered, HandScored
from balatro_gym.sim.items import new_item
from balatro_gym.sim.state import BlindSlot, BlindStatus, HandType, Phase, RunState, Suit
from scenarios import give, set_hand


def boss_game(
    key: str, seed: str = "boss", setup: Callable[[RunState], None] | None = None
) -> Game:
    """A game facing boss ``key``, with the blind selected and the hand drawn."""
    game = Game.new(seed)
    state = game.state
    state.blind_states[BlindSlot.SMALL] = BlindStatus.DEFEATED
    state.blind_states[BlindSlot.BIG] = BlindStatus.DEFEATED
    state.blind_states[BlindSlot.BOSS] = BlindStatus.SELECT
    state.blind_on_deck = BlindSlot.BOSS
    state.blind_choices[BlindSlot.BOSS] = key
    if setup is not None:
        setup(state)
    game.step(SelectBlind())
    return game


def play(game: Game, labels: str) -> list[object]:
    """Put the cards ``labels`` first in hand and play them."""
    state = game.state
    cards = labels.split()
    rest = [card.label for card in state.hand if card.label not in cards]
    set_hand(state, " ".join([*cards, *rest][: len(state.hand)]))
    events: list[object] = []
    for action in macros.play_cards(state, range(len(cards))):
        events += game.step(action)
    return events


def scored(events: list[object]) -> HandScored:
    return next(event for event in events if isinstance(event, HandScored))


def win_round(game: Game) -> None:
    state = game.state
    assert state.blind is not None
    state.blind.chips = 0
    for action in macros.play_cards(state, [0]):
        game.step(action)


class TestDebuffs:
    @pytest.mark.parametrize(
        ("key", "suit"),
        [
            ("bl_club", Suit.CLUBS),
            ("bl_goad", Suit.SPADES),
            ("bl_head", Suit.HEARTS),
            ("bl_window", Suit.DIAMONDS),
        ],
    )
    def test_a_suit_is_debuffed_until_the_boss_is_beaten(self, key: str, suit: Suit) -> None:
        game = boss_game(key)
        state = game.state
        cards = state.playing_cards()
        assert all(card.debuffed == (card.suit is suit) for card in cards)
        win_round(game)
        assert not any(card.debuffed for card in state.playing_cards())

    def test_a_wild_card_counts_as_every_suit(self) -> None:
        def setup(state: RunState) -> None:
            for card in state.deck:
                if card.suit is Suit.HEARTS:
                    card.enhancement = "m_wild"

        state = boss_game("bl_club", setup=setup).state
        assert all(card.debuffed for card in state.playing_cards() if card.suit is Suit.HEARTS)

    def test_the_plant_debuffs_face_cards_and_pareidolia_makes_all_faces(self) -> None:
        state = boss_game("bl_plant").state
        assert all(card.debuffed == (11 <= card.rank <= 13) for card in state.playing_cards())
        state = boss_game("bl_plant", setup=lambda s: give(s, "j_pareidolia") and None).state
        assert all(card.debuffed for card in state.playing_cards())

    def test_the_pillar_debuffs_cards_played_this_ante(self) -> None:
        def setup(state: RunState) -> None:
            state.deck[0].played_this_ante = True

        state = boss_game("bl_pillar", setup=setup).state
        assert [card.played_this_ante for card in state.playing_cards()] == [
            card.debuffed for card in state.playing_cards()
        ]

    def test_selling_a_joker_disables_verdant_leaf(self) -> None:
        game = boss_game("bl_final_leaf", setup=lambda s: give(s, "j_joker") and None)
        state = game.state
        assert all(card.debuffed for card in state.playing_cards())
        events = game.step(SellJoker(0))
        assert BlindDisabled("bl_final_leaf") in events
        assert not any(card.debuffed for card in state.playing_cards())


class TestPlayedHands:
    def test_the_psychic_needs_five_cards(self) -> None:
        game = boss_game("bl_psychic")
        events = play(game, "AS AH")
        assert BlindTriggered("bl_psychic") in events
        assert scored(events).score == 0
        assert scored(play(game, "2S 3S 4S 5S 7H")).score > 0

    def test_the_eye_allows_each_hand_once(self) -> None:
        game = boss_game("bl_eye")
        assert scored(play(game, "AS")).score > 0
        assert scored(play(game, "KS")).score == 0
        assert scored(play(game, "QS QH")).score > 0

    def test_the_mouth_allows_only_the_first_hand(self) -> None:
        game = boss_game("bl_mouth")
        assert scored(play(game, "AS AH")).score > 0
        assert scored(play(game, "KS")).score == 0
        assert scored(play(game, "QS QH")).score > 0

    def test_the_arm_lowers_the_hand_level(self) -> None:
        def setup(state: RunState) -> None:
            state.hands[HandType.HIGH_CARD].level = 2

        game = boss_game("bl_arm", setup=setup)
        events = play(game, "AS")
        assert game.state.hands[HandType.HIGH_CARD].level == 1
        assert scored(events).chips == 5 + 11

    def test_the_ox_takes_all_money_for_the_most_played_hand(self) -> None:
        def setup(state: RunState) -> None:
            state.most_played_hand = HandType.PAIR
            state.money = 30

        game = boss_game("bl_ox", setup=setup)
        play(game, "AS")
        assert game.state.money == 30
        play(game, "KS KH")
        assert game.state.money == 0

    def test_the_flint_halves_base_chips_and_mult(self) -> None:
        events = play(boss_game("bl_flint"), "AS")
        assert (scored(events).chips, scored(events).mult) == (3 + 11, 1)

    def test_the_hook_discards_two_cards_without_using_a_discard(self) -> None:
        game = boss_game("bl_hook")
        state = game.state
        discards, size = state.discards_left, len(state.hand)
        events = play(game, "AS")
        assert BlindTriggered("bl_hook") in events
        assert state.discards_left == discards
        assert len(state.discard) == 3
        assert len(state.hand) == size

    def test_the_tooth_costs_a_dollar_per_card_played(self) -> None:
        game = boss_game("bl_tooth", setup=lambda s: setattr(s, "money", 10))
        play(game, "AS AH KS")
        assert game.state.money == 7


class TestRoundLimits:
    def test_the_water_takes_the_discards(self) -> None:
        assert boss_game("bl_water").state.discards_left == 0

    def test_the_needle_leaves_one_hand(self) -> None:
        assert boss_game("bl_needle").state.hands_left == 1

    def test_the_manacle_takes_one_hand_size_until_beaten(self) -> None:
        game = boss_game("bl_manacle")
        state = game.state
        assert (state.hand_size, len(state.hand)) == (7, 7)
        win_round(game)
        assert state.hand_size == 8

    def test_the_serpent_always_draws_three(self) -> None:
        game = boss_game("bl_serpent")
        state = game.state
        for action in macros.discard_cards(state, [0]):
            game.step(action)
        assert len(state.hand) == 8 - 1 + 3

    @pytest.mark.parametrize(("key", "factor"), [("bl_wall", 4), ("bl_final_vessel", 6)])
    def test_big_bosses_raise_the_target(self, key: str, factor: int) -> None:
        state = boss_game(key).state
        assert state.blind is not None
        assert state.blind.chips == 300 * factor


class TestFaceDown:
    def test_the_house_deals_the_first_hand_face_down(self) -> None:
        game = boss_game("bl_house")
        state = game.state
        assert all(card.face_down for card in state.hand)
        game.step(ToggleCard(0))
        game.step(Discard())
        assert sum(card.face_down for card in state.hand) == 7

    def test_the_mark_deals_face_cards_face_down(self) -> None:
        state = boss_game("bl_mark").state
        assert all(card.face_down == (11 <= card.rank <= 13) for card in state.hand)

    def test_the_wheel_turns_cards_down_one_in_seven(self) -> None:
        state = boss_game("bl_wheel", setup=lambda s: setattr(s, "probabilities_normal", 7)).state
        assert all(card.face_down for card in state.hand)

    def test_the_fish_deals_cards_face_down_after_a_hand(self) -> None:
        game = boss_game("bl_fish")
        state = game.state
        assert not any(card.face_down for card in state.hand)
        play(game, "AS")
        assert sum(card.face_down for card in state.hand) == 1

    def test_played_cards_are_revealed(self) -> None:
        game = boss_game("bl_house")
        play(game, "AS")
        assert not any(card.face_down for card in game.state.discard)

    def test_amber_acorn_turns_and_shuffles_the_jokers(self) -> None:
        def setup(state: RunState) -> None:
            for key in ("j_joker", "j_greedy_joker", "j_lusty_joker", "j_wrathful_joker"):
                give(state, key)

        game = boss_game("bl_final_acorn", setup=setup)
        state = game.state
        assert all(joker.face_down for joker in state.jokers)
        assert len({joker.key for joker in state.jokers}) == 4
        assert "~" in render_state(state)
        win_round(game)
        assert not any(joker.face_down for joker in state.jokers)

    def test_determinize_deals_face_down_cards_again(self) -> None:
        state = boss_game("bl_house").state
        hidden = [card.uid for card in state.hand]
        copies = [determinize(state, seed) for seed in range(5)]
        assert any([card.uid for card in copy.hand] != hidden for copy in copies)
        for copy in copies:
            assert all(card.face_down for card in copy.hand)
            assert {card.uid for card in copy.playing_cards()} == {
                card.uid for card in state.playing_cards()
            }

    def test_determinize_keeps_cards_face_up_in_place(self) -> None:
        state = boss_game("bl_club").state
        assert [card.uid for card in determinize(state, 1).hand] == [
            card.uid for card in state.hand
        ]

    def test_determinize_shuffles_face_down_jokers(self) -> None:
        def setup(state: RunState) -> None:
            for _ in range(6):
                give(state, "j_joker")

        state = boss_game("bl_final_acorn", setup=setup).state
        orders = {
            tuple(joker.uid for joker in determinize(state, seed).jokers) for seed in range(5)
        }
        assert len(orders) > 1


class TestFinalBosses:
    def test_cerulean_bell_forces_a_card_to_stay_selected(self) -> None:
        game = boss_game("bl_final_bell")
        state = game.state
        forced = [i for i, card in enumerate(state.hand) if card.forced_selection]
        assert len(forced) == 1
        assert state.hand[forced[0]].selected
        assert not game.is_legal(ToggleCard(forced[0]))
        game.step(Play())
        assert sum(card.forced_selection for card in state.hand) == 1

    def test_crimson_heart_debuffs_one_joker_each_hand(self) -> None:
        def setup(state: RunState) -> None:
            for key in ("j_joker", "j_greedy_joker", "j_juggler"):
                give(state, key)

        game = boss_game("bl_final_heart", setup=setup)
        state = game.state
        first = [joker.uid for joker in state.jokers if joker.debuffed]
        assert len(first) == 1
        play(game, "AS")
        second = [joker.uid for joker in state.jokers if joker.debuffed]
        assert len(second) == 1
        assert second != first

    def test_a_debuffed_joker_loses_its_lasting_effect(self) -> None:
        def setup(state: RunState) -> None:
            give(state, "j_juggler")
            state.hand_size += 1  # what the Juggler gives while it is owned

        game = boss_game("bl_final_heart", setup=setup)
        state = game.state
        assert state.jokers[0].debuffed
        assert state.hand_size == 8


class TestChicot:
    def test_chicot_disables_each_boss_as_it_is_set(self) -> None:
        game = boss_game("bl_water", setup=lambda s: give(s, "j_chicot") and None)
        state = game.state
        assert state.blind is not None
        assert state.blind.disabled
        assert state.discards_left > 0

    def test_chicot_lowers_the_wall(self) -> None:
        state = boss_game("bl_wall", setup=lambda s: give(s, "j_chicot") and None).state
        assert state.blind is not None
        assert state.blind.chips == 300 * 2

    def test_chicot_brings_back_the_manacles_hand_size(self) -> None:
        state = boss_game("bl_manacle", setup=lambda s: give(s, "j_chicot") and None).state
        assert (state.hand_size, len(state.hand)) == (8, 8)

    def test_chicot_ignores_small_and_big_blinds(self) -> None:
        game = Game.new("small")
        give(game.state, "j_chicot")
        game.step(SelectBlind())
        assert game.state.blind is not None
        assert not game.state.blind.disabled

    def test_chicot_gained_mid_round_can_end_it(self) -> None:
        # Find a seed where The Soul creates Chicot.
        for seed in range(200):
            game = boss_game("bl_wall", seed=f"soul{seed}")
            state = game.state
            assert state.blind is not None
            state.chips = state.blind.chips * 0.6
            state.consumables = [new_item(state, "c_soul")]
            events = game.step(UseConsumable(0))
            if any(joker.key == "j_chicot" for joker in state.jokers):
                break
        else:
            pytest.fail("no seed gave Chicot")
        assert BlindDisabled("bl_wall") in events
        assert state.phase is Phase.ROUND_EVAL
        game.step(CashOut())


def test_a_worn_out_perishable_joker_loses_its_lasting_effect() -> None:
    game = Game.new("perishable")
    state = game.state
    juggler = give(state, "j_juggler")
    state.hand_size += 1
    juggler.perishable = True
    juggler.perish_tally = 1
    game.step(SelectBlind())
    win_round(game)
    assert juggler.debuffed
    assert state.hand_size == 8


def test_the_most_played_hand_breaks_ties_by_the_games_hand_order() -> None:
    state = Game.new("most").state
    state.hands[HandType.PAIR].played = 3
    state.hands[HandType.FLUSH].played = 3
    assert most_played_hand(state) is HandType.FLUSH
