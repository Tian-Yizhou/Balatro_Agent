from __future__ import annotations

import pytest

from balatro_gym.sim import Game, IllegalActionError, macros, snapshot
from balatro_gym.sim.actions import (
    CashOut,
    ContinueEndless,
    Discard,
    EndRun,
    EndShop,
    MoveCard,
    MoveJoker,
    Play,
    SelectBlind,
    SortBySuit,
    ToggleCard,
)
from balatro_gym.sim.blinds import BLINDS
from balatro_gym.sim.engine import legal_actions, new_run
from balatro_gym.sim.events import Event, HandScored, RunEnded
from balatro_gym.sim.state import (
    BlindSlot,
    BlindStatus,
    HandSort,
    HandType,
    Item,
    Phase,
    RunConfig,
)
from scenarios import set_hand, started_round


def kinds(events: list[Event]) -> list[str]:
    return [event.kind for event in events]


def play(game: Game, labels: str) -> list[Event]:
    """Put ``labels`` in hand and play all of them."""
    set_hand(game.state, labels)
    events: list[Event] = []
    for action in macros.play_cards(game.state, range(len(game.state.hand))):
        events += game.step(action)
    return events


def beat_blind(game: Game) -> None:
    """Win the current blind with one hand and cash out."""
    game.state.hands[HandType.HIGH_CARD].mult = 10**9
    play(game, "AS")
    if game.state.phase is Phase.WIN:
        game.step(ContinueEndless())
    game.step(CashOut())


class TestNewRun:
    def test_starting_values(self) -> None:
        state = new_run("seed")
        assert state.phase is Phase.BLIND_SELECT
        assert (state.ante, state.round, state.money) == (1, 0, 4)
        assert (state.hands_per_round, state.discards_per_round, state.hand_size) == (4, 4, 8)
        assert len(state.deck) == 52
        assert len({card.label for card in state.deck}) == 52
        assert state.blind_on_deck is BlindSlot.SMALL
        assert state.blind_states[BlindSlot.SMALL] is BlindStatus.SELECT

    def test_first_boss_is_eligible_in_ante_one(self) -> None:
        for seed in range(50):
            boss = BLINDS[new_run(seed).blind_choices[BlindSlot.BOSS]]
            assert boss.boss
            assert boss.boss_min == 1
            assert not boss.showdown

    @pytest.mark.parametrize(
        ("stake", "discards", "scaling", "small_reward"),
        [(1, 4, 1, 3), (2, 4, 1, 0), (3, 4, 2, 0), (5, 3, 2, 0), (6, 3, 3, 0), (8, 3, 3, 0)],
    )
    def test_stakes(self, stake: int, discards: int, scaling: int, small_reward: int) -> None:
        game = Game.new("seed", RunConfig(stake=stake))
        assert game.state.discards_per_round == discards
        assert game.state.scaling == scaling
        game.step(SelectBlind())
        assert game.state.blind is not None
        assert game.state.blind.dollars == small_reward

    def test_unsupported_config(self) -> None:
        with pytest.raises(ValueError, match="deck"):
            new_run("seed", RunConfig(deck="b_blue"))
        with pytest.raises(ValueError, match="stake"):
            new_run("seed", RunConfig(stake=9))


class TestRound:
    def test_select_blind_draws_a_sorted_hand(self) -> None:
        game = Game.new("seed")
        events = game.step(SelectBlind())
        state = game.state
        assert state.phase is Phase.SELECTING_HAND
        assert state.round == 1
        assert len(state.hand) == 8
        assert len(state.deck) == 44
        assert state.blind is not None
        assert (state.blind.key, state.blind.chips, state.blind.dollars) == ("bl_small", 300, 3)
        ranks = [card.rank for card in state.hand]
        assert ranks == sorted(ranks, reverse=True)
        assert kinds(events) == ["blind_selected", "cards_drawn", "phase_changed"]

    @pytest.mark.parametrize(
        ("labels", "hand", "score"),
        [
            ("AS", HandType.HIGH_CARD, (5 + 11) * 1),
            ("AS AH", HandType.PAIR, (10 + 11 + 11) * 2),
            ("AS AH 2C", HandType.PAIR, (10 + 11 + 11) * 2),
            ("KS KH KC 2S 2D", HandType.FULL_HOUSE, (40 + 30 + 2 + 2) * 4),
            ("9S 8H 7C 6D 5S", HandType.STRAIGHT, (30 + 35) * 4),
            ("TS JS QS KS AS", HandType.STRAIGHT_FLUSH, (100 + 51) * 8),
        ],
    )
    def test_scoring(self, labels: str, hand: HandType, score: int) -> None:
        game = started_round()
        events = play(game, labels)
        scored = next(event for event in events if isinstance(event, HandScored))
        assert (scored.hand, scored.score) == (hand, score)
        assert game.state.chips == score
        assert game.state.hands[hand].played == 1
        assert game.state.last_hand_played is hand

    def test_debuffed_cards_add_no_chips(self) -> None:
        game = started_round()
        cards = set_hand(game.state, "AS AH")
        cards[0].debuffed = True
        game.step(ToggleCard(0))
        game.step(ToggleCard(1))
        events = game.step(Play())
        assert (
            next(event for event in events if isinstance(event, HandScored)).score == (10 + 11) * 2
        )

    def test_play_moves_cards_to_discard_and_draws(self) -> None:
        game = started_round()
        state = game.state
        played = set_hand(state, "AS AH 2C 3D 4H 5S 9C TD")[:2]
        game.step(ToggleCard(0))
        game.step(ToggleCard(1))
        game.step(Play())
        assert state.hands_left == 3
        assert [card.uid for card in state.discard] == [card.uid for card in played]
        assert all(card.times_played == 1 for card in played)
        assert len(state.hand) == 8
        assert state.phase is Phase.SELECTING_HAND

    def test_discard(self) -> None:
        game = started_round()
        state = game.state
        set_hand(state, "AS AH 2C 3D 4H 5S 9C TD")
        for action in macros.discard_cards(state, [2, 3, 4]):
            game.step(action)
        assert state.discards_left == 3
        assert {card.label for card in state.discard} == {"2C", "3D", "4H"}
        assert all(card.discarded for card in state.discard)
        assert len(state.hand) == 8

    def test_drawing_resorts_the_hand(self) -> None:
        game = started_round()
        state = game.state
        set_hand(state, "2C AS KH QD JC TS 9H 8D")
        game.step(SortBySuit())
        assert state.hand_sort is HandSort.SUIT
        assert [card.suit.value for card in state.hand] == sorted(
            (card.suit.value for card in state.hand), key="SHCD".index
        )
        game.step(MoveCard(0, 7))
        for action in macros.discard_cards(state, [0]):
            game.step(action)
        suits = [card.suit.value for card in state.hand]
        assert suits == sorted(suits, key="SHCD".index)

    def test_selection_limit(self) -> None:
        game = started_round()
        for i in range(5):
            game.step(ToggleCard(i))
        assert not game.is_legal(ToggleCard(5))
        assert game.is_legal(ToggleCard(4))
        with pytest.raises(IllegalActionError):
            game.step(ToggleCard(5))

    def test_play_and_discard_need_a_selection(self) -> None:
        game = started_round()
        assert not game.is_legal(Play())
        assert not game.is_legal(Discard())
        game.state.discards_left = 0
        game.step(ToggleCard(0))
        assert game.is_legal(Play())
        assert not game.is_legal(Discard())

    def test_move_card(self) -> None:
        game = started_round()
        set_hand(game.state, "AS KS QS JS")
        game.step(MoveCard(0, 3))
        assert [card.label for card in game.state.hand] == ["KS", "QS", "JS", "AS"]
        assert not game.is_legal(MoveCard(1, 1))
        assert not game.is_legal(MoveCard(0, 4))

    def test_illegal_action_leaves_state_unchanged(self) -> None:
        game = started_round()
        before = snapshot(game.state)
        with pytest.raises(IllegalActionError):
            game.step(CashOut())
        assert snapshot(game.state) == before

    def test_losing_ends_the_run(self) -> None:
        game = started_round()
        events: list[Event] = []
        for _ in range(4):
            events += play(game, "2C")
        assert game.state.phase is Phase.GAME_OVER
        assert game.is_over
        assert legal_actions(game.state) == []
        ended = events[-1]
        assert isinstance(ended, RunEnded)
        assert not ended.won

    def test_running_out_of_cards_ends_the_round(self) -> None:
        game = started_round()
        state = game.state
        set_hand(state, "2C")
        state.discard.extend(state.deck)
        state.deck.clear()
        play(game, "2C")
        assert state.phase is Phase.GAME_OVER

    def test_no_hand_size_ends_the_run(self) -> None:
        game = Game.new("seed")
        game.state.hand_size = 0
        game.step(SelectBlind())
        assert game.state.phase is Phase.GAME_OVER


class TestProgression:
    def test_win_cash_out_and_shop(self) -> None:
        game = started_round()
        state = game.state
        events = play(game, "TS JS QS KS AS")
        assert state.phase is Phase.ROUND_EVAL
        assert [(payout.source, payout.dollars) for payout in state.payouts] == [
            ("blind", 3),
            ("hands", 3),
        ]
        assert state.blind_states[BlindSlot.SMALL] is BlindStatus.DEFEATED
        assert len(state.deck) == 52
        assert not state.hand
        assert "round_ended" in kinds(events)

        game.step(CashOut())
        assert state.money == 4 + 3 + 3
        assert state.phase is Phase.SHOP
        assert state.chips == 0

        game.step(EndShop())
        assert state.phase is Phase.BLIND_SELECT
        assert state.blind_on_deck is BlindSlot.BIG
        game.step(SelectBlind())
        assert state.blind is not None
        assert state.blind.chips == 450

    def test_interest(self) -> None:
        game = started_round()
        game.state.money = 23
        play(game, "TS JS QS KS AS")
        assert game.state.payouts[-1].source == "interest"
        assert game.state.payouts[-1].dollars == 4
        game.state.money = 100
        assert game.state.interest_cap // 5 == 5

    def test_beating_the_boss_starts_the_next_ante(self) -> None:
        game = started_round()
        beat_blind(game)
        for _ in range(2):
            game.step(EndShop())
            game.step(SelectBlind())
            beat_blind(game)
        state = game.state
        assert state.ante == 2
        assert state.round == 3
        assert state.blind_on_deck is BlindSlot.SMALL
        assert all(status is BlindStatus.UPCOMING for status in state.blind_states.values())
        assert state.bosses_used[state.blind_choices[BlindSlot.BOSS]] >= 1

    def test_winning_the_run(self) -> None:
        game = started_round()
        state = game.state
        state.ante = 8
        state.blind_on_deck = BlindSlot.BOSS
        state.hands[HandType.HIGH_CARD].mult = 10**9
        set_hand(state, "AS")
        state.blind = None
        state.phase = Phase.BLIND_SELECT
        game.step(SelectBlind())
        play(game, "AS")
        assert state.phase is Phase.WIN
        assert state.won
        assert legal_actions(state) == [ContinueEndless(), EndRun()]

        endless = game.clone()
        endless.step(ContinueEndless())
        assert endless.state.phase is Phase.ROUND_EVAL
        assert endless.state.ante == 9

        events = game.step(EndRun())
        assert game.is_over
        assert isinstance(events[-1], RunEnded)
        assert events[-1].won


def test_move_joker() -> None:
    game = Game.new("seed")
    state = game.state
    state.jokers = [Item(uid=state.new_uid(), key=key) for key in ("j_a", "j_b", "j_c")]
    assert MoveJoker(0, 2) in game.legal_actions()
    game.step(MoveJoker(0, 2))
    assert [joker.key for joker in state.jokers] == ["j_b", "j_c", "j_a"]


def test_same_seed_and_actions_give_the_same_game() -> None:
    def run() -> Game:
        game = started_round("same")
        for action in macros.discard_cards(game.state, [0, 1, 2]):
            game.step(action)
        for action in macros.play_cards(game.state, [0, 1]):
            game.step(action)
        return game

    assert snapshot(run().state) == snapshot(run().state)


def test_different_seeds_give_different_hands() -> None:
    hands = {tuple(card.label for card in started_round(seed).state.hand) for seed in range(10)}
    assert len(hands) == 10
