"""Random and greedy games, checking invariants after every step."""

from __future__ import annotations

import dataclasses
import json
import random

import pytest

from balatro_gym.sim import Game, IllegalActionError, is_legal, legal_actions, snapshot
from balatro_gym.sim import actions as act
from balatro_gym.sim.engine import SELECTION_LIMIT
from balatro_gym.sim.state import Phase, RunState
from scenarios import Policy, greedy_policy, play_game, random_policy


def candidate_actions(state: RunState) -> list[act.Action]:
    """Every action with arguments near the valid ranges, legal or not."""
    hand = range(-1, len(state.hand) + 1)
    jokers = range(-1, len(state.jokers) + 1)
    candidates: list[act.Action] = [
        cls() for cls in act.ACTION_TYPES.values() if not dataclasses.fields(cls)
    ]
    candidates += [act.ToggleCard(i) for i in hand]
    candidates += [act.MoveCard(i, j) for i in hand for j in hand]
    candidates += [act.MoveJoker(i, j) for i in jokers for j in jokers]
    candidates += [cls(i) for cls in (act.SellJoker, act.Buy, act.Pick) for i in range(2)]
    return candidates


def check_invariants(game: Game, uids: set[int]) -> None:
    state = game.state
    assert {card.uid for card in state.playing_cards()} == uids
    assert len(state.playing_cards()) == len(uids)
    assert not any(card.selected for card in state.deck + state.discard)
    assert sum(card.selected for card in state.hand) <= SELECTION_LIMIT
    assert state.hands_left >= 0
    assert state.discards_left >= 0
    if state.phase is Phase.SELECTING_HAND:
        assert len(state.hand) <= max(state.hand_size, 0) or not state.deck
    legal = legal_actions(state)
    assert len(legal) == len(set(legal))
    assert set(legal) == {a for a in candidate_actions(state) if is_legal(state, a)}
    assert bool(legal) != state.is_over


@pytest.mark.parametrize(
    ("policy", "seeds"),
    [(random_policy, range(25)), (greedy_policy, range(8))],
    ids=["random", "greedy"],
)
def test_invariants_hold(policy: Policy, seeds: range) -> None:
    for seed in seeds:
        game = Game.new(seed)
        uids = {card.uid for card in game.state.playing_cards()}
        rnd = random.Random(seed)
        steps = 0
        while not game.is_over and steps < 3000:
            for action in policy(game, rnd):
                game.step(action)
                steps += 1
                check_invariants(game, uids)
            if steps % 50 == 0:
                data = json.loads(json.dumps(game.snapshot()))
                assert Game.from_snapshot(data).state == game.state
        assert game.is_over


def test_illegal_actions_are_rejected() -> None:
    game, _ = play_game(3, greedy_policy, max_steps=30)
    before = snapshot(game.state)
    for action in candidate_actions(game.state):
        if not game.is_legal(action):
            with pytest.raises(IllegalActionError):
                game.step(action)
    assert snapshot(game.state) == before


def test_replaying_actions_reproduces_the_game() -> None:
    for seed in range(5):
        game, taken = play_game(seed, greedy_policy)
        again = Game.new(seed)
        for action in taken:
            again.step(action)
        assert snapshot(again.state) == snapshot(game.state)


def test_greedy_player_reaches_later_antes() -> None:
    antes = [play_game(seed, greedy_policy)[0].state.ante for seed in range(8)]
    assert max(antes) >= 2
