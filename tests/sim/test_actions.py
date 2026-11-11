from __future__ import annotations

import dataclasses

import pytest

from balatro_gym.sim import actions as act
from balatro_gym.sim import macros, parse_action
from scenarios import set_hand, started_round


@pytest.mark.parametrize("cls", list(act.ACTION_TYPES.values()))
def test_every_action_round_trips(cls: type[act.Action]) -> None:
    action = cls(*range(1, len(dataclasses.fields(cls)) + 1))
    assert parse_action(action.to_list()) == action


def test_action_text() -> None:
    assert act.MoveCard(0, 3).to_list() == ["move_card", 0, 3]
    assert str(act.MoveCard(0, 3)) == "move_card(0, 3)"
    assert str(act.Play()) == "play()"


@pytest.mark.parametrize(
    "data",
    [[], ["fly"], ["play", 1], ["toggle_card"], ["toggle_card", "1"], ["toggle_card", True]],
)
def test_malformed_actions(data: list[str | int]) -> None:
    with pytest.raises(ValueError, match=r"action|arguments"):
        parse_action(data)


def test_select_cards_deselects_first() -> None:
    game = started_round()
    game.step(act.ToggleCard(0))
    game.step(act.ToggleCard(1))
    actions = macros.select_cards(game.state, [3, 1])
    assert actions == [act.ToggleCard(0), act.ToggleCard(3)]
    for action in actions:
        game.step(action)
    assert [c.selected for c in game.state.hand][:4] == [False, True, False, True]


def test_play_and_discard_macros() -> None:
    game = started_round()
    assert macros.play_cards(game.state, [2])[-1] == act.Play()
    assert macros.discard_cards(game.state, [2])[-1] == act.Discard()
    with pytest.raises(ValueError, match="at most"):
        macros.select_cards(game.state, range(6))
    with pytest.raises(ValueError, match="out of range"):
        macros.select_cards(game.state, [8])


@pytest.mark.parametrize("order", [[3, 2, 1, 0], [0, 1, 2, 3], [1, 3, 0, 2]])
def test_arrange_hand(order: list[int]) -> None:
    game = started_round()
    labels = [c.label for c in set_hand(game.state, "AS KS QS JS")]
    for action in macros.arrange_hand(game.state, order):
        game.step(action)
    assert [c.label for c in game.state.hand] == [labels[i] for i in order]


def test_arrange_hand_needs_a_permutation() -> None:
    game = started_round()
    with pytest.raises(ValueError, match="permutation"):
        macros.arrange_hand(game.state, [0, 0, 1])
