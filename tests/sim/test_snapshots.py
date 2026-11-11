from __future__ import annotations

import json

import pytest

from balatro_gym.sim import (
    STATE_SCHEMA_VERSION,
    Game,
    SnapshotError,
    clone,
    determinize,
    restore,
    snapshot,
)
from balatro_gym.sim.actions import Discard, SelectBlind, ToggleCard
from balatro_gym.sim.events import EVENT_TYPES, CardsDrawn, event_to_data
from balatro_gym.sim.serialize import from_data, to_data
from scenarios import greedy_policy, play_game, started_round


def test_snapshot_round_trips_through_json() -> None:
    game, _ = play_game("snap", greedy_policy, max_steps=40)
    data = json.loads(json.dumps(game.snapshot()))
    restored = Game.from_snapshot(data)
    assert restored.state == game.state
    assert restored.snapshot() == game.snapshot()


def test_restored_game_continues_identically() -> None:
    game = started_round("continue")
    copy = Game.from_snapshot(json.loads(json.dumps(game.snapshot())))
    for each in (game, copy):
        each.step(ToggleCard(0))
        each.step(Discard())
    assert game.snapshot() == copy.snapshot()


def test_clone_is_independent() -> None:
    game = started_round("clone")
    copy = game.clone()
    assert copy.state == game.state
    copy.step(ToggleCard(0))
    assert copy.state != game.state
    assert not game.state.hand[0].selected


def test_clone_of_state() -> None:
    state = started_round("clone").state
    copy = clone(state)
    assert copy == state
    assert copy is not state
    assert copy.hand[0] is not state.hand[0]


def test_determinize_keeps_visible_state_and_hides_the_future() -> None:
    game = started_round("det")
    state = game.state
    other = determinize(state, 1)
    assert [card.uid for card in other.hand] == [card.uid for card in state.hand]
    assert (other.money, other.ante, other.blind_choices) == (
        state.money,
        state.ante,
        state.blind_choices,
    )
    assert sorted(card.uid for card in other.deck) == sorted(card.uid for card in state.deck)
    assert [card.uid for card in other.deck] != [card.uid for card in state.deck]
    assert determinize(state, 1) == other
    assert determinize(state, 2) != other


def test_schema_version_is_checked() -> None:
    data = started_round().snapshot()
    data["schema"] = STATE_SCHEMA_VERSION + 1
    with pytest.raises(SnapshotError, match="schema"):
        restore(data)


def test_malformed_snapshot() -> None:
    data = started_round().snapshot()
    del data["state"]["money"]
    with pytest.raises(SnapshotError, match="malformed"):
        restore(data)
    data = started_round().snapshot()
    data["state"]["money"] = "4"
    with pytest.raises(SnapshotError, match="malformed"):
        restore(data)


def test_events_serialize() -> None:
    game = Game.new("events")
    for event in game.step(SelectBlind()):
        data = event_to_data(event)
        assert data["kind"] == event.kind
        restored = from_data(EVENT_TYPES[data.pop("kind")], data)
        assert restored == event
    assert to_data(CardsDrawn((1, 2))) == {"uids": [1, 2]}


def test_snapshot_is_small() -> None:
    assert len(json.dumps(snapshot(started_round().state))) < 20_000
