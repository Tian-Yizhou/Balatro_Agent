from __future__ import annotations

import random
from pathlib import Path

import pytest

from balatro_gym.replay import KeyframePolicy, Recorder, Replay, ReplayDriftError
from balatro_gym.sim import Game, snapshot
from balatro_gym.sim.actions import CashOut, Play
from balatro_gym.sim.state import Phase
from scenarios import Policy, greedy_policy, random_policy


def record(
    seed: str, policy: Policy = greedy_policy, keyframes: KeyframePolicy = "round"
) -> tuple[Replay, list[dict[str, object]]]:
    """Record a full game, also returning a snapshot of every step for comparison."""
    game = Game.new(seed)
    recorder = Recorder(game, game_id=f"test-{seed}", keyframes=keyframes)
    states = [snapshot(game.state)]
    randomizer = random.Random(seed)
    while not game.is_over:
        for action in policy(game, randomizer):
            annotation = {"choice": action.name} if isinstance(action, Play) else None
            recorder.step(action, annotation)
            states.append(snapshot(game.state))
    return recorder.replay(), states


@pytest.fixture(scope="module")
def recorded() -> tuple[Replay, list[dict[str, object]]]:
    return record("replay")


def test_state_at_every_step(recorded: tuple[Replay, list[dict[str, object]]]) -> None:
    replay, states = recorded
    assert replay.num_steps == len(states) - 1
    for index in range(0, len(states), 7):
        assert snapshot(replay.state_at(index)) == states[index]
    assert snapshot(replay.final_state()) == states[-1]


def test_round_keyframes(recorded: tuple[Replay, list[dict[str, object]]]) -> None:
    replay, _ = recorded
    keyframe_steps = sorted(replay.keyframes)
    assert keyframe_steps[-1] == replay.num_steps
    for index in keyframe_steps[:-1]:
        assert replay.state_at(index).phase is Phase.BLIND_SELECT


@pytest.mark.parametrize(("policy", "expected"), [(5, None), ("none", 1)])
def test_keyframe_policies(policy: KeyframePolicy, expected: int | None) -> None:
    replay, _ = record("policies", keyframes=policy)
    if isinstance(policy, int):
        assert all(step % policy == 0 or step == replay.num_steps for step in replay.keyframes)
        assert len(replay.keyframes) >= replay.num_steps // policy
    else:
        assert list(replay.keyframes) == [replay.num_steps]
    replay.verify()


def test_invalid_keyframe_interval() -> None:
    with pytest.raises(ValueError, match="interval"):
        Recorder(Game.new(1), keyframes=0)


@pytest.mark.parametrize("name", ["game.json", "game.json.gz"])
def test_save_and_load(
    tmp_path: Path, recorded: tuple[Replay, list[dict[str, object]]], name: str
) -> None:
    replay, _ = recorded
    path = tmp_path / name
    replay.save(path)
    loaded = Replay.load(path)
    assert loaded.to_dict() == replay.to_dict()
    assert loaded.header.metadata == {}
    assert loaded.annotations == replay.annotations
    assert loaded.annotations
    loaded.verify()


def test_compressed_files_are_deterministic(tmp_path: Path) -> None:
    replay, _ = record("same")
    replay.save(tmp_path / "a.json.gz")
    replay.save(tmp_path / "b.json.gz")
    assert (tmp_path / "a.json.gz").read_bytes() == (tmp_path / "b.json.gz").read_bytes()


def test_branch_plays_on_independently(recorded: tuple[Replay, list[dict[str, object]]]) -> None:
    replay, states = recorded
    game = replay.branch(1)
    game.step(game.legal_actions()[-1])
    assert snapshot(replay.state_at(1)) == states[1]


def test_drift_is_detected(recorded: tuple[Replay, list[dict[str, object]]]) -> None:
    replay, _ = recorded
    tampered = Replay.from_dict(replay.to_dict())
    step = sorted(tampered.keyframes)[0]
    tampered.keyframes[step]["state"]["money"] += 1
    with pytest.raises(ReplayDriftError, match=f"step {step}"):
        tampered.verify()


def test_illegal_recorded_action_is_drift(recorded: tuple[Replay, list[dict[str, object]]]) -> None:
    replay, _ = recorded
    tampered = Replay.from_dict(replay.to_dict())
    tampered.actions[0] = CashOut()
    with pytest.raises(ReplayDriftError, match="step 0"):
        tampered.verify()


def test_steps_yield_events_and_annotations(
    recorded: tuple[Replay, list[dict[str, object]]],
) -> None:
    replay, _ = recorded
    steps = list(replay.steps())
    assert [step.index for step in steps] == list(range(replay.num_steps))
    assert all(step.events for step in steps)
    assert {step.index for step in steps if step.annotation} == set(replay.annotations)


def test_random_games_replay_exactly() -> None:
    for seed in ("r1", "r2", "r3"):
        replay, states = record(seed, random_policy, keyframes="none")
        assert snapshot(replay.final_state()) == states[-1]
        replay.verify()


def test_out_of_range_step(recorded: tuple[Replay, list[dict[str, object]]]) -> None:
    replay, _ = recorded
    with pytest.raises(IndexError):
        replay.state_at(replay.num_steps + 1)


@pytest.mark.parametrize(
    ("change", "message"),
    [({"format": "other"}, "not a balatro-gym replay"), ({"format_version": 99}, "version")],
)
def test_unknown_formats_are_rejected(
    recorded: tuple[Replay, list[dict[str, object]]], change: dict[str, object], message: str
) -> None:
    data = {**recorded[0].to_dict(), **change}
    with pytest.raises(ValueError, match=message):
        Replay.from_dict(data)
