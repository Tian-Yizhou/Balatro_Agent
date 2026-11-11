"""Golden replays: recorded games whose outcome must never change by accident.

If a change to the engine alters game behavior on purpose, bump ``RULES_VERSION``
and re-record them::

    uv run pytest tests/replay/test_golden.py --update-golden
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from balatro_gym.replay import Recorder, Replay
from balatro_gym.sim import RULES_VERSION, Game
from scenarios import Policy, greedy_policy, random_policy

GOLDEN = Path(__file__).resolve().parents[1] / "golden"

GAMES: dict[str, tuple[str, Policy]] = {
    "greedy": ("golden-greedy", greedy_policy),
    "random": ("golden-random", random_policy),
}


def record(seed: str, policy: Policy) -> Replay:
    game = Game.new(seed)
    recorder = Recorder(game, game_id=seed, metadata={"policy": policy.__name__})
    rnd = random.Random(seed)
    while not game.is_over:
        for action in policy(game, rnd):
            recorder.step(action)
    return recorder.replay()


@pytest.mark.parametrize("name", sorted(GAMES))
def test_golden_replay(name: str, request: pytest.FixtureRequest) -> None:
    path = GOLDEN / f"{name}.json.gz"
    seed, policy = GAMES[name]
    if request.config.getoption("--update-golden"):
        GOLDEN.mkdir(exist_ok=True)
        record(seed, policy).save(path)
    replay = Replay.load(path)
    assert replay.header.rules_version == RULES_VERSION, (
        "RULES_VERSION changed: re-record the golden replays with --update-golden"
    )
    replay.verify()
    assert replay.final_state().is_over
