"""The replay file: a recorded game that can be restored at any step.

A replay stores the initial state, every atomic action, and keyframe snapshots
(by default at the start of every round, plus the final state). Any intermediate
state is rebuilt by restoring the nearest keyframe and re-applying the actions
after it, so replays stay small while every step remains inspectable.

Files are JSON, gzip-compressed when the file name ends in ``.gz``.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from balatro_gym.sim import (
    GAME_VERSION,
    RULES_VERSION,
    STATE_SCHEMA_VERSION,
    Action,
    Event,
    Game,
    RunState,
    parse_action,
    restore,
    snapshot,
    step,
)

__all__ = ["FORMAT_NAME", "FORMAT_VERSION", "Replay", "ReplayDriftError", "ReplayHeader", "Step"]

FORMAT_NAME = "balatro-gym-replay"
#: Bumped whenever the layout of a replay file changes.
FORMAT_VERSION = 1


class ReplayDriftError(AssertionError):
    """Re-simulating a replay no longer reproduces its recorded states."""

    def __init__(self, step: int, message: str) -> None:
        super().__init__(f"step {step}: {message}")
        self.step = step


@dataclass(frozen=True, slots=True)
class ReplayHeader:
    """Identifies a recorded game and the engine that recorded it."""

    game_id: str
    seed: str
    config: dict[str, Any]
    rules_version: int = RULES_VERSION
    schema_version: int = STATE_SCHEMA_VERSION
    game_version: str = GAME_VERSION
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True, slots=True)
class Step:
    """One action of a replay as it is re-simulated.

    ``index`` is the number of actions applied before this one, so the action
    moves the game from step ``index`` to step ``index + 1``. ``state`` is the
    live state after the action; it changes as iteration continues.
    """

    index: int
    action: Action
    events: list[Event]
    state: RunState
    annotation: Any


@dataclass(slots=True)
class Replay:
    """A recorded game.

    Step ``n`` is the state after the first ``n`` actions: step 0 is the initial
    state and step ``len(actions)`` is the final state. ``keyframes`` maps step
    numbers to snapshots, and ``annotations`` maps step numbers to whatever the
    agent attached to the action it took at that step.
    """

    header: ReplayHeader
    initial: dict[str, Any]
    actions: list[Action] = field(default_factory=list[Action])
    keyframes: dict[int, dict[str, Any]] = field(default_factory=dict[int, dict[str, Any]])
    annotations: dict[int, Any] = field(default_factory=dict[int, Any])

    @property
    def num_steps(self) -> int:
        return len(self.actions)

    def state_at(self, index: int) -> RunState:
        """The state after the first ``index`` actions."""
        if not 0 <= index <= self.num_steps:
            raise IndexError(f"step {index} is outside 0..{self.num_steps}")
        start = max((step for step in self.keyframes if step <= index), default=0)
        state = restore(self.keyframes[start] if start else self.initial)
        for action in self.actions[start:index]:
            step(state, action)
        return state

    def branch(self, index: int) -> Game:
        """A live game resumed at step ``index``, to play on from there."""
        return Game(self.state_at(index))

    def final_state(self) -> RunState:
        return self.state_at(self.num_steps)

    def steps(self, start: int = 0, stop: int | None = None) -> Iterator[Step]:
        """Re-simulate actions ``start`` to ``stop`` and yield each with its events."""
        stop = self.num_steps if stop is None else min(stop, self.num_steps)
        state = self.state_at(start)
        for index in range(start, stop):
            action = self.actions[index]
            events = step(state, action)
            yield Step(index, action, events, state, self.annotations.get(index))

    def verify(self) -> None:
        """Re-simulate the whole game and check every keyframe.

        Raises:
            ReplayDriftError: at the first keyframe the engine no longer
                reproduces, for example after a change to the rules.
        """
        state = restore(self.initial)
        for index, action in enumerate(self.actions):
            try:
                step(state, action)
            except ValueError as exc:
                raise ReplayDriftError(index, f"{action} failed: {exc}") from exc
            expected = self.keyframes.get(index + 1)
            if expected is not None and _canonical(snapshot(state)) != _canonical(expected):
                raise ReplayDriftError(index + 1, "state differs from the recorded keyframe")

    # Serialization.

    def to_dict(self) -> dict[str, Any]:
        header = self.header
        return {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "header": {
                "game_id": header.game_id,
                "seed": header.seed,
                "config": header.config,
                "rules_version": header.rules_version,
                "schema_version": header.schema_version,
                "game_version": header.game_version,
                "metadata": header.metadata,
            },
            "initial": self.initial,
            "actions": [action.to_list() for action in self.actions],
            "keyframes": [
                {"step": step, "state": state} for step, state in sorted(self.keyframes.items())
            ],
            "annotations": [
                {"step": step, "data": data} for step, data in sorted(self.annotations.items())
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Replay:
        if data.get("format") != FORMAT_NAME:
            raise ValueError("not a balatro-gym replay")
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported replay format version {data.get('format_version')!r}")
        header = data["header"]
        return cls(
            header=ReplayHeader(
                game_id=header["game_id"],
                seed=header["seed"],
                config=header["config"],
                rules_version=header["rules_version"],
                schema_version=header["schema_version"],
                game_version=header["game_version"],
                metadata=header["metadata"],
            ),
            initial=data["initial"],
            actions=[parse_action(item) for item in data["actions"]],
            keyframes={item["step"]: item["state"] for item in data["keyframes"]},
            annotations={item["step"]: item["data"] for item in data["annotations"]},
        )

    def save(self, path: str | Path) -> None:
        """Write the replay as JSON, gzip-compressed if ``path`` ends in ``.gz``."""
        path = Path(path)
        text = json.dumps(self.to_dict(), separators=(",", ":"))
        if path.suffix == ".gz":
            path.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
        else:
            path.write_text(text + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Replay:
        path = Path(path)
        raw = path.read_bytes()
        if path.suffix == ".gz":
            raw = gzip.decompress(raw)
        return cls.from_dict(json.loads(raw))


def _canonical(data: Any) -> str:
    """Comparable text form of a snapshot; unlike ``==`` it treats NaN as equal to NaN."""
    return json.dumps(data, sort_keys=True)
