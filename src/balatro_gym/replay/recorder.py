"""Recording games as they are played."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from balatro_gym.sim import Action, Event, Game, Phase, snapshot
from balatro_gym.sim.events import PhaseChanged
from balatro_gym.sim.serialize import to_data

from .format import Replay, ReplayHeader

__all__ = ["KeyframePolicy", "Recorder"]

#: When to store a keyframe: at the start of every round (``"round"``), every
#: ``n`` steps (an ``int``), or only at the end (``"none"``).
KeyframePolicy: TypeAlias = Literal["round", "none"] | int


class Recorder:
    """Plays a game and records it as a :class:`Replay`.

    Send every action through :meth:`step`. The final state is always stored as
    a keyframe, so a replay can be verified end to end.

    Example::

        recorder = Recorder(Game.new(seed=7), game_id="run-7")
        while not recorder.game.is_over:
            recorder.step(policy(recorder.game), annotation={"value": 0.4})
        recorder.replay().save("run-7.json.gz")
    """

    __slots__ = ("_keyframes", "_replay", "game")

    def __init__(
        self,
        game: Game,
        *,
        game_id: str = "",
        keyframes: KeyframePolicy = "round",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(keyframes, int) and keyframes < 1:
            raise ValueError("a keyframe interval must be at least 1 step")
        state = game.state
        self.game = game
        self._keyframes: KeyframePolicy = keyframes
        self._replay = Replay(
            header=ReplayHeader(
                game_id=game_id or state.rng.seed,
                seed=state.rng.seed,
                config=to_data(state.config),
                metadata=dict(metadata or {}),
            ),
            initial=snapshot(state),
        )

    def step(self, action: Action, annotation: Any = None) -> list[Event]:
        """Apply ``action``, record it, and return its events.

        ``annotation`` is any JSON-compatible value the agent wants to keep with
        this decision, such as action probabilities or its reasoning.
        """
        index = len(self._replay.actions)
        events = self.game.step(action)
        self._replay.actions.append(action)
        if annotation is not None:
            self._replay.annotations[index] = annotation
        if self._is_keyframe(index + 1, events):
            self._replay.keyframes[index + 1] = snapshot(self.game.state)
        return events

    def replay(self) -> Replay:
        """The game so far as a replay, with the current state as a keyframe."""
        replay = self._replay
        last = len(replay.actions)
        keyframes = dict(replay.keyframes)
        if last:
            keyframes[last] = snapshot(self.game.state)
        return Replay(
            header=replay.header,
            initial=replay.initial,
            actions=list(replay.actions),
            keyframes=keyframes,
            annotations=dict(replay.annotations),
        )

    def _is_keyframe(self, index: int, events: list[Event]) -> bool:
        policy = self._keyframes
        if isinstance(policy, int):
            return index % policy == 0
        if policy == "round":
            return any(
                isinstance(event, PhaseChanged) and event.new is Phase.BLIND_SELECT
                for event in events
            )
        return False
