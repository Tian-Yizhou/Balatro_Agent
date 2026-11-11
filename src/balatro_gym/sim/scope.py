"""The working context of one engine step: the state, its events, deferred work."""

from __future__ import annotations

from collections.abc import Callable

from .events import Event, MoneyChanged, PhaseChanged
from .rng import Rng
from .state import Phase, RunState

__all__ = ["Scope"]


class Scope:
    """The state being changed by one action, and the events emitted so far.

    Some effects in the game run through its event queue after the code that
    triggers them has finished, for example a joker destroying itself while the
    other jokers are still being evaluated. ``defer`` queues such work and
    ``flush`` runs it at the point where the game's queue would.
    """

    __slots__ = ("_deferred", "events", "state")

    def __init__(self, state: RunState) -> None:
        self.state = state
        self.events: list[Event] = []
        self._deferred: list[Callable[[], None]] = []

    @property
    def rng(self) -> Rng:
        return self.state.rng

    def emit(self, event: Event) -> None:
        self.events.append(event)

    def set_phase(self, phase: Phase) -> None:
        old = self.state.phase
        if phase is not old:
            self.state.phase = phase
            self.emit(PhaseChanged(old, phase))

    def change_money(self, delta: int, reason: str) -> None:
        """Source: functions/common_events.lua ``ease_dollars``."""
        if delta == 0:
            return
        self.state.money += delta
        self.emit(MoneyChanged(delta, self.state.money, reason))

    def chance(self, key: str, odds: float) -> bool:
        """A "1 in ``odds``" roll, scaled by the run's probability multiplier.

        Source: the ``pseudorandom(key) < G.GAME.probabilities.normal / odds`` idiom.
        """
        return self.rng.random(key) < self.state.probabilities_normal / odds

    def defer(self, work: Callable[[], None]) -> None:
        self._deferred.append(work)

    def flush(self) -> None:
        while self._deferred:
            work = self._deferred.pop(0)
            work()
