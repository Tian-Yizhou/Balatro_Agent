"""Plain-text views of states, events and replays, for people to read."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from typing import Any, Literal

from balatro_gym.sim import Event, Phase, RunState
from balatro_gym.sim import events as ev
from balatro_gym.sim.blinds import BLINDS
from balatro_gym.sim.state import BlindSlot, BlindStatus, HandSort, PlayingCard

from .format import Replay, Step

__all__ = ["render_events", "render_replay", "render_state", "render_step"]

StateDetail = Literal["all", "phases", "none"]


def render_state(state: RunState) -> str:
    """A few lines describing what the player sees."""
    lines = [_headline(state)]
    lines.append(
        f"${state.money} · hands {state.hands_left} · discards {state.discards_left}"
        f" · deck {len(state.deck)}/{len(state.playing_cards())}"
    )
    if state.phase is Phase.BLIND_SELECT:
        lines.append("Blinds  " + " · ".join(_blind_choice(state, slot) for slot in BlindSlot))
    if state.hand:
        lines.append("Hand    " + " ".join(_card(card) for card in state.hand))
    if state.jokers:
        lines.append("Jokers  " + " ".join(joker.key for joker in state.jokers))
    if state.consumables:
        lines.append("Items   " + " ".join(item.key for item in state.consumables))
    if state.phase is Phase.ROUND_EVAL:
        rows = " + ".join(f"{p.source} ${p.dollars}" for p in state.payouts)
        lines.append(f"Cash out  {rows} = ${sum(p.dollars for p in state.payouts)}")
    return "\n".join(lines)


def render_events(events: Iterable[Event], state: RunState) -> list[str]:
    """One line per notable event. ``state`` is used to name cards by their uid."""
    labels = {card.uid: card.label for card in state.playing_cards()}
    lines: list[str] = []
    for event in events:
        formatter = _FORMATTERS.get(type(event))
        if formatter is not None:
            lines.append(formatter(event, labels))
    return lines


def render_step(step: Step) -> str:
    """The action taken at a step, followed by what happened."""
    lines = [f"#{step.index} {step.action}"]
    lines.extend(f"    {line}" for line in render_events(step.events, step.state))
    if step.annotation is not None:
        note = json.dumps(step.annotation, ensure_ascii=False)
        lines.append(f"    note: {note if len(note) <= 200 else note[:197] + '...'}")
    return "\n".join(lines)


def render_replay(
    replay: Replay,
    start: int = 0,
    stop: int | None = None,
    *,
    states: StateDetail = "phases",
) -> str:
    """The game step by step.

    ``states`` controls how often the full state is printed: after every step
    (``"all"``), whenever the phase changes (``"phases"``), or never (``"none"``).
    """
    header = replay.header
    config = ", ".join(f"{key} {value}" for key, value in header.config.items())
    out = [
        f"Replay {header.game_id} · seed {header.seed} · {config} · rules v{header.rules_version}"
        f" · {replay.num_steps} steps"
    ]
    if states != "none":
        out.append(_indent(render_state(replay.state_at(start))))
    for step in replay.steps(start, stop):
        out.append(render_step(step))
        changed = any(isinstance(event, ev.PhaseChanged) for event in step.events)
        if states == "all" or (states == "phases" and changed):
            out.append(_indent(render_state(step.state)))
    return "\n".join(out)


def _headline(state: RunState) -> str:
    where = f"Ante {state.ante} · Round {state.round}"
    blind = state.blind
    if state.phase is Phase.SELECTING_HAND and blind is not None:
        name = BLINDS[blind.key].name
        return f"{where} · {name} · score {_number(state.chips)} / {_number(blind.chips)}"
    if state.phase is Phase.GAME_OVER:
        return f"{where} · run over: {'won' if state.won else 'lost'}"
    if state.phase is Phase.WIN:
        return f"{where} · won the run; continue in endless mode or end it"
    return f"{where} · {state.phase.value.replace('_', ' ')}"


def _blind_choice(state: RunState, slot: BlindSlot) -> str:
    key = state.blind_choices.get(slot)
    name = BLINDS[key].name if key else slot.value
    status = state.blind_states[slot]
    marker = "> " if slot is state.blind_on_deck and status is BlindStatus.SELECT else ""
    return f"{marker}{name} ({status.value.lower()})"


def _card(card: PlayingCard) -> str:
    text = card.label
    if card.debuffed:
        text += "!"
    return f"[{text}]" if card.selected else text


def _number(value: float) -> str:
    if math.isfinite(value) and value == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}" if math.isfinite(value) else str(value)


def _names(uids: Iterable[int], labels: dict[int, str]) -> str:
    return " ".join(labels.get(uid, f"#{uid}") for uid in uids)


def _indent(text: str) -> str:
    return "\n".join(f"  | {line}" for line in text.splitlines())


def _money(event: ev.MoneyChanged, labels: dict[int, str]) -> str:
    sign = "+" if event.delta >= 0 else "-"
    reason = event.reason.replace("_", " ")
    return f"{sign}${abs(event.delta)} from {reason} (now ${event.balance})"


def _round_ended(event: ev.RoundEnded, labels: dict[int, str]) -> str:
    name = BLINDS[event.blind].name
    result = "Beat" if event.won else "Lost to"
    return f"{result} the {name}: {_number(event.score)} / {_number(event.target)}"


def _hand_scored(event: ev.HandScored, labels: dict[int, str]) -> str:
    return (
        f"{event.hand.value}: {_number(event.chips)} chips x {_number(event.mult)} mult"
        f" = {_number(event.score)} (round score {_number(event.total)})"
    )


_FORMATTERS: dict[type[Event], Callable[[Any, dict[int, str]], str]] = {
    ev.BossChosen: lambda e, _: f"Boss for ante {e.ante}: {BLINDS[e.boss].name}",
    ev.BlindSelected: lambda e, _: (
        f"Selected the {BLINDS[e.blind].name}: score {_number(e.target)} to win"
    ),
    ev.CardsDrawn: lambda e, labels: f"Drew {_names(e.uids, labels)}",
    ev.CardToggled: lambda e, labels: (
        f"{'Selected' if e.selected else 'Deselected'} {_names([e.uid], labels)}"
    ),
    ev.CardMoved: lambda e, labels: (
        f"Moved {_names([e.uid], labels)} from position {e.source} to {e.target}"
    ),
    ev.JokerMoved: lambda e, _: f"Moved joker #{e.uid} from position {e.source} to {e.target}",
    ev.HandSorted: lambda e, _: (
        f"Sorted the hand by {'suit' if e.mode is HandSort.SUIT else 'rank'}"
    ),
    ev.HandPlayed: lambda e, labels: (
        f"Played {_names(e.uids, labels)}: {e.hand.value}, scoring {_names(e.scoring, labels)}"
    ),
    ev.CardScored: lambda e, labels: f"  {_names([e.uid], labels)} +{_number(e.chips)} chips",
    ev.HandScored: _hand_scored,
    ev.CardsDiscarded: lambda e, labels: f"Discarded {_names(e.uids, labels)}",
    ev.RoundEnded: _round_ended,
    ev.AnteChanged: lambda e, _: f"Reached ante {e.ante}",
    ev.MoneyChanged: _money,
    ev.RunEnded: lambda e, _: f"Run {'won' if e.won else 'lost'} at ante {e.ante}, round {e.round}",
}
