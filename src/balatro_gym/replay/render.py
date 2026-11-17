"""Plain-text views of states, events and replays, for people to read."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from typing import Any, Literal

from balatro_gym.sim import Event, Phase, RunState
from balatro_gym.sim.blinds import BLINDS
from balatro_gym.sim.events import (
    AnteChanged,
    BlindSelected,
    BossChosen,
    CardDestroyed,
    CardMoved,
    CardRetriggered,
    CardsDiscarded,
    CardsDrawn,
    CardToggled,
    ConsumableUsed,
    EffectApplied,
    HandLevelChanged,
    HandPlayed,
    HandScored,
    HandSorted,
    ItemBought,
    ItemCreated,
    ItemSold,
    JokerDestroyed,
    JokerMoved,
    MoneyChanged,
    PackOpened,
    PackPicked,
    PackSkipped,
    PhaseChanged,
    PlayingCardCreated,
    RoundEnded,
    RunEnded,
    ShopRerolled,
)
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
    if state.phase is Phase.SHOP:
        offers = " · ".join(f"{item.key} ${item.cost}" for item in state.shop_cards)
        lines.append(f"Shop    {offers or '(empty)'} · reroll ${state.reroll_cost}")
        if state.shop_boosters:
            packs = " · ".join(f"{item.key} ${item.cost}" for item in state.shop_boosters)
            lines.append(f"Packs   {packs}")
    if state.pack is not None:
        contents = [item.key for item in state.pack.items]
        contents += [_card(card) for card in state.pack.playing_cards]
        lines.append(
            f"Pack    {state.pack.key}: {' '.join(contents)} · {state.pack.choices_left} to choose"
        )
    if state.phase is Phase.ROUND_EVAL:
        rows = " + ".join(f"{payout.source} ${payout.dollars}" for payout in state.payouts)
        lines.append(f"Cash out  {rows} = ${sum(payout.dollars for payout in state.payouts)}")
    return "\n".join(lines)


def render_events(events: Iterable[Event], state: RunState) -> list[str]:
    """One line per notable event. ``state`` is used to name cards by their uid."""
    labels = {card.uid: card.label for card in state.playing_cards()}
    if state.pack is not None:
        labels.update((item.uid, item.key) for item in state.pack.items)
        labels.update((card.uid, card.label) for card in state.pack.playing_cards)
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
        changed = any(isinstance(event, PhaseChanged) for event in step.events)
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


def _money(event: MoneyChanged, labels: dict[int, str]) -> str:
    sign = "+" if event.delta >= 0 else "-"
    reason = event.reason.replace("_", " ")
    return f"{sign}${abs(event.delta)} from {reason} (now ${event.balance})"


def _effect(event: EffectApplied, labels: dict[int, str]) -> str:
    parts = []
    if event.chips:
        parts.append(f"{event.chips:+g} chips")
    if event.mult:
        parts.append(f"{event.mult:+g} mult")
    if event.x_mult != 1:
        parts.append(f"x{event.x_mult:g} mult")
    if event.dollars:
        parts.append(f"{event.dollars:+d} $")
    from_card = event.key in ("card", "held")
    source = labels.get(event.source, event.key) if from_card else event.key
    if not from_card and event.target is not None:
        source += f" on {_names([event.target], labels)}"
    if event.via is not None:
        source += " (copied)"
    return f"  {source}: {', '.join(parts)}"


def _round_ended(event: RoundEnded, labels: dict[int, str]) -> str:
    name = BLINDS[event.blind].name
    result = "Beat" if event.won else "Lost to"
    return f"{result} the {name}: {_number(event.score)} / {_number(event.target)}"


def _hand_scored(event: HandScored, labels: dict[int, str]) -> str:
    return (
        f"{event.hand.value}: {_number(event.chips)} chips x {_number(event.mult)} mult"
        f" = {_number(event.score)} (round score {_number(event.total)})"
    )


_FORMATTERS: dict[type[Event], Callable[[Any, dict[int, str]], str]] = {
    BossChosen: lambda event, _: f"Boss for ante {event.ante}: {BLINDS[event.boss].name}",
    BlindSelected: lambda event, _: (
        f"Selected the {BLINDS[event.blind].name}: score {_number(event.target)} to win"
    ),
    CardsDrawn: lambda event, labels: f"Drew {_names(event.uids, labels)}",
    CardToggled: lambda event, labels: (
        f"{'Selected' if event.selected else 'Deselected'} {_names([event.uid], labels)}"
    ),
    CardMoved: lambda event, labels: (
        f"Moved {_names([event.uid], labels)} from position {event.source} to {event.target}"
    ),
    JokerMoved: lambda event, _: (
        f"Moved joker #{event.uid} from position {event.source} to {event.target}"
    ),
    HandSorted: lambda event, _: (
        f"Sorted the hand by {'suit' if event.mode is HandSort.SUIT else 'rank'}"
    ),
    HandPlayed: lambda event, labels: (
        f"Played {_names(event.uids, labels)}: {event.hand.value},"
        f" scoring {_names(event.scoring, labels)}"
    ),
    EffectApplied: lambda event, labels: _effect(event, labels),
    CardRetriggered: lambda event, labels: (
        f"  {_names([event.uid], labels)} retriggered by {event.key}"
    ),
    CardDestroyed: lambda event, labels: (
        f"{_names([event.uid], labels)} destroyed ({event.reason})"
    ),
    JokerDestroyed: lambda event, _: f"{event.key} destroyed ({event.reason})",
    HandLevelChanged: lambda event, _: f"{event.hand.value} is now level {event.level}",
    HandScored: _hand_scored,
    CardsDiscarded: lambda event, labels: f"Discarded {_names(event.uids, labels)}",
    RoundEnded: _round_ended,
    AnteChanged: lambda event, _: f"Reached ante {event.ante}",
    MoneyChanged: _money,
    RunEnded: lambda event, _: (
        f"Run over ({event.reason.replace('_', ' ')}): {'won' if event.won else 'not won'}"
        f" at ante {event.ante}, round {event.round}"
    ),
    ItemBought: lambda event, _: f"Bought {event.key} for ${event.cost}",
    ItemSold: lambda event, _: f"Sold {event.key} for ${event.value}",
    ItemCreated: lambda event, _: f"{event.key} appeared in {event.area}",
    ConsumableUsed: lambda event, labels: (
        f"Used {event.key}" + (f" on {_names(event.targets, labels)}" if event.targets else "")
    ),
    ShopRerolled: lambda event, _: f"Rerolled the shop for ${event.cost}",
    PackOpened: lambda event, labels: f"Opened {event.key}: {_names(event.cards, labels)}",
    PackPicked: lambda event, _: f"Took {event.name} from the pack",
    PackSkipped: lambda event, _: "Skipped the rest of the pack",
    PlayingCardCreated: lambda event, _: f"Added {event.label} to the {event.area}",
}
