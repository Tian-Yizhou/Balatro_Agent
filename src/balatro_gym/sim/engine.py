"""Game rules: starting a run, listing legal actions and applying them.

The engine is deterministic: the same state and action always give the same
result, because all randomness comes from the state's own :class:`Rng`. Applying
an action changes the state in place and returns the events it caused; copy a
state first with ``serialize.clone`` to keep the original (ADR 0001).

The game runs its effects through an animation queue. Here each action runs to
completion synchronously, in the same order as the game's queued events.

Implemented so far: the core loop with plain cards. A run starts at blind
selection; a round is played with toggle, move, sort, play and discard; the round
is evaluated and cashed out; the shop can be left; the boss of each ante is chosen
by the game's rule (without its effect); and the run is won after the boss of
ante 8, with the choice to continue in endless mode.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .actions import (
    Action,
    CashOut,
    ContinueEndless,
    Discard,
    EndRun,
    EndShop,
    MoveCard,
    MoveJoker,
    Play,
    SelectBlind,
    SortByRank,
    SortBySuit,
    ToggleCard,
)
from .blinds import BLINDS, choose_boss, get_blind_amount
from .cards import sort_key
from .effects import Area, Context, Ctx, Effect, calculate_joker
from .events import (
    AnteChanged,
    BlindSelected,
    BossChosen,
    CardDestroyed,
    CardMoved,
    CardsDiscarded,
    CardsDrawn,
    CardToggled,
    Event,
    HandSorted,
    JokerMoved,
    RoundEnded,
    RunEnded,
)
from .hands import initial_hand_levels
from .jokers import DOLLAR_BONUS
from .modifiers import end_of_round_dollars, red_seal_repetitions
from .rng import Rng
from .scope import Scope
from .scoring import evaluate_play
from .serialize import clone
from .state import (
    BlindSlot,
    BlindState,
    BlindStatus,
    HandSort,
    Item,
    Payout,
    Phase,
    PlayingCard,
    RunConfig,
    RunState,
    Suit,
)

__all__ = [
    "SELECTION_LIMIT",
    "SUPPORTED_DECKS",
    "IllegalActionError",
    "determinize",
    "is_legal",
    "legal_actions",
    "new_run",
    "step",
]

#: The most cards that can be selected in hand (``CardArea`` ``highlighted_limit``).
SELECTION_LIMIT = 5

#: Decks implemented so far, with their changes to the starting parameters.
#: Source: game.lua ``P_CENTERS`` (``b_red``); back.lua ``Back:apply_to_run``.
SUPPORTED_DECKS: dict[str, dict[str, int]] = {"b_red": {"discards": 1}}

# Phases in which jokers can be dragged into a new order.
_JOKER_MOVE_PHASES = frozenset(
    {Phase.BLIND_SELECT, Phase.SELECTING_HAND, Phase.ROUND_EVAL, Phase.SHOP, Phase.PACK_OPEN}
)


class IllegalActionError(ValueError):
    """The action is not legal in the current state."""


# Starting a run.


def new_run(seed: str | int, config: RunConfig | None = None) -> RunState:
    """Start a run at the first blind selection.

    Source: game.lua ``Game:start_run``; functions/misc_functions.lua
    ``get_starting_params``.
    """
    config = config or RunConfig()
    if config.deck not in SUPPORTED_DECKS:
        raise ValueError(f"deck {config.deck!r} is not supported; choose from {SUPPORTED_DECKS}")
    if not 1 <= config.stake <= 8:
        raise ValueError(f"stake must be between 1 and 8, not {config.stake}")

    params = {
        "dollars": 4,
        "hand_size": 8,
        "discards": 3,
        "hands": 4,
        "joker_slots": 5,
        "consumable_slots": 2,
    }
    no_blind_reward: list[BlindSlot] = []
    scaling = 1
    if config.stake >= 2:
        no_blind_reward.append(BlindSlot.SMALL)
    if config.stake >= 3:
        scaling = 2
    if config.stake >= 5:
        params["discards"] -= 1
    if config.stake >= 6:
        scaling = 3
    for name, change in SUPPORTED_DECKS[config.deck].items():
        params[name] += change

    rng = Rng(str(seed))
    state = RunState(
        config=config,
        rng=rng,
        next_uid=1,
        phase=Phase.BLIND_SELECT,
        ante=1,
        round=0,
        win_ante=8,
        won=False,
        money=params["dollars"],
        chips=0.0,
        hands=initial_hand_levels(),
        last_hand_played=None,
        hands_per_round=params["hands"],
        discards_per_round=params["discards"],
        hand_size=params["hand_size"],
        joker_slots=params["joker_slots"],
        consumable_slots=params["consumable_slots"],
        hands_left=params["hands"],
        discards_left=params["discards"],
        hands_played_this_round=0,
        discards_used_this_round=0,
        hands_played=0,
        unused_discards=0,
        blind=None,
        blind_on_deck=BlindSlot.SMALL,
        blind_states={
            BlindSlot.SMALL: BlindStatus.SELECT,
            BlindSlot.BIG: BlindStatus.UPCOMING,
            BlindSlot.BOSS: BlindStatus.UPCOMING,
        },
        blind_choices={BlindSlot.SMALL: "bl_small", BlindSlot.BIG: "bl_big"},
        bosses_used={key: 0 for key, info in BLINDS.items() if info.boss},
        interest_cap=25,
        interest_amount=1,
        inflation=0,
        discount_percent=0,
        scaling=scaling,
        ante_scaling=1.0,
        no_blind_reward=no_blind_reward,
        payouts=[],
        probabilities_normal=1.0,
        pool_flags={},
        hand_sort=HandSort.RANK,
        deck=[],
        hand=[],
        discard=[],
        jokers=[],
        consumables=[],
    )
    for suit in Suit:
        for rank in range(2, 15):
            state.deck.append(PlayingCard(uid=state.new_uid(), rank=rank, suit=suit))
    state.blind_choices[BlindSlot.BOSS] = choose_boss(
        rng, state.ante, state.win_ante, state.bosses_used
    )
    return state


# Legal actions.


def legal_actions(state: RunState) -> list[Action]:
    """Every action that is legal in ``state``, in a fixed order."""
    actions: list[Action] = []
    phase = state.phase
    if phase is Phase.BLIND_SELECT:
        actions.append(SelectBlind())
    elif phase is Phase.SELECTING_HAND:
        selected = _count_selected(state)
        if 1 <= selected <= SELECTION_LIMIT:
            actions.append(Play())
        if selected >= 1 and state.discards_left > 0:
            actions.append(Discard())
        for i, card in enumerate(state.hand):
            if card.selected or selected < SELECTION_LIMIT:
                actions.append(ToggleCard(i))
        size = len(state.hand)
        actions.extend(MoveCard(i, j) for i in range(size) for j in range(size) if i != j)
        if state.hand:
            actions.extend((SortByRank(), SortBySuit()))
    elif phase is Phase.ROUND_EVAL:
        actions.append(CashOut())
    elif phase is Phase.SHOP:
        actions.append(EndShop())
    elif phase is Phase.WIN:
        actions.extend((ContinueEndless(), EndRun()))
    if phase in _JOKER_MOVE_PHASES:
        size = len(state.jokers)
        actions.extend(MoveJoker(i, j) for i in range(size) for j in range(size) if i != j)
    return actions


def is_legal(state: RunState, action: Action) -> bool:
    """Whether ``action`` can be applied to ``state``."""
    check = _CHECKS.get(type(action))
    return check is not None and check(state, action)


def _count_selected(state: RunState) -> int:
    return sum(card.selected for card in state.hand)


def _is_move_legal(size: int, source: int, target: int) -> bool:
    return 0 <= source < size and 0 <= target < size and source != target


def _check_toggle(state: RunState, action: ToggleCard) -> bool:
    if state.phase is not Phase.SELECTING_HAND or not 0 <= action.index < len(state.hand):
        return False
    return state.hand[action.index].selected or _count_selected(state) < SELECTION_LIMIT


def _check_play(state: RunState, _: Play) -> bool:
    return state.phase is Phase.SELECTING_HAND and 1 <= _count_selected(state) <= SELECTION_LIMIT


def _check_discard(state: RunState, _: Discard) -> bool:
    return (
        state.phase is Phase.SELECTING_HAND
        and state.discards_left > 0
        and _count_selected(state) >= 1
    )


_CHECKS: dict[type[Action], Callable[[RunState, Any], bool]] = {
    SelectBlind: lambda s, _: s.phase is Phase.BLIND_SELECT,
    ToggleCard: _check_toggle,
    MoveCard: lambda s, a: (
        s.phase is Phase.SELECTING_HAND and _is_move_legal(len(s.hand), a.source, a.target)
    ),
    SortByRank: lambda s, _: s.phase is Phase.SELECTING_HAND and bool(s.hand),
    SortBySuit: lambda s, _: s.phase is Phase.SELECTING_HAND and bool(s.hand),
    Play: _check_play,
    Discard: _check_discard,
    CashOut: lambda s, _: s.phase is Phase.ROUND_EVAL,
    EndShop: lambda s, _: s.phase is Phase.SHOP,
    ContinueEndless: lambda s, _: s.phase is Phase.WIN,
    EndRun: lambda s, _: s.phase is Phase.WIN,
    MoveJoker: lambda s, a: (
        s.phase in _JOKER_MOVE_PHASES and _is_move_legal(len(s.jokers), a.source, a.target)
    ),
}


# Applying actions.


def step(state: RunState, action: Action) -> list[Event]:
    """Apply ``action`` to ``state`` in place and return the events it caused.

    Raises:
        IllegalActionError: if the action is not legal; the state is unchanged.
    """
    if not is_legal(state, action):
        raise IllegalActionError(f"{action} is not legal in phase {state.phase.value}")
    scope = Scope(state)
    _HANDLERS[type(action)](scope, action)
    scope.flush()
    return scope.events


def determinize(state: RunState, seed: str | int) -> RunState:
    """Copy ``state``, re-randomizing everything the player cannot see.

    The copy keeps all visible information but shuffles the draw pile and uses a
    fresh random seed, so a planning agent that searches it learns nothing about
    the real run's future.
    """
    copy = clone(state)
    copy.rng = Rng(f"determinize:{seed}")
    _shuffle(copy, "determinize")
    return copy


def _toggle_card(scope: Scope, action: ToggleCard) -> None:
    card = scope.state.hand[action.index]
    card.selected = not card.selected
    scope.emit(CardToggled(card.uid, card.selected))


def _move_card(scope: Scope, action: MoveCard) -> None:
    hand = scope.state.hand
    card = hand.pop(action.source)
    hand.insert(action.target, card)
    scope.emit(CardMoved(card.uid, action.source, action.target))


def _move_joker(scope: Scope, action: MoveJoker) -> None:
    jokers = scope.state.jokers
    joker: Item = jokers.pop(action.source)
    jokers.insert(action.target, joker)
    scope.emit(JokerMoved(joker.uid, action.source, action.target))


def _sort_by_rank(scope: Scope, _: SortByRank) -> None:
    """Source: functions/button_callbacks.lua ``G.FUNCS.sort_hand_value``."""
    scope.state.hand_sort = HandSort.RANK
    _sort_hand(scope.state)
    scope.emit(HandSorted(HandSort.RANK))


def _sort_by_suit(scope: Scope, _: SortBySuit) -> None:
    """Source: functions/button_callbacks.lua ``G.FUNCS.sort_hand_suit``."""
    scope.state.hand_sort = HandSort.SUIT
    _sort_hand(scope.state)
    scope.emit(HandSorted(HandSort.SUIT))


def _sort_hand(state: RunState) -> None:
    """Source: cardarea.lua ``CardArea:sort``."""
    mode = "suit" if state.hand_sort is HandSort.SUIT else "rank"
    state.hand.sort(key=lambda card: sort_key(card, mode), reverse=True)


def _shuffle(state: RunState, key: str) -> None:
    """Shuffle the draw pile on stream ``key``.

    The game first puts the cards in creation order, so the result depends only
    on which cards are in the pile, not on their previous order.

    Source: cardarea.lua ``CardArea:shuffle``; functions/misc_functions.lua
    ``pseudoshuffle``.
    """
    state.deck.sort(key=lambda card: card.uid)
    state.rng.shuffle(key, state.deck)


def _select_blind(scope: Scope, _: SelectBlind) -> None:
    """Source: functions/button_callbacks.lua ``G.FUNCS.select_blind``."""
    state = scope.state
    slot = state.blind_on_deck
    assert slot is not None
    state.round += 1
    state.blind_states[slot] = BlindStatus.CURRENT
    _new_round(scope, slot)


def _new_round(scope: Scope, slot: BlindSlot) -> None:
    """Source: functions/state_events.lua ``new_round``."""
    state = scope.state
    state.discards_left = max(0, state.discards_per_round)
    state.hands_left = max(1, state.hands_per_round)
    state.hands_played_this_round = 0
    state.discards_used_this_round = 0
    for level in state.hands.values():
        level.played_this_round = 0
    blind = _set_blind(state, state.blind_choices[slot], slot)
    scope.emit(BlindSelected(slot, blind.key, blind.chips))
    for joker in list(state.jokers):
        calculate_joker(scope, joker, Context(Ctx.SETTING_BLIND))
    scope.flush()
    _shuffle(state, f"nr{state.ante}")
    _draw_to_hand(scope)


def _set_blind(state: RunState, key: str, slot: BlindSlot) -> BlindState:
    """Source: blind.lua ``Blind:set_blind``."""
    info = BLINDS[key]
    dollars = 0 if slot in state.no_blind_reward else info.dollars
    chips = get_blind_amount(state.ante, state.scaling) * info.mult * state.ante_scaling
    state.blind = BlindState(key=key, slot=slot, chips=chips, mult=info.mult, dollars=dollars)
    return state.blind


def _draw_to_hand(scope: Scope) -> None:
    """Draw up to the hand size, then wait for the player.

    Source: functions/state_events.lua ``G.FUNCS.draw_from_deck_to_hand``;
    game.lua ``Game:update_draw_to_hand`` and ``Game:update_selecting_hand``.
    """
    state = scope.state
    if state.hand_size <= 0 and not state.hand:
        _game_over(scope)
        return
    count = min(len(state.deck), state.hand_size - len(state.hand))
    drawn = [state.deck.pop() for _ in range(max(0, count))]
    if drawn:
        state.hand.extend(drawn)
        _sort_hand(state)
        scope.emit(CardsDrawn(tuple(card.uid for card in drawn)))
    if state.hands_played_this_round == 0 and state.discards_used_this_round == 0:
        for joker in list(state.jokers):
            calculate_joker(scope, joker, Context(Ctx.FIRST_HAND_DRAWN))
        scope.flush()
    scope.set_phase(Phase.SELECTING_HAND)
    if not state.hand and not state.deck:
        _end_round(scope)


def _play(scope: Scope, _: Play) -> None:
    """Source: functions/state_events.lua ``G.FUNCS.play_cards_from_highlighted``;
    game.lua ``Game:update_hand_played``.
    """
    state = scope.state
    played = _take_selected(state)
    state.hands_left -= 1
    for card in played:
        card.times_played += 1
        card.played_this_ante = True
    destroyed = evaluate_play(scope, played)
    state.discard.extend(card for card in played if card not in destroyed)
    state.hands_played += 1
    state.hands_played_this_round += 1
    assert state.blind is not None
    if state.chips >= state.blind.chips or state.hands_left < 1:
        _end_round(scope)
    else:
        _draw_to_hand(scope)


def _take_selected(state: RunState) -> list[PlayingCard]:
    """Remove the selected cards from the hand, keeping their left-to-right order."""
    taken = [card for card in state.hand if card.selected]
    state.hand = [card for card in state.hand if not card.selected]
    for card in taken:
        card.selected = False
    return taken


def _discard(scope: Scope, _: Discard) -> None:
    """Source: functions/state_events.lua ``G.FUNCS.discard_cards_from_highlighted``."""
    state = scope.state
    cards = _take_selected(state)
    for joker in list(state.jokers):
        calculate_joker(scope, joker, Context(Ctx.PRE_DISCARD, full_hand=cards))
    destroyed: list[PlayingCard] = []
    for card in cards:
        removed = False
        for joker in list(state.jokers):
            effect = calculate_joker(
                scope, joker, Context(Ctx.DISCARD, full_hand=cards, other_card=card)
            )
            if effect is not None and effect.remove:
                removed = True
        if removed:
            destroyed.append(card)
        else:
            card.discarded = True
            state.discard.append(card)
    scope.emit(CardsDiscarded(tuple(card.uid for card in cards)))
    for card in destroyed:
        scope.emit(CardDestroyed(card.uid, "discarded"))
    if destroyed:
        for joker in list(state.jokers):
            calculate_joker(
                scope, joker, Context(Ctx.REMOVE_PLAYING_CARDS, Area.JOKERS, removed=destroyed)
            )
    scope.flush()
    state.discards_left -= 1
    state.discards_used_this_round += 1
    _draw_to_hand(scope)


def _end_round(scope: Scope) -> None:
    """Source: functions/state_events.lua ``end_round``."""
    state = scope.state
    blind = state.blind
    assert blind is not None
    beaten = state.chips >= blind.chips
    game_over = not beaten
    for joker in list(state.jokers):
        effect = calculate_joker(
            scope, joker, Context(Ctx.END_OF_ROUND, end_of_round=True, game_over=game_over)
        )
        if effect is not None and effect.saved:
            game_over = False
    scope.flush()
    scope.emit(RoundEnded(blind=blind.key, won=beaten, score=state.chips, target=blind.chips))
    if game_over:
        _game_over(scope)
        return
    game_won = state.ante == state.win_ante and blind.slot is BlindSlot.BOSS
    state.unused_discards += state.discards_left
    for card in list(state.hand):
        _end_of_round_held_card(scope, card)
    scope.flush()

    for card in state.hand:
        card.selected = False
    state.discard.extend(state.hand)
    state.hand.clear()
    if blind.slot is BlindSlot.BOSS:
        state.ante += 1
        scope.emit(AnteChanged(state.ante))
    # Source: functions/state_events.lua G.FUNCS.draw_from_discard_to_deck.
    state.deck[:0] = state.discard
    state.discard.clear()

    state.blind_states[blind.slot] = BlindStatus.DEFEATED
    for card in state.deck:
        card.discarded = False
        if blind.slot is BlindSlot.BOSS:
            card.played_this_ante = False
    state.payouts = _evaluate_round(scope, beaten)

    if game_won and not state.won:
        state.won = True
        scope.set_phase(Phase.WIN)
    else:
        scope.set_phase(Phase.ROUND_EVAL)


def _end_of_round_held_card(scope: Scope, card: PlayingCard) -> None:
    """Effects of a card held in hand when the round ends, with retriggers."""
    state = scope.state
    reps = 1
    index = 0
    while index < reps:
        dollars, acted = end_of_round_dollars(scope, card)
        if index == 0:
            effects: list[Effect | None] = [None if not acted else Effect(dollars=dollars)]
            if red_seal_repetitions(card) and acted:
                reps += 1
            for joker in list(state.jokers):
                effect = calculate_joker(
                    scope,
                    joker,
                    Context(
                        Ctx.REPETITION,
                        Area.HAND,
                        other_card=card,
                        end_of_round=True,
                        card_effects=effects,
                    ),
                )
                if effect is not None and effect.repetitions:
                    reps += effect.repetitions
        if dollars:
            scope.change_money(dollars, "held_card")
        index += 1


def _evaluate_round(scope: Scope, beaten: bool) -> list[Payout]:
    """The money earned for the round, row by row.

    Source: functions/state_events.lua ``G.FUNCS.evaluate_round``; card.lua
    ``Card:calculate_dollar_bonus``.
    """
    state = scope.state
    assert state.blind is not None
    payouts = [Payout("blind", state.blind.dollars if beaten else 0)]
    if state.hands_left > 0:
        payouts.append(Payout("hands", state.hands_left))
    for joker in state.jokers:
        bonus = DOLLAR_BONUS.get(joker.key)
        dollars = bonus(scope, joker) if bonus is not None and not joker.debuffed else None
        if dollars:
            payouts.append(Payout(joker.key, dollars))
    if state.money >= 5:
        interest = state.interest_amount * min(state.money // 5, state.interest_cap // 5)
        payouts.append(Payout("interest", interest))
    return payouts


def _cash_out(scope: Scope, _: CashOut) -> None:
    """Source: functions/button_callbacks.lua ``G.FUNCS.cash_out``."""
    state = scope.state
    _shuffle(state, f"cashout{state.ante}")
    scope.change_money(sum(payout.dollars for payout in state.payouts), "cash_out")
    state.payouts = []
    state.discards_left = max(0, state.discards_per_round)
    state.hands_left = max(1, state.hands_per_round)
    state.chips = 0.0
    state.blind = None
    _reset_blinds(scope)
    scope.set_phase(Phase.SHOP)


def _reset_blinds(scope: Scope) -> None:
    """After the boss is beaten, start the next ante's blinds and pick its boss.

    Source: functions/common_events.lua ``reset_blinds``.
    """
    state = scope.state
    if state.blind_states[BlindSlot.BOSS] is not BlindStatus.DEFEATED:
        return
    for slot in BlindSlot:
        state.blind_states[slot] = BlindStatus.UPCOMING
    state.blind_on_deck = BlindSlot.SMALL
    boss = choose_boss(state.rng, state.ante, state.win_ante, state.bosses_used)
    state.blind_choices[BlindSlot.BOSS] = boss
    scope.emit(BossChosen(state.ante, boss))


def _end_shop(scope: Scope, _: EndShop) -> None:
    """Source: functions/button_callbacks.lua ``G.FUNCS.toggle_shop``;
    functions/UI_definitions.lua ``create_UIBox_blind_select``.
    """
    state = scope.state
    done = (BlindStatus.DEFEATED, BlindStatus.SKIPPED)
    if state.blind_states[BlindSlot.SMALL] not in done:
        slot = BlindSlot.SMALL
    elif state.blind_states[BlindSlot.BIG] not in done:
        slot = BlindSlot.BIG
    else:
        slot = BlindSlot.BOSS
    state.blind_on_deck = slot
    state.blind_states[slot] = BlindStatus.SELECT
    scope.set_phase(Phase.BLIND_SELECT)


def _continue_endless(scope: Scope, _: ContinueEndless) -> None:
    scope.set_phase(Phase.ROUND_EVAL)


def _end_run(scope: Scope, _: EndRun) -> None:
    scope.set_phase(Phase.GAME_OVER)
    state = scope.state
    scope.emit(RunEnded(won=True, ante=state.ante, round=state.round))


def _game_over(scope: Scope) -> None:
    scope.set_phase(Phase.GAME_OVER)
    state = scope.state
    scope.emit(RunEnded(won=False, ante=state.ante, round=state.round))


_HANDLERS: dict[type[Action], Callable[[Scope, Any], None]] = {
    SelectBlind: _select_blind,
    ToggleCard: _toggle_card,
    MoveCard: _move_card,
    MoveJoker: _move_joker,
    SortByRank: _sort_by_rank,
    SortBySuit: _sort_by_suit,
    Play: _play,
    Discard: _discard,
    CashOut: _cash_out,
    EndShop: _end_shop,
    ContinueEndless: _continue_endless,
    EndRun: _end_run,
}
