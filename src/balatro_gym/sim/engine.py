"""Game rules: starting a run, listing legal actions and applying them.

The engine is deterministic: the same state and action always give the same
result, because all randomness comes from the state's own :class:`Rng`. Applying
an action changes the state in place and returns the events it caused; copy a
state first with ``serialize.clone`` to keep the original (ADR 0001).

The game runs its effects through an animation queue. Here each action runs to
completion synchronously, in the same order as the game's queued events.

Implemented so far: a run starts at blind selection; a round is played with
toggle, move, sort, play and discard and scored with jokers and card modifiers;
the round is evaluated and cashed out; the shop sells jokers, consumables,
playing cards, vouchers and booster packs, which open into the pack phase; the
boss of each ante is chosen by the game's rule, can be rerolled with a voucher,
and applies its effect; and the run is won after the boss of ante 8, with the
choice to continue in endless mode.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .actions import (
    Action,
    Buy,
    BuyAndUse,
    BuyPack,
    CashOut,
    ContinueEndless,
    Discard,
    EndRun,
    EndShop,
    MoveCard,
    MoveJoker,
    Pick,
    Play,
    RedeemVoucher,
    Reroll,
    RerollBoss,
    SelectBlind,
    SellConsumable,
    SellJoker,
    SkipPack,
    SortByRank,
    SortBySuit,
    ToggleCard,
    UseConsumable,
)
from .blinds import BLINDS, choose_boss, get_blind_amount
from .bosses import (
    defeat_blind,
    draw_count,
    drawn_to_hand,
    most_played_hand,
    press_play,
    start_blind,
)
from .cards import sort_key
from .consumables import can_use, consumable_room, joker_room, use
from .content import CENTERS
from .effects import Area, Context, ContextKind, Effect, calculate_joker
from .events import (
    AnteChanged,
    BlindSelected,
    BossChosen,
    CardDestroyed,
    CardMoved,
    CardsDiscarded,
    CardToggled,
    ConsumableUsed,
    Event,
    HandSorted,
    ItemBought,
    ItemCreated,
    ItemSold,
    JokerMoved,
    PackPicked,
    PackSkipped,
    RoundEnded,
    RunEnded,
    ShopRerolled,
    VoucherRedeemed,
)
from .hands import initial_hand_levels
from .inventory import (
    add_playing_cards,
    add_to_deck,
    calculate_reroll_cost,
    disable_blind,
    draw_cards,
    remove_from_deck,
    set_joker_debuff,
)
from .items import new_item
from .jokers import DOLLAR_BONUS
from .modifiers import end_of_round_dollars, red_seal_repetitions
from .packs import close_pack, fill_boosters, finish_pick, open_pack
from .rng import Rng
from .scope import Scope
from .scoring import evaluate_play
from .serialize import clone
from .shop import create_card, fill_shop, get_next_voucher_key
from .state import (
    BlindSlot,
    BlindState,
    BlindStatus,
    HandSort,
    HandType,
    Item,
    Payout,
    Phase,
    PlayingCard,
    RunConfig,
    RunState,
    Suit,
)
from .vouchers import can_reroll_boss, redeem, reroll_boss

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
    modifiers: dict[str, bool] = {}
    if config.stake >= 2:
        no_blind_reward.append(BlindSlot.SMALL)
    if config.stake >= 3:
        scaling = 2
    if config.stake >= 4:
        modifiers["eternals_in_shop"] = True
    if config.stake >= 5:
        params["discards"] -= 1
    if config.stake >= 6:
        scaling = 3
    if config.stake >= 7:
        modifiers["perishables_in_shop"] = True
    if config.stake >= 8:
        modifiers["rentals_in_shop"] = True
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
        most_played_hand=HandType.HIGH_CARD,
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
        bankrupt_at=0,
        joker_rate=20,
        tarot_rate=4,
        planet_rate=4,
        spectral_rate=0,
        playing_card_rate=0,
        edition_rate=1,
        shop_joker_max=2,
        base_reroll_cost=5,
        reroll_cost=5,
        reroll_cost_increase=0,
        free_rerolls=0,
        shop_cards=[],
        shop_boosters=[],
        first_shop_buffoon=False,
        pack=None,
        ecto_minus=1,
        used_vouchers=[],
        shop_voucher=None,
        shop_vouchers=[],
        boss_rerolled=False,
        last_tarot_planet=None,
        consumable_usage={},
        modifiers=modifiers,
        perishable_rounds=5,
        rental_rate=3,
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
    state.shop_voucher = get_next_voucher_key(Scope(state))
    return state


# Legal actions.


def legal_actions(state: RunState) -> list[Action]:
    """Every action that is legal in ``state``, in a fixed order."""
    actions: list[Action] = []
    phase = state.phase
    if phase is Phase.BLIND_SELECT:
        actions.append(SelectBlind())
        if can_reroll_boss(state):
            actions.append(RerollBoss())
    elif phase is Phase.SELECTING_HAND:
        selected = _count_selected(state)
        if 1 <= selected <= SELECTION_LIMIT:
            actions.append(Play())
        if selected >= 1 and state.discards_left > 0:
            actions.append(Discard())
        _hand_actions(state, actions)
        if state.hand:
            actions.extend((SortByRank(), SortBySuit()))
    elif phase is Phase.ROUND_EVAL:
        actions.append(CashOut())
    elif phase is Phase.SHOP:
        actions.append(EndShop())
        if _check_reroll(state, Reroll()):
            actions.append(Reroll())
        for i in range(len(state.shop_cards)):
            for action in (Buy(i), BuyAndUse(i)):
                if is_legal(state, action):
                    actions.append(action)
        actions.extend(
            RedeemVoucher(i)
            for i, voucher in enumerate(state.shop_vouchers)
            if _can_redeem(state, voucher)
        )
        actions.extend(
            BuyPack(i)
            for i, pack in enumerate(state.shop_boosters)
            if _affordable(state, pack.cost)
        )
    elif phase is Phase.PACK_OPEN:
        assert state.pack is not None
        actions.extend(Pick(i) for i in range(state.pack.size) if _can_pick(state, i))
        # The game greys out Skip only while the pack's hand is being drawn.
        actions.append(SkipPack())
        _hand_actions(state, actions)
    elif phase is Phase.WIN:
        if state.config.endless:
            actions.append(ContinueEndless())
        actions.append(EndRun())
    if phase in _JOKER_MOVE_PHASES:
        size = len(state.jokers)
        actions.extend(
            MoveJoker(source, target)
            for source in range(size)
            for target in range(size)
            if source != target
        )
        actions.extend(
            SellJoker(index) for index, joker in enumerate(state.jokers) if not joker.eternal
        )
        actions.extend(SellConsumable(index) for index in range(len(state.consumables)))
        actions.extend(
            UseConsumable(index)
            for index, item in enumerate(state.consumables)
            if can_use(state, item, "consumables")
        )
    return actions


def _hand_actions(state: RunState, actions: list[Action]) -> None:
    """Selecting and moving cards in hand, wherever the hand is shown."""
    selected = _count_selected(state)
    for i, card in enumerate(state.hand):
        if _can_toggle(card, selected):
            actions.append(ToggleCard(i))
    size = len(state.hand)
    actions.extend(
        MoveCard(source, target)
        for source in range(size)
        for target in range(size)
        if source != target
    )


def is_legal(state: RunState, action: Action) -> bool:
    """Whether ``action`` can be applied to ``state``."""
    check = _CHECKS.get(type(action))
    return check is not None and check(state, action)


def _count_selected(state: RunState) -> int:
    return sum(card.selected for card in state.hand)


def _is_move_legal(size: int, source: int, target: int) -> bool:
    return 0 <= source < size and 0 <= target < size and source != target


def _hand_shown(state: RunState) -> bool:
    """Whether the hand can be worked with: while playing, or in an Arcana or Spectral pack."""
    return state.phase is Phase.SELECTING_HAND or (
        state.phase is Phase.PACK_OPEN and bool(state.hand)
    )


def _can_toggle(card: PlayingCard, selected: int) -> bool:
    """A selected card can be deselected unless Cerulean Bell forces it; a card can
    be selected while fewer than five are. Source: cardarea.lua
    ``CardArea:remove_from_highlighted``."""
    if card.selected:
        return not card.forced_selection
    return selected < SELECTION_LIMIT


def _check_toggle(state: RunState, action: ToggleCard) -> bool:
    if not _hand_shown(state) or not 0 <= action.index < len(state.hand):
        return False
    return _can_toggle(state.hand[action.index], _count_selected(state))


def _check_play(state: RunState, _: Play) -> bool:
    return state.phase is Phase.SELECTING_HAND and 1 <= _count_selected(state) <= SELECTION_LIMIT


def _check_discard(state: RunState, _: Discard) -> bool:
    return (
        state.phase is Phase.SELECTING_HAND
        and state.discards_left > 0
        and _count_selected(state) >= 1
    )


def _affordable(state: RunState, cost: int) -> bool:
    """Source: functions/button_callbacks.lua ``G.FUNCS.can_buy``."""
    return cost <= state.money - state.bankrupt_at or cost <= 0


def _check_buy(state: RunState, action: Buy) -> bool:
    if state.phase is not Phase.SHOP or not 0 <= action.index < len(state.shop_cards):
        return False
    item = state.shop_cards[action.index]
    if not _affordable(state, item.cost):
        return False
    if item.playing_card is not None:
        return True
    bonus = 1 if item.edition == "e_negative" else 0
    if CENTERS[item.key].is_joker:
        return joker_room(state) + bonus > 0
    return consumable_room(state) + bonus > 0


def _check_buy_and_use(state: RunState, action: BuyAndUse) -> bool:
    if state.phase is not Phase.SHOP or not 0 <= action.index < len(state.shop_cards):
        return False
    item = state.shop_cards[action.index]
    return (
        CENTERS[item.key].is_consumable
        and _affordable(state, item.cost)
        and can_use(state, item, "shop")
    )


def _check_reroll(state: RunState, _: Reroll) -> bool:
    """Source: functions/button_callbacks.lua ``G.FUNCS.can_reroll``."""
    if state.phase is not Phase.SHOP:
        return False
    return state.money - state.bankrupt_at - state.reroll_cost >= 0 or state.reroll_cost == 0


def _check_sell_joker(state: RunState, action: SellJoker) -> bool:
    return (
        state.phase in _JOKER_MOVE_PHASES
        and 0 <= action.index < len(state.jokers)
        and not state.jokers[action.index].eternal
    )


def _can_redeem(state: RunState, voucher: Item) -> bool:
    """Source: functions/button_callbacks.lua ``G.FUNCS.can_redeem``."""
    return voucher.cost <= state.money - state.bankrupt_at


def _check_redeem(state: RunState, action: RedeemVoucher) -> bool:
    return (
        state.phase is Phase.SHOP
        and 0 <= action.index < len(state.shop_vouchers)
        and _can_redeem(state, state.shop_vouchers[action.index])
    )


def _check_buy_pack(state: RunState, action: BuyPack) -> bool:
    """Source: functions/button_callbacks.lua ``G.FUNCS.can_open``."""
    return (
        state.phase is Phase.SHOP
        and 0 <= action.index < len(state.shop_boosters)
        and _affordable(state, state.shop_boosters[action.index].cost)
    )


def _can_pick(state: RunState, index: int) -> bool:
    """Whether the card at ``index`` in the open pack can be taken.

    A consumable must be usable; a joker needs a free slot unless it is Negative.
    Source: functions/button_callbacks.lua ``G.FUNCS.can_use_consumeable`` and
    ``G.FUNCS.can_select_card``.
    """
    pack = state.pack
    if pack is None or not 0 <= index < pack.size:
        return False
    if pack.playing_cards:
        return True
    item = pack.items[index]
    if CENTERS[item.key].is_consumable:
        return can_use(state, item, "pack")
    return item.edition == "e_negative" or joker_room(state) > 0


def _check_use(state: RunState, action: UseConsumable) -> bool:
    return (
        state.phase in _JOKER_MOVE_PHASES
        and 0 <= action.index < len(state.consumables)
        and can_use(state, state.consumables[action.index], "consumables")
    )


_CHECKS: dict[type[Action], Callable[[RunState, Any], bool]] = {
    SelectBlind: lambda state, _: state.phase is Phase.BLIND_SELECT,
    RerollBoss: lambda state, _: state.phase is Phase.BLIND_SELECT and can_reroll_boss(state),
    ToggleCard: _check_toggle,
    MoveCard: lambda state, action: (
        _hand_shown(state) and _is_move_legal(len(state.hand), action.source, action.target)
    ),
    SortByRank: lambda state, _: state.phase is Phase.SELECTING_HAND and bool(state.hand),
    SortBySuit: lambda state, _: state.phase is Phase.SELECTING_HAND and bool(state.hand),
    Play: _check_play,
    Discard: _check_discard,
    CashOut: lambda state, _: state.phase is Phase.ROUND_EVAL,
    EndShop: lambda state, _: state.phase is Phase.SHOP,
    ContinueEndless: lambda state, _: state.phase is Phase.WIN and state.config.endless,
    Buy: _check_buy,
    BuyAndUse: _check_buy_and_use,
    Reroll: _check_reroll,
    RedeemVoucher: _check_redeem,
    BuyPack: _check_buy_pack,
    Pick: lambda state, action: state.phase is Phase.PACK_OPEN and _can_pick(state, action.index),
    SkipPack: lambda state, _: state.phase is Phase.PACK_OPEN,
    SellJoker: _check_sell_joker,
    SellConsumable: lambda state, action: (
        state.phase in _JOKER_MOVE_PHASES and 0 <= action.index < len(state.consumables)
    ),
    UseConsumable: _check_use,
    EndRun: lambda state, _: state.phase is Phase.WIN,
    MoveJoker: lambda state, action: (
        state.phase in _JOKER_MOVE_PHASES
        and _is_move_legal(len(state.jokers), action.source, action.target)
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

    The copy keeps all visible information but shuffles the draw pile, deals the
    face-down cards in hand again from the unseen cards, shuffles the jokers that
    are face down, and uses a fresh random seed, so a planning agent that searches
    it learns nothing about the real run's future.
    """
    copy = clone(state)
    copy.rng = Rng(f"determinize:{seed}")
    slots = [i for i, card in enumerate(copy.hand) if card.face_down]
    unseen = [*copy.deck, *(copy.hand[i] for i in slots)]
    unseen.sort(key=lambda card: card.uid)
    copy.rng.shuffle("determinize", unseen)
    for i in slots:
        old = copy.hand[i]
        new = unseen.pop()
        new.face_down, new.selected, new.forced_selection = True, old.selected, old.forced_selection
        copy.hand[i] = new
    for card in unseen:
        card.face_down = card.selected = card.forced_selection = False
    copy.deck = unseen
    hidden_jokers = [i for i, joker in enumerate(copy.jokers) if joker.face_down]
    order = [copy.jokers[i] for i in hidden_jokers]
    copy.rng.shuffle("determinize_jokers", order)
    for i, joker in zip(hidden_jokers, order, strict=True):
        copy.jokers[i] = joker
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
    state.reroll_cost_increase = 0
    for level in state.hands.values():
        level.played_this_round = 0
    state.free_rerolls = sum(
        joker.key == "j_chaos" and not joker.debuffed for joker in state.jokers
    )
    calculate_reroll_cost(scope, skip_increment=True)
    blind = _set_blind(state, state.blind_choices[slot], slot)
    scope.emit(BlindSelected(slot, blind.key, blind.chips))
    start_blind(scope)
    for joker in list(state.jokers):
        calculate_joker(scope, joker, Context(ContextKind.SETTING_BLIND))
    # The shuffle is queued after the jokers' reactions, as in the game.
    scope.defer(lambda: _shuffle(state, f"nr{state.ante}"))
    scope.flush()
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
    draw_cards(scope, draw_count(state))
    if state.hands_played_this_round == 0 and state.discards_used_this_round == 0:
        for joker in list(state.jokers):
            calculate_joker(scope, joker, Context(ContextKind.FIRST_HAND_DRAWN))
        scope.flush()
    scope.set_phase(Phase.SELECTING_HAND)
    drawn_to_hand(scope)
    scope.flush()
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
    hooked = press_play(scope, played)
    if hooked:
        _discard_cards(scope, hooked)
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
    """Remove the selected cards from the hand, keeping their left-to-right order.

    Playing or discarding ends Cerulean Bell's forced selection.
    """
    for card in state.playing_cards():
        card.forced_selection = False
    return _take(state, [card for card in state.hand if card.selected])


def _take(state: RunState, cards: list[PlayingCard]) -> list[PlayingCard]:
    """Remove ``cards`` from the hand; they leave it face up and unselected."""
    leaving = {id(card) for card in cards}
    state.hand = [card for card in state.hand if id(card) not in leaving]
    for card in cards:
        card.selected = False
        card.face_down = False
    return cards


def _discard(scope: Scope, _: Discard) -> None:
    state = scope.state
    _discard_cards(scope, _take_selected(state))
    state.discards_left -= 1
    state.discards_used_this_round += 1
    _draw_to_hand(scope)


def _discard_cards(scope: Scope, cards: list[PlayingCard]) -> None:
    """Discard ``cards`` from the hand: by the player, or by The Hook, which uses
    no discard. Source: functions/state_events.lua
    ``G.FUNCS.discard_cards_from_highlighted``."""
    state = scope.state
    _take(state, cards)
    for joker in list(state.jokers):
        calculate_joker(scope, joker, Context(ContextKind.PRE_DISCARD, full_hand=cards))
    destroyed: list[PlayingCard] = []
    for card in cards:
        _purple_seal(scope, card)
        removed = False
        for joker in list(state.jokers):
            effect = calculate_joker(
                scope, joker, Context(ContextKind.DISCARD, full_hand=cards, other_card=card)
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
                scope,
                joker,
                Context(ContextKind.REMOVE_PLAYING_CARDS, Area.JOKERS, removed=destroyed),
            )
    scope.flush()


def _purple_seal(scope: Scope, card: PlayingCard) -> None:
    """A discarded Purple Seal card creates a Tarot card, if there is room.

    Source: card.lua ``Card:calculate_seal`` (``context.discard``).
    """
    if card.seal != "Purple" or card.debuffed:
        return
    if consumable_room(scope.state) - scope.consumable_buffer <= 0:
        return
    scope.consumable_buffer += 1

    def create() -> None:
        item = create_card(scope, "Tarot", "consumables", key_append="8ba")
        scope.state.consumables.append(item)
        scope.emit(ItemCreated(item.uid, item.key, "consumables"))
        scope.consumable_buffer = 0

    scope.defer(create)


def _apply_stickers(scope: Scope, joker: Item) -> None:
    """Rental jokers cost money and perishable jokers wear out at the end of a round.

    Source: card.lua ``Card:calculate_rental`` and ``Card:calculate_perishable``.
    """
    state = scope.state
    if joker.rental:
        scope.change_money(-state.rental_rate, "rental")
    if joker.perishable and joker.perish_tally > 0:
        joker.perish_tally -= 1
        if joker.perish_tally == 0:
            set_joker_debuff(scope, joker, True)


def _end_round(scope: Scope) -> None:
    """Source: functions/state_events.lua ``end_round``."""
    state = scope.state
    blind = state.blind
    assert blind is not None
    beaten = state.chips >= blind.chips
    game_over = not beaten
    for joker in list(state.jokers):
        effect = calculate_joker(
            scope, joker, Context(ContextKind.END_OF_ROUND, end_of_round=True, game_over=game_over)
        )
        if effect is not None and effect.saved:
            game_over = False
        _apply_stickers(scope, joker)
    scope.flush()
    scope.emit(RoundEnded(blind=blind.key, won=beaten, score=state.chips, target=blind.chips))
    if game_over:
        _game_over(scope)
        return
    game_won = state.ante == state.win_ante and blind.slot is BlindSlot.BOSS
    state.unused_discards += state.discards_left
    if BLINDS[blind.key].boss:
        state.most_played_hand = most_played_hand(state)
    for card in list(state.hand):
        _end_of_round_held_card(scope, card)
    scope.flush()

    for card in state.playing_cards():
        card.forced_selection = False
    state.discard.extend(_take(state, list(state.hand)))
    if blind.slot is BlindSlot.BOSS:
        state.ante += 1
        scope.emit(AnteChanged(state.ante))
        state.shop_voucher = get_next_voucher_key(scope)
    # Source: functions/state_events.lua G.FUNCS.draw_from_discard_to_deck.
    state.deck[:0] = state.discard
    state.discard.clear()

    state.blind_states[blind.slot] = BlindStatus.DEFEATED
    for card in state.deck:
        card.discarded = False
        if blind.slot is BlindSlot.BOSS:
            card.played_this_ante = False
    state.payouts = _evaluate_round(scope, beaten)
    defeat_blind(scope)

    if game_won and not state.won:
        state.won = True
        scope.set_phase(Phase.WIN)
    else:
        scope.set_phase(Phase.ROUND_EVAL)
    max_ante = state.config.max_ante
    if max_ante is not None and blind.slot is BlindSlot.BOSS and state.ante > max_ante:
        _game_over(scope, reason="max_ante")


def _end_of_round_held_card(scope: Scope, card: PlayingCard) -> None:
    """Effects of a card held in hand when the round ends, with retriggers."""
    state = scope.state
    passes = 1
    index = 0
    while index < passes:
        dollars, acted = end_of_round_dollars(scope, card)
        if index == 0:
            effects: list[Effect | None] = [None if not acted else Effect(dollars=dollars)]
            if red_seal_repetitions(card) and acted:
                passes += 1
            for joker in list(state.jokers):
                effect = calculate_joker(
                    scope,
                    joker,
                    Context(
                        ContextKind.REPETITION,
                        Area.HAND,
                        other_card=card,
                        end_of_round=True,
                        card_effects=effects,
                    ),
                )
                if effect is not None and effect.repetitions:
                    passes += effect.repetitions
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
    fill_shop(scope)
    if state.shop_voucher is not None:
        state.shop_vouchers.append(new_item(state, state.shop_voucher))
    fill_boosters(scope)
    for item in (*state.shop_cards, *state.shop_vouchers, *state.shop_boosters):
        scope.emit(ItemCreated(item.uid, item.key, "shop"))


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
    state.boss_rerolled = False
    scope.emit(BossChosen(state.ante, boss))


def _end_shop(scope: Scope, _: EndShop) -> None:
    """Source: functions/button_callbacks.lua ``G.FUNCS.toggle_shop``;
    functions/UI_definitions.lua ``create_UIBox_blind_select``.
    """
    state = scope.state
    for joker in list(state.jokers):
        calculate_joker(scope, joker, Context(ContextKind.ENDING_SHOP))
    scope.flush()
    state.shop_cards.clear()
    state.shop_vouchers.clear()
    state.shop_boosters.clear()
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
    scope.emit(RunEnded(won=True, ante=state.ante, round=state.round, reason="won"))


def _game_over(scope: Scope, *, reason: str = "lost") -> None:
    scope.set_phase(Phase.GAME_OVER)
    state = scope.state
    scope.emit(RunEnded(won=state.won, ante=state.ante, round=state.round, reason=reason))


def _buy(scope: Scope, action: Buy) -> None:
    """Source: functions/button_callbacks.lua ``G.FUNCS.buy_from_shop``."""
    state = scope.state
    item = state.shop_cards.pop(action.index)
    if item.playing_card is not None:
        _buy_playing_card(scope, item, item.playing_card)
        return
    add_to_deck(scope, item)
    if CENTERS[item.key].is_joker:
        state.jokers.append(item)
    else:
        state.consumables.append(item)
    scope.emit(ItemBought(item.uid, item.key, item.cost))
    calculate_joker(scope, item, Context(ContextKind.BUYING_CARD, card=item))
    for joker in list(state.jokers):
        calculate_joker(scope, joker, Context(ContextKind.BUYING_CARD, card=item))
    scope.change_money(-item.cost, "buy")


def _buy_playing_card(scope: Scope, item: Item, card: PlayingCard) -> None:
    """A bought playing card goes under the deck."""
    state = scope.state
    scope.emit(ItemBought(item.uid, item.key, item.cost))
    state.deck.insert(0, card)
    add_playing_cards(scope, [card], "deck")
    for joker in list(state.jokers):
        calculate_joker(scope, joker, Context(ContextKind.BUYING_CARD, card=item))
    scope.change_money(-item.cost, "buy")


def _redeem_voucher(scope: Scope, action: RedeemVoucher) -> None:
    """Source: card.lua ``Card:redeem``."""
    voucher = scope.state.shop_vouchers.pop(action.index)
    scope.emit(VoucherRedeemed(voucher.uid, voucher.key, voucher.cost))
    redeem(scope, voucher)


def _reroll_boss(scope: Scope, _: RerollBoss) -> None:
    """Source: functions/button_callbacks.lua ``G.FUNCS.reroll_boss``."""
    reroll_boss(scope)


def _buy_and_use(scope: Scope, action: BuyAndUse) -> None:
    state = scope.state
    item = state.shop_cards.pop(action.index)
    scope.emit(ItemBought(item.uid, item.key, item.cost))
    for joker in list(state.jokers):
        calculate_joker(scope, joker, Context(ContextKind.BUYING_CARD, card=item))
    scope.change_money(-item.cost, "buy")
    _use(scope, item)


def _use_consumable(scope: Scope, action: UseConsumable) -> None:
    """Source: functions/button_callbacks.lua ``G.FUNCS.use_card``."""
    state = scope.state
    item = state.consumables.pop(action.index)
    _use(scope, item)
    # A boss disabled mid-round (Chicot from The Soul) may lower the target below
    # the score, which ends the round. Source: blind.lua Blind:disable.
    blind = state.blind
    if state.phase is Phase.SELECTING_HAND and blind is not None and state.chips >= blind.chips:
        _end_round(scope)


def _use(scope: Scope, item: Item) -> None:
    state = scope.state
    targets = tuple(card.uid for card in state.hand if card.selected)
    before = {i.uid for i in (*state.jokers, *state.consumables)}
    use(scope, item)
    scope.emit(ConsumableUsed(item.uid, item.key, targets))
    for new in (*state.jokers, *state.consumables):
        if new.uid not in before:
            area = "jokers" if new in state.jokers else "consumables"
            scope.emit(ItemCreated(new.uid, new.key, area))
            if area == "jokers":
                add_to_deck(scope, new)
    for joker in list(state.jokers):
        calculate_joker(scope, joker, Context(ContextKind.USING_CONSUMABLE, card=item))
    scope.flush()


def _buy_pack(scope: Scope, action: BuyPack) -> None:
    """Source: functions/button_callbacks.lua ``G.FUNCS.use_card`` (a booster)."""
    pack = scope.state.shop_boosters.pop(action.index)
    scope.emit(ItemBought(pack.uid, pack.key, pack.cost))
    open_pack(scope, pack, Phase.SHOP)


def _pick(scope: Scope, action: Pick) -> None:
    """Take a card from the open pack: use a consumable, or keep a joker or card.

    Source: functions/button_callbacks.lua ``G.FUNCS.use_card``.
    """
    state = scope.state
    pack = state.pack
    assert pack is not None
    if pack.playing_cards:
        card = pack.playing_cards.pop(action.index)
        scope.emit(PackPicked(card.uid, card.label))
        state.deck.insert(0, card)
        add_playing_cards(scope, [card], "deck")
    else:
        item = pack.items.pop(action.index)
        scope.emit(PackPicked(item.uid, item.key))
        if CENTERS[item.key].is_consumable:
            _use(scope, item)
        else:
            add_to_deck(scope, item)
            state.jokers.append(item)
    scope.flush()
    finish_pick(scope)


def _skip_pack(scope: Scope, _: SkipPack) -> None:
    """Source: functions/button_callbacks.lua ``G.FUNCS.skip_booster``."""
    state = scope.state
    for joker in list(state.jokers):
        calculate_joker(scope, joker, Context(ContextKind.SKIPPING_BOOSTER))
    scope.flush()
    scope.emit(PackSkipped())
    close_pack(scope)


def _sell_joker(scope: Scope, action: SellJoker) -> None:
    _sell(scope, scope.state.jokers, action.index)


def _sell_consumable(scope: Scope, action: SellConsumable) -> None:
    _sell(scope, scope.state.consumables, action.index)


def _sell(scope: Scope, area: list[Item], index: int) -> None:
    """Source: card.lua ``Card:sell_card``; button_callbacks.lua ``G.FUNCS.sell_card``."""
    state = scope.state
    item = area[index]
    calculate_joker(scope, item, Context(ContextKind.SELLING_SELF, card=item))
    for joker in list(state.jokers):
        if joker is not item:
            calculate_joker(scope, joker, Context(ContextKind.SELLING_CARD, card=item))
    scope.change_money(item.sell_value, "sell")
    area.remove(item)
    remove_from_deck(scope, item)
    scope.emit(ItemSold(item.uid, item.key, item.sell_value))
    # Selling a joker disables Verdant Leaf.
    blind = state.blind
    verdant_leaf = blind is not None and blind.key == "bl_final_leaf" and not blind.disabled
    if verdant_leaf and CENTERS[item.key].is_joker:
        disable_blind(scope)
    scope.flush()


def _reroll(scope: Scope, _: Reroll) -> None:
    """Source: functions/button_callbacks.lua ``G.FUNCS.reroll_shop``."""
    state = scope.state
    cost = state.reroll_cost
    if cost > 0:
        scope.change_money(-cost, "reroll")
    final_free = state.free_rerolls > 0
    state.free_rerolls = max(state.free_rerolls - 1, 0)
    calculate_reroll_cost(scope, skip_increment=final_free)
    state.shop_cards.clear()
    fill_shop(scope)
    scope.emit(ShopRerolled(cost))
    for item in state.shop_cards:
        scope.emit(ItemCreated(item.uid, item.key, "shop"))
    for joker in list(state.jokers):
        calculate_joker(scope, joker, Context(ContextKind.REROLL_SHOP))
    scope.flush()


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
    Buy: _buy,
    BuyAndUse: _buy_and_use,
    UseConsumable: _use_consumable,
    SellJoker: _sell_joker,
    SellConsumable: _sell_consumable,
    Reroll: _reroll,
    RedeemVoucher: _redeem_voucher,
    RerollBoss: _reroll_boss,
    BuyPack: _buy_pack,
    Pick: _pick,
    SkipPack: _skip_pack,
}
