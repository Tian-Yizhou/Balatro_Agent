"""Scoring a played hand: a step-by-step port of the game's ``evaluate_play``.

1. Find the poker hand and the scoring cards.
2. Jokers react ``before`` scoring (some level up the hand).
3. Each scoring card scores from left to right, once per retrigger: its own chips,
   mult, x-mult, money and edition, then each joker's ``individual`` effect.
4. Each card held in hand applies its effects, with retriggers.
5. Each joker applies its edition, its main effect and other jokers' reactions
   to it (``other_joker``), from left to right.
6. Cards are destroyed where needed (Glass Card, some jokers).
7. The score is added, and jokers react ``after`` scoring.

Source: functions/state_events.lua ``G.FUNCS.evaluate_play``;
functions/common_events.lua ``eval_card`` and ``level_up_hand``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .blinds import lua_floor
from .cards import STONE
from .effects import Area, Context, Ctx, Effect, calculate_joker, find_joker
from .events import (
    CardDestroyed,
    CardRetriggered,
    EffectApplied,
    HandLevelChanged,
    HandPlayed,
    HandScored,
)
from .hands import HAND_BASE, PokerHands, evaluate_poker_hand
from .modifiers import (
    chip_bonus,
    chip_h_mult,
    chip_h_x_mult,
    chip_mult,
    chip_x_mult,
    edition_effect,
    p_dollars,
    red_seal_repetitions,
)
from .scope import Scope
from .state import HandType, Item, PlayingCard

__all__ = ["GLASS", "evaluate_play", "level_up_hand", "poker_hands"]

GLASS = "m_glass"


def poker_hands(scope: Scope, cards: list[PlayingCard]) -> PokerHands:
    """Evaluate cards with the rule changes from the player's jokers."""
    state = scope.state
    return evaluate_poker_hand(
        cards,
        four_fingers=bool(find_joker(state, "j_four_fingers")),
        shortcut=bool(find_joker(state, "j_shortcut")),
        smeared=bool(find_joker(state, "j_smeared")),
    )


def level_up_hand(scope: Scope, hand: HandType, amount: int = 1) -> None:
    """Change a poker hand's level by ``amount`` and recompute its chips and mult."""
    level = scope.state.hands[hand]
    base = HAND_BASE[hand]
    level.level = max(0, level.level + amount)
    level.mult = max(base.mult + base.level_mult * (level.level - 1), 1)
    level.chips = max(base.chips + base.level_chips * (level.level - 1), 0)
    scope.emit(HandLevelChanged(hand, level.level))


class _ContextFactory(Protocol):
    def __call__(self, kind: Ctx, area: Area | None = Area.JOKERS, **fields: object) -> Context: ...


@dataclass(slots=True)
class _Score:
    chips: float
    mult: float


def evaluate_play(scope: Scope, played: list[PlayingCard]) -> list[PlayingCard]:
    """Score ``played`` (in play order) and return the cards it destroyed."""
    state = scope.state
    hands = poker_hands(scope, played)
    hand = hands.top
    assert hand is not None
    level = state.hands[hand]
    level.played += 1
    level.played_this_round += 1
    level.visible = True
    state.last_hand_played = hand

    if find_joker(state, "j_splash"):
        positions = set(range(len(played)))
    else:
        positions = set(hands.scoring)
        positions.update(i for i, card in enumerate(played) if card.enhancement == STONE)
    scoring = [played[i] for i in sorted(positions)]
    scope.emit(
        HandPlayed(
            uids=tuple(card.uid for card in played),
            hand=hand,
            scoring=tuple(card.uid for card in scoring),
        )
    )

    def context(kind: Ctx, area: Area | None = Area.JOKERS, **fields: object) -> Context:
        ctx = Context(
            kind,
            area=area,
            full_hand=played,
            scoring_hand=scoring,
            scoring_name=hand,
            poker_hands=hands,
        )
        for name, field_value in fields.items():
            setattr(ctx, name, field_value)
        return ctx

    destroyed: list[PlayingCard] = []
    for joker in list(state.jokers):
        effect = calculate_joker(scope, joker, context(Ctx.BEFORE))
        if effect is not None and effect.level_up:
            level_up_hand(scope, hand, effect.level_up)

    score = _Score(chips=float(level.chips), mult=float(level.mult))
    for card in scoring:
        if not card.debuffed:
            _score_played_card(scope, card, score, context)
    for card in list(state.hand):
        _score_held_card(scope, card, score, context)
    for joker in list(state.jokers):
        _score_joker(scope, joker, score, context)

    for card in scoring:
        is_destroyed = False
        for joker in list(state.jokers):
            effect = calculate_joker(scope, joker, context(Ctx.DESTROYING_CARD, other_card=card))
            if effect is not None and effect.remove:
                is_destroyed = True
                break
        glass_odds = 4
        if card.enhancement == GLASS and not card.debuffed and scope.chance("glass", glass_odds):
            is_destroyed = True
        if is_destroyed:
            destroyed.append(card)
    for joker in list(state.jokers):
        calculate_joker(scope, joker, context(Ctx.REMOVE_PLAYING_CARDS, removed=destroyed))
    for card in destroyed:
        reason = "shattered" if card.enhancement == GLASS else "destroyed"
        scope.emit(CardDestroyed(card.uid, reason))

    total = lua_floor(score.chips * score.mult)
    state.chips += total
    scope.emit(
        HandScored(hand=hand, chips=score.chips, mult=score.mult, score=total, total=state.chips)
    )
    scope.flush()
    for joker in list(state.jokers):
        calculate_joker(scope, joker, context(Ctx.AFTER))
    scope.flush()
    return destroyed


def _score_played_card(
    scope: Scope, card: PlayingCard, score: _Score, context: _ContextFactory
) -> None:
    state = scope.state
    retriggers: list[Effect] = [Effect()] * red_seal_repetitions(card)
    for joker in list(state.jokers):
        effect = calculate_joker(scope, joker, context(Ctx.REPETITION, Area.PLAY, other_card=card))
        if effect is not None and effect.repetitions:
            retriggers.extend([effect] * effect.repetitions)

    for rep in [None, *retriggers]:
        if rep is not None:
            source = rep.via or rep.source
            scope.emit(
                CardRetriggered(card.uid, source.uid if source else card.uid, _key(rep, "seal"))
            )
        own, lucky = _evaluate_played_card(scope, card)
        effects: list[Effect] = [own]
        for joker in list(state.jokers):
            effect = calculate_joker(
                scope,
                joker,
                context(Ctx.INDIVIDUAL, Area.PLAY, other_card=card, lucky_trigger=lucky),
            )
            if effect is not None:
                effects.append(effect)
        for effect in effects:
            _apply_card_effect(scope, effect, card, score)


def _evaluate_played_card(scope: Scope, card: PlayingCard) -> tuple[Effect, bool]:
    """A scoring card's own effects (``eval_card`` with ``cardarea = G.play``)."""
    own = Effect()
    chips = chip_bonus(card)
    if chips > 0:
        own.chips = chips
    mult, lucky_mult = chip_mult(scope, card)
    if mult > 0:
        own.mult = mult
    x_mult = chip_x_mult(card)
    if x_mult > 0:
        own.x_mult = x_mult
    dollars, lucky_money = p_dollars(scope, card)
    if dollars > 0:
        own.dollars = dollars
    if card.edition is not None and not card.debuffed:
        own.edition = edition_effect(card.edition, card.debuffed)
    return own, lucky_mult or lucky_money


def _apply_card_effect(scope: Scope, effect: Effect, card: PlayingCard, score: _Score) -> None:
    """Apply one effect on a scoring card, in the game's order of fields."""
    if effect.chips is not None:
        score.chips += effect.chips
    if effect.mult is not None:
        score.mult += effect.mult
    if effect.dollars is not None:
        scope.change_money(effect.dollars, "scoring")
    if effect.mult_mod is not None:
        score.mult += effect.mult_mod
    if effect.chip_mod is not None:
        score.chips += effect.chip_mod
    if effect.swap:
        score.chips, score.mult = score.mult, score.chips
    if effect.x_mult is not None:
        score.mult *= effect.x_mult
    _emit_effect(scope, effect, card, own_key="card")
    if effect.edition is not None:
        chips, mult, x_mult = effect.edition
        score.chips += chips
        score.mult += mult
        score.mult *= x_mult
        _emit_edition(scope, card.uid, card.edition or "", card.uid, effect.edition)


def _score_held_card(
    scope: Scope, card: PlayingCard, score: _Score, context: _ContextFactory
) -> None:
    state = scope.state
    reps: list[Effect | None] = [None]
    index = 0
    while index < len(reps):
        rep = reps[index]
        if rep is not None:
            source = rep.via or rep.source
            scope.emit(
                CardRetriggered(card.uid, source.uid if source else card.uid, _key(rep, "seal"))
            )
        own = Effect()
        h_mult = chip_h_mult(card)
        if h_mult > 0:
            own.h_mult = h_mult
        h_x_mult = chip_h_x_mult(card)
        if h_x_mult > 0:
            own.x_mult = h_x_mult
        effects: list[Effect] = [own]
        for joker in list(state.jokers):
            effect = calculate_joker(
                scope, joker, context(Ctx.INDIVIDUAL, Area.HAND, other_card=card)
            )
            if effect is not None:
                effects.append(effect)
        if index == 0:
            if red_seal_repetitions(card) and (not own.is_empty() or len(effects) > 1):
                reps.append(Effect())
            for joker in list(state.jokers):
                effect = calculate_joker(
                    scope,
                    joker,
                    context(Ctx.REPETITION, Area.HAND, other_card=card, card_effects=effects),
                )
                if effect is not None and effect.repetitions:
                    reps.extend([effect] * effect.repetitions)
        for effect in effects:
            if effect.dollars is not None:
                scope.change_money(effect.dollars, "scoring")
            if effect.h_mult is not None:
                score.mult += effect.h_mult
            if effect.x_mult is not None:
                score.mult *= effect.x_mult
            _emit_effect(scope, effect, card, own_key="held")
        index += 1


def _score_joker(scope: Scope, joker: Item, score: _Score, context: _ContextFactory) -> None:
    state = scope.state
    chips, mult, x_mult = edition_effect(joker.edition, joker.debuffed)
    if chips or mult:
        score.chips += chips
        score.mult += mult
        _emit_edition(scope, joker.uid, joker.edition or "", None, (chips, mult, 1))

    effect = calculate_joker(scope, joker, context(Ctx.JOKER_MAIN))
    if effect is not None:
        _apply_joker_effect(scope, effect, score)
    for other in list(state.jokers):
        effect = calculate_joker(scope, other, context(Ctx.OTHER_JOKER, other_joker=joker))
        if effect is not None:
            _apply_joker_effect(scope, effect, score)

    if x_mult != 1:
        score.mult *= x_mult
        _emit_edition(scope, joker.uid, joker.edition or "", None, (0, 0, x_mult))


def _apply_joker_effect(scope: Scope, effect: Effect, score: _Score) -> None:
    if effect.mult_mod is not None:
        score.mult += effect.mult_mod
    if effect.chip_mod is not None:
        score.chips += effect.chip_mod
    if effect.x_mult_mod is not None:
        score.mult *= effect.x_mult_mod
    _emit_effect(scope, effect, None, own_key="joker")


def _key(effect: Effect, fallback: str) -> str:
    return effect.source.key if effect.source is not None else fallback


def _emit_effect(scope: Scope, effect: Effect, card: PlayingCard | None, *, own_key: str) -> None:
    """Record an applied effect for viewers; neutral effects are not recorded."""
    chips = (effect.chips or 0) + (effect.chip_mod or 0)
    mult = (effect.mult or 0) + (effect.mult_mod or 0) + (effect.h_mult or 0)
    x_mult = effect.x_mult if effect.x_mult is not None else (effect.x_mult_mod or 1)
    dollars = effect.dollars or 0
    if not (chips or mult or x_mult != 1 or dollars or effect.swap):
        return
    if effect.source is not None:
        source, key = effect.source.uid, effect.source.key
    elif card is not None:
        source, key = card.uid, card.enhancement or own_key
    else:
        source, key = 0, own_key
    scope.emit(
        EffectApplied(
            source=source,
            key=key,
            target=card.uid if card is not None else None,
            chips=chips,
            mult=mult,
            x_mult=x_mult,
            dollars=dollars,
            via=effect.via.uid if effect.via is not None else None,
        )
    )


def _emit_edition(
    scope: Scope, owner: int, edition: str, target: int | None, change: tuple[float, float, float]
) -> None:
    chips, mult, x_mult = change
    scope.emit(
        EffectApplied(
            source=owner, key=edition, target=target, chips=chips, mult=mult, x_mult=x_mult
        )
    )
