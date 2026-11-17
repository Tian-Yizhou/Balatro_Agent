"""Scoring a played hand: a step-by-step port of the game's ``evaluate_play``.

1. Find the poker hand and the scoring cards. The boss may stop the hand from
   scoring (The Psychic, The Eye, The Mouth): jokers then react to the debuffed
   hand and steps 2 to 6 are skipped. The Arm and The Ox act here too.
2. Jokers react ``before`` scoring (some level up the hand); The Flint halves the
   hand's base chips and mult.
3. Each scoring card scores from left to right, once per retrigger: its own chips,
   mult, x-mult, money and edition, then each joker's ``individual`` effect.
4. Each card held in hand applies its effects, with retriggers.
5. Each joker, then each consumable, applies its edition, its main effect and
   other jokers' reactions to it (``other_joker``), from left to right. The only
   consumable with a main effect is a Planet card with Observatory.
6. Cards are destroyed where needed (Glass Card, some jokers).
7. The score is added, and jokers react ``after`` scoring.

Source: functions/state_events.lua ``G.FUNCS.evaluate_play``;
functions/common_events.lua ``eval_card`` and ``level_up_hand``; blind.lua
``Blind:debuff_hand`` and ``Blind:modify_hand``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .blinds import BLINDS, lua_floor
from .cards import STONE
from .content import CENTERS
from .effects import Area, Context, ContextKind, Effect, calculate_joker, find_joker
from .events import (
    BlindTriggered,
    CardDestroyed,
    CardRetriggered,
    EffectApplied,
    HandLevelChanged,
    HandPlayed,
    HandScored,
)
from .hands import HAND_BASE, PokerHands, evaluate_poker_hand
from .modifiers import (
    edition_effect,
    held_mult,
    held_x_mult,
    red_seal_repetitions,
    scoring_chips,
    scoring_dollars,
    scoring_mult,
    scoring_x_mult,
)
from .scope import Scope
from .state import HandType, Item, PlayingCard, RunState

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
    def __call__(
        self, kind: ContextKind, area: Area | None = Area.JOKERS, **fields: object
    ) -> Context: ...


@dataclass(slots=True)
class _RunningScore:
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

    def make_context(
        kind: ContextKind, area: Area | None = Area.JOKERS, **fields: object
    ) -> Context:
        context = Context(
            kind,
            area=area,
            full_hand=played,
            scoring_hand=scoring,
            scoring_name=hand,
            poker_hands=hands,
        )
        for name, field_value in fields.items():
            setattr(context, name, field_value)
        return context

    destroyed: list[PlayingCard] = []
    if _blind_debuffs_hand(scope, played, hand):
        # The hand scores nothing. Source: the debuffed branch of evaluate_play.
        score = _RunningScore(chips=0.0, mult=0.0)
        for joker in list(state.jokers):
            calculate_joker(scope, joker, make_context(ContextKind.DEBUFFED_HAND))
    else:
        score = _score_hand(scope, hand, played, scoring, destroyed, make_context)

    total = lua_floor(score.chips * score.mult)
    state.chips += total
    scope.emit(
        HandScored(hand=hand, chips=score.chips, mult=score.mult, score=total, total=state.chips)
    )
    scope.flush()
    for joker in list(state.jokers):
        calculate_joker(scope, joker, make_context(ContextKind.AFTER))
    scope.flush()
    return destroyed


def _blind_debuffs_hand(scope: Scope, played: list[PlayingCard], hand: HandType) -> bool:
    """Whether the boss stops the hand from scoring; also The Arm's and The Ox's
    effects, which apply to a hand that does score.

    The Psychic needs 5 cards, The Eye allows each hand once, and The Mouth only
    the first hand played. Source: blind.lua ``Blind:debuff_hand``.
    """
    state = scope.state
    blind = state.blind
    if blind is None or blind.disabled:
        return False
    key = blind.key
    minimum = BLINDS[key].debuff.get("h_size_ge")
    debuffed = False
    if isinstance(minimum, int) and len(played) < minimum:
        debuffed = True
    elif key == "bl_eye":
        debuffed = hand in blind.hands_used
        if not debuffed:
            blind.hands_used.append(hand)
    elif key == "bl_mouth":
        debuffed = blind.only_hand is not None and blind.only_hand is not hand
        if not debuffed:
            blind.only_hand = hand
    if debuffed:
        scope.emit(BlindTriggered(key))
        return True
    if key == "bl_arm" and state.hands[hand].level > 1:
        scope.emit(BlindTriggered(key))
        level_up_hand(scope, hand, -1)
    if key == "bl_ox" and hand is state.most_played_hand:
        scope.emit(BlindTriggered(key))
        scope.change_money(-state.money, "ox")
    return False


def _score_hand(
    scope: Scope,
    hand: HandType,
    played: list[PlayingCard],
    scoring: list[PlayingCard],
    destroyed: list[PlayingCard],
    make_context: _ContextFactory,
) -> _RunningScore:
    """Score a hand the blind allows; destroyed cards are added to ``destroyed``."""
    state = scope.state
    for joker in list(state.jokers):
        effect = calculate_joker(scope, joker, make_context(ContextKind.BEFORE))
        if effect is not None and effect.level_up:
            level_up_hand(scope, hand, effect.level_up)

    level = state.hands[hand]
    score = _RunningScore(chips=float(level.chips), mult=float(level.mult))
    blind = state.blind
    if blind is not None and blind.key == "bl_flint" and not blind.disabled:
        # Source: blind.lua Blind:modify_hand.
        scope.emit(BlindTriggered(blind.key))
        score.mult = max(lua_floor(score.mult * 0.5 + 0.5), 1)
        score.chips = max(lua_floor(score.chips * 0.5 + 0.5), 0)
    for card in scoring:
        if not card.debuffed:
            _score_played_card(scope, card, score, make_context)
    for card in list(state.hand):
        _score_held_card(scope, card, score, make_context)
    for joker in list(state.jokers):
        _score_joker(scope, joker, score, make_context)
    for consumable in list(state.consumables):
        _score_joker(scope, consumable, score, make_context, consumable=True)

    for card in scoring:
        is_destroyed = False
        for joker in list(state.jokers):
            effect = calculate_joker(
                scope, joker, make_context(ContextKind.DESTROYING_CARD, other_card=card)
            )
            if effect is not None and effect.remove:
                is_destroyed = True
                break
        glass_odds = 4
        if card.enhancement == GLASS and not card.debuffed and scope.chance("glass", glass_odds):
            is_destroyed = True
        if is_destroyed:
            destroyed.append(card)
    for joker in list(state.jokers):
        calculate_joker(
            scope, joker, make_context(ContextKind.REMOVE_PLAYING_CARDS, removed=destroyed)
        )
    for card in destroyed:
        reason = "shattered" if card.enhancement == GLASS else "destroyed"
        scope.emit(CardDestroyed(card.uid, reason))
    return score


def _score_played_card(
    scope: Scope, card: PlayingCard, score: _RunningScore, make_context: _ContextFactory
) -> None:
    """Score one played card, once plus once per retrigger."""
    state = scope.state
    retriggers: list[Effect] = [Effect()] * red_seal_repetitions(card)
    for joker in list(state.jokers):
        effect = calculate_joker(
            scope, joker, make_context(ContextKind.REPETITION, Area.PLAY, other_card=card)
        )
        if effect is not None and effect.repetitions:
            retriggers.extend([effect] * effect.repetitions)

    for retrigger in [None, *retriggers]:
        if retrigger is not None:
            _emit_retrigger(scope, card, retrigger)
        card_effect, lucky = _evaluate_played_card(scope, card)
        effects: list[Effect] = [card_effect]
        for joker in list(state.jokers):
            effect = calculate_joker(
                scope,
                joker,
                make_context(
                    ContextKind.INDIVIDUAL, Area.PLAY, other_card=card, lucky_trigger=lucky
                ),
            )
            if effect is not None:
                effects.append(effect)
        for effect in effects:
            _apply_played_card_effect(scope, effect, card, score)


def _evaluate_played_card(scope: Scope, card: PlayingCard) -> tuple[Effect, bool]:
    """A scoring card's own effects (``eval_card`` with ``cardarea = G.play``).

    Returns the effect and whether a Lucky Card triggered.
    """
    card_effect = Effect()
    chips = scoring_chips(card)
    if chips > 0:
        card_effect.chips = chips
    mult, lucky_mult = scoring_mult(scope, card)
    if mult > 0:
        card_effect.mult = mult
    x_mult = scoring_x_mult(card)
    if x_mult > 0:
        card_effect.x_mult = x_mult
    dollars, lucky_money = scoring_dollars(scope, card)
    if dollars > 0:
        card_effect.dollars = dollars
    if card.edition is not None and not card.debuffed:
        card_effect.edition = edition_effect(card.edition, card.debuffed)
    return card_effect, lucky_mult or lucky_money


def _apply_played_card_effect(
    scope: Scope, effect: Effect, card: PlayingCard, score: _RunningScore
) -> None:
    """Apply one effect on a scoring card, in the game's order: chips, mult, money,
    swap, x-mult, then the card's edition."""
    if effect.chips is not None:
        score.chips += effect.chips
    if effect.mult is not None:
        score.mult += effect.mult
    if effect.dollars is not None:
        scope.change_money(effect.dollars, "scoring")
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
    scope: Scope, card: PlayingCard, score: _RunningScore, make_context: _ContextFactory
) -> None:
    """Apply the effects of one card held in hand, once plus once per retrigger.

    Retriggers are only counted on the first pass, and a Red Seal only retriggers
    a card that has an effect.
    """
    state = scope.state
    passes: list[Effect | None] = [None]
    index = 0
    while index < len(passes):
        retrigger = passes[index]
        if retrigger is not None:
            _emit_retrigger(scope, card, retrigger)
        card_effect = Effect()
        mult = held_mult(card)
        if mult > 0:
            card_effect.mult = mult
        x_mult = held_x_mult(card)
        if x_mult > 0:
            card_effect.x_mult = x_mult
        effects: list[Effect] = [card_effect]
        for joker in list(state.jokers):
            effect = calculate_joker(
                scope, joker, make_context(ContextKind.INDIVIDUAL, Area.HAND, other_card=card)
            )
            if effect is not None:
                effects.append(effect)
        if index == 0:
            if red_seal_repetitions(card) and (not card_effect.is_empty() or len(effects) > 1):
                passes.append(Effect())
            for joker in list(state.jokers):
                effect = calculate_joker(
                    scope,
                    joker,
                    make_context(
                        ContextKind.REPETITION, Area.HAND, other_card=card, card_effects=effects
                    ),
                )
                if effect is not None and effect.repetitions:
                    passes.extend([effect] * effect.repetitions)
        for effect in effects:
            if effect.dollars is not None:
                scope.change_money(effect.dollars, "scoring")
            if effect.mult is not None:
                score.mult += effect.mult
            if effect.x_mult is not None:
                score.mult *= effect.x_mult
            _emit_effect(scope, effect, card, own_key="held")
        index += 1


def _score_joker(
    scope: Scope,
    joker: Item,
    score: _RunningScore,
    make_context: _ContextFactory,
    *,
    consumable: bool = False,
) -> None:
    """Apply a joker's (or consumable's) edition, its main effect, and other
    jokers' reactions to it."""
    state = scope.state
    chips, mult, x_mult = edition_effect(joker.edition, joker.debuffed)
    if chips or mult:
        score.chips += chips
        score.mult += mult
        _emit_edition(scope, joker.uid, joker.edition or "", None, (chips, mult, 1))

    context = make_context(ContextKind.JOKER_MAIN)
    if consumable:
        effect = _observatory(state, joker, context)
    else:
        effect = calculate_joker(scope, joker, context)
    if effect is not None:
        _apply_joker_effect(scope, effect, score)
    for other in list(state.jokers):
        effect = calculate_joker(
            scope, other, make_context(ContextKind.OTHER_JOKER, other_joker=joker)
        )
        if effect is not None:
            _apply_joker_effect(scope, effect, score)

    if x_mult != 1:
        score.mult *= x_mult
        _emit_edition(scope, joker.uid, joker.edition or "", None, (0, 0, x_mult))


def _observatory(state: RunState, planet: Item, context: Context) -> Effect | None:
    """With Observatory, a Planet card held gives x-mult to the hand it levels up.

    Source: card.lua ``Card:calculate_joker`` (the Planet set).
    """
    if "v_observatory" not in state.used_vouchers or planet.debuffed:
        return None
    if CENTERS[planet.key].config.get("hand_type") != context.scoring_name:
        return None
    return Effect(x_mult=CENTERS["v_observatory"].config["extra"], source=planet)


def _apply_joker_effect(scope: Scope, effect: Effect, score: _RunningScore) -> None:
    if effect.mult is not None:
        score.mult += effect.mult
    if effect.chips is not None:
        score.chips += effect.chips
    if effect.x_mult is not None:
        score.mult *= effect.x_mult
    _emit_effect(scope, effect, None, own_key="joker")


def _emit_retrigger(scope: Scope, card: PlayingCard, retrigger: Effect) -> None:
    source = retrigger.via or retrigger.source
    key = retrigger.source.key if retrigger.source is not None else "seal"
    scope.emit(CardRetriggered(card.uid, source.uid if source else card.uid, key))


def _emit_effect(scope: Scope, effect: Effect, card: PlayingCard | None, *, own_key: str) -> None:
    """Record an applied effect for viewers; neutral effects are not recorded."""
    chips = effect.chips or 0
    mult = effect.mult or 0
    x_mult = effect.x_mult if effect.x_mult is not None else 1
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
