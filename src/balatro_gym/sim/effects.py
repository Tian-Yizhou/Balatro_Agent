"""The effect system: contexts, effect results and the joker registry.

The game asks every joker how it reacts to a *context*, such as a card being
scored or the round ending (``Card:calculate_joker``). A joker answers with a
table of changes to apply, or nothing. This module mirrors that design: a joker is
a function ``(scope, joker, context) -> Effect | None`` registered under its key.

Jokers that are not registered have no effect. The shop treats them as
unavailable (ADR 0003).

Source: card.lua ``Card:calculate_joker``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .cards import STONE, WILD, card_id
from .content import CENTERS
from .events import JokerDestroyed
from .hands import PokerHands
from .scope import Scope
from .state import HandType, Item, PlayingCard, RunState, Suit

__all__ = [
    "JOKERS",
    "SUIT_NAMES",
    "Area",
    "Context",
    "ContextKind",
    "Effect",
    "JokerEffect",
    "calculate_joker",
    "copy_joker",
    "destroy_joker",
    "find_joker",
    "is_face",
    "is_implemented",
    "is_suit",
    "register_joker",
    "value",
]


class ContextKind(StrEnum):
    """The situations a joker can react to, named as in the game."""

    SETTING_BLIND = "setting_blind"
    FIRST_HAND_DRAWN = "first_hand_drawn"
    BEFORE = "before"
    INDIVIDUAL = "individual"
    REPETITION = "repetition"
    OTHER_JOKER = "other_joker"
    JOKER_MAIN = "joker_main"
    AFTER = "after"
    DEBUFFED_HAND = "debuffed_hand"
    DESTROYING_CARD = "destroying_card"
    REMOVE_PLAYING_CARDS = "remove_playing_cards"
    PRE_DISCARD = "pre_discard"
    DISCARD = "discard"
    END_OF_ROUND = "end_of_round"
    BUYING_CARD = "buying_card"
    SELLING_SELF = "selling_self"
    SELLING_CARD = "selling_card"
    REROLL_SHOP = "reroll_shop"
    ENDING_SHOP = "ending_shop"
    USING_CONSUMABLE = "using_consumeable"
    OPEN_BOOSTER = "open_booster"
    SKIPPING_BOOSTER = "skipping_booster"
    PLAYING_CARD_ADDED = "playing_card_added"


class Area(StrEnum):
    """Which card area a context concerns (``context.cardarea``)."""

    PLAY = "play"
    HAND = "hand"
    JOKERS = "jokers"


@dataclass(slots=True)
class Context:
    """What is happening, as passed to ``calculate_joker``.

    Only the fields that the context kind uses are set. ``blueprint`` counts how
    many copy jokers the call has passed through, and ``blueprint_card`` is the
    first of them; jokers that must not be copied check ``blueprint``.
    """

    kind: ContextKind
    area: Area | None = None
    full_hand: Sequence[PlayingCard] = ()
    scoring_hand: Sequence[PlayingCard] = ()
    scoring_name: HandType | None = None
    poker_hands: PokerHands | None = None
    other_card: PlayingCard | None = None
    other_joker: Item | None = None
    #: Effects already collected for ``other_card``; read by retrigger jokers.
    card_effects: Sequence[Effect | None] = ()
    #: Whether ``other_card`` triggered its Lucky Card bonus in this evaluation.
    lucky_trigger: bool = False
    end_of_round: bool = False
    game_over: bool = False
    removed: Sequence[PlayingCard] = ()
    #: Playing cards just added to the deck (``playing_card_added``).
    cards: Sequence[PlayingCard] = ()
    #: The card bought or sold (``context.card``), the consumable used, or the
    #: booster pack opened.
    card: Item | None = None
    blueprint: int = 0
    blueprint_card: Item | None = None

    def contains(self, hand: HandType) -> bool:
        """Whether the played cards contain ``hand`` (``next(context.poker_hands[...])``)."""
        return self.poker_hands is not None and self.poker_hands.contains(hand)


@dataclass(slots=True)
class Effect:
    """A joker's or card's answer to a context. Unset numbers have no effect.

    The game uses different names for the same changes depending on the context
    (``chips``/``mult``/``x_mult`` per card, ``chip_mod``/``mult_mod``/``Xmult_mod``
    for a joker's main effect, ``h_mult`` for cards held in hand). Here they are
    always ``chips``, ``mult`` and ``x_mult``; the context decides how they apply.
    """

    #: Chips added.
    chips: float | None = None
    #: Mult added.
    mult: float | None = None
    #: Factor the mult is multiplied by.
    x_mult: float | None = None
    #: Money paid to the player.
    dollars: int | None = None
    #: Swap chips and mult (``extra.swap``).
    swap: bool = False
    #: An edition's ``(chips, mult, x_mult)``, for a card's own evaluation.
    edition: tuple[float, float, float] | None = None
    repetitions: int = 0
    level_up: int = 0
    remove: bool = False
    saved: bool = False
    #: The joker the effect comes from, after any copying.
    source: Item | None = None
    #: The Blueprint or Brainstorm that copied it.
    via: Item | None = None

    def is_empty(self) -> bool:
        """Whether the effect does nothing, like an empty table in the game."""
        return self == Effect(source=self.source, via=self.via)


JokerEffect = Callable[[Scope, Item, Context], Effect | None]

#: Implemented jokers, by key.
JOKERS: dict[str, JokerEffect] = {}


def register_joker(*keys: str) -> Callable[[JokerEffect], JokerEffect]:
    """Register a function as the effect of the jokers ``keys``."""

    def register(effect_function: JokerEffect) -> JokerEffect:
        for key in keys:
            if key not in CENTERS or not CENTERS[key].is_joker:
                raise KeyError(f"{key!r} is not a joker")
            if key in JOKERS:
                raise KeyError(f"{key!r} is registered twice")
            JOKERS[key] = effect_function
        return effect_function

    return register


def is_implemented(key: str) -> bool:
    return key in JOKERS


def find_joker(state: RunState, key: str) -> list[Item]:
    """The player's non-debuffed jokers and consumables with key ``key``.

    Source: functions/misc_functions.lua ``find_joker``.
    """
    return [
        item
        for item in (*state.jokers, *state.consumables)
        if item.key == key and not item.debuffed
    ]


def is_face(state: RunState, card: PlayingCard, *, from_boss: bool = False) -> bool:
    """Whether a card counts as a face card. Source: card.lua ``Card:is_face``."""
    if card.debuffed and not from_boss:
        return False
    return card_id(card) in (11, 12, 13) or bool(find_joker(state, "j_pareidolia"))


def is_suit(state: RunState, card: PlayingCard, suit: Suit, *, bypass_debuff: bool = False) -> bool:
    """Whether a card counts as ``suit`` for jokers. Source: card.lua ``Card:is_suit``."""
    if card.debuffed and not bypass_debuff:
        return False
    if card.enhancement == STONE:
        return False
    if card.enhancement == WILD:
        return True
    if find_joker(state, "j_smeared") and _is_red(card.suit) == _is_red(suit):
        return True
    return card.suit is suit


def _is_red(suit: Suit) -> bool:
    return suit is Suit.HEARTS or suit is Suit.DIAMONDS


#: The game's suit names, as used in item configs.
SUIT_NAMES = {
    "Spades": Suit.SPADES,
    "Hearts": Suit.HEARTS,
    "Clubs": Suit.CLUBS,
    "Diamonds": Suit.DIAMONDS,
}


def destroy_joker(scope: Scope, item: Item, reason: str) -> None:
    """Remove a joker from the run (after the current effects, as the game does)."""

    def remove() -> None:
        if item in scope.state.jokers:
            scope.state.jokers.remove(item)
            scope.emit(JokerDestroyed(item.uid, item.key, reason))

    scope.defer(remove)


def value(item: Item, name: str, default: float = 0) -> float:
    """A number from an item's values, such as ``value(joker, "extra.mult")``."""
    result = item.values.get(name, default)
    if not isinstance(result, int | float) or isinstance(result, bool):
        raise TypeError(f"{item.key} value {name!r} is not a number")
    return result


def calculate_joker(scope: Scope, item: Item, context: Context) -> Effect | None:
    """How ``item`` reacts to ``context``; ``None`` if it doesn't.

    Source: card.lua ``Card:calculate_joker``. For the main joker context the game
    first applies rules shared by many jokers (a ``x_mult`` above 1, and the
    ``t_mult`` and ``t_chips`` bonuses for containing a poker hand), then the
    joker's own rule.
    """
    if item.debuffed:
        return None
    if context.end_of_round and context.kind is ContextKind.INDIVIDUAL:
        return None
    if (
        context.kind is ContextKind.END_OF_ROUND
        and context.blueprint
        and item.key not in _COPY_JOKERS
    ):
        return None
    if context.kind is ContextKind.JOKER_MAIN:
        effect = _shared_main_rules(item, context)
        if effect is not None:
            return _attribute(effect, item, context)
    effect_function = JOKERS.get(item.key)
    if effect_function is None:
        return None
    effect = effect_function(scope, item, context)
    return None if effect is None else _attribute(effect, item, context)


_COPY_JOKERS = frozenset({"j_blueprint", "j_brainstorm"})


def copy_joker(scope: Scope, copier: Item, target: Item | None, context: Context) -> Effect | None:
    """Run ``target``'s effect on behalf of ``copier`` (Blueprint, Brainstorm)."""
    if target is None or target is copier:
        return None
    depth = context.blueprint + 1
    if depth > len(scope.state.jokers) + 1:
        return None
    copied = dataclasses.replace(
        context, blueprint=depth, blueprint_card=context.blueprint_card or copier
    )
    effect = calculate_joker(scope, target, copied)
    if effect is not None:
        effect.via = context.blueprint_card or copier
    return effect


def _attribute(effect: Effect, item: Item, context: Context) -> Effect:
    if effect.source is None:
        effect.source = item
    return effect


def _shared_main_rules(item: Item, context: Context) -> Effect | None:
    x_mult = value(item, "x_mult", 1)
    hand_type = item.values.get("type", "")
    applies = hand_type == "" or context.contains(HandType(hand_type))
    if item.key != "j_seeing_double" and x_mult > 1 and applies:
        return Effect(x_mult=x_mult)
    if value(item, "t_mult") > 0 and hand_type and context.contains(HandType(hand_type)):
        return Effect(mult=value(item, "t_mult"))
    if value(item, "t_chips") > 0 and hand_type and context.contains(HandType(hand_type)):
        return Effect(chips=value(item, "t_chips"))
    return None
