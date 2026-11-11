"""Joker effects, grouped by mechanism.

Each function ports one joker's branches of ``Card:calculate_joker`` (and of
``Card:calculate_dollar_bonus`` for jokers that pay at the end of the round).
Changes to a joker's own values are skipped when the call is a copy by Blueprint
or Brainstorm, exactly where the game checks ``context.blueprint``.

Each docstring says what the joker does, naming the game's config values it uses
(``extra`` is the game's catch-all name for a joker's main number).

Source: card.lua ``Card:calculate_joker`` and ``Card:calculate_dollar_bonus``,
under the joker's name.
"""

from __future__ import annotations

from collections.abc import Callable

from .cards import card_id
from .content import CENTERS
from .effects import (
    SUIT_NAMES,
    Area,
    Context,
    ContextKind,
    Effect,
    copy_joker,
    destroy_joker,
    is_face,
    is_suit,
    register_joker,
    value,
)
from .scope import Scope
from .state import Item

__all__ = ["DOLLAR_BONUS"]

#: Jokers that pay money when the round is cashed out, by key.
DOLLAR_BONUS: dict[str, Callable[[Scope, Item], int | None]] = {}


def _other_card_rank(context: Context) -> int:
    """``context.other_card:get_id()``; Stone cards never match a rank."""
    assert context.other_card is not None
    rank = card_id(context.other_card)
    return -1 if rank is None else rank


def _is_scoring_card(context: Context) -> bool:
    """Whether the context is a played card being scored."""
    return context.kind is ContextKind.INDIVIDUAL and context.area is Area.PLAY


def _is_held_card(context: Context) -> bool:
    """Whether the context is a card held in hand being scored."""
    return context.kind is ContextKind.INDIVIDUAL and context.area is Area.HAND


# Jokers whose only effect is one of the game's shared main rules: a flat
# x-mult, or a bonus when the played cards contain a poker hand.
@register_joker(
    "j_jolly",
    "j_zany",
    "j_mad",
    "j_crazy",
    "j_droll",
    "j_sly",
    "j_wily",
    "j_clever",
    "j_devious",
    "j_crafty",
    "j_duo",
    "j_trio",
    "j_family",
    "j_order",
    "j_tribe",
)
def _shared_rule_only(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """Jolly/Zany/Mad/Crazy/Droll: +``t_mult`` mult, Sly/Wily/Clever/Devious/Crafty: +``t_chips``
    chips, The Duo/Trio/Family/Order/Tribe: x``x_mult``, when the played cards contain the
    joker's poker hand (``type``). The engine applies these shared rules itself."""
    return None


# Rule changes, read by hand evaluation and scoring wherever they apply.
@register_joker("j_four_fingers", "j_shortcut", "j_smeared", "j_splash", "j_pareidolia")
def _rule_change(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """Four Fingers, Shortcut, Smeared Joker, Splash and Pareidolia change how hands and
    cards are evaluated; the evaluation code checks for them."""
    return None


@register_joker("j_joker")
def _joker(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """+4 mult (``mult``)."""
    if context.kind is ContextKind.JOKER_MAIN:
        return Effect(mult=value(joker, "mult"))
    return None


# Effects on each scoring card.


@register_joker("j_greedy_joker", "j_lusty_joker", "j_wrathful_joker", "j_gluttenous_joker")
def _suit_mult(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """Greedy/Lusty/Wrathful/Gluttonous Joker: +``extra.s_mult`` mult for each scoring card
    of the joker's suit (``extra.suit``)."""
    suit = SUIT_NAMES[str(joker.values["extra.suit"])]
    if (
        _is_scoring_card(context)
        and context.other_card
        and is_suit(scope.state, context.other_card, suit)
    ):
        return Effect(mult=value(joker, "extra.s_mult"))
    return None


@register_joker("j_scary_face")
def _scary_face(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """+``extra`` (30) chips for each scoring face card."""
    if (
        _is_scoring_card(context)
        and context.other_card
        and is_face(scope.state, context.other_card)
    ):
        return Effect(chips=value(joker, "extra"))
    return None


@register_joker("j_smiley")
def _smiley_face(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """+``extra`` (5) mult for each scoring face card."""
    if (
        _is_scoring_card(context)
        and context.other_card
        and is_face(scope.state, context.other_card)
    ):
        return Effect(mult=value(joker, "extra"))
    return None


@register_joker("j_fibonacci")
def _fibonacci(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """+``extra`` (8) mult for each scoring Ace, 2, 3, 5 or 8."""
    if _is_scoring_card(context) and _other_card_rank(context) in (2, 3, 5, 8, 14):
        return Effect(mult=value(joker, "extra"))
    return None


@register_joker("j_even_steven")
def _even_steven(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """+``extra`` (4) mult for each scoring 10, 8, 6, 4 or 2."""
    if (
        _is_scoring_card(context)
        and 0 <= _other_card_rank(context) <= 10
        and _other_card_rank(context) % 2 == 0
    ):
        return Effect(mult=value(joker, "extra"))
    return None


@register_joker("j_odd_todd")
def _odd_todd(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """+``extra`` (31) chips for each scoring Ace, 9, 7, 5 or 3."""
    if not _is_scoring_card(context):
        return None
    rank = _other_card_rank(context)
    if (0 <= rank <= 10 and rank % 2 == 1) or rank == 14:
        return Effect(chips=value(joker, "extra"))
    return None


@register_joker("j_scholar")
def _scholar(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """+``extra.chips`` chips and +``extra.mult`` mult for each scoring Ace."""
    if _is_scoring_card(context) and _other_card_rank(context) == 14:
        return Effect(chips=value(joker, "extra.chips"), mult=value(joker, "extra.mult"))
    return None


@register_joker("j_walkie_talkie")
def _walkie_talkie(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """+``extra.chips`` chips and +``extra.mult`` mult for each scoring 10 or 4."""
    if _is_scoring_card(context) and _other_card_rank(context) in (10, 4):
        return Effect(chips=value(joker, "extra.chips"), mult=value(joker, "extra.mult"))
    return None


@register_joker("j_photograph")
def _photograph(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """x``extra`` (2) mult for the first scoring face card."""
    if not _is_scoring_card(context):
        return None
    first_face = next((card for card in context.scoring_hand if is_face(scope.state, card)), None)
    if first_face is not None and context.other_card is first_face:
        return Effect(x_mult=value(joker, "extra"))
    return None


# Effects of cards held in hand.


@register_joker("j_baron")
def _baron(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """x``extra`` (1.5) mult for each King held in hand."""
    if _is_held_card(context) and context.other_card and _other_card_rank(context) == 13:
        return Effect() if context.other_card.debuffed else Effect(x_mult=value(joker, "extra"))
    return None


@register_joker("j_shoot_the_moon")
def _shoot_the_moon(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """+13 mult for each Queen held in hand."""
    if _is_held_card(context) and context.other_card and _other_card_rank(context) == 12:
        return Effect() if context.other_card.debuffed else Effect(mult=13)
    return None


# Retriggers.


@register_joker("j_hanging_chad")
def _hanging_chad(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """Retrigger the first scoring card ``extra`` (2) more times."""
    repetition = context.kind is ContextKind.REPETITION and context.area is Area.PLAY
    if repetition and context.scoring_hand and context.other_card is context.scoring_hand[0]:
        return Effect(repetitions=int(value(joker, "extra")))
    return None


@register_joker("j_sock_and_buskin")
def _sock_and_buskin(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """Retrigger each scoring face card ``extra`` (1) more time."""
    repetition = context.kind is ContextKind.REPETITION and context.area is Area.PLAY
    if repetition and context.other_card and is_face(scope.state, context.other_card):
        return Effect(repetitions=int(value(joker, "extra")))
    return None


@register_joker("j_hack")
def _hack(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """Retrigger each scoring 2, 3, 4 or 5 ``extra`` (1) more time."""
    if (
        context.kind is ContextKind.REPETITION
        and context.area is Area.PLAY
        and _other_card_rank(context) in (2, 3, 4, 5)
    ):
        return Effect(repetitions=int(value(joker, "extra")))
    return None


@register_joker("j_mime")
def _mime(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """Retrigger each effect of cards held in hand ``extra`` (1) more time."""
    if (
        context.kind is ContextKind.REPETITION
        and context.area is Area.HAND
        and context.card_effects
    ):
        first = context.card_effects[0]
        if (first is not None and not first.is_empty()) or len(context.card_effects) > 1:
            return Effect(repetitions=int(value(joker, "extra")))
    return None


# Main effects that depend on the hand or the round.


@register_joker("j_half")
def _half(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """+``extra.mult`` mult if at most ``extra.size`` cards are played."""
    if context.kind is ContextKind.JOKER_MAIN and len(context.full_hand) <= value(
        joker, "extra.size"
    ):
        return Effect(mult=value(joker, "extra.mult"))
    return None


@register_joker("j_abstract")
def _abstract(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """+``extra`` (3) mult for each joker owned."""
    if context.kind is ContextKind.JOKER_MAIN:
        count = sum(CENTERS[other.key].is_joker for other in scope.state.jokers)
        return Effect(mult=count * value(joker, "extra"))
    return None


@register_joker("j_acrobat")
def _acrobat(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """x``extra`` (3) mult on the last hand of the round."""
    if context.kind is ContextKind.JOKER_MAIN and scope.state.hands_left == 0:
        return Effect(x_mult=value(joker, "extra"))
    return None


@register_joker("j_mystic_summit")
def _mystic_summit(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """+``extra.mult`` mult when no discards are left (``extra.d_remaining``)."""
    if context.kind is ContextKind.JOKER_MAIN and scope.state.discards_left == value(
        joker, "extra.d_remaining"
    ):
        return Effect(mult=value(joker, "extra.mult"))
    return None


@register_joker("j_misprint")
def _misprint(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """A random +``extra.min`` to +``extra.max`` mult."""
    if context.kind is ContextKind.JOKER_MAIN:
        low, high = int(value(joker, "extra.min")), int(value(joker, "extra.max"))
        return Effect(mult=scope.rng.randint("misprint", low, high))
    return None


@register_joker("j_banner")
def _banner(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """+``extra`` (30) chips for each discard left."""
    if context.kind is ContextKind.JOKER_MAIN and scope.state.discards_left > 0:
        return Effect(chips=scope.state.discards_left * value(joker, "extra"))
    return None


@register_joker("j_baseball")
def _baseball(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """x``extra`` (1.5) mult for each Uncommon joker."""
    other = context.other_joker
    if context.kind is ContextKind.OTHER_JOKER and other is not None and other is not joker:
        return Effect(x_mult=value(joker, "extra")) if CENTERS[other.key].rarity == 2 else None
    return None


# Jokers whose values change as the run goes on.


@register_joker("j_ride_the_bus")
def _ride_the_bus(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """Gains ``extra`` mult for each hand played without a scoring face card, and
    resets to 0 when one scores; adds its ``mult``."""
    if context.kind is ContextKind.BEFORE and not context.blueprint:
        if any(is_face(scope.state, card) for card in context.scoring_hand):
            last = value(joker, "mult")
            joker.values["mult"] = 0
            return Effect() if last > 0 else None
        joker.values["mult"] = value(joker, "mult") + value(joker, "extra")
    if context.kind is ContextKind.JOKER_MAIN and value(joker, "mult") > 0:
        return Effect(mult=value(joker, "mult"))
    return None


@register_joker("j_green_joker")
def _green_joker(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """Gains ``extra.hand_add`` mult per hand played and loses ``extra.discard_sub`` per
    discard; adds its ``mult``."""
    if context.kind is ContextKind.BEFORE and not context.blueprint:
        joker.values["mult"] = value(joker, "mult") + value(joker, "extra.hand_add")
        return Effect()
    last_discarded = bool(context.full_hand) and context.other_card is context.full_hand[-1]
    if context.kind is ContextKind.DISCARD and not context.blueprint and last_discarded:
        previous = value(joker, "mult")
        joker.values["mult"] = max(0, previous - value(joker, "extra.discard_sub"))
        return Effect() if joker.values["mult"] != previous else None
    if context.kind is ContextKind.JOKER_MAIN and value(joker, "mult") > 0:
        return Effect(mult=value(joker, "mult"))
    return None


@register_joker("j_ice_cream")
def _ice_cream(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """+``extra.chips`` chips, which drop by ``extra.chip_mod`` after each hand; melts at 0."""
    if context.kind is ContextKind.AFTER and not context.blueprint:
        if value(joker, "extra.chips") - value(joker, "extra.chip_mod") <= 0:
            destroy_joker(scope, joker, "melted")
        else:
            joker.values["extra.chips"] = value(joker, "extra.chips") - value(
                joker, "extra.chip_mod"
            )
        return Effect()
    if context.kind is ContextKind.JOKER_MAIN:
        return Effect(chips=value(joker, "extra.chips"))
    return None


@register_joker("j_popcorn")
def _popcorn(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """+``mult`` mult, which drops by ``extra`` at the end of each round; eaten at 0."""
    if context.kind is ContextKind.END_OF_ROUND and not context.blueprint:
        if value(joker, "mult") - value(joker, "extra") <= 0:
            destroy_joker(scope, joker, "eaten")
        else:
            joker.values["mult"] = value(joker, "mult") - value(joker, "extra")
        return Effect()
    if context.kind is ContextKind.JOKER_MAIN and value(joker, "mult") > 0:
        return Effect(mult=value(joker, "mult"))
    return None


@register_joker("j_gros_michel")
def _gros_michel(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """+``extra.mult`` mult; a 1 in ``extra.odds`` chance each round to go extinct."""
    if context.kind is ContextKind.END_OF_ROUND and not context.blueprint:
        if scope.chance("gros_michel", value(joker, "extra.odds")):
            destroy_joker(scope, joker, "extinct")
            scope.state.pool_flags["gros_michel_extinct"] = True
        return Effect()
    if context.kind is ContextKind.JOKER_MAIN:
        return Effect(mult=value(joker, "extra.mult"))
    return None


@register_joker("j_cavendish")
def _cavendish(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """x``extra.Xmult`` mult; a 1 in ``extra.odds`` chance each round to be destroyed."""
    if context.kind is ContextKind.END_OF_ROUND and not context.blueprint:
        if scope.chance("cavendish", value(joker, "extra.odds")):
            destroy_joker(scope, joker, "extinct")
        return Effect()
    if context.kind is ContextKind.JOKER_MAIN:
        return Effect(x_mult=value(joker, "extra.Xmult"))
    return None


# Money.


@register_joker("j_golden")
def _golden(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """Earn ``extra`` (4) dollars at the end of each round (see ``DOLLAR_BONUS``)."""
    return None


DOLLAR_BONUS["j_golden"] = lambda scope, joker: int(value(joker, "extra"))


# Copying other jokers.


@register_joker("j_blueprint")
def _blueprint(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """Copies the effect of the joker to its right."""
    jokers = scope.state.jokers
    position = next((i for i, other in enumerate(jokers) if other is joker), None)
    if position is None or position + 1 >= len(jokers):
        return None
    return copy_joker(scope, joker, jokers[position + 1], context)


@register_joker("j_brainstorm")
def _brainstorm(scope: Scope, joker: Item, context: Context) -> Effect | None:
    """Copies the effect of the leftmost joker."""
    jokers = scope.state.jokers
    return copy_joker(scope, joker, jokers[0] if jokers else None, context)
