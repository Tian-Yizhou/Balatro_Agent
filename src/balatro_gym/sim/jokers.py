"""Joker effects, grouped by mechanism.

Each function ports one joker's branches of ``Card:calculate_joker`` (and of
``Card:calculate_dollar_bonus`` for jokers that pay at the end of the round).
Changes to a joker's own values are skipped when the call is a copy by Blueprint
or Brainstorm, exactly where the game checks ``context.blueprint``.

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
    Ctx,
    Effect,
    copy_joker,
    destroy_joker,
    is_face,
    is_suit,
    joker,
    value,
)
from .scope import Scope
from .state import Item

__all__ = ["DOLLAR_BONUS"]

#: Jokers that pay money when the round is cashed out, by key.
DOLLAR_BONUS: dict[str, Callable[[Scope, Item], int | None]] = {}


def _rank(ctx: Context) -> int:
    """``context.other_card:get_id()``; Stone cards never match a rank."""
    assert ctx.other_card is not None
    rank = card_id(ctx.other_card)
    return -1 if rank is None else rank


def _scoring_card(ctx: Context) -> bool:
    return ctx.kind is Ctx.INDIVIDUAL and ctx.area is Area.PLAY


def _held_card(ctx: Context) -> bool:
    return ctx.kind is Ctx.INDIVIDUAL and ctx.area is Area.HAND


# Jokers whose only effect is one of the game's shared main rules: a flat
# x-mult, or a bonus when the played cards contain a poker hand.
@joker(
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
def _shared_rule_only(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    return None


# Rule changes, read by hand evaluation and scoring wherever they apply.
@joker("j_four_fingers", "j_shortcut", "j_smeared", "j_splash", "j_pareidolia")
def _rule_change(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    return None


@joker("j_joker")
def _joker(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if ctx.kind is Ctx.JOKER_MAIN:
        return Effect(mult_mod=value(j, "mult"))
    return None


# Effects on each scoring card.


@joker("j_greedy_joker", "j_lusty_joker", "j_wrathful_joker", "j_gluttenous_joker")
def _suit_mult(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    suit = SUIT_NAMES[str(j.values["extra.suit"])]
    if _scoring_card(ctx) and ctx.other_card and is_suit(scope.state, ctx.other_card, suit):
        return Effect(mult=value(j, "extra.s_mult"))
    return None


@joker("j_scary_face")
def _scary_face(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if _scoring_card(ctx) and ctx.other_card and is_face(scope.state, ctx.other_card):
        return Effect(chips=value(j, "extra"))
    return None


@joker("j_smiley")
def _smiley_face(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if _scoring_card(ctx) and ctx.other_card and is_face(scope.state, ctx.other_card):
        return Effect(mult=value(j, "extra"))
    return None


@joker("j_fibonacci")
def _fibonacci(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if _scoring_card(ctx) and _rank(ctx) in (2, 3, 5, 8, 14):
        return Effect(mult=value(j, "extra"))
    return None


@joker("j_even_steven")
def _even_steven(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if _scoring_card(ctx) and 0 <= _rank(ctx) <= 10 and _rank(ctx) % 2 == 0:
        return Effect(mult=value(j, "extra"))
    return None


@joker("j_odd_todd")
def _odd_todd(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if not _scoring_card(ctx):
        return None
    rank = _rank(ctx)
    if (0 <= rank <= 10 and rank % 2 == 1) or rank == 14:
        return Effect(chips=value(j, "extra"))
    return None


@joker("j_scholar")
def _scholar(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if _scoring_card(ctx) and _rank(ctx) == 14:
        return Effect(chips=value(j, "extra.chips"), mult=value(j, "extra.mult"))
    return None


@joker("j_walkie_talkie")
def _walkie_talkie(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if _scoring_card(ctx) and _rank(ctx) in (10, 4):
        return Effect(chips=value(j, "extra.chips"), mult=value(j, "extra.mult"))
    return None


@joker("j_photograph")
def _photograph(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if not _scoring_card(ctx):
        return None
    first_face = next((c for c in ctx.scoring_hand if is_face(scope.state, c)), None)
    if first_face is not None and ctx.other_card is first_face:
        return Effect(x_mult=value(j, "extra"))
    return None


# Effects of cards held in hand.


@joker("j_baron")
def _baron(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if _held_card(ctx) and ctx.other_card and _rank(ctx) == 13:
        return Effect() if ctx.other_card.debuffed else Effect(x_mult=value(j, "extra"))
    return None


@joker("j_shoot_the_moon")
def _shoot_the_moon(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if _held_card(ctx) and ctx.other_card and _rank(ctx) == 12:
        return Effect() if ctx.other_card.debuffed else Effect(h_mult=13)
    return None


# Retriggers.


@joker("j_hanging_chad")
def _hanging_chad(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    repetition = ctx.kind is Ctx.REPETITION and ctx.area is Area.PLAY
    if repetition and ctx.scoring_hand and ctx.other_card is ctx.scoring_hand[0]:
        return Effect(repetitions=int(value(j, "extra")))
    return None


@joker("j_sock_and_buskin")
def _sock_and_buskin(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    repetition = ctx.kind is Ctx.REPETITION and ctx.area is Area.PLAY
    if repetition and ctx.other_card and is_face(scope.state, ctx.other_card):
        return Effect(repetitions=int(value(j, "extra")))
    return None


@joker("j_hack")
def _hack(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if ctx.kind is Ctx.REPETITION and ctx.area is Area.PLAY and _rank(ctx) in (2, 3, 4, 5):
        return Effect(repetitions=int(value(j, "extra")))
    return None


@joker("j_mime")
def _mime(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if ctx.kind is Ctx.REPETITION and ctx.area is Area.HAND and ctx.card_effects:
        first = ctx.card_effects[0]
        if (first is not None and not first.is_empty()) or len(ctx.card_effects) > 1:
            return Effect(repetitions=int(value(j, "extra")))
    return None


# Main effects that depend on the hand or the round.


@joker("j_half")
def _half(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if ctx.kind is Ctx.JOKER_MAIN and len(ctx.full_hand) <= value(j, "extra.size"):
        return Effect(mult_mod=value(j, "extra.mult"))
    return None


@joker("j_abstract")
def _abstract(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if ctx.kind is Ctx.JOKER_MAIN:
        count = sum(CENTERS[other.key].is_joker for other in scope.state.jokers)
        return Effect(mult_mod=count * value(j, "extra"))
    return None


@joker("j_acrobat")
def _acrobat(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if ctx.kind is Ctx.JOKER_MAIN and scope.state.hands_left == 0:
        return Effect(x_mult_mod=value(j, "extra"))
    return None


@joker("j_mystic_summit")
def _mystic_summit(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if ctx.kind is Ctx.JOKER_MAIN and scope.state.discards_left == value(j, "extra.d_remaining"):
        return Effect(mult_mod=value(j, "extra.mult"))
    return None


@joker("j_misprint")
def _misprint(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if ctx.kind is Ctx.JOKER_MAIN:
        low, high = int(value(j, "extra.min")), int(value(j, "extra.max"))
        return Effect(mult_mod=scope.rng.randint("misprint", low, high))
    return None


@joker("j_banner")
def _banner(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if ctx.kind is Ctx.JOKER_MAIN and scope.state.discards_left > 0:
        return Effect(chip_mod=scope.state.discards_left * value(j, "extra"))
    return None


@joker("j_baseball")
def _baseball(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    other = ctx.other_joker
    if ctx.kind is Ctx.OTHER_JOKER and other is not None and other is not j:
        return Effect(x_mult_mod=value(j, "extra")) if CENTERS[other.key].rarity == 2 else None
    return None


# Jokers whose values change as the run goes on.


@joker("j_ride_the_bus")
def _ride_the_bus(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if ctx.kind is Ctx.BEFORE and not ctx.blueprint:
        if any(is_face(scope.state, c) for c in ctx.scoring_hand):
            last = value(j, "mult")
            j.values["mult"] = 0
            return Effect() if last > 0 else None
        j.values["mult"] = value(j, "mult") + value(j, "extra")
    if ctx.kind is Ctx.JOKER_MAIN and value(j, "mult") > 0:
        return Effect(mult_mod=value(j, "mult"))
    return None


@joker("j_green_joker")
def _green_joker(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if ctx.kind is Ctx.BEFORE and not ctx.blueprint:
        j.values["mult"] = value(j, "mult") + value(j, "extra.hand_add")
        return Effect()
    last_discarded = bool(ctx.full_hand) and ctx.other_card is ctx.full_hand[-1]
    if ctx.kind is Ctx.DISCARD and not ctx.blueprint and last_discarded:
        previous = value(j, "mult")
        j.values["mult"] = max(0, previous - value(j, "extra.discard_sub"))
        return Effect() if j.values["mult"] != previous else None
    if ctx.kind is Ctx.JOKER_MAIN and value(j, "mult") > 0:
        return Effect(mult_mod=value(j, "mult"))
    return None


@joker("j_ice_cream")
def _ice_cream(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if ctx.kind is Ctx.AFTER and not ctx.blueprint:
        if value(j, "extra.chips") - value(j, "extra.chip_mod") <= 0:
            destroy_joker(scope, j, "melted")
        else:
            j.values["extra.chips"] = value(j, "extra.chips") - value(j, "extra.chip_mod")
        return Effect()
    if ctx.kind is Ctx.JOKER_MAIN:
        return Effect(chip_mod=value(j, "extra.chips"))
    return None


@joker("j_popcorn")
def _popcorn(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if ctx.kind is Ctx.END_OF_ROUND and not ctx.blueprint:
        if value(j, "mult") - value(j, "extra") <= 0:
            destroy_joker(scope, j, "eaten")
        else:
            j.values["mult"] = value(j, "mult") - value(j, "extra")
        return Effect()
    if ctx.kind is Ctx.JOKER_MAIN and value(j, "mult") > 0:
        return Effect(mult_mod=value(j, "mult"))
    return None


@joker("j_gros_michel")
def _gros_michel(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if ctx.kind is Ctx.END_OF_ROUND and not ctx.blueprint:
        if scope.chance("gros_michel", value(j, "extra.odds")):
            destroy_joker(scope, j, "extinct")
            scope.state.pool_flags["gros_michel_extinct"] = True
        return Effect()
    if ctx.kind is Ctx.JOKER_MAIN:
        return Effect(mult_mod=value(j, "extra.mult"))
    return None


@joker("j_cavendish")
def _cavendish(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    if ctx.kind is Ctx.END_OF_ROUND and not ctx.blueprint:
        if scope.chance("cavendish", value(j, "extra.odds")):
            destroy_joker(scope, j, "extinct")
        return Effect()
    if ctx.kind is Ctx.JOKER_MAIN:
        return Effect(x_mult_mod=value(j, "extra.Xmult"))
    return None


# Money.


@joker("j_golden")
def _golden(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    return None


DOLLAR_BONUS["j_golden"] = lambda scope, j: int(value(j, "extra"))


# Copying other jokers.


@joker("j_blueprint")
def _blueprint(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    jokers = scope.state.jokers
    position = next((i for i, other in enumerate(jokers) if other is j), None)
    if position is None or position + 1 >= len(jokers):
        return None
    return copy_joker(scope, j, jokers[position + 1], ctx)


@joker("j_brainstorm")
def _brainstorm(scope: Scope, j: Item, ctx: Context) -> Effect | None:
    jokers = scope.state.jokers
    return copy_joker(scope, j, jokers[0] if jokers else None, ctx)
