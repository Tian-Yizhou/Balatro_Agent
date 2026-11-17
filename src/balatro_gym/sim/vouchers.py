"""Vouchers: redeeming them in the shop, and what each one changes for the rest of
the run.

Most vouchers change a number of the run once, when redeemed. The others are
checked where they matter: Omen Globe and Telescope when a pack opens
(``packs``), Observatory when a hand scores (``scoring``), Illusion when the shop
creates a playing card (``shop``), and Director's Cut and Retcon by the
``reroll_boss`` action (ADR 0004).

Source: card.lua ``Card:redeem`` and ``Card:apply_to_run``;
functions/button_callbacks.lua ``G.FUNCS.reroll_boss_button`` and
``G.FUNCS.reroll_boss``; functions/common_events.lua ``change_shop_size``.
"""

from __future__ import annotations

from collections.abc import Callable

from .blinds import choose_boss
from .content import CENTERS
from .effects import Context, ContextKind, calculate_joker
from .events import AnteChanged, BossRerolled, ItemCreated
from .inventory import change_hand_size
from .items import set_cost
from .scope import Scope
from .shop import fill_shop
from .state import BlindSlot, Item, Phase, RunState

__all__ = [
    "BOSS_REROLL_COST",
    "IMPLEMENTED",
    "apply_to_run",
    "can_reroll_boss",
    "redeem",
    "reroll_boss",
    "set_all_costs",
]

#: What rerolling the boss costs. Source: functions/button_callbacks.lua
#: ``G.FUNCS.reroll_boss``.
BOSS_REROLL_COST = 10


def redeem(scope: Scope, voucher: Item) -> None:
    """Pay for ``voucher`` and apply it for the rest of the run.

    Redeeming any voucher removes this ante's voucher from later shops.
    """
    state = scope.state
    state.used_vouchers.append(voucher.key)
    scope.change_money(-voucher.cost, "voucher")
    state.shop_voucher = None
    apply_to_run(scope, voucher.key)
    for joker in list(state.jokers):
        calculate_joker(scope, joker, Context(ContextKind.BUYING_CARD, card=voucher))
    scope.flush()


def apply_to_run(scope: Scope, key: str) -> None:
    """Apply the lasting change of voucher ``key``. Source: card.lua ``Card:apply_to_run``."""
    extra = CENTERS[key].config.get("extra", 0)
    _EFFECTS[key](scope, extra)


def set_all_costs(state: RunState) -> None:
    """Re-price every joker, consumable, pack and voucher, as a discount changes."""
    in_pack = state.pack.items if state.pack is not None else ()
    for item in (
        *state.jokers,
        *state.consumables,
        *state.shop_cards,
        *state.shop_boosters,
        *state.shop_vouchers,
        *in_pack,
    ):
        set_cost(state, item)


def can_reroll_boss(state: RunState) -> bool:
    """Whether the boss can be rerolled now: Retcon allows any number of rerolls,
    Director's Cut one per ante. Source: ``G.FUNCS.reroll_boss_button``."""
    if state.money - state.bankrupt_at - BOSS_REROLL_COST < 0:
        return False
    used = state.used_vouchers
    return "v_retcon" in used or ("v_directors_cut" in used and not state.boss_rerolled)


def reroll_boss(scope: Scope) -> None:
    """Pay to replace the ante's boss blind with a new one."""
    state = scope.state
    state.boss_rerolled = True
    scope.change_money(-BOSS_REROLL_COST, "reroll_boss")
    boss = choose_boss(state.rng, state.ante, state.win_ante, state.bosses_used)
    state.blind_choices[BlindSlot.BOSS] = boss
    scope.emit(BossRerolled(boss, BOSS_REROLL_COST))


# The lasting effects, by voucher. Each takes the voucher's ``extra`` number.


def _no_lasting_effect(scope: Scope, extra: float) -> None:
    """Vouchers that act elsewhere, when their moment comes."""


def _overstock(scope: Scope, extra: float) -> None:
    """One more card slot in the shop, filled at once if the shop is open."""
    state = scope.state
    state.shop_joker_max += 1
    if state.phase is Phase.SHOP:
        before = len(state.shop_cards)
        fill_shop(scope)
        for item in state.shop_cards[before:]:
            scope.emit(ItemCreated(item.uid, item.key, "shop"))


def _tarot_rate(scope: Scope, extra: float) -> None:
    scope.state.tarot_rate = 4 * extra


def _planet_rate(scope: Scope, extra: float) -> None:
    scope.state.planet_rate = 4 * extra


def _edition_rate(scope: Scope, extra: float) -> None:
    scope.state.edition_rate = extra


def _playing_card_rate(scope: Scope, extra: float) -> None:
    scope.state.playing_card_rate = extra


def _consumable_slot(scope: Scope, extra: float) -> None:
    scope.state.consumable_slots += 1


def _joker_slot(scope: Scope, extra: float) -> None:
    scope.state.joker_slots += 1


def _discount(scope: Scope, extra: float) -> None:
    state = scope.state
    state.discount_percent = int(extra)
    set_all_costs(state)


def _cheaper_rerolls(scope: Scope, extra: float) -> None:
    state = scope.state
    state.base_reroll_cost -= int(extra)
    state.reroll_cost = max(0, state.reroll_cost - int(extra))


def _interest_cap(scope: Scope, extra: float) -> None:
    scope.state.interest_cap = int(extra)


def _more_hands(scope: Scope, extra: float) -> None:
    state = scope.state
    state.hands_per_round += int(extra)
    state.hands_left += int(extra)


def _more_discards(scope: Scope, extra: float) -> None:
    state = scope.state
    state.discards_per_round += int(extra)
    state.discards_left += int(extra)


def _bigger_hand(scope: Scope, extra: float) -> None:
    change_hand_size(scope, 1)


def _ante_back(scope: Scope, amount: int) -> None:
    state = scope.state
    state.ante -= amount
    scope.emit(AnteChanged(state.ante))


def _hieroglyph(scope: Scope, extra: float) -> None:
    """Back one ante, for one hand less each round."""
    state = scope.state
    _ante_back(scope, int(extra))
    state.hands_per_round -= int(extra)
    state.hands_left -= int(extra)


def _petroglyph(scope: Scope, extra: float) -> None:
    """Back one ante, for one discard less each round."""
    state = scope.state
    _ante_back(scope, int(extra))
    state.discards_per_round -= int(extra)
    state.discards_left -= min(state.discards_left, int(extra))


_EFFECTS: dict[str, Callable[[Scope, float], None]] = {
    "v_overstock_norm": _overstock,
    "v_overstock_plus": _overstock,
    "v_clearance_sale": _discount,
    "v_liquidation": _discount,
    "v_hone": _edition_rate,
    "v_glow_up": _edition_rate,
    "v_reroll_surplus": _cheaper_rerolls,
    "v_reroll_glut": _cheaper_rerolls,
    "v_crystal_ball": _consumable_slot,
    "v_omen_globe": _no_lasting_effect,
    "v_telescope": _no_lasting_effect,
    "v_observatory": _no_lasting_effect,
    "v_grabber": _more_hands,
    "v_nacho_tong": _more_hands,
    "v_wasteful": _more_discards,
    "v_recyclomancy": _more_discards,
    "v_tarot_merchant": _tarot_rate,
    "v_tarot_tycoon": _tarot_rate,
    "v_planet_merchant": _planet_rate,
    "v_planet_tycoon": _planet_rate,
    "v_seed_money": _interest_cap,
    "v_money_tree": _interest_cap,
    "v_blank": _no_lasting_effect,
    "v_antimatter": _joker_slot,
    "v_magic_trick": _playing_card_rate,
    "v_illusion": _playing_card_rate,
    "v_hieroglyph": _hieroglyph,
    "v_petroglyph": _petroglyph,
    "v_directors_cut": _no_lasting_effect,
    "v_retcon": _no_lasting_effect,
    "v_paint_brush": _bigger_hand,
    "v_palette": _bigger_hand,
}

#: Vouchers with an implemented effect: all of them.
IMPLEMENTED = frozenset(_EFFECTS)
