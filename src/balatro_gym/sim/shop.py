"""Creating cards: item pools, the shop, and rolled editions and stickers.

Source: functions/common_events.lua ``get_current_pool``, ``poll_edition`` and
``create_card``; functions/UI_definitions.lua ``create_card_for_shop``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from .content import CENTERS, Center, pool
from .effects import JOKERS, find_joker
from .items import initial_values, set_cost
from .scope import Scope
from .state import HandType, Item, RunState

__all__ = [
    "Area",
    "create_card",
    "create_card_for_shop",
    "fill_shop",
    "is_available",
    "poll_edition",
    "register_consumables",
]

Area = Literal["shop", "jokers", "consumables", "pack"]

# The fallback when every item of a pool is unavailable.
_FALLBACK = {"Tarot": "c_strength", "Tarot_Planet": "c_strength", "Planet": "c_pluto"}
_FALLBACK_DEFAULT = "j_joker"

_consumables: Callable[[str], bool] = lambda key: False  # noqa: E731


def register_consumables(is_implemented: Callable[[str], bool]) -> None:
    """Tell the shop which consumables have an implemented effect."""
    global _consumables
    _consumables = is_implemented


def is_available(state: RunState, center: Center) -> bool:
    """Whether ``center`` may be drawn from its pool right now.

    Items the player already has (or that are in the shop) are unavailable unless
    they own Showman, as are items without an implemented effect (ADR 0003).
    """
    key = center.key
    implemented = key in JOKERS if center.is_joker else _consumables(key)
    if not implemented:
        return False
    if key in _used_keys(state) and not find_joker(state, "j_ring_master"):
        return False
    if not center.unlocked and center.rarity != 4:
        return False
    if center.set == "Planet" and center.config.get("softlock"):
        hand = HandType(center.config["hand_type"])
        if state.hands[hand].played == 0:
            return False
    gate = center.raw.get("enhancement_gate")
    if gate and not any(card.enhancement == gate for card in state.playing_cards()):
        return False
    if key in ("c_black_hole", "c_soul"):
        return False
    no_flag = center.raw.get("no_pool_flag")
    if no_flag and state.pool_flags.get(str(no_flag)):
        return False
    yes_flag = center.raw.get("yes_pool_flag")
    return not (yes_flag and not state.pool_flags.get(str(yes_flag)))


def _used_keys(state: RunState) -> set[str]:
    """Keys of every joker and consumable that currently exists (``G.GAME.used_jokers``)."""
    return {item.key for item in (*state.jokers, *state.consumables, *state.shop_cards)}


def poll_edition(
    scope: Scope, key: str, *, mod: float = 1, no_negative: bool = False, guaranteed: bool = False
) -> str | None:
    """Roll an edition. Source: functions/common_events.lua ``poll_edition``."""
    roll = scope.rng.random(key)
    rate = 25 if guaranteed else scope.state.edition_rate * mod
    negative_rate = 25 if guaranteed else mod
    if roll > 1 - 0.003 * negative_rate and not no_negative:
        return "e_negative"
    if roll > 1 - 0.006 * rate:
        return "e_polychrome"
    if roll > 1 - 0.02 * rate:
        return "e_holo"
    if roll > 1 - 0.04 * rate:
        return "e_foil"
    return None


def create_card(
    scope: Scope,
    set_name: str,
    area: Area,
    *,
    key_append: str = "",
    rarity: float | None = None,
    legendary: bool = False,
    forced_key: str | None = None,
) -> Item:
    """Create a joker or consumable of ``set_name``, drawn from its pool.

    ``area`` is where the card is going, which decides whether a shop joker
    rolls stickers. The card is not placed anywhere.
    """
    state = scope.state
    ante = state.ante
    if forced_key is not None:
        key = forced_key
    else:
        key = _draw_from_pool(scope, set_name, key_append, rarity, legendary)
    item = Item(uid=state.new_uid(), key=key, values=initial_values(key))
    if CENTERS[key].is_joker:
        if area in ("shop", "pack"):
            _roll_stickers(scope, item, area)
        item.edition = poll_edition(scope, f"edi{key_append}{ante}")
    set_cost(state, item)
    return item


def _draw_from_pool(
    scope: Scope, set_name: str, key_append: str, rarity: float | None, legendary: bool
) -> str:
    state = scope.state
    ante = state.ante
    if set_name == "Joker":
        roll = rarity if rarity is not None else scope.rng.random(f"rarity{ante}{key_append}")
        level = 4 if legendary else 3 if roll > 0.95 else 2 if roll > 0.7 else 1
        members = pool("Joker", level)
        pool_key = f"Joker{level}{'' if legendary else key_append}{'' if legendary else ante}"
    else:
        members = _set_pool(set_name)
        pool_key = f"{set_name}{key_append}{ante}"
    entries = [center.key if is_available(state, center) else None for center in members]
    if not any(entries):
        return _FALLBACK.get(set_name, _FALLBACK_DEFAULT)
    choice = scope.rng.choice(pool_key, entries)
    attempt = 1
    while choice is None:
        attempt += 1
        choice = scope.rng.choice(f"{pool_key}_resample{attempt}", entries)
    return choice


def _set_pool(set_name: str) -> tuple[Center, ...]:
    if set_name == "Tarot_Planet":
        return tuple(sorted((*pool("Tarot"), *pool("Planet")), key=lambda center: center.order))
    return pool(set_name)


def _roll_stickers(scope: Scope, item: Item, area: Area) -> None:
    """Eternal, perishable and rental stickers on shop and pack jokers, by stake."""
    state = scope.state
    center = CENTERS[item.key]
    ante = state.ante
    poll = scope.rng.random(f"{'packetper' if area == 'pack' else 'etperpoll'}{ante}")
    if state.modifiers.get("eternals_in_shop") and poll > 0.7:
        item.eternal = center.eternal_compat
    elif state.modifiers.get("perishables_in_shop") and 0.4 < poll <= 0.7:
        item.perishable = center.perishable_compat
        item.perish_tally = state.perishable_rounds if center.perishable_compat else 0
    if state.modifiers.get("rentals_in_shop"):
        rental_key = f"{'packssjr' if area == 'pack' else 'ssjr'}{ante}"
        if scope.rng.random(rental_key) > 0.7:
            item.rental = True


def create_card_for_shop(scope: Scope) -> Item:
    """Roll a card type by the shop's rates, then create a card of that type."""
    state = scope.state
    rates = (
        ("Joker", state.joker_rate),
        ("Tarot", state.tarot_rate),
        ("Planet", state.planet_rate),
        ("Base", state.playing_card_rate),
        ("Spectral", state.spectral_rate),
    )
    total = sum(rate for _, rate in rates)
    polled = scope.rng.random(f"cdt{state.ante}") * total
    checked = 0.0
    for set_name, rate in rates:
        if checked < polled <= checked + rate:
            return create_card(scope, set_name, "shop", key_append="sho")
        checked += rate
    return create_card(scope, "Joker", "shop", key_append="sho")


def fill_shop(scope: Scope) -> None:
    """Fill the shop's card slots. Source: game.lua ``Game:update_shop``."""
    state = scope.state
    while len(state.shop_cards) < state.shop_joker_max:
        state.shop_cards.append(create_card_for_shop(scope))
