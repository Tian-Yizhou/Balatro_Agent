"""Creating cards: item pools, the shop, and rolled editions and stickers.

Source: functions/common_events.lua ``get_current_pool``, ``poll_edition``,
``create_card`` and ``get_next_voucher_key``; functions/UI_definitions.lua
``create_card_for_shop``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from .content import CENTERS, Center, pool
from .effects import JOKERS, find_joker
from .items import initial_values, new_playing_card, set_cost
from .scope import Scope
from .state import HandType, Item, PlayingCard, RunState, Suit

__all__ = [
    "Area",
    "create_card",
    "create_card_for_shop",
    "create_playing_card",
    "fill_shop",
    "get_next_voucher_key",
    "is_available",
    "poll_edition",
    "register_consumables",
]

Area = Literal["shop", "jokers", "consumables", "pack"]

# The fallback when every item of a pool is unavailable.
_FALLBACK = {
    "Tarot": "c_strength",
    "Tarot_Planet": "c_strength",
    "Planet": "c_pluto",
    "Spectral": "c_incantation",
    "Voucher": "v_blank",
}
_FALLBACK_DEFAULT = "j_joker"

# Every rank and suit, for a random playing card's front (``G.P_CARDS``).
_FRONTS = tuple((rank, suit) for suit in Suit for rank in range(2, 15))

_consumables: Callable[[str], bool] = lambda key: False  # noqa: E731


def register_consumables(is_implemented: Callable[[str], bool]) -> None:
    """Tell the shop which consumables have an implemented effect."""
    global _consumables
    _consumables = is_implemented


def is_available(state: RunState, center: Center) -> bool:
    """Whether ``center`` may be drawn from its pool right now.

    Items the player already has (or that are in the shop) are unavailable unless
    they own Showman, as are items without an implemented effect (ADR 0003).
    A voucher is available until it is redeemed, once the vouchers it upgrades
    have been redeemed; every voucher is implemented.
    """
    key = center.key
    if center.set == "Voucher":
        return _voucher_available(state, center)
    implemented = key in JOKERS if center.is_joker else _consumables(key)
    if not implemented:
        return False
    if key in _used_keys(state) and not find_joker(state, "j_ring_master"):
        return False
    # Runs use the "everything unlocked" mode (plan D6); progression comes later.
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


def _voucher_available(state: RunState, center: Center) -> bool:
    if center.key in state.used_vouchers:
        return False
    if any(required not in state.used_vouchers for required in center.raw.get("requires", ())):
        return False
    # Showman doesn't bring back a voucher that is already for sale.
    return all(item.key != center.key for item in state.shop_vouchers)


def _used_keys(state: RunState) -> set[str]:
    """Keys of every joker and consumable that currently exists (``G.GAME.used_jokers``)."""
    in_pack = state.pack.items if state.pack is not None else ()
    return {item.key for item in (*state.jokers, *state.consumables, *state.shop_cards, *in_pack)}


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
    soulable: bool = False,
) -> Item:
    """Create a joker or consumable of ``set_name``, drawn from its pool.

    ``area`` is where the card is going, which decides whether a shop joker
    rolls stickers. The card is not placed anywhere. A ``soulable`` card (one
    in a booster pack) may instead be The Soul or Black Hole, which never appear
    in their pools.
    """
    state = scope.state
    ante = state.ante
    if forced_key is None and soulable:
        forced_key = _soul_roll(scope, set_name)
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


def _soul_roll(scope: Scope, set_name: str) -> str | None:
    """The rare chance for a pack card to be The Soul or Black Hole.

    Source: functions/common_events.lua ``create_card`` (``soulable``).
    """
    state = scope.state
    key = f"soul_{set_name}{state.ante}"
    showman = bool(find_joker(state, "j_ring_master"))
    used = _used_keys(state)
    forced = None
    # Each check draws from the same stream; the second can override the first.
    soul_allowed = "c_soul" not in used or showman
    if (
        set_name in ("Tarot", "Spectral", "Tarot_Planet")
        and soul_allowed
        and scope.rng.random(key) > 0.997
    ):
        forced = "c_soul"
    black_hole_allowed = "c_black_hole" not in used or showman
    if set_name in ("Planet", "Spectral") and black_hole_allowed and scope.rng.random(key) > 0.997:
        forced = "c_black_hole"
    return forced


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
    return _choose(scope, set_name, members, pool_key)


def _choose(scope: Scope, set_name: str, members: tuple[Center, ...], pool_key: str) -> str:
    """Pick an available member on stream ``pool_key``, resampling past the
    unavailable ones."""
    entries = [center.key if is_available(scope.state, center) else None for center in members]
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


def create_playing_card(scope: Scope, set_name: str, key_append: str) -> PlayingCard:
    """A random playing card: plain for the ``Base`` set, or with a random
    enhancement for the ``Enhanced`` set. It has no edition or seal yet.

    Source: functions/common_events.lua ``create_card`` (the ``front``).
    """
    ante = scope.state.ante
    enhancement = None
    if set_name == "Enhanced":
        enhancements = [center.key for center in pool("Enhanced")]
        enhancement = scope.rng.choice(f"Enhanced{key_append}{ante}", enhancements)
    rank, suit = scope.rng.choice(f"front{key_append}{ante}", _FRONTS)
    return new_playing_card(scope.state, rank, suit, enhancement)


def create_card_for_shop(scope: Scope) -> Item:
    """Roll a card type by the shop's rates, then create a card of that type.

    With Illusion, a playing card may be enhanced and may get an edition.
    """
    state = scope.state
    illusion = "v_illusion" in state.used_vouchers
    total = (
        state.joker_rate
        + state.tarot_rate
        + state.planet_rate
        + state.playing_card_rate
        + state.spectral_rate
    )
    polled = scope.rng.random(f"cdt{state.ante}") * total
    # The game builds this list on every call, rolling Illusion's chance each time.
    playing_set = "Enhanced" if illusion and scope.rng.random("illusion") > 0.6 else "Base"
    rates = (
        ("Joker", state.joker_rate),
        ("Tarot", state.tarot_rate),
        ("Planet", state.planet_rate),
        (playing_set, state.playing_card_rate),
        ("Spectral", state.spectral_rate),
    )
    checked = 0.0
    for set_name, rate in rates:
        if checked < polled <= checked + rate:
            if set_name in ("Base", "Enhanced"):
                return _shop_playing_card(scope, set_name, illusion)
            return create_card(scope, set_name, "shop", key_append="sho")
        checked += rate
    return create_card(scope, "Joker", "shop", key_append="sho")


def _shop_playing_card(scope: Scope, set_name: str, illusion: bool) -> Item:
    card = create_playing_card(scope, set_name, "sho")
    if illusion and scope.rng.random("illusion") > 0.8:
        roll = scope.rng.random("illusion")
        card.edition = "e_polychrome" if roll > 0.85 else "e_holo" if roll > 0.5 else "e_foil"
    item = Item(uid=card.uid, key=card.enhancement or "c_base", playing_card=card)
    set_cost(scope.state, item)
    return item


def get_next_voucher_key(scope: Scope, *, from_tag: bool = False) -> str:
    """Draw the next voucher to offer. Source: functions/common_events.lua
    ``get_next_voucher_key``."""
    # A Voucher Tag's draw has its own stream, the same in every ante.
    pool_key = "Voucher_fromtag" if from_tag else f"Voucher{scope.state.ante}"
    return _choose(scope, "Voucher", pool("Voucher"), pool_key)


def fill_shop(scope: Scope) -> None:
    """Fill the shop's card slots. Source: game.lua ``Game:update_shop``."""
    state = scope.state
    while len(state.shop_cards) < state.shop_joker_max:
        state.shop_cards.append(create_card_for_shop(scope))
