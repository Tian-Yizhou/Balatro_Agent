"""Booster packs: offering them in the shop, opening them, and closing them.

A pack shows ``extra`` cards, of which the player may take ``choose``. Arcana and
Spectral packs also draw a hand from the deck, so their cards have targets; the
hand goes back to the deck when the pack closes. Taking a card from a pack is an
engine step (``engine._pick``), since it may use a consumable.

Source: functions/common_events.lua ``get_pack``; game.lua ``Game:update_shop``
(the booster slots) and ``Game:update_arcana_pack`` (drawing the hand); card.lua
``Card:open``; functions/button_callbacks.lua ``G.FUNCS.use_card``,
``G.FUNCS.skip_booster`` and ``G.FUNCS.end_consumeable``.
"""

from __future__ import annotations

from .content import CENTERS, pool
from .effects import Context, ContextKind, calculate_joker, value
from .events import PackOpened
from .inventory import draw_cards
from .items import new_item, new_playing_card
from .scope import Scope
from .shop import create_card, poll_edition
from .state import Item, OpenPack, Phase, PlayingCard, Suit

__all__ = [
    "BOOSTER_SLOTS",
    "close_pack",
    "fill_boosters",
    "finish_pick",
    "get_pack",
    "is_hand_pack",
    "open_pack",
    "pack_kind",
]

#: Packs offered in each shop. Source: game.lua ``Game:update_shop``.
BOOSTER_SLOTS = 2

# The set of the cards each kind of pack holds, and the ``key_append`` it creates
# them with. Standard packs make playing cards instead (``_standard_card``).
_CONTENTS = {
    "Arcana": ("Tarot", "ar1"),
    "Celestial": ("Planet", "pl1"),
    "Spectral": ("Spectral", "spe"),
    "Buffoon": ("Joker", "buf"),
}

# Every rank and suit, for a Standard pack card's random front (``G.P_CARDS``).
_FRONTS = tuple((rank, suit) for suit in Suit for rank in range(2, 15))


def pack_kind(key: str) -> str:
    """The kind of a pack: Arcana, Celestial, Spectral, Standard or Buffoon."""
    return str(CENTERS[key].raw["kind"])


def is_hand_pack(key: str) -> bool:
    """Whether opening the pack draws a hand from the deck (Arcana and Spectral)."""
    return pack_kind(key) in ("Arcana", "Spectral")


def get_pack(scope: Scope, key: str = "pack_generic", kind: str | None = None) -> str:
    """Draw a pack by the packs' weights; the first one of a run is a Buffoon pack."""
    state = scope.state
    if not state.first_shop_buffoon:
        state.first_shop_buffoon = True
        return scope.rng.choice("first_shop_buffoon", ("p_buffoon_normal_1", "p_buffoon_normal_2"))
    members = [center for center in pool("Booster") if kind is None or center.raw["kind"] == kind]
    weights = [float(center.raw.get("weight", 1)) for center in members]
    poll = scope.rng.random(f"{key}{state.ante}") * sum(weights)
    cumulative = 0.0
    for center, weight in zip(members, weights, strict=True):
        cumulative += weight
        if cumulative >= poll and cumulative - weight <= poll:
            return center.key
    return members[-1].key


def fill_boosters(scope: Scope) -> None:
    """Put this round's packs in the shop."""
    state = scope.state
    state.shop_boosters = [
        new_item(state, get_pack(scope, "shop_pack")) for _ in range(BOOSTER_SLOTS)
    ]


def open_pack(scope: Scope, pack: Item, return_phase: Phase) -> None:
    """Pay for ``pack``, fill it, and wait for the player's picks.

    Source: card.lua ``Card:open``.
    """
    state = scope.state
    opened = OpenPack(
        key=pack.key,
        items=[],
        playing_cards=[],
        choices_left=int(CENTERS[pack.key].config.get("choose", 1)),
        return_phase=return_phase,
    )
    state.pack = opened
    scope.set_phase(Phase.PACK_OPEN)
    if pack.cost > 0:
        scope.change_money(-pack.cost, "pack")
    kind = pack_kind(pack.key)
    for _ in range(int(value(pack, "extra"))):
        if kind == "Standard":
            opened.playing_cards.append(_standard_card(scope))
        else:
            set_name, key_append = _CONTENTS[kind]
            opened.items.append(
                create_card(scope, set_name, "pack", key_append=key_append, soulable=True)
            )
    cards = (*opened.items, *opened.playing_cards)
    scope.emit(PackOpened(pack.uid, pack.key, tuple(card.uid for card in cards)))
    for joker in list(state.jokers):
        calculate_joker(scope, joker, Context(ContextKind.OPEN_BOOSTER, card=pack))
    scope.flush()
    if is_hand_pack(pack.key):
        draw_cards(scope, state.hand_size - len(state.hand))


def _standard_card(scope: Scope) -> PlayingCard:
    """A random playing card for a Standard pack: maybe enhanced, with an edition
    and a seal at their own odds."""
    state = scope.state
    ante = state.ante
    enhancement = None
    if scope.rng.random(f"stdset{ante}") > 0.6:
        enhancements = [center.key for center in pool("Enhanced")]
        enhancement = scope.rng.choice(f"Enhancedsta{ante}", enhancements)
    rank, suit = scope.rng.choice(f"frontsta{ante}", _FRONTS)
    card = new_playing_card(state, rank, suit, enhancement)
    card.edition = poll_edition(scope, f"standard_edition{ante}", mod=2, no_negative=True)
    seal_rate = 10
    if scope.rng.random(f"stdseal{ante}") > 1 - 0.02 * seal_rate:
        roll = scope.rng.random(f"stdsealtype{ante}")
        if roll > 0.75:
            card.seal = "Red"
        elif roll > 0.5:
            card.seal = "Blue"
        elif roll > 0.25:
            card.seal = "Gold"
        else:
            card.seal = "Purple"
    return card


def finish_pick(scope: Scope) -> None:
    """Count a card taken from the pack; close the pack after the last choice."""
    pack = scope.state.pack
    assert pack is not None
    if pack.choices_left > 1 and pack.size > 0:
        pack.choices_left -= 1
    else:
        close_pack(scope)


def close_pack(scope: Scope) -> None:
    """Return the hand to the bottom of the deck, discard the rest of the pack, and
    go back to the phase the pack was opened from."""
    state = scope.state
    pack = state.pack
    assert pack is not None
    # Each card leaves from the left of the hand and goes under the deck.
    for card in state.hand:
        card.selected = False
        state.deck.insert(0, card)
    state.hand.clear()
    state.pack = None
    scope.set_phase(pack.return_phase)
