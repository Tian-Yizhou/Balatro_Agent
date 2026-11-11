"""Planet, Tarot and Spectral cards: when they can be used, and what they do.

Source: card.lua ``Card:can_use_consumeable``, ``Card:check_use`` and
``Card:use_consumeable``; functions/misc_functions.lua ``set_consumeable_usage``.
"""

from __future__ import annotations

from collections.abc import Sequence

from .cards import STONE
from .content import CENTERS, pool
from .effects import SUIT_NAMES, Context, ContextKind, calculate_joker
from .effects import Area as ContextArea
from .events import CardDestroyed
from .inventory import add_playing_cards, change_hand_size, destroy_jokers
from .items import copy_item, copy_playing_card, new_playing_card, set_cost
from .scope import Scope
from .scoring import level_up_hand
from .shop import Area, create_card, poll_edition, register_consumables
from .state import HandType, Item, Phase, PlayingCard, RunState, Suit

__all__ = [
    "IMPLEMENTED",
    "can_use",
    "consumable_room",
    "joker_limit",
    "joker_room",
    "use",
]

_TARGETED_PHASES = frozenset({Phase.SELECTING_HAND, Phase.PACK_OPEN})

#: Consumables whose effect is implemented: every Planet, Tarot and Spectral card.
IMPLEMENTED: frozenset[str] = frozenset(
    key for key, center in CENTERS.items() if center.set in ("Planet", "Tarot", "Spectral")
)
register_consumables(IMPLEMENTED.__contains__)

#: Spectral cards that change every card in hand, or destroy some of them, and
#: need at least two cards in hand.
_WHOLE_HAND = frozenset(
    {"c_familiar", "c_grim", "c_incantation", "c_immolate", "c_sigil", "c_ouija"}
)

#: Spectral cards that add a seal to one selected card.
_SEALS = frozenset({"c_talisman", "c_deja_vu", "c_trance", "c_medium"})

#: Spectral cards that destroy a random card in hand and add new ones.
_CARD_CREATORS = frozenset({"c_familiar", "c_grim", "c_incantation"})

#: Ranks the card creators choose from (11-13 are Jack, Queen and King; 14 is Ace).
_CREATED_RANKS = {
    "c_familiar": (11, 12, 13),
    "c_grim": (14,),
    "c_incantation": (2, 3, 4, 5, 6, 7, 8, 9, 10),
}

#: Suits in the order the game lists them for random choices.
_SUIT_ORDER = (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS)


def joker_limit(state: RunState) -> int:
    """How many jokers fit (``G.jokers.config.card_limit``); Negative jokers add one."""
    negatives = sum(item.edition == "e_negative" for item in state.jokers)
    return state.joker_slots + negatives


def joker_room(state: RunState) -> int:
    """Free joker slots (Negative jokers don't take one)."""
    return joker_limit(state) - len(state.jokers)


def consumable_room(state: RunState) -> int:
    negatives = sum(item.edition == "e_negative" for item in state.consumables)
    return state.consumable_slots + negatives - len(state.consumables)


def _editionless_jokers(state: RunState) -> list[Item]:
    """Jokers without an edition, in creation order (the game sorts by ``sort_id``)."""
    return sorted(
        (joker for joker in state.jokers if joker.edition is None), key=lambda joker: joker.uid
    )


def can_use(state: RunState, item: Item, area: Area) -> bool:
    """Whether a consumable can be used now from ``area`` (consumables, shop or pack)."""
    center = CENTERS[item.key]
    config = center.config
    key = item.key
    if key not in IMPLEMENTED or item.debuffed:
        return False
    if center.set == "Planet" or key in ("c_hermit", "c_temperance", "c_black_hole"):
        return True
    if key == "c_wheel_of_fortune":
        return bool(_editionless_jokers(state))
    if key == "c_ankh":
        # Card:check_use refuses Ankh when the joker slots are full.
        limit = joker_limit(state)
        return bool(state.jokers) and limit > 1 and len(state.jokers) < limit
    if key == "c_aura":
        selected = state.selected_cards()
        return len(selected) == 1 and selected[0].edition is None
    if key in ("c_ectoplasm", "c_hex"):
        return bool(_editionless_jokers(state))
    if key in ("c_emperor", "c_high_priestess"):
        return consumable_room(state) > 0 or area == "consumables"
    if key == "c_fool":
        has_room = consumable_room(state) > 0 or area == "consumables"
        return has_room and state.last_tarot_planet not in (None, "c_fool")
    if key in ("c_judgement", "c_soul", "c_wraith"):
        return joker_room(state) > 0
    if state.phase in _TARGETED_PHASES:
        if "max_highlighted" in config:
            selected = len(state.selected_cards())
            most = min(5, int(config["max_highlighted"]))
            if int(config.get("min_highlighted", 1)) <= selected <= most:
                return True
        if key in _WHOLE_HAND and len(state.hand) > 1:
            return True
    return False


def use(scope: Scope, item: Item) -> None:
    """Apply a consumable's effect. The caller has already removed it from its area."""
    state = scope.state
    center = CENTERS[item.key]
    config = center.config
    key = item.key
    state.consumable_usage[key] = state.consumable_usage.get(key, 0) + 1
    selected = state.selected_cards()

    if "hand_type" in config:
        level_up_hand(scope, HandType(config["hand_type"]))
    elif "mod_conv" in config:
        _convert(state, key, str(config["mod_conv"]), selected)
    elif "suit_conv" in config:
        for card in selected:
            card.suit = SUIT_NAMES[str(config["suit_conv"])]
    elif key == "c_hanged_man":
        _destroy_hand_cards(scope, list(reversed(selected)), "hanged_man")
    elif key == "c_hermit":
        scope.change_money(max(0, min(state.money, int(config["extra"]))), "hermit")
    elif key == "c_temperance":
        total = sum(joker.sell_value for joker in state.jokers if CENTERS[joker.key].is_joker)
        scope.change_money(min(total, int(config["extra"])), "temperance")
    elif key in ("c_emperor", "c_high_priestess"):
        emperor = key == "c_emperor"
        count = int(config["tarots" if emperor else "planets"])
        for _ in range(min(count, consumable_room(state))):
            state.consumables.append(
                create_card(
                    scope,
                    "Tarot" if emperor else "Planet",
                    "consumables",
                    key_append="emp" if emperor else "pri",
                )
            )
    elif key in ("c_judgement", "c_soul"):
        soul = key == "c_soul"
        state.jokers.append(
            create_card(
                scope, "Joker", "jokers", legendary=soul, key_append="sou" if soul else "jud"
            )
        )
    elif key == "c_fool":
        if consumable_room(state) > 0 and state.last_tarot_planet is not None:
            state.consumables.append(
                create_card(
                    scope,
                    "Tarot_Planet",
                    "consumables",
                    key_append="fool",
                    forced_key=state.last_tarot_planet,
                )
            )
    elif key == "c_wheel_of_fortune":
        _wheel_of_fortune(scope, int(config["extra"]))
    else:
        _use_spectral(scope, item, selected)

    for card in state.hand:
        card.selected = False
    if center.set in ("Tarot", "Planet"):
        state.last_tarot_planet = key


def _use_spectral(scope: Scope, item: Item, selected: list[PlayingCard]) -> None:
    """The Spectral cards. Source: card.lua ``Card:use_consumeable``, by card name."""
    state = scope.state
    key = item.key
    config = CENTERS[key].config
    if key == "c_black_hole":
        for hand in HandType:
            level_up_hand(scope, hand)
    elif key in _SEALS:
        selected[0].seal = str(config["extra"])
    elif key == "c_aura":
        selected[0].edition = poll_edition(scope, "aura", no_negative=True, guaranteed=True)
    elif key == "c_cryptid":
        copies = [copy_playing_card(state, selected[0]) for _ in range(int(config["extra"]))]
        state.hand.extend(copies)
        add_playing_cards(scope, copies, "hand")
    elif key == "c_sigil":
        suit = scope.rng.choice("sigil", _SUIT_ORDER)
        for card in state.hand:
            card.suit = suit
    elif key == "c_ouija":
        rank = scope.rng.choice("ouija", range(2, 15))
        for card in state.hand:
            card.rank = rank
        change_hand_size(scope, -1)
    elif key in _CARD_CREATORS:
        _destroy_and_create(scope, key, int(config["extra"]))
    elif key == "c_immolate":
        in_hand = sorted(state.hand, key=lambda card: card.uid)
        scope.rng.shuffle("immolate", in_hand)
        extra = config["extra"]
        _destroy_hand_cards(scope, in_hand[: int(extra["destroy"])], "immolate")
        scope.change_money(int(extra["dollars"]), "immolate")
    elif key == "c_ankh":
        _ankh(scope)
    elif key in ("c_hex", "c_ectoplasm"):
        _hex_or_ectoplasm(scope, key)
    elif key == "c_wraith":
        state.jokers.append(create_card(scope, "Joker", "jokers", rarity=0.99, key_append="wra"))
        if state.money != 0:
            scope.change_money(-state.money, "wraith")


def _convert(state: RunState, key: str, conversion: str, cards: list[PlayingCard]) -> None:
    if key == "c_death":
        rightmost = cards[-1]
        for card in cards[:-1]:
            card.rank = rightmost.rank
            card.suit = rightmost.suit
            card.enhancement = rightmost.enhancement
            card.edition = rightmost.edition
            card.seal = rightmost.seal
            card.perma_bonus = rightmost.perma_bonus
    elif key == "c_strength":
        for card in cards:
            card.rank = 2 if card.rank == 14 else min(card.rank + 1, 14)
    else:
        for card in cards:
            card.enhancement = conversion


def _destroy_hand_cards(scope: Scope, cards: Sequence[PlayingCard], reason: str) -> None:
    """Destroy cards in hand; jokers then react to the removed cards."""
    state = scope.state
    state.hand = [card for card in state.hand if card not in cards]
    for card in cards:
        scope.emit(CardDestroyed(card.uid, reason))
    for joker in list(state.jokers):
        calculate_joker(
            scope,
            joker,
            Context(ContextKind.REMOVE_PLAYING_CARDS, ContextArea.JOKERS, removed=list(cards)),
        )


def _destroy_and_create(scope: Scope, key: str, count: int) -> None:
    """Familiar, Grim and Incantation: destroy one random card in hand, then add
    ``count`` new enhanced cards of the card's ranks to the hand."""
    state = scope.state
    victim = scope.rng.choice("random_destroy", sorted(state.hand, key=lambda card: card.uid))
    _destroy_hand_cards(scope, [victim], key.removeprefix("c_"))
    enhancements = [center.key for center in pool("Enhanced") if center.key != STONE]
    created: list[PlayingCard] = []
    for _ in range(count):
        rank = scope.rng.choice(f"{key.removeprefix('c_')}_create", _CREATED_RANKS[key])
        suit = scope.rng.choice(f"{key.removeprefix('c_')}_create", _SUIT_ORDER)
        enhancement = scope.rng.choice("spe_card", enhancements)
        created.append(new_playing_card(state, rank, suit, enhancement))
    state.hand.extend(created)
    add_playing_cards(scope, created, "hand")


def _ankh(scope: Scope) -> None:
    """Copy a random joker and destroy every other joker that isn't eternal."""
    state = scope.state
    chosen = scope.rng.choice("ankh_choice", sorted(state.jokers, key=lambda joker: joker.uid))
    others = [joker for joker in state.jokers if joker is not chosen and not joker.eternal]
    destroy_jokers(scope, others, "ankh")
    state.jokers.append(copy_item(state, chosen, strip_edition=chosen.edition == "e_negative"))


def _hex_or_ectoplasm(scope: Scope, key: str) -> None:
    """Hex makes a random joker Polychrome and destroys the others that aren't
    eternal; Ectoplasm makes one Negative and shrinks the hand, by one more card
    each time it is used."""
    state = scope.state
    target = scope.rng.choice(key.removeprefix("c_"), _editionless_jokers(state))
    if key == "c_hex":
        target.edition = "e_polychrome"
        set_cost(state, target)
        others = [joker for joker in state.jokers if joker is not target and not joker.eternal]
        destroy_jokers(scope, others, "hex")
    else:
        target.edition = "e_negative"
        set_cost(state, target)
        change_hand_size(scope, -state.ecto_minus)
        state.ecto_minus += 1


def _wheel_of_fortune(scope: Scope, odds: int) -> None:
    state = scope.state
    eligible = _editionless_jokers(state)
    if not scope.chance("wheel_of_fortune", odds):
        return
    target = scope.rng.choice("wheel_of_fortune", eligible)
    target.edition = poll_edition(scope, "wheel_of_fortune", no_negative=True, guaranteed=True)
    set_cost(state, target)
