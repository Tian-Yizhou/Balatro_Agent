"""Planet and Tarot cards: when they can be used, and what they do.

Source: card.lua ``Card:can_use_consumeable`` and ``Card:use_consumeable``;
functions/misc_functions.lua ``set_consumeable_usage``.
"""

from __future__ import annotations

from .content import CENTERS
from .effects import SUIT_NAMES, Context, ContextKind, calculate_joker
from .effects import Area as ContextArea
from .events import CardDestroyed
from .items import set_cost
from .scope import Scope
from .scoring import level_up_hand
from .shop import Area, create_card, poll_edition, register_consumables
from .state import HandType, Item, Phase, PlayingCard, RunState

__all__ = [
    "IMPLEMENTED",
    "can_use",
    "consumable_room",
    "joker_room",
    "use",
]

_TARGETED_PHASES = frozenset({Phase.SELECTING_HAND, Phase.PACK_OPEN})


#: Consumables whose effect is implemented: every Planet and every Tarot.
IMPLEMENTED: frozenset[str] = frozenset(
    key for key, center in CENTERS.items() if center.set in ("Planet", "Tarot")
)
register_consumables(IMPLEMENTED.__contains__)


def joker_room(state: RunState) -> int:
    """Free joker slots (Negative jokers don't take one)."""
    negatives = sum(item.edition == "e_negative" for item in state.jokers)
    return state.joker_slots + negatives - len(state.jokers)


def consumable_room(state: RunState) -> int:
    negatives = sum(item.edition == "e_negative" for item in state.consumables)
    return state.consumable_slots + negatives - len(state.consumables)


def can_use(state: RunState, item: Item, area: Area) -> bool:
    """Whether a consumable can be used now from ``area`` (consumables or shop)."""
    center = CENTERS[item.key]
    config = center.config
    if item.key not in IMPLEMENTED or item.debuffed:
        return False
    if center.set == "Planet" or item.key in ("c_hermit", "c_temperance"):
        return True
    if item.key == "c_wheel_of_fortune":
        return any(CENTERS[joker.key].is_joker and joker.edition is None for joker in state.jokers)
    if item.key in ("c_emperor", "c_high_priestess"):
        return consumable_room(state) > 0 or area == "consumables"
    if item.key == "c_fool":
        has_room = consumable_room(state) > 0 or area == "consumables"
        return has_room and state.last_tarot_planet not in (None, "c_fool")
    if item.key == "c_judgement":
        return joker_room(state) > 0
    if state.phase in _TARGETED_PHASES and "max_highlighted" in config:
        selected = sum(card.selected for card in state.hand)
        most = min(5, int(config["max_highlighted"]))
        return int(config.get("min_highlighted", 1)) <= selected <= most
    return False


def use(scope: Scope, item: Item) -> None:
    """Apply a consumable's effect. The caller has already removed it from its area."""
    state = scope.state
    center = CENTERS[item.key]
    config = center.config
    state.consumable_usage[item.key] = state.consumable_usage.get(item.key, 0) + 1
    selected = [card for card in state.hand if card.selected]

    if "hand_type" in config:
        level_up_hand(scope, HandType(config["hand_type"]))
    elif "mod_conv" in config:
        _convert(state, item.key, str(config["mod_conv"]), selected)
    elif "suit_conv" in config:
        for card in selected:
            card.suit = SUIT_NAMES[str(config["suit_conv"])]
    elif config.get("remove_card"):
        _hanged_man(scope, selected)
    elif item.key == "c_hermit":
        scope.change_money(max(0, min(state.money, int(config["extra"]))), "hermit")
    elif item.key == "c_temperance":
        total = sum(joker.sell_value for joker in state.jokers if CENTERS[joker.key].is_joker)
        scope.change_money(min(total, int(config["extra"])), "temperance")
    elif item.key in ("c_emperor", "c_high_priestess"):
        emperor = item.key == "c_emperor"
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
    elif item.key == "c_judgement":
        state.jokers.append(create_card(scope, "Joker", "jokers", key_append="jud"))
    elif item.key == "c_fool":
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
    elif item.key == "c_wheel_of_fortune":
        _wheel_of_fortune(scope, int(config["extra"]))

    for card in state.hand:
        card.selected = False
    if center.set in ("Tarot", "Planet"):
        state.last_tarot_planet = item.key


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


def _hanged_man(scope: Scope, cards: list[PlayingCard]) -> None:
    state = scope.state
    state.hand = [card for card in state.hand if card not in cards]
    for card in reversed(cards):
        scope.emit(CardDestroyed(card.uid, "hanged_man"))
    for joker in list(state.jokers):
        calculate_joker(
            scope,
            joker,
            Context(ContextKind.REMOVE_PLAYING_CARDS, ContextArea.JOKERS, removed=cards),
        )


def _wheel_of_fortune(scope: Scope, odds: int) -> None:
    state = scope.state
    eligible = sorted(
        (joker for joker in state.jokers if CENTERS[joker.key].is_joker and joker.edition is None),
        key=lambda joker: joker.uid,
    )
    if not scope.chance("wheel_of_fortune", odds):
        return
    target = scope.rng.choice("wheel_of_fortune", eligible)
    target.edition = poll_edition(scope, "wheel_of_fortune", no_negative=True, guaranteed=True)
    set_cost(state, target)
