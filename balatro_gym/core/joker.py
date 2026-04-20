"""Joker system: registry, base class, score modification, and joker definitions.

Reference: Balatro Lua source — card.lua (calculate_joker), game.lua (joker configs).

The scoring pipeline calls jokers in several contexts:
1. `before` — called before individual card scoring (stat updates)
2. `individual` — called per scoring card (returns chips/mult/x_mult per card)
3. `held_individual` — called per held-in-hand card
4. `main` — called once for the whole hand (returns chip_mod/mult_mod/Xmult_mod)
5. `after` — called after scoring (stat updates, e.g. Ice Cream losing chips)
6. `end_of_round` — called after beating a blind
7. `on_discard` — called when cards are discarded
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from balatro_gym.core.card import Card, Suit, Rank
from balatro_gym.core.hand_evaluator import HandResult, HandType


# ---------------------------------------------------------------------------
# Score modification (returned by jokers during scoring)
# ---------------------------------------------------------------------------

@dataclass
class ScoreModification:
    """Modification to chips/mult from a joker effect.

    For individual context: chips, mult, x_mult are per-card effects.
    For main context: chip_mod, mult_mod, Xmult_mod are whole-hand effects.
    """
    # Per-card (individual context)
    chips: int = 0
    mult: int = 0
    x_mult: float = 0.0
    # Per-joker (main context)
    chip_mod: int = 0
    mult_mod: int = 0
    Xmult_mod: float = 0.0
    # Money
    dollars: int = 0
    # Dollar bonus (end of round)
    dollar_bonus: int = 0


# ---------------------------------------------------------------------------
# Joker metadata
# ---------------------------------------------------------------------------

@dataclass
class JokerInfo:
    id: str
    name: str
    description: str
    rarity: int          # 1=common, 2=uncommon, 3=rare, 4=legendary
    cost: int


# ---------------------------------------------------------------------------
# GameStateView protocol (avoids circular import with game_state.py)
# ---------------------------------------------------------------------------

@runtime_checkable
class GameStateView(Protocol):
    """Read-only view of the game state. Implemented by game_state.GameStateSnapshot."""
    hand: list[Card]
    jokers: list[BaseJoker]
    money: int
    ante: int
    blind_type: str
    score_target: int
    current_score: int
    hands_remaining: int
    discards_remaining: int
    deck_size: int
    max_jokers: int
    hands_played_this_round: int
    hand_type_played_counts: dict[str, int]


# ---------------------------------------------------------------------------
# Joker registry
# ---------------------------------------------------------------------------

_JOKER_REGISTRY: dict[str, type[BaseJoker]] = {}


def register_joker(cls: type[BaseJoker]) -> type[BaseJoker]:
    """Class decorator to register a joker in the global registry."""
    _JOKER_REGISTRY[cls.INFO.id] = cls
    return cls


def get_joker_class(joker_id: str) -> type[BaseJoker]:
    """Get joker class by ID. Raises KeyError if not found."""
    return _JOKER_REGISTRY[joker_id]


def get_all_joker_ids() -> list[str]:
    """Return all registered joker IDs."""
    return list(_JOKER_REGISTRY.keys())


def create_joker(joker_id: str) -> BaseJoker:
    """Create a joker instance by ID."""
    return _JOKER_REGISTRY[joker_id]()


def get_jokers_by_rarity(rarity: int) -> list[str]:
    """Return joker IDs filtered by rarity."""
    return [jid for jid, cls in _JOKER_REGISTRY.items() if cls.INFO.rarity == rarity]


# ---------------------------------------------------------------------------
# Base joker class
# ---------------------------------------------------------------------------

class BaseJoker:
    """Base class for all jokers.

    Subclasses override hook methods for their specific effects.
    The scoring pipeline calls these in order: before -> individual -> held_individual -> main -> after.
    """

    INFO: ClassVar[JokerInfo]

    def __init__(self) -> None:
        self._internal_state: dict[str, Any] = {}

    def on_before(self, hand_result: HandResult, scoring_hand: list[Card],
                  full_hand: list[Card], poker_hands: dict[str, bool],
                  scoring_name: str, view: GameStateView) -> ScoreModification | None:
        """Called before individual card scoring. For stat updates (Runner, Ride the Bus, etc.)."""
        return None

    def on_individual(self, card: Card, hand_result: HandResult,
                      scoring_hand: list[Card], scoring_name: str,
                      view: GameStateView) -> ScoreModification | None:
        """Called per scoring card. Returns per-card chips/mult/x_mult."""
        return None

    def on_held_individual(self, card: Card, hand_result: HandResult,
                           scoring_hand: list[Card], scoring_name: str,
                           view: GameStateView) -> ScoreModification | None:
        """Called per held-in-hand card."""
        return None

    def on_main(self, hand_result: HandResult, scoring_hand: list[Card],
                full_hand: list[Card], poker_hands: dict[str, bool],
                scoring_name: str, view: GameStateView) -> ScoreModification | None:
        """Called once for main joker scoring. Returns chip_mod/mult_mod/Xmult_mod."""
        return None

    def on_after(self, hand_result: HandResult, scoring_hand: list[Card],
                 scoring_name: str, view: GameStateView) -> ScoreModification | None:
        """Called after scoring is complete. For stat updates (Ice Cream, etc.)."""
        return None

    def on_discard(self, discarded: list[Card], view: GameStateView) -> dict[str, Any]:
        """Called when cards are discarded."""
        return {}

    def on_end_of_round(self, view: GameStateView) -> dict[str, Any]:
        """Called after beating a blind."""
        return {}

    def on_round_start(self, view: GameStateView) -> None:
        """Called at the start of a blind."""
        return

    def calculate_dollar_bonus(self, view: GameStateView) -> int:
        """Return dollar bonus at end of round (e.g., Delayed Gratification)."""
        return 0

    def passive_effects(self, view: GameStateView) -> dict[str, Any]:
        """Return passive modifications like {'hand_size': +1}."""
        return {}

    def get_state(self) -> dict[str, Any]:
        """Serialize internal state for observation."""
        return dict(self._internal_state)

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore internal state from a dict (as returned by ``get_state``)."""
        self._internal_state.update(state)

    def __repr__(self) -> str:
        return f"{self.INFO.name} ({self.INFO.id})"


# ===========================================================================
# PRIORITY 1 JOKERS (10 jokers — simple, for easy difficulty)
# ===========================================================================

@register_joker
class BasicJoker(BaseJoker):
    """Lua: j_joker, config = {mult = 4}. +4 Mult always."""
    INFO = JokerInfo(id="joker_basic", name="Joker", description="+4 Mult",
                     rarity=1, cost=2)

    def on_main(self, hand_result, scoring_hand, full_hand, poker_hands,
                scoring_name, view):
        return ScoreModification(mult_mod=4)


@register_joker
class GreedyJoker(BaseJoker):
    """Lua: j_greedy_joker, individual context, +3 Mult per played Diamond."""
    INFO = JokerInfo(id="greedy_joker", name="Greedy Joker",
                     description="+3 Mult for each played Diamond card",
                     rarity=1, cost=5)

    def on_individual(self, card, hand_result, scoring_hand, scoring_name, view):
        if (card.suit == Suit.DIAMONDS or card.is_wild) and not card.face_down:
            return ScoreModification(mult=3)
        return None


@register_joker
class LustyJoker(BaseJoker):
    """Lua: j_lusty_joker, individual context, +3 Mult per played Heart."""
    INFO = JokerInfo(id="lusty_joker", name="Lusty Joker",
                     description="+3 Mult for each played Heart card",
                     rarity=1, cost=5)

    def on_individual(self, card, hand_result, scoring_hand, scoring_name, view):
        if (card.suit == Suit.HEARTS or card.is_wild) and not card.face_down:
            return ScoreModification(mult=3)
        return None


@register_joker
class WrathfulJoker(BaseJoker):
    """Lua: j_wrathful_joker, individual context, +3 Mult per played Spade."""
    INFO = JokerInfo(id="wrathful_joker", name="Wrathful Joker",
                     description="+3 Mult for each played Spade card",
                     rarity=1, cost=5)

    def on_individual(self, card, hand_result, scoring_hand, scoring_name, view):
        if (card.suit == Suit.SPADES or card.is_wild) and not card.face_down:
            return ScoreModification(mult=3)
        return None


@register_joker
class GluttonousJoker(BaseJoker):
    """Lua: j_gluttenous_joker, individual context, +3 Mult per played Club."""
    INFO = JokerInfo(id="gluttonous_joker", name="Gluttonous Joker",
                     description="+3 Mult for each played Club card",
                     rarity=1, cost=5)

    def on_individual(self, card, hand_result, scoring_hand, scoring_name, view):
        if (card.suit == Suit.CLUBS or card.is_wild) and not card.face_down:
            return ScoreModification(mult=3)
        return None


@register_joker
class JollyJoker(BaseJoker):
    """Lua: j_jolly, config = {t_mult = 8, type = 'Pair'}. +8 Mult if hand contains Pair."""
    INFO = JokerInfo(id="jolly_joker", name="Jolly Joker",
                     description="+8 Mult if played hand contains a Pair",
                     rarity=1, cost=3)

    def on_main(self, hand_result, scoring_hand, full_hand, poker_hands,
                scoring_name, view):
        if poker_hands.get("Pair", False):
            return ScoreModification(mult_mod=8)
        return None


@register_joker
class ZanyJoker(BaseJoker):
    """Lua: j_zany, config = {t_mult = 12, type = 'Three of a Kind'}."""
    INFO = JokerInfo(id="zany_joker", name="Zany Joker",
                     description="+12 Mult if played hand contains Three of a Kind",
                     rarity=1, cost=4)

    def on_main(self, hand_result, scoring_hand, full_hand, poker_hands,
                scoring_name, view):
        if poker_hands.get("Three of a Kind", False):
            return ScoreModification(mult_mod=12)
        return None


@register_joker
class Banner(BaseJoker):
    """Lua: j_banner, config = {extra = 30}. +30 Chips per discard remaining."""
    INFO = JokerInfo(id="banner", name="Banner",
                     description="+30 Chips for each discard remaining",
                     rarity=1, cost=5)

    def on_main(self, hand_result, scoring_hand, full_hand, poker_hands,
                scoring_name, view):
        if view.discards_remaining > 0:
            return ScoreModification(chip_mod=view.discards_remaining * 30)
        return None


@register_joker
class MysticSummit(BaseJoker):
    """Lua: j_mystic_summit, config = {extra = {mult = 15, d_remaining = 0}}."""
    INFO = JokerInfo(id="mystic_summit", name="Mystic Summit",
                     description="+15 Mult when 0 discards remaining",
                     rarity=1, cost=5)

    def on_main(self, hand_result, scoring_hand, full_hand, poker_hands,
                scoring_name, view):
        if view.discards_remaining == 0:
            return ScoreModification(mult_mod=15)
        return None


@register_joker
class IceCream(BaseJoker):
    """Lua: j_ice_cream, config = {extra = {chips = 100, chip_mod = 5}}.
    +chips to scoring (via main), -5 chips after each hand (via after).
    Self-destructs when chips reach 0 (handled by game_state).
    """
    INFO = JokerInfo(id="ice_cream", name="Ice Cream",
                     description="+100 Chips. -5 Chips for every hand played",
                     rarity=1, cost=5)

    def __init__(self) -> None:
        super().__init__()
        self._internal_state["chips"] = 100

    def on_main(self, hand_result, scoring_hand, full_hand, poker_hands,
                scoring_name, view):
        chips = self._internal_state["chips"]
        if chips > 0:
            return ScoreModification(chip_mod=chips)
        return None

    def on_after(self, hand_result, scoring_hand, scoring_name, view):
        self._internal_state["chips"] = max(0, self._internal_state["chips"] - 5)
        return None


# ===========================================================================
# PRIORITY 2 JOKERS (10 more — for medium difficulty)
# ===========================================================================

@register_joker
class RaisedFist(BaseJoker):
    """Lua: j_raised_fist. Held card individual: adds 2x nominal of lowest held card as Mult."""
    INFO = JokerInfo(id="raised_fist", name="Raised Fist",
                     description="Adds double the rank of lowest held card as Mult",
                     rarity=1, cost=5)

    def on_held_individual(self, card, hand_result, scoring_hand, scoring_name, view):
        # Find the lowest-ID non-debuffed held card
        held_cards = [c for c in view.hand if c not in scoring_hand]
        if not held_cards:
            return None
        lowest = min(held_cards, key=lambda c: c.id)
        if card is lowest or (card.id == lowest.id and card.suit == lowest.suit):
            if card.face_down:
                return None
            return ScoreModification(mult=2 * card.nominal)
        return None


@register_joker
class Fibonacci(BaseJoker):
    """Lua: j_fibonacci, individual, +8 Mult per Ace/2/3/5/8 played."""
    INFO = JokerInfo(id="fibonacci", name="Fibonacci",
                     description="+8 Mult for each played Ace, 2, 3, 5, or 8",
                     rarity=2, cost=8)

    _FIB_IDS = {2, 3, 5, 8, 14}  # Card IDs

    def on_individual(self, card, hand_result, scoring_hand, scoring_name, view):
        if card.id in self._FIB_IDS and not card.face_down:
            return ScoreModification(mult=8)
        return None


@register_joker
class EvenSteven(BaseJoker):
    """Lua: j_even_steven, individual, +4 Mult per even-ranked played card (2,4,6,8,10)."""
    INFO = JokerInfo(id="even_steven", name="Even Steven",
                     description="+4 Mult for each played even-ranked card",
                     rarity=1, cost=4)

    def on_individual(self, card, hand_result, scoring_hand, scoring_name, view):
        cid = card.id
        if 0 <= cid <= 10 and cid % 2 == 0 and not card.face_down:
            return ScoreModification(mult=4)
        return None


@register_joker
class OddTodd(BaseJoker):
    """Lua: j_odd_todd, individual, +31 Chips per odd-ranked played card (3,5,7,9,Ace)."""
    INFO = JokerInfo(id="odd_todd", name="Odd Todd",
                     description="+31 Chips for each played odd-ranked card",
                     rarity=1, cost=4)

    def on_individual(self, card, hand_result, scoring_hand, scoring_name, view):
        cid = card.id
        if not card.face_down:
            if (0 <= cid <= 10 and cid % 2 == 1) or cid == 14:  # Ace is odd
                return ScoreModification(chips=31)
        return None


@register_joker
class Scholar(BaseJoker):
    """Lua: j_scholar, individual, +20 Chips and +4 Mult per Ace played."""
    INFO = JokerInfo(id="scholar", name="Scholar",
                     description="+20 Chips and +4 Mult for each played Ace",
                     rarity=1, cost=4)

    def on_individual(self, card, hand_result, scoring_hand, scoring_name, view):
        if card.id == 14 and not card.face_down:
            return ScoreModification(chips=20, mult=4)
        return None


@register_joker
class BusinessCard(BaseJoker):
    """Lua: j_business, individual, face cards have 1 in 2 chance to give $2."""
    INFO = JokerInfo(id="business_card", name="Business Card",
                     description="Played face cards have a 1 in 2 chance to give $2 each",
                     rarity=1, cost=4)

    def on_individual(self, card, hand_result, scoring_hand, scoring_name, view):
        if card.is_face_card and not card.face_down:
            # Probability handled by game_state via dollars field
            return ScoreModification(dollars=2)
        return None


@register_joker
class Stencil(BaseJoker):
    """Lua: j_stencil. X(1+empty_slots) Mult. Uses generic x_mult path."""
    INFO = JokerInfo(id="stencil", name="Joker Stencil",
                     description="X1 Mult for each empty Joker slot",
                     rarity=2, cost=8)

    def on_main(self, hand_result, scoring_hand, full_hand, poker_hands,
                scoring_name, view):
        empty_slots = view.max_jokers - len(view.jokers)
        # Stencil itself counts as a joker, but it contributes X1 per OTHER empty slot
        # Plus X1 for itself being the stencil = total Xmult of (empty + 1)?
        # Lua: the x_mult is set elsewhere. Config is empty, x_mult starts at 1.
        # Actually from Lua code, Stencil uses the generic x_mult > 1 path.
        # Its x_mult is dynamically calculated as empty_slots + 1.
        x = empty_slots + 1
        if x > 1:
            return ScoreModification(Xmult_mod=float(x))
        return None


@register_joker
class HalfJoker(BaseJoker):
    """Lua: j_half, config = {extra = {mult = 20, size = 3}}.
    +20 Mult if played hand contains 3 or fewer cards (full_hand, not scoring_hand).
    """
    INFO = JokerInfo(id="half_joker", name="Half Joker",
                     description="+20 Mult if played hand contains 3 or fewer cards",
                     rarity=1, cost=5)

    def on_main(self, hand_result, scoring_hand, full_hand, poker_hands,
                scoring_name, view):
        if len(full_hand) <= 3:
            return ScoreModification(mult_mod=20)
        return None


@register_joker
class Blueprint(BaseJoker):
    """Lua: j_blueprint. Copies the ability of the Joker to the right."""
    INFO = JokerInfo(id="blueprint", name="Blueprint",
                     description="Copies the ability of the Joker to the right",
                     rarity=3, cost=10)

    def _get_right_neighbor(self, view: GameStateView) -> BaseJoker | None:
        for i, j in enumerate(view.jokers):
            if j is self and i + 1 < len(view.jokers):
                return view.jokers[i + 1]
        return None

    def on_before(self, hand_result, scoring_hand, full_hand, poker_hands,
                  scoring_name, view):
        neighbor = self._get_right_neighbor(view)
        if neighbor:
            return neighbor.on_before(hand_result, scoring_hand, full_hand,
                                      poker_hands, scoring_name, view)
        return None

    def on_individual(self, card, hand_result, scoring_hand, scoring_name, view):
        neighbor = self._get_right_neighbor(view)
        if neighbor:
            return neighbor.on_individual(card, hand_result, scoring_hand,
                                          scoring_name, view)
        return None

    def on_held_individual(self, card, hand_result, scoring_hand, scoring_name, view):
        neighbor = self._get_right_neighbor(view)
        if neighbor:
            return neighbor.on_held_individual(card, hand_result, scoring_hand,
                                               scoring_name, view)
        return None

    def on_main(self, hand_result, scoring_hand, full_hand, poker_hands,
                scoring_name, view):
        neighbor = self._get_right_neighbor(view)
        if neighbor:
            return neighbor.on_main(hand_result, scoring_hand, full_hand,
                                    poker_hands, scoring_name, view)
        return None


@register_joker
class DNA(BaseJoker):
    """Lua: j_dna. Before context: if first hand of round and 1 card played,
    copy first card played to hand/deck."""
    INFO = JokerInfo(id="dna", name="DNA",
                     description="If first hand of round has 1 card, create a copy of it",
                     rarity=3, cost=8)

    def on_before(self, hand_result, scoring_hand, full_hand, poker_hands,
                  scoring_name, view):
        if view.hands_played_this_round == 0 and len(full_hand) == 1:
            first_card = full_hand[0]
            return ScoreModification()  # Side effect handled by game_state
        return None


# ===========================================================================
# PRIORITY 3 JOKERS (10 more — for hard difficulty)
# ===========================================================================

@register_joker
class AbstractJoker(BaseJoker):
    """Lua: j_abstract, config = {extra = 3}. +3 Mult per Joker held."""
    INFO = JokerInfo(id="abstract_joker", name="Abstract Joker",
                     description="+3 Mult for each Joker held",
                     rarity=1, cost=4)

    def on_main(self, hand_result, scoring_hand, full_hand, poker_hands,
                scoring_name, view):
        joker_count = sum(1 for j in view.jokers if isinstance(j, BaseJoker))
        return ScoreModification(mult_mod=joker_count * 3)


@register_joker
class Blackboard(BaseJoker):
    """Lua: j_blackboard, config = {extra = 3}. X3 Mult if all held cards are Spades or Clubs."""
    INFO = JokerInfo(id="blackboard", name="Blackboard",
                     description="X3 Mult if all held-in-hand cards are Spades or Clubs",
                     rarity=2, cost=6)

    def on_main(self, hand_result, scoring_hand, full_hand, poker_hands,
                scoring_name, view):
        held = [c for c in view.hand if c not in scoring_hand]
        if not held or all(c.suit in (Suit.SPADES, Suit.CLUBS) for c in held):
            return ScoreModification(Xmult_mod=3.0)
        return None


@register_joker
class TheDuo(BaseJoker):
    """Lua: j_duo, config = {Xmult = 2, type = 'Pair'}. X2 Mult if Pair."""
    INFO = JokerInfo(id="the_duo", name="The Duo",
                     description="X2 Mult if played hand contains a Pair",
                     rarity=3, cost=8)

    def on_main(self, hand_result, scoring_hand, full_hand, poker_hands,
                scoring_name, view):
        if poker_hands.get("Pair", False):
            return ScoreModification(Xmult_mod=2.0)
        return None


@register_joker
class TheTrio(BaseJoker):
    """Lua: j_trio, config = {Xmult = 3, type = 'Three of a Kind'}."""
    INFO = JokerInfo(id="the_trio", name="The Trio",
                     description="X3 Mult if played hand contains Three of a Kind",
                     rarity=3, cost=8)

    def on_main(self, hand_result, scoring_hand, full_hand, poker_hands,
                scoring_name, view):
        if poker_hands.get("Three of a Kind", False):
            return ScoreModification(Xmult_mod=3.0)
        return None


@register_joker
class TheFamily(BaseJoker):
    """Lua: j_family, config = {Xmult = 4, type = 'Four of a Kind'}."""
    INFO = JokerInfo(id="the_family", name="The Family",
                     description="X4 Mult if played hand contains Four of a Kind",
                     rarity=3, cost=8)

    def on_main(self, hand_result, scoring_hand, full_hand, poker_hands,
                scoring_name, view):
        if poker_hands.get("Four of a Kind", False):
            return ScoreModification(Xmult_mod=4.0)
        return None


@register_joker
class LoyaltyCard(BaseJoker):
    """Lua: j_loyalty_card, config = {extra = {Xmult = 4, every = 5}}.
    X4 Mult every 6th hand played (0-indexed: triggers when loyalty_remaining cycles to `every`).
    """
    INFO = JokerInfo(id="loyalty_card", name="Loyalty Card",
                     description="X4 Mult every 6 hands played",
                     rarity=2, cost=5)

    def __init__(self) -> None:
        super().__init__()
        self._internal_state["hands_played_at_create"] = 0
        self._internal_state["every"] = 5

    def on_main(self, hand_result, scoring_hand, full_hand, poker_hands,
                scoring_name, view):
        # Lua formula: loyalty_remaining = (every - 1 - (hands_played - hands_played_at_create)) % (every + 1)
        # Triggers when loyalty_remaining == every (which means it cycles back)
        every = self._internal_state["every"]
        total = view.hands_played_this_round  # Simplified: use round hands
        # Actually in Lua it uses G.GAME.hands_played (total hands played in the run)
        # For simplicity, we track internally
        hands_since = self._internal_state.get("total_hands_seen", 0)
        loyalty_remaining = (every - hands_since) % (every + 1)
        if loyalty_remaining == every:
            return ScoreModification(Xmult_mod=4.0)
        return None

    def on_after(self, hand_result, scoring_hand, scoring_name, view):
        self._internal_state["total_hands_seen"] = self._internal_state.get("total_hands_seen", 0) + 1
        return None


@register_joker
class CeremonialDagger(BaseJoker):
    """Lua: j_ceremonial, config = {mult = 0}.
    At blind start (setting_blind), slices the joker to its right, gaining 2x its sell cost as mult.
    During scoring, adds accumulated mult.
    """
    INFO = JokerInfo(id="ceremonial_dagger", name="Ceremonial Dagger",
                     description="When blind is set, destroy right joker and gain mult",
                     rarity=2, cost=6)

    def __init__(self) -> None:
        super().__init__()
        self._internal_state["mult"] = 0

    def on_main(self, hand_result, scoring_hand, full_hand, poker_hands,
                scoring_name, view):
        m = self._internal_state["mult"]
        if m > 0:
            return ScoreModification(mult_mod=m)
        return None


@register_joker
class RideTheBus(BaseJoker):
    """Lua: j_ride_the_bus, config = {extra = 1}.
    Before context: +1 mult per consecutive hand without face card.
    Resets to 0 when a face card is in the scoring hand.
    """
    INFO = JokerInfo(id="ride_the_bus", name="Ride the Bus",
                     description="+1 Mult per consecutive hand without a face card",
                     rarity=1, cost=6)

    def __init__(self) -> None:
        super().__init__()
        self._internal_state["mult"] = 0

    def on_before(self, hand_result, scoring_hand, full_hand, poker_hands,
                  scoring_name, view):
        has_face = any(c.is_face_card for c in scoring_hand)
        if has_face:
            self._internal_state["mult"] = 0
        else:
            self._internal_state["mult"] += 1
        return None

    def on_main(self, hand_result, scoring_hand, full_hand, poker_hands,
                scoring_name, view):
        m = self._internal_state["mult"]
        if m > 0:
            return ScoreModification(mult_mod=m)
        return None


@register_joker
class Runner(BaseJoker):
    """Lua: j_runner, config = {extra = {chips = 0, chip_mod = 15}}.
    Before context: gains +15 chips permanently if played hand contains a Straight.
    Main context: adds accumulated chips.
    """
    INFO = JokerInfo(id="runner", name="Runner",
                     description="Gains +15 Chips if played hand contains a Straight",
                     rarity=1, cost=5)

    def __init__(self) -> None:
        super().__init__()
        self._internal_state["chips"] = 0

    def on_before(self, hand_result, scoring_hand, full_hand, poker_hands,
                  scoring_name, view):
        if poker_hands.get("Straight", False):
            self._internal_state["chips"] += 15
        return None

    def on_main(self, hand_result, scoring_hand, full_hand, poker_hands,
                scoring_name, view):
        c = self._internal_state["chips"]
        if c > 0:
            return ScoreModification(chip_mod=c)
        return None


@register_joker
class Supernova(BaseJoker):
    """Lua: j_supernova, config = {extra = 1}.
    +Mult equal to the number of times the played hand type has been played this run.
    Uses the count BEFORE incrementing for this hand (Lua reads G.GAME.hands[name].played
    which was already incremented at the top of evaluate_play).
    """
    INFO = JokerInfo(id="supernova", name="Supernova",
                     description="+Mult for times this hand type has been played",
                     rarity=1, cost=5)

    def on_main(self, hand_result, scoring_hand, full_hand, poker_hands,
                scoring_name, view):
        count = view.hand_type_played_counts.get(scoring_name, 0)
        if count > 0:
            return ScoreModification(mult_mod=count)
        return None
