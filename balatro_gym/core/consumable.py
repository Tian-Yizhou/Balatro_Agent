"""Consumable system: Tarot, Planet, and Spectral cards.

Reference: Balatro Lua source — card.lua (use_consumeable, lines 1091-1240),
game.lua (consumable definitions, lines 530-588),
common_events.lua (level_up_hand, lines 464-493).

Registry-based design mirrors joker.py for extensibility.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from balatro_gym.core.card import (
    Card, Enhancement, Edition, Seal, Rank, Suit, make_standard_deck,
)
from balatro_gym.core.hand_evaluator import HandType


# ---------------------------------------------------------------------------
# Consumable type enum
# ---------------------------------------------------------------------------

class ConsumableType(enum.Enum):
    TAROT = "Tarot"
    PLANET = "Planet"
    SPECTRAL = "Spectral"


# ---------------------------------------------------------------------------
# Consumable metadata
# ---------------------------------------------------------------------------

@dataclass
class ConsumableInfo:
    id: str
    name: str
    consumable_type: ConsumableType
    cost: int
    description: str
    max_highlighted: int = 0  # 0 = no card targeting needed
    min_highlighted: int = 0


# ---------------------------------------------------------------------------
# Game state view protocol (avoids circular import)
# ---------------------------------------------------------------------------

@runtime_checkable
class ConsumableGameView(Protocol):
    """Minimal view of game state needed by consumables."""
    hand: list[Card]
    money: int
    jokers: list[Any]
    max_jokers: int
    consumables: list[BaseConsumable]
    consumable_slots: int
    rng: np.random.Generator


# ---------------------------------------------------------------------------
# Consumable registry
# ---------------------------------------------------------------------------

_CONSUMABLE_REGISTRY: dict[str, type[BaseConsumable]] = {}


def register_consumable(cls: type[BaseConsumable]) -> type[BaseConsumable]:
    """Class decorator to register a consumable in the global registry."""
    _CONSUMABLE_REGISTRY[cls.INFO.id] = cls
    return cls


def get_consumable_class(consumable_id: str) -> type[BaseConsumable]:
    """Get consumable class by ID. Raises KeyError if not found."""
    return _CONSUMABLE_REGISTRY[consumable_id]


def get_all_consumable_ids() -> list[str]:
    """Return all registered consumable IDs."""
    return list(_CONSUMABLE_REGISTRY.keys())


def create_consumable(consumable_id: str) -> BaseConsumable:
    """Create a consumable instance by ID."""
    return _CONSUMABLE_REGISTRY[consumable_id]()


def get_consumables_by_type(ctype: ConsumableType) -> list[str]:
    """Return consumable IDs filtered by type."""
    return [
        cid for cid, cls in _CONSUMABLE_REGISTRY.items()
        if cls.INFO.consumable_type == ctype
    ]


# ---------------------------------------------------------------------------
# Base consumable class
# ---------------------------------------------------------------------------

class BaseConsumable:
    """Base class for all consumables (Tarots, Planets, Spectrals).

    Subclasses must define INFO and implement can_use() / use().
    """

    INFO: ClassVar[ConsumableInfo]

    def can_use(self, view: ConsumableGameView,
                highlighted_indices: list[int]) -> bool:
        """Check if this consumable can be used with the given highlighted cards.

        Default: validates highlighted count against min/max.
        """
        n = len(highlighted_indices)
        info = self.INFO
        if info.min_highlighted > 0 and n < info.min_highlighted:
            return False
        if info.max_highlighted > 0 and n > info.max_highlighted:
            return False
        if info.min_highlighted == 0 and info.max_highlighted == 0 and n > 0:
            return False
        # Validate indices in range
        for idx in highlighted_indices:
            if idx < 0 or idx >= len(view.hand):
                return False
        return True

    def use(self, game_state: Any,
            highlighted_indices: list[int]) -> dict[str, Any]:
        """Apply this consumable's effect.

        Args:
            game_state: The mutable GameState object.
            highlighted_indices: Indices into game_state.hand of targeted cards.

        Returns:
            Side effects dict. Common keys:
            - "money": int — money gained/lost
            - "create_cards": list[Card] — cards to add to deck
            - "destroy_cards": list[Card] — cards to remove from deck
            - "create_consumable": list[str] — consumable IDs to create
            - "create_joker": str — joker ID to create
            - "message": str — description of what happened
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.INFO.name} ({self.INFO.id})"


# ===========================================================================
# PLANET CARDS (12) — Each levels up one hand type
# ===========================================================================

def _make_planet(planet_id: str, name: str, hand_type: HandType,
                 cost: int = 3) -> type[BaseConsumable]:
    """Factory for planet card classes."""

    class PlanetCard(BaseConsumable):
        INFO = ConsumableInfo(
            id=planet_id,
            name=name,
            consumable_type=ConsumableType.PLANET,
            cost=cost,
            description=f"Level up {hand_type.name.replace('_', ' ').title()}",
            max_highlighted=0,
            min_highlighted=0,
        )

        _hand_type = hand_type

        def can_use(self, view: ConsumableGameView,
                    highlighted_indices: list[int]) -> bool:
            return len(highlighted_indices) == 0

        def use(self, game_state: Any,
                highlighted_indices: list[int]) -> dict[str, Any]:
            game_state.hand_levels.level_up(self._hand_type)
            return {"message": f"Leveled up {self.INFO.name}"}

    PlanetCard.__name__ = f"Planet_{name.replace(' ', '')}"
    PlanetCard.__qualname__ = PlanetCard.__name__
    return PlanetCard


# Register all 12 planet cards (Lua: c_mercury through c_eris)
_PLANET_DEFS: list[tuple[str, str, HandType]] = [
    ("c_mercury",  "Mercury",  HandType.PAIR),
    ("c_venus",    "Venus",    HandType.THREE_OF_A_KIND),
    ("c_earth",    "Earth",    HandType.FULL_HOUSE),
    ("c_mars",     "Mars",     HandType.FOUR_OF_A_KIND),
    ("c_jupiter",  "Jupiter",  HandType.FLUSH),
    ("c_saturn",   "Saturn",   HandType.STRAIGHT),
    ("c_uranus",   "Uranus",   HandType.TWO_PAIR),
    ("c_neptune",  "Neptune",  HandType.STRAIGHT_FLUSH),
    ("c_pluto",    "Pluto",    HandType.HIGH_CARD),
    ("c_ceres",    "Ceres",    HandType.FLUSH_FIVE),
    ("c_planet_x", "Planet X", HandType.FLUSH_HOUSE),
    ("c_eris",     "Eris",     HandType.FIVE_OF_A_KIND),
]

for _pid, _pname, _ht in _PLANET_DEFS:
    _cls = _make_planet(_pid, _pname, _ht)
    _CONSUMABLE_REGISTRY[_pid] = _cls


# ===========================================================================
# TAROT CARDS (22)
# ===========================================================================

# --- Enhancement Tarots (8) ---
# Each sets an enhancement on up to N highlighted cards

def _make_enhancement_tarot(
    tarot_id: str, name: str, enhancement: Enhancement,
    max_cards: int, cost: int = 3,
) -> type[BaseConsumable]:
    """Factory for enhancement tarot classes."""

    class EnhancementTarot(BaseConsumable):
        INFO = ConsumableInfo(
            id=tarot_id,
            name=name,
            consumable_type=ConsumableType.TAROT,
            cost=cost,
            description=f"Enhance up to {max_cards} card(s) with {enhancement.name}",
            max_highlighted=max_cards,
            min_highlighted=1,
        )

        _enhancement = enhancement

        def use(self, game_state: Any,
                highlighted_indices: list[int]) -> dict[str, Any]:
            for idx in highlighted_indices:
                game_state.hand[idx].enhancement = self._enhancement
            return {"message": f"Applied {self._enhancement.name} to {len(highlighted_indices)} card(s)"}

    EnhancementTarot.__name__ = f"Tarot_{name.replace(' ', '').replace('The', '')}"
    EnhancementTarot.__qualname__ = EnhancementTarot.__name__
    return EnhancementTarot


# Lua: c_magician(Lucky/1), c_empress(Mult/2), c_hierophant(Bonus/2),
#      c_lovers(Wild/1), c_chariot(Steel/1), c_justice(Glass/1),
#      c_devil(Gold/1), c_tower(Stone/1)
_ENHANCEMENT_TAROT_DEFS: list[tuple[str, str, Enhancement, int]] = [
    ("c_magician",   "The Magician",   Enhancement.LUCKY, 1),
    ("c_empress",    "The Empress",    Enhancement.MULT,  2),
    ("c_hierophant", "The Hierophant", Enhancement.BONUS, 2),
    ("c_lovers",     "The Lovers",     Enhancement.WILD,  1),
    ("c_chariot",    "The Chariot",    Enhancement.STEEL, 1),
    ("c_justice",    "Justice",        Enhancement.GLASS, 1),
    ("c_devil",      "The Devil",      Enhancement.GOLD,  1),
    ("c_tower",      "The Tower",      Enhancement.STONE, 1),
]

for _tid, _tname, _enh, _mc in _ENHANCEMENT_TAROT_DEFS:
    _cls = _make_enhancement_tarot(_tid, _tname, _enh, _mc)
    _CONSUMABLE_REGISTRY[_tid] = _cls


# --- Suit Conversion Tarots (4) ---

def _make_suit_tarot(
    tarot_id: str, name: str, target_suit: Suit,
    max_cards: int = 3, cost: int = 3,
) -> type[BaseConsumable]:
    """Factory for suit conversion tarot classes."""

    class SuitTarot(BaseConsumable):
        INFO = ConsumableInfo(
            id=tarot_id,
            name=name,
            consumable_type=ConsumableType.TAROT,
            cost=cost,
            description=f"Convert up to {max_cards} card(s) to {target_suit.name}",
            max_highlighted=max_cards,
            min_highlighted=1,
        )

        _target_suit = target_suit

        def use(self, game_state: Any,
                highlighted_indices: list[int]) -> dict[str, Any]:
            for idx in highlighted_indices:
                game_state.hand[idx].suit = self._target_suit
            return {"message": f"Converted {len(highlighted_indices)} card(s) to {self._target_suit.name}"}

    SuitTarot.__name__ = f"Tarot_{name.replace(' ', '').replace('The', '')}"
    SuitTarot.__qualname__ = SuitTarot.__name__
    return SuitTarot


# Lua: c_star(Diamonds/3), c_moon(Clubs/3), c_sun(Hearts/3), c_world(Spades/3)
_SUIT_TAROT_DEFS: list[tuple[str, str, Suit]] = [
    ("c_star",  "The Star",  Suit.DIAMONDS),
    ("c_moon",  "The Moon",  Suit.CLUBS),
    ("c_sun",   "The Sun",   Suit.HEARTS),
    ("c_world", "The World", Suit.SPADES),
]

for _tid, _tname, _suit in _SUIT_TAROT_DEFS:
    _cls = _make_suit_tarot(_tid, _tname, _suit)
    _CONSUMABLE_REGISTRY[_tid] = _cls


# --- Strength (rank +1) ---

@register_consumable
class TarotStrength(BaseConsumable):
    """Lua: c_strength — Increase rank of up to 2 selected cards by 1.
    Ace wraps to 2 (matching Lua behavior)."""

    INFO = ConsumableInfo(
        id="c_strength", name="Strength",
        consumable_type=ConsumableType.TAROT, cost=3,
        description="Increase rank of up to 2 cards by 1 (Ace wraps to 2)",
        max_highlighted=2, min_highlighted=1,
    )

    def use(self, game_state: Any,
            highlighted_indices: list[int]) -> dict[str, Any]:
        for idx in highlighted_indices:
            card = game_state.hand[idx]
            if card.rank == Rank.ACE:
                card.rank = Rank.TWO
            else:
                card.rank = Rank(int(card.rank) + 1)
        return {"message": f"Increased rank of {len(highlighted_indices)} card(s)"}


# --- The Hanged Man (destroy cards) ---

@register_consumable
class TarotHangedMan(BaseConsumable):
    """Lua: c_hanged_man — Destroy up to 2 selected cards."""

    INFO = ConsumableInfo(
        id="c_hanged_man", name="The Hanged Man",
        consumable_type=ConsumableType.TAROT, cost=3,
        description="Destroy up to 2 selected cards",
        max_highlighted=2, min_highlighted=1,
    )

    def use(self, game_state: Any,
            highlighted_indices: list[int]) -> dict[str, Any]:
        # Remove cards from hand and deck (sorted descending to preserve indices)
        destroyed = []
        for idx in sorted(highlighted_indices, reverse=True):
            card = game_state.hand.pop(idx)
            game_state.deck.remove_card(card)
            destroyed.append(card)
        return {
            "destroy_cards": destroyed,
            "message": f"Destroyed {len(destroyed)} card(s)",
        }


# --- Death (copy card) ---

@register_consumable
class TarotDeath(BaseConsumable):
    """Lua: c_death — Select exactly 2 cards. Left card becomes a copy of right card."""

    INFO = ConsumableInfo(
        id="c_death", name="Death",
        consumable_type=ConsumableType.TAROT, cost=3,
        description="Left selected card becomes a copy of right selected card",
        max_highlighted=2, min_highlighted=2,
    )

    def use(self, game_state: Any,
            highlighted_indices: list[int]) -> dict[str, Any]:
        left_idx, right_idx = sorted(highlighted_indices)
        source = game_state.hand[right_idx]
        target = game_state.hand[left_idx]

        # Copy rank, suit, enhancement, edition, seal from source to target
        target.rank = source.rank
        target.suit = source.suit
        target.enhancement = source.enhancement
        target.edition = source.edition
        target.seal = source.seal
        return {"message": f"Copied {source} to position {left_idx}"}


# --- The Hermit ($$ card) ---

@register_consumable
class TarotHermit(BaseConsumable):
    """Lua: c_hermit — Double your money (cap at $20 gained)."""

    INFO = ConsumableInfo(
        id="c_hermit", name="The Hermit",
        consumable_type=ConsumableType.TAROT, cost=3,
        description="Double your money (max $20 gained)",
        max_highlighted=0, min_highlighted=0,
    )

    def use(self, game_state: Any,
            highlighted_indices: list[int]) -> dict[str, Any]:
        gain = min(game_state.money, 20)
        game_state.money += gain
        return {"money": gain, "message": f"Gained ${gain}"}


# --- Temperance ---

@register_consumable
class TarotTemperance(BaseConsumable):
    """Lua: c_temperance — Gain sum of sell values of all jokers (cap at $50)."""

    INFO = ConsumableInfo(
        id="c_temperance", name="Temperance",
        consumable_type=ConsumableType.TAROT, cost=3,
        description="Gain total sell value of current jokers (max $50)",
        max_highlighted=0, min_highlighted=0,
    )

    def use(self, game_state: Any,
            highlighted_indices: list[int]) -> dict[str, Any]:
        total = 0
        for joker in game_state.jokers:
            # Sell value = ceil(cost / 2) in Lua, but we use shop.sell_value
            total += max(1, joker.INFO.cost // 2)
        gain = min(total, 50)
        game_state.money += gain
        return {"money": gain, "message": f"Gained ${gain} from joker values"}


# --- The Fool (create random Tarot/Planet) ---

@register_consumable
class TarotFool(BaseConsumable):
    """Lua: c_fool — Create a random Tarot or Planet card (if consumable slot available)."""

    INFO = ConsumableInfo(
        id="c_fool", name="The Fool",
        consumable_type=ConsumableType.TAROT, cost=3,
        description="Create a random Tarot or Planet card",
        max_highlighted=0, min_highlighted=0,
    )

    def can_use(self, view: ConsumableGameView,
                highlighted_indices: list[int]) -> bool:
        return len(highlighted_indices) == 0

    def use(self, game_state: Any,
            highlighted_indices: list[int]) -> dict[str, Any]:
        # Pick a random Tarot or Planet
        pool = (get_consumables_by_type(ConsumableType.TAROT) +
                get_consumables_by_type(ConsumableType.PLANET))
        if not pool:
            return {"message": "No consumables available"}
        chosen = game_state.rng.choice(pool)
        return {"create_consumable": [chosen],
                "message": f"Created {chosen}"}


# --- High Priestess (create 2 Planets) ---

@register_consumable
class TarotHighPriestess(BaseConsumable):
    """Lua: c_high_priestess — Create up to 2 random Planet cards."""

    INFO = ConsumableInfo(
        id="c_high_priestess", name="The High Priestess",
        consumable_type=ConsumableType.TAROT, cost=3,
        description="Create up to 2 random Planet cards",
        max_highlighted=0, min_highlighted=0,
    )

    def use(self, game_state: Any,
            highlighted_indices: list[int]) -> dict[str, Any]:
        pool = get_consumables_by_type(ConsumableType.PLANET)
        if not pool:
            return {"message": "No planets available"}
        created = []
        for _ in range(2):
            if len(game_state.consumables) < game_state.consumable_slots:
                chosen = str(game_state.rng.choice(pool))
                created.append(chosen)
        return {"create_consumable": created,
                "message": f"Created {len(created)} planet(s)"}


# --- The Emperor (create 2 Tarots) ---

@register_consumable
class TarotEmperor(BaseConsumable):
    """Lua: c_emperor — Create up to 2 random Tarot cards."""

    INFO = ConsumableInfo(
        id="c_emperor", name="The Emperor",
        consumable_type=ConsumableType.TAROT, cost=3,
        description="Create up to 2 random Tarot cards",
        max_highlighted=0, min_highlighted=0,
    )

    def use(self, game_state: Any,
            highlighted_indices: list[int]) -> dict[str, Any]:
        pool = get_consumables_by_type(ConsumableType.TAROT)
        if not pool:
            return {"message": "No tarots available"}
        created = []
        for _ in range(2):
            if len(game_state.consumables) < game_state.consumable_slots:
                chosen = str(game_state.rng.choice(pool))
                created.append(chosen)
        return {"create_consumable": created,
                "message": f"Created {len(created)} tarot(s)"}


# --- Judgement (create random Joker) ---

@register_consumable
class TarotJudgement(BaseConsumable):
    """Lua: c_judgement — Create a random Joker (if joker slot available)."""

    INFO = ConsumableInfo(
        id="c_judgement", name="Judgement",
        consumable_type=ConsumableType.TAROT, cost=3,
        description="Create a random Joker",
        max_highlighted=0, min_highlighted=0,
    )

    def can_use(self, view: ConsumableGameView,
                highlighted_indices: list[int]) -> bool:
        return len(highlighted_indices) == 0 and len(view.jokers) < view.max_jokers

    def use(self, game_state: Any,
            highlighted_indices: list[int]) -> dict[str, Any]:
        from balatro_gym.core.joker import get_all_joker_ids
        pool = game_state.available_joker_ids or get_all_joker_ids()
        if not pool:
            return {"message": "No jokers available"}
        chosen = str(game_state.rng.choice(pool))
        return {"create_joker": chosen,
                "message": f"Created joker {chosen}"}


# --- Wheel of Fortune (add edition to joker) ---

@register_consumable
class TarotWheelOfFortune(BaseConsumable):
    """Lua: c_wheel_of_fortune — 1 in 4 chance to add random edition to random joker."""

    INFO = ConsumableInfo(
        id="c_wheel_of_fortune", name="Wheel of Fortune",
        consumable_type=ConsumableType.TAROT, cost=3,
        description="1 in 4 chance to add random edition to a random Joker",
        max_highlighted=0, min_highlighted=0,
    )

    def can_use(self, view: ConsumableGameView,
                highlighted_indices: list[int]) -> bool:
        return len(highlighted_indices) == 0 and len(view.jokers) > 0

    def use(self, game_state: Any,
            highlighted_indices: list[int]) -> dict[str, Any]:
        if game_state.rng.random() < 1 / 4:
            editions = [Edition.FOIL, Edition.HOLO, Edition.POLYCHROME]
            edition = editions[int(game_state.rng.integers(0, len(editions)))]
            # Wheel of Fortune applies edition to joker — but our jokers don't have
            # edition fields yet. Store as internal state for now.
            idx = int(game_state.rng.integers(0, len(game_state.jokers)))
            joker = game_state.jokers[idx]
            joker._internal_state["edition"] = edition
            return {"message": f"Added {edition.name} to {joker.INFO.name}"}
        return {"message": "Wheel of Fortune missed (3/4 chance)"}


# ===========================================================================
# SPECTRAL CARDS (10 of 18)
# ===========================================================================

# --- Familiar (destroy 1 random, create 3 enhanced face cards) ---

@register_consumable
class SpectralFamiliar(BaseConsumable):
    """Lua: c_familiar — Destroy 1 random card in hand, create 3 random enhanced face cards."""

    INFO = ConsumableInfo(
        id="c_familiar", name="Familiar",
        consumable_type=ConsumableType.SPECTRAL, cost=4,
        description="Destroy 1 random card, add 3 random enhanced face cards to deck",
        max_highlighted=0, min_highlighted=0,
    )

    def can_use(self, view: ConsumableGameView,
                highlighted_indices: list[int]) -> bool:
        return len(highlighted_indices) == 0 and len(view.hand) > 0

    def use(self, game_state: Any,
            highlighted_indices: list[int]) -> dict[str, Any]:
        rng = game_state.rng
        # Destroy 1 random card from hand
        idx = int(rng.integers(0, len(game_state.hand)))
        destroyed = game_state.hand.pop(idx)
        game_state.deck.remove_card(destroyed)

        # Create 3 random enhanced face cards
        face_ranks = [Rank.JACK, Rank.QUEEN, Rank.KING]
        suits = list(Suit)
        enhancements = list(Enhancement)
        created = []
        for _ in range(3):
            card = Card(
                rank=face_ranks[int(rng.integers(0, len(face_ranks)))],
                suit=suits[int(rng.integers(0, len(suits)))],
                enhancement=enhancements[int(rng.integers(0, len(enhancements)))],
            )
            game_state.deck.add_card(card)
            created.append(card)

        return {
            "destroy_cards": [destroyed],
            "create_cards": created,
            "message": f"Destroyed {destroyed}, created 3 enhanced face cards",
        }


# --- Grim (destroy 1 random, create 2 random Aces) ---

@register_consumable
class SpectralGrim(BaseConsumable):
    """Lua: c_grim — Destroy 1 random card in hand, create 2 random Aces."""

    INFO = ConsumableInfo(
        id="c_grim", name="Grim",
        consumable_type=ConsumableType.SPECTRAL, cost=4,
        description="Destroy 1 random card, add 2 random Aces to deck",
        max_highlighted=0, min_highlighted=0,
    )

    def can_use(self, view: ConsumableGameView,
                highlighted_indices: list[int]) -> bool:
        return len(highlighted_indices) == 0 and len(view.hand) > 0

    def use(self, game_state: Any,
            highlighted_indices: list[int]) -> dict[str, Any]:
        rng = game_state.rng
        idx = int(rng.integers(0, len(game_state.hand)))
        destroyed = game_state.hand.pop(idx)
        game_state.deck.remove_card(destroyed)

        suits = list(Suit)
        created = []
        for _ in range(2):
            card = Card(
                rank=Rank.ACE,
                suit=suits[int(rng.integers(0, len(suits)))],
            )
            game_state.deck.add_card(card)
            created.append(card)

        return {
            "destroy_cards": [destroyed],
            "create_cards": created,
            "message": f"Destroyed {destroyed}, created 2 Aces",
        }


# --- Incantation (destroy 1 random, create 4 random number cards 2-10) ---

@register_consumable
class SpectralIncantation(BaseConsumable):
    """Lua: c_incantation — Destroy 1 random card, create 4 random number cards (2-10)."""

    INFO = ConsumableInfo(
        id="c_incantation", name="Incantation",
        consumable_type=ConsumableType.SPECTRAL, cost=4,
        description="Destroy 1 random card, add 4 random number cards (2-10) to deck",
        max_highlighted=0, min_highlighted=0,
    )

    def can_use(self, view: ConsumableGameView,
                highlighted_indices: list[int]) -> bool:
        return len(highlighted_indices) == 0 and len(view.hand) > 0

    def use(self, game_state: Any,
            highlighted_indices: list[int]) -> dict[str, Any]:
        rng = game_state.rng
        idx = int(rng.integers(0, len(game_state.hand)))
        destroyed = game_state.hand.pop(idx)
        game_state.deck.remove_card(destroyed)

        number_ranks = [Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
                        Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN]
        suits = list(Suit)
        created = []
        for _ in range(4):
            card = Card(
                rank=number_ranks[int(rng.integers(0, len(number_ranks)))],
                suit=suits[int(rng.integers(0, len(suits)))],
            )
            game_state.deck.add_card(card)
            created.append(card)

        return {
            "destroy_cards": [destroyed],
            "create_cards": created,
            "message": f"Destroyed {destroyed}, created 4 number cards",
        }


# --- Seal Spectrals (4): Talisman, Deja Vu, Trance, Medium ---

def _make_seal_spectral(
    spectral_id: str, name: str, seal: Seal, cost: int = 4,
) -> type[BaseConsumable]:
    """Factory for seal-applying spectral classes."""

    class SealSpectral(BaseConsumable):
        INFO = ConsumableInfo(
            id=spectral_id,
            name=name,
            consumable_type=ConsumableType.SPECTRAL,
            cost=cost,
            description=f"Add {seal.name} Seal to 1 selected card",
            max_highlighted=1,
            min_highlighted=1,
        )

        _seal = seal

        def use(self, game_state: Any,
                highlighted_indices: list[int]) -> dict[str, Any]:
            idx = highlighted_indices[0]
            game_state.hand[idx].seal = self._seal
            return {"message": f"Applied {self._seal.name} Seal to card"}

    SealSpectral.__name__ = f"Spectral_{name.replace(' ', '')}"
    SealSpectral.__qualname__ = SealSpectral.__name__
    return SealSpectral


# Lua: c_talisman(Gold), c_deja_vu(Red), c_trance(Blue), c_medium(Purple)
_SEAL_SPECTRAL_DEFS: list[tuple[str, str, Seal]] = [
    ("c_talisman", "Talisman", Seal.GOLD),
    ("c_deja_vu",  "Deja Vu",  Seal.RED),
    ("c_trance",   "Trance",   Seal.BLUE),
    ("c_medium",   "Medium",   Seal.PURPLE),
]

for _sid, _sname, _seal in _SEAL_SPECTRAL_DEFS:
    _cls = _make_seal_spectral(_sid, _sname, _seal)
    _CONSUMABLE_REGISTRY[_sid] = _cls


# --- Aura (apply random edition to 1 highlighted card) ---

@register_consumable
class SpectralAura(BaseConsumable):
    """Lua: c_aura — Add a random edition (Foil/Holo/Polychrome) to 1 selected card."""

    INFO = ConsumableInfo(
        id="c_aura", name="Aura",
        consumable_type=ConsumableType.SPECTRAL, cost=4,
        description="Add a random edition to 1 selected card",
        max_highlighted=1, min_highlighted=1,
    )

    def use(self, game_state: Any,
            highlighted_indices: list[int]) -> dict[str, Any]:
        editions = [Edition.FOIL, Edition.HOLO, Edition.POLYCHROME]
        edition = editions[int(game_state.rng.integers(0, len(editions)))]
        idx = highlighted_indices[0]
        game_state.hand[idx].edition = edition
        return {"message": f"Applied {edition.name} edition to card"}


# --- Cryptid (create 2 copies of 1 highlighted card, add to deck) ---

@register_consumable
class SpectralCryptid(BaseConsumable):
    """Lua: c_cryptid — Create 2 copies of 1 selected card and add them to deck."""

    INFO = ConsumableInfo(
        id="c_cryptid", name="Cryptid",
        consumable_type=ConsumableType.SPECTRAL, cost=4,
        description="Create 2 copies of 1 selected card, add to deck",
        max_highlighted=1, min_highlighted=1,
    )

    def use(self, game_state: Any,
            highlighted_indices: list[int]) -> dict[str, Any]:
        source = game_state.hand[highlighted_indices[0]]
        created = []
        for _ in range(2):
            copy = source.copy()
            game_state.deck.add_card(copy)
            created.append(copy)
        return {
            "create_cards": created,
            "message": f"Created 2 copies of {source}",
        }


# --- Immolate (destroy 5 random cards, gain $20) ---

@register_consumable
class SpectralImmolate(BaseConsumable):
    """Lua: c_immolate — Destroy 5 random cards in hand, gain $20."""

    INFO = ConsumableInfo(
        id="c_immolate", name="Immolate",
        consumable_type=ConsumableType.SPECTRAL, cost=4,
        description="Destroy 5 random cards in hand, gain $20",
        max_highlighted=0, min_highlighted=0,
    )

    def can_use(self, view: ConsumableGameView,
                highlighted_indices: list[int]) -> bool:
        return len(highlighted_indices) == 0 and len(view.hand) >= 5

    def use(self, game_state: Any,
            highlighted_indices: list[int]) -> dict[str, Any]:
        rng = game_state.rng
        # Pick 5 random indices
        indices = list(rng.choice(len(game_state.hand), size=5, replace=False))
        destroyed = []
        for idx in sorted(indices, reverse=True):
            card = game_state.hand.pop(idx)
            game_state.deck.remove_card(card)
            destroyed.append(card)

        game_state.money += 20
        return {
            "destroy_cards": destroyed,
            "money": 20,
            "message": f"Destroyed 5 cards, gained $20",
        }
