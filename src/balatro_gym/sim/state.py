"""The run state: plain, serializable data with no behavior of its own.

Field names follow the game's ``G.GAME`` table where one exists, so the original
code is easy to cross-reference. Rules live in ``engine``; this module only
describes what a run consists of.

Source: game.lua ``Game:init_game_object`` and ``Game:start_run``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

from .rng import Rng

__all__ = [
    "RANK_CHARS",
    "BlindSlot",
    "BlindState",
    "BlindStatus",
    "HandLevel",
    "HandSort",
    "HandType",
    "Item",
    "OpenPack",
    "Payout",
    "Phase",
    "PlayingCard",
    "RunConfig",
    "RunState",
    "Scalar",
    "Suit",
]

#: Rank characters indexed by ``rank - 2``: ranks run from 2 to 14 (Ace).
RANK_CHARS = "23456789TJQKA"

#: A JSON scalar: the type of the values an item keeps in ``Item.values``.
Scalar: TypeAlias = int | float | str | bool | None


class Phase(StrEnum):
    """The phase of a run, mirroring the game's ``G.STATES`` without animations.

    ``GAME_OVER`` is the only terminal phase; ``RunState.won`` tells whether the
    run ended in a win.
    """

    BLIND_SELECT = "blind_select"
    SELECTING_HAND = "selecting_hand"
    ROUND_EVAL = "round_eval"
    SHOP = "shop"
    PACK_OPEN = "pack_open"
    WIN = "win"
    GAME_OVER = "game_over"


class Suit(StrEnum):
    SPADES = "S"
    HEARTS = "H"
    CLUBS = "C"
    DIAMONDS = "D"


class HandType(StrEnum):
    """Poker hands, named as in the game's ``G.GAME.hands``."""

    FLUSH_FIVE = "Flush Five"
    FLUSH_HOUSE = "Flush House"
    FIVE_OF_A_KIND = "Five of a Kind"
    STRAIGHT_FLUSH = "Straight Flush"
    FOUR_OF_A_KIND = "Four of a Kind"
    FULL_HOUSE = "Full House"
    FLUSH = "Flush"
    STRAIGHT = "Straight"
    THREE_OF_A_KIND = "Three of a Kind"
    TWO_PAIR = "Two Pair"
    PAIR = "Pair"
    HIGH_CARD = "High Card"


class HandSort(StrEnum):
    """How the hand is re-sorted after each draw (``G.hand.config.sort``)."""

    RANK = "desc"
    SUIT = "suit desc"


class BlindSlot(StrEnum):
    """The three blinds of an ante, as the game names them."""

    SMALL = "Small"
    BIG = "Big"
    BOSS = "Boss"


class BlindStatus(StrEnum):
    SELECT = "Select"
    UPCOMING = "Upcoming"
    CURRENT = "Current"
    DEFEATED = "Defeated"
    SKIPPED = "Skipped"


@dataclass(slots=True)
class PlayingCard:
    """One playing card. ``uid`` is unique within a run and never reused."""

    uid: int
    rank: int
    suit: Suit
    enhancement: str | None = None
    edition: str | None = None
    seal: str | None = None
    perma_bonus: int = 0
    debuffed: bool = False
    selected: bool = False
    times_played: int = 0
    played_this_ante: bool = False
    discarded: bool = False

    @property
    def label(self) -> str:
        """Two-character name such as ``"AS"`` (Ace of Spades) or ``"TD"``."""
        return RANK_CHARS[self.rank - 2] + self.suit.value


@dataclass(slots=True)
class Item:
    """A joker or consumable. ``values`` holds its changing internal numbers."""

    uid: int
    key: str
    edition: str | None = None
    eternal: bool = False
    perishable: bool = False
    perish_tally: int = 0
    rental: bool = False
    debuffed: bool = False
    selected: bool = False
    cost: int = 0
    sell_value: int = 0
    values: dict[str, Scalar] = field(default_factory=dict[str, Scalar])


@dataclass(slots=True)
class OpenPack:
    """A booster pack being opened.

    A Standard pack holds playing cards; every other pack holds jokers or
    consumables. Source: card.lua ``Card:open`` (``G.pack_cards``,
    ``G.GAME.pack_choices`` and ``G.GAME.PACK_INTERRUPT``).

    Attributes:
        key: The pack's key, such as ``p_arcana_normal_1``.
        items: The jokers or consumables in the pack, left to right.
        playing_cards: The playing cards in a Standard pack, left to right.
        choices_left: How many more cards may be picked.
        return_phase: The phase to go back to once the pack is closed.
    """

    key: str
    items: list[Item]
    playing_cards: list[PlayingCard]
    choices_left: int
    return_phase: Phase

    @property
    def size(self) -> int:
        """How many cards are left in the pack."""
        return len(self.items) + len(self.playing_cards)


@dataclass(slots=True)
class HandLevel:
    """The level and play counts of one poker hand (``G.GAME.hands[name]``)."""

    level: int
    chips: int
    mult: int
    visible: bool
    played: int = 0
    played_this_round: int = 0


@dataclass(slots=True)
class BlindState:
    """The blind being played (``G.GAME.blind``). ``chips`` is the score target."""

    key: str
    slot: BlindSlot
    chips: float
    mult: float
    dollars: int
    disabled: bool = False


@dataclass(slots=True)
class Payout:
    """One row of the cash-out screen: where money comes from and how much."""

    source: str
    dollars: int


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Options fixed when a run starts.

    Attributes:
        deck: The deck (``Back``) key.
        stake: The stake level, 1 (White) to 8 (Gold).
        endless: Whether the run may continue after it is won.
        max_ante: End the run, as a win, once this ante is reached; ``None``
            for no limit. Useful for curricula.
    """

    deck: str = "b_red"
    stake: int = 1
    endless: bool = True
    max_ante: int | None = None


@dataclass(slots=True)
class RunState:
    """Everything needed to continue a run exactly, including its RNG state."""

    config: RunConfig
    rng: Rng
    next_uid: int
    phase: Phase
    ante: int
    round: int
    win_ante: int
    won: bool
    money: int
    chips: float
    hands: dict[HandType, HandLevel]
    last_hand_played: HandType | None
    # Per-round allowances and what is left of them (G.GAME.round_resets and
    # G.GAME.current_round).
    hands_per_round: int
    discards_per_round: int
    hand_size: int
    joker_slots: int
    consumable_slots: int
    hands_left: int
    discards_left: int
    hands_played_this_round: int
    discards_used_this_round: int
    hands_played: int
    unused_discards: int
    # Blinds.
    blind: BlindState | None
    blind_on_deck: BlindSlot | None
    blind_states: dict[BlindSlot, BlindStatus]
    blind_choices: dict[BlindSlot, str]
    bosses_used: dict[str, int]
    # Economy and difficulty modifiers.
    interest_cap: int
    interest_amount: int
    inflation: int
    discount_percent: int
    scaling: int
    ante_scaling: float
    no_blind_reward: list[BlindSlot]
    payouts: list[Payout]
    #: Scales every "1 in N" chance (``G.GAME.probabilities.normal``).
    probabilities_normal: float
    #: Flags that add or remove items from pools, such as ``gros_michel_extinct``.
    pool_flags: dict[str, bool]
    #: The player may spend down to this amount of money (Credit Card).
    bankrupt_at: int
    # Shop (G.GAME rates, G.GAME.shop and the reroll fields of current_round).
    joker_rate: float
    tarot_rate: float
    planet_rate: float
    spectral_rate: float
    playing_card_rate: float
    edition_rate: float
    shop_joker_max: int
    base_reroll_cost: int
    reroll_cost: int
    reroll_cost_increase: int
    free_rerolls: int
    shop_cards: list[Item]
    #: Booster packs for sale this round (``G.GAME.current_round.used_packs``).
    shop_boosters: list[Item]
    #: Whether the first shop's free Buffoon pack has been offered.
    first_shop_buffoon: bool
    #: The booster pack being opened, while the phase is ``PACK_OPEN``.
    pack: OpenPack | None
    #: How much the next Ectoplasm reduces the hand size (``G.GAME.ecto_minus``).
    ecto_minus: int
    # Consumable history.
    last_tarot_planet: str | None
    consumable_usage: dict[str, int]
    #: Extra modifiers set by stakes, such as ``"eternals_in_shop"``.
    modifiers: dict[str, bool]
    perishable_rounds: int
    rental_rate: int
    # Card areas. The deck is the draw pile: cards are drawn from its end, and
    # its order is hidden from the player.
    hand_sort: HandSort
    deck: list[PlayingCard]
    hand: list[PlayingCard]
    discard: list[PlayingCard]
    jokers: list[Item]
    consumables: list[Item]

    def new_uid(self) -> int:
        uid = self.next_uid
        self.next_uid += 1
        return uid

    @property
    def is_over(self) -> bool:
        return self.phase is Phase.GAME_OVER

    def playing_cards(self) -> list[PlayingCard]:
        """Every playing card the player owns, in no particular order."""
        return [*self.deck, *self.hand, *self.discard]

    def selected_cards(self) -> list[PlayingCard]:
        """The selected cards in hand, from left to right."""
        return [card for card in self.hand if card.selected]
