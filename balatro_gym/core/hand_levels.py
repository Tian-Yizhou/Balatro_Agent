"""Hand leveling system: mutable per-hand-type chip/mult scores.

Reference: Balatro Lua source — game.lua (G.GAME.hands table, lines 2002-2013),
common_events.lua (level_up_hand, lines 464-493).

Each hand type has a starting chips/mult and a per-level increment.
Planet cards level up specific hand types, increasing their base scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

from balatro_gym.core.hand_evaluator import HandType


@dataclass
class HandLevelData:
    """Level data for a single hand type."""
    level: int
    s_chips: int    # starting chips (at level 1)
    s_mult: int     # starting mult (at level 1)
    l_chips: int    # chips added per level above 1
    l_mult: int     # mult added per level above 1

    @property
    def chips(self) -> int:
        """Current chip value at this level."""
        return max(self.s_chips + self.l_chips * (self.level - 1), 0)

    @property
    def mult(self) -> int:
        """Current mult value at this level."""
        return max(self.s_mult + self.l_mult * (self.level - 1), 1)


# Default hand level data from Lua game.lua lines 2002-2013
# Format: HandType -> (s_chips, s_mult, l_chips, l_mult)
_DEFAULTS: dict[HandType, tuple[int, int, int, int]] = {
    HandType.HIGH_CARD:       (5, 1, 10, 1),
    HandType.PAIR:            (10, 2, 15, 1),
    HandType.TWO_PAIR:        (20, 2, 20, 1),
    HandType.THREE_OF_A_KIND: (30, 3, 20, 2),
    HandType.STRAIGHT:        (30, 4, 30, 3),
    HandType.FLUSH:           (35, 4, 15, 2),
    HandType.FULL_HOUSE:      (40, 4, 25, 2),
    HandType.FOUR_OF_A_KIND:  (60, 7, 30, 3),
    HandType.STRAIGHT_FLUSH:  (100, 8, 40, 4),
    HandType.FIVE_OF_A_KIND:  (120, 12, 35, 3),
    HandType.FLUSH_HOUSE:     (140, 14, 40, 4),
    HandType.FLUSH_FIVE:      (160, 16, 50, 3),
}


class HandLevelManager:
    """Manages mutable hand type levels.

    Tracks the current level of each poker hand type. Planet cards
    call `level_up()` to increase a hand's base scoring.
    """

    def __init__(self) -> None:
        self._levels: dict[HandType, HandLevelData] = {}
        self.reset()

    def reset(self) -> None:
        """Reset all hand types to level 1."""
        self._levels = {
            ht: HandLevelData(level=1, s_chips=sc, s_mult=sm, l_chips=lc, l_mult=lm)
            for ht, (sc, sm, lc, lm) in _DEFAULTS.items()
        }

    def get_level(self, hand_type: HandType) -> HandLevelData:
        """Get the level data for a hand type."""
        return self._levels[hand_type]

    def get_score(self, hand_type: HandType) -> tuple[int, int]:
        """Get the current (chips, mult) for a hand type at its current level."""
        data = self._levels[hand_type]
        return (data.chips, data.mult)

    def level_up(self, hand_type: HandType, amount: int = 1) -> None:
        """Level up a hand type.

        Matches Lua: level_up_hand(card, hand, instant, amount)
        """
        data = self._levels[hand_type]
        data.level = max(0, data.level + amount)

    def get_all_levels(self) -> dict[HandType, HandLevelData]:
        """Return a copy of all hand levels (for observation)."""
        return dict(self._levels)
