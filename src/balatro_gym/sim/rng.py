"""Keyed, counter-based random numbers.

Every random decision names a *stream key*. Keys are the seed strings the game
passes to ``pseudoseed``, for example ``"nr1"`` for the deck shuffle at the start
of a round in ante 1, or ``"boss"`` for choosing a boss blind, so each draw can be
traced back to the code that makes it in the original game.

Draw number ``n`` on stream ``key`` is a pure function of ``(seed, key, n)``. The
generator's whole state is the seed plus one counter per stream, so it is cheap to
store in every snapshot. Streams are independent: drawing more cards never changes
which boss or shop a later draw on another stream produces.

Source for the key scheme: functions/misc_functions.lua ``pseudoseed``. The values
drawn differ from the game's (see ADR D1); only the probabilities match.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, MutableSequence, Sequence
from typing import Any, TypeVar

__all__ = ["Rng", "stable_hash"]

_T = TypeVar("_T")

_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_TWO_POW_MINUS_53 = 2.0**-53


def _splitmix64(x: int) -> int:
    """Advance a SplitMix64 state by one step and return its output."""
    z = (x + _GOLDEN) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def stable_hash(text: str) -> int:
    """Return a 64-bit hash of ``text`` that is identical on every machine.

    Python's built-in ``hash()`` is salted per process, so it must never be used
    for anything that is saved or replayed.
    """
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


class Rng:
    """A seed plus one draw counter per stream key.

    Each method that draws advances the counter of its stream by exactly one, so
    a shuffle of any length costs one tick on its stream.
    """

    __slots__ = ("_bases", "_seed_word", "counters", "seed")

    seed: str
    counters: dict[str, int]
    _seed_word: int
    _bases: dict[str, int]

    def __init__(self, seed: str, counters: dict[str, int] | None = None) -> None:
        self.seed = seed
        self.counters = dict(counters) if counters else {}
        self._seed_word = stable_hash(seed)
        self._bases = {}

    def _next(self, key: str) -> int:
        base = self._bases.get(key)
        if base is None:
            base = _splitmix64(self._seed_word ^ stable_hash(key))
            self._bases[key] = base
        n = self.counters.get(key, 0)
        self.counters[key] = n + 1
        return _splitmix64((base + n * _GOLDEN) & _MASK64)

    def random(self, key: str) -> float:
        """Return a float uniformly distributed in ``[0, 1)``."""
        return (self._next(key) >> 11) * _TWO_POW_MINUS_53

    def randint(self, key: str, low: int, high: int) -> int:
        """Return an integer uniformly distributed in ``[low, high]``."""
        span = high - low + 1
        if span <= 0:
            raise ValueError(f"empty range [{low}, {high}]")
        return low + ((self._next(key) * span) >> 64)

    def choice(self, key: str, items: Sequence[_T]) -> _T:
        """Return one element of ``items``, chosen uniformly."""
        if not items:
            raise ValueError("cannot choose from an empty sequence")
        return items[self.randint(key, 0, len(items) - 1)]

    def shuffle(self, key: str, items: MutableSequence[_T]) -> None:
        """Shuffle ``items`` in place with a Fisher-Yates shuffle."""
        state = self._next(key)
        for i in range(len(items) - 1, 0, -1):
            state = _splitmix64(state)
            j = (state * (i + 1)) >> 64
            items[i], items[j] = items[j], items[i]

    def to_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "counters": dict(sorted(self.counters.items()))}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Rng:
        counters: Mapping[str, Any] = data["counters"]
        return cls(str(data["seed"]), {str(k): int(v) for k, v in counters.items()})

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rng):
            return NotImplemented
        return self.seed == other.seed and self.counters == other.counters

    def __repr__(self) -> str:
        return f"Rng(seed={self.seed!r}, streams={len(self.counters)})"
