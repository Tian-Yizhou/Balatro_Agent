"""Episode seed ID generation and parsing.

Produces human-readable, roguelite-style seed strings that uniquely
identify episodes.  The format encodes a creation timestamp and the
numeric game seed into a compact alphanumeric string.

Format
------
``YYYYMMDD-HHMM-XXXXXXXX``

- **YYYYMMDD-HHMM** — minute-precision UTC timestamp of episode creation.
- **XXXXXXXX** — 8-character base-36 encoding of the numeric game seed
  (digits 0-9 and uppercase letters A-Z).

Capacity: 36^8 = 2,821,109,907,455 (~2.8 trillion unique seeds), far
exceeding the "billions of trajectories" requirement.

Examples::

    >>> from balatro_gym.core.seed_id import generate_seed_id, parse_seed_id
    >>> sid = generate_seed_id(game_seed=42)
    >>> sid
    '20260418-1430-00000016'
    >>> parse_seed_id(sid)
    {'timestamp': '20260418-1430', 'game_seed': 42, 'seed_str': '00000016'}
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

# Base-36 alphabet (0-9, A-Z)
_B36_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_B36_DECODE = {c: i for i, c in enumerate(_B36_CHARS)}
_SEED_DIGITS = 8  # 36^8 ≈ 2.8 T
_MAX_SEED = 36**_SEED_DIGITS - 1


def _encode_base36(n: int, width: int = _SEED_DIGITS) -> str:
    """Encode a non-negative integer as zero-padded base-36 string."""
    if n < 0:
        raise ValueError(f"Seed must be non-negative, got {n}")
    if n > _MAX_SEED:
        raise ValueError(
            f"Seed {n} exceeds maximum encodable value ({_MAX_SEED})"
        )
    chars: list[str] = []
    if n == 0:
        chars.append("0")
    else:
        val = n
        while val > 0:
            chars.append(_B36_CHARS[val % 36])
            val //= 36
    return "".join(reversed(chars)).zfill(width)


def _decode_base36(s: str) -> int:
    """Decode a base-36 string to an integer."""
    result = 0
    for c in s.upper():
        if c not in _B36_DECODE:
            raise ValueError(f"Invalid base-36 character: {c!r}")
        result = result * 36 + _B36_DECODE[c]
    return result


def generate_seed_id(
    game_seed: int,
    timestamp: datetime | None = None,
) -> str:
    """Generate an episode seed ID string.

    Args:
        game_seed: The numeric RNG seed that determines the game.
        timestamp: Episode creation time (defaults to current UTC time).

    Returns:
        Seed string in ``YYYYMMDD-HHMM-XXXXXXXX`` format.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    ts_str = timestamp.strftime("%Y%m%d-%H%M")
    seed_str = _encode_base36(game_seed)
    return f"{ts_str}-{seed_str}"


def parse_seed_id(seed_id: str) -> dict:
    """Parse a seed ID string back into its components.

    Args:
        seed_id: A string in ``YYYYMMDD-HHMM-XXXXXXXX`` format.

    Returns:
        Dict with keys ``timestamp`` (str), ``game_seed`` (int),
        and ``seed_str`` (the raw base-36 portion).

    Raises:
        ValueError: If the format is invalid.
    """
    parts = seed_id.split("-")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid seed ID format (expected YYYYMMDD-HHMM-XXXXXXXX): {seed_id!r}"
        )
    date_part, time_part, seed_part = parts
    if len(date_part) != 8 or len(time_part) != 4 or len(seed_part) != _SEED_DIGITS:
        raise ValueError(
            f"Invalid seed ID component lengths: {seed_id!r}"
        )
    game_seed = _decode_base36(seed_part)
    return {
        "timestamp": f"{date_part}-{time_part}",
        "game_seed": game_seed,
        "seed_str": seed_part,
    }


def seed_id_to_game_seed(seed_id: str) -> int:
    """Extract just the numeric game seed from a seed ID string."""
    return parse_seed_id(seed_id)["game_seed"]
