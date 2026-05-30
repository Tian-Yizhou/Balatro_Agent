"""Pluggable difficulty configuration system.

Each difficulty is defined in its own module under :mod:`balatro_gym.difficulty`
and exposes a single ``build_config(seed=None) -> GameConfig`` function.

The loader discovers modules by filename — no registration step required.

Adding a new difficulty
-----------------------
1. Create ``balatro_gym/difficulty/<name>.py``.
2. Define ``build_config(seed=None) -> GameConfig`` in it.
3. Use it::

       env = balatro_gym.make("<name>")

See ``customized.py`` for a worked example you can copy.

Programmatic registration
-------------------------
For runtime experimentation (notebooks, tests) you can also register a
builder without creating a file::

    from balatro_gym.difficulty import register_difficulty
    register_difficulty("aggressive", lambda seed=None: GameConfig(...))
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from balatro_gym.envs.configs import GameConfig

BuildConfigFn = Callable[..., "GameConfig"]

_DIFFICULTY_DIR = Path(__file__).parent
_EXCLUDED_STEMS = {"__init__", "base"}

# Runtime overrides — populated by register_difficulty(). Takes priority over
# file-backed difficulties of the same name.
_OVERRIDES: dict[str, BuildConfigFn] = {}


def list_difficulties() -> list[str]:
    """Return all available difficulty names (sorted)."""
    names = set(_OVERRIDES)
    for path in _DIFFICULTY_DIR.glob("*.py"):
        stem = path.stem
        if stem.startswith("_") or stem in _EXCLUDED_STEMS:
            continue
        names.add(stem)
    return sorted(names)


def get_difficulty(name: str, seed: int | None = None) -> "GameConfig":
    """Build the :class:`GameConfig` for a difficulty by name.

    Looks up runtime overrides first, then loads
    ``balatro_gym.difficulty.<name>`` and calls its ``build_config(seed)``.
    """
    if name in _OVERRIDES:
        return _OVERRIDES[name](seed=seed)

    module_name = f"balatro_gym.difficulty.{name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Only swallow "this module doesn't exist". Re-raise import errors
        # caused by something the difficulty module itself imports.
        if exc.name == module_name:
            raise ValueError(
                f"Unknown difficulty {name!r}. "
                f"Available: {list_difficulties()}"
            ) from None
        raise

    builder = getattr(module, "build_config", None)
    if builder is None:
        raise ValueError(
            f"Difficulty module {module_name!r} must define a "
            f"'build_config(seed=None) -> GameConfig' function."
        )
    return builder(seed=seed)


def register_difficulty(name: str, builder: BuildConfigFn) -> None:
    """Register a difficulty builder programmatically (no file required).

    To make a difficulty permanent and visible to other users, create a
    file in :mod:`balatro_gym.difficulty` instead.
    """
    _OVERRIDES[name] = builder


def unregister_difficulty(name: str) -> None:
    """Remove a programmatically-registered difficulty."""
    _OVERRIDES.pop(name, None)


__all__ = [
    "list_difficulties",
    "get_difficulty",
    "register_difficulty",
    "unregister_difficulty",
]
