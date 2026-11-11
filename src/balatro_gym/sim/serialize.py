"""Snapshots: conversion between engine objects and JSON-compatible data.

``to_data`` and ``from_data`` work on any dataclass in ``sim`` by reading its
type hints, so a new field is serialized without extra code. The output uses only
dicts, lists, strings, numbers, booleans and ``None``, and can be written with
``json.dumps``. Floats keep their exact value, including infinities, which
``json`` writes as ``Infinity``.
"""

from __future__ import annotations

import dataclasses
import enum
import types
import typing
from collections.abc import Mapping
from functools import cache
from typing import Any, TypeVar, cast

from .rng import Rng
from .state import RunState
from .version import RULES_VERSION, STATE_SCHEMA_VERSION

__all__ = [
    "SnapshotError",
    "clone",
    "from_data",
    "restore",
    "snapshot",
    "to_data",
]

Value = TypeVar("Value")
_SCALARS: tuple[type, ...] = (bool, int, float, str)


class SnapshotError(ValueError):
    """A snapshot is malformed or was written by an incompatible schema."""


def to_data(value: object) -> Any:
    """Convert an engine object into JSON-compatible data."""
    if value is None or (isinstance(value, _SCALARS) and not isinstance(value, enum.Enum)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Rng):
        return value.to_dict()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {name: to_data(getattr(value, name)) for name in _field_names(type(value))}
    if isinstance(value, list | tuple):
        return [to_data(item) for item in cast("list[object]", value)]
    if isinstance(value, dict):
        items = cast("dict[object, object]", value).items()
        return {_key_to_data(key): to_data(entry) for key, entry in items}
    raise TypeError(f"cannot serialize {type(value).__name__}")


def from_data(target_type: type[Value], data: Any) -> Value:
    """Rebuild an object of type ``target_type`` from the output of :func:`to_data`."""
    return cast("Value", _from_data(target_type, data))


def snapshot(state: RunState) -> dict[str, Any]:
    """Return a self-describing snapshot of ``state``."""
    return {
        "schema": STATE_SCHEMA_VERSION,
        "rules": RULES_VERSION,
        "state": to_data(state),
    }


def restore(data: Mapping[str, Any]) -> RunState:
    """Rebuild the state saved by :func:`snapshot`.

    Raises:
        SnapshotError: if the snapshot uses a different state schema version.
    """
    schema = data.get("schema")
    if schema != STATE_SCHEMA_VERSION:
        raise SnapshotError(
            f"snapshot schema {schema!r} does not match engine schema {STATE_SCHEMA_VERSION}"
        )
    try:
        return from_data(RunState, data["state"])
    except (KeyError, TypeError, ValueError) as error:
        raise SnapshotError(f"malformed snapshot: {error}") from error


def clone(state: RunState) -> RunState:
    """Return an exact, independent copy of ``state``, RNG included.

    The copy knows the run's real future. Planning agents that must not see
    hidden information use ``engine.determinize`` instead.
    """
    return from_data(RunState, to_data(state))


@cache
def _field_names(cls: type) -> tuple[str, ...]:
    return tuple(field.name for field in dataclasses.fields(cls))


@cache
def _hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls)


def _key_to_data(key: object) -> str:
    if isinstance(key, enum.Enum):
        return str(key.value)
    if isinstance(key, str):
        return key
    raise TypeError(f"dict keys must be strings or enums, not {type(key).__name__}")


def _from_data(target_type: Any, data: Any) -> Any:
    origin = typing.get_origin(target_type)
    if origin is typing.Union or origin is types.UnionType:
        return _from_union(typing.get_args(target_type), data)
    if origin is list:
        (item_type,) = typing.get_args(target_type)
        return [_from_data(item_type, item) for item in data]
    if origin is tuple:
        item_type = typing.get_args(target_type)[0]
        return tuple(_from_data(item_type, item) for item in data)
    if origin is dict:
        key_type, value_type = typing.get_args(target_type)
        return {
            _from_data(key_type, key): _from_data(value_type, value)
            for key, value in cast("dict[str, Any]", data).items()
        }
    if target_type is Any:
        return data
    if target_type is float:
        if not isinstance(data, int | float) or isinstance(data, bool):
            raise TypeError(f"expected a number, got {data!r}")
        return float(data)
    if target_type in (bool, int, str):
        if type(data) is not target_type:
            raise TypeError(f"expected {target_type.__name__}, got {data!r}")
        return data
    if isinstance(target_type, type):
        cls = cast("type[Any]", target_type)
        if issubclass(cls, enum.Enum):
            return cls(data)
        if cls is Rng:
            return Rng.from_dict(data)
        if dataclasses.is_dataclass(cls):
            hints = _hints(cls)
            fields = cast("dict[str, Any]", data)
            return cls(
                **{name: _from_data(hints[name], fields[name]) for name in _field_names(cls)}
            )
    raise TypeError(f"cannot deserialize type {target_type!r}")


def _from_union(member_types: tuple[Any, ...], data: Any) -> Any:
    if data is None:
        if type(None) in member_types:
            return None
        raise TypeError("unexpected null")
    members = [member for member in member_types if member is not type(None)]
    if len(members) == 1:
        return _from_data(members[0], data)
    if all(member in _SCALARS for member in members) and isinstance(data, _SCALARS):
        return data
    raise TypeError(f"cannot deserialize union {member_types!r} from {data!r}")
