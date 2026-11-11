"""Enforce the package dependency rules (docs/REFACTOR_PLAN.md §3)."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
ROOT = "balatro_gym"

# Each layer may import only itself and the layers listed here.
ALLOWED: dict[str, frozenset[str]] = {
    "sim": frozenset({"sim"}),
    "replay": frozenset({"sim", "replay"}),
    "env": frozenset({"sim", "replay", "env"}),
    "serve": frozenset({"sim", "replay", "env", "serve"}),
}


def _module_name(path: Path) -> str:
    parts = path.relative_to(SRC).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imported_names(path: Path) -> list[str]:
    """Return the absolute dotted name of everything imported by ``path``."""
    package = _module_name(path).split(".")
    if path.name != "__init__.py":
        package = package[:-1]

    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                anchor = package[: len(package) - node.level + 1]
                base = ".".join([*anchor, node.module] if node.module else anchor)
            names.extend(
                base if alias.name == "*" else f"{base}.{alias.name}" for alias in node.names
            )
    return names


def _layer_files(layer: str) -> list[Path]:
    return sorted((SRC / ROOT / layer).rglob("*.py"))


def test_every_subpackage_has_rules() -> None:
    subpackages = {path.parent.name for path in (SRC / ROOT).glob("*/__init__.py")}
    assert subpackages == set(ALLOWED)


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_layer_imports_only_allowed_layers(layer: str) -> None:
    violations = []
    for path in _layer_files(layer):
        for name in _imported_names(path):
            parts = name.split(".")
            if parts[0] == ROOT and (len(parts) < 2 or parts[1] not in ALLOWED[layer]):
                violations.append(f"{path.relative_to(SRC)} imports {name}")
    assert not violations, "\n".join(violations)


def test_sim_imports_only_the_standard_library() -> None:
    violations = []
    for path in _layer_files("sim"):
        for name in _imported_names(path):
            top = name.split(".")[0]
            if top != ROOT and top not in sys.stdlib_module_names:
                violations.append(f"{path.relative_to(SRC)} imports {name}")
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_layer_is_importable(layer: str) -> None:
    importlib.import_module(f"{ROOT}.{layer}")


def test_version_is_set() -> None:
    import balatro_gym

    assert isinstance(balatro_gym.__version__, str)
    assert balatro_gym.__version__
