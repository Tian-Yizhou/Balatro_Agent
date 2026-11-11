from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        help="re-record the golden replays in tests/golden/ instead of checking them",
    )
