# Guide for AI assistants

Read these before changing anything:

- [docs/REFACTOR_PLAN.md](docs/REFACTOR_PLAN.md): design, decisions (§2) and milestones (§11).
- [CONTRIBUTING.md](CONTRIBUTING.md): branch, commit and PR rules. Follow them exactly.

## Hard rules

- Never list an AI tool as a co-author. Commits and PRs carry no AI attribution.
- Significant changes go through a branch and one PR; small, low-risk changes are
  committed directly to `main` (CONTRIBUTING.md §1). Commit finished units of work,
  not every edit. Never merge PRs and never force-push `main`.
- Never commit any part of the original game: code, assets or text. The
  maintainer's local reference copy of the source is found through `BALATRO_LUA_DIR`.
- Raise design decisions with the maintainer before building on them. Record
  agreed decisions in `docs/adr/`.

## Commands

```bash
uv sync
uv run pytest
uv run ruff check
uv run ruff format --check
uv run pyright
```

## Architecture rules

- Layers: `sim` ← `replay` ← `env` ← `serve`. A layer imports only itself and the
  layers to its left. `tests/test_architecture.py` enforces this.
- `sim` uses only the standard library. It has no global mutable state, no wall
  clock, no unseeded randomness, and never uses the built-in `hash()` for values
  that are saved. All state is plain, serializable data.
- Ported rules cite their source location in the docstring, for example
  `Source: card.lua Card:calculate_joker ("Blueprint")`. Never copy code or text
  word for word.
- A change to game behavior bumps `RULES_VERSION` and updates the golden replays.
