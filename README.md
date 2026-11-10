# Balatro-Agent

A rules-accurate simulator of the deck-building game Balatro, built as a research
environment for AI agents: reinforcement learning, LLM agents and search.

> **v2 is under active development.** The design, decisions and roadmap are in
> [docs/REFACTOR_PLAN.md](docs/REFACTOR_PLAN.md). The previous implementation is
> preserved at tag [`v1-legacy`](../../tree/v1-legacy).

## Goals

- **Faithful rules:** the full game as of version 1.0.1o, including jokers, consumables,
  vouchers, tags, boss blinds, decks, stakes, endless mode and unlock progression.
- **Scale:** many games run in parallel across processes and nodes.
- **Inspectable:** every game can be restored and replayed step by step.

## Layout

| Path | Contents |
|---|---|
| `src/balatro_gym/sim` | Pure, deterministic rules engine |
| `src/balatro_gym/replay` | Replay recording, loading and re-simulation |
| `src/balatro_gym/env` | Agent-facing environment (Gymnasium) |
| `src/balatro_gym/serve` | Game dispatcher and remote step server |
| `tests/` | Test suite |
| `docs/` | Plan and architecture decision records |

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # create the environment
uv run pytest                # tests
uv run ruff check            # lint
uv run ruff format --check   # formatting
uv run pyright               # type check
uv run pre-commit install    # optional: lint and format on every commit
```

## Contributing

Branch, commit and pull request rules are in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE). This project is not affiliated with or endorsed by LocalThunk or
Playstack, and it contains no code, art, audio or text from Balatro.
