# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Engine foundations in `balatro_gym.sim`: the run state, a keyed counter-based RNG,
  snapshots with `clone` and `determinize`, atomic actions with macro helpers, and
  typed events.
- The round loop with plain cards: blind selection, selecting, sorting, moving,
  playing and discarding cards, scoring, cash out, boss selection, antes, and the
  win at ante 8 with the choice to continue in endless mode.
- Poker hand evaluation and blind score targets ported from the game and checked
  against its Lua code, with a harness in `tools/lua_harness/`.

### Changed

- Restructure the repository for v2: a `src/` layout with the `sim`, `replay`, `env`
  and `serve` subpackages, managed with uv.

### Removed

- The v1 implementation. It remains available at tag `v1-legacy`.
