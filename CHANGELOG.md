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
- Replays in `balatro_gym.replay`: a recorder with round keyframes and per-step
  annotations, `.json` and `.json.gz` files, restoring any step, branching, a drift
  check, a text viewer (`python -m balatro_gym.replay`), and golden replays in
  `tests/golden/`.

- Item data for all 296 jokers, consumables, vouchers, packs, editions,
  enhancements and decks, extracted from the game by `tools/extract_content/`.
- The full scoring pipeline: retriggers, cards held in hand, joker editions,
  effects between jokers, destroyed cards, and the joker contexts before and after
  scoring, on discards and at the end of the round.
- Enhancements, editions and seals on playing cards.
- 52 jokers, including Blueprint and Brainstorm. Jokers without an implemented
  effect are unavailable (ADR 0003).
- `EffectApplied`, `CardRetriggered`, `CardDestroyed`, `JokerDestroyed` and
  `HandLevelChanged` events, which the replay viewer shows as a scoring breakdown.

### Changed

- Restructure the repository for v2: a `src/` layout with the `sim`, `replay`, `env`
  and `serve` subpackages, managed with uv.

### Removed

- The v1 implementation. It remains available at tag `v1-legacy`.
