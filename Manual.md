# Project Manual

Quick-reference guide to the Balatro-Agent codebase. Designed for AI assistants and new contributors to understand the project structure and locate components.

## What This Project Is

A Gymnasium-compatible card game environment inspired by Balatro (a roguelite deck-building poker game), plus an RL training pipeline using Ray RLlib. Built as a final project for COMP_SCI 496 (Agent AI) at Northwestern.

Two independent packages in one repo:
- **`balatro_gym`** — Environment package (gymnasium + numpy + pyyaml only). Installable via `pip install -e .`
- **`agent`** — Training/baselines package (depends on balatro_gym + Ray + Torch)
- **Tests**: 382 tests across 13 files, run with `pytest tests/`

## Directory Layout

```
Balatro-Agent/
├── balatro_gym/              # ENVIRONMENT PACKAGE (no Ray/Torch dependency)
│   ├── __init__.py           # make(), make_vec(), version
│   ├── core/                 # Game engine (no Gymnasium dependency)
│   ├── envs/                 # Gymnasium environment wrapper + configs
│   ├── wrappers/             # Recording wrappers (trajectories, statistics)
│   ├── rendering/            # (placeholder — not yet implemented)
│   └── utils/                # (placeholder — not yet implemented)
├── agent/                    # AGENT PACKAGE (depends on balatro_gym + Ray + Torch)
│   ├── __init__.py
│   ├── baselines/            # Baseline agents (random, heuristic)
│   └── rllib/                # Ray RLlib integration (training, evaluation)
├── configs/                  # YAML configuration files
│   ├── defaults.yaml         # Full default config (medium preset, all fields)
│   └── example_custom.yaml   # Example: override a few fields from easy preset
├── tests/                    # Unit and integration tests
├── docs/                     # Documentation
│   ├── Balatro_Gym_Guidance.md   # Full usage guide (obs space, actions, training, recording)
│   ├── Project_Proposal.md       # Course project proposal (historical)
│   ├── PLAN.md                   # Original implementation plan (historical)
│   └── Tech_Log.md               # Technical decisions log
├── experiments/              # (placeholder for training experiment scripts)
├── CLAUDE.md                 # Project context for Claude Code
├── Manual.md                 # This file
├── README.md                 # Public project README
├── setup.py                  # Package installation (extras: recording, agent, all)
└── requirements.txt          # Direct dependencies
```

### Package Dependency

```
balatro_gym (environment)  ←── agent (training/baselines)
    gymnasium, numpy, pyyaml   ray[rllib], torch
```

One-way dependency: `agent` imports from `balatro_gym`. The environment package has zero dependency on agent code.

---

## Core Engine (`balatro_gym/core/`)

The game engine is framework-independent — no Gymnasium, no Ray, no PyTorch. All game logic lives here.

### Module Dependency Graph

```
card.py  (leaf — no internal imports)
  ↑
hand_evaluator.py  (imports: card)
  ↑
hand_levels.py  (imports: hand_evaluator)

joker.py  (imports: card, hand_evaluator)
consumable.py  (imports: card, hand_evaluator)
blind.py  (imports: card; TYPE_CHECKING: game_state)
  ↑
shop.py  (imports: joker, consumable)
  ↑
game_state.py  (imports: card, hand_evaluator, hand_levels, joker, consumable, blind, shop)

seed_id.py  (leaf — no internal imports)
```

### `card.py` (458 lines)
Core card types and deck management.

| Export | Type | Description |
|--------|------|-------------|
| `Suit` | IntEnum | HEARTS=0, DIAMONDS=1, CLUBS=2, SPADES=3 |
| `Rank` | IntEnum | TWO=2 through ACE=14 |
| `Enhancement` | Enum | 8 types: BONUS, MULT, WILD, GLASS, STEEL, STONE, GOLD, LUCKY |
| `Edition` | Enum | 3 types: FOIL, HOLO, POLYCHROME |
| `Seal` | Enum | 4 types: GOLD, RED, BLUE, PURPLE |
| `Card` | dataclass | `(rank, suit, uid, enhancement, edition, seal, face_down)` |
| `Deck` | class | Draw pile + discard pile, seeded RNG, auto-reshuffle |

Key methods:
- `Card.chip_value` — chip bonus including enhancement effects
- `Card.to_dict()` / `Card.from_dict()` — serialization
- `Card.is_wild` — True if Enhancement.WILD
- `Deck.draw(n)`, `Deck.return_cards(cards)`, `Deck.add_card(card)`, `Deck.remove_card(card)`

### `hand_evaluator.py` (223 lines)
Poker hand detection. Supports 12 hand types including Balatro-specific ones (Five of a Kind, Flush House, Flush Five).

| Export | Type | Description |
|--------|------|-------------|
| `HandType` | IntEnum | 12 types: HIGH_CARD(0) through FLUSH_FIVE(11) |
| `HandResult` | dataclass | `(hand_type, scoring_cards, held_cards, base_chips, base_mult)` |
| `evaluate_hand(played, held)` | function | Returns `HandResult` for given cards |
| `HAND_BASE_SCORES` | dict | HandType -> (base_chips, base_mult) |

### `hand_levels.py` (92 lines)
Mutable hand-type scoring levels. Planet cards level up hand types.

| Export | Type | Description |
|--------|------|-------------|
| `HandLevelData` | dataclass | `(level, s_chips, s_mult, l_chips, l_mult)` with `chips`/`mult` properties |
| `HandLevelManager` | class | `get_score(hand_type)` -> (chips, mult), `level_up(hand_type)` |

### `joker.py` (740 lines)
30 joker implementations using a registry pattern.

| Export | Type | Description |
|--------|------|-------------|
| `ScoreModification` | dataclass | `(add_chips, add_mult, x_mult)` |
| `JokerInfo` | dataclass | `(id, name, rarity, cost, description)` |
| `BaseJoker` | ABC | Base class with hook methods |
| `register_joker` | decorator | Registers a joker class in `_JOKER_REGISTRY` |
| `create_joker(id)` | function | Factory that creates joker instance by ID |
| `get_all_joker_ids()` | function | Returns list of all registered joker IDs |

Hook methods (called by `game_state._apply_scoring()`):
- `on_before(view)` — before per-card scoring
- `on_individual(card, view)` — per scoring card
- `on_held_individual(card, view)` — per held (non-played) card
- `on_main(view)` — main left-to-right joker pass
- `on_after(view)` — after all scoring
- `on_discard(cards, view)` — when cards are discarded
- `on_end_of_round(view)` — when a blind is beaten
- `on_round_start(view)` — when a new blind starts

Circular import avoidance: `GameStateView` is a `@runtime_checkable Protocol`.

### `consumable.py` (839 lines)
40 consumable implementations: 12 Planets, 22 Tarots, 6 Spectrals.

| Export | Type | Description |
|--------|------|-------------|
| `ConsumableType` | Enum | TAROT, PLANET, SPECTRAL |
| `ConsumableInfo` | dataclass | `(id, name, consumable_type, cost, description, max_highlighted, min_highlighted)` |
| `BaseConsumable` | ABC | `can_use(view, indices)`, `use(game_state, indices)` |
| `register_consumable` | decorator | Registry pattern (same as jokers) |
| `create_consumable(id)` | function | Factory |
| `get_consumables_by_type(type)` | function | Filter by ConsumableType |

### `blind.py` (315 lines)
Blind progression and boss blind effects.

| Export | Type | Description |
|--------|------|-------------|
| `BlindType` | Enum | SMALL, BIG, BOSS |
| `BlindManager` | class | Manages blind progression across antes |
| `BOSS_BLINDS` | list | 9 boss blind definitions |

Boss effects: DebuffSuit (4 variants), DebuffFaceCards, TheNeedle (1 hand only), TheWall (4x mult), TheFlint (halve base), TheHook (discard 2).

### `shop.py` (222 lines)
Shop offering generation, buying, selling, rerolling.

| Export | Type | Description |
|--------|------|-------------|
| `ShopOffering` | dataclass | `(item, cost, sold, item_type)` where item_type is "joker" or "consumable" |
| `Shop` | class | `generate_offerings()`, `buy_item()`, `sell_value()`, `reroll()` |

### `game_state.py` (1001 lines)
Central game state manager. Owns the full game lifecycle and 10-step scoring pipeline.

| Export | Type | Description |
|--------|------|-------------|
| `GamePhase` | Enum | PLAY, SHOP, GAME_OVER, GAME_WON |
| `GameState` | class | Full mutable game state |
| `GameStateSnapshot` | dataclass | Read-only snapshot (satisfies `GameStateView` Protocol) |

Key methods:
- `reset()` — initialize a new game
- `play_hand(indices)` — play selected cards, score them
- `discard(indices)` — discard selected cards, draw replacements
- `shop_buy(slot)`, `shop_sell(joker_idx)`, `shop_reroll()`, `shop_skip()`
- `serialize()` / `deserialize(data)` — full state save/load (including RNG state)
- `_apply_scoring()` — 10-step pipeline matching Lua's scoring order

Scoring pipeline order:
1. Hand-type base chips/mult from `hand_levels`
2. Blind modify_hand (The Flint)
3. Joker `on_before`
4. Per scoring card: chip_value + mult + x_mult + edition + joker on_individual + Red Seal retrigger
5. Per held card: Steel x_mult + edition + joker on_held_individual + Red Seal retrigger
6. Per joker: `on_main` (chip_mod, mult_mod, x_mult_mod)
7. Joker `on_after`
8. Glass Card shatter chance
9. Gold Card/Seal dollar bonuses
10. Final score = chips * mult

### `seed_id.py` (122 lines)
Episode seed ID system for roguelite-style replay.

| Export | Type | Description |
|--------|------|-------------|
| `generate_seed_id(seed, timestamp)` | function | Returns `YYYYMMDD-HHMM-XXXXXXXX` |
| `parse_seed_id(seed_id)` | function | Returns dict with `game_seed`, `timestamp`, `seed_str` |
| `seed_id_to_game_seed(seed_id)` | function | Extracts integer seed |

Format: `YYYYMMDD-HHMM-XXXXXXXX` (base-36, 2.8T capacity).

### `core/__init__.py` (18 lines)
Re-exports all key types from submodules. Import from `balatro_gym.core` for convenience.

---

## Gymnasium Environment (`balatro_gym/envs/`)

### `balatro_env.py` (700 lines)
Main Gymnasium wrapper. Converts `GameState` into Gymnasium-compatible obs/action/reward.

| Export | Type | Description |
|--------|------|-------------|
| `BalatroEnv` | gym.Env | Main environment class |
| `TOTAL_ACTIONS` | int | 446 |
| `CARD_SUBSETS` | list[tuple] | Pre-computed C(8,1..5) = 218 card index combinations |
| `PLAY_OFFSET` | int | 0 (actions 0-217) |
| `DISCARD_OFFSET` | int | 218 (actions 218-435) |
| `BUY_OFFSET` | int | 436 (actions 436-438) |
| `SELL_OFFSET` | int | 439 (actions 439-443) |
| `REROLL_ACTION` | int | 444 |
| `SKIP_ACTION` | int | 445 |

Key methods:
- `reset(seed=)` — returns `(obs, info)` where `info["action_mask"]` is bool array
- `step(action)` — returns `(obs, reward, terminated, truncated, info)`
- `action_masks()` — returns bool array of shape `(446,)`
- `save_state()` / `load_state(checkpoint)` — full state persistence
- `from_seed_id(seed_id, config)` — classmethod to replay a seed

Observation dimensions vary by config: Easy=756, Medium=868, Hard=950.

### `configs.py`
Difficulty presets and YAML config system.

| Preset | Antes | Hands/Round | Discards | Start Money | Joker Pool | Consumable Pool |
|--------|-------|-------------|----------|-------------|------------|-----------------|
| Easy | 4 | 5 | 4 | $6 | 10 jokers | Planets + simple Tarots |
| Medium | 6 | 4 | 3 | $4 | 20 jokers | Planets + all Tarots + simple Spectrals |
| Hard | 8 | 4 | 3 | $4 | 30 jokers | All consumables |

Key methods:
- `GameConfig.from_file(path, base="medium")` — Load YAML, merge over base preset. The YAML `base` key selects the preset. Only specified fields override.
- `GameConfig.to_yaml(path)` — Serialize config to YAML.
- `GameConfig.to_dict()` — Serialize to plain dict.

### `envs/__init__.py`
Registers four Gymnasium IDs: `Balatro-v0` (medium), `Balatro-Easy-v0`, `Balatro-Medium-v0`, `Balatro-Hard-v0`.

### `balatro_gym/__init__.py` — Top-level API
Convenience factory functions (MineDojo-style):

| Export | Description |
|--------|-------------|
| `balatro_gym.make(preset, config=, config_path=, seed=)` | Create a single env |
| `balatro_gym.make_vec(preset, num_envs=, seed=, vectorization_mode=)` | Create parallel envs |

### YAML Config Files (`configs/`)
- `configs/defaults.yaml` — Full reference config showing every field (medium preset)
- `configs/example_custom.yaml` — Example: start from easy, override a few fields

---

## RLlib Integration (`agent/rllib/`)

Lives in the `agent` package (not `balatro_gym`). Depends on Ray and Torch.

### `env_wrapper.py` (121 lines)
Wraps `BalatroEnv` for RLlib action masking.

| Export | Type | Description |
|--------|------|-------------|
| `BalatroRLlibEnv` | gym.Wrapper | Observation becomes `Dict(action_mask=Box, observations=Box)` |
| `make_balatro_env(config)` | function | Factory for `ray.tune.register_env` |

The `make_balatro_env` function reads `config["difficulty"]` and `config["seed"]` from the `env_config` dict.

### `action_mask_model.py` (155 lines)
PPO RLModule with action masking.

| Export | Type | Description |
|--------|------|-------------|
| `ActionMaskingTorchRLModule` | RLModule | Masks invalid action logits to -inf |

Extends `PPOTorchRLModule`. Strips `action_mask` from Dict obs, passes clean obs to parent network, then masks logits before distribution construction.

### `train.py` (309 lines)
CLI training script. Run with `python -m agent.rllib.train`.

| Export | Type | Description |
|--------|------|-------------|
| `parse_args()` | function | argparse with all distributed/PPO/architecture knobs |
| `build_config(args)` | function | Constructs `PPOConfig` from CLI args |
| `train(args)` | function | Runs training loop, returns final checkpoint path |

Key CLI flags:
- `--num-env-runners N` — CPU rollout workers
- `--num-gpus-per-learner N` — GPU training (0=CPU, 1=GPU)
- `--num-learners N` — remote learner workers
- `--difficulty easy|medium|hard`
- `--fcnet-hiddens 256 256` — MLP architecture
- `--checkpoint-dir`, `--checkpoint-freq`

### `evaluate.py` (236 lines)
CLI evaluation script. Run with `python -m agent.rllib.evaluate`. Loads checkpoint, runs episodes, prints aggregate stats.

### `rllib/__init__.py` (14 lines)
Exports: `BalatroRLlibEnv`, `make_balatro_env`, `ActionMaskingTorchRLModule`, `build_config`.

---

## Baseline Agents (`agent/baselines/`)

Lives in the `agent` package. Framework-independent baseline agents.

### `random_agent.py` (43 lines)
`RandomAgent` — picks uniformly random valid action from the action mask.

### `heuristic_agent.py` (197 lines)
`HeuristicAgent` — greedy rule-based strategy:
- Play: scores all valid hands, picks the highest-scoring one
- Discard: removes weakest unpaired cards
- Shop: buys cheapest affordable joker, else skips

Note: accesses `env._game` directly (tight coupling, baseline only).

---

## Recording Wrappers (`balatro_gym/wrappers/`)

### `rollout_recorder.py` (196 lines)
`RolloutRecorder(env, save_dir, save_action_mask=False)` — saves per-episode `.npz` files.

Arrays stored: `obs(T+1, obs_dim)`, `actions(T)`, `rewards(T)`, `terminated(T)`, `truncated(T)`, `phases(T)`, `antes(T)`, `scores(T)`, `money(T)`, optionally `action_masks(T, 446)`.

Static method: `RolloutRecorder.load(path)` — returns dict of numpy arrays.

### `episode_stats_recorder.py` (300 lines)
`EpisodeStatsRecorder(env, save_path, flush_every=100)` — appends rows to Snappy-compressed Parquet.

23 columns per episode: `episode_id`, `episode_seed_id`, `timestamp`, `seed`, `won`, `antes_beaten`, `blinds_beaten`, `total_steps`, `total_hands_played`, `total_reward`, `max_score`, `final_money`, `final_ante`, `num_jokers_final`, `num_consumables_final`, `max_ante_reached`, `max_blinds_beaten`, `play_actions`, `discard_actions`, `buy_actions`, `sell_actions`, `reroll_actions`, `skip_actions`.

Static method: `EpisodeStatsRecorder.load(path)` — returns PyArrow Table.

---

## Tests (`tests/`)

382 tests across 13 files. Run with `pytest tests/` (the RLlib smoke test is marked `@pytest.mark.slow`).

| File | Tests | What it covers |
|------|-------|----------------|
| `test_card.py` | 42 | Card properties, uid, enhancements/editions/seals, serialization, Deck ops |
| `test_hand_evaluator.py` | 34 | All 12 hand types, edge cases, wild card flush |
| `test_hand_levels.py` | 18 | HandLevelData math, HandLevelManager level_up |
| `test_joker.py` | 40 | All 30 jokers, scoring contexts, stateful jokers |
| `test_consumable.py` | 50 | All 40 consumables, can_use validation, use effects |
| `test_blind.py` | 26 | BlindType, BlindManager, boss effects |
| `test_shop.py` | 15 | Shop generation, buying, selling, reroll cost |
| `test_game_state.py` | 25 | GameState lifecycle, play_hand, discard, shop, scoring pipeline |
| `test_integration.py` | 26 | 100 random games, enhanced card scoring, consumable effects E2E |
| `test_env.py` | 26 | BalatroEnv init, obs encoding, action masking, reset/step, seed |
| `test_wrappers.py` | 22 | RolloutRecorder (npz), EpisodeStatsRecorder (parquet) |
| `test_seed_and_state.py` | 35 | Seed ID gen/parse, state serialization, env save/load/resume |
| `test_rllib.py` | 24 | RLlib env wrapper, action masking module, config builder, smoke test |

---

## Key Design Patterns

### Registry Pattern (Jokers and Consumables)
Both `joker.py` and `consumable.py` use `@register_joker` / `@register_consumable` decorators to auto-register classes in a module-level dict. `create_joker(id)` / `create_consumable(id)` are factory functions. This makes it easy to add new jokers/consumables.

### Protocol for Circular Import Avoidance
`joker.py` defines `GameStateView` as a `@runtime_checkable Protocol`. `game_state.py` defines `GameStateSnapshot` dataclass that satisfies this protocol. Jokers receive the snapshot (read-only view), not the mutable `GameState`.

Same pattern: `consumable.py` defines `ConsumableGameView` Protocol. `blind.py` uses `TYPE_CHECKING` import.

### Pre-computed Action Subsets
`CARD_SUBSETS` in `balatro_env.py` pre-computes all C(8,k) for k=1..5 = 218 combinations. Action index directly maps to a card subset.

### Seeded Reproducibility
`GameState` uses `numpy.random.Generator` seeded at init. `BalatroEnv.reset(seed=N)` derives a game seed from Gymnasium's `np_random`. `seed_id.py` encodes seeds as human-readable base-36 strings.

---

## Common Tasks

### Adding a new joker
1. Add a class in `core/joker.py` decorated with `@register_joker`
2. Define `INFO = JokerInfo(id="my_joker", name="My Joker", rarity=1, cost=5, description="...")`
3. Override relevant hook methods (`on_main`, `on_individual`, etc.)
4. Add the ID to the appropriate pool in `envs/configs.py`
5. Add tests in `tests/test_joker.py`

### Adding a new consumable
1. Add a class in `core/consumable.py` decorated with `@register_consumable`
2. Define `INFO = ConsumableInfo(id="c_my_card", name="My Card", consumable_type=ConsumableType.TAROT, cost=3, description="...", max_highlighted=1)`
3. Implement `can_use()` and `use()`
4. Add the ID to pools in `envs/configs.py`
5. Add tests in `tests/test_consumable.py`

### Running training
```bash
# Quick local training
python -m agent.rllib.train --difficulty easy --num-env-runners 2 --num-iterations 50

# GPU training with more workers
python -m agent.rllib.train --difficulty easy --num-env-runners 8 --num-gpus-per-learner 1
```

### Evaluating a checkpoint
```bash
python -m agent.rllib.evaluate --checkpoint path/to/checkpoint --num-episodes 100 --difficulty easy
```

---

## File Quick Reference

When looking for specific functionality:

| I want to... | Look in |
|---------------|---------|
| Create an environment | `balatro_gym/__init__.py` (make, make_vec) |
| Understand card scoring | `balatro_gym/core/card.py` (chip_value, enhancement effects) |
| See how hands are detected | `balatro_gym/core/hand_evaluator.py` (evaluate_hand) |
| See the full scoring pipeline | `balatro_gym/core/game_state.py` (_apply_scoring) |
| Find a specific joker | `balatro_gym/core/joker.py` (search by ID or class name) |
| Find a consumable effect | `balatro_gym/core/consumable.py` (search by ID) |
| See blind score targets | `balatro_gym/core/blind.py` (get_blind_amount, BlindManager) |
| See shop mechanics | `balatro_gym/core/shop.py` |
| See observation encoding | `balatro_gym/envs/balatro_env.py` (_encode_observation, _compute_obs_dim) |
| See action masking logic | `balatro_gym/envs/balatro_env.py` (action_masks) |
| See difficulty presets | `balatro_gym/envs/configs.py` (GameConfig.easy/medium/hard) |
| Configure via YAML | `configs/defaults.yaml`, `configs/example_custom.yaml` |
| See training CLI args | `agent/rllib/train.py` (parse_args) |
| See how action masking works in RLlib | `agent/rllib/action_mask_model.py` |
| See baseline agents | `agent/baselines/` (random_agent.py, heuristic_agent.py) |
| See trajectory recording | `balatro_gym/wrappers/rollout_recorder.py` |
| See episode statistics | `balatro_gym/wrappers/episode_stats_recorder.py` |
| See state serialization | `balatro_gym/core/game_state.py` (serialize/deserialize) |
| See seed ID format | `balatro_gym/core/seed_id.py` |
