# Balatro-Agent: Card Game Reasoning Gym + RL-Trained Agent

## What This Project Is

A final project for COMP_SCI 496 (Agent AI, Prof. Manling Li, Northwestern, Spring 2026). Two deliverables:

1. **A Gymnasium-compatible card game environment** inspired by Balatro (a deck-building roguelite poker game). This is a reasoning gym that tests agents on strategic decision-making: probability, expected value, combinatorics, long-term planning, and risk assessment. Designed as reusable open-source infrastructure.

2. **An RL agent trained via PPO** (Proximal Policy Optimization) with action masking to play the card game, demonstrating that RL training can learn strategies beyond hand-crafted heuristics.

## Game Mechanics (Simplified Balatro)

The game is a **single-player** deck-building poker game:

- **Deck**: Standard 52 cards (4 suits x 13 ranks), mutable (cards can be added/removed by consumables)
- **Hand**: Player is dealt 8 cards, selects up to 5 to play as a poker hand
- **Card properties**: Cards can have enhancements (8 types), editions (3 types), and seals (4 types) that modify scoring
- **Discards**: Player gets a limited number of discards per round (draw replacement cards from deck)
- **Scoring**: Each poker hand type has base chips and base multiplier (upgradable via Planet cards). Final score = chips x mult. Jokers and card properties can modify both.
- **Jokers**: 30 special cards that modify scoring rules (e.g., "+4 mult for each pair", "x1.5 if hand contains a heart", "+30 chips for each face card played"). Player can hold up to 5 jokers.
- **Consumables**: Tarot cards (22), Planet cards (12), and Spectral cards (6) that modify cards, level up hands, or alter the deck. Player can hold up to 2 consumables.
- **Shop**: Between rounds, spend money ($) to buy jokers/consumables, reroll shop offerings, or sell owned jokers.
- **Blinds**: Each ante has 3 blinds (small, big, boss) with increasing score targets. Player must meet the target to advance.
- **Boss blinds**: Special blinds with debuff effects (e.g., "all hearts are face-down", "first hand played scores 0").
- **Win**: Survive all antes (e.g., 8 antes = 24 blinds). **Lose**: Fail to reach a blind's score target.
- **Economy**: Earn $ from winning rounds + interest on savings (1$ per 5$ held, max 5$ interest).

This is a **clean-room implementation** based on publicly known game mechanics. No copyrighted code is used.

## Architecture

```
Balatro-Agent/
├── balatro_gym/                         # ENVIRONMENT PACKAGE (no Ray/Torch dependency)
│   ├── environment_gym.yml             # Conda env: balatro-gym (env only)
│   ├── __init__.py                     # make(), make_vec(), version, re-exports
│   ├── core/
│   │   ├── __init__.py                 # Re-exports all core types
│   │   ├── card.py                     # Card, Deck, Enhancement, Edition, Seal enums
│   │   ├── hand_evaluator.py           # Poker hand detection + base scoring (12 hand types)
│   │   ├── hand_levels.py              # Mutable hand-type chip/mult levels (Planet card targets)
│   │   ├── joker.py                    # 30 joker definitions with registry pattern
│   │   ├── consumable.py              # 40 consumables (Tarots, Planets, Spectrals) with registry
│   │   ├── blind.py                    # Blind progression, 9 boss blind effects
│   │   ├── shop.py                     # Shop: offerings, buying, selling, rerolling
│   │   ├── game_state.py              # Full game state manager + 10-step scoring pipeline
│   │   └── seed_id.py                 # Episode seed IDs (YYYYMMDD-HHMM-XXXXXXXX)
│   ├── envs/
│   │   ├── __init__.py                 # Gymnasium environment registration (4 env IDs)
│   │   ├── balatro_env.py              # Main Gymnasium environment (Discrete(446), action masking)
│   │   ├── configs.py                  # GameConfig + YAML merge + easy/medium/hard presets
│   │   └── rewards.py                  # Pluggable reward system (DefaultReward, SparseReward)
│   ├── wrappers/
│   │   ├── __init__.py                 # Exports RolloutRecorder, EpisodeStatsRecorder
│   │   ├── rollout_recorder.py         # Per-episode .npz trajectory recording
│   │   └── episode_stats_recorder.py   # Per-episode Parquet summary statistics
│   ├── rendering/
│   │   └── __init__.py                 # (placeholder for future text rendering)
│   └── utils/
│       └── __init__.py                 # (placeholder for future math/metric utils)
├── agent/                               # AGENT PACKAGE (depends on balatro_gym + Ray + Torch)
│   ├── __init__.py                     # Exports Agent protocol, run_episode, evaluate_agent
│   ├── base.py                         # Agent Protocol + run_episode + evaluate_agent helpers
│   ├── baselines/
│   │   ├── __init__.py                 # Exports RandomAgent, HeuristicAgent
│   │   ├── random_agent.py             # Baseline: uniform random valid action
│   │   └── heuristic_agent.py          # Baseline: greedy rule-based strategy
│   ├── rllib/
│   │   ├── __init__.py                 # Exports BalatroRLlibEnv, ActionMaskingTorchRLModule
│   │   ├── env_wrapper.py              # Dict obs wrapper for RLlib action masking
│   │   ├── action_mask_model.py        # PPO TorchRLModule with action masking
│   │   ├── callbacks.py                # BalatroMetricsCallback (game-specific RLlib metrics)
│   │   ├── train.py                    # CLI training script with distributed config
│   │   └── evaluate.py                 # CLI evaluation script
│   └── llm/
│       ├── __init__.py                 # Exports GameStateRenderer
│       ├── renderer.py                 # JSON game state renderer for LLM agents
│       └── backends.py                 # ModelBackend Protocol + TODO implementations
├── configs/                             # YAML configuration files
│   ├── defaults.yaml                   # Full default config (medium preset, all fields)
│   └── example_custom.yaml             # Example custom config (overrides a few fields)
├── tests/                              # 383 unit tests across 13 test files
├── docs/
│   ├── Balatro_Gym_Guidance.md         # Full usage guide: obs/action space, training, recording
│   ├── Project_Proposal.md            # Course project proposal
│   ├── PLAN.md                         # Original implementation plan
│   └── Tech_Log.md                     # Technical decisions and debugging log
├── experiments/                        # (placeholder for experiment scripts)
├── Manual.md                           # Project structure guide for AI assistants
├── environment.yml                     # Conda env: balatro-agent (full project)
├── setup.py                            # pip install (extras: recording, agent, dev, all)
├── requirements.txt                    # Legacy pip deps (prefer conda YAMLs)
├── CLAUDE.md                           # This file
└── README.md
```

### Package Dependency

```
balatro_gym (environment)  ←── agent (training/baselines)
    gymnasium, numpy           ray[rllib], torch
```

One-way dependency: `agent` imports from `balatro_gym`. The environment package has zero dependency on agent code.

## Key Technical Decisions

### Gymnasium API
The environment follows the standard Gymnasium interface:
- `env = balatro_gym.make("easy")` — create environment (preferred entry point)
- `env = BalatroEnv(config=GameConfig.easy())` — direct construction
- `env = BalatroEnv(config_path="configs/example_custom.yaml")` — from YAML config
- `obs, info = env.reset()` — start a new game
- `obs, reward, terminated, truncated, info = env.step(action)` — take an action
- `info["action_mask"]` — boolean mask over valid actions
- `vec_env = balatro_gym.make_vec("easy", num_envs=8)` — parallel envs for data collection
- Registered env IDs: `Balatro-v0`, `Balatro-Easy-v0`, `Balatro-Medium-v0`, `Balatro-Hard-v0`

### Game Phases (Action Space)
The game has distinct phases with different action spaces:
1. **Play phase**: Select which cards from hand to play (or discard). Action = card subset index.
2. **Shop phase**: Buy a joker/consumable, sell a joker, reroll shop, or skip. Action = shop action index.

### Reward Design (Pluggable)
The reward system is pluggable via `balatro_gym/envs/rewards.py`:
- `DefaultReward` — shaped reward with tunable coefficients:
  - +1 for beating a blind (scaled by progress), -1 for failing, +10 for winning
  - Partial credit based on score progress toward blind target
  - -0.01 penalty for invalid actions
- `SparseReward` — only +1 (win) / -1 (lose), no shaping
- Custom rewards: any callable matching `RewardFunction` Protocol (`RewardContext -> float`)
- `RewardContext` is a frozen dataclass with all game state needed for reward computation

Usage: `balatro_gym.make("easy", reward_fn=SparseReward())` or pass custom callable

### Observation Space
Flat float32 vector (dimension varies by config: Easy=756, Medium=868, Hard=950):
- 8 x 68-dim per-card features (52 card one-hot + 8 enhancement + 3 edition + 4 seal + 1 face_down)
- 5 x joker one-hot slots
- 2 x consumable one-hot slots
- 12 x 3 hand level features (level, chips, mult)
- Normalized scalars (money, ante, blind type, score, hands/discards remaining, deck size, phase)
- Shop state features

### Action Space
Discrete(446) with action masking:
- Actions 0-217: Play card subsets (all C(8,1..5) = 218 combinations)
- Actions 218-435: Discard card subsets (same 218 combinations)
- Actions 436-438: Buy shop slots 0-2
- Actions 439-443: Sell joker slots 0-4
- Action 444: Reroll shop
- Action 445: Skip shop

### Agent Protocol (`agent/base.py`)
All agents implement a common protocol:
- `act(obs: np.ndarray, info: dict) -> int` — choose an action
- `reset() -> None` — reset internal state between episodes

Helper functions:
- `run_episode(env, agent, seed=, max_steps=)` → dict with episode stats
- `evaluate_agent(env, agent, num_episodes=, seed=, verbose=)` → aggregate stats

### LLM Agent Infrastructure (`agent/llm/`)
For LLM-based agents that read game state as JSON:
- `GameStateRenderer(env)` — converts env internal state to structured JSON dict
  - Sections: game_progress, hand, jokers, consumables, economy, hand_levels, valid_actions, shop
  - Valid actions include human-readable descriptions (card names, types)
  - `render(info)` → dict, `render_json(info)` → formatted JSON string
- `ModelBackend` Protocol — interface for LLM backends (`generate(prompt) -> str`)
  - Planned: HuggingFaceBackend, OpenAIBackend, AnthropicBackend (currently TODO stubs)

### Game-Specific Metrics (`agent/rllib/callbacks.py`)
`BalatroMetricsCallback` extracts per-episode metrics during training:
- win_rate, blinds_beaten, ante_reached, final_money, final_score, episode_length

## Training Pipeline

1. **Baselines first**: Run random and heuristic agents, record performance.
2. **PPO training**: Train MLP policy via Ray RLlib with action masking.
3. **Curriculum training**: Easy → medium → hard difficulty progression.
4. **Evaluation**: Compare all agents on win rate, average ante reached, and survival curves.

## Experiments to Run

1. **Baseline comparison**: Random vs. Heuristic vs. PPO vs. PPO-Curriculum
2. **Reward shaping ablation**: Sparse (win/lose only) vs. shaped. Does shaping help?
3. **Curriculum learning**: Easy→hard vs. hard-only. Does curriculum help?
4. **Network architecture**: MLP [256,256] vs. [512,256] vs. [128,128,128]

## Compute

- Simulator: CPU only
- PPO training: CPU or 1x GPU, ~2-8 hours per run (distributed rollout collection via Ray)
- Available hardware: 2x H100/A100 GPUs on remote server
- Total budget: ~20-40 CPU/GPU-hours

## Environment Setup

Preferred method is conda:
```bash
# Full project (environment + agent training)
conda env create -f environment.yml
conda activate balatro-agent

# Environment only (for users who just need balatro_gym)
conda env create -f balatro_gym/environment_gym.yml
conda activate balatro-gym
```

Alternative (pip):
```bash
pip install -e ".[all]"   # everything
pip install -e .           # environment only
```

## Tech Stack

- Python 3.10+
- `gymnasium` — environment API
- `ray[rllib]` — distributed PPO training with action masking
- `torch` — neural network backend
- `pyarrow` — Parquet statistics recording
- `numpy` — array operations
- `pytest` — testing
- `pyyaml` — config file loading

## Development Status

All infrastructure is DONE. Remaining work is experiments and evaluation.

### Done:
1. `balatro_gym/core/` — Full game engine (card, hand_evaluator, hand_levels, joker, consumable, blind, shop, game_state, seed_id)
2. `balatro_gym/envs/` — Gymnasium env, configs, pluggable rewards
3. `balatro_gym/wrappers/` — RolloutRecorder, EpisodeStatsRecorder
4. `agent/baselines/` — RandomAgent, HeuristicAgent
5. `agent/rllib/` — RLlib integration (env wrapper, action mask model, callbacks, train/evaluate CLIs)
6. `agent/llm/` — GameStateRenderer, ModelBackend Protocol
7. `agent/base.py` — Agent Protocol + evaluation helpers
8. Tests — 383 tests across 13 test files (all passing)
9. Conda environment YAMLs, configs, documentation

### TODO (Experiments — the remaining deliverables):
1. **PPO training runs** — Train on easy/medium/hard, ~200+ iterations
2. **Reward shaping ablation** — DefaultReward vs SparseReward comparison
3. **Curriculum learning** — Easy→hard vs hard-only
4. **Network architecture search** — MLP [256,256] vs [512,256] vs [128,128,128]
5. **Baseline comparison** — Random vs Heuristic vs PPO (win rate, blinds beaten, ante reached)
6. **LLM agent** — Implement a HuggingFace backend, run Qwen on the game (stretch goal)
7. **Final report + presentation** — Due Jun 1 (presentation) and Jun 8 (report)

### Baseline Performance (verified):
- RandomAgent on easy: ~1.4 blinds beaten per game, 0% win rate
- HeuristicAgent on easy: ~9.3 blinds beaten per game, high win rate
- This gap proves the environment rewards strategic play

## Course Context

This project is for COMP_SCI 496 (Agent AI) at Northwestern. Key syllabus topics it covers:
- MDP formulation (Week 1)
- Reward modeling and world modeling (Week 1-2)
- RL training with PPO, reward shaping, curriculum learning (Week 3)
- Agent planning and reasoning (Week 3, Week 8)
- Evaluation metrics (Week 5)

Proposal presentation: Apr 20. Final presentation: Jun 1. Report due: Jun 8.

## Coding Guidelines

- The developer is most comfortable with Python. Keep code straightforward and well-structured.
- Avoid over-engineering. Start simple, add complexity only when needed.
- Write unit tests for core game logic (hand evaluation, scoring, joker effects).
- Use type hints for clarity.
- Keep the Gymnasium API contract strict — other researchers should be able to use this env.
