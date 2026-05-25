# Balatro-Agent

An RL agent that learns to play the roguelite deck-building game Balatro.

## Introduction

This project has two parts:

1. **Balatro-Gym** -- A Gymnasium-compatible simulation engine that recreates Balatro's game mechanics in Python. It serves as a reasoning gym for training and evaluating AI agents on strategic decision-making: probability estimation, expected value calculation, combinatorial optimization, and long-horizon planning.

2. **Balatro-Agent** -- An RL agent trained via PPO (Proximal Policy Optimization) with action masking to play the game, learning strategies that go beyond hand-crafted heuristics.

### Project Status

| Component | Status |
|-----------|--------|
| Core game engine (cards, hand evaluation, jokers, blinds, shop, game state) | Done |
| Card properties (8 enhancements, 3 editions, 4 seals) | Done |
| Hand level system (Planet card upgrades) | Done |
| Consumables (22 Tarots, 12 Planets, 6 Spectrals) | Done |
| Gymnasium environment wrapper (observation encoding, action masking) | Done |
| Difficulty configs (easy/medium/hard presets) | Done |
| Baseline agents (random, heuristic) | Done |
| Recording wrappers (RolloutRecorder, EpisodeStatsRecorder) | Done |
| Episode seed IDs and state serialization | Done |
| Ray RLlib integration (distributed PPO with action masking) | Done |
| PPO training experiments | TODO |
| Curriculum learning (easy -> hard) | TODO |
| Evaluation and ablation experiments | TODO |

### About Balatro

[Balatro](https://www.playbalatro.com/) is a single-player roguelite where you score points by playing poker hands -- but with a twist. You collect **Jokers** (modifier cards) that warp the scoring rules, and you must build synergies between your Jokers and your deck to hit exponentially growing score targets.

**Why is Balatro a good environment for agent planning research?**

Consider a state where your current hand cannot satisfy the score requirement, and you have one discard remaining. The agent faces a high-dimensional planning problem:

- **Option A (The Safe Play):** Discard for a **Flush**. Your current Jokers provide a mult boost that guarantees current round win immediately. This ensures survival but offers zero long-term growth.
- **Option B (The Scaling Play):** Discard for a **Straight**. While this hand won't win the round in one go, you have a "Scaling Joker" (like *Square Joker* or *Bus*) that permanently increases its power whenever a Straight is played.

**Reasoning and Planning:** An agent must evaluate if its current health (hands remaining) and deck probability allow it to take the "suboptimal" short-term play (the Straight) to ensure its "scaling" is high enough to survive the exponential difficulty of later Antes. This requires the model to value **future state utility** over **immediate reward.**

## Quick Start

### Setup (conda)

```bash
# Option 1: Full project (environment + agent training)
conda env create -f environment.yml
conda activate balatro-agent

# Option 2: Environment only (for users who only need balatro_gym)
conda env create -f ./balatro_gym/environment_gym.yml
conda activate balatro-gym
```

### Setup (pip)

```bash
# Environment only
pip install -e .

# Full project (environment + agent training)
pip install -e ".[all]"
```

```python
# Basic usage
import balatro_gym

env = balatro_gym.make("easy")
obs, info = env.reset(seed=42)
mask = info["action_mask"]

# Parallel environments for data collection
vec_env = balatro_gym.make_vec("easy", num_envs=8, seed=42)
obs_batch, infos = vec_env.reset()

# Custom config from YAML
env = balatro_gym.make(config_path="configs/example_custom.yaml")
```

```bash
# Train with Ray RLlib
python -m agent.rllib.train --difficulty easy --num-env-runners 4
```

See `docs/Balatro_Gym_Guidance.md` for full documentation.

## Tech Stack

- Python 3.10+
- `gymnasium` -- environment API
- `ray[rllib]` -- distributed PPO training with action masking
- `torch` -- neural network backend
- `numpy` -- numerical computation
- `pyarrow` -- Parquet recording
- `pytest` -- testing (383 tests)
- `pyyaml` -- config loading
