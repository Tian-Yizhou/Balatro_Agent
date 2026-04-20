# Balatro Gym Guidance

A Gymnasium-compatible environment for the Balatro card game, designed for reinforcement learning research and agent evaluation.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Environment Overview](#environment-overview)
- [Observation Space](#observation-space)
- [Action Space](#action-space)
- [Reward Structure](#reward-structure)
- [Difficulty Presets](#difficulty-presets)
- [Custom Configuration](#custom-configuration)
- [Training with PPO (Ray RLlib)](#training-with-ppo-ray-rllib)
- [Recording and Logging](#recording-and-logging)
- [Episode Seed IDs and State Resume](#episode-seed-ids-and-state-resume)
- [Game Mechanics Reference](#game-mechanics-reference)
- [API Reference](#api-reference)

---

## Installation

```bash
# Clone and install in development mode
git clone <repo-url>
cd Balatro-Agent
pip install -e .

# Dependencies
pip install gymnasium numpy

# For RL training with Ray RLlib
pip install -e ".[rllib]"
# or manually:
pip install "ray[rllib]" torch

# For recording wrappers (trajectory + stats)
pip install -e ".[recording]"

# Everything at once
pip install -e ".[all]"
```

After installation, the following Gymnasium environment IDs are available:

| ID | Difficulty | Description |
|----|-----------|-------------|
| `Balatro-v0` | Medium | Default — 6 antes, 20 jokers, 40 consumables |
| `Balatro-Easy-v0` | Easy | 4 antes, extra hands/discards, starter joker |
| `Balatro-Medium-v0` | Medium | Same as `Balatro-v0` |
| `Balatro-Hard-v0` | Hard | 8 antes, all 30 jokers, all 44 consumables |

## Quick Start

### Basic Usage

```python
import balatro_gym
import gymnasium as gym
import numpy as np

# Create environment
env = gym.make("Balatro-Easy-v0")

# Reset and get initial observation + action mask
obs, info = env.reset(seed=42)
mask = info["action_mask"]  # boolean array, shape (446,)

# Game loop
terminated = False
while not terminated:
    # Pick a random valid action
    valid_actions = np.where(mask)[0]
    action = int(np.random.choice(valid_actions))

    obs, reward, terminated, truncated, info = env.step(action)
    mask = info["action_mask"]

print(f"Game ended — phase: {info['phase']}, blinds beaten: {info['blinds_beaten']}")
```

### With Rendering

```python
env = gym.make("Balatro-Easy-v0", render_mode="ansi")
obs, info = env.reset(seed=42)

# Print game state
text = env.render()
print(text)
# === Ante 1 | Small Blind | Phase: play ===
# Score: 0 / 300
# Hands: 5  Discards: 4  Money: $6
# Hand: 5H JC AH 6H 2D 3D 6C 7H
# Jokers: Joker (joker_basic)
```

### Direct Instantiation (without gym.make)

```python
from balatro_gym.envs.balatro_env import BalatroEnv
from balatro_gym.envs.configs import GameConfig

# Using a preset
env = BalatroEnv(config_preset="easy")

# Using a custom config
config = GameConfig(
    num_antes=4,
    hands_per_round=5,
    discards_per_round=4,
    joker_pool=["joker_basic", "greedy_joker", "fibonacci"],
    consumable_pool=["c_pluto", "c_mercury", "c_earth"],
    seed=42,
)
env = BalatroEnv(config=config)
obs, info = env.reset()
```

---

## Environment Overview

The Balatro environment simulates a simplified version of the Balatro card game. The player must beat a series of "blinds" (score targets) by playing poker hands from an 8-card hand. Between rounds, the player visits a shop to buy jokers (permanent scoring modifiers) and consumables (single-use cards).

### Game Flow

```
For each Ante (1 to num_antes):
    For each Blind (Small, Big, Boss):
        PLAY PHASE:
            - Player is dealt 8 cards from a 52-card deck
            - Player selects 1-5 cards to PLAY as a poker hand, or DISCARD cards to draw replacements
            - Playing a hand scores chips * mult (modified by jokers, card properties, hand levels)
            - Repeat until score >= target (advance) or hands run out (game over)
        SHOP PHASE:
            - Player can BUY jokers or consumables, SELL jokers, REROLL shop, or SKIP
            - Interest earned: $1 per $5 held (max $5 interest)

Game ends when all antes are cleared (WIN) or a blind's target is not met (LOSE).
```

### Game Phases

The environment alternates between two phases, each with different valid actions:

| Phase | Valid Actions | Description |
|-------|-------------|-------------|
| `play` | Play hand, Discard | Select cards to play or discard |
| `shop` | Buy, Sell, Reroll, Skip | Manage jokers and consumables between rounds |

---

## Observation Space

The observation is a flat `float32` vector. Its size varies by configuration because the joker and consumable pools differ across difficulty levels.

| Preset | Observation Dim | Joker Pool | Consumable Pool |
|--------|----------------|------------|-----------------|
| Easy | 756 | 10 | 26 |
| Medium | 868 | 20 | 40 |
| Hard | 950 | 30 | 44 |

### Observation Layout

The observation vector is divided into these sections (in order):

| Section | Size | Description |
|---------|------|-------------|
| Hand cards | 8 x 68 = 544 | Per-card features for each hand slot |
| Joker slots | max_jokers x (J+1) | One-hot encoding per joker slot |
| Consumable slots | 2 x (C+1) | One-hot encoding per consumable slot |
| Hand levels | 12 x 3 = 36 | Level/chips/mult per hand type |
| Money | 1 | Normalized by 100 |
| Ante | 1 | Normalized by num_antes |
| Blind type | 3 | One-hot: small/big/boss |
| Score target | 1 | Log-normalized |
| Score progress | 1 | current_score / score_target |
| Hands remaining | 1 | Normalized |
| Discards remaining | 1 | Normalized |
| Deck size | 1 | Normalized by 52 |
| Phase | 2 | One-hot: play/shop |
| Shop joker slots | shop_slots x (J+2) | Joker one-hot + cost + sold flag |
| Shop consumable slot | 1 x (C+2) | Consumable one-hot + cost + sold flag |

Where `J` = number of joker types in pool, `C` = number of consumable types in pool.

### Per-Card Features (68 dimensions)

Each of the 8 hand card slots is encoded as a 68-dimensional vector:

| Offset | Size | Description |
|--------|------|-------------|
| 0-51 | 52 | One-hot: which card (4 suits x 13 ranks) |
| 52-59 | 8 | One-hot: enhancement (Bonus/Mult/Wild/Glass/Steel/Stone/Gold/Lucky) |
| 60-62 | 3 | One-hot: edition (Foil/Holo/Polychrome) |
| 63-66 | 4 | One-hot: seal (Gold/Red/Blue/Purple) |
| 67 | 1 | Face-down flag (boss blind debuff) |

Empty card slots are all zeros.

---

## Action Space

**Type:** `Discrete(446)` with action masking.

Actions are divided into six groups:

| Action Range | Count | Phase | Description |
|-------------|-------|-------|-------------|
| 0-217 | 218 | Play | Play a subset of hand cards as a poker hand |
| 218-435 | 218 | Play | Discard a subset of hand cards and draw replacements |
| 436-438 | 3 | Shop | Buy shop offering at slot 0, 1, or 2 |
| 439-443 | 5 | Shop | Sell joker at slot 0, 1, 2, 3, or 4 |
| 444 | 1 | Shop | Reroll shop offerings |
| 445 | 1 | Shop | Skip shop (advance to next blind) |

### Card Subset Encoding

Actions 0-217 (play) and 218-435 (discard) each encode one of the 218 possible subsets of indices from an 8-card hand. The subsets cover all combinations of 1 to 5 cards:

- C(8,1) = 8 single-card subsets
- C(8,2) = 28 two-card subsets
- C(8,3) = 56 three-card subsets
- C(8,4) = 70 four-card subsets
- C(8,5) = 56 five-card subsets
- **Total: 218 subsets**

You can access the mapping from action index to card indices:

```python
from balatro_gym.envs.balatro_env import CARD_SUBSETS

# Action 0 plays card index (0,)
# Action 8 plays card indices (0, 1)
# etc.
print(CARD_SUBSETS[0])   # (0,)
print(CARD_SUBSETS[8])   # (0, 1)
print(CARD_SUBSETS[217]) # (3, 4, 5, 6, 7)
```

### Action Masking

The environment provides a boolean action mask indicating which actions are currently valid. **Always use the mask** — taking an invalid action returns a small penalty (-0.01) with no game state change.

```python
obs, info = env.reset(seed=42)
mask = info["action_mask"]  # np.ndarray, shape (446,), dtype=bool

# After each step
obs, reward, terminated, truncated, info = env.step(action)
mask = info["action_mask"]

# The unwrapped env also exposes action_masks() directly
# (used by BalatroRLlibEnv wrapper for RLlib action masking)
mask = env.unwrapped.action_masks()
```

---

## Reward Structure

| Event | Reward |
|-------|--------|
| Beat a blind | +1.0 + progress_bonus (blinds_beaten / total_blinds) |
| Score progress toward blind target | +0.0 to +0.01 (proportional to score delta) |
| Win the game (all antes cleared) | +10.0 |
| Lose the game (failed a blind) | -1.0 |
| Invalid action | -0.01 |

The reward is shaped to guide learning: agents receive incremental reward for scoring progress and larger bonuses for clearing blinds and winning the game.

---

## Difficulty Presets

### Easy

Best for initial training and debugging. Forgiving parameters with a starter joker.

```python
env = gym.make("Balatro-Easy-v0")
```

| Parameter | Value |
|-----------|-------|
| Antes | 4 |
| Hands per round | 5 |
| Discards per round | 4 |
| Starting money | $6 |
| Starting joker | Joker (basic) |
| Joker pool | 10 (priority 1) |
| Consumable pool | 26 (12 planets + 14 simple tarots) |

### Medium (Default)

Standard difficulty. No starting advantages.

```python
env = gym.make("Balatro-v0")  # or "Balatro-Medium-v0"
```

| Parameter | Value |
|-----------|-------|
| Antes | 6 |
| Hands per round | 4 |
| Discards per round | 3 |
| Starting money | $4 |
| Starting joker | None |
| Joker pool | 20 (priority 1 + 2) |
| Consumable pool | 40 (planets + all tarots + simple spectrals) |

### Hard

Full game with all content.

```python
env = gym.make("Balatro-Hard-v0")
```

| Parameter | Value |
|-----------|-------|
| Antes | 8 |
| Hands per round | 4 |
| Discards per round | 3 |
| Starting money | $4 |
| Starting joker | None |
| Joker pool | 30 (all jokers) |
| Consumable pool | 44 (all consumables) |

---

## Custom Configuration

### GameConfig

Create a fully custom environment by constructing a `GameConfig`:

```python
from balatro_gym.envs.balatro_env import BalatroEnv
from balatro_gym.envs.configs import GameConfig

config = GameConfig(
    num_antes=4,
    hands_per_round=6,
    discards_per_round=4,
    hand_size=8,
    max_jokers=5,
    starting_money=10,
    shop_slots=2,
    reroll_base_cost=3,
    consumable_slots=2,
    joker_pool=["joker_basic", "greedy_joker", "fibonacci", "half_joker"],
    starting_joker_ids=["joker_basic"],
    consumable_pool=["c_pluto", "c_mercury", "c_earth", "c_strength"],
    seed=42,
)

env = BalatroEnv(config=config)
obs, info = env.reset()
```

### GameConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_antes` | int | 8 | Number of antes (3 blinds each) |
| `hands_per_round` | int | 4 | Hands allowed per blind |
| `discards_per_round` | int | 3 | Discards allowed per blind |
| `hand_size` | int | 8 | Cards dealt to hand |
| `max_jokers` | int | 5 | Maximum joker slots |
| `starting_money` | int | 4 | Starting dollars |
| `shop_slots` | int | 2 | Number of joker offerings in shop |
| `reroll_base_cost` | int | 5 | Base cost to reroll shop |
| `consumable_slots` | int | 2 | Maximum consumable slots |
| `joker_pool` | list[str] | [] | Available joker IDs for shop |
| `starting_joker_ids` | list[str] | [] | Jokers given at game start |
| `consumable_pool` | list[str] | [] | Available consumable IDs for shop |
| `seed` | int \| None | None | RNG seed for reproducibility |

### Loading Config from YAML

```python
config = GameConfig.from_file("my_config.yaml")
env = BalatroEnv(config=config)
```

Example YAML:

```yaml
num_antes: 4
hands_per_round: 5
discards_per_round: 4
max_jokers: 5
starting_money: 6
joker_pool:
  - joker_basic
  - greedy_joker
  - fibonacci
consumable_pool:
  - c_pluto
  - c_mercury
starting_joker_ids:
  - joker_basic
```

---

## Training with PPO (Ray RLlib)

The environment uses [Ray RLlib](https://docs.ray.io/en/latest/rllib/) for distributed PPO training with action masking. The `balatro_gym.rllib` package provides:

- **`BalatroRLlibEnv`** — wraps the base env with a Dict observation space (`{"observations": ..., "action_mask": ...}`)
- **`ActionMaskingTorchRLModule`** — PPO RLModule that masks invalid actions to `-inf` logits
- **`train.py`** — CLI script with full control over distributed resource allocation
- **`evaluate.py`** — CLI script for checkpoint evaluation

### Basic Training (CLI)

```bash
# Minimal: 2 CPU rollout workers, CPU training, easy difficulty
python -m balatro_gym.rllib.train --difficulty easy --num-env-runners 2

# GPU training with 8 CPU rollout workers
python -m balatro_gym.rllib.train \
    --difficulty easy \
    --num-env-runners 8 \
    --num-gpus-per-learner 1

# Multi-GPU: 2 learner workers each with 1 GPU, 16 CPU rollout workers
python -m balatro_gym.rllib.train \
    --difficulty medium \
    --num-env-runners 16 \
    --num-learners 2 \
    --num-gpus-per-learner 1

# Vectorized envs on each runner (faster sampling)
python -m balatro_gym.rllib.train \
    --difficulty easy \
    --num-env-runners 8 \
    --num-envs-per-env-runner 4

# Custom hyperparameters and architecture
python -m balatro_gym.rllib.train \
    --difficulty easy \
    --lr 1e-4 \
    --gamma 0.995 \
    --entropy-coeff 0.005 \
    --fcnet-hiddens 512 256 \
    --train-batch-size 8000 \
    --num-iterations 500 \
    --checkpoint-freq 50
```

### Resource Allocation

The key idea: **rollout collection** runs on CPU workers and **training** runs on a separate learner (optionally on GPU). You control each independently:

| Flag | Controls | Typical value |
|------|----------|---------------|
| `--num-env-runners N` | CPU rollout workers (sampling) | 2–16 |
| `--num-envs-per-env-runner N` | Vectorized envs per worker | 1–4 |
| `--num-learners N` | Remote learner workers | 0 (local) or 1–2 |
| `--num-gpus-per-learner N` | GPUs per learner | 0 (CPU) or 1 |
| `--num-cpus-per-env-runner N` | CPUs per rollout worker | 1 |
| `--num-gpus-per-env-runner N` | GPUs per rollout worker | 0 |

Example resource layouts:

```
# Laptop (4 CPU cores):
--num-env-runners 2

# Workstation (16 cores + 1 GPU):
--num-env-runners 12 --num-gpus-per-learner 1

# Multi-GPU server (64 cores + 4 GPUs):
--num-env-runners 48 --num-learners 4 --num-gpus-per-learner 1

# Remote Ray cluster:
--ray-address auto --num-env-runners 32 --num-gpus-per-learner 1
```

### Programmatic Training

```python
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from balatro_gym.rllib import (
    make_balatro_env,
    ActionMaskingTorchRLModule,
)

ray.init()
register_env("Balatro", make_balatro_env)

config = (
    PPOConfig()
    .environment(
        env="Balatro",
        env_config={"difficulty": "easy"},
    )
    .env_runners(
        num_env_runners=4,
        num_envs_per_env_runner=2,
    )
    .learners(
        num_gpus_per_learner=0,  # set to 1 for GPU training
    )
    .training(
        lr=3e-4,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        entropy_coeff=0.01,
        train_batch_size_per_learner=4000,
        minibatch_size=256,
        num_epochs=10,
    )
    .rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=ActionMaskingTorchRLModule,
            model_config={
                "head_fcnet_hiddens": [256, 256],
                "head_fcnet_activation": "relu",
            },
        ),
    )
)

algo = config.build()

for i in range(200):
    result = algo.train()
    mean_reward = result["env_runners"]["episode_reward_mean"]
    print(f"Iter {i}: reward_mean={mean_reward:.2f}")

    if i % 50 == 0:
        algo.save("checkpoints/balatro_ppo")

algo.stop()
ray.shutdown()
```

### Evaluation (CLI)

```bash
# Evaluate a checkpoint over 100 episodes
python -m balatro_gym.rllib.evaluate \
    --checkpoint checkpoints/balatro_ppo/checkpoint_000200 \
    --num-episodes 100 \
    --difficulty easy \
    --verbose
```

### Curriculum Training

Train on easy first, then restore the checkpoint for harder difficulties:

```bash
# Phase 1: Easy (200 iterations)
python -m balatro_gym.rllib.train \
    --difficulty easy \
    --num-iterations 200 \
    --checkpoint-dir checkpoints/phase1

# Phase 2: Medium — restore from Phase 1 checkpoint
# (programmatic — load checkpoint, rebuild with new env_config)
```

```python
# Curriculum in code
config_easy = build_ppo_config(difficulty="easy")
algo = config_easy.build()
for i in range(200):
    algo.train()
checkpoint = algo.save("checkpoints/curriculum")
algo.stop()

config_medium = build_ppo_config(difficulty="medium")
algo = config_medium.build()
algo.restore(checkpoint)
for i in range(200):
    algo.train()
algo.stop()
```

> **Note:** The observation dimension changes across difficulty levels (756 for Easy, 868 for Medium, 950 for Hard) because the joker/consumable pool sizes differ. For curriculum training across difficulties, consider using a fixed pool size config across all stages.

---

## Recording and Logging

The package includes two Gymnasium wrappers for data collection: **RolloutRecorder** (per-step trajectory data for RL training) and **EpisodeStatsRecorder** (per-episode summary statistics for analysis). Both are composable and can be used independently or together.

```bash
# pyarrow is required for EpisodeStatsRecorder
pip install pyarrow
# or install with the recording extra:
pip install -e ".[recording]"
```

### RolloutRecorder — Trajectory Data for RL Training

Records every `(obs, action, reward, done)` transition. Each episode produces one compressed `.npz` file. Use these for offline RL, imitation learning, or replay buffers.

```python
import balatro_gym
import gymnasium as gym
from balatro_gym.wrappers import RolloutRecorder

env = gym.make("Balatro-Easy-v0")
env = RolloutRecorder(env, save_dir="data/rollouts")

# Play episodes — .npz files are saved automatically on termination
for seed in range(100):
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        mask = info["action_mask"]
        action = ...  # your agent's action
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
```

**Loading rollout data:**

```python
data = RolloutRecorder.load("data/rollouts/rollout_20260418_143022_000000.npz")

data["obs"]         # shape (T+1, obs_dim), float32 — observations (includes terminal obs)
data["actions"]     # shape (T,), int32 — actions taken
data["rewards"]     # shape (T,), float32 — rewards received
data["terminated"]  # shape (T,), bool
data["truncated"]   # shape (T,), bool
data["phases"]      # shape (T,), uint8 — game phase (0=play, 1=shop, 2=game_over, 3=game_won)
data["antes"]       # shape (T,), uint8 — ante number at each step
data["scores"]      # shape (T,), int32 — cumulative score at each step
data["money"]       # shape (T,), int16 — money at each step
```

**With action masks** (for offline masked-action training):

```python
env = RolloutRecorder(env, save_dir="data/rollouts", save_action_mask=True)

# After loading:
data = RolloutRecorder.load("data/rollouts/rollout_20260418_143022_000000.npz")
data["action_masks"]  # shape (T, 446), bool — valid actions at each step
```

Action masks are bit-packed on disk for efficiency and automatically unpacked when loaded.

**File naming:** `rollout_{session_timestamp}_{episode_id:06d}.npz` — unique across sessions and episodes.

**File size:** A typical Easy episode (~100-300 steps) compresses to ~50-200 KB.

### EpisodeStatsRecorder — Game Summary Statistics

Records one row per completed episode to a Parquet file. Ideal for tracking win rates, comparing agents, and statistical analysis.

```python
import balatro_gym
import gymnasium as gym
from balatro_gym.wrappers import EpisodeStatsRecorder

env = gym.make("Balatro-Easy-v0")
env = EpisodeStatsRecorder(env, save_path="data/results/experiment_01.parquet")

for seed in range(1000):
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        action = ...
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

env.close()  # important: flushes remaining buffered rows
```

**Loading and analyzing results:**

```python
import pandas as pd

df = pd.read_parquet("data/results/experiment_01.parquet")

print(f"Win rate: {df['won'].mean():.1%}")
print(f"Avg antes beaten: {df['antes_beaten'].mean():.1f}")
print(f"Avg max score: {df['max_score'].mean():.0f}")
print(f"Avg final money: {df['final_money'].mean():.1f}")
print(f"Avg steps/episode: {df['total_steps'].mean():.0f}")
```

**Columns recorded:**

| Column | Type | Description |
|--------|------|-------------|
| `episode_id` | int32 | Sequential episode counter (0-based) |
| `timestamp` | float64 | Unix timestamp when episode ended |
| `seed` | int64 | Seed passed to `reset()`, -1 if None |
| `won` | bool | True if all antes were beaten |
| `antes_beaten` | int32 | Full antes beaten (blinds_beaten // 3) |
| `blinds_beaten` | int32 | Total blinds beaten |
| `total_steps` | int32 | Steps taken in the episode |
| `total_hands_played` | int32 | Poker hands played (not discards) |
| `total_reward` | float32 | Sum of rewards across the episode |
| `max_score` | int64 | Highest score achieved in any single blind |
| `final_money` | int32 | Money at episode end |
| `final_ante` | int32 | Ante number at episode end |
| `num_jokers_final` | int32 | Jokers held at episode end |
| `num_consumables_final` | int32 | Consumables held at episode end |
| `max_ante_reached` | int32 | Highest ante entered |
| `max_blinds_beaten` | int32 | Total blinds beaten |
| `play_actions` | int32 | Count of play actions |
| `discard_actions` | int32 | Count of discard actions |
| `buy_actions` | int32 | Count of buy actions |
| `sell_actions` | int32 | Count of sell actions |
| `reroll_actions` | int32 | Count of reroll actions |
| `skip_actions` | int32 | Count of skip actions |

**File size:** ~2-4 KB per 1000 episodes (Snappy-compressed Parquet).

### Using Both Wrappers Together

```python
env = gym.make("Balatro-Easy-v0")
env = RolloutRecorder(env, save_dir="data/rollouts")
env = EpisodeStatsRecorder(env, save_path="data/results/run_001.parquet")

# RolloutRecorder captures every transition to .npz
# EpisodeStatsRecorder captures game summaries to .parquet
```

### Comparing Agents

```python
import pandas as pd

df_random = pd.read_parquet("data/results/random_agent.parquet")
df_ppo = pd.read_parquet("data/results/ppo_agent.parquet")

comparison = pd.DataFrame({
    "Random": [df_random["won"].mean(), df_random["antes_beaten"].mean()],
    "PPO":    [df_ppo["won"].mean(),    df_ppo["antes_beaten"].mean()],
}, index=["Win Rate", "Avg Antes Beaten"])

print(comparison)
```

---

## Episode Seed IDs and State Resume

### Seed IDs

Every episode is assigned a unique, human-readable seed ID in the format:

```
YYYYMMDD-HHMM-XXXXXXXX
```

- **YYYYMMDD-HHMM** — UTC timestamp of episode creation (minute precision)
- **XXXXXXXX** — 8-character base-36 encoding of the game seed (0-9, A-Z)

This supports over 2.8 trillion unique seeds. The game seed fully determines the game (deck shuffle, shop offerings, boss blind selection), so sharing a seed ID lets another player start the exact same game — just like roguelite seeds.

```python
import balatro_gym
import gymnasium as gym

env = gym.make("Balatro-Easy-v0")
obs, info = env.reset(seed=42)

seed_id = info["episode_seed_id"]
print(seed_id)  # e.g., "20260418-1430-00000016"
```

**Replay a game from a seed ID:**

```python
from balatro_gym.envs.balatro_env import BalatroEnv
from balatro_gym.envs.configs import GameConfig

# Another player can start the same game:
env, obs, info = BalatroEnv.from_seed_id(
    "20260418-1430-00000016",
    config=GameConfig.easy(),
)
# obs is identical to the original game's initial observation
```

**Parse a seed ID:**

```python
from balatro_gym.core.seed_id import parse_seed_id, seed_id_to_game_seed

parsed = parse_seed_id("20260418-1430-00000016")
# {'timestamp': '20260418-1430', 'game_seed': 42, 'seed_str': '00000016'}

game_seed = seed_id_to_game_seed("20260418-1430-00000016")
# 42
```

### Save and Resume State

The environment supports full state checkpointing. You can save the complete game state at any point and resume from it later — including the RNG state, so the game continues deterministically.

**Save a checkpoint:**

```python
env = BalatroEnv(config=GameConfig.easy(seed=42))
obs, info = env.reset(seed=42)

# Play some steps...
for _ in range(20):
    mask = info["action_mask"]
    valid = np.where(mask)[0]
    obs, _, term, _, info = env.step(int(valid[0]))
    if term:
        break

# Save the full state
checkpoint = env.save_state()

# checkpoint is a JSON-compatible dict — save it however you like:
import json
with open("checkpoint.json", "w") as f:
    json.dump(checkpoint, f)
```

**Resume from a checkpoint:**

```python
import json

with open("checkpoint.json") as f:
    checkpoint = json.load(f)

env = BalatroEnv(config=GameConfig.easy())
env.reset(seed=0)  # dummy reset to initialize
obs, info = env.load_state(checkpoint)

# Continue playing from the exact saved point
mask = info["action_mask"]
# ... the game continues as if it was never interrupted
```

**Replay from a rollout file:**

Since rollout `.npz` files store the episode seed ID, you can replay any recorded game:

```python
from balatro_gym.wrappers import RolloutRecorder
from balatro_gym.envs.balatro_env import BalatroEnv, CARD_SUBSETS, PLAY_OFFSET, DISCARD_OFFSET
from balatro_gym.envs.configs import GameConfig

# Load a rollout
data = RolloutRecorder.load("data/rollouts/rollout_20260418_1430_000042.npz")
seed_id = data["episode_seed_id"]
actions = data["actions"]

# Recreate the game
env, obs, info = BalatroEnv.from_seed_id(seed_id, config=GameConfig.easy())

# Replay to step N, then continue with new actions
N = 50
for action in actions[:N]:
    obs, reward, term, trunc, info = env.step(int(action))

# Now at step N — save state and branch off
checkpoint = env.save_state()
```

---

## Game Mechanics Reference

### Poker Hands

Scored from lowest to highest. Each hand type has base chips and base multiplier that increase with hand levels (via Planet cards).

| Hand Type | Level 1 (chips x mult) | Planet Card |
|-----------|----------------------|-------------|
| High Card | 5 x 1 | Pluto |
| Pair | 10 x 2 | Mercury |
| Two Pair | 20 x 2 | Uranus |
| Three of a Kind | 30 x 3 | Earth |
| Straight | 30 x 4 | Saturn |
| Flush | 35 x 4 | Jupiter |
| Full House | 40 x 4 | Venus |
| Four of a Kind | 60 x 7 | Mars |
| Straight Flush | 100 x 8 | Neptune |
| Five of a Kind | 120 x 12 | Planet X |
| Flush House | 140 x 14 | Ceres |
| Flush Five | 160 x 16 | Eris |

### Scoring Pipeline

The scoring pipeline for a played hand follows this order:

1. **Hand-type base** chips and mult from hand levels
2. **Boss blind** modifiers (e.g., The Flint halves base)
3. **Per scoring card** (left to right):
   - Card chip value (rank nominal + enhancement bonus)
   - Enhancement mult (Mult Card +4, Lucky Card 1/5 chance +20)
   - Enhancement x_mult (Glass Card x2)
   - Edition bonus (Foil +50 chips, Holo +10 mult, Polychrome x1.5 mult)
   - Joker individual triggers
   - Red Seal retrigger (repeats all card effects once)
4. **Per held card** (cards in hand, not played):
   - Steel Card x1.5 mult
   - Edition bonuses
   - Joker held-card triggers
   - Red Seal retrigger
5. **Per joker** (left to right): main scoring effects
6. **Final score** = max(0, chips x mult)

### Card Enhancements

Applied to individual playing cards via Tarot cards.

| Enhancement | Effect |
|------------|--------|
| Bonus | +30 chips when scored |
| Mult | +4 mult when scored |
| Wild | Counts as every suit (enables flushes) |
| Glass | x2 mult when scored, 1/4 chance to shatter (destroyed) |
| Steel | x1.5 mult when held in hand (not played) |
| Stone | +50 chips, no rank (always +50, no nominal value) |
| Gold | +$3 when held at end of round |
| Lucky | 1/5 chance +20 mult, 1/15 chance +$20 |

### Card Editions

| Edition | Effect |
|---------|--------|
| Foil | +50 chips |
| Holo | +10 mult |
| Polychrome | x1.5 mult |

### Card Seals

| Seal | Effect |
|------|--------|
| Gold | +$3 when card is played |
| Red | Retrigger: card scores twice |
| Blue | Create a Planet card when round ends (if consumable slot available) |
| Purple | Create a Tarot card when discarded (if consumable slot available) |

### Jokers (30 total)

Jokers are permanent modifiers bought from the shop. They trigger during scoring to add chips, mult, or x_mult. Player can hold up to 5 jokers (order matters for some effects).

### Consumables (44 total)

Single-use cards that modify the game state:

- **Planet Cards (12):** Level up a specific hand type (+chips, +mult per level)
- **Tarot Cards (22):** Apply enhancements, change suits, modify cards, generate money
- **Spectral Cards (10):** Powerful effects like duplicating cards, applying seals, destroying cards for money

### Economy

- Earn $3-5 for beating a blind (varies by blind type)
- Interest: $1 per $5 held, up to $5 max interest per round
- Sell jokers for half their purchase price
- Gold Card: $3 per card held at round end
- Gold Seal: $3 when played

### Boss Blinds

Each ante's third blind is a "boss" with a debuff:

| Boss | Effect |
|------|--------|
| The Hook | Discards 2 random cards from hand each play |
| The Wall | Score target x2 |
| The Wheel | 1/7 chance each card is face-down |
| The Arm | Reduces level of played hand type by 1 |
| The Flint | Halves base chips and mult |
| The Mark | All face cards are face-down |

---

## API Reference

### Environment Constants

```python
from balatro_gym.envs.balatro_env import (
    TOTAL_ACTIONS,       # 446
    CARD_SUBSETS,        # list of 218 tuples — index-to-card-indices mapping
    PLAY_OFFSET,         # 0
    DISCARD_OFFSET,      # 218
    BUY_OFFSET,          # 436
    SELL_OFFSET,         # 439
    REROLL_ACTION,       # 444
    SKIP_ACTION,         # 445
    CARD_FEATURE_DIM,    # 68 (per-card observation features)
    MAX_HAND_SIZE,       # 8
)
```

### Info Dict

The `info` dict returned by `reset()` and `step()` contains:

| Key | Type | Description |
|-----|------|-------------|
| `action_mask` | np.ndarray (bool) | Boolean mask over 446 actions |
| `phase` | str | Current phase: "play", "shop", "game_over", "game_won" |
| `ante` | int | Current ante number |
| `blind_index` | int | Current blind (0=small, 1=big, 2=boss) |
| `score` | int | Current score this blind |
| `score_target` | int | Required score to beat this blind |
| `money` | int | Current money |
| `hands_remaining` | int | Hands left this blind |
| `discards_remaining` | int | Discards left this blind |
| `blinds_beaten` | int | Total blinds beaten so far |
| `num_jokers` | int | Number of jokers held |
| `num_consumables` | int | Number of consumables held |

### Reproducibility

Seeding is fully deterministic:

```python
# Method 1: Seed via reset
env = gym.make("Balatro-Easy-v0")
obs1, _ = env.reset(seed=42)
obs2, _ = env.reset(seed=42)
assert np.array_equal(obs1, obs2)  # True

# Method 2: Seed via config
config = GameConfig.easy(seed=42)
env = BalatroEnv(config=config)
obs, _ = env.reset()
```

### Available Joker IDs

```
joker_basic, greedy_joker, lusty_joker, wrathful_joker, gluttonous_joker,
jolly_joker, zany_joker, banner, mystic_summit, ice_cream,
raised_fist, fibonacci, even_steven, odd_todd, scholar,
business_card, stencil, half_joker, blueprint, dna,
abstract_joker, blackboard, the_duo, the_trio, the_family,
loyalty_card, ceremonial_dagger, ride_the_bus, runner, supernova
```

### Available Consumable IDs

**Planet Cards:**
```
c_pluto, c_mercury, c_uranus, c_earth, c_saturn,
c_jupiter, c_venus, c_mars, c_neptune, c_planet_x, c_ceres, c_eris
```

**Tarot Cards:**
```
c_magician, c_empress, c_hierophant, c_lovers, c_chariot, c_justice,
c_devil, c_tower, c_star, c_moon, c_sun, c_world, c_strength,
c_hanged_man, c_death, c_hermit, c_temperance, c_fool,
c_high_priestess, c_emperor, c_judgement, c_wheel_of_fortune
```

**Spectral Cards:**
```
c_familiar, c_grim, c_incantation, c_talisman, c_deja_vu,
c_trance, c_medium, c_aura, c_cryptid, c_immolate
```

---

## Tips for Training

1. **Start with Easy.** The easy preset has only 4 antes, more hands/discards, and a free starter joker. This makes it feasible for agents to win during early training.

2. **Use action masking.** The action space is large (446) but most actions are invalid at any given state. The `ActionMaskingTorchRLModule` in `balatro_gym.rllib` handles this natively with RLlib PPO.

3. **Observation varies by config.** The observation dimension changes with the joker/consumable pool size. If you plan curriculum training across difficulties, consider using a fixed pool size for all stages.

4. **Monitor these metrics:**
   - Win rate (most important)
   - Average ante reached (more granular than win/loss)
   - Average blinds beaten per game
   - Average money at game end (proxy for economic decisions)

5. **Reward shaping matters.** The default reward includes score-progress bonuses and per-blind rewards. For ablation studies, you can modify reward shaping in `BalatroEnv.step()`.

6. **Seed for reproducibility.** Always pass `seed=` to `env.reset()` when evaluating or comparing agents.
