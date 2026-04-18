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
- [Training with PPO (sb3-contrib)](#training-with-ppo-sb3-contrib)
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

# For RL training
pip install stable-baselines3 sb3-contrib torch
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
# (required by sb3-contrib MaskablePPO)
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

## Training with PPO (sb3-contrib)

The environment is designed for `MaskablePPO` from `sb3-contrib`, which handles the discrete action space with action masking.

### Basic Training

```python
import balatro_gym
import gymnasium as gym
from sb3_contrib import MaskablePPO

env = gym.make("Balatro-Easy-v0")

model = MaskablePPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    tensorboard_log="./logs/",
)

model.learn(total_timesteps=1_000_000)
model.save("balatro_ppo_easy")
```

### Evaluation

```python
from sb3_contrib.common.maskable.utils import get_action_masks

model = MaskablePPO.load("balatro_ppo_easy")
env = gym.make("Balatro-Easy-v0")

wins = 0
num_episodes = 100

for _ in range(num_episodes):
    obs, info = env.reset()
    terminated = False
    while not terminated:
        action_masks = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

    if info["phase"] == "game_won":
        wins += 1

print(f"Win rate: {wins}/{num_episodes} = {wins/num_episodes:.1%}")
```

### Curriculum Training

Train on easy first, then fine-tune on harder difficulties:

```python
# Phase 1: Easy
env_easy = gym.make("Balatro-Easy-v0")
model = MaskablePPO("MlpPolicy", env_easy, verbose=1)
model.learn(total_timesteps=500_000)

# Phase 2: Medium
env_medium = gym.make("Balatro-Medium-v0")
model.set_env(env_medium)
model.learn(total_timesteps=500_000)

# Phase 3: Hard
env_hard = gym.make("Balatro-Hard-v0")
model.set_env(env_hard)
model.learn(total_timesteps=500_000)
```

> **Note:** When switching environments with `set_env()`, the observation dimension changes across difficulty levels (756 for Easy, 868 for Medium, 950 for Hard) because the joker/consumable pool sizes differ. You may need to create a new model for each difficulty or use a fixed-size config across all stages.

### Vectorized Training

```python
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("Balatro-Easy-v0", n_envs=8)
model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2_000_000)
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

2. **Use action masking.** The action space is large (446) but most actions are invalid at any given state. MaskablePPO from sb3-contrib handles this natively.

3. **Observation varies by config.** The observation dimension changes with the joker/consumable pool size. If you plan curriculum training across difficulties, consider using a fixed pool size for all stages.

4. **Monitor these metrics:**
   - Win rate (most important)
   - Average ante reached (more granular than win/loss)
   - Average blinds beaten per game
   - Average money at game end (proxy for economic decisions)

5. **Reward shaping matters.** The default reward includes score-progress bonuses and per-blind rewards. For ablation studies, you can modify reward shaping in `BalatroEnv.step()`.

6. **Seed for reproducibility.** Always pass `seed=` to `env.reset()` when evaluating or comparing agents.
