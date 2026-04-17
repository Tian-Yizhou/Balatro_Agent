# Balatro-Agent Implementation Plan

## Context

Building a Gymnasium-compatible card game environment inspired by Balatro for COMP_SCI 496 (Agent AI, Northwestern). Two deliverables: (1) a reasoning gym environment, (2) an RL-trained agent using PPO to play the game with structured (numerical) observations.

**Approach**: Reference-guided clean-room. Read the Balatro Lua source to understand exact formulas, scoring rules, and joker effects, then implement from scratch in Python with original code structure. The Lua source is never committed to the repo.

**Timeline**: Proposal presentation Apr 20 (slides only) | Final presentation Jun 1 | Report due Jun 8

---

## Architecture

```
Balatro-Agent/
├── balatro_gym/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── card.py                 # Card, Deck classes
│   │   ├── hand_evaluator.py       # Poker hand detection + base scoring
│   │   ├── joker.py                # Joker registry, base class, all joker definitions
│   │   ├── blind.py                # Blind progression, boss blind effects
│   │   ├── shop.py                 # Shop: offerings, buying, selling, rerolling
│   │   └── game_state.py           # Full game state manager + scoring pipeline
│   ├── envs/
│   │   ├── __init__.py             # Gymnasium registration
│   │   ├── balatro_env.py          # Main Gymnasium environment
│   │   └── configs.py              # GameConfig dataclass + difficulty presets
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── random_agent.py         # Baseline: random valid actions
│   │   └── heuristic_agent.py      # Baseline: greedy best-hand strategy
│   ├── rendering/
│   │   ├── __init__.py
│   │   └── text_renderer.py        # Human-readable game state (for debugging)
│   └── utils/
│       ├── __init__.py
│       └── metrics.py              # Win rate, avg score, tracking
├── experiments/
│   ├── train_ppo.py                # PPO training script
│   ├── run_baselines.py            # Run random/heuristic baselines
│   └── evaluation.py               # Evaluation + comparison suite
├── configs/
│   ├── easy.yaml
│   ├── medium.yaml
│   └── hard.yaml
├── tests/
│   ├── test_card.py
│   ├── test_hand_evaluator.py
│   ├── test_joker.py
│   ├── test_blind.py
│   ├── test_shop.py
│   ├── test_game_state.py
│   └── test_env.py
├── setup.py
├── requirements.txt
└── README.md
```

---

## Phase 1: Core Game Engine (DONE)

All core modules are implemented. See `docs/Tech_Log.md` for code details.

| Module | File | Status |
|--------|------|--------|
| Card, Deck | `balatro_gym/core/card.py` | Done |
| Hand evaluator | `balatro_gym/core/hand_evaluator.py` | Done |
| Joker registry + 30 jokers | `balatro_gym/core/joker.py` | Done |
| Blinds + boss effects | `balatro_gym/core/blind.py` | Done |
| Shop | `balatro_gym/core/shop.py` | Done |
| Game state manager | `balatro_gym/core/game_state.py` | Done |

---

## Phase 2: Gymnasium Environment

### Step 1 — `balatro_gym/envs/configs.py`

GameConfig dataclass holding all game parameters. Provides `easy()`, `medium()`, `hard()` presets and `from_file()` for YAML loading. Validates joker IDs against the registry.

```python
@dataclass
class GameConfig:
    num_antes: int = 8
    hands_per_round: int = 4
    discards_per_round: int = 3
    hand_size: int = 8
    max_jokers: int = 5
    starting_money: int = 4
    shop_slots: int = 2
    reroll_base_cost: int = 5
    reward_mode: str = "shaped"
    available_joker_ids: list[str] = field(default_factory=list)
    starting_joker_ids: list[str] = field(default_factory=list)
```

### Step 2 — `balatro_gym/envs/balatro_env.py`

The Gymnasium wrapper. This is the most complex new code because it must encode the game state and action space for a neural network.

#### Observation Space

A flat `Box` vector that a standard MLP policy can consume. All values normalized to roughly [0, 1] or [-1, 1].

```
Observation vector layout (total ~240 dimensions):

1. Hand cards: 52-dim binary vector (1 = card is in hand)
   - Index = suit * 13 + (rank - 2)
   - Example: K♠ = 3*13 + 11 = index 50

2. Hand card face-down flags: 52-dim binary (1 = face-down, applies to boss debuffs)

3. Joker slots: 5 slots × (num_joker_types + 1) one-hot
   - With 30 joker types: 5 × 31 = 155 dims
   - Slot is all-zeros if empty, one-hot for joker type otherwise
   - ORDER IS PRESERVED (slot 0 = leftmost joker, matters for scoring)

4. Game scalars (normalized):
   - money / 100                    (float, rough max ~$100)
   - ante / num_antes               (float, 0 to 1)
   - blind_type: 3-dim one-hot      (small, big, boss)
   - score_target (log-normalized):  log(score_target) / log(100000)
   - current_score / score_target    (float, 0 to ~2)
   - hands_remaining / hands_per_round
   - discards_remaining / discards_per_round
   - deck_size / 52

5. Phase: 2-dim one-hot (play, shop)

6. Shop offerings (during shop phase, zeros during play):
   - 2 slots × (num_joker_types + 1) one-hot + cost_normalized + sold_flag
   - 2 × (31 + 1 + 1) = 66 dims
```

Implementation:

```python
self.observation_space = gymnasium.spaces.Box(
    low=0.0, high=1.0,
    shape=(obs_dim,),
    dtype=np.float32,
)
```

The `_get_obs()` method builds this flat vector from `game_state.get_view()`.

#### Action Space

A single `Discrete` space enumerating ALL possible actions across both phases. Invalid actions are masked.

**Action enumeration:**

```
PLAY PHASE:
  Actions 0 to 217:    Play a subset of cards (all C(8,1)+C(8,2)+C(8,3)+C(8,4)+C(8,5) = 218 subsets)
  Actions 218 to 435:  Discard a subset of cards (same 218 subsets, offset by 218)

SHOP PHASE:
  Action 436: Buy shop slot 0
  Action 437: Buy shop slot 1
  Actions 438-442: Sell joker from slot 0-4
  Action 443: Reroll shop
  Action 444: Skip (leave shop, go to next blind)

TOTAL: 445 discrete actions
```

Pre-compute the card subset mapping at init time:

```python
# Build lookup: action_index -> list of card indices
from itertools import combinations

self._card_subsets: list[tuple[int, ...]] = []
for size in range(1, 6):          # 1 to 5 cards
    for combo in combinations(range(8), size):
        self._card_subsets.append(combo)
# len(self._card_subsets) == 218

self.action_space = gymnasium.spaces.Discrete(445)
```

**Action mask function** — returns a boolean array of length 445:

```python
def action_masks(self) -> np.ndarray:
    """Return valid action mask for current state. Required by MaskablePPO."""
    mask = np.zeros(445, dtype=bool)

    if self.game.phase == GamePhase.PLAY:
        if self.game.hands_remaining > 0:
            # All play subsets valid (actions 0-217) as long as we have enough cards
            for i, subset in enumerate(self._card_subsets):
                if max(subset) < len(self.game.hand):
                    mask[i] = True

        if self.game.discards_remaining > 0:
            # All discard subsets valid (actions 218-435)
            for i, subset in enumerate(self._card_subsets):
                if max(subset) < len(self.game.hand):
                    mask[218 + i] = True

    elif self.game.phase == GamePhase.SHOP:
        # Buy actions (436-437): valid if can afford and have joker slots
        for slot_idx, offering in enumerate(self.game.shop.offerings):
            if (not offering.sold
                and self.game.money >= offering.cost
                and len(self.game.jokers) < self.game.max_jokers):
                mask[436 + slot_idx] = True

        # Sell actions (438-442): valid if joker exists in that slot
        for j_idx in range(len(self.game.jokers)):
            mask[438 + j_idx] = True

        # Reroll (443): valid if can afford
        if self.game.money >= self.game.shop.reroll_cost:
            mask[443] = True

        # Skip (444): always valid in shop
        mask[444] = True

    return mask
```

#### Reward Function

```python
def _compute_reward(self) -> float:
    if self.game.phase == GamePhase.GAME_WON:
        return 10.0
    elif self.game.phase == GamePhase.GAME_OVER:
        return -1.0
    elif blind_just_beaten:
        # Shaped: reward proportional to progress
        progress = self.game.blinds_beaten / self.game.blind_manager.total_blinds
        efficiency = hands_saved / self.game.hands_per_round
        return 1.0 + progress + 0.5 * efficiency
    else:
        # Mid-blind: small reward for scoring (encourages learning to score)
        score_ratio = min(1.0, self.game.current_score / self.game.score_target)
        return 0.01 * score_ratio
```

Design rationale:
- **Win (+10)**: Large positive signal for completing the game
- **Lose (-1)**: Negative but not too punishing (we want the agent to explore, not become overly conservative)
- **Blind beaten (+1 to +2.5)**: Scaled by how far through the game the agent is. Beating late blinds is worth more.
- **Mid-step (0 to 0.01)**: Tiny reward for making score progress. Prevents the agent from learning to just skip/do nothing.

#### Step function dispatch

```python
def step(self, action: int):
    prev_score = self.game.current_score
    prev_blinds = self.game.blinds_beaten
    reward = 0.0

    if self.game.phase == GamePhase.PLAY:
        if action < 218:
            # Play cards
            card_indices = list(self._card_subsets[action])
            self.game.play_hand(card_indices)
        elif action < 436:
            # Discard cards
            card_indices = list(self._card_subsets[action - 218])
            self.game.discard(card_indices)

    elif self.game.phase == GamePhase.SHOP:
        if action == 436 or action == 437:
            self.game.shop_buy(action - 436)
        elif 438 <= action <= 442:
            self.game.shop_sell(action - 438)
        elif action == 443:
            self.game.shop_reroll()
        elif action == 444:
            self.game.shop_skip()

    reward = self._compute_reward(prev_score, prev_blinds)
    terminated = self.game.phase in (GamePhase.GAME_OVER, GamePhase.GAME_WON)
    obs = self._get_obs()
    info = {"action_mask": self.action_masks(), ...}

    return obs, reward, terminated, False, info
```

#### Tests (`tests/test_env.py`)

- `env.reset()` returns observation of correct shape
- `env.step()` with masked valid action → no error
- `env.action_masks()` returns correct shape, at least one True
- Random agent (respecting masks) plays 100 full games without crashing
- Seeded env → identical observation sequences
- Reward values are in expected ranges

---

## Phase 3: Agents

### Step 3 — `balatro_gym/agents/random_agent.py`

Selects a random action from the valid action mask each step. Lower bound baseline.

```python
def act(obs, action_mask):
    valid = np.where(action_mask)[0]
    return np.random.choice(valid)
```

### Step 4 — `balatro_gym/agents/heuristic_agent.py`

Rule-based agent that sets the upper bound for non-learned play.

**Play phase strategy:**
1. Evaluate all 218 possible card subsets (1-5 cards), compute the score each would produce (using `evaluate_hand` + joker scoring simulation).
2. Play the highest-scoring subset.
3. If best hand scores poorly and discards remain, discard the cards not part of any promising partial hand.

**Shop phase strategy:**
1. Buy the cheapest affordable joker (if have slots).
2. Never reroll.
3. Skip after buying (or if nothing affordable).

```python
def act(obs, game_state_view, action_mask):
    if phase == "play":
        best_action = -1
        best_score = -1
        for i in range(218):
            if action_mask[i]:
                subset = card_subsets[i]
                score = simulate_score(hand, subset, jokers)
                if score > best_score:
                    best_score = score
                    best_action = i
        return best_action
    elif phase == "shop":
        # Buy cheapest, else skip
        ...
```

### Step 5 — `balatro_gym/rendering/text_renderer.py`

Converts GameStateView to a human-readable string for debugging and visualization. Not used by the RL agent — purely for development and manual inspection.

---

## Phase 4: RL Training Pipeline

### Step 6 — `experiments/train_ppo.py`

PPO training using Stable Baselines3 with MaskablePPO from `sb3-contrib`.

**Setup:**

```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env):
    return env.action_masks()

env = ActionMasker(BalatroEnv(config), mask_fn)

model = MaskablePPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.01,           # Encourage exploration
    tensorboard_log="./logs/",
    policy_kwargs={
        "net_arch": [256, 256],  # Two hidden layers
    },
)

model.learn(total_timesteps=1_000_000)
model.save("ppo_balatro")
```

**Key hyperparameters to tune:**
- `learning_rate`: Start with 3e-4, decay if unstable
- `n_steps`: Rollout buffer size. 2048 is standard. May need larger if episodes are long.
- `gamma`: Discount factor. 0.99 is standard. Could try 0.995 for long-horizon play.
- `ent_coef`: Entropy bonus. 0.01 to encourage exploration. Increase to 0.05 if agent converges to suboptimal policy too fast.
- `net_arch`: [256, 256] for MLP. Could try [512, 256] if underfitting.

**Curriculum training:**

```python
# Stage 1: Easy (fewer antes, more hands)
easy_config = GameConfig.easy()
env = ActionMasker(BalatroEnv(easy_config), mask_fn)
model = MaskablePPO("MlpPolicy", env, ...)
model.learn(total_timesteps=500_000)

# Stage 2: Medium
medium_config = GameConfig.medium()
env = ActionMasker(BalatroEnv(medium_config), mask_fn)
model.set_env(env)
model.learn(total_timesteps=500_000)

# Stage 3: Hard
hard_config = GameConfig.hard()
env = ActionMasker(BalatroEnv(hard_config), mask_fn)
model.set_env(env)
model.learn(total_timesteps=1_000_000)
```

**Monitoring (via TensorBoard or wandb):**
- Episode reward (should trend upward)
- Episode length (longer = surviving more blinds)
- Win rate (rolling average over last 100 episodes)
- Average ante reached

### Step 7 — `experiments/run_baselines.py`

Run random and heuristic agents for 1000+ games, record:
- Win rate
- Average ante reached
- Average total score
- Average money at game end

### Step 8 — `experiments/evaluation.py`

Compare all agents (random, heuristic, PPO, PPO-curriculum):
- Win rate by difficulty (easy/medium/hard)
- Average ante reached
- Learning curves (timesteps vs. win rate)
- Per-blind survival rate (what percentage of agents beat blind X?)

---

## Phase 5: Experiments and Ablations

1. **Baseline comparison**: Random vs. Heuristic vs. PPO vs. PPO-curriculum
2. **Reward shaping ablation**: Sparse reward (win/lose only) vs. shaped reward. Does shaping help?
3. **Curriculum learning**: Train on easy→medium→hard vs. train on hard directly. Does curriculum help?
4. **Observation ablation**: Does including shop information help? Does joker encoding matter?
5. **Network architecture**: MLP [256,256] vs. [512,256] vs. [128,128,128]

---

## Key Design Principles

### 1. GameState separate from BalatroEnv
Core game logic lives in `GameState` which knows nothing about Gymnasium. The env is a thin wrapper. This means you can test game logic without Gymnasium and use GameState directly for fast rollouts.

### 2. Joker scoring returns ScoreModification, not raw mutation
Each joker returns a `ScoreModification` object. This makes the scoring pipeline debuggable — you can log exactly what each joker contributed.

### 3. GameStateView is read-only
Jokers receive a read-only view. Prevents accidental mutation. Defined as a Protocol in `joker.py` to avoid circular imports.

### 4. Left-to-right joker order matters
In Balatro, the order of jokers changes the outcome. A +4 mult joker before a x2 mult joker gives different results than reversed. The scoring pipeline respects `self.jokers` list order. The observation preserves slot ordering.

### 5. Config-selected joker pools
All jokers are code-defined and registered. The config specifies which IDs are available per experiment.

### 6. Single Discrete action space with masking
All actions (play, discard, buy, sell, reroll, skip) are enumerated in one Discrete(445) space. Action masks handle phase-dependent validity. This is the cleanest approach for MaskablePPO.

---

## Potential Challenges

| Challenge | Mitigation |
|-----------|------------|
| Scoring pipeline correctness | Parameterized tests; cross-reference Lua source |
| Large action space (445 actions) | Action masking reduces effective space to ~20-50 per step |
| Sparse rewards (long episodes) | Shaped reward: partial credit for score progress and blind completion |
| Observation encoding quality | Ablation study on different encodings; ensure all game-relevant info is included |
| Curriculum learning transitions | Warm-start from previous stage; monitor for performance dips |
| Joker ordering in observation | Preserve slot order in one-hot encoding; test that agent learns order-dependent strategies |

---

## Verification Checkpoints

1. **After Phase 1**: `pytest tests/` passes. Can run a full game via `GameState` API. **(DONE)**
2. **After Phase 2**: `gymnasium.make("Balatro-v0")` works. `env.step()` accepts masked Discrete actions. Observation shape is correct.
3. **After Phase 3**: Random and heuristic agents play 1000 games without crashes. Heuristic win rate > random.
4. **After Phase 4**: PPO training runs without errors. Training curve shows improvement. PPO win rate > random (at minimum).
5. **After Phase 5**: All experiments complete. Comparison figures ready for presentation/report.
