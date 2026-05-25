# Balatro-Agent: Project Design Map

## 1. Vision

A Gymnasium-compatible reasoning gym for strategic decision-making research, paired with multiple agent paradigms (RL, LLM, hybrid) that serve as reference implementations and baselines. The environment tests agents on: probability estimation, expected value calculation, combinatorial optimization, resource management, and long-horizon planning under uncertainty.

Analogous to how Atari provides arcade-game benchmarks for vision-based RL, and Voyager provides open-ended exploration benchmarks for LLM agents — Balatro-Gym provides a **deck-building strategy benchmark** that requires both mathematical reasoning and multi-step planning.

---

## 2. Environment Design

### 2.1 Core Game Engine (`balatro_gym/core/`)

A framework-independent game engine implementing Balatro's mechanics:

| Module | Responsibility |
|--------|---------------|
| `card.py` | Card/Deck primitives, 8 enhancements, 3 editions, 4 seals |
| `hand_evaluator.py` | 12 poker hand types, base scoring |
| `hand_levels.py` | Mutable hand-type levels (Planet card upgrades) |
| `joker.py` | 30 jokers via registry pattern, 8 scoring hooks per joker |
| `consumable.py` | 40 consumables (22 Tarots, 12 Planets, 6 Spectrals) |
| `blind.py` | Blind progression, 9 boss blind debuff effects |
| `shop.py` | Shop offering generation, buy/sell/reroll |
| `game_state.py` | Full lifecycle + 10-step scoring pipeline |
| `seed_id.py` | Reproducible episode seeds (YYYYMMDD-HHMM-XXXXXXXX) |

**Status**: Complete. 383 tests passing.

### 2.2 Gymnasium Interface (`balatro_gym/envs/`)

| Component | Specification |
|-----------|--------------|
| Observation | Flat float32 vector (Easy=756, Medium=868, Hard=950 dims) |
| Action Space | Discrete(446) with boolean action mask |
| Actions 0-217 | Play card subsets (all C(8,1..5) = 218 combinations) |
| Actions 218-435 | Discard card subsets (same 218 combinations) |
| Actions 436-438 | Buy shop slots 0-2 |
| Actions 439-443 | Sell joker slots 0-4 |
| Action 444 | Reroll shop |
| Action 445 | Skip shop |
| Registered IDs | `Balatro-v0`, `Balatro-Easy-v0`, `Balatro-Medium-v0`, `Balatro-Hard-v0` |

**Factory API** (MineDojo-style):
```python
env = balatro_gym.make("easy", seed=42, reward_fn=DefaultReward())
vec_env = balatro_gym.make_vec("easy", num_envs=8)
```

**Status**: Complete.

### 2.3 Pluggable Reward System (`balatro_gym/envs/rewards.py`)

| Reward Function | Design |
|----------------|--------|
| `DefaultReward` | Shaped: blind_beaten bonus (+1 scaled by progress), score progress credit, win (+10), lose (-1), invalid action penalty (-0.01) |
| `SparseReward` | Terminal only: win (+1), lose (-1) |
| Custom | Any callable satisfying `RewardFunction` Protocol (`RewardContext -> float`) |

`RewardContext` is a frozen dataclass exposing: `game_won`, `game_over`, `blind_beaten`, `score_progress`, `invalid_action`, `blinds_beaten`, `total_blinds`.

**Status**: Complete.

### 2.4 Difficulty Configuration (`balatro_gym/envs/configs.py`)

| Parameter | Easy | Medium | Hard |
|-----------|------|--------|------|
| Antes | 4 | 6 | 8 |
| Hands/round | 5 | 4 | 4 |
| Discards/round | 4 | 3 | 3 |
| Starting money | $6 | $4 | $4 |
| Joker pool | 10 | 20 | 30 |
| Consumable pool | Planets + simple Tarots | + all Tarots + simple Spectrals | All 40 |

Custom configs via YAML merge-over-defaults:
```yaml
base: easy
num_antes: 5
starting_money: 8
```

**Status**: Complete.

### 2.5 Recording Infrastructure (`balatro_gym/wrappers/`)

| Wrapper | Output | Contents |
|---------|--------|----------|
| `RolloutRecorder` | `.npz` per episode | obs, actions, rewards, terminated, truncated, phases, antes, scores, money, [action_masks] |
| `EpisodeStatsRecorder` | Parquet (streaming append) | 23 columns: won, blinds_beaten, total_steps, reward, scores, action counts, etc. |

**Status**: Complete.

---

## 3. Agent Paradigms

### 3.1 Baseline Agents (`agent/baselines/`)

| Agent | Strategy | Purpose |
|-------|----------|---------|
| `RandomAgent` | Uniform sample from valid actions | Lower bound |
| `HeuristicAgent` | Greedy: play highest-scoring hand, discard weak unpaired cards, buy cheapest joker | Upper bound for non-learned play |

All agents satisfy the `Agent` Protocol: `act(obs, info) -> int`, `reset() -> None`.

**Verified baselines (easy mode)**:
- Random: ~1.4 blinds beaten, 0% win rate
- Heuristic: ~9.3 blinds beaten, high win rate

**Status**: Complete.

### 3.2 RL Agent — PPO with Action Masking (`agent/rllib/`)

**Architecture**:
- Framework: Ray RLlib (new API stack)
- Algorithm: PPO with `ActionMaskingTorchRLModule`
- Policy network: MLP (configurable hidden layers, default [256, 256])
- Observation: Dict(`observations`: flat vector, `action_mask`: bool[446])
- Training: distributed env_runners for rollout collection, GPU learner

**Training CLI**:
```bash
python -m agent.rllib.train \
    --difficulty easy \
    --num-env-runners 4 \
    --num-gpus-per-learner 1 \
    --num-iterations 200 \
    --fcnet-hiddens 256 256
```

**Metrics** (`BalatroMetricsCallback`): win_rate, blinds_beaten, ante_reached, final_money, episode_length.

**Status**: Infrastructure complete. Training runs needed.

### 3.3 LLM Agent (`agent/llm/`)

**Architecture**:
- Input: Full game state rendered as structured JSON via `GameStateRenderer`
- Model: Any model satisfying `ModelBackend` Protocol (`generate(prompt) -> str`)
- Output: Parsed `action_id` from model response

**Game State Renderer** produces JSON with sections:
- `game_progress` — phase, ante, blind, score target, hands/discards remaining
- `hand` — cards with rank, suit, chip_value, enhancements, editions, seals
- `jokers` — owned jokers with descriptions
- `consumables` — owned consumables with descriptions
- `economy` — money, deck size
- `hand_levels` — current level/chips/mult for all 12 hand types
- `valid_actions` — enumerated with type, card_indices, human-readable descriptions
- `shop` — offerings with cost, name, description (if in shop phase)

**Planned backends**:
| Backend | Model | Use Case |
|---------|-------|----------|
| `HuggingFaceBackend` | Qwen2.5-3B/7B-Instruct | Local inference, fine-tunable |
| `OpenAIBackend` | GPT-4o | API-based, strong zero-shot |
| `AnthropicBackend` | Claude | API-based, strong reasoning |

**Status**: Renderer complete, backends are Protocol stubs (TODO: implement).

### 3.4 Hybrid Agent (Future)

Combine RL policy with LLM reasoning:
- LLM provides high-level strategy (which hand type to aim for, when to save money)
- RL policy executes low-level card selection within that strategy
- Or: LLM acts as a reward model / critic for RL training

**Status**: Not started. Future extension.

---

## 4. Experiment Design

### 4.1 PPO Training Experiments

| Experiment | Variable | Control | Metric |
|-----------|----------|---------|--------|
| E1: Basic PPO | Train PPO on easy, 200 iter | — | Win rate, learning curve |
| E2: Reward ablation | DefaultReward vs SparseReward | Same architecture, same seed | Learning speed, final win rate |
| E3: Curriculum | Easy→Medium vs Medium-only | Same total timesteps | Win rate on medium |
| E4: Architecture | [256,256] vs [512,256] vs [128,128,128] | Same difficulty, reward | Final performance |
| E5: Difficulty scaling | Train on easy / medium / hard | Same architecture | Win rate per difficulty |

### 4.2 LLM Agent Experiments

| Experiment | Variable | Metric |
|-----------|----------|--------|
| L1: Zero-shot | Qwen-3B / Qwen-7B on easy | Win rate, blinds beaten, action quality |
| L2: Prompt engineering | Minimal prompt vs strategy-augmented prompt | Win rate delta |
| L3: Few-shot | Include 3 expert trajectory snippets in prompt | Win rate vs zero-shot |
| L4: Fine-tuning (SFT) | Fine-tune on heuristic agent trajectories | Win rate before/after |

### 4.3 Cross-Paradigm Comparison

| Agent | Easy | Medium | Hard |
|-------|------|--------|------|
| Random | baseline | baseline | baseline |
| Heuristic | baseline | baseline | baseline |
| PPO (shaped) | E1 | E5 | E5 |
| PPO (curriculum) | — | E3 | — |
| LLM (zero-shot) | L1 | — | — |
| LLM (fine-tuned) | L4 | — | — |

---

## 5. Evaluation Protocol

### 5.1 Metrics

| Metric | Definition | Granularity |
|--------|-----------|-------------|
| Win rate | % of episodes where agent wins | Per 100 episodes |
| Blinds beaten | Mean blinds beaten per episode | Per episode |
| Ante reached | Mean maximum ante reached | Per episode |
| Survival curve | P(agent beats blind N) for each N | Per blind index |
| Score efficiency | Mean(score / target) per blind | Per blind |
| Episode length | Mean steps per episode | Per episode |
| Action distribution | % play / discard / buy / sell / reroll / skip | Per episode |

### 5.2 Evaluation Procedure

1. Fix 100 evaluation seeds (shared across all agents for fair comparison)
2. Run each agent on all 100 seeds per difficulty level
3. Report mean ± std for all metrics
4. Use `evaluate_agent()` from `agent/base.py` for consistent execution

### 5.3 Recording

All evaluation runs are recorded via `EpisodeStatsRecorder` to Parquet files for post-hoc analysis. Training runs save per-iteration metrics to `metrics.json`.

---

## 6. Implementation Roadmap

### Phase A: RL Training (Current Priority)

```
A1. Run PPO on easy mode (200 iterations, DefaultReward)
    → Verify learning signal exists (reward curve trends upward)
    
A2. Run PPO on easy mode (200 iterations, SparseReward)
    → Compare learning speed with A1

A3. Run curriculum: easy (100 iter) → medium (100 iter)
    → Compare medium-mode performance vs training on medium directly

A4. Run architecture variants on easy mode
    → Identify best architecture for downstream experiments

A5. Evaluate best PPO checkpoint on all difficulties
    → Produce cross-difficulty comparison table
```

### Phase B: LLM Agent

```
B1. Implement HuggingFaceBackend
    → Load Qwen2.5-3B-Instruct, wrap generate() with prompt template

B2. Design system prompt
    → Game rules summary + valid action format specification

B3. Run LLM zero-shot on easy (50 episodes)
    → Measure baseline LLM performance

B4. Iterate on prompt (strategy hints, few-shot examples)
    → Measure improvement from prompt engineering

B5. (Optional) Generate expert trajectories with HeuristicAgent
    → SFT fine-tune Qwen on (state_json, action) pairs
    → Evaluate fine-tuned model
```

### Phase C: Evaluation & Analysis

```
C1. Run all agents on standardized 100-seed evaluation set
    → Easy, medium, hard

C2. Generate comparison tables and figures
    → Win rate, survival curves, learning curves

C3. Analyze failure modes
    → Where does each agent type fail? (which blind, which decisions)

C4. Record full trajectory dataset
    → For future research (imitation learning, offline RL)
```

---

## 7. Compute Plan

| Task | Hardware | Time |
|------|----------|------|
| PPO experiments (5 runs × ~200 iter) | 1× GPU + 4-8 CPU workers | ~10h total |
| LLM inference (Qwen-3B, 50 episodes) | 1× GPU | ~1-2h |
| LLM fine-tuning (optional, SFT) | 1× GPU | ~2-4h |
| Evaluation runs (all agents × 3 difficulties) | CPU | <1h |
| Trajectory recording | CPU | <1h |
| **Total** | 2× H100/A100 available | **~15-20h** |

---

## 8. Project Structure

```
Balatro-Agent/
├── balatro_gym/                  # Environment package (standalone, no RL deps)
│   ├── core/                    # Game engine
│   ├── envs/                    # Gymnasium interface + rewards + configs
│   ├── wrappers/                # Recording (trajectories, statistics)
│   └── environment_gym.yml      # Conda env for env-only users
├── agent/                        # Agent package (depends on balatro_gym)
│   ├── base.py                  # Agent Protocol + evaluation helpers
│   ├── baselines/               # Random, Heuristic
│   ├── rllib/                   # PPO training (Ray RLlib + action masking)
│   └── llm/                     # LLM agent (renderer + backends)
├── configs/                      # YAML game configurations
├── experiments/                  # Experiment scripts (TODO)
├── tests/                        # 383 unit tests
├── docs/                         # Documentation
└── environment.yml               # Conda env for full project
```

**Dependency graph**:
```
balatro_gym (gymnasium, numpy, pyyaml)
    ↑
agent.baselines (no extra deps)
agent.rllib (ray[rllib], torch)
agent.llm (transformers / openai / anthropic)
```
