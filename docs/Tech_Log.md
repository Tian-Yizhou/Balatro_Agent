# Tech Log

Technical decisions, implementation notes, and key details for the Balatro-Agent project.

---

## 2026-04-16: Project Pivot — Traditional RL Instead of LLM Agent

**Decision**: Dropped the LLM-based agent (SFT + GRPO on Qwen2.5-7B) in favor of traditional RL (PPO with MLP policy). Reasons:
- Limited GPU access makes LLM training impractical (~100-150 GPU-hours for GRPO)
- PPO with Stable Baselines3 can train on CPU or a single GPU in hours
- Structured observations are simpler to implement and debug than text rendering + parsing
- Still covers core course topics: MDP formulation, reward design, RL training, curriculum learning

**What stays the same**: Core game engine, Gymnasium API, joker system, configurable difficulty.
**What changes**: Observation = flat numerical vector (not text), action = Discrete(445) with masking (not Dict), agent = MLP policy (not LLM).

---

## 2026-04-17: Engine Reconstruction from Lua Source

**What changed**: Reconstructed all core engine modules by reading the Balatro Lua source code (card.lua, game.lua, blind.lua, state_events.lua, misc_functions.lua) as a specification reference. Key corrections:

### Scoring Pipeline (game_state.py)
The scoring pipeline was fundamentally restructured to match Lua's `evaluate_play`:
1. **3-phase scoring**: Individual card effects → held card effects → main joker effects (was: flat joker iteration)
2. **Per-card joker triggers**: Suit-based jokers (Greedy/Lusty/etc.), Fibonacci, Even Steven, Odd Todd, Scholar, Business Card now trigger per scoring card in `on_individual()` context (was: summed over all cards at once)
3. **Poker hand sub-types**: `_get_poker_hands()` builds a dict of all contained hand types (e.g., Full House contains Pair + Three of a Kind). Jolly/Zany/Duo/Trio/Family check this dict, matching Lua's `next(context.poker_hands[type])`.
4. **Before/After phases**: Runner and Ride the Bus update state in `on_before()` before scoring. Ice Cream decrements chips in `on_after()` after scoring.

### Blind System (blind.py)
- **Score targets**: Now use `get_blind_amount(ante) * blind.mult` matching Lua exactly. Base amounts: [300, 800, 2000, 5000, 11000, 20000, 35000, 50000]. Blind mults: Small=1.0, Big=1.5, Boss varies (2.0 typical, The Wall=4.0).
- **Boss blinds**: Added BlindDef dataclass with name, dollars, mult. Implemented The Flint (halves base chips/mult with rounding), The Hook (discards 2 random cards), The Needle (1 hand only), The Wall (4x mult).
- **Economy**: Fixed to match Lua — blind.dollars (3/4/5) + $1 per unused hand + interest (was: win_bonus = 3 + blind_index, which was wrong).

### Joker System (joker.py)
- **Architecture change**: Replaced single `on_score()` method with context-specific methods: `on_before()`, `on_individual()`, `on_held_individual()`, `on_main()`, `on_after()`. This matches Lua's context-based calculate_joker dispatch.
- **Odd Todd**: Fixed to +31 chips (was +30).
- **Half Joker**: Now checks `len(full_hand) <= 3` (was checking scoring_cards).
- **Ceremonial Dagger**: Corrected — gains mult from slicing right neighbor at blind start, not from blind score target.
- **Supernova**: Uses global hand type play counts, incremented before scoring (matching Lua).
- **Raised Fist**: Moved to `on_held_individual()` context, uses 2x nominal of lowest held card.
- **Stencil**: X-mult of (empty_slots + 1).

### Card System (card.py)
- `chip_value` now returns 0 for debuffed (face_down) cards, matching `Card:get_chip_bonus()` in Lua.
- Added `nominal` property (raw chip value ignoring debuff) for Raised Fist.
- Added `id` property mapping to Lua's card.base.id (2-14, Ace=14).

### Verification
- All hand base scores match Lua exactly
- All blind base amounts match Lua exactly  
- Score targets match Lua's formula
- Economy matches Lua's evaluate_round
- 100 random-agent games run with zero crashes

---

## Core Engine Implementation Status

All modules in `balatro_gym/core/` are implemented. Key details:

### card.py
- `Suit(IntEnum)`: HEARTS=0, DIAMONDS=1, CLUBS=2, SPADES=3
- `Rank(IntEnum)`: TWO=2 through ACE=14
- `Card` dataclass: rank, suit, face_down (mutable for boss debuffs)
- `Card.chip_value`: Ace=11, K/Q/J=10, others=rank value
- `Deck`: uses `np.random.Generator` for seeded reproducibility. draw_pile + discard_pile. Auto-reshuffles discard into draw when draw runs out.

### hand_evaluator.py
- `HandType(IntEnum)`: HIGH_CARD(0) through FLUSH_FIVE(11). Includes Balatro-specific types: FIVE_OF_A_KIND, FLUSH_HOUSE, FLUSH_FIVE.
- `HAND_BASE_SCORES`: dict mapping HandType to (base_chips, base_mult)
- `HandResult` dataclass: hand_type, scoring_cards, held_cards, base_chips, base_mult
- `evaluate_hand(played_cards, held_cards)`: checks from highest to lowest hand type. Returns HandResult. `base_chips` already includes the chip values of scoring cards.
- Edge cases handled: ace-low straight (A-2-3-4-5), ace-high straight (10-J-Q-K-A), playing fewer than 5 cards.

### joker.py
- `GameStateView` defined as `@runtime_checkable Protocol` to avoid circular imports with game_state.py
- `ScoreModification(add_chips, add_mult, x_mult)` -- returned by each joker's `on_score()`
- `@register_joker` decorator + `_JOKER_REGISTRY` dict
- 30 jokers implemented across 3 priority tiers (10 each)
- Stateful jokers use `self._internal_state` dict (e.g., IceCream, LoyaltyCard, CeremonialDagger, RideTheBus, Runner, Supernova)
- Blueprint joker copies the ability of the joker to its right (finds itself in the joker list, returns neighbor's on_score result)

### blind.py
- `BLIND_TARGETS` lookup table: ante 1-8 x (small, big, boss). Values escalate non-linearly (300 at ante 1 small to 100,000 at ante 8 boss).
- 7 boss effects implemented: DebuffHearts, DebuffSpades, DebuffClubs, DebuffDiamonds, DebuffFaceCards, FirstHandScoresZero, NoDiscards. Each has apply/remove methods.
- `BlindManager.get_score_target()` extrapolates beyond ante 8 with 1.5x scaling.

### shop.py
- `Shop.generate_offerings()`: draws random jokers from the configured pool
- `buy_joker()` validates: slot exists, not sold, can afford, have joker slots
- `sell_value()` = max(1, cost // 2)
- `reroll()` costs money (increasing by 1 each reroll) and regenerates offerings

### game_state.py
- `GameStateSnapshot` dataclass satisfies the `GameStateView` Protocol from joker.py
- Scoring pipeline in `_apply_scoring()`: start with base_chips + base_mult, iterate jokers left-to-right, apply add_chips then add_mult then x_mult per joker, return chips * mult
- `_process_side_effects()` handles joker side effects: money, money_chance (probabilistic), copy_card
- `_calculate_economy()`: win_bonus (3 + blind_index) + interest (min(money//5, 5))
- Phase transitions: PLAY -> (beat blind) -> SHOP -> (skip) -> PLAY, or PLAY -> (fail) -> GAME_OVER, or SHOP -> (last ante done) -> GAME_WON

---

## Observation Space Design

**Total dimensions**: ~240 (exact number depends on joker pool size)

| Component | Encoding | Dimensions |
|-----------|----------|------------|
| Cards in hand | 52-dim binary (1 = in hand) | 52 |
| Face-down flags | 52-dim binary (1 = face-down) | 52 |
| Joker slots | 5 x (num_jokers+1) one-hot | 5 x 31 = 155 |
| Money | normalized by 100 | 1 |
| Ante | normalized by num_antes | 1 |
| Blind type | 3-dim one-hot | 3 |
| Score target | log-normalized | 1 |
| Current score / target | ratio, capped at 2 | 1 |
| Hands remaining | normalized | 1 |
| Discards remaining | normalized | 1 |
| Deck size | normalized by 52 | 1 |
| Phase | 2-dim one-hot (play, shop) | 2 |
| Shop offerings | 2 x (joker one-hot + cost + sold) | ~66 |

**Card index formula**: `suit * 13 + (rank - 2)`. Example: K of spades = 3*13 + 11 = 50.

**Why 52-dim binary for hand** (not 8 x card_encoding): Binary bitmap is simpler, fixed-size regardless of hand size, and naturally encodes "which cards exist" rather than "what's in each slot." The neural network doesn't need to track card ordering within the hand.

**Why one-hot for joker slots** (not bitmap): Joker ORDER matters for scoring. Slot 0 = leftmost joker, applied first. A bitmap would lose ordering information.

---

## Action Space Design

**Total: 445 discrete actions**

Pre-computed at env init:
```python
from itertools import combinations

card_subsets = []
for size in range(1, 6):
    for combo in combinations(range(8), size):
        card_subsets.append(combo)
# len = C(8,1)+C(8,2)+C(8,3)+C(8,4)+C(8,5) = 8+28+56+70+56 = 218
```

| Action range | Meaning |
|-------------|---------|
| 0-217 | Play card subset (index into card_subsets) |
| 218-435 | Discard card subset (same mapping, offset by 218) |
| 436-437 | Buy shop slot 0 or 1 |
| 438-442 | Sell joker from slot 0-4 |
| 443 | Reroll shop |
| 444 | Skip shop (advance to next blind) |

**Action mask**: boolean array of length 445. During play phase, only 0-435 can be True. During shop phase, only 436-444 can be True. Further filtered by game state (e.g., can't discard if discards_remaining == 0, can't buy if can't afford).

---

## Reward Function Design

| Event | Reward | Rationale |
|-------|--------|-----------|
| Win game | +10.0 | Strong signal for completing all antes |
| Lose game | -1.0 | Negative but not crushing (encourage exploration) |
| Beat a blind | +1.0 to +2.5 | Scaled by progress: 1.0 + (blinds_beaten / total_blinds) + efficiency bonus |
| Mid-step score progress | 0.0 to 0.01 | Tiny reward proportional to current_score / score_target |

The asymmetric win/lose rewards (+10 vs -1) reflect that winning is rare and should be heavily reinforced, while losing is common early in training and shouldn't overly penalize exploration.

---

## Dependencies (Updated)

```
gymnasium>=0.29.0
numpy>=1.24.0
stable-baselines3>=2.0.0
sb3-contrib>=2.0.0
torch>=2.0.0
pytest>=7.0.0
pyyaml>=6.0
tensorboard>=2.0.0
```

Note: `transformers`, `trl`, `vllm`, `peft` are NO LONGER needed after the pivot to traditional RL.
