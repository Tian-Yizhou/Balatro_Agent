# Balatro-Agent

## Introduction

**Target**

Construct an AI agent that can play Balatro, a single-player deck-building roguelite card game that uses poker hands as its core mechanic. The goal is to use reinforcement learning to train an agent that wins as many rounds as possible.

**Two Contributions**

1. **Balatro-Gym**: A Gymnasium-compatible simulation engine that provides an interactive environment for training and evaluating agents on strategic reasoning and planning.
2. **Balatro-Agent**: An RL agent trained via PPO (Proximal Policy Optimization) with action masking to play this game, demonstrating that RL training can learn non-trivial strategies beyond hand-crafted heuristics.

**Motivation**

Why is a card game environment valuable for agent AI research?

1. **Virtual environments are necessary middle steps.** Before deploying agents in the physical world, we need controlled virtual environments to develop and evaluate reasoning capabilities. Physical environments face hardware limitations, high experiment costs, and safety concerns. Virtual environments provide cheap, fast, reproducible experiments.
2. **Most existing task benchmarks are too narrow.** Standard benchmarks (e.g., "put the turkey on the table") have finite horizons and a single expected output. Real-world tasks require multi-step planning under uncertainty with many valid strategies. A card game with stochastic draws, resource management, and compounding decisions better reflects this complexity.
3. **Poker-style games are ideal testbeds for mathematical reasoning, but existing ones have limitations.** Poker games have computable transition probabilities, making them well-suited for studying decision-making. However, existing simulators like Texas Hold'em focus on adversarial play (bluffing, opponent modeling) rather than combinatorial planning and resource optimization. Balatro fills this gap: it is single-player (no adversary), but demands probability estimation, expected value calculation, long-horizon planning, and dynamic strategy adaptation.

**Advantages of Balatro as a Reasoning Gym**

1. **Deep MDP with controllable depth.** The game target can be configured: "beat the first ante" (short), "beat 4 antes" (medium), or "survive all 8 antes" (long). This naturally supports curriculum learning, gradually increasing task difficulty.
2. **Curriculum learning support.** Training can progress from easy configurations (fewer antes, simpler jokers) to hard configurations (all antes, complex joker interactions), allowing the agent to build up strategic capabilities incrementally.
3. **Short-term vs. long-term planning trade-offs.** The game constantly forces decisions between immediate reward and future payoff: Should you play a weaker hand now to save discards? Should you save money for interest instead of buying a joker? Should you sell a joker to fund a better one?
4. **Richer than standard poker environments.** Compared to Texas Hold'em or other poker RL environments, Balatro offers:
   - **Non-linear scoring growth**: Score targets grow exponentially across antes, requiring multiplicative strategies, not just additive play.
   - **Dynamic deck-building**: Joker acquisition and synergy planning add a combinatorial optimization layer beyond simple probability calculation.
   - **Single-player**: No adversarial hidden information or opponent modeling needed, isolating the reasoning and planning challenge. Similar to other Gymnasium environments (Atari, MuJoCo), it is a single-agent MDP.
   - **Partial but non-adversarial information**: The deck composition is known but the draw order is random. Shop offerings are stochastic. This creates uncertainty that requires probabilistic reasoning, without the confound of adversarial bluffing.

## Game Overview

Balatro is a single-player deck-building roguelite where the player uses poker hands to score points:

- **Deck**: Standard 52 cards (4 suits $\times$ 13 ranks). Player is dealt 8 cards per round.
- **Play**: Select 1-5 cards from hand to form a poker hand (pair, flush, straight, etc.). Each hand type has base chips and a base multiplier. Score = chips $\times$ mult.
- **Discards**: Limited discards per round to swap cards for better draws.
- **Jokers**: Special modifier cards (up to 5 held) that alter scoring rules. Examples: "+4 mult for each pair", "$$\times$$1.5 mult if hand contains a heart", "+30 chips for each face card played". Joker ordering matters (applied left-to-right).
- **Shop**: Between rounds, spend money to buy jokers, sell jokers, or reroll shop offerings.
- **Blinds**: Each ante has 3 blinds (small, big, boss) with increasing score targets. Boss blinds impose debuff effects (e.g., "all hearts are face-down", "only 1 hand allowed").
- **Economy**: Earn \$ from winning rounds + interest on savings (\$1 per \$5 held, max \$5 interest).
- **Win condition**: Beat all blinds across all antes (e.g., 8 antes = 24 blinds).
- **Lose condition**: Fail to reach a blind's score target with available hands.

The strategic depth comes from the interaction between poker hand selection, joker synergies, resource management (money, hands, discards), and the escalating difficulty curve.

## Methodology

### Simulation Engine Construction

**Process**

1. **Reference-guided clean-room implementation.** Read the Balatro Lua source code (LOVE2D engine) to understand exact formulas, scoring rules, joker effects, and game flow. Then implement from scratch in Python with an original code structure. The Lua source is used purely as a specification reference and is never committed to the repository. This ensures accuracy while avoiding copyright issues (game mechanics are not copyrightable; only specific code expression is).
2. **Build modular core game logic.** Implement the game as independent, testable modules:
   - Card and Deck primitives (standard 52-card deck, shuffling, drawing)
   - Poker hand detection and base scoring
   - Joker system with a registry pattern (each joker is a self-contained class with a standard interface)
   - Blind progression and boss blind debuff effects
   - Shop mechanics (buying, selling, rerolling)
   - Central game state manager that orchestrates the scoring pipeline and game flow
3. **Wrap with Gymnasium API.** Expose the game as a standard Gymnasium environment (`reset()` / `step()` / `render()`) so it can interface with any RL framework. The observation space is a flat numerical vector encoding the full game state (card positions, joker slots, money, score, etc.). The action space is a single Discrete space with action masking.
4. **Design configurable difficulty.** Create a configuration system that controls game parameters (number of antes, hands per round, discards per round) and joker pools (which jokers are available). Provide preset difficulty levels (easy/medium/hard) and support custom YAML configs for experiments.

**Techniques Needed**

1. **OpenAI Gymnasium API**
   - *What it is*: Gymnasium (formerly OpenAI Gym) is the standard Python API for reinforcement learning environments. It defines a universal interface (`reset`, `step`, `render`) that any RL agent can interact with, regardless of the specific environment.
   - *What problem it solves*: Without a standard interface, every new environment requires custom integration code. Gymnasium makes our Balatro environment immediately compatible with any RL library (Stable Baselines3, RLlib, CleanRL, etc.) and reusable by other researchers.
   - *Resources needed*: The `gymnasium` Python package. CPU only for the environment itself.

2. **Markov Decision Process (MDP) formulation**
   - *What it is*: MDP is the mathematical framework for modeling sequential decision-making under uncertainty. It defines states, actions, transition probabilities, and rewards. In our formulation, the state is the full game state (hand, jokers, money, deck, blind), actions are card selections or shop decisions, transitions are stochastic (card draws, shop offerings), and rewards are based on blind outcomes.
   - *What problem it solves*: Formalizing Balatro as an MDP allows us to apply RL algorithms to train agents. It also clarifies the observation space and action space design: what information the agent sees and what decisions it can make.
   - *Resources needed*: No special resources. This is a design framework applied during implementation.

3. **Action masking for multi-phase action spaces**
   - *What it is*: The game has two distinct phases (play and shop) with different valid actions. We enumerate all possible actions into a single Discrete(445) space: 218 play subsets + 218 discard subsets + 9 shop actions. An action mask (boolean array) indicates which actions are valid at each step, preventing the agent from selecting invalid actions.
   - *What problem it solves*: Standard RL environments assume a fixed action space, but Balatro's valid actions change by phase and game state (e.g., can't buy if no money, can't discard if no discards left). Action masking lets us use a single policy network for all phases while guaranteeing only valid actions are taken. This avoids the need for separate networks per phase.
   - *Resources needed*: `sb3-contrib` package (provides `MaskablePPO`). No GPU needed for the masking logic itself.

4. **Observation space encoding**
   - *What it is*: A method to convert the rich game state (cards, jokers, money, score, etc.) into a flat numerical vector that a neural network can process. We use: a 52-dim binary vector for cards in hand, one-hot encoding for joker slots (preserving order), and normalized scalars for money, score, ante, etc.
   - *What problem it solves*: Neural networks require fixed-size numerical inputs. The encoding must capture all strategically relevant information: which cards are available, which jokers are active (and in what order), the current scoring target, resource counts, and the game phase. Poor encoding would prevent the agent from learning effective strategies.
   - *Resources needed*: No special resources. Pure numpy array construction.

**Challenges**

1. **Scoring pipeline correctness.** Jokers apply their effects in left-to-right order, and within each joker the order is: add chips $\rightarrow$ add mult $\rightarrow$ multiply mult. Different joker orderings produce different scores. Getting this exactly right is critical, as incorrect scoring would corrupt all downstream training signals.
2. **Variable action space across phases.** The game alternates between play and shop phases with completely different valid actions. The Gymnasium wrapper must handle this cleanly while maintaining a consistent API for the agent.
3. **Stateful joker interactions.** Some jokers accumulate state across hands or rounds (e.g., "gains +15 chips permanently each time a straight is played"). Others copy abilities of neighboring jokers. These interactions create edge cases that must be carefully tested.

### Agent Construction

**Process**

1. **Build baseline agents.** Implement a random agent (selects random valid actions) and a heuristic agent (always plays the highest-scoring hand, buys the cheapest joker, uses simple discard rules). These set the lower and upper bounds for non-learned play.
2. **Train PPO agent.** Using Stable Baselines3's MaskablePPO, train a neural network policy on the structured observation space. The policy network is a standard MLP (multi-layer perceptron) that maps the observation vector to action probabilities, with invalid actions masked out.
3. **Curriculum training.** Train in stages: first on easy difficulty (4 antes, simple jokers), then transfer the learned policy to medium (6 antes), then hard (8 antes). This allows the agent to learn basic card-playing strategy before facing the full complexity.
4. **Evaluation.** Compare all agents (random, heuristic, PPO, PPO-curriculum) on win rate, average ante reached, and survival curves.

**Techniques Needed**

1. **Proximal Policy Optimization (PPO)**
   - *What it is*: PPO is a policy gradient reinforcement learning algorithm that updates the policy by maximizing a clipped surrogate objective. It alternates between collecting rollouts (playing games) and optimizing the policy network via stochastic gradient descent. The clipping mechanism prevents overly large policy updates, improving training stability.
   - *What problem it solves*: We need an RL algorithm that can learn from episodic game outcomes in a large, discrete action space. PPO is the most widely used algorithm for this setting because it balances sample efficiency, stability, and simplicity. It works with both discrete and continuous action spaces, and handles the stochastic nature of card games well.
   - *Resources needed*: `stable-baselines3` and `sb3-contrib` (for MaskablePPO). Training can run on CPU (slower, ~hours) or single GPU (faster). No large GPU cluster needed.

2. **Action masking (MaskablePPO)**
   - *What it is*: An extension of PPO that zeroes out the probability of invalid actions before sampling. At each step, the environment provides a boolean mask indicating which of the 445 actions are valid. The policy's softmax output is masked so only valid actions receive probability mass.
   - *What problem it solves*: Without masking, the agent would frequently attempt invalid actions (e.g., buying a joker during the play phase, or discarding when no discards remain), receiving error signals that waste training time and confuse learning. Masking guarantees every sampled action is valid, dramatically improving sample efficiency.
   - *Resources needed*: `sb3-contrib` package. The `ActionMasker` wrapper connects the environment's mask function to MaskablePPO.

3. **Reward shaping**
   - *What it is*: Augmenting the sparse win/lose reward with intermediate signals that guide learning. Our shaped reward includes: (a) large positive reward for winning the game, (b) scaled positive reward for beating each blind (more for later blinds), (c) small per-step reward proportional to score progress toward the current target.
   - *What problem it solves*: With only a win/lose signal at the end of a 50+ step episode, the agent receives almost no gradient signal early in training (it almost never wins randomly). Shaped rewards provide a learning signal at every step, helping the agent first learn to score points, then to beat blinds, then to beat the full game.
   - *Resources needed*: No special resources. Implemented in the environment's reward function.

4. **Curriculum learning**
   - *What it is*: A training strategy where the agent is first trained on easy tasks and gradually exposed to harder ones. In our case: start with easy config (4 antes, simple jokers, more hands/discards), then progress to medium and hard configs.
   - *What problem it solves*: Training directly on the hardest difficulty may be too challenging for the agent to learn any useful signal (reward is too sparse -- the agent almost never wins). Curriculum learning provides a smoother learning curve, allowing the agent to first master basic strategy before tackling complex scenarios.
   - *Resources needed*: Multiple difficulty configurations (already built into the environment). Additional training time (~1.5$\times$ compared to single-difficulty training).

**Challenges**

1. **Reward sparsity.** The primary signal is binary: win or lose, determined only after many sequential decisions. The agent must figure out which of its many actions led to winning or losing. Mitigation: shaped rewards provide intermediate signals for score progress and blind completion.
2. **Credit assignment across long horizons.** A single game involves 20+ sequential decisions (multiple hands per blind, multiple blinds per ante, shop decisions between blinds). Determining which decisions were good vs. bad is inherently difficult. Mitigation: PPO's value function baseline helps with temporal credit assignment, and shaped rewards shorten the effective horizon.
3. **Large discrete action space.** 445 possible actions is larger than typical Gymnasium environments (e.g., Atari has 18). However, action masking reduces the effective space to ~20-50 valid actions per step, which is manageable for PPO.
4. **Observation encoding quality.** If the observation vector fails to capture strategically relevant information (e.g., joker interactions, remaining deck composition), the agent cannot learn effective play. Mitigation: we include all game-relevant information in the observation and run ablation studies on encoding choices.

## Evaluation Plan

**Agents to compare:**
| Agent | Description |
|-------|-------------|
| Random | Selects random valid actions each step |
| Heuristic | Greedy rule-based (always plays best hand, buys cheapest joker) |
| PPO | Trained on single difficulty with shaped rewards |
| PPO-Curriculum | Trained with progressive difficulty (easy $\rightarrow$ medium $\rightarrow$ hard) |

**Metrics:**
- **Win rate**: Percentage of games where the agent beats all blinds
- **Average ante reached**: How far the agent progresses before losing
- **Average score in a given blind**: Raw scoring performance
- **Survival curve**: Per-blind survival rate (what percentage of agents beat blind $X$?)

**Ablation studies:**
1. **Reward shaping**: Sparse reward (win/lose only) vs. shaped reward. Does shaping help?
2. **Curriculum learning**: Train on easy $\rightarrow$ medium $\rightarrow$ hard vs. train on hard directly. Does curriculum help?
3. **Network architecture**: MLP [256, 256] vs. [512, 256] vs. [128, 128, 128]. What capacity is needed?

## Timeline

| Period | Milestone |
|--------|-----------|
| Apr 20 | Proposal presentation |
| Apr 20 - May 4 | Core game engine (done). Gymnasium env wrapper with observation/action encoding. Unit tests. |
| May 4 - May 11 | Baseline agents (random + heuristic). Integration testing. Validate env with SB3. |
| May 11 - May 25 | PPO training. Curriculum training. Hyperparameter tuning. |
| May 25 - Jun 1 | Run experiments, ablation studies. Prepare final presentation. |
| Jun 1 | Final presentation |
| Jun 1 - Jun 8 | Write final report with results and analysis. |
| Jun 8 | Report due |

## Compute Resources

| Component | Hardware | Estimated Time |
|-----------|----------|----------------|
| Simulation engine | CPU only | N/A |
| PPO training (MLP policy) | CPU or 1$\times$ GPU | ~2-8 hours per run |
| Baseline evaluation (1000 games) | CPU | ~minutes |
| **Total compute** | | **~20-40 CPU/GPU-hours** |

## References

- Schulman et al. "Proximal Policy Optimization Algorithms." 2017.
- Huang et al. "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms." 2022.
- Raffin et al. "Stable-Baselines3: Reliable Reinforcement Learning Implementations." 2021.
- Brockman et al. "OpenAI Gym." 2016. (Gymnasium)
