# Balatro-Agent

An RL agent that learns to play the roguelite deck-building game Balatro.

## Introduction

This project has two parts:

1. **Balatro-Gym** -- A Gymnasium-compatible simulation engine that recreates Balatro's game mechanics in Python. It serves as a reasoning gym for training and evaluating AI agents on strategic decision-making: probability estimation, expected value calculation, combinatorial optimization, and long-horizon planning.

2. **Balatro-Agent** -- An RL agent trained via PPO (Proximal Policy Optimization) with action masking to play the game, learning strategies that go beyond hand-crafted heuristics.

### Project Status

| Component | Status |
|-----------|--------|
| Core game engine (cards, hand evaluation, jokers, blinds, shop, game state) | In progess |
| Gymnasium environment wrapper (observation encoding, action masking) | In progress |
| Difficulty configs (easy/medium/hard YAML presets) | In progress |
| Baseline agents (random, heuristic) | Waiting |
| PPO training pipeline (MaskablePPO via SB3) | Waiting |
| Curriculum learning (easy -> hard) | Waiting |
| Evaluation and ablation experiments | Waiting |

### About Balatro

[Balatro](https://www.playbalatro.com/) is a single-player roguelite where you score points by playing poker hands -- but with a twist. You collect **Jokers** (modifier cards) that warp the scoring rules, and you must build synergies between your Jokers and your deck to hit exponentially growing score targets.

**Why is Balatro a good environment for agent planning research?**

Consider this situation: you are in Ante 3, facing the Boss Blind with a target score of 4,000. You have 4 hands and 3 discards. Your hand is:

```
[K of spades] [K of hearts] [7 of diamonds] [4 of clubs] [J of spades] [10 of hearts] [3 of diamonds] [9 of clubs]
```

Your Jokers are:
```
[Greedy Joker]: +3 Mult for each Diamond card played
[Banner]: +30 Chips for each discard remaining
```

A naive agent plays the obvious pair of Kings for a safe ~60 points. But a strategic agent reasons differently:

- *"I have Banner, which gives +30 chips per discard remaining. If I play now with 3 discards left, that's +90 chips. But if I use discards, I lose that bonus."*
- *"I have Greedy Joker, which rewards Diamonds. My current hand only has 7 of diamonds and 3 of diamonds. I could discard non-diamond cards and fish for a Diamond flush -- but that sacrifices Banner's bonus."*
- *"The target is 4,000. I have 4 hands. I need ~1,000 per hand on average. A pair of Kings scores about 50. I need to do much better -- should I risk discards to chase a stronger hand type, or preserve discards for Banner's bonus?"*

This single decision involves **probability estimation** (what's the chance of drawing diamonds?), **expected value calculation** (is the flush attempt worth the Banner loss?), **joker synergy reasoning** (which Joker benefits more from my play style?), and **multi-turn planning** (I have 4 hands to reach 4,000 -- how should I budget them?).

Unlike Texas Hold'em, which focuses on adversarial bluffing, Balatro isolates the **planning and reasoning** challenge: there is no opponent to model, but the combinatorial space of Joker interactions and the exponentially growing score targets demand genuine strategic thinking.

## Tech Stack

- Python 3.12+
- `gymnasium` -- environment API
- `stable-baselines3` + `sb3-contrib` -- PPO with action masking
- `numpy` -- numerical computation
- `pytest` -- testing
- `pyyaml` -- config loading
- `tensorboard` or `wandb` -- experiment tracking (optional)
