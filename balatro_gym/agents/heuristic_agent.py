"""Heuristic agent: greedy rule-based strategy.

Strategy:
- Play phase: evaluate all playable subsets, pick the highest-scoring one.
  If score won't beat target and discards remain, discard low-value cards.
- Shop phase: buy the cheapest affordable joker, otherwise skip.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from balatro_gym.core.hand_evaluator import evaluate_hand, HAND_BASE_SCORES
from balatro_gym.envs.balatro_env import (
    BalatroEnv,
    TOTAL_ACTIONS,
    CARD_SUBSETS,
    PLAY_OFFSET,
    DISCARD_OFFSET,
    NUM_PLAY_ACTIONS,
    BUY_OFFSET,
    NUM_BUY_ACTIONS,
    SELL_OFFSET,
    REROLL_ACTION,
    SKIP_ACTION,
)


class HeuristicAgent:
    """Greedy heuristic: always play the highest-scoring valid hand.

    Shop strategy: buy cheapest joker if affordable, else skip.
    Discard strategy: if best hand is weak and discards remain, discard
    the lowest-value cards not part of any pair/triple.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, info: dict, env: BalatroEnv) -> int:
        mask = info["action_mask"]
        phase = info["phase"]

        if phase == "play":
            return self._play_action(mask, env)
        elif phase == "shop":
            return self._shop_action(mask, env)
        else:
            valid = np.where(mask)[0]
            return int(valid[0]) if len(valid) > 0 else 0

    def _play_action(self, mask: np.ndarray, env: BalatroEnv) -> int:
        """Choose the best play or discard action."""
        game = env._game
        assert game is not None
        hand = game.hand

        # Score all valid play actions
        best_play_action = -1
        best_play_score = -1

        for i in range(NUM_PLAY_ACTIONS):
            action_idx = PLAY_OFFSET + i
            if not mask[action_idx]:
                continue

            subset = CARD_SUBSETS[i]
            played = [hand[j] for j in subset]
            held = [hand[j] for j in range(len(hand)) if j not in subset]

            result = evaluate_hand(played, held)
            # Estimate score as base_chips * base_mult (ignoring jokers for speed)
            est_score = result.base_chips * result.base_mult

            if est_score > best_play_score:
                best_play_score = est_score
                best_play_action = action_idx

        # Decision: play or discard?
        score_needed = game.score_target - game.current_score
        hands_left = game.hands_remaining

        # If our best hand likely beats what's needed, or no discards left, play it
        if (best_play_score >= score_needed * 0.3
                or game.discards_remaining == 0
                or hands_left <= 1):
            if best_play_action >= 0:
                return best_play_action

        # Otherwise, try to discard weak cards
        discard_action = self._choose_discard(mask, hand)
        if discard_action >= 0:
            return discard_action

        # Fall back to best play
        if best_play_action >= 0:
            return best_play_action

        # Shouldn't reach here, but pick any valid action
        valid = np.where(mask)[0]
        return int(valid[0]) if len(valid) > 0 else 0

    def _choose_discard(self, mask: np.ndarray, hand: list) -> int:
        """Choose which cards to discard: drop the weakest non-paired cards."""
        from collections import Counter
        from balatro_gym.core.card import Card

        # Find ranks that appear multiple times (pairs/triples)
        rank_counts = Counter(c.rank for c in hand)
        paired_ranks = {r for r, count in rank_counts.items() if count >= 2}

        # Cards not in any pair, sorted by chip value (lowest first)
        unpaired = []
        for i, card in enumerate(hand):
            if card.rank not in paired_ranks:
                unpaired.append((i, card.chip_value))
        unpaired.sort(key=lambda x: x[1])

        # Discard up to 5 of the weakest unpaired cards
        to_discard = [idx for idx, _ in unpaired[:5]]
        if not to_discard:
            # All cards are paired; discard the lowest-value singles
            singles = [(i, c.chip_value) for i, c in enumerate(hand)]
            singles.sort(key=lambda x: x[1])
            to_discard = [idx for idx, _ in singles[:3]]

        if not to_discard:
            return -1

        # Find the matching discard action in CARD_SUBSETS
        to_discard_set = tuple(sorted(to_discard))
        for i, subset in enumerate(CARD_SUBSETS):
            action_idx = DISCARD_OFFSET + i
            if mask[action_idx] and subset == to_discard_set:
                return action_idx

        # Exact match not found; try subsets of what we want to discard
        for size in range(min(5, len(to_discard)), 0, -1):
            for combo in combinations(to_discard, size):
                combo_sorted = tuple(sorted(combo))
                for i, subset in enumerate(CARD_SUBSETS):
                    action_idx = DISCARD_OFFSET + i
                    if mask[action_idx] and subset == combo_sorted:
                        return action_idx

        return -1

    def _shop_action(self, mask: np.ndarray, env: BalatroEnv) -> int:
        """Buy cheapest affordable joker, else skip."""
        game = env._game
        assert game is not None

        # Try to buy the cheapest available joker
        best_buy = -1
        best_cost = float("inf")
        for slot_idx in range(NUM_BUY_ACTIONS):
            action_idx = BUY_OFFSET + slot_idx
            if mask[action_idx]:
                if slot_idx < len(game.shop.offerings):
                    cost = game.shop.offerings[slot_idx].cost
                    if cost < best_cost:
                        best_cost = cost
                        best_buy = action_idx

        if best_buy >= 0:
            return best_buy

        # Skip
        if mask[SKIP_ACTION]:
            return SKIP_ACTION

        valid = np.where(mask)[0]
        return int(valid[0]) if len(valid) > 0 else 0

    def run_episode(self, env: BalatroEnv) -> dict:
        """Run a full episode and return stats."""
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            action = self.act(obs, info, env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        return {
            "total_reward": total_reward,
            "steps": steps,
            "blinds_beaten": info["blinds_beaten"],
            "phase": info["phase"],
            "won": info["phase"] == "game_won",
        }
