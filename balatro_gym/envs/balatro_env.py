"""Gymnasium environment wrapper for the Balatro card game.

Wraps the core GameState with a Gymnasium-compatible API:
- Discrete(445) action space with action masking
- Flat observation vector (normalized floats)
- Reward shaping for RL training
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from balatro_gym.core.game_state import GamePhase, GameState
from balatro_gym.core.joker import get_all_joker_ids
from balatro_gym.envs.configs import GameConfig


# ---------------------------------------------------------------------------
# Pre-compute card subsets: all C(8,k) for k=1..5
# ---------------------------------------------------------------------------
CARD_SUBSETS: list[tuple[int, ...]] = []
for _size in range(1, 6):
    for _combo in combinations(range(8), _size):
        CARD_SUBSETS.append(_combo)
# len = C(8,1)+C(8,2)+C(8,3)+C(8,4)+C(8,5) = 8+28+56+70+56 = 218

NUM_PLAY_ACTIONS = len(CARD_SUBSETS)       # 218
NUM_DISCARD_ACTIONS = len(CARD_SUBSETS)    # 218
NUM_BUY_ACTIONS = 2                        # shop slot 0, 1
NUM_SELL_ACTIONS = 5                       # joker slot 0-4
NUM_REROLL = 1
NUM_SKIP = 1

PLAY_OFFSET = 0                                        # 0-217
DISCARD_OFFSET = NUM_PLAY_ACTIONS                      # 218-435
BUY_OFFSET = DISCARD_OFFSET + NUM_DISCARD_ACTIONS      # 436-437
SELL_OFFSET = BUY_OFFSET + NUM_BUY_ACTIONS             # 438-442
REROLL_ACTION = SELL_OFFSET + NUM_SELL_ACTIONS          # 443
SKIP_ACTION = REROLL_ACTION + NUM_REROLL                # 444

TOTAL_ACTIONS = SKIP_ACTION + NUM_SKIP                  # 445


class BalatroEnv(gym.Env):
    """Gymnasium environment for simplified Balatro.

    Observation: flat float32 vector (all values in [0, 1] or small positive range).
    Action: Discrete(445) with action masking via `action_masks()`.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        config: GameConfig | None = None,
        config_preset: str | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()

        # Resolve config
        if config is not None:
            self.config = config
        elif config_preset == "easy":
            self.config = GameConfig.easy()
        elif config_preset == "hard":
            self.config = GameConfig.hard()
        elif config_preset == "medium" or config_preset is None:
            self.config = GameConfig.medium()
        else:
            raise ValueError(f"Unknown config preset: {config_preset!r}")

        self.render_mode = render_mode

        # Build joker ID -> index mapping for one-hot encoding
        self._joker_ids = sorted(self.config.joker_pool)
        self._joker_id_to_idx: dict[str, int] = {
            jid: i for i, jid in enumerate(self._joker_ids)
        }
        self._num_joker_types = len(self._joker_ids)
        self._joker_one_hot_size = self._num_joker_types + 1  # +1 for empty

        # Compute observation dimension
        self._obs_dim = self._compute_obs_dim()

        # Spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=10.0,  # most values 0-1, but some can exceed 1
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)

        # Game state (created on reset)
        self._game: GameState | None = None

        # Tracking for reward shaping
        self._prev_score = 0
        self._prev_score_target = 0
        self._prev_blinds_beaten = 0

    def _compute_obs_dim(self) -> int:
        """Calculate total observation vector length."""
        dim = 0
        dim += 52                                  # hand bitmap
        dim += 52                                  # face-down flags
        dim += self.config.max_jokers * self._joker_one_hot_size  # joker slots
        dim += 1                                   # money (normalized)
        dim += 1                                   # ante (normalized)
        dim += 3                                   # blind type one-hot
        dim += 1                                   # score target (log-normalized)
        dim += 1                                   # score progress (ratio)
        dim += 1                                   # hands remaining (normalized)
        dim += 1                                   # discards remaining (normalized)
        dim += 1                                   # deck size (normalized)
        dim += 2                                   # phase one-hot (play, shop)
        dim += self.config.shop_slots * (self._joker_one_hot_size + 2)  # shop
        return dim

    # -------------------------------------------------------------------
    # Gymnasium API
    # -------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Create a fresh game state
        game_seed = seed if seed is not None else self.config.seed
        self._game = GameState(
            num_antes=self.config.num_antes,
            hands_per_round=self.config.hands_per_round,
            discards_per_round=self.config.discards_per_round,
            hand_size=self.config.hand_size,
            max_jokers=self.config.max_jokers,
            starting_money=self.config.starting_money,
            available_joker_ids=list(self.config.joker_pool),
            starting_joker_ids=list(self.config.starting_joker_ids),
            shop_slots=self.config.shop_slots,
            reroll_base_cost=self.config.reroll_base_cost,
            seed=game_seed,
        )
        self._game.reset()

        self._prev_score = 0
        self._prev_score_target = self._game.score_target
        self._prev_blinds_beaten = 0

        obs = self._encode_observation()
        info = self._build_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._game is not None, "Must call reset() before step()"

        reward = 0.0
        prev_phase = self._game.phase
        prev_blinds = self._game.blinds_beaten
        prev_score = self._game.current_score
        score_target = self._game.score_target

        # Execute action
        if self._game.phase == GamePhase.PLAY:
            reward += self._execute_play_action(action)
        elif self._game.phase == GamePhase.SHOP:
            reward += self._execute_shop_action(action)

        # Reward shaping: score progress
        if self._game.phase == GamePhase.PLAY and score_target > 0:
            new_score = self._game.current_score
            score_delta = (new_score - prev_score) / score_target
            reward += np.clip(score_delta * 0.01, 0.0, 0.01)

        # Reward for beating a blind
        if self._game.blinds_beaten > prev_blinds:
            total_blinds = self._game.blind_manager.total_blinds
            progress_bonus = self._game.blinds_beaten / total_blinds
            reward += 1.0 + progress_bonus

        # Terminal rewards
        terminated = False
        if self._game.phase == GamePhase.GAME_WON:
            reward += 10.0
            terminated = True
        elif self._game.phase == GamePhase.GAME_OVER:
            reward += -1.0
            terminated = True

        # Update tracking
        self._prev_score = self._game.current_score
        self._prev_score_target = (
            self._game.score_target
            if self._game.phase in (GamePhase.PLAY, GamePhase.SHOP)
            else self._prev_score_target
        )
        self._prev_blinds_beaten = self._game.blinds_beaten

        obs = self._encode_observation()
        info = self._build_info()

        return obs, reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        """Return a boolean mask over the 445 actions. True = valid."""
        mask = np.zeros(TOTAL_ACTIONS, dtype=bool)

        if self._game is None:
            return mask

        if self._game.phase == GamePhase.PLAY:
            hand_size = len(self._game.hand)
            can_play = self._game.hands_remaining > 0
            can_discard = self._game.discards_remaining > 0

            for i, subset in enumerate(CARD_SUBSETS):
                # Check all indices in subset are valid for current hand
                if max(subset) < hand_size:
                    if can_play:
                        mask[PLAY_OFFSET + i] = True
                    if can_discard:
                        mask[DISCARD_OFFSET + i] = True

        elif self._game.phase == GamePhase.SHOP:
            # Buy actions
            for slot_idx in range(self.config.shop_slots):
                action_idx = BUY_OFFSET + slot_idx
                if slot_idx < len(self._game.shop.offerings):
                    offering = self._game.shop.offerings[slot_idx]
                    if (not offering.sold
                            and self._game.money >= offering.cost
                            and len(self._game.jokers) < self._game.max_jokers):
                        mask[action_idx] = True

            # Sell actions
            for joker_idx in range(min(NUM_SELL_ACTIONS, len(self._game.jokers))):
                mask[SELL_OFFSET + joker_idx] = True

            # Reroll
            if self._game.money >= self._game.shop.reroll_cost:
                mask[REROLL_ACTION] = True

            # Skip is always valid in shop
            mask[SKIP_ACTION] = True

        # Game over / won: no valid actions
        return mask

    # -------------------------------------------------------------------
    # Action execution
    # -------------------------------------------------------------------

    def _execute_play_action(self, action: int) -> float:
        """Execute a play or discard action. Returns any immediate reward."""
        assert self._game is not None

        if PLAY_OFFSET <= action < PLAY_OFFSET + NUM_PLAY_ACTIONS:
            # Play a hand
            subset_idx = action - PLAY_OFFSET
            card_indices = list(CARD_SUBSETS[subset_idx])
            try:
                self._game.play_hand(card_indices)
            except ValueError:
                # Invalid action (shouldn't happen with proper masking)
                return -0.01
            return 0.0

        elif DISCARD_OFFSET <= action < DISCARD_OFFSET + NUM_DISCARD_ACTIONS:
            # Discard cards
            subset_idx = action - DISCARD_OFFSET
            card_indices = list(CARD_SUBSETS[subset_idx])
            try:
                self._game.discard(card_indices)
            except ValueError:
                return -0.01
            return 0.0

        # Invalid action for this phase
        return -0.01

    def _execute_shop_action(self, action: int) -> float:
        """Execute a shop action. Returns any immediate reward."""
        assert self._game is not None

        if BUY_OFFSET <= action < BUY_OFFSET + NUM_BUY_ACTIONS:
            slot_idx = action - BUY_OFFSET
            success = self._game.shop_buy(slot_idx)
            return 0.0

        elif SELL_OFFSET <= action < SELL_OFFSET + NUM_SELL_ACTIONS:
            joker_idx = action - SELL_OFFSET
            if joker_idx < len(self._game.jokers):
                self._game.shop_sell(joker_idx)
            return 0.0

        elif action == REROLL_ACTION:
            self._game.shop_reroll()
            return 0.0

        elif action == SKIP_ACTION:
            self._game.shop_skip()
            return 0.0

        return -0.01

    # -------------------------------------------------------------------
    # Observation encoding
    # -------------------------------------------------------------------

    def _encode_observation(self) -> np.ndarray:
        """Encode the current game state as a flat float32 vector."""
        assert self._game is not None

        obs = np.zeros(self._obs_dim, dtype=np.float32)
        offset = 0

        # 1. Hand bitmap (52-dim): 1 if card is in hand
        for card in self._game.hand:
            card_idx = int(card.suit) * 13 + (int(card.rank) - 2)
            obs[offset + card_idx] = 1.0
        offset += 52

        # 2. Face-down flags (52-dim): 1 if card is face-down
        for card in self._game.hand:
            if card.face_down:
                card_idx = int(card.suit) * 13 + (int(card.rank) - 2)
                obs[offset + card_idx] = 1.0
        offset += 52

        # 3. Joker slots (max_jokers x joker_one_hot_size)
        for slot in range(self.config.max_jokers):
            if slot < len(self._game.jokers):
                joker = self._game.jokers[slot]
                jid = joker.INFO.id
                if jid in self._joker_id_to_idx:
                    idx = self._joker_id_to_idx[jid]
                    obs[offset + idx] = 1.0
                # else: joker not in pool (shouldn't happen), leave as zeros
            else:
                # Empty slot: last position in one-hot
                obs[offset + self._num_joker_types] = 1.0
            offset += self._joker_one_hot_size

        # 4. Money (normalized by 100)
        obs[offset] = min(self._game.money / 100.0, 2.0)
        offset += 1

        # 5. Ante (normalized by num_antes)
        obs[offset] = self._game.ante / self.config.num_antes
        offset += 1

        # 6. Blind type one-hot (small=0, big=1, boss=2)
        blind_idx = self._game.blind_index
        if 0 <= blind_idx < 3:
            obs[offset + blind_idx] = 1.0
        offset += 3

        # 7. Score target (log-normalized)
        target = self._game.score_target
        if target > 0:
            obs[offset] = math.log1p(target) / 15.0  # log(50000) ~ 10.8
        offset += 1

        # 8. Score progress (current_score / score_target, capped at 2.0)
        if target > 0:
            obs[offset] = min(self._game.current_score / target, 2.0)
        offset += 1

        # 9. Hands remaining (normalized)
        obs[offset] = self._game.hands_remaining / max(self.config.hands_per_round, 1)
        offset += 1

        # 10. Discards remaining (normalized)
        obs[offset] = self._game.discards_remaining / max(self.config.discards_per_round, 1)
        offset += 1

        # 11. Deck size (normalized by 52)
        obs[offset] = self._game.deck.cards_remaining / 52.0
        offset += 1

        # 12. Phase one-hot (play=0, shop=1)
        if self._game.phase == GamePhase.PLAY:
            obs[offset] = 1.0
        elif self._game.phase == GamePhase.SHOP:
            obs[offset + 1] = 1.0
        offset += 2

        # 13. Shop offerings (shop_slots x (joker_one_hot_size + 2))
        for slot in range(self.config.shop_slots):
            if slot < len(self._game.shop.offerings):
                offering = self._game.shop.offerings[slot]
                jid = offering.joker.INFO.id
                if jid in self._joker_id_to_idx:
                    idx = self._joker_id_to_idx[jid]
                    obs[offset + idx] = 1.0
                # Cost (normalized by 20)
                obs[offset + self._joker_one_hot_size] = offering.cost / 20.0
                # Sold flag
                obs[offset + self._joker_one_hot_size + 1] = float(offering.sold)
            offset += self._joker_one_hot_size + 2

        assert offset == self._obs_dim, f"Obs size mismatch: {offset} != {self._obs_dim}"
        return obs

    # -------------------------------------------------------------------
    # Info dict
    # -------------------------------------------------------------------

    def _build_info(self) -> dict[str, Any]:
        """Build the info dict returned with each step."""
        assert self._game is not None

        info: dict[str, Any] = {
            "action_mask": self.action_masks(),
            "phase": self._game.phase.value,
            "ante": self._game.ante,
            "blind_index": self._game.blind_index,
            "score": self._game.current_score,
            "score_target": self._game.score_target,
            "money": self._game.money,
            "hands_remaining": self._game.hands_remaining,
            "discards_remaining": self._game.discards_remaining,
            "blinds_beaten": self._game.blinds_beaten,
            "num_jokers": len(self._game.jokers),
        }
        return info

    # -------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------

    def render(self) -> str | None:
        if self._game is None:
            return None

        if self.render_mode == "ansi" or self.render_mode == "human":
            return self._render_text()
        return None

    def _render_text(self) -> str:
        assert self._game is not None
        g = self._game
        lines = []

        lines.append(f"=== Ante {g.ante} | "
                      f"{g.current_blind_type.value.title()} Blind | "
                      f"Phase: {g.phase.value} ===")
        lines.append(f"Score: {g.current_score} / {g.score_target}")
        lines.append(f"Hands: {g.hands_remaining}  Discards: {g.discards_remaining}  "
                      f"Money: ${g.money}")
        lines.append(f"Hand: {' '.join(str(c) for c in g.hand)}")

        if g.jokers:
            lines.append(f"Jokers: {', '.join(str(j) for j in g.jokers)}")
        else:
            lines.append("Jokers: (none)")

        if g.phase == GamePhase.SHOP:
            available = g.shop.get_available_offerings()
            shop_str = ", ".join(
                f"[{i}] {o.joker.INFO.name} (${o.cost})"
                for i, o in available
            )
            lines.append(f"Shop: {shop_str or '(empty)'}")
            lines.append(f"Reroll cost: ${g.shop.reroll_cost}")

        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
        return text
