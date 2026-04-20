"""Gymnasium environment wrapper for the Balatro card game.

Wraps the core GameState with a Gymnasium-compatible API:
- Discrete action space with action masking
- Flat observation vector (normalized floats)
- Reward shaping for RL training

Designed for compatibility with sb3-contrib MaskablePPO and standard Gymnasium
tooling (env_checker, VecEnv wrappers, etc.).
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from balatro_gym.core.card import Enhancement, Edition, Seal
from balatro_gym.core.game_state import GamePhase, GameState
from balatro_gym.core.joker import get_all_joker_ids
from balatro_gym.core.consumable import get_all_consumable_ids, ConsumableType
from balatro_gym.core.seed_id import generate_seed_id, parse_seed_id
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
NUM_BUY_ACTIONS = 3                        # shop slot 0, 1 (jokers) + 1 (consumable)
NUM_SELL_ACTIONS = 5                       # joker slot 0-4
NUM_REROLL = 1
NUM_SKIP = 1

PLAY_OFFSET = 0                                        # 0-217
DISCARD_OFFSET = NUM_PLAY_ACTIONS                      # 218-435
BUY_OFFSET = DISCARD_OFFSET + NUM_DISCARD_ACTIONS      # 436-438
SELL_OFFSET = BUY_OFFSET + NUM_BUY_ACTIONS             # 439-443
REROLL_ACTION = SELL_OFFSET + NUM_SELL_ACTIONS          # 444
SKIP_ACTION = REROLL_ACTION + NUM_REROLL                # 445

TOTAL_ACTIONS = SKIP_ACTION + NUM_SKIP                  # 446

# Card property counts for observation encoding
NUM_ENHANCEMENTS = len(Enhancement)    # 8
NUM_EDITIONS = len(Edition)            # 3
NUM_SEALS = len(Seal)                  # 4

# Per-card feature size: 52 (which card) + 8 (enhancement) + 3 (edition) + 4 (seal) + 1 (face_down)
CARD_FEATURE_DIM = 52 + NUM_ENHANCEMENTS + NUM_EDITIONS + NUM_SEALS + 1  # 68

# Max hand slots
MAX_HAND_SIZE = 8
MAX_CONSUMABLE_SLOTS = 2
NUM_HAND_TYPES = 12  # for hand level encoding


class BalatroEnv(gym.Env):
    """Gymnasium environment for Balatro.

    Observation: flat float32 vector encoding hand cards (with properties),
    jokers, consumables, game scalars, shop state, and hand levels.

    Action: Discrete(446) with action masking via ``action_masks()``.
    Compatible with sb3-contrib ``MaskablePPO``.

    Registered IDs:
        - ``Balatro-v0`` (medium difficulty, default)
        - ``Balatro-Easy-v0``
        - ``Balatro-Medium-v0``
        - ``Balatro-Hard-v0``

    Example::

        import gymnasium as gym
        env = gym.make("Balatro-v0")
        obs, info = env.reset()
        mask = info["action_mask"]
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

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

        # Build consumable ID -> index mapping
        self._consumable_ids = sorted(self.config.consumable_pool)
        self._consumable_id_to_idx: dict[str, int] = {
            cid: i for i, cid in enumerate(self._consumable_ids)
        }
        self._num_consumable_types = len(self._consumable_ids)
        self._consumable_one_hot_size = self._num_consumable_types + 1  # +1 for empty

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

        # Episode seed ID (set on reset)
        self._game_seed: int = 0
        self.episode_seed_id: str = ""

    def _compute_obs_dim(self) -> int:
        """Calculate total observation vector length.

        Layout:
            [0]    Hand cards: MAX_HAND_SIZE * CARD_FEATURE_DIM
            [1]    Joker slots: max_jokers * joker_one_hot_size
            [2]    Consumable slots: MAX_CONSUMABLE_SLOTS * consumable_one_hot_size
            [3]    Hand levels: NUM_HAND_TYPES * 3 (level, chips, mult normalized)
            [4-14] Scalar features (money, ante, blind, score, etc.)
            [15]   Shop: shop_slots * (joker_one_hot_size + 2)
            [16]   Shop consumable slot: 1 * (consumable_one_hot_size + 2)
        """
        dim = 0
        dim += MAX_HAND_SIZE * CARD_FEATURE_DIM               # hand cards with properties
        dim += self.config.max_jokers * self._joker_one_hot_size  # joker slots
        dim += MAX_CONSUMABLE_SLOTS * self._consumable_one_hot_size  # consumable slots
        dim += NUM_HAND_TYPES * 3                              # hand levels
        dim += 1                                               # money (normalized)
        dim += 1                                               # ante (normalized)
        dim += 3                                               # blind type one-hot
        dim += 1                                               # score target (log-normalized)
        dim += 1                                               # score progress (ratio)
        dim += 1                                               # hands remaining (normalized)
        dim += 1                                               # discards remaining (normalized)
        dim += 1                                               # deck size (normalized)
        dim += 2                                               # phase one-hot (play, shop)
        dim += self.config.shop_slots * (self._joker_one_hot_size + 2)  # shop joker slots
        dim += 1 * (self._consumable_one_hot_size + 2)        # shop consumable slot
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
        """Reset the environment to a new game.

        Args:
            seed: RNG seed for reproducibility.
            options: Unused, for Gymnasium API compatibility.

        Returns:
            (observation, info) tuple. ``info["action_mask"]`` contains the
            boolean action mask for the initial state.
        """
        super().reset(seed=seed)

        # Use explicit seed, or derive from Gymnasium's np_random (which was
        # seeded by super().reset) for deterministic replay.
        if seed is not None:
            game_seed = seed
        elif self.np_random is not None:
            game_seed = int(self.np_random.integers(0, 2**31))
        else:
            game_seed = self.config.seed
        self._game = GameState(
            num_antes=self.config.num_antes,
            hands_per_round=self.config.hands_per_round,
            discards_per_round=self.config.discards_per_round,
            hand_size=self.config.hand_size,
            max_jokers=self.config.max_jokers,
            starting_money=self.config.starting_money,
            available_joker_ids=list(self.config.joker_pool),
            starting_joker_ids=list(self.config.starting_joker_ids),
            available_consumable_ids=list(self.config.consumable_pool),
            shop_slots=self.config.shop_slots,
            reroll_base_cost=self.config.reroll_base_cost,
            consumable_slots=self.config.consumable_slots,
            seed=game_seed,
        )
        self._game.reset()

        # Generate episode seed ID
        self._game_seed = game_seed
        self.episode_seed_id = generate_seed_id(game_seed)

        self._prev_score = 0
        self._prev_score_target = self._game.score_target
        self._prev_blinds_beaten = 0

        obs = self._encode_observation()
        info = self._build_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take one step in the environment.

        Args:
            action: Integer action index (0 to TOTAL_ACTIONS-1).

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        assert self._game is not None, "Must call reset() before step()"

        reward = 0.0
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

        return obs, float(reward), terminated, False, info

    def action_masks(self) -> np.ndarray:
        """Return a boolean mask over actions. ``True`` = valid.

        Compatible with sb3-contrib ``MaskablePPO``.
        """
        mask = np.zeros(TOTAL_ACTIONS, dtype=bool)

        if self._game is None:
            return mask

        if self._game.phase == GamePhase.PLAY:
            hand_size = len(self._game.hand)
            can_play = self._game.hands_remaining > 0
            can_discard = self._game.discards_remaining > 0

            for i, subset in enumerate(CARD_SUBSETS):
                if max(subset) < hand_size:
                    if can_play:
                        mask[PLAY_OFFSET + i] = True
                    if can_discard:
                        mask[DISCARD_OFFSET + i] = True

        elif self._game.phase == GamePhase.SHOP:
            # Buy actions — iterate over all offerings
            for slot_idx in range(min(NUM_BUY_ACTIONS, len(self._game.shop.offerings))):
                offering = self._game.shop.offerings[slot_idx]
                if offering.sold or self._game.money < offering.cost:
                    continue
                if offering.item_type == "joker":
                    if len(self._game.jokers) < self._game.max_jokers:
                        mask[BUY_OFFSET + slot_idx] = True
                elif offering.item_type == "consumable":
                    if len(self._game.consumables) < self._game.consumable_slots:
                        mask[BUY_OFFSET + slot_idx] = True

            # Sell actions
            for joker_idx in range(min(NUM_SELL_ACTIONS, len(self._game.jokers))):
                mask[SELL_OFFSET + joker_idx] = True

            # Reroll
            if self._game.money >= self._game.shop.reroll_cost:
                mask[REROLL_ACTION] = True

            # Skip is always valid in shop
            mask[SKIP_ACTION] = True

        return mask

    # -------------------------------------------------------------------
    # Action execution
    # -------------------------------------------------------------------

    def _execute_play_action(self, action: int) -> float:
        """Execute a play or discard action. Returns any immediate reward."""
        assert self._game is not None

        if PLAY_OFFSET <= action < PLAY_OFFSET + NUM_PLAY_ACTIONS:
            subset_idx = action - PLAY_OFFSET
            card_indices = list(CARD_SUBSETS[subset_idx])
            try:
                self._game.play_hand(card_indices)
            except ValueError:
                return -0.01
            return 0.0

        elif DISCARD_OFFSET <= action < DISCARD_OFFSET + NUM_DISCARD_ACTIONS:
            subset_idx = action - DISCARD_OFFSET
            card_indices = list(CARD_SUBSETS[subset_idx])
            try:
                self._game.discard(card_indices)
            except ValueError:
                return -0.01
            return 0.0

        return -0.01

    def _execute_shop_action(self, action: int) -> float:
        """Execute a shop action. Returns any immediate reward."""
        assert self._game is not None

        if BUY_OFFSET <= action < BUY_OFFSET + NUM_BUY_ACTIONS:
            slot_idx = action - BUY_OFFSET
            self._game.shop_buy(slot_idx)
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

    # Pre-build lookup dicts for fast encoding
    _ENHANCEMENT_IDX: dict[Enhancement, int] = {e: i for i, e in enumerate(Enhancement)}
    _EDITION_IDX: dict[Edition, int] = {e: i for i, e in enumerate(Edition)}
    _SEAL_IDX: dict[Seal, int] = {s: i for i, s in enumerate(Seal)}

    def _encode_observation(self) -> np.ndarray:
        """Encode the current game state as a flat float32 vector."""
        assert self._game is not None

        obs = np.zeros(self._obs_dim, dtype=np.float32)
        offset = 0

        # 1. Hand cards (MAX_HAND_SIZE * CARD_FEATURE_DIM)
        # Each card slot: 52-dim one-hot (which card) + 8 enhancement + 3 edition
        #                 + 4 seal + 1 face_down
        for slot in range(MAX_HAND_SIZE):
            if slot < len(self._game.hand):
                card = self._game.hand[slot]
                # Which card (52-dim one-hot)
                card_idx = int(card.suit) * 13 + (int(card.rank) - 2)
                obs[offset + card_idx] = 1.0
                # Enhancement (8-dim one-hot)
                if card.enhancement is not None:
                    enh_idx = self._ENHANCEMENT_IDX[card.enhancement]
                    obs[offset + 52 + enh_idx] = 1.0
                # Edition (3-dim one-hot)
                if card.edition is not None:
                    ed_idx = self._EDITION_IDX[card.edition]
                    obs[offset + 52 + NUM_ENHANCEMENTS + ed_idx] = 1.0
                # Seal (4-dim one-hot)
                if card.seal is not None:
                    seal_idx = self._SEAL_IDX[card.seal]
                    obs[offset + 52 + NUM_ENHANCEMENTS + NUM_EDITIONS + seal_idx] = 1.0
                # Face-down
                if card.face_down:
                    obs[offset + 52 + NUM_ENHANCEMENTS + NUM_EDITIONS + NUM_SEALS] = 1.0
            offset += CARD_FEATURE_DIM

        # 2. Joker slots (max_jokers * joker_one_hot_size)
        for slot in range(self.config.max_jokers):
            if slot < len(self._game.jokers):
                joker = self._game.jokers[slot]
                jid = joker.INFO.id
                if jid in self._joker_id_to_idx:
                    obs[offset + self._joker_id_to_idx[jid]] = 1.0
            else:
                obs[offset + self._num_joker_types] = 1.0  # empty marker
            offset += self._joker_one_hot_size

        # 3. Consumable slots (MAX_CONSUMABLE_SLOTS * consumable_one_hot_size)
        for slot in range(MAX_CONSUMABLE_SLOTS):
            if slot < len(self._game.consumables):
                consumable = self._game.consumables[slot]
                cid = consumable.INFO.id
                if cid in self._consumable_id_to_idx:
                    obs[offset + self._consumable_id_to_idx[cid]] = 1.0
            else:
                obs[offset + self._num_consumable_types] = 1.0  # empty marker
            offset += self._consumable_one_hot_size

        # 4. Hand levels (NUM_HAND_TYPES * 3): level/20, chips/500, mult/100
        from balatro_gym.core.hand_evaluator import HandType
        for ht in HandType:
            data = self._game.hand_levels.get_level(ht)
            obs[offset] = data.level / 20.0
            obs[offset + 1] = data.chips / 500.0
            obs[offset + 2] = data.mult / 100.0
            offset += 3

        # 5. Money (normalized by 100)
        obs[offset] = min(self._game.money / 100.0, 2.0)
        offset += 1

        # 6. Ante (normalized by num_antes)
        obs[offset] = self._game.ante / self.config.num_antes
        offset += 1

        # 7. Blind type one-hot (small=0, big=1, boss=2)
        blind_idx = self._game.blind_index
        if 0 <= blind_idx < 3:
            obs[offset + blind_idx] = 1.0
        offset += 3

        # 8. Score target (log-normalized)
        target = self._game.score_target
        if target > 0:
            obs[offset] = math.log1p(target) / 15.0
        offset += 1

        # 9. Score progress (current_score / score_target, capped at 2.0)
        if target > 0:
            obs[offset] = min(self._game.current_score / target, 2.0)
        offset += 1

        # 10. Hands remaining (normalized)
        obs[offset] = self._game.hands_remaining / max(self.config.hands_per_round, 1)
        offset += 1

        # 11. Discards remaining (normalized)
        obs[offset] = self._game.discards_remaining / max(self.config.discards_per_round, 1)
        offset += 1

        # 12. Deck size (normalized by 52)
        obs[offset] = self._game.deck.cards_remaining / 52.0
        offset += 1

        # 13. Phase one-hot (play=0, shop=1)
        if self._game.phase == GamePhase.PLAY:
            obs[offset] = 1.0
        elif self._game.phase == GamePhase.SHOP:
            obs[offset + 1] = 1.0
        offset += 2

        # 14. Shop joker offerings (shop_slots * (joker_one_hot_size + 2))
        for slot in range(self.config.shop_slots):
            if slot < len(self._game.shop.offerings):
                offering = self._game.shop.offerings[slot]
                if offering.item_type == "joker" and offering.joker is not None:
                    jid = offering.joker.INFO.id
                    if jid in self._joker_id_to_idx:
                        obs[offset + self._joker_id_to_idx[jid]] = 1.0
                obs[offset + self._joker_one_hot_size] = offering.cost / 20.0
                obs[offset + self._joker_one_hot_size + 1] = float(offering.sold)
            offset += self._joker_one_hot_size + 2

        # 15. Shop consumable offering (1 * (consumable_one_hot_size + 2))
        # Find the first consumable offering (if any)
        consumable_offering = None
        for o in self._game.shop.offerings:
            if o.item_type == "consumable":
                consumable_offering = o
                break
        if consumable_offering is not None:
            cid = consumable_offering.consumable.INFO.id if consumable_offering.consumable else None
            if cid and cid in self._consumable_id_to_idx:
                obs[offset + self._consumable_id_to_idx[cid]] = 1.0
            obs[offset + self._consumable_one_hot_size] = consumable_offering.cost / 20.0
            obs[offset + self._consumable_one_hot_size + 1] = float(consumable_offering.sold)
        offset += self._consumable_one_hot_size + 2

        assert offset == self._obs_dim, f"Obs size mismatch: {offset} != {self._obs_dim}"
        return obs

    # -------------------------------------------------------------------
    # Info dict
    # -------------------------------------------------------------------

    def _build_info(self) -> dict[str, Any]:
        """Build the info dict returned with each step.

        Always contains ``action_mask`` for use with masked RL algorithms.
        """
        assert self._game is not None

        return {
            "action_mask": self.action_masks(),
            "episode_seed_id": self.episode_seed_id,
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
            "num_consumables": len(self._game.consumables),
        }

    # -------------------------------------------------------------------
    # State save / load
    # -------------------------------------------------------------------

    def save_state(self) -> dict[str, Any]:
        """Serialize the full environment state to a JSON-compatible dict.

        Returns a checkpoint that can be passed to ``load_state()`` to
        restore the environment to this exact point and continue playing.

        The dict includes the game state, reward-shaping trackers, and
        the episode seed ID.
        """
        assert self._game is not None, "Must call reset() before save_state()"
        return {
            "game_state": self._game.serialize(),
            "episode_seed_id": self.episode_seed_id,
            "game_seed": self._game_seed,
            "prev_score": self._prev_score,
            "prev_score_target": self._prev_score_target,
            "prev_blinds_beaten": self._prev_blinds_beaten,
        }

    def load_state(self, checkpoint: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Restore the environment from a checkpoint dict.

        Args:
            checkpoint: Dict as returned by ``save_state()``.

        Returns:
            (observation, info) as if ``reset()`` had just returned at
            the checkpoint moment.
        """
        self._game = GameState.deserialize(checkpoint["game_state"])
        self.episode_seed_id = checkpoint["episode_seed_id"]
        self._game_seed = checkpoint["game_seed"]
        self._prev_score = checkpoint["prev_score"]
        self._prev_score_target = checkpoint["prev_score_target"]
        self._prev_blinds_beaten = checkpoint["prev_blinds_beaten"]

        obs = self._encode_observation()
        info = self._build_info()
        return obs, info

    @classmethod
    def from_seed_id(
        cls,
        seed_id: str,
        config: GameConfig | None = None,
        config_preset: str | None = None,
        render_mode: str | None = None,
    ) -> tuple["BalatroEnv", np.ndarray, dict[str, Any]]:
        """Create an environment and start the game encoded by a seed ID.

        This lets players share seed IDs (like roguelite seeds) to replay
        the same game.

        Args:
            seed_id: A seed string in ``YYYYMMDD-HHMM-XXXXXXXX`` format.
            config: GameConfig to use (overrides preset).
            config_preset: Difficulty preset if no config given.
            render_mode: Gymnasium render mode.

        Returns:
            (env, obs, info) — ready to play.
        """
        parsed = parse_seed_id(seed_id)
        game_seed = parsed["game_seed"]

        env = cls(config=config, config_preset=config_preset, render_mode=render_mode)
        obs, info = env.reset(seed=game_seed)
        # Overwrite the auto-generated seed ID with the original one
        # so the timestamp portion is preserved
        env.episode_seed_id = seed_id
        info["episode_seed_id"] = seed_id
        return env, obs, info

    # -------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------

    def render(self) -> str | None:
        if self._game is None:
            return None

        if self.render_mode == "ansi":
            return self._render_text()
        elif self.render_mode == "human":
            print(self._render_text())
            return None
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

        if g.consumables:
            lines.append(f"Consumables: {', '.join(str(c) for c in g.consumables)}")

        if g.phase == GamePhase.SHOP:
            available = g.shop.get_available_offerings()
            parts = []
            for i, o in available:
                parts.append(f"[{i}] {o.name} (${o.cost}, {o.item_type})")
            lines.append(f"Shop: {', '.join(parts) or '(empty)'}")
            lines.append(f"Reroll cost: ${g.shop.reroll_cost}")

        return "\n".join(lines)
