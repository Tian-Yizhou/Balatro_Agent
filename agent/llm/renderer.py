"""Game state renderer: converts BalatroEnv state to structured JSON.

The renderer reads the environment's internal game state and produces a
JSON-serializable dict containing all information an agent needs to make
decisions. This structured representation can be:

1. Sent directly to an LLM as JSON in the prompt.
2. Converted to natural language via a template (future extension).
3. Logged for debugging or trajectory analysis.

The JSON output includes:
- Game phase and progression (ante, blind, score target)
- Hand cards with full properties
- Jokers with descriptions
- Consumables with descriptions
- Valid actions (enumerated with descriptions)
- Economy (money, shop state)
- Hand levels

Example::

    from agent.llm.renderer import GameStateRenderer
    import balatro_gym

    env = balatro_gym.make("easy", seed=42)
    renderer = GameStateRenderer(env)

    obs, info = env.reset()
    state_json = renderer.render(info)
    # state_json is a dict ready for json.dumps() or LLM prompting
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from balatro_gym.core.card import Card, Rank, Suit, Enhancement, Edition, Seal
from balatro_gym.core.game_state import GamePhase, GameState
from balatro_gym.core.joker import BaseJoker
from balatro_gym.core.consumable import BaseConsumable
from balatro_gym.core.hand_evaluator import HandType
from balatro_gym.envs.balatro_env import (
    BalatroEnv,
    CARD_SUBSETS,
    PLAY_OFFSET,
    DISCARD_OFFSET,
    BUY_OFFSET,
    SELL_OFFSET,
    REROLL_ACTION,
    SKIP_ACTION,
    NUM_PLAY_ACTIONS,
    NUM_DISCARD_ACTIONS,
    NUM_BUY_ACTIONS,
    NUM_SELL_ACTIONS,
    TOTAL_ACTIONS,
)

# Human-readable names
_RANK_NAMES = {
    Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5",
    Rank.SIX: "6", Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9",
    Rank.TEN: "10", Rank.JACK: "Jack", Rank.QUEEN: "Queen",
    Rank.KING: "King", Rank.ACE: "Ace",
}
_SUIT_NAMES = {
    Suit.HEARTS: "Hearts", Suit.DIAMONDS: "Diamonds",
    Suit.CLUBS: "Clubs", Suit.SPADES: "Spades",
}


class GameStateRenderer:
    """Renders BalatroEnv game state as structured JSON.

    The renderer needs access to the environment to read its internal
    ``_game`` state. It produces a complete JSON-serializable dict.

    Args:
        env: A :class:`BalatroEnv` instance.
    """

    def __init__(self, env: BalatroEnv):
        self.env = env

    def render(self, info: dict[str, Any] | None = None) -> dict[str, Any]:
        """Render the current game state as a structured dict.

        Args:
            info: The info dict from the last step/reset. If None,
                the renderer will call env._build_info() internally.

        Returns:
            A JSON-serializable dict containing the full game state.
        """
        game: GameState = self.env._game
        if game is None:
            return {"error": "Environment not initialized. Call env.reset() first."}

        if info is None:
            info = self.env._build_info()

        action_mask = info["action_mask"]

        state = {
            "game_progress": self._render_progress(game),
            "hand": self._render_hand(game),
            "jokers": self._render_jokers(game),
            "consumables": self._render_consumables(game),
            "economy": self._render_economy(game),
            "hand_levels": self._render_hand_levels(game),
            "valid_actions": self._render_valid_actions(game, action_mask),
        }

        # Add shop state if in shop phase
        if game.phase == GamePhase.SHOP:
            state["shop"] = self._render_shop(game)

        return state

    def render_json(self, info: dict[str, Any] | None = None, indent: int = 2) -> str:
        """Render as a formatted JSON string (convenience method).

        Useful for directly inserting into an LLM prompt.
        """
        return json.dumps(self.render(info), indent=indent)

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _render_progress(self, game: GameState) -> dict[str, Any]:
        """Render game progression info."""
        blind_type = "small"
        if game.blind_index == 1:
            blind_type = "big"
        elif game.blind_index == 2:
            blind_type = "boss"

        result = {
            "phase": game.phase.value,
            "ante": game.ante,
            "total_antes": game.num_antes,
            "blind_type": blind_type,
            "blind_name": game.current_blind_def.name if game.current_blind_def else blind_type,
            "score_target": game.score_target,
            "current_score": game.current_score,
            "hands_remaining": game.hands_remaining,
            "discards_remaining": game.discards_remaining,
            "blinds_beaten": game.blinds_beaten,
            "total_blinds": game.blind_manager.total_blinds,
        }

        # Boss effect
        if game.active_boss_effect:
            result["boss_effect"] = {
                "name": game.current_blind_def.name,
                "description": game.current_blind_def.description,
            }

        return result

    def _render_hand(self, game: GameState) -> list[dict[str, Any]]:
        """Render cards currently in hand."""
        cards = []
        for i, card in enumerate(game.hand):
            cards.append(self._render_card(card, index=i))
        return cards

    def _render_card(self, card: Card, index: int | None = None) -> dict[str, Any]:
        """Render a single card."""
        result: dict[str, Any] = {}
        if index is not None:
            result["index"] = index
        result["rank"] = _RANK_NAMES.get(card.rank, str(card.rank))
        result["suit"] = _SUIT_NAMES.get(card.suit, str(card.suit))
        result["chip_value"] = card.chip_value
        if card.enhancement:
            result["enhancement"] = card.enhancement.value
        if card.edition:
            result["edition"] = card.edition.value
        if card.seal:
            result["seal"] = card.seal.value
        if card.face_down:
            result["face_down"] = True
        return result

    def _render_jokers(self, game: GameState) -> list[dict[str, Any]]:
        """Render owned jokers."""
        jokers = []
        for i, joker in enumerate(game.jokers):
            info = joker.INFO
            entry: dict[str, Any] = {
                "index": i,
                "id": info.id,
                "name": info.name,
                "description": info.description,
                "rarity": info.rarity,
                "sell_value": info.cost // 2,
            }
            jokers.append(entry)
        return jokers

    def _render_consumables(self, game: GameState) -> list[dict[str, Any]]:
        """Render owned consumables."""
        consumables = []
        for i, cons in enumerate(game.consumables):
            info = cons.INFO
            entry: dict[str, Any] = {
                "index": i,
                "id": info.id,
                "name": info.name,
                "type": info.consumable_type.value,
                "description": info.description,
            }
            consumables.append(entry)
        return consumables

    def _render_economy(self, game: GameState) -> dict[str, Any]:
        """Render economy state."""
        return {
            "money": game.money,
            "deck_size": game.deck.cards_remaining,
        }

    def _render_hand_levels(self, game: GameState) -> dict[str, Any]:
        """Render current hand type levels and scoring."""
        levels = {}
        for ht in HandType:
            chips, mult = game.hand_levels.get_score(ht)
            data = game.hand_levels.get_level(ht)
            levels[ht.name.lower()] = {
                "level": data.level,
                "chips": chips,
                "mult": mult,
            }
        return levels

    def _render_shop(self, game: GameState) -> dict[str, Any]:
        """Render shop offerings."""
        offerings = []
        for i, offering in enumerate(game.shop.offerings):
            if offering.sold:
                continue
            entry: dict[str, Any] = {
                "slot": i,
                "cost": offering.cost,
                "type": offering.item_type,
            }
            if offering.item_type == "joker" and offering.joker is not None:
                info = offering.joker.INFO
                entry["name"] = info.name
                entry["id"] = info.id
                entry["description"] = info.description
            elif offering.item_type == "consumable" and offering.consumable is not None:
                info = offering.consumable.INFO
                entry["name"] = info.name
                entry["id"] = info.id
                entry["description"] = info.description
            offerings.append(entry)

        return {
            "offerings": offerings,
            "reroll_cost": game.shop.reroll_cost,
        }

    def _render_valid_actions(
        self, game: GameState, action_mask: np.ndarray
    ) -> list[dict[str, Any]]:
        """Render valid actions with human-readable descriptions.

        Only includes actions where action_mask is True.
        """
        actions = []

        for action_idx in range(TOTAL_ACTIONS):
            if not action_mask[action_idx]:
                continue

            desc = self._describe_action(action_idx, game)
            if desc:
                actions.append({"action_id": action_idx, **desc})

        return actions

    def _describe_action(
        self, action: int, game: GameState
    ) -> dict[str, Any] | None:
        """Produce a human-readable description of an action."""
        if PLAY_OFFSET <= action < PLAY_OFFSET + NUM_PLAY_ACTIONS:
            subset_idx = action - PLAY_OFFSET
            card_indices = list(CARD_SUBSETS[subset_idx])
            cards_desc = self._describe_card_subset(card_indices, game)
            return {"type": "play", "card_indices": card_indices, "cards": cards_desc}

        elif DISCARD_OFFSET <= action < DISCARD_OFFSET + NUM_DISCARD_ACTIONS:
            subset_idx = action - DISCARD_OFFSET
            card_indices = list(CARD_SUBSETS[subset_idx])
            cards_desc = self._describe_card_subset(card_indices, game)
            return {"type": "discard", "card_indices": card_indices, "cards": cards_desc}

        elif BUY_OFFSET <= action < BUY_OFFSET + NUM_BUY_ACTIONS:
            slot_idx = action - BUY_OFFSET
            return {"type": "buy", "shop_slot": slot_idx}

        elif SELL_OFFSET <= action < SELL_OFFSET + NUM_SELL_ACTIONS:
            joker_idx = action - SELL_OFFSET
            joker_name = ""
            if joker_idx < len(game.jokers):
                joker_name = game.jokers[joker_idx].INFO.name
            return {"type": "sell", "joker_index": joker_idx, "joker_name": joker_name}

        elif action == REROLL_ACTION:
            return {"type": "reroll"}

        elif action == SKIP_ACTION:
            return {"type": "skip"}

        return None

    def _describe_card_subset(
        self, indices: list[int], game: GameState
    ) -> list[str]:
        """Produce short card descriptions for a subset of hand indices."""
        descs = []
        for i in indices:
            if i < len(game.hand):
                card = game.hand[i]
                rank = _RANK_NAMES.get(card.rank, str(card.rank))
                suit = _SUIT_NAMES.get(card.suit, str(card.suit))
                descs.append(f"{rank} of {suit}")
            else:
                descs.append(f"[index {i}]")
        return descs
