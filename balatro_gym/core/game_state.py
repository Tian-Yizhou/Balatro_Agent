"""Full game state manager: orchestrates all game phases and the scoring pipeline.

Reference: Balatro Lua source — state_events.lua (evaluate_play, evaluate_round),
game.lua (starting params), misc_functions.lua (get_starting_params).
"""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from balatro_gym.core.card import Card, Deck, Rank
from balatro_gym.core.hand_evaluator import HandResult, HandType, evaluate_hand
from balatro_gym.core.joker import BaseJoker, ScoreModification, create_joker
from balatro_gym.core.blind import (
    BlindType, BlindDef, BlindManager, BossEffect,
    SMALL_BLIND, BIG_BLIND,
)
from balatro_gym.core.shop import Shop


class GamePhase(enum.Enum):
    PLAY = "play"
    SHOP = "shop"
    GAME_OVER = "game_over"
    GAME_WON = "game_won"


# Map HandType enum to the Lua hand name strings used in poker_hands dict
HAND_TYPE_NAMES: dict[HandType, str] = {
    HandType.HIGH_CARD: "High Card",
    HandType.PAIR: "Pair",
    HandType.TWO_PAIR: "Two Pair",
    HandType.THREE_OF_A_KIND: "Three of a Kind",
    HandType.STRAIGHT: "Straight",
    HandType.FLUSH: "Flush",
    HandType.FULL_HOUSE: "Full House",
    HandType.FOUR_OF_A_KIND: "Four of a Kind",
    HandType.STRAIGHT_FLUSH: "Straight Flush",
    HandType.FIVE_OF_A_KIND: "Five of a Kind",
    HandType.FLUSH_HOUSE: "Flush House",
    HandType.FLUSH_FIVE: "Flush Five",
}


def _get_poker_hands(hand_type: HandType) -> dict[str, bool]:
    """Build the poker_hands dict: True for the hand type and all sub-hands contained.

    In Lua, G.FUNCS.get_poker_hand_info returns all poker hands present.
    E.g., a Full House also contains Pair and Three of a Kind.
    """
    result: dict[str, bool] = {}
    ht = hand_type

    # Always set the detected hand type
    result[HAND_TYPE_NAMES[ht]] = True

    # Set contained sub-hands
    if ht in (HandType.PAIR, HandType.TWO_PAIR, HandType.THREE_OF_A_KIND,
              HandType.FULL_HOUSE, HandType.FOUR_OF_A_KIND, HandType.FIVE_OF_A_KIND,
              HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE):
        result["Pair"] = True
    if ht in (HandType.THREE_OF_A_KIND, HandType.FULL_HOUSE,
              HandType.FLUSH_HOUSE):
        result["Three of a Kind"] = True
    if ht in (HandType.FOUR_OF_A_KIND, HandType.FIVE_OF_A_KIND):
        result["Four of a Kind"] = True
    if ht in (HandType.TWO_PAIR,):
        result["Two Pair"] = True
    if ht in (HandType.STRAIGHT, HandType.STRAIGHT_FLUSH):
        result["Straight"] = True
    if ht in (HandType.FLUSH, HandType.STRAIGHT_FLUSH,
              HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE):
        result["Flush"] = True
    if ht in (HandType.FULL_HOUSE, HandType.FLUSH_HOUSE):
        result["Full House"] = True

    return result


@dataclass
class GameStateSnapshot:
    """Read-only snapshot of game state. Satisfies the GameStateView protocol in joker.py."""
    hand: list[Card]
    jokers: list[BaseJoker]
    money: int
    ante: int
    blind_type: str
    score_target: int
    current_score: int
    hands_remaining: int
    discards_remaining: int
    deck_size: int
    max_jokers: int
    hands_played_this_round: int
    hand_type_played_counts: dict[str, int]


class GameState:
    """Central game state manager.

    Owns the deck, hand, jokers, money, blind progression, shop,
    and the critical scoring pipeline.
    """

    def __init__(
        self,
        num_antes: int = 8,
        hands_per_round: int = 4,
        discards_per_round: int = 3,
        hand_size: int = 8,
        max_jokers: int = 5,
        starting_money: int = 4,
        available_joker_ids: list[str] | None = None,
        starting_joker_ids: list[str] | None = None,
        shop_slots: int = 2,
        reroll_base_cost: int = 5,
        seed: int | None = None,
    ):
        # Config
        self.num_antes = num_antes
        self.hands_per_round = hands_per_round
        self.discards_per_round = discards_per_round
        self.hand_size = hand_size
        self.max_jokers = max_jokers
        self.starting_money = starting_money
        self.available_joker_ids = available_joker_ids or []
        self.starting_joker_ids = starting_joker_ids or []
        self.shop_slots = shop_slots
        self.reroll_base_cost = reroll_base_cost

        # RNG
        self.rng = np.random.default_rng(seed)

        # Game objects
        self.deck = Deck(self.rng)
        self.blind_manager = BlindManager(num_antes)
        self.shop = Shop(
            joker_pool=self.available_joker_ids,
            rng=self.rng,
            num_slots=shop_slots,
            reroll_base_cost=reroll_base_cost,
        )

        # Mutable state (set in reset)
        self.hand: list[Card] = []
        self.jokers: list[BaseJoker] = []
        self.money: int = 0
        self.ante: int = 1
        self.blind_index: int = 0  # 0=small, 1=big, 2=boss
        self.current_score: int = 0
        self.hands_remaining: int = 0
        self.discards_remaining: int = 0
        self.phase: GamePhase = GamePhase.PLAY
        self.hands_played_this_round: int = 0
        self.active_boss_effect: BossEffect | None = None
        self.current_blind_def: BlindDef = SMALL_BLIND

        # Hand type play counts (for Supernova, etc.)
        self.hand_type_played_counts: dict[str, int] = {}

        # Statistics
        self.total_hands_played: int = 0
        self.blinds_beaten: int = 0

    @property
    def current_blind_type(self) -> BlindType:
        return BlindManager.BLINDS_PER_ANTE[self.blind_index]

    @property
    def score_target(self) -> int:
        return self.blind_manager.get_score_target(self.ante, self.current_blind_def)

    @property
    def effective_hand_size(self) -> int:
        """Hand size including passive joker effects."""
        bonus = 0
        view = self.get_view()
        for joker in self.jokers:
            effects = joker.passive_effects(view)
            bonus += effects.get("hand_size", 0)
        return self.hand_size + bonus

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def reset(self) -> GameStateSnapshot:
        """Reset to a fresh game."""
        self.deck.reset()
        self.hand = []
        self.jokers = [create_joker(jid) for jid in self.starting_joker_ids]
        self.money = self.starting_money
        self.ante = 1
        self.blind_index = 0
        self.current_score = 0
        self.hands_remaining = self.hands_per_round
        self.discards_remaining = self.discards_per_round
        self.phase = GamePhase.PLAY
        self.hands_played_this_round = 0
        self.active_boss_effect = None
        self.current_blind_def = SMALL_BLIND
        self.hand_type_played_counts = {}
        self.total_hands_played = 0
        self.blinds_beaten = 0

        self._start_blind()
        return self.get_view()

    def _start_blind(self) -> None:
        """Set up for a new blind: determine blind def, reset score, deal hand, apply boss effects."""
        self.current_score = 0
        self.hands_remaining = self.hands_per_round
        self.discards_remaining = self.discards_per_round
        self.hands_played_this_round = 0
        self.phase = GamePhase.PLAY

        # Determine blind definition
        bt = self.current_blind_type
        if bt == BlindType.SMALL:
            self.current_blind_def = SMALL_BLIND
        elif bt == BlindType.BIG:
            self.current_blind_def = BIG_BLIND
        else:
            self.current_blind_def = self.blind_manager.choose_boss(self.ante, self.rng)

        # Reset hand type played counts per round
        for key in self.hand_type_played_counts:
            pass  # Keep cumulative counts (they persist across the run)

        # Reset deck and deal
        self.deck.reset()
        self.hand = self.deck.draw(self.effective_hand_size)

        # Apply boss effect if boss blind
        self.active_boss_effect = None
        if bt == BlindType.BOSS:
            self.active_boss_effect = self.blind_manager.get_boss_effect(self.current_blind_def)
            self.active_boss_effect.apply(self)

        # Debuff newly drawn cards if boss is active
        if self.active_boss_effect:
            for card in self.hand:
                if self.active_boss_effect.debuff_card(card):
                    card.face_down = True

        # Notify jokers of round start
        view = self.get_view()
        for joker in self.jokers:
            joker.on_round_start(view)

    def _end_blind(self) -> None:
        """Handle beating a blind: award money, notify jokers, advance."""
        self.blinds_beaten += 1

        # Remove boss effect
        if self.active_boss_effect is not None:
            self.active_boss_effect.remove(self)
            self.active_boss_effect = None

        # Notify jokers
        view = self.get_view()
        for joker in self.jokers:
            effects = joker.on_end_of_round(view)
            self._process_side_effects(effects)

        # Award money (Lua: evaluate_round)
        self.money += self._calculate_economy()

        # Advance blind
        self.blind_index += 1
        if self.blind_index >= len(BlindManager.BLINDS_PER_ANTE):
            # Finished all blinds in this ante
            self.blind_index = 0
            self.ante += 1
            if self.ante > self.num_antes:
                self.phase = GamePhase.GAME_WON
                return

        # Enter shop phase
        self.phase = GamePhase.SHOP
        self.shop.generate_offerings()

    def _calculate_economy(self) -> int:
        """Calculate end-of-round money award.

        Lua (evaluate_round):
        1. blind.dollars (Small=3, Big=4, Boss=5)
        2. +$1 per unused hand remaining
        3. Joker dollar bonuses (calculate_dollar_bonus)
        4. Interest: $1 per $5 held, max $5
        """
        dollars = 0

        # 1. Blind reward
        dollars += self.current_blind_def.dollars

        # 2. $1 per unused hand
        dollars += self.hands_remaining

        # 3. Joker dollar bonuses
        view = self.get_view()
        for joker in self.jokers:
            dollars += joker.calculate_dollar_bonus(view)

        # 4. Interest (must calculate BEFORE adding to money)
        # Interest is on current money, not including this round's earnings
        interest = min(self.money // 5, 5)
        dollars += interest

        return dollars

    # -------------------------------------------------------------------
    # Play phase actions
    # -------------------------------------------------------------------

    def play_hand(self, card_indices: list[int]) -> tuple[int, HandResult]:
        """Play selected cards from hand as a poker hand.

        Returns:
            (score_earned, hand_result)
        """
        if self.phase != GamePhase.PLAY:
            raise ValueError(f"Cannot play hand in phase {self.phase}")
        if not card_indices:
            raise ValueError("Must select at least 1 card to play")
        if len(card_indices) > 5:
            raise ValueError("Cannot play more than 5 cards")
        if self.hands_remaining <= 0:
            raise ValueError("No hands remaining")

        # Validate indices
        for idx in card_indices:
            if idx < 0 or idx >= len(self.hand):
                raise ValueError(f"Invalid card index: {idx}")
        if len(set(card_indices)) != len(card_indices):
            raise ValueError("Duplicate card indices")

        # Extract played cards (full_hand) and held cards
        sorted_indices = sorted(card_indices)
        full_hand = [self.hand[i] for i in sorted_indices]
        held_cards = [c for i, c in enumerate(self.hand) if i not in card_indices]

        # Evaluate hand type
        hand_result = evaluate_hand(full_hand, held_cards)
        scoring_name = HAND_TYPE_NAMES[hand_result.hand_type]
        scoring_hand = hand_result.scoring_cards
        poker_hands = _get_poker_hands(hand_result.hand_type)

        # Update hand type play count (Lua does this BEFORE scoring)
        self.hand_type_played_counts[scoring_name] = (
            self.hand_type_played_counts.get(scoring_name, 0) + 1
        )

        # Boss blind press_play hook (e.g., The Hook discards cards)
        if self.active_boss_effect is not None:
            self.active_boss_effect.press_play(self)

        # Check if blind debuffs the entire hand type
        # (Not implemented for simplicity — only card-level debuffs for now)

        # Apply scoring pipeline
        score = self._apply_scoring(hand_result, scoring_hand, full_hand,
                                     poker_hands, scoring_name, held_cards)
        self.current_score += score

        # Update state
        self.hands_remaining -= 1
        self.hands_played_this_round += 1
        self.total_hands_played += 1

        # Return played cards to deck discard pile
        self.deck.return_cards(full_hand)

        # Remove played cards from hand and draw replacements
        self.hand = held_cards
        new_cards = self.deck.draw(len(full_hand))
        self.hand.extend(new_cards)

        # Debuff new cards if boss is active
        if self.active_boss_effect:
            for card in new_cards:
                if self.active_boss_effect.debuff_card(card):
                    card.face_down = True

        # Check if blind is beaten
        if self.current_score >= self.score_target:
            self._end_blind()
        elif self.hands_remaining <= 0:
            self.phase = GamePhase.GAME_OVER

        return score, hand_result

    def discard(self, card_indices: list[int]) -> list[Card]:
        """Discard selected cards and draw replacements."""
        if self.phase != GamePhase.PLAY:
            raise ValueError(f"Cannot discard in phase {self.phase}")
        if not card_indices:
            raise ValueError("Must select at least 1 card to discard")
        if len(card_indices) > 5:
            raise ValueError("Cannot discard more than 5 cards")
        if self.discards_remaining <= 0:
            raise ValueError("No discards remaining")

        for idx in card_indices:
            if idx < 0 or idx >= len(self.hand):
                raise ValueError(f"Invalid card index: {idx}")
        if len(set(card_indices)) != len(card_indices):
            raise ValueError("Duplicate card indices")

        # Extract discarded cards
        discarded = [self.hand[i] for i in sorted(card_indices)]

        # Notify jokers
        view = self.get_view()
        for joker in self.jokers:
            effects = joker.on_discard(discarded, view)
            self._process_side_effects(effects)

        # Return discarded cards to deck and draw new ones
        self.deck.return_cards(discarded)
        self.hand = [c for i, c in enumerate(self.hand) if i not in card_indices]
        new_cards = self.deck.draw(len(discarded))
        self.hand.extend(new_cards)

        # Debuff new cards if boss is active
        if self.active_boss_effect:
            for card in new_cards:
                if self.active_boss_effect.debuff_card(card):
                    card.face_down = True

        self.discards_remaining -= 1
        return new_cards

    # -------------------------------------------------------------------
    # Shop phase actions
    # -------------------------------------------------------------------

    def shop_buy(self, slot_index: int) -> bool:
        """Buy a joker from the shop. Returns True if successful."""
        if self.phase != GamePhase.SHOP:
            raise ValueError(f"Cannot buy in phase {self.phase}")

        joker, remaining = self.shop.buy_joker(
            slot_index, self.money, self.jokers, self.max_jokers
        )
        if joker is not None:
            self.jokers.append(joker)
            self.money = remaining
            return True
        return False

    def shop_sell(self, joker_index: int) -> int:
        """Sell a joker from the player's collection. Returns money gained."""
        if self.phase != GamePhase.SHOP:
            raise ValueError(f"Cannot sell in phase {self.phase}")
        if joker_index < 0 or joker_index >= len(self.jokers):
            raise ValueError(f"Invalid joker index: {joker_index}")

        joker = self.jokers.pop(joker_index)
        sell_value = self.shop.sell_value(joker)
        self.money += sell_value
        return sell_value

    def shop_reroll(self) -> bool:
        """Reroll shop offerings. Returns True if successful."""
        if self.phase != GamePhase.SHOP:
            raise ValueError(f"Cannot reroll in phase {self.phase}")

        success, remaining = self.shop.reroll(self.money)
        if success:
            self.money = remaining
        return success

    def shop_skip(self) -> None:
        """Skip the shop and advance to the next blind."""
        if self.phase != GamePhase.SHOP:
            raise ValueError(f"Cannot skip in phase {self.phase}")

        self._start_blind()

    # -------------------------------------------------------------------
    # Scoring pipeline
    # -------------------------------------------------------------------

    def _apply_scoring(self, hand_result: HandResult, scoring_hand: list[Card],
                       full_hand: list[Card], poker_hands: dict[str, bool],
                       scoring_name: str, held_cards: list[Card]) -> int:
        """Apply the full scoring pipeline matching Lua's evaluate_play.

        Scoring order (from state_events.lua):
        1. Start with base chips and base mult from hand type
        2. Blind modify_hand (The Flint halves them)
        3. For each scoring card (left to right):
           a. Card chip bonus (nominal + bonuses) -> add to chips
           b. Card mult bonus -> add to mult  (enhanced cards, not in our simplified version)
           c. Card x_mult -> multiply mult (enhanced cards, not in our simplified version)
           d. For each joker: individual card effects (chips/mult/x_mult per card)
        4. For each held card in hand:
           a. Card held-mult -> add to mult
           b. For each joker: held-individual effects
        5. For each joker (left to right):
           a. Edition effects (not in our simplified version)
           b. Main joker effects: chip_mod -> mult_mod -> Xmult_mod
           c. Joker-on-joker effects (not in our simplified version)
           d. Edition x_mult (not in our simplified version)
        6. Final score = max(0, chips * mult)
        """
        # 1. Base chips and mult
        chips = hand_result.base_chips
        mult = hand_result.base_mult
        view = self.get_view()

        # 2. Blind modify_hand (e.g., The Flint)
        if self.active_boss_effect:
            new_mult, new_chips, modified = self.active_boss_effect.modify_hand(mult, chips)
            mult = new_mult
            chips = new_chips

        # Joker "before" phase (stat updates like Runner, Ride the Bus)
        for joker in self.jokers:
            mod = joker.on_before(hand_result, scoring_hand, full_hand,
                                  poker_hands, scoring_name, view)
            # "before" effects don't directly modify score, they update joker state

        # 3. Score each scoring card individually
        for card in scoring_hand:
            if card.face_down:
                # Debuffed cards don't score
                continue

            # a. Card chip bonus (already included in base_chips via hand_evaluator)
            # In Lua, card:get_chip_bonus() is added here. Our hand_evaluator already
            # adds scoring card chip values to base_chips, so we don't double-count.
            # Actually, let me reconsider: in Lua the base chips/mult come from the
            # hand type table, and THEN each scoring card's chip_bonus is added
            # individually. Our hand_evaluator adds them in base_chips. This is
            # equivalent for the simple case (no enhanced cards).

            # d. Joker individual effects per scoring card
            for joker in self.jokers:
                mod = joker.on_individual(card, hand_result, scoring_hand,
                                          scoring_name, view)
                if mod:
                    if mod.chips:
                        chips += mod.chips
                    if mod.mult:
                        mult += mod.mult
                    if mod.x_mult > 0:
                        mult = int(mult * mod.x_mult)
                    if mod.dollars:
                        # Probabilistic money (e.g., Business Card: 1 in 2 chance)
                        if self.rng.random() < 0.5:
                            self.money += mod.dollars

        # 4. Held card effects
        for card in held_cards:
            if card.face_down:
                continue

            for joker in self.jokers:
                mod = joker.on_held_individual(card, hand_result, scoring_hand,
                                               scoring_name, view)
                if mod:
                    if mod.mult:
                        mult += mod.mult
                    if mod.x_mult > 0:
                        mult = int(mult * mod.x_mult)

        # 5. Main joker effects (left to right)
        for joker in self.jokers:
            mod = joker.on_main(hand_result, scoring_hand, full_hand,
                                poker_hands, scoring_name, view)
            if mod:
                if mod.chip_mod:
                    chips += mod.chip_mod
                if mod.mult_mod:
                    mult += mod.mult_mod
                if mod.Xmult_mod > 0:
                    mult = int(mult * mod.Xmult_mod)

        # Joker "after" phase (Ice Cream loses chips, etc.)
        for joker in self.jokers:
            joker.on_after(hand_result, scoring_hand, scoring_name, view)

        # 6. Final score
        return max(0, chips * mult)

    # -------------------------------------------------------------------
    # Side effects processing
    # -------------------------------------------------------------------

    def _process_side_effects(self, effects: dict[str, Any]) -> None:
        """Process side effects returned by joker hooks."""
        if not effects:
            return

        if "money" in effects:
            self.money += effects["money"]

    # -------------------------------------------------------------------
    # Observation
    # -------------------------------------------------------------------

    def get_view(self) -> GameStateSnapshot:
        """Create a read-only snapshot of the current game state."""
        return GameStateSnapshot(
            hand=list(self.hand),
            jokers=list(self.jokers),
            money=self.money,
            ante=self.ante,
            blind_type=self.current_blind_type.value,
            score_target=self.score_target,
            current_score=self.current_score,
            hands_remaining=self.hands_remaining,
            discards_remaining=self.discards_remaining,
            deck_size=self.deck.cards_remaining,
            max_jokers=self.max_jokers,
            hands_played_this_round=self.hands_played_this_round,
            hand_type_played_counts=dict(self.hand_type_played_counts),
        )

    def get_valid_actions(self) -> dict[str, Any]:
        """Return valid actions for the current phase."""
        if self.phase == GamePhase.PLAY:
            return {
                "can_play": self.hands_remaining > 0,
                "can_discard": self.discards_remaining > 0,
                "hand_size": len(self.hand),
                "max_play": min(5, len(self.hand)),
            }
        elif self.phase == GamePhase.SHOP:
            available = self.shop.get_available_offerings()
            return {
                "can_buy": [
                    (i, o.joker.INFO.name, o.cost, o.cost <= self.money
                     and len(self.jokers) < self.max_jokers)
                    for i, o in available
                ],
                "can_sell": [(i, j.INFO.name, self.shop.sell_value(j))
                             for i, j in enumerate(self.jokers)],
                "can_reroll": self.money >= self.shop.reroll_cost,
                "reroll_cost": self.shop.reroll_cost,
                "can_skip": True,
            }
        else:
            return {}
