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

from balatro_gym.core.card import Card, Deck, Rank, Enhancement, Seal
from balatro_gym.core.hand_evaluator import HandResult, HandType, evaluate_hand
from balatro_gym.core.hand_levels import HandLevelManager
from balatro_gym.core.joker import BaseJoker, ScoreModification, create_joker
from balatro_gym.core.consumable import (
    BaseConsumable, ConsumableType, create_consumable, get_consumables_by_type,
)
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
    consumables: list[BaseConsumable] = field(default_factory=list)
    consumable_slots: int = 2


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
        available_consumable_ids: list[str] | None = None,
        shop_slots: int = 2,
        reroll_base_cost: int = 5,
        consumable_slots: int = 2,
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
        self.available_consumable_ids = available_consumable_ids or []
        self.shop_slots = shop_slots
        self.reroll_base_cost = reroll_base_cost
        self.consumable_slots = consumable_slots

        # RNG
        self.rng = np.random.default_rng(seed)

        # Game objects
        self.deck = Deck(self.rng)
        self.blind_manager = BlindManager(num_antes)
        self.hand_levels = HandLevelManager()
        self.shop = Shop(
            joker_pool=self.available_joker_ids,
            rng=self.rng,
            num_slots=shop_slots,
            reroll_base_cost=reroll_base_cost,
            consumable_pool=self.available_consumable_ids,
        )

        # Mutable state (set in reset)
        self.hand: list[Card] = []
        self.jokers: list[BaseJoker] = []
        self.consumables: list[BaseConsumable] = []
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
        self.consumables = []
        self.hand_levels.reset()
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

        # Blue Seal: for each card in hand with Blue Seal, create planet card
        for card in self.hand:
            if card.seal == Seal.BLUE and not card.face_down:
                if len(self.consumables) < self.consumable_slots:
                    planet_pool = get_consumables_by_type(ConsumableType.PLANET)
                    if planet_pool:
                        chosen = str(self.rng.choice(planet_pool))
                        self.consumables.append(create_consumable(chosen))

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

        # Purple Seal: create a random Tarot if consumable slot available
        for card in discarded:
            if card.seal == Seal.PURPLE and not card.face_down:
                if len(self.consumables) < self.consumable_slots:
                    tarot_pool = get_consumables_by_type(ConsumableType.TAROT)
                    if tarot_pool:
                        chosen = str(self.rng.choice(tarot_pool))
                        self.consumables.append(create_consumable(chosen))

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
        """Buy a joker or consumable from the shop. Returns True if successful."""
        if self.phase != GamePhase.SHOP:
            raise ValueError(f"Cannot buy in phase {self.phase}")

        item, remaining = self.shop.buy_item(
            slot_index, self.money, self.jokers, self.max_jokers,
            self.consumables, self.consumable_slots,
        )
        if item is not None:
            if isinstance(item, BaseJoker):
                self.jokers.append(item)
            elif isinstance(item, BaseConsumable):
                self.consumables.append(item)
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
        1. Hand-type base chips/mult from hand_levels (NOT from hand_evaluator)
        2. Blind modify_hand (The Flint halves them)
        3. Joker on_before
        4. Per scoring card (left to right):
           a. Skip if debuffed
           b. card.chip_value -> add to chips (includes nominal + enhancement bonus)
           c. card.get_chip_mult(rng) -> add to mult (Mult +4, Lucky 1/5 +20)
           d. card.get_chip_x_mult() -> multiply mult (Glass X2)
           e. Card edition: Foil +50 chips, Holo +10 mult, Polychrome X1.5 mult
           f. Gold Seal: +$3
           g. Joker on_individual per card
           h. If Red Seal: repeat steps b-g once (retrigger)
        5. Per held card:
           a. Skip if debuffed
           b. card.get_held_x_mult() -> multiply mult (Steel X1.5)
           c. Card edition effects (if held card has edition)
           d. Joker on_held_individual per card
           e. If Red Seal: repeat steps b-d once
        6. Per joker (main, left to right):
           a. Joker on_main (chip_mod, mult_mod, Xmult_mod)
        7. Joker on_after
        8. Glass Card shatter check (1/4 chance to destroy)
        9. Gold Card held dollars ($3 per held Gold Card)
        10. Score = max(0, chips * mult)
        """
        # 1. Hand-type base chips/mult from hand levels
        level_chips, level_mult = self.hand_levels.get_score(hand_result.hand_type)
        chips = level_chips
        mult = level_mult
        view = self.get_view()

        # 2. Blind modify_hand (e.g., The Flint)
        if self.active_boss_effect:
            new_mult, new_chips, modified = self.active_boss_effect.modify_hand(mult, chips)
            mult = new_mult
            chips = new_chips

        # 3. Joker "before" phase (stat updates like Runner, Ride the Bus)
        for joker in self.jokers:
            joker.on_before(hand_result, scoring_hand, full_hand,
                            poker_hands, scoring_name, view)

        # 4. Score each scoring card individually
        for card in scoring_hand:
            if card.face_down:
                continue

            retriggers = 1
            if card.seal == Seal.RED:
                retriggers = 2

            for _ in range(retriggers):
                # b. Card chip bonus (nominal + enhancement)
                chips += card.chip_value
                # c. Card mult bonus (Mult Card +4, Lucky 1/5 +20)
                chips_mult_add = card.get_chip_mult(self.rng)
                if chips_mult_add:
                    mult += chips_mult_add
                # d. Card x_mult (Glass X2)
                card_x = card.get_chip_x_mult()
                if card_x > 0:
                    mult = int(mult * card_x)
                # e. Card edition bonus
                ed_chips, ed_mult, ed_x = card.get_edition_bonus()
                if ed_chips:
                    chips += ed_chips
                if ed_mult:
                    mult += ed_mult
                if ed_x > 0:
                    mult = int(mult * ed_x)
                # f. Gold Seal: +$3
                played_dollars = card.get_played_dollars()
                if played_dollars:
                    self.money += played_dollars
                # g. Joker individual effects per scoring card
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
                            self.money += mod.dollars

        # 5. Held card effects
        for card in held_cards:
            if card.face_down:
                continue

            retriggers = 1
            if card.seal == Seal.RED:
                retriggers = 2

            for _ in range(retriggers):
                # b. Steel Card X1.5
                held_x = card.get_held_x_mult()
                if held_x > 0:
                    mult = int(mult * held_x)
                # c. Held card edition
                ed_chips, ed_mult, ed_x = card.get_edition_bonus()
                if ed_chips:
                    chips += ed_chips
                if ed_mult:
                    mult += ed_mult
                if ed_x > 0:
                    mult = int(mult * ed_x)
                # d. Joker held-individual effects
                for joker in self.jokers:
                    mod = joker.on_held_individual(card, hand_result, scoring_hand,
                                                   scoring_name, view)
                    if mod:
                        if mod.mult:
                            mult += mod.mult
                        if mod.x_mult > 0:
                            mult = int(mult * mod.x_mult)

        # 6. Main joker effects (left to right)
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

        # 7. Joker "after" phase (Ice Cream loses chips, etc.)
        for joker in self.jokers:
            joker.on_after(hand_result, scoring_hand, scoring_name, view)

        # 8. Glass Card shatter (1/4 chance for each Glass scoring card)
        for card in scoring_hand:
            if card.enhancement == Enhancement.GLASS and not card.face_down:
                if self.rng.random() < 0.25:
                    self.deck.remove_card(card)

        # 9. Gold Card held dollars ($3 per held Gold Card at end of round —
        #    actually this happens in economy, not scoring. But we track it here
        #    since held_cards is available.)
        for card in held_cards:
            if not card.face_down:
                held_dollars = card.get_held_dollars()
                if held_dollars:
                    self.money += held_dollars

        # 10. Final score
        return max(0, chips * mult)

    # -------------------------------------------------------------------
    # Side effects processing
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------
    # Consumable actions
    # -------------------------------------------------------------------

    def use_consumable(self, slot_index: int,
                       highlighted_indices: list[int] | None = None) -> dict[str, Any]:
        """Use a consumable from the player's consumable slots.

        Args:
            slot_index: Index into self.consumables.
            highlighted_indices: Indices into self.hand of targeted cards.

        Returns:
            Side effects dict from the consumable.
        """
        if slot_index < 0 or slot_index >= len(self.consumables):
            raise ValueError(f"Invalid consumable slot: {slot_index}")

        highlighted = highlighted_indices or []
        consumable = self.consumables[slot_index]

        if not consumable.can_use(self, highlighted):
            raise ValueError(f"Cannot use {consumable.INFO.name} with {len(highlighted)} highlighted cards")

        # Remove consumable before use (it's consumed)
        self.consumables.pop(slot_index)

        # Apply effect
        result = consumable.use(self, highlighted)

        # Process side effects
        self._process_side_effects(result)

        return result

    # -------------------------------------------------------------------
    # Side effects processing
    # -------------------------------------------------------------------

    def _process_side_effects(self, effects: dict[str, Any]) -> None:
        """Process side effects returned by joker/consumable hooks."""
        if not effects:
            return

        if "money" in effects:
            # Money already applied by the consumable in most cases;
            # only apply here for joker hooks that return money
            pass

        if "create_consumable" in effects:
            for cid in effects["create_consumable"]:
                if len(self.consumables) < self.consumable_slots:
                    self.consumables.append(create_consumable(cid))

        if "create_joker" in effects:
            if len(self.jokers) < self.max_jokers:
                self.jokers.append(create_joker(effects["create_joker"]))

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
            consumables=list(self.consumables),
            consumable_slots=self.consumable_slots,
        )

    def get_valid_actions(self) -> dict[str, Any]:
        """Return valid actions for the current phase."""
        if self.phase == GamePhase.PLAY:
            return {
                "can_play": self.hands_remaining > 0,
                "can_discard": self.discards_remaining > 0,
                "hand_size": len(self.hand),
                "max_play": min(5, len(self.hand)),
                "consumables": [
                    (i, c.INFO.name, c.INFO.min_highlighted, c.INFO.max_highlighted)
                    for i, c in enumerate(self.consumables)
                ],
            }
        elif self.phase == GamePhase.SHOP:
            available = self.shop.get_available_offerings()
            can_buy = []
            for i, o in available:
                if o.item_type == "joker":
                    affordable = (o.cost <= self.money
                                  and len(self.jokers) < self.max_jokers)
                else:
                    affordable = (o.cost <= self.money
                                  and len(self.consumables) < self.consumable_slots)
                can_buy.append((i, o.name, o.cost, affordable, o.item_type))
            return {
                "can_buy": can_buy,
                "can_sell": [(i, j.INFO.name, self.shop.sell_value(j))
                             for i, j in enumerate(self.jokers)],
                "can_reroll": self.money >= self.shop.reroll_cost,
                "reroll_cost": self.shop.reroll_cost,
                "can_skip": True,
            }
        else:
            return {}
