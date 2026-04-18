"""Integration tests: consumables through game state, enhanced scoring, wild cards, 100 random games."""

import numpy as np
import pytest

from balatro_gym.core.card import Card, Rank, Suit, Enhancement, Edition, Seal
from balatro_gym.core.game_state import GamePhase, GameState
from balatro_gym.core.hand_evaluator import HandType, evaluate_hand
from balatro_gym.core.consumable import create_consumable, ConsumableType
from balatro_gym.envs.configs import GameConfig
from balatro_gym.envs.balatro_env import BalatroEnv


class TestWildCardFlush:
    def test_wild_card_completes_flush(self):
        """A Wild card should count as any suit to complete a flush."""
        cards = [
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.FIVE, Suit.HEARTS),
            Card(Rank.SEVEN, Suit.HEARTS),
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.KING, Suit.CLUBS, enhancement=Enhancement.WILD),
        ]
        result = evaluate_hand(cards, [])
        assert result.hand_type == HandType.FLUSH

    def test_all_wild_is_flush(self):
        cards = [
            Card(Rank.TWO, Suit.HEARTS, enhancement=Enhancement.WILD),
            Card(Rank.FIVE, Suit.DIAMONDS, enhancement=Enhancement.WILD),
            Card(Rank.SEVEN, Suit.CLUBS, enhancement=Enhancement.WILD),
            Card(Rank.NINE, Suit.SPADES, enhancement=Enhancement.WILD),
            Card(Rank.KING, Suit.HEARTS, enhancement=Enhancement.WILD),
        ]
        result = evaluate_hand(cards, [])
        assert result.hand_type == HandType.FLUSH

    def test_non_wild_mixed_suits_no_flush(self):
        cards = [
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.FIVE, Suit.HEARTS),
            Card(Rank.SEVEN, Suit.CLUBS),  # not wild, different suit
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
        ]
        result = evaluate_hand(cards, [])
        assert result.hand_type != HandType.FLUSH


class TestEnhancedScoring:
    def test_bonus_card_adds_chips(self):
        """Bonus enhancement should add +30 chips in scoring pipeline."""
        gs = GameState(seed=42, available_joker_ids=[])
        gs.reset()
        gs.hand = [
            Card(Rank.ACE, Suit.HEARTS, enhancement=Enhancement.BONUS),
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.SIX, Suit.CLUBS),
            Card(Rank.SEVEN, Suit.DIAMONDS),
        ]
        score, result = gs.play_hand([0, 1])
        # Pair base: chips=10, mult=2
        # Scoring cards: Bonus Ace = 11+30 = 41, normal Ace = 11
        # Total chips = 10 + 41 + 11 = 62. Score = 62 * 2 = 124
        assert result.hand_type == HandType.PAIR
        assert score == 124

    def test_foil_edition_adds_chips(self):
        """Foil edition adds +50 chips per scoring card."""
        gs = GameState(seed=42, available_joker_ids=[])
        gs.reset()
        gs.hand = [
            Card(Rank.ACE, Suit.HEARTS, edition=Edition.FOIL),
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.SIX, Suit.CLUBS),
            Card(Rank.SEVEN, Suit.DIAMONDS),
        ]
        score, result = gs.play_hand([0, 1])
        # Pair base: chips=10, mult=2
        # Foil Ace: 11 + 50 (edition) = 61 chip contribution
        # Normal Ace: 11
        # Total chips = 10 + 61 + 11 = 82. Score = 82 * 2 = 164
        assert score == 164

    def test_holo_edition_adds_mult(self):
        """Holo edition adds +10 mult per scoring card."""
        gs = GameState(seed=42, available_joker_ids=[])
        gs.reset()
        gs.hand = [
            Card(Rank.ACE, Suit.HEARTS, edition=Edition.HOLO),
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.SIX, Suit.CLUBS),
            Card(Rank.SEVEN, Suit.DIAMONDS),
        ]
        score, result = gs.play_hand([0, 1])
        # Pair base: chips=10, mult=2
        # Holo Ace: chips+=11, mult+=10 -> chips=21, mult=12
        # Normal Ace: chips+=11 -> chips=32, mult=12
        # Score = 32 * 12 = 384
        assert score == 384

    def test_glass_card_x_mult(self):
        """Glass enhancement gives X2 mult."""
        gs = GameState(seed=100, available_joker_ids=[])
        gs.reset()
        gs.hand = [
            Card(Rank.ACE, Suit.HEARTS, enhancement=Enhancement.GLASS),
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.SIX, Suit.CLUBS),
            Card(Rank.SEVEN, Suit.DIAMONDS),
        ]
        score, result = gs.play_hand([0, 1])
        # Pair base: chips=10, mult=2
        # Glass Ace: chips+=11, mult*=2 -> chips=21, mult=4
        # Normal Ace: chips+=11 -> chips=32, mult=4
        # Score = 32 * 4 = 128
        assert score == 128

    def test_stone_card_50_chips_only(self):
        """Stone card: +50 chips, no nominal."""
        gs = GameState(seed=42, available_joker_ids=[])
        gs.reset()
        gs.hand = [
            Card(Rank.ACE, Suit.HEARTS, enhancement=Enhancement.STONE),
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.SIX, Suit.CLUBS),
            Card(Rank.SEVEN, Suit.DIAMONDS),
        ]
        score, result = gs.play_hand([0, 1])
        # Pair base: chips=10, mult=2
        # Stone Ace: chip_value=50 (no nominal)
        # Normal Ace: 11
        # Total chips = 10 + 50 + 11 = 71. Score = 71 * 2 = 142
        assert score == 142

    def test_gold_seal_gives_money(self):
        """Gold Seal should give $3 when played."""
        gs = GameState(seed=42, available_joker_ids=[])
        gs.reset()
        starting_money = gs.money
        gs.hand = [
            Card(Rank.ACE, Suit.HEARTS, seal=Seal.GOLD),
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.SIX, Suit.CLUBS),
            Card(Rank.SEVEN, Suit.DIAMONDS),
        ]
        gs.play_hand([0, 1])
        # Gold Seal gives $3 during scoring
        assert gs.money >= starting_money + 3

    def test_red_seal_retrigger(self):
        """Red Seal should make a card score twice."""
        gs = GameState(seed=42, available_joker_ids=[])
        gs.reset()
        gs.hand = [
            Card(Rank.ACE, Suit.HEARTS, seal=Seal.RED),
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.SIX, Suit.CLUBS),
            Card(Rank.SEVEN, Suit.DIAMONDS),
        ]
        score, result = gs.play_hand([0, 1])
        # Pair base: chips=10, mult=2
        # Red Seal Ace: 11 chips twice = 22 chip contribution
        # Normal Ace: 11
        # Total chips = 10 + 22 + 11 = 43. Score = 43 * 2 = 86
        assert score == 86

    def test_steel_card_held_x_mult(self):
        """Steel Card in hand (not played) should give X1.5 mult."""
        gs = GameState(seed=42, available_joker_ids=[])
        gs.reset()
        gs.hand = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.TWO, Suit.CLUBS, enhancement=Enhancement.STEEL),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.SIX, Suit.CLUBS),
            Card(Rank.SEVEN, Suit.DIAMONDS),
        ]
        score, result = gs.play_hand([0, 1])
        # Pair base: chips=10, mult=2
        # Aces: 11+11=22. chips=32, mult=2
        # Held Steel card: mult *= 1.5 -> mult=3
        # Score = 32 * 3 = 96
        assert score == 96


class TestHandLeveling:
    def test_planet_card_increases_score(self):
        """Using a planet card should increase hand type scoring."""
        gs = GameState(seed=42, available_joker_ids=[])
        gs.reset()

        # Level up Pair
        gs.consumables = [create_consumable("c_mercury")]
        gs.use_consumable(0, [])

        # Pair should now be level 2: chips=25, mult=3
        gs.hand = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.SIX, Suit.CLUBS),
            Card(Rank.SEVEN, Suit.DIAMONDS),
        ]
        score, result = gs.play_hand([0, 1])
        # Level 2 Pair: chips=25, mult=3
        # Aces: 11+11=22. Total chips = 25+22 = 47
        # Score = 47 * 3 = 141
        assert score == 141


class TestConsumableUseThroughGameState:
    def test_use_tarot_enhancement(self):
        """Using The Tower (Stone) through GameState should enhance card."""
        gs = GameState(seed=42, available_joker_ids=[])
        gs.reset()
        gs.hand = [Card(Rank.FIVE, Suit.HEARTS)] * 8  # Simple hand

        gs.consumables = [create_consumable("c_tower")]
        gs.use_consumable(0, [0])
        assert gs.hand[0].enhancement == Enhancement.STONE
        assert len(gs.consumables) == 0  # consumed

    def test_use_planet_through_state(self):
        """Planet card through GameState should level up hand."""
        gs = GameState(seed=42, available_joker_ids=[])
        gs.reset()

        gs.consumables = [create_consumable("c_jupiter")]
        gs.use_consumable(0, [])
        chips, mult = gs.hand_levels.get_score(HandType.FLUSH)
        assert chips == 50   # 35 + 15
        assert mult == 6     # 4 + 2

    def test_use_consumable_invalid_slot_raises(self):
        gs = GameState(seed=42, available_joker_ids=[])
        gs.reset()
        with pytest.raises(ValueError):
            gs.use_consumable(0, [])

    def test_use_consumable_wrong_targets_raises(self):
        gs = GameState(seed=42, available_joker_ids=[])
        gs.reset()
        gs.hand = [Card(Rank.FIVE, Suit.HEARTS)] * 8
        gs.consumables = [create_consumable("c_tower")]  # needs 1 highlighted
        with pytest.raises(ValueError):
            gs.use_consumable(0, [])  # 0 highlighted

    def test_cryptid_grows_deck(self):
        """Cryptid should add 2 copies of highlighted card to deck."""
        gs = GameState(seed=42, available_joker_ids=[])
        gs.reset()
        initial_deck = gs.deck.card_count
        gs.consumables = [create_consumable("c_cryptid")]
        gs.use_consumable(0, [0])
        assert gs.deck.card_count == initial_deck + 2

    def test_hanged_man_shrinks_hand(self):
        """Hanged Man should destroy selected cards."""
        gs = GameState(seed=42, available_joker_ids=[])
        gs.reset()
        initial_hand = len(gs.hand)
        gs.consumables = [create_consumable("c_hanged_man")]
        gs.use_consumable(0, [0])
        assert len(gs.hand) == initial_hand - 1

    def test_fool_creates_consumable(self):
        """Fool should create a random Tarot or Planet in consumable slots."""
        gs = GameState(seed=42, available_joker_ids=[])
        gs.reset()
        gs.consumables = [create_consumable("c_fool")]
        gs.use_consumable(0, [])
        # Should have created 1 new consumable
        assert len(gs.consumables) == 1

    def test_judgement_creates_joker(self):
        """Judgement should create a random joker."""
        gs = GameState(seed=42, available_joker_ids=["joker_basic"])
        gs.reset()
        initial_jokers = len(gs.jokers)
        gs.consumables = [create_consumable("c_judgement")]
        gs.use_consumable(0, [])
        assert len(gs.jokers) == initial_jokers + 1


class TestPurpleSealOnDiscard:
    def test_purple_seal_creates_tarot(self):
        """Discarding a Purple Seal card should create a random Tarot."""
        gs = GameState(seed=42, available_joker_ids=[])
        gs.reset()
        gs.hand = [
            Card(Rank.FIVE, Suit.HEARTS, seal=Seal.PURPLE),
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
            Card(Rank.SIX, Suit.SPADES),
            Card(Rank.SEVEN, Suit.CLUBS),
            Card(Rank.EIGHT, Suit.DIAMONDS),
            Card(Rank.NINE, Suit.SPADES),
        ]
        assert len(gs.consumables) == 0
        gs.discard([0])
        assert len(gs.consumables) == 1


class TestShopConsumables:
    def test_shop_generates_consumable_offerings(self):
        """Shop with consumable_pool should generate consumable offerings."""
        gs = GameState(
            seed=42,
            available_joker_ids=["joker_basic"],
            available_consumable_ids=["c_mercury", "c_pluto"],
        )
        gs.reset()
        # Force to shop
        gs.current_score = gs.score_target
        gs._end_blind()
        if gs.phase == GamePhase.SHOP:
            offerings = gs.shop.get_available_offerings()
            types = [o.item_type for _, o in offerings]
            assert "consumable" in types

    def test_buy_consumable_from_shop(self):
        """Buying a consumable from shop should add it to consumables list."""
        gs = GameState(
            seed=42,
            available_joker_ids=["joker_basic"],
            available_consumable_ids=["c_mercury"],
        )
        gs.reset()
        gs.money = 100
        gs.current_score = gs.score_target
        gs._end_blind()
        if gs.phase == GamePhase.SHOP:
            offerings = gs.shop.get_available_offerings()
            consumable_slots = [
                (i, o) for i, o in offerings if o.item_type == "consumable"
            ]
            if consumable_slots:
                idx, _ = consumable_slots[0]
                assert gs.shop_buy(idx)
                assert len(gs.consumables) == 1


class TestRandomGamesWithConsumables:
    def test_100_random_games_with_consumables(self):
        """100 random games with consumables enabled should not crash."""
        for seed in range(100):
            config = GameConfig.easy(seed=seed)
            env = BalatroEnv(config=config)
            obs, info = env.reset(seed=seed)
            rng = np.random.default_rng(seed)

            for _ in range(2000):
                mask = info["action_mask"]
                valid = np.where(mask)[0]
                if len(valid) == 0:
                    break
                obs, _, term, _, info = env.step(int(rng.choice(valid)))
                if term:
                    break

    def test_50_medium_games_no_crashes(self):
        """50 medium-difficulty games with full consumable pool."""
        for seed in range(50):
            config = GameConfig.medium(seed=seed)
            env = BalatroEnv(config=config)
            obs, info = env.reset(seed=seed)
            rng = np.random.default_rng(seed)

            for _ in range(3000):
                mask = info["action_mask"]
                valid = np.where(mask)[0]
                if len(valid) == 0:
                    break
                obs, _, term, _, info = env.step(int(rng.choice(valid)))
                if term:
                    break

    def test_20_hard_games_no_crashes(self):
        """20 hard-difficulty games with all consumables + spectrals."""
        for seed in range(20):
            config = GameConfig.hard(seed=seed)
            env = BalatroEnv(config=config)
            obs, info = env.reset(seed=seed)
            rng = np.random.default_rng(seed)

            for _ in range(4000):
                mask = info["action_mask"]
                valid = np.where(mask)[0]
                if len(valid) == 0:
                    break
                obs, _, term, _, info = env.step(int(rng.choice(valid)))
                if term:
                    break
