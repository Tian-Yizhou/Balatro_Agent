"""Tests for consumable.py: Tarot, Planet, and Spectral cards."""

import numpy as np
import pytest

from balatro_gym.core.card import Card, Rank, Suit, Enhancement, Edition, Seal, Deck
from balatro_gym.core.hand_evaluator import HandType
from balatro_gym.core.hand_levels import HandLevelManager
from balatro_gym.core.consumable import (
    ConsumableType, BaseConsumable,
    get_all_consumable_ids, get_consumables_by_type, create_consumable,
)


# ---------------------------------------------------------------------------
# Helper: minimal game state mock for consumable.use()
# ---------------------------------------------------------------------------

class MockGameState:
    """Minimal game state mock satisfying consumable use() requirements."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.deck = Deck(self.rng)
        self.deck.reset()
        self.hand: list[Card] = self.deck.draw(8)
        self.jokers: list = []
        self.max_jokers: int = 5
        self.money: int = 10
        self.consumables: list[BaseConsumable] = []
        self.consumable_slots: int = 2
        self.hand_levels = HandLevelManager()
        self.available_joker_ids: list[str] = ["joker_basic"]

    def _set_hand(self, cards: list[Card]) -> None:
        """Replace hand with specific cards (for controlled testing)."""
        self.hand = cards


class TestRegistry:
    def test_total_consumable_count(self):
        assert len(get_all_consumable_ids()) == 44

    def test_planet_count(self):
        assert len(get_consumables_by_type(ConsumableType.PLANET)) == 12

    def test_tarot_count(self):
        assert len(get_consumables_by_type(ConsumableType.TAROT)) == 22

    def test_spectral_count(self):
        assert len(get_consumables_by_type(ConsumableType.SPECTRAL)) == 10

    def test_create_all(self):
        for cid in get_all_consumable_ids():
            c = create_consumable(cid)
            assert c.INFO.id == cid

    def test_unknown_id_raises(self):
        with pytest.raises(KeyError):
            create_consumable("c_nonexistent")


class TestPlanetCards:
    def test_mercury_levels_pair(self):
        gs = MockGameState()
        planet = create_consumable("c_mercury")
        assert planet.can_use(gs, [])
        planet.use(gs, [])
        chips, mult = gs.hand_levels.get_score(HandType.PAIR)
        assert chips == 25   # 10 + 15
        assert mult == 3     # 2 + 1

    def test_pluto_levels_high_card(self):
        gs = MockGameState()
        create_consumable("c_pluto").use(gs, [])
        chips, mult = gs.hand_levels.get_score(HandType.HIGH_CARD)
        assert chips == 15   # 5 + 10
        assert mult == 2     # 1 + 1

    def test_jupiter_levels_flush(self):
        gs = MockGameState()
        create_consumable("c_jupiter").use(gs, [])
        chips, mult = gs.hand_levels.get_score(HandType.FLUSH)
        assert chips == 50   # 35 + 15
        assert mult == 6     # 4 + 2

    def test_planet_cannot_use_with_highlighted(self):
        gs = MockGameState()
        planet = create_consumable("c_mercury")
        assert not planet.can_use(gs, [0])

    def test_all_planets_level_correct_hand(self):
        expected = {
            "c_mercury":  HandType.PAIR,
            "c_venus":    HandType.THREE_OF_A_KIND,
            "c_earth":    HandType.FULL_HOUSE,
            "c_mars":     HandType.FOUR_OF_A_KIND,
            "c_jupiter":  HandType.FLUSH,
            "c_saturn":   HandType.STRAIGHT,
            "c_uranus":   HandType.TWO_PAIR,
            "c_neptune":  HandType.STRAIGHT_FLUSH,
            "c_pluto":    HandType.HIGH_CARD,
            "c_ceres":    HandType.FLUSH_FIVE,
            "c_planet_x": HandType.FLUSH_HOUSE,
            "c_eris":     HandType.FIVE_OF_A_KIND,
        }
        for pid, ht in expected.items():
            gs = MockGameState()
            create_consumable(pid).use(gs, [])
            level = gs.hand_levels.get_level(ht)
            assert level.level == 2, f"{pid} should level up {ht.name}"

    def test_double_level_up(self):
        gs = MockGameState()
        create_consumable("c_mercury").use(gs, [])
        create_consumable("c_mercury").use(gs, [])
        level = gs.hand_levels.get_level(HandType.PAIR)
        assert level.level == 3


class TestEnhancementTarots:
    def test_magician_adds_lucky(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.ACE, Suit.HEARTS)])
        create_consumable("c_magician").use(gs, [0])
        assert gs.hand[0].enhancement == Enhancement.LUCKY

    def test_empress_adds_mult_to_two(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.SPADES)])
        create_consumable("c_empress").use(gs, [0, 1])
        assert gs.hand[0].enhancement == Enhancement.MULT
        assert gs.hand[1].enhancement == Enhancement.MULT

    def test_hierophant_adds_bonus(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.FIVE, Suit.HEARTS)])
        create_consumable("c_hierophant").use(gs, [0])
        assert gs.hand[0].enhancement == Enhancement.BONUS

    def test_lovers_adds_wild(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.FIVE, Suit.HEARTS)])
        create_consumable("c_lovers").use(gs, [0])
        assert gs.hand[0].enhancement == Enhancement.WILD

    def test_chariot_adds_steel(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.FIVE, Suit.HEARTS)])
        create_consumable("c_chariot").use(gs, [0])
        assert gs.hand[0].enhancement == Enhancement.STEEL

    def test_justice_adds_glass(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.FIVE, Suit.HEARTS)])
        create_consumable("c_justice").use(gs, [0])
        assert gs.hand[0].enhancement == Enhancement.GLASS

    def test_devil_adds_gold(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.FIVE, Suit.HEARTS)])
        create_consumable("c_devil").use(gs, [0])
        assert gs.hand[0].enhancement == Enhancement.GOLD

    def test_tower_adds_stone(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.FIVE, Suit.HEARTS)])
        create_consumable("c_tower").use(gs, [0])
        assert gs.hand[0].enhancement == Enhancement.STONE

    def test_cannot_use_with_no_cards(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.FIVE, Suit.HEARTS)])
        tarot = create_consumable("c_magician")
        assert not tarot.can_use(gs, [])  # min_highlighted = 1

    def test_cannot_exceed_max_highlighted(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.FIVE, Suit.HEARTS)] * 3)
        tarot = create_consumable("c_magician")  # max = 1
        assert not tarot.can_use(gs, [0, 1])


class TestSuitTarots:
    def test_star_converts_to_diamonds(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.ACE, Suit.HEARTS)])
        create_consumable("c_star").use(gs, [0])
        assert gs.hand[0].suit == Suit.DIAMONDS

    def test_moon_converts_to_clubs(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.ACE, Suit.HEARTS)])
        create_consumable("c_moon").use(gs, [0])
        assert gs.hand[0].suit == Suit.CLUBS

    def test_sun_converts_to_hearts(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.ACE, Suit.SPADES)])
        create_consumable("c_sun").use(gs, [0])
        assert gs.hand[0].suit == Suit.HEARTS

    def test_world_converts_to_spades(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.ACE, Suit.HEARTS)])
        create_consumable("c_world").use(gs, [0])
        assert gs.hand[0].suit == Suit.SPADES

    def test_converts_multiple(self):
        gs = MockGameState()
        gs._set_hand([
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.CLUBS),
            Card(Rank.QUEEN, Suit.DIAMONDS),
        ])
        create_consumable("c_star").use(gs, [0, 1, 2])
        assert all(c.suit == Suit.DIAMONDS for c in gs.hand)


class TestSpecialTarots:
    def test_strength_increases_rank(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.FIVE, Suit.HEARTS)])
        create_consumable("c_strength").use(gs, [0])
        assert gs.hand[0].rank == Rank.SIX

    def test_strength_ace_wraps_to_two(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.ACE, Suit.HEARTS)])
        create_consumable("c_strength").use(gs, [0])
        assert gs.hand[0].rank == Rank.TWO

    def test_hanged_man_destroys_cards(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.SPADES)])
        initial_deck_size = gs.deck.card_count
        result = create_consumable("c_hanged_man").use(gs, [0])
        assert len(gs.hand) == 1
        assert gs.hand[0].rank == Rank.KING
        assert "destroy_cards" in result

    def test_death_copies_card(self):
        gs = MockGameState()
        gs._set_hand([
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.ACE, Suit.SPADES, enhancement=Enhancement.GOLD),
        ])
        create_consumable("c_death").use(gs, [0, 1])
        # Left card (index 0) should now be a copy of right card (index 1)
        assert gs.hand[0].rank == Rank.ACE
        assert gs.hand[0].suit == Suit.SPADES
        assert gs.hand[0].enhancement == Enhancement.GOLD

    def test_hermit_doubles_money_capped(self):
        gs = MockGameState()
        gs.money = 15
        create_consumable("c_hermit").use(gs, [])
        assert gs.money == 30  # 15 + min(15, 20) = 30

    def test_hermit_cap_at_20(self):
        gs = MockGameState()
        gs.money = 50
        create_consumable("c_hermit").use(gs, [])
        assert gs.money == 70  # 50 + min(50, 20) = 70

    def test_temperance_gains_joker_values(self):
        gs = MockGameState()
        from balatro_gym.core.joker import create_joker
        gs.jokers = [create_joker("joker_basic")]  # cost=2, sell=1
        gs.money = 0
        create_consumable("c_temperance").use(gs, [])
        assert gs.money == 1  # max(1, 2//2) = 1

    def test_fool_creates_consumable(self):
        gs = MockGameState()
        result = create_consumable("c_fool").use(gs, [])
        assert "create_consumable" in result
        assert len(result["create_consumable"]) == 1

    def test_high_priestess_creates_planets(self):
        gs = MockGameState()
        result = create_consumable("c_high_priestess").use(gs, [])
        assert "create_consumable" in result
        assert len(result["create_consumable"]) <= 2

    def test_emperor_creates_tarots(self):
        gs = MockGameState()
        result = create_consumable("c_emperor").use(gs, [])
        assert "create_consumable" in result
        assert len(result["create_consumable"]) <= 2

    def test_judgement_creates_joker(self):
        gs = MockGameState()
        result = create_consumable("c_judgement").use(gs, [])
        assert "create_joker" in result

    def test_judgement_cannot_use_full_jokers(self):
        gs = MockGameState()
        gs.jokers = [None] * gs.max_jokers  # fill slots
        j = create_consumable("c_judgement")
        assert not j.can_use(gs, [])


class TestSpectralCards:
    def test_familiar_destroys_and_creates(self):
        gs = MockGameState()
        hand_size = len(gs.hand)
        result = create_consumable("c_familiar").use(gs, [])
        assert len(gs.hand) == hand_size - 1
        assert "create_cards" in result
        assert len(result["create_cards"]) == 3
        # Created cards should be face cards
        for card in result["create_cards"]:
            assert card.rank in (Rank.JACK, Rank.QUEEN, Rank.KING)
            assert card.enhancement is not None

    def test_grim_creates_aces(self):
        gs = MockGameState()
        hand_size = len(gs.hand)
        result = create_consumable("c_grim").use(gs, [])
        assert len(gs.hand) == hand_size - 1
        assert len(result["create_cards"]) == 2
        for card in result["create_cards"]:
            assert card.rank == Rank.ACE

    def test_incantation_creates_number_cards(self):
        gs = MockGameState()
        hand_size = len(gs.hand)
        result = create_consumable("c_incantation").use(gs, [])
        assert len(gs.hand) == hand_size - 1
        assert len(result["create_cards"]) == 4
        for card in result["create_cards"]:
            assert 2 <= int(card.rank) <= 10

    def test_talisman_adds_gold_seal(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.ACE, Suit.HEARTS)])
        create_consumable("c_talisman").use(gs, [0])
        assert gs.hand[0].seal == Seal.GOLD

    def test_deja_vu_adds_red_seal(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.ACE, Suit.HEARTS)])
        create_consumable("c_deja_vu").use(gs, [0])
        assert gs.hand[0].seal == Seal.RED

    def test_trance_adds_blue_seal(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.ACE, Suit.HEARTS)])
        create_consumable("c_trance").use(gs, [0])
        assert gs.hand[0].seal == Seal.BLUE

    def test_medium_adds_purple_seal(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.ACE, Suit.HEARTS)])
        create_consumable("c_medium").use(gs, [0])
        assert gs.hand[0].seal == Seal.PURPLE

    def test_aura_adds_random_edition(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.ACE, Suit.HEARTS)])
        create_consumable("c_aura").use(gs, [0])
        assert gs.hand[0].edition in (Edition.FOIL, Edition.HOLO, Edition.POLYCHROME)

    def test_cryptid_creates_copies(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.ACE, Suit.SPADES, enhancement=Enhancement.GOLD)])
        initial_deck_count = gs.deck.card_count
        result = create_consumable("c_cryptid").use(gs, [0])
        assert len(result["create_cards"]) == 2
        assert gs.deck.card_count == initial_deck_count + 2
        for copy in result["create_cards"]:
            assert copy.rank == Rank.ACE
            assert copy.suit == Suit.SPADES
            assert copy.enhancement == Enhancement.GOLD
            assert copy.uid != gs.hand[0].uid

    def test_immolate_destroys_5_gains_20(self):
        gs = MockGameState()
        gs.money = 0
        assert len(gs.hand) == 8
        result = create_consumable("c_immolate").use(gs, [])
        assert len(gs.hand) == 3
        assert gs.money == 20
        assert len(result["destroy_cards"]) == 5

    def test_immolate_cannot_use_with_few_cards(self):
        gs = MockGameState()
        gs._set_hand([Card(Rank.ACE, Suit.HEARTS)] * 4)
        assert not create_consumable("c_immolate").can_use(gs, [])
