"""Tests for joker.py: registry, individual joker effects, scoring contexts."""

import pytest

from balatro_gym.core.card import Card, Rank, Suit
from balatro_gym.core.hand_evaluator import HandType, HandResult, evaluate_hand
from balatro_gym.core.joker import (
    BaseJoker,
    ScoreModification,
    create_joker,
    get_all_joker_ids,
    get_joker_class,
    get_jokers_by_rarity,
)


def _make_view(**kwargs):
    """Create a minimal GameStateView-like object."""
    defaults = dict(
        hand=[], jokers=[], money=0, ante=1, blind_type="small",
        score_target=300, current_score=0, hands_remaining=4,
        discards_remaining=3, deck_size=44, max_jokers=5,
        hands_played_this_round=0, hand_type_played_counts={},
    )
    defaults.update(kwargs)

    class View:
        pass

    v = View()
    for k, val in defaults.items():
        setattr(v, k, val)
    return v


def _cards(*specs: str) -> list[Card]:
    rank_map = {
        "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE,
        "6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE,
        "10": Rank.TEN, "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE,
    }
    suit_map = {"H": Suit.HEARTS, "D": Suit.DIAMONDS, "C": Suit.CLUBS, "S": Suit.SPADES}
    result = []
    for s in specs:
        result.append(Card(rank_map[s[:-1]], suit_map[s[-1]]))
    return result


class TestRegistry:
    def test_all_30_jokers_registered(self):
        ids = get_all_joker_ids()
        assert len(ids) == 30

    def test_create_joker(self):
        joker = create_joker("joker_basic")
        assert joker.INFO.id == "joker_basic"
        assert isinstance(joker, BaseJoker)

    def test_unknown_joker_raises(self):
        with pytest.raises(KeyError):
            create_joker("nonexistent_joker")

    def test_rarity_filter(self):
        common = get_jokers_by_rarity(1)
        assert "joker_basic" in common
        rare = get_jokers_by_rarity(3)
        assert "blueprint" in rare


class TestBasicJoker:
    def test_basic_joker_4_mult(self):
        joker = create_joker("joker_basic")
        view = _make_view()
        played = _cards("AH", "AD")
        result = evaluate_hand(played)
        mod = joker.on_main(result, result.scoring_cards, played,
                            {"Pair": True}, "Pair", view)
        assert mod is not None
        assert mod.mult_mod == 4


class TestSuitJokers:
    """Test suit-based jokers trigger per individual card."""

    def test_greedy_joker_diamonds(self):
        joker = create_joker("greedy_joker")
        view = _make_view()
        diamond = Card(Rank.KING, Suit.DIAMONDS)
        spade = Card(Rank.KING, Suit.SPADES)

        result = evaluate_hand([diamond, spade])
        mod = joker.on_individual(diamond, result, result.scoring_cards, "Pair", view)
        assert mod is not None and mod.mult == 3

        mod2 = joker.on_individual(spade, result, result.scoring_cards, "Pair", view)
        assert mod2 is None

    def test_lusty_joker_hearts(self):
        joker = create_joker("lusty_joker")
        view = _make_view()
        heart = Card(Rank.ACE, Suit.HEARTS)
        mod = joker.on_individual(heart, evaluate_hand([heart]), [heart], "High Card", view)
        assert mod is not None and mod.mult == 3

    def test_wrathful_joker_spades(self):
        joker = create_joker("wrathful_joker")
        view = _make_view()
        spade = Card(Rank.FIVE, Suit.SPADES)
        mod = joker.on_individual(spade, evaluate_hand([spade]), [spade], "High Card", view)
        assert mod is not None and mod.mult == 3

    def test_gluttonous_joker_clubs(self):
        joker = create_joker("gluttonous_joker")
        view = _make_view()
        club = Card(Rank.SEVEN, Suit.CLUBS)
        mod = joker.on_individual(club, evaluate_hand([club]), [club], "High Card", view)
        assert mod is not None and mod.mult == 3

    def test_suit_joker_debuffed_no_trigger(self):
        joker = create_joker("greedy_joker")
        view = _make_view()
        card = Card(Rank.KING, Suit.DIAMONDS, face_down=True)
        result = evaluate_hand([Card(Rank.KING, Suit.SPADES)])
        mod = joker.on_individual(card, result, [card], "High Card", view)
        assert mod is None


class TestHandTypeJokers:
    def test_jolly_joker_pair(self):
        joker = create_joker("jolly_joker")
        view = _make_view()
        mod = joker.on_main(None, [], [], {"Pair": True}, "Pair", view)
        assert mod is not None and mod.mult_mod == 8

    def test_jolly_joker_no_pair(self):
        joker = create_joker("jolly_joker")
        view = _make_view()
        mod = joker.on_main(None, [], [], {"High Card": True}, "High Card", view)
        assert mod is None

    def test_jolly_joker_full_house_contains_pair(self):
        joker = create_joker("jolly_joker")
        view = _make_view()
        # Full House contains Pair as a sub-hand
        mod = joker.on_main(None, [], [], {"Full House": True, "Pair": True,
                            "Three of a Kind": True}, "Full House", view)
        assert mod is not None and mod.mult_mod == 8

    def test_zany_joker_three_of_a_kind(self):
        joker = create_joker("zany_joker")
        view = _make_view()
        mod = joker.on_main(None, [], [], {"Three of a Kind": True}, "Three of a Kind", view)
        assert mod is not None and mod.mult_mod == 12


class TestConditionalJokers:
    def test_banner_discards_remaining(self):
        joker = create_joker("banner")
        view = _make_view(discards_remaining=3)
        mod = joker.on_main(None, [], [], {}, "Pair", view)
        assert mod is not None and mod.chip_mod == 90  # 3 * 30

    def test_banner_zero_discards(self):
        joker = create_joker("banner")
        view = _make_view(discards_remaining=0)
        mod = joker.on_main(None, [], [], {}, "Pair", view)
        assert mod is None

    def test_mystic_summit_zero_discards(self):
        joker = create_joker("mystic_summit")
        view = _make_view(discards_remaining=0)
        mod = joker.on_main(None, [], [], {}, "Pair", view)
        assert mod is not None and mod.mult_mod == 15

    def test_mystic_summit_has_discards(self):
        joker = create_joker("mystic_summit")
        view = _make_view(discards_remaining=1)
        mod = joker.on_main(None, [], [], {}, "Pair", view)
        assert mod is None

    def test_half_joker_3_cards(self):
        joker = create_joker("half_joker")
        view = _make_view()
        cards = _cards("AH", "AD", "AS")
        mod = joker.on_main(None, [], cards, {}, "Three of a Kind", view)
        assert mod is not None and mod.mult_mod == 20

    def test_half_joker_4_cards(self):
        joker = create_joker("half_joker")
        view = _make_view()
        cards = _cards("AH", "AD", "AS", "2C")
        mod = joker.on_main(None, [], cards, {}, "Three of a Kind", view)
        assert mod is None


class TestStatefulJokers:
    def test_ice_cream_starts_100_decrements(self):
        joker = create_joker("ice_cream")
        view = _make_view()
        result = evaluate_hand(_cards("AH"))

        # First hand: 100 chips
        mod = joker.on_main(result, [], [], {}, "High Card", view)
        assert mod.chip_mod == 100

        # After playing a hand, chips decrease by 5
        joker.on_after(result, [], "High Card", view)
        mod2 = joker.on_main(result, [], [], {}, "High Card", view)
        assert mod2.chip_mod == 95

    def test_runner_gains_chips_on_straight(self):
        joker = create_joker("runner")
        view = _make_view()
        cards = _cards("5H", "6D", "7C", "8S", "9H")
        result = evaluate_hand(cards)
        poker_hands = {"Straight": True}

        joker.on_before(result, cards, cards, poker_hands, "Straight", view)
        mod = joker.on_main(result, cards, cards, poker_hands, "Straight", view)
        assert mod.chip_mod == 15

        # Second straight: cumulative
        joker.on_before(result, cards, cards, poker_hands, "Straight", view)
        mod2 = joker.on_main(result, cards, cards, poker_hands, "Straight", view)
        assert mod2.chip_mod == 30

    def test_runner_no_gain_without_straight(self):
        joker = create_joker("runner")
        view = _make_view()
        cards = _cards("AH", "AD")
        result = evaluate_hand(cards)
        poker_hands = {"Pair": True}

        joker.on_before(result, cards, cards, poker_hands, "Pair", view)
        mod = joker.on_main(result, cards, cards, poker_hands, "Pair", view)
        assert mod is None  # 0 chips accumulated

    def test_ride_the_bus(self):
        joker = create_joker("ride_the_bus")
        view = _make_view()

        # Hand without face cards: +1
        no_face = _cards("2H", "5D")
        result = evaluate_hand(no_face)
        joker.on_before(result, no_face, no_face, {}, "High Card", view)
        mod = joker.on_main(result, no_face, no_face, {}, "High Card", view)
        assert mod.mult_mod == 1

        # Another hand without face cards: +2
        joker.on_before(result, no_face, no_face, {}, "High Card", view)
        mod2 = joker.on_main(result, no_face, no_face, {}, "High Card", view)
        assert mod2.mult_mod == 2

        # Hand with face card: reset
        face = _cards("JH", "QD")
        result2 = evaluate_hand(face)
        joker.on_before(result2, face, face, {}, "Pair", view)
        mod3 = joker.on_main(result2, face, face, {}, "Pair", view)
        assert mod3 is None  # reset to 0

    def test_supernova_uses_played_counts(self):
        joker = create_joker("supernova")
        view = _make_view(hand_type_played_counts={"Pair": 3})
        result = evaluate_hand(_cards("AH", "AD"))
        mod = joker.on_main(result, [], [], {"Pair": True}, "Pair", view)
        assert mod.mult_mod == 3


class TestPerCardJokers:
    def test_fibonacci(self):
        joker = create_joker("fibonacci")
        view = _make_view()
        # Fibonacci cards: Ace(14), 2, 3, 5, 8
        ace = Card(Rank.ACE, Suit.HEARTS)
        two = Card(Rank.TWO, Suit.HEARTS)
        six = Card(Rank.SIX, Suit.HEARTS)

        mod_ace = joker.on_individual(ace, None, [], "High Card", view)
        assert mod_ace is not None and mod_ace.mult == 8

        mod_two = joker.on_individual(two, None, [], "High Card", view)
        assert mod_two is not None and mod_two.mult == 8

        mod_six = joker.on_individual(six, None, [], "High Card", view)
        assert mod_six is None

    def test_even_steven(self):
        joker = create_joker("even_steven")
        view = _make_view()
        even = Card(Rank.EIGHT, Suit.HEARTS)
        odd = Card(Rank.SEVEN, Suit.HEARTS)

        mod_even = joker.on_individual(even, None, [], "High Card", view)
        assert mod_even is not None and mod_even.mult == 4

        mod_odd = joker.on_individual(odd, None, [], "High Card", view)
        assert mod_odd is None

    def test_odd_todd_31_chips(self):
        joker = create_joker("odd_todd")
        view = _make_view()
        odd = Card(Rank.SEVEN, Suit.HEARTS)
        mod = joker.on_individual(odd, None, [], "High Card", view)
        assert mod is not None and mod.chips == 31

    def test_odd_todd_ace_is_odd(self):
        joker = create_joker("odd_todd")
        view = _make_view()
        ace = Card(Rank.ACE, Suit.HEARTS)
        mod = joker.on_individual(ace, None, [], "High Card", view)
        assert mod is not None and mod.chips == 31

    def test_scholar_ace(self):
        joker = create_joker("scholar")
        view = _make_view()
        ace = Card(Rank.ACE, Suit.HEARTS)
        mod = joker.on_individual(ace, None, [], "High Card", view)
        assert mod is not None
        assert mod.chips == 20
        assert mod.mult == 4

    def test_scholar_non_ace(self):
        joker = create_joker("scholar")
        view = _make_view()
        king = Card(Rank.KING, Suit.HEARTS)
        mod = joker.on_individual(king, None, [], "High Card", view)
        assert mod is None

    def test_business_card_face(self):
        joker = create_joker("business_card")
        view = _make_view()
        jack = Card(Rank.JACK, Suit.HEARTS)
        mod = joker.on_individual(jack, None, [], "High Card", view)
        assert mod is not None and mod.dollars == 2

    def test_business_card_non_face(self):
        joker = create_joker("business_card")
        view = _make_view()
        ten = Card(Rank.TEN, Suit.HEARTS)
        mod = joker.on_individual(ten, None, [], "High Card", view)
        assert mod is None


class TestXMultJokers:
    def test_stencil_empty_slots(self):
        joker = create_joker("stencil")
        view = _make_view(max_jokers=5, jokers=[joker, create_joker("joker_basic")])
        mod = joker.on_main(None, [], [], {}, "High Card", view)
        # 5 - 2 = 3 empty + 1 = X4
        assert mod is not None and mod.Xmult_mod == 4.0

    def test_the_duo_pair(self):
        joker = create_joker("the_duo")
        view = _make_view()
        mod = joker.on_main(None, [], [], {"Pair": True}, "Pair", view)
        assert mod is not None and mod.Xmult_mod == 2.0

    def test_the_trio(self):
        joker = create_joker("the_trio")
        view = _make_view()
        mod = joker.on_main(None, [], [], {"Three of a Kind": True}, "Three of a Kind", view)
        assert mod is not None and mod.Xmult_mod == 3.0

    def test_the_family(self):
        joker = create_joker("the_family")
        view = _make_view()
        mod = joker.on_main(None, [], [], {"Four of a Kind": True}, "Four of a Kind", view)
        assert mod is not None and mod.Xmult_mod == 4.0

    def test_blackboard_all_black(self):
        joker = create_joker("blackboard")
        held = _cards("2S", "5C", "8S")
        scoring = _cards("AH", "AD")
        view = _make_view(hand=held + scoring)
        mod = joker.on_main(None, scoring, [], {}, "Pair", view)
        assert mod is not None and mod.Xmult_mod == 3.0

    def test_blackboard_has_red(self):
        joker = create_joker("blackboard")
        held = _cards("2S", "5H")  # 5H is red
        scoring = _cards("AH", "AD")
        view = _make_view(hand=held + scoring)
        mod = joker.on_main(None, scoring, [], {}, "Pair", view)
        assert mod is None


class TestAbstractJoker:
    def test_mult_per_joker(self):
        joker = create_joker("abstract_joker")
        jokers = [joker, create_joker("joker_basic"), create_joker("banner")]
        view = _make_view(jokers=jokers)
        mod = joker.on_main(None, [], [], {}, "High Card", view)
        assert mod.mult_mod == 9  # 3 jokers * 3
