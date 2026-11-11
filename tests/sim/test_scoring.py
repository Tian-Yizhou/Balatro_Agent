"""Scenario tests for the scoring pipeline, card modifiers and the starter jokers.

Expected scores are worked out by hand from the game's rules. "AS" alone is a
High Card: 5 base chips + 11 for the Ace, x1 mult = 16 chips x 1.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from balatro_gym.sim import actions as act
from balatro_gym.sim import events as ev
from balatro_gym.sim import macros
from balatro_gym.sim.effects import JOKERS
from balatro_gym.sim.items import new_item
from balatro_gym.sim.state import HandType, Phase, PlayingCard, RunState
from scenarios import play_hand, started_round

Setup = Callable[[RunState, list[PlayingCard]], None]


def enhance(position: int, enhancement: str) -> Setup:
    def setup(state: RunState, cards: list[PlayingCard]) -> None:
        cards[position].enhancement = enhancement

    return setup


def with_field(position: int, name: str, value: str) -> Setup:
    def setup(state: RunState, cards: list[PlayingCard]) -> None:
        setattr(cards[position], name, value)

    return setup


def chain(*setups: Setup) -> Setup:
    def setup(state: RunState, cards: list[PlayingCard]) -> None:
        for each in setups:
            each(state, cards)

    return setup


def certain(state: RunState, cards: list[PlayingCard]) -> None:
    """Make every "1 in N" chance (N <= 1000) succeed."""
    state.probabilities_normal = 1000


def score(
    played: str,
    *,
    held: str = "",
    jokers: Sequence[str] = (),
    setup: Setup | None = None,
    seed: str = "score",
) -> float:
    return play_hand(played, held=held, jokers=jokers, setup=setup, seed=seed)[2].score


class TestModifiers:
    @pytest.mark.parametrize(
        ("played", "held", "setup", "expected"),
        [
            ("AS", "", enhance(0, "m_bonus"), (16 + 30) * 1),
            ("AS", "", enhance(0, "m_mult"), 16 * (1 + 4)),
            ("AS", "", enhance(0, "m_glass"), 16 * 2),
            ("AS 2C", "", enhance(1, "m_stone"), (16 + 50) * 1),
            ("AS", "KS", enhance(1, "m_steel"), 16 * 1.5),
            ("AS", "", with_field(0, "edition", "e_foil"), (16 + 50) * 1),
            ("AS", "", with_field(0, "edition", "e_holo"), 16 * (1 + 10)),
            ("AS", "", with_field(0, "edition", "e_polychrome"), 16 * 1.5),
            ("AS", "", with_field(0, "seal", "Red"), (16 + 11) * 1),
            (
                "AS",
                "KS",
                chain(enhance(1, "m_steel"), with_field(1, "seal", "Red")),
                16 * 1.5 * 1.5,
            ),
            ("AS", "KS", with_field(1, "seal", "Red"), 16),
        ],
    )
    def test_score(self, played: str, held: str, setup: Setup, expected: float) -> None:
        assert score(played, held=held, setup=setup) == expected

    def test_debuffed_card_has_no_modifiers(self) -> None:
        def setup(state: RunState, cards: list[PlayingCard]) -> None:
            cards[0].enhancement = "m_mult"
            cards[0].edition = "e_holo"
            cards[0].debuffed = True

        assert score("AS AH", setup=setup) == (10 + 11) * 2

    def test_lucky_card(self) -> None:
        _, events, scored = play_hand("AS", setup=chain(enhance(0, "m_lucky"), certain))
        assert scored.score == 16 * (1 + 20)
        assert any(isinstance(e, ev.MoneyChanged) and e.delta == 20 for e in events)
        assert play_hand("AS", setup=enhance(0, "m_lucky"), seed="unlucky")[2].score in (16, 336)

    def test_gold_seal_pays_when_scored(self) -> None:
        game, _, _ = play_hand("AS", setup=with_field(0, "seal", "Gold"))
        assert game.state.money == 4 + 3

    def test_glass_card_shatters(self) -> None:
        game, events, _ = play_hand("AS", setup=chain(enhance(0, "m_glass"), certain))
        destroyed = [e for e in events if isinstance(e, ev.CardDestroyed)]
        assert [e.reason for e in destroyed] == ["shattered"]
        assert destroyed[0].uid not in {c.uid for c in game.state.playing_cards()}
        assert len(game.state.playing_cards()) == 51

    def test_glass_card_can_survive(self) -> None:
        def never(state: RunState, cards: list[PlayingCard]) -> None:
            state.probabilities_normal = 0
            cards[0].enhancement = "m_glass"

        game, events, _ = play_hand("AS", setup=never)
        assert not any(isinstance(e, ev.CardDestroyed) for e in events)
        assert len(game.state.playing_cards()) == 52

    def test_end_of_round_gold_card_and_blue_seal(self) -> None:
        def setup(state: RunState, cards: list[PlayingCard]) -> None:
            assert state.blind is not None
            state.blind.chips = 1
            cards[1].enhancement = "m_gold"
            cards[1].seal = "Blue"

        game, _, _ = play_hand("AS", held="KS", setup=setup)
        state = game.state
        assert state.phase is Phase.ROUND_EVAL
        assert state.money == 4 + 3
        assert [item.key for item in state.consumables] == ["c_pluto"]


class TestJokers:
    @pytest.mark.parametrize(
        ("jokers", "played", "held", "expected"),
        [
            (["j_joker"], "AS", "", 16 * 5),
            (["j_jolly"], "AS AH", "", 32 * (2 + 8)),
            (["j_jolly"], "AS", "", 16),
            (["j_sly"], "AS AH", "", (32 + 50) * 2),
            (["j_duo"], "AS AH", "", 32 * 2 * 2),
            (["j_duo"], "AS", "", 16),
            (["j_greedy_joker"], "AD AH", "", 32 * (2 + 3)),
            (["j_scary_face"], "KS KH", "", (30 + 60) * 2),
            (["j_smiley"], "KS KH", "", 30 * (2 + 10)),
            (["j_fibonacci"], "8S 8H", "", 26 * (2 + 16)),
            (["j_even_steven"], "8S 8H", "", 26 * (2 + 8)),
            (["j_odd_todd"], "AS AH", "", (32 + 62) * 2),
            (["j_scholar"], "AS AH", "", (32 + 40) * (2 + 8)),
            (["j_walkie_talkie"], "TS TH", "", (30 + 20) * (2 + 8)),
            (["j_photograph"], "KS KH", "", 30 * 2 * 2),
            (["j_baron"], "AS", "KS KH", 16 * 1.5 * 1.5),
            (["j_shoot_the_moon"], "AS", "QS", 16 * (1 + 13)),
            (["j_hanging_chad"], "AS", "", (16 + 22) * 1),
            (["j_sock_and_buskin"], "KS KH", "", (30 + 20) * 2),
            (["j_hack"], "2S 2H", "", (14 + 4) * 2),
            (["j_mime", "j_baron"], "AS", "KS", 16 * 1.5 * 1.5),
            (["j_mime"], "AS", "KS", 16),
            (["j_ride_the_bus"], "AS", "", 16 * 2),
            (["j_green_joker"], "AS", "", 16 * 2),
            (["j_ice_cream"], "AS", "", 116),
            (["j_popcorn"], "AS", "", 16 * 21),
            (["j_gros_michel"], "AS", "", 16 * 16),
            (["j_cavendish"], "AS", "", 16 * 3),
            (["j_half"], "AS", "", 16 * 21),
            (["j_half"], "AS AH 2C 3D", "", 32 * 2),
            (["j_abstract", "j_joker"], "AS", "", 16 * (1 + 6 + 4)),
            (["j_banner"], "AS", "", (16 + 4 * 30) * 1),
            (["j_fibonacci", "j_baseball"], "AS", "", 16 * 9 * 1.5),
            (["j_blueprint", "j_joker"], "AS", "", 16 * 9),
            (["j_joker", "j_brainstorm"], "AS", "", 16 * 9),
            (["j_blueprint"], "AS", "", 16),
            (["j_splash"], "AS 2C", "", 18),
            (["j_four_fingers"], "2S 3H 4C 5D", "", (30 + 14) * 4),
            (["j_shortcut"], "2S 3H 5C 6D 7S", "", (30 + 23) * 4),
            (["j_smeared"], "AH KD QH JD 9H", "", (35 + 50) * 4),
            (["j_pareidolia", "j_scary_face"], "AS", "", (16 + 30) * 1),
        ],
    )
    def test_score(self, jokers: list[str], played: str, held: str, expected: float) -> None:
        assert score(played, held=held, jokers=jokers) == expected

    def test_acrobat_on_the_last_hand(self) -> None:
        def last_hand(state: RunState, cards: list[PlayingCard]) -> None:
            state.hands_left = 1

        assert score("AS", jokers=["j_acrobat"], setup=last_hand) == 16 * 3
        assert score("AS", jokers=["j_acrobat"]) == 16

    def test_mystic_summit_with_no_discards(self) -> None:
        def no_discards(state: RunState, cards: list[PlayingCard]) -> None:
            state.discards_left = 0

        assert score("AS", jokers=["j_mystic_summit"], setup=no_discards) == 16 * 16
        assert score("AS", jokers=["j_mystic_summit"]) == 16

    def test_misprint_is_random_and_reproducible(self) -> None:
        results = {score("AS", jokers=["j_misprint"], seed=str(seed)) for seed in range(30)}
        assert all(16 <= s <= 16 * 24 and s % 16 == 0 for s in results)
        assert len(results) > 5
        assert score("AS", jokers=["j_misprint"]) == score("AS", jokers=["j_misprint"])

    def test_joker_editions(self) -> None:
        def edition(name: str) -> Setup:
            def setup(state: RunState, cards: list[PlayingCard]) -> None:
                state.jokers[0].edition = name

            return setup

        assert score("AS", jokers=["j_joker"], setup=edition("e_foil")) == 66 * 5
        assert score("AS", jokers=["j_joker"], setup=edition("e_holo")) == 16 * 15
        assert score("AS", jokers=["j_joker"], setup=edition("e_polychrome")) == 16 * 5 * 1.5

    def test_debuffed_joker_does_nothing(self) -> None:
        def debuff(state: RunState, cards: list[PlayingCard]) -> None:
            state.jokers[0].debuffed = True
            state.jokers[0].edition = "e_foil"

        assert score("AS", jokers=["j_joker"], setup=debuff) == 16

    def test_ride_the_bus_grows_and_resets(self) -> None:
        game, _, _ = play_hand("AS", jokers=["j_ride_the_bus"])
        bus = game.state.jokers[0]
        assert bus.values["mult"] == 1
        game.state.hands[HandType.HIGH_CARD].mult = 1
        hand = game.state.hand
        face = next(i for i, c in enumerate(hand) if c.rank in (11, 12, 13))
        for action in macros.play_cards(game.state, [face]):
            game.step(action)
        assert bus.values["mult"] == 0

    def test_blueprint_does_not_double_scaling(self) -> None:
        game, _, scored = play_hand("AS", jokers=["j_blueprint", "j_ride_the_bus"])
        assert game.state.jokers[1].values["mult"] == 1
        assert scored.score == 16 * (1 + 1 + 1)

    def test_green_joker_loses_mult_on_discard(self) -> None:
        game, _, _ = play_hand("AS", jokers=["j_green_joker"])
        green = game.state.jokers[0]
        for action in macros.discard_cards(game.state, [0, 1]):
            game.step(action)
        assert green.values["mult"] == 0

    def test_ice_cream_melts(self) -> None:
        def nearly_gone(state: RunState, cards: list[PlayingCard]) -> None:
            state.jokers[0].values["extra.chips"] = 5

        game, events, scored = play_hand("AS", jokers=["j_ice_cream"], setup=nearly_gone)
        assert scored.score == 16 + 5
        assert game.state.jokers == []
        assert any(isinstance(e, ev.JokerDestroyed) and e.reason == "melted" for e in events)

    def test_popcorn_and_gros_michel_at_end_of_round(self) -> None:
        def win(state: RunState, cards: list[PlayingCard]) -> None:
            assert state.blind is not None
            state.blind.chips = 1
            state.probabilities_normal = 1000

        game, _, _ = play_hand("AS", jokers=["j_popcorn", "j_gros_michel"], setup=win)
        state = game.state
        assert [j.key for j in state.jokers] == ["j_popcorn"]
        assert state.jokers[0].values["mult"] == 16
        assert state.pool_flags == {"gros_michel_extinct": True}

    def test_golden_joker_pays_at_cash_out(self) -> None:
        def win(state: RunState, cards: list[PlayingCard]) -> None:
            assert state.blind is not None
            state.blind.chips = 1

        game, _, _ = play_hand("AS", jokers=["j_golden"], setup=win)
        assert ("j_golden", 4) in [(p.source, p.dollars) for p in game.state.payouts]

    def test_effect_events_describe_the_score(self) -> None:
        _, events, _ = play_hand("AS", jokers=["j_blueprint", "j_joker"])
        applied = [e for e in events if isinstance(e, ev.EffectApplied)]
        assert [(e.key, e.mult) for e in applied if e.mult] == [("j_joker", 4), ("j_joker", 4)]
        assert applied[-2].via is not None
        assert applied[-1].via is None


def test_every_registered_joker_has_a_scenario() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert [key for key in JOKERS if f'"{key}"' not in source] == []


@pytest.mark.parametrize(
    ("key", "played", "expected"),
    [
        ("j_zany", "AS AH AC", 63 * (3 + 12)),
        ("j_mad", "AS AH KS KH", 62 * (2 + 10)),
        ("j_crazy", "2S 3H 4C 5D 6S", 50 * (4 + 12)),
        ("j_droll", "2S 4S 6S 8S TS", 65 * (4 + 10)),
        ("j_wily", "AS AH AC", (63 + 100) * 3),
        ("j_clever", "AS AH KS KH", (62 + 80) * 2),
        ("j_devious", "2S 3H 4C 5D 6S", (50 + 100) * 4),
        ("j_crafty", "2S 4S 6S 8S TS", (65 + 80) * 4),
        ("j_trio", "AS AH AC", 63 * 3 * 3),
        ("j_family", "AS AH AC AD", 104 * 7 * 4),
        ("j_order", "2S 3H 4C 5D 6S", 50 * 4 * 3),
        ("j_tribe", "2S 4S 6S 8S TS", 65 * 4 * 2),
        ("j_lusty_joker", "AH", 16 * 4),
        ("j_wrathful_joker", "AS", 16 * 4),
        ("j_gluttenous_joker", "AC", 16 * 4),
    ],
)
def test_shared_rule_jokers(key: str, played: str, expected: float) -> None:
    assert score(played, jokers=[key]) == expected


def test_new_item_values() -> None:
    state = started_round().state
    bus = new_item(state, "j_ride_the_bus")
    assert bus.values == {"extra": 1}
    assert (bus.cost, bus.sell_value) == (6, 3)
    duo = new_item(state, "j_duo")
    assert duo.values == {"x_mult": 2, "type": "Pair"}


def test_cash_out_still_works_with_jokers() -> None:
    game, _, _ = play_hand("TS JS QS KS AS", jokers=["j_joker"])
    game.step(act.CashOut())
    assert game.state.phase is Phase.SHOP
