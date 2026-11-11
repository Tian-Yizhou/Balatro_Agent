"""Helpers that build game situations for tests."""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable
from typing import Any

from balatro_gym.sim import Game, macros
from balatro_gym.sim import actions as act
from balatro_gym.sim.cards import chip_bonus
from balatro_gym.sim.hands import display_name, evaluate_poker_hand
from balatro_gym.sim.state import Phase, PlayingCard, RunState, Suit

Policy = Callable[[Game, random.Random], list[act.Action]]

_RANKS = "23456789TJQKA"
_ENHANCEMENTS = {"w": "m_wild", "s": "m_stone"}


def cards(text: str) -> list[PlayingCard]:
    """Cards from text such as ``"AS KHw 2Cs 5D!"`` (Wild, Stone, debuffed)."""
    result = []
    for uid, token in enumerate(text.split(), start=1):
        debuffed = token.endswith("!")
        token = token.rstrip("!")
        result.append(
            PlayingCard(
                uid=uid,
                rank=_RANKS.index(token[0]) + 2,
                suit=Suit(token[1]),
                enhancement=_ENHANCEMENTS.get(token[2:]) if len(token) > 2 else None,
                debuffed=debuffed,
            )
        )
    return result


def describe(text: str, flags: str = "") -> dict[str, Any]:
    """Evaluate cards and return the result in the vector format."""
    played = cards(text)
    hands = evaluate_poker_hand(
        played, four_fingers="f" in flags, shortcut="s" in flags, smeared="m" in flags
    )
    return {
        "top": hands.top.value if hands.top else None,
        "display": display_name(played, hands),
        "scoring": list(hands.scoring),
        "groups": {
            hand.value: [list(g) for g in groups] for hand, groups in hands.groups.items() if groups
        },
    }


def started_round(seed: str | int = "test") -> Game:
    """A new game with the small blind selected and the first hand drawn."""
    game = Game.new(seed)
    game.step(act.SelectBlind())
    return game


def set_hand(state: RunState, labels: str) -> list[PlayingCard]:
    """Put the cards named in ``labels`` (e.g. ``"AS KH 2C"``) in hand, in that order.

    Cards are taken from the draw or discard pile, and the previous hand
    goes back to the draw pile, so every card is still accounted for.
    """
    state.deck.extend(state.hand)
    state.hand.clear()
    for label in labels.split():
        area = next(a for a in (state.deck, state.discard) if any(c.label == label for c in a))
        card = next(card for card in area if card.label == label)
        area.remove(card)
        card.selected = False
        state.hand.append(card)
    return state.hand


def best_hand(state: RunState) -> tuple[int, ...]:
    """Hand positions of the selection that scores the most with plain cards."""
    best: tuple[float, tuple[int, ...]] = (-1.0, ())
    for size in range(1, 6):
        for combo in itertools.combinations(range(len(state.hand)), size):
            cards = [state.hand[i] for i in combo]
            hands = evaluate_poker_hand(cards)
            assert hands.top is not None
            level = state.hands[hands.top]
            chips = level.chips + sum(chip_bonus(cards[i]) for i in hands.scoring)
            best = max(best, (chips * level.mult, combo))
    return best[1]


def greedy_policy(game: Game, rnd: random.Random) -> list[act.Action]:
    """Play the best plain-card hand; take the first legal action elsewhere."""
    state = game.state
    if state.phase is Phase.SELECTING_HAND:
        return macros.play_cards(state, best_hand(state))
    if state.phase is Phase.WIN:
        return [act.ContinueEndless()]
    return [game.legal_actions()[0]]


def random_policy(game: Game, rnd: random.Random) -> list[act.Action]:
    """A uniformly random legal action, with a bias towards ending the round."""
    actions = game.legal_actions()
    if rnd.random() < 0.3:
        return [actions[0]]
    return [rnd.choice(actions)]


def play_game(
    seed: str | int, policy: Policy, max_steps: int = 5000
) -> tuple[Game, list[act.Action]]:
    """Play a game to the end and return it with the atomic actions taken."""
    game = Game.new(seed)
    rnd = random.Random(str(seed))
    taken: list[act.Action] = []
    while not game.is_over and len(taken) < max_steps:
        for action in policy(game, rnd):
            game.step(action)
            taken.append(action)
    return game, taken
