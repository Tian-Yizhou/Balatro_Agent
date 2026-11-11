"""Helpers that build game situations for tests."""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Sequence
from typing import Any

from balatro_gym.sim import Game, macros
from balatro_gym.sim.actions import Action, ContinueEndless, SelectBlind
from balatro_gym.sim.events import Event, HandScored
from balatro_gym.sim.hands import display_name, evaluate_poker_hand
from balatro_gym.sim.items import new_item
from balatro_gym.sim.modifiers import scoring_chips
from balatro_gym.sim.state import Phase, PlayingCard, RunState, Suit

Policy = Callable[[Game, random.Random], list[Action]]

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
            hand.value: [list(group) for group in groups]
            for hand, groups in hands.groups.items()
            if groups
        },
    }


def started_round(seed: str | int = "test") -> Game:
    """A new game with the small blind selected and the first hand drawn."""
    game = Game.new(seed)
    game.step(SelectBlind())
    return game


def set_hand(state: RunState, labels: str) -> list[PlayingCard]:
    """Put the cards named in ``labels`` (event.g. ``"AS KH 2C"``) in hand, in that order.

    Cards are taken from the draw or discard pile, and the previous hand
    goes back to the draw pile, so every card is still accounted for.
    """
    state.deck.extend(state.hand)
    state.hand.clear()
    for label in labels.split():
        pile = next(
            candidate
            for candidate in (state.deck, state.discard)
            if any(card.label == label for card in candidate)
        )
        card = next(card for card in pile if card.label == label)
        pile.remove(card)
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
            chips = level.chips + sum(scoring_chips(cards[i]) for i in hands.scoring)
            best = max(best, (chips * level.mult, combo))
    return best[1]


def greedy_policy(game: Game, randomizer: random.Random) -> list[Action]:
    """Play the best plain-card hand; take the first legal action elsewhere."""
    state = game.state
    if state.phase is Phase.SELECTING_HAND:
        return macros.play_cards(state, best_hand(state))
    if state.phase is Phase.WIN:
        return [ContinueEndless()]
    return [game.legal_actions()[0]]


def random_policy(game: Game, randomizer: random.Random) -> list[Action]:
    """A uniformly random legal action, with a bias towards ending the round."""
    actions = game.legal_actions()
    if randomizer.random() < 0.3:
        return [actions[0]]
    return [randomizer.choice(actions)]


def play_game(seed: str | int, policy: Policy, max_steps: int = 5000) -> tuple[Game, list[Action]]:
    """Play a game to the end and return it with the atomic actions taken."""
    game = Game.new(seed)
    randomizer = random.Random(str(seed))
    taken: list[Action] = []
    while not game.is_over and len(taken) < max_steps:
        for action in policy(game, randomizer):
            game.step(action)
            taken.append(action)
    return game, taken


def play_hand(
    played: str,
    *,
    held: str = "",
    jokers: Sequence[str] = (),
    setup: Callable[[RunState, list[PlayingCard]], None] | None = None,
    seed: str = "score",
) -> tuple[Game, list[Event], HandScored]:
    """Play the cards in ``played`` with ``held`` kept in hand and ``jokers`` owned.

    ``setup`` can change the state and the cards (played first, then held)
    before the hand is played. Returns the game, the events of the play, and the
    hand's score.
    """
    game = started_round(seed)
    state = game.state
    cards = list(set_hand(state, f"{played} {held}"))
    for key in jokers:
        state.jokers.append(new_item(state, key))
    if setup is not None:
        setup(state, cards)
    events: list[Event] = []
    for action in macros.play_cards(state, range(len(played.split()))):
        events += game.step(action)
    scored = next(event for event in events if isinstance(event, HandScored))
    return game, events, scored
