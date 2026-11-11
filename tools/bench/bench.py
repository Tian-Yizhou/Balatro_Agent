"""Measure engine throughput: steps per second for random and greedy players.

Usage::

    uv run python tools/bench/bench.py            # default: 200 games per policy
    uv run python tools/bench/bench.py --games 1000
"""

from __future__ import annotations

import argparse
import itertools
import random
import time
from collections.abc import Callable

from balatro_gym.sim import Game, Phase, clone, macros
from balatro_gym.sim.actions import Action, Buy, ContinueEndless, UseConsumable
from balatro_gym.sim.hands import evaluate_poker_hand
from balatro_gym.sim.modifiers import scoring_chips

Policy = Callable[[Game, random.Random], list[Action]]


def random_policy(game: Game, randomizer: random.Random) -> list[Action]:
    actions = game.legal_actions()
    return [actions[0] if randomizer.random() < 0.3 else randomizer.choice(actions)]


def greedy_policy(game: Game, randomizer: random.Random) -> list[Action]:
    """Plays the best plain-card hand, buys what it can afford, uses what it can."""
    state = game.state
    legal = game.legal_actions()
    if state.phase is Phase.SELECTING_HAND:
        best: tuple[float, tuple[int, ...]] = (-1.0, ())
        for size in range(1, 6):
            for combo in itertools.combinations(range(len(state.hand)), size):
                cards = [state.hand[i] for i in combo]
                hands = evaluate_poker_hand(cards)
                assert hands.top is not None
                level = state.hands[hands.top]
                chips = level.chips + sum(scoring_chips(cards[i]) for i in hands.scoring)
                best = max(best, (chips * level.mult, combo))
        return macros.play_cards(state, best[1])
    for kind in (UseConsumable, Buy, ContinueEndless):
        for action in legal:
            if isinstance(action, kind):
                return [action]
    return [legal[0]]


def run(policy: Policy, games: int) -> tuple[int, float, float, list[int]]:
    """Play ``games`` games; return steps, engine seconds, legal-action seconds, antes.

    Only the engine's own work is timed: ``step`` and ``legal_actions``, not the
    policy's thinking.
    """
    steps = 0
    engine = 0.0
    listing = 0.0
    antes = []
    clock = time.perf_counter
    for seed in range(games):
        start = clock()
        game = Game.new(seed)
        engine += clock() - start
        randomizer = random.Random(seed)
        while not game.is_over:
            for action in policy(game, randomizer):
                start = clock()
                game.step(action)
                engine += clock() - start
                steps += 1
            start = clock()
            game.legal_actions()
            listing += clock() - start
        antes.append(game.state.ante)
    return steps, engine, listing, antes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    args = parser.parse_args()
    for name, policy in (("random", random_policy), ("greedy", greedy_policy)):
        steps, seconds, listing, antes = run(policy, args.games)
        print(
            f"{name:7s} {args.games} games, {steps} steps: engine {steps / seconds:,.0f} steps/s"
            f" ({seconds / steps * 1e6:.0f} us/step), legal_actions {listing / steps * 1e6:.0f} us,"
            f" mean ante {sum(antes) / len(antes):.2f}, best ante {max(antes)}"
        )
    state = Game.new(0).state
    start = time.perf_counter()
    for _ in range(2000):
        clone(state)
    print(f"clone   {(time.perf_counter() - start) / 2000 * 1e6:.0f} us per state")


if __name__ == "__main__":
    main()
