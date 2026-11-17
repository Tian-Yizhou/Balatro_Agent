"""Random and greedy games, checking invariants after every step."""

from __future__ import annotations

import dataclasses
import json
import random

import pytest

from balatro_gym.sim import Game, IllegalActionError, is_legal, legal_actions, snapshot
from balatro_gym.sim.actions import (
    ACTION_TYPES,
    Action,
    Buy,
    BuyAndUse,
    BuyPack,
    MoveCard,
    MoveJoker,
    Pick,
    SellConsumable,
    SellJoker,
    ToggleCard,
    UseConsumable,
)
from balatro_gym.sim.consumables import IMPLEMENTED as CONSUMABLES
from balatro_gym.sim.effects import JOKERS
from balatro_gym.sim.engine import SELECTION_LIMIT
from balatro_gym.sim.events import CardDestroyed, PlayingCardCreated
from balatro_gym.sim.items import new_item
from balatro_gym.sim.state import RunState
from scenarios import Policy, greedy_policy, play_game, random_policy


def candidate_actions(state: RunState) -> list[Action]:
    """Every action with arguments near the valid ranges, legal or not."""
    hand = range(-1, len(state.hand) + 1)
    jokers = range(-1, len(state.jokers) + 1)
    candidates: list[Action] = [
        cls() for cls in ACTION_TYPES.values() if not dataclasses.fields(cls)
    ]
    candidates += [ToggleCard(i) for i in hand]
    candidates += [MoveCard(source, target) for source in hand for target in hand]
    candidates += [MoveJoker(source, target) for source in jokers for target in jokers]
    shop = range(-1, len(state.shop_cards) + 1)
    consumables = range(-1, len(state.consumables) + 1)
    candidates += [cls(i) for cls in (Buy, BuyAndUse) for i in shop]
    candidates += [SellJoker(i) for i in jokers]
    candidates += [cls(i) for cls in (SellConsumable, UseConsumable) for i in consumables]
    candidates += [BuyPack(i) for i in range(-1, len(state.shop_boosters) + 1)]
    pack_size = state.pack.size if state.pack is not None else 0
    candidates += [Pick(i) for i in range(-1, pack_size + 1)]
    return candidates


def check_invariants(game: Game, uids: set[int]) -> None:
    """``uids`` are the playing cards still owned; destroyed cards are removed from it."""
    state = game.state
    assert {card.uid for card in state.playing_cards()} == uids
    assert len(state.playing_cards()) == len(uids)
    assert not any(card.selected for card in state.deck + state.discard)
    assert sum(card.selected for card in state.hand) <= SELECTION_LIMIT
    assert state.hands_left >= 0
    assert state.discards_left >= 0
    legal = legal_actions(state)
    assert len(legal) == len(set(legal))
    assert set(legal) == {action for action in candidate_actions(state) if is_legal(state, action)}
    assert bool(legal) != state.is_over


def equip(game: Game, randomizer: random.Random) -> None:
    """Give the player random implemented jokers, consumables and card modifiers."""
    state = game.state
    state.consumables = [new_item(state, key) for key in randomizer.sample(sorted(CONSUMABLES), 2)]
    for key in randomizer.sample(sorted(JOKERS), 5):
        joker = new_item(state, key)
        joker.edition = randomizer.choice([None, None, "e_foil", "e_holo", "e_polychrome"])
        state.jokers.append(joker)
    for card in state.deck:
        card.enhancement = randomizer.choice([None, None, None, *ENHANCEMENTS])
        card.edition = randomizer.choice(
            [None, None, None, None, "e_foil", "e_holo", "e_polychrome"]
        )
        card.seal = randomizer.choice([None, None, None, None, "Red", "Gold", "Blue"])


ENHANCEMENTS = ["m_bonus", "m_mult", "m_wild", "m_glass", "m_steel", "m_stone", "m_gold", "m_lucky"]


@pytest.mark.parametrize(
    ("policy", "seeds", "equipped"),
    [
        (random_policy, range(25), False),
        (greedy_policy, range(8), False),
        (greedy_policy, range(15), True),
        (random_policy, range(15), True),
    ],
    ids=["random", "greedy", "greedy-jokers", "random-jokers"],
)
def test_invariants_hold(policy: Policy, seeds: range, equipped: bool) -> None:
    for seed in seeds:
        game = Game.new(seed)
        randomizer = random.Random(seed)
        if equipped:
            equip(game, randomizer)
        uids = {card.uid for card in game.state.playing_cards()}
        steps = 0
        while not game.is_over and steps < 3000:
            for action in policy(game, randomizer):
                for event in game.step(action):
                    if isinstance(event, CardDestroyed):
                        uids.discard(event.uid)
                    elif isinstance(event, PlayingCardCreated):
                        uids.add(event.uid)
                steps += 1
                check_invariants(game, uids)
            if steps % 50 == 0:
                text = json.dumps(game.snapshot(), sort_keys=True)
                restored = Game.from_snapshot(json.loads(text))
                assert restored.state == game.state
                assert json.dumps(restored.snapshot(), sort_keys=True) == text
        assert game.is_over


def test_illegal_actions_are_rejected() -> None:
    game, _ = play_game(3, greedy_policy, max_steps=30)
    before = snapshot(game.state)
    for action in candidate_actions(game.state):
        if not game.is_legal(action):
            with pytest.raises(IllegalActionError):
                game.step(action)
    assert snapshot(game.state) == before


def test_replaying_actions_reproduces_the_game() -> None:
    for seed in range(5):
        game, taken = play_game(seed, greedy_policy)
        again = Game.new(seed)
        for action in taken:
            again.step(action)
        assert snapshot(again.state) == snapshot(game.state)


def test_greedy_player_reaches_later_antes() -> None:
    antes = [play_game(seed, greedy_policy)[0].state.ante for seed in range(8)]
    assert max(antes) >= 2
