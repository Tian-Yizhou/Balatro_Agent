"""Helpers that build game situations for tests."""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Sequence
from typing import Any

from balatro_gym.sim import Game, macros
from balatro_gym.sim.actions import (
    Action,
    ContinueEndless,
    Discard,
    Reroll,
    SelectBlind,
    SellJoker,
    UseConsumable,
)
from balatro_gym.sim.content import CENTERS, pool
from balatro_gym.sim.engine import _end_round as end_round
from balatro_gym.sim.engine import new_run
from balatro_gym.sim.events import Event, HandScored
from balatro_gym.sim.hands import display_name, evaluate_poker_hand
from balatro_gym.sim.inventory import add_to_deck, remove_from_deck
from balatro_gym.sim.items import new_item, set_cost
from balatro_gym.sim.modifiers import scoring_chips
from balatro_gym.sim.scope import Scope
from balatro_gym.sim.scoring import evaluate_play, level_up_hand
from balatro_gym.sim.shop import is_available
from balatro_gym.sim.state import (
    BlindSlot,
    BlindState,
    HandType,
    Phase,
    PlayingCard,
    RunState,
    Suit,
)

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


def engine_score(scenario: dict[str, Any]) -> dict[str, Any]:
    """Play a scoring scenario (see tools/lua_harness/scoring_scenarios.py) through
    the engine's ``evaluate_play`` and report the same results as the Lua harness."""
    state = new_run(str(scenario.get("seed", "lua-scenario")))
    state.deck.clear()
    state.money = scenario["dollars"]
    state.probabilities_normal = scenario["probability"]
    state.hands_left = scenario["hands_left"]
    state.discards_left = scenario["discards_left"]
    if scenario.get("tarots_used"):
        state.consumable_usage = {"c_hermit": scenario["tarots_used"]}
    for name, level in scenario["levels"].items():
        hand = HandType(name)
        level_up_hand(Scope(state), hand, level - state.hands[hand].level)
    played = [_scenario_card(card) for card in scenario["played"]]
    state.hand = [_scenario_card(card) for card in scenario["held"]]
    jokers = []
    for joker in scenario["jokers"]:
        item = new_item(state, joker["key"], edition=joker.get("edition"))
        item.uid = joker["uid"]
        item.values.update(joker.get("values", {}))
        jokers.append(item)
    state.jokers = list(jokers)
    scope = Scope(state)
    chips_before = state.chips
    destroyed = evaluate_play(scope, played)
    scope.flush()
    return {
        "score": state.chips - chips_before,
        "dollars": state.money,
        "jokers": [dict(joker.values) for joker in jokers],
        "destroyed": [card.uid for card in destroyed],
    }


def _scenario_card(card: dict[str, Any]) -> PlayingCard:
    return PlayingCard(
        uid=card["uid"],
        rank=card["rank"],
        suit=Suit(card["suit"]),
        enhancement=card.get("enhancement"),
        edition=card.get("edition"),
        seal=card.get("seal"),
        debuffed=card.get("debuffed", False),
    )


def joker_values_match(engine: dict[str, Any], game: dict[str, Any]) -> bool:
    """Whether the engine's values for a joker agree with the game's ability table."""
    defaults = {"mult": 0, "x_mult": 1}
    for name in {*engine, "mult", "x_mult"}:
        if name not in game or name in ("type", "t_mult", "t_chips"):
            continue
        ours = engine.get(name, defaults.get(name))
        theirs = game[name]
        if isinstance(theirs, str) or isinstance(ours, str):
            if ours != theirs:
                return False
        elif ours is None or abs(float(ours) - float(theirs)) > 1e-9:
            return False
    return True


def engine_end_of_round(scenario: dict[str, Any]) -> dict[str, Any]:
    """Run a round-end scenario through the engine; report what the Lua harness does."""
    state = new_run("lua-scenario")
    state.deck.clear()
    state.money = scenario["dollars"]
    state.probabilities_normal = scenario["probability"]
    state.hands_left = scenario["hands_left"]
    state.discards_left = scenario["discards_left"]
    state.last_hand_played = HandType(scenario["last_hand"])
    slot = BlindSlot.BOSS if scenario["boss"] else BlindSlot.SMALL
    state.blind = BlindState(
        key="bl_hook" if scenario["boss"] else "bl_small",
        slot=slot,
        chips=100,
        mult=1,
        dollars=5 if scenario["boss"] else 3,
    )
    state.blind_on_deck = slot
    state.chips = 100 if scenario["beaten"] else 50
    state.phase = Phase.SELECTING_HAND
    state.hand = [_scenario_card(card) for card in scenario["held"]]
    jokers = []
    for joker in scenario["jokers"]:
        item = new_item(state, joker["key"], edition=joker.get("edition"))
        item.uid = joker["uid"]
        item.values.update(joker.get("values", {}))
        jokers.append(item)
    state.jokers = list(jokers)
    scope = Scope(state)
    end_round(scope)
    scope.flush()
    beaten = scenario["beaten"]
    return {
        "dollars": state.money,
        "payout": sum(payout.dollars for payout in state.payouts) if beaten else None,
        "jokers": {joker.uid: dict(joker.values) for joker in state.jokers},
        "consumables": [item.key for item in state.consumables],
        "pool_flags": dict(state.pool_flags),
    }


def engine_discard(scenario: dict[str, Any]) -> dict[str, Any]:
    """Run a discard scenario through the engine; report what the Lua harness does."""
    state = new_run("lua-scenario")
    state.deck.clear()
    state.money = scenario["dollars"]
    state.probabilities_normal = scenario["probability"]
    state.discards_left = scenario["discards_left"]
    state.phase = Phase.SELECTING_HAND
    state.blind = BlindState(key="bl_small", slot=BlindSlot.SMALL, chips=300, mult=1, dollars=3)
    # The harness runs only the game's discard function, so keep cards to draw:
    # an empty hand and draw pile would end the round here.
    state.deck = [PlayingCard(uid=900 + rank, rank=rank, suit=Suit.CLUBS) for rank in range(2, 15)]
    state.hand = [_scenario_card(card) for card in scenario["held"]]
    for position in scenario["selected"]:
        state.hand[position].selected = True
    jokers = []
    for joker in scenario["jokers"]:
        item = new_item(state, joker["key"], edition=joker.get("edition"))
        item.uid = joker["uid"]
        item.values.update(joker.get("values", {}))
        jokers.append(item)
    state.jokers = list(jokers)
    Game(state).step(Discard())
    return {
        "dollars": state.money,
        "jokers": [dict(joker.values) for joker in jokers],
        "consumables": [CENTERS[item.key].set for item in state.consumables],
    }


def compare_results(ours: dict[str, Any], game: dict[str, Any], jokers: Any) -> list[str]:
    """Differences between the engine's and the game's results for one scenario.

    ``jokers`` is the scenario's joker list, used to name joker differences. Scores
    are compared with a relative tolerance of 1e-9, other values exactly.
    """
    problems: list[str] = []
    for field, theirs in game.items():
        mine: Any = ours[field]
        if field == "score":
            if abs(mine - theirs) > 1e-9 * max(1.0, abs(theirs)):
                problems.append(f"score {mine} vs {theirs}")
        elif field == "destroyed":
            if sorted(mine) != sorted(theirs):
                problems.append(f"destroyed {mine} vs {theirs}")
        elif field == "jokers":
            problems.extend(_compare_jokers(mine, theirs, jokers))
        elif mine != theirs:
            problems.append(f"{field} {mine} vs {theirs}")
    return problems


def _compare_jokers(mine: Any, theirs: Any, jokers: Any) -> list[str]:
    if isinstance(theirs, dict):
        if set(mine) != {int(uid) for uid in theirs}:
            return [f"jokers left {sorted(mine)} vs {sorted(theirs)}"]
        pairs = [(mine[int(uid)], values, uid) for uid, values in theirs.items()]
    else:
        pairs = [
            (values, game_values, joker["key"])
            for values, game_values, joker in zip(mine, theirs, jokers, strict=True)
        ]
    return [
        f"{name} values {values} vs {game_values}"
        for values, game_values, name in pairs
        if not joker_values_match(values, game_values)
    ]


def engine_use_consumable(scenario: dict[str, Any]) -> dict[str, Any]:
    """Use a consumable in the engine; report what ``rules.use_consumable`` does."""
    state = new_run("lua-scenario")
    state.deck.clear()
    state.money = scenario["dollars"]
    state.probabilities_normal = scenario["probability"]
    state.last_tarot_planet = scenario.get("last_tarot_planet")
    state.phase = Phase(scenario["phase"])
    scope = Scope(state)
    for name, level in scenario.get("levels", {}).items():
        hand = HandType(name)
        level_up_hand(scope, hand, level - state.hands[hand].level)
    state.hand = [_scenario_card(card) for card in scenario["held"]]
    for position in scenario["selected"]:
        state.hand[position].selected = True
    jokers = []
    for joker in scenario["jokers"]:
        item = new_item(state, joker["key"], edition=joker.get("edition"))
        item.uid = joker["uid"]
        item.values.update(joker.get("values", {}))
        jokers.append(item)
    state.jokers = list(jokers)
    others = [new_item(state, key) for key in scenario.get("consumables", [])]
    state.consumables = [*others, new_item(state, scenario["consumable"])]
    game = Game(state)
    action = UseConsumable(len(state.consumables) - 1)
    if not game.is_legal(action):
        return {"usable": False}
    game.step(action)
    return {
        "usable": True,
        "hand": [
            {
                "uid": card.uid,
                "rank": card.rank,
                "suit": card.suit.value,
                "enhancement": card.enhancement,
                "edition": card.edition,
                "seal": card.seal,
            }
            for card in state.hand
        ],
        "levels": {hand.value: level.level for hand, level in state.hands.items()},
        "dollars": state.money,
        "consumables": [item.key for item in state.consumables],
        "jokers": [{"key": item.key, "edition": item.edition is not None} for item in state.jokers],
        "joker_values": [dict(item.values) for item in jokers],
        "last_tarot_planet": state.last_tarot_planet,
    }


#: Consumables that create random cards: only the kind and count of the created
#: cards are compared, since the engine's random draws differ from the game's.
RANDOM_CREATORS = frozenset({"c_emperor", "c_high_priestess", "c_judgement", "c_wheel_of_fortune"})


def compare_consumable_use(
    scenario: dict[str, Any], ours: dict[str, Any], game: dict[str, Any]
) -> list[str]:
    """Differences between the engine's and the game's results of using a consumable."""
    if ours["usable"] != game["usable"]:
        return [f"usable {ours['usable']} vs {game['usable']}"]
    if not game["usable"]:
        return []
    problems = [
        f"{field} {ours[field]} vs {game[field]}"
        for field in ("hand", "levels", "dollars", "last_tarot_planet")
        if ours[field] != game[field]
    ]
    random_creation = scenario["consumable"] in RANDOM_CREATORS
    for field, describe in (
        ("consumables", lambda key: CENTERS[key].set if random_creation else key),
        ("jokers", lambda joker: joker["key"] if not random_creation else "joker"),
    ):
        mine = sorted(map(describe, ours[field]))
        theirs = sorted(map(describe, game[field]))
        if mine != theirs:
            problems.append(f"{field} {mine} vs {theirs}")
    our_editions = [joker["edition"] for joker in ours["jokers"]]
    game_editions = [joker["edition"] for joker in game["jokers"]]
    if scenario["consumable"] in RANDOM_CREATORS:
        # The Wheel of Fortune picks a random joker, and Judgement's joker may
        # roll an edition: compare how many jokers have one.
        our_editions, game_editions = [sum(our_editions)], [sum(game_editions)]
        if scenario["consumable"] == "c_judgement":
            our_editions = game_editions
    if our_editions != game_editions:
        problems.append(f"joker editions {ours['jokers']} vs {game['jokers']}")
    problems.extend(_compare_jokers(ours["joker_values"], game["joker_values"], scenario["jokers"]))
    return problems


def engine_shop(scenario: dict[str, Any]) -> dict[str, Any]:
    """Run a shop scenario through the engine; report what ``rules.shop`` does."""
    state = new_run("lua-scenario")
    state.deck.clear()
    state.money = scenario["dollars"]
    state.inflation = scenario["inflation"]
    state.discount_percent = scenario["discount"]
    state.discards_per_round = state.discards_left = 3
    state.phase = Phase.SHOP
    for name in scenario["played_hands"]:
        state.hands[HandType(name)].played = 1
    for flag in scenario["pool_flags"]:
        state.pool_flags[flag] = True
    jokers = []
    for joker in scenario["jokers"]:
        item = new_item(state, joker["key"], edition=joker.get("edition"))
        item.uid = joker["uid"]
        item.rental = joker.get("rental", False)
        item.values.update(joker.get("values", {}))
        set_cost(state, item)
        jokers.append(item)
    state.jokers = list(jokers)
    state.consumables = [new_item(state, key) for key in scenario["consumables"]]

    pools = {
        f"Joker{rarity}": sorted(c.key for c in pool("Joker", rarity) if is_available(state, c))
        for rarity in (1, 2, 3)
    }
    for set_name in ("Tarot", "Planet"):
        pools[set_name] = sorted(c.key for c in pool(set_name) if is_available(state, c))
    prices = [[item.cost, item.sell_value] for item in jokers]

    def run_numbers() -> dict[str, int]:
        return {
            "hand_size": state.hand_size,
            "discards_per_round": state.discards_per_round,
            "discards_left": state.discards_left,
            "bankrupt_at": state.bankrupt_at,
            "free_rerolls": state.free_rerolls,
            "reroll_cost": state.reroll_cost,
        }

    scope = Scope(state)
    gained = new_item(state, scenario["gained"])
    add_to_deck(scope, gained)
    after_gain = run_numbers()
    remove_from_deck(scope, gained)
    after_loss = run_numbers()
    game = Game(state)
    for _ in range(scenario.get("rerolls", 0)):
        if not game.is_legal(Reroll()):
            break
        game.step(Reroll())
    reroll_cost = state.reroll_cost
    if scenario["sell"] is not None:
        game.step(SellJoker(scenario["sell"]))
    return {
        "pools": pools,
        "prices": prices,
        "after_gain": after_gain,
        "after_loss": after_loss,
        "reroll_cost": reroll_cost,
        "dollars": state.money,
        "jokers": {joker.uid: dict(joker.values) for joker in state.jokers},
    }


def compare_shop(ours: dict[str, Any], game: dict[str, Any], implemented: set[str]) -> list[str]:
    """Differences in a shop scenario. The game's pools are narrowed to the items the
    engine implements, which are the only ones it offers (ADR 0003)."""
    problems = []
    for name, keys in game["pools"].items():
        expected = sorted(key for key in keys if key in implemented)
        if ours["pools"][name] != expected:
            problems.append(f"pool {name}: {ours['pools'][name]} vs {expected}")
    for field in ("prices", "after_gain", "after_loss", "reroll_cost", "dollars"):
        if ours[field] != game[field]:
            problems.append(f"{field} {ours[field]} vs {game[field]}")
    problems.extend(_compare_jokers(ours["jokers"], game["jokers"], None))
    return problems
