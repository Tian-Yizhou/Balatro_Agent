"""Helpers that build game situations for tests."""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Sequence
from typing import Any

from balatro_gym.sim import Game, macros
from balatro_gym.sim.actions import (
    Action,
    Buy,
    BuyPack,
    CashOut,
    ContinueEndless,
    Discard,
    Pick,
    RedeemVoucher,
    Reroll,
    RerollBoss,
    SelectBlind,
    SellJoker,
    UseConsumable,
)
from balatro_gym.sim.blinds import boss_candidates, stays_face_down
from balatro_gym.sim.bosses import defeat_blind, drawn_to_hand, press_play, start_blind
from balatro_gym.sim.content import CENTERS, pool
from balatro_gym.sim.effects import Context as JokerContext
from balatro_gym.sim.effects import ContextKind, calculate_joker
from balatro_gym.sim.engine import _discard_cards as discard_cards
from balatro_gym.sim.engine import _end_round as end_round
from balatro_gym.sim.engine import _set_blind as set_blind
from balatro_gym.sim.engine import _take as take_from_hand
from balatro_gym.sim.engine import new_run
from balatro_gym.sim.events import Event, HandScored
from balatro_gym.sim.hands import display_name, evaluate_poker_hand
from balatro_gym.sim.inventory import add_to_deck, disable_blind, remove_from_deck
from balatro_gym.sim.items import new_item, set_cost
from balatro_gym.sim.modifiers import scoring_chips
from balatro_gym.sim.packs import pack_kind
from balatro_gym.sim.scope import Scope
from balatro_gym.sim.scoring import evaluate_play, level_up_hand
from balatro_gym.sim.shop import create_card_for_shop, is_available
from balatro_gym.sim.state import (
    BlindSlot,
    BlindState,
    HandType,
    Item,
    OpenPack,
    Phase,
    PlayingCard,
    RunConfig,
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


def in_shop(seed: str = "shop", config: RunConfig | None = None) -> Game:
    """A game standing in the shop after winning the small blind."""
    game = Game.new(seed, config)
    game.step(SelectBlind())
    state = game.state
    assert state.blind is not None
    state.blind.chips = 1
    for action in macros.play_cards(state, [0]):
        game.step(action)
    game.step(CashOut())
    assert state.phase is Phase.SHOP
    return game


def give(state: RunState, key: str) -> Item:
    item = new_item(state, key)
    (state.jokers if CENTERS[key].is_joker else state.consumables).append(item)
    return item


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
    """Hand positions of the selection that scores the most with plain cards,
    always including a card Cerulean Bell forces."""
    best: tuple[float, tuple[int, ...]] = (-1.0, ())
    forced = {i for i, card in enumerate(state.hand) if card.forced_selection}
    for size in range(1, 6):
        for combo in itertools.combinations(range(len(state.hand)), size):
            if not forced <= set(combo):
                continue
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
    state.consumables = [new_item(state, key) for key in scenario.get("consumables", [])]
    state.used_vouchers = list(scenario.get("vouchers", []))
    blind = scenario.get("blind")
    if blind is not None:
        state.blind = set_blind(state, blind["key"], BlindSlot.BOSS)
        if blind["key"] == "bl_eye":
            state.blind.hands_used = [HandType(hand) for hand in blind.get("hands_used", [])]
        if blind["key"] == "bl_mouth" and blind.get("only_hand"):
            state.blind.only_hand = HandType(blind["only_hand"])
        state.blind.disabled = blind.get("disabled", False)
        state.most_played_hand = HandType(blind.get("most_played", "High Card"))
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
    used = new_item(state, scenario["consumable"])
    action: Action
    if state.phase is Phase.PACK_OPEN:
        # Picked from a Spectral pack that stays open, so the hand stays in view.
        state.consumables = others
        state.pack = OpenPack(
            key="p_spectral_mega_1",
            items=[used, new_item(state, "c_pluto")],
            playing_cards=[],
            choices_left=2,
            return_phase=Phase.SHOP,
        )
        action = Pick(0)
    else:
        state.consumables = [*others, used]
        action = UseConsumable(len(state.consumables) - 1)
    game = Game(state)
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
        "hand_size": state.hand_size,
        "consumables": [item.key for item in state.consumables],
        "jokers": [{"key": item.key, "edition": item.edition is not None} for item in state.jokers],
        "joker_values": [dict(item.values) for item in jokers],
        "last_tarot_planet": state.last_tarot_planet,
    }


#: Consumables that create or change random items: only the kind and count of the
#: items are compared, since the engine's random draws differ from the game's.
RANDOM_CREATORS = frozenset(
    {
        "c_emperor",
        "c_high_priestess",
        "c_judgement",
        "c_wheel_of_fortune",
        "c_wraith",
        "c_soul",
        "c_ankh",
        "c_hex",
        "c_ectoplasm",
    }
)

#: Of those, the ones whose jokers are compared by rarity: a rare or a legendary.
_BY_RARITY = frozenset({"c_wraith", "c_soul"})

#: Consumables whose resulting joker editions depend on which joker was drawn.
_RANDOM_EDITIONS = frozenset({"c_judgement", "c_wraith", "c_soul", "c_ankh"})

#: Ranks the card-creating Spectral cards may add.
_CREATED_RANKS = {"c_familiar": {11, 12, 13}, "c_grim": {14}, "c_incantation": set(range(2, 11))}


def _hand_outline(key: str, scenario: dict[str, Any], hand: list[dict[str, Any]]) -> Any:
    """What a consumable's effect on the hand must give, whatever its random draws.

    Effects without randomness are compared card by card. Sigil and Ouija must
    give every card one suit or one rank, Aura some edition, and Cryptid exact
    copies with new uids. Familiar, Grim and Incantation must destroy one card
    and add cards of their ranks with an enhancement other than Stone; Immolate
    destroys up to five cards.
    """
    originals = {card["uid"] for card in scenario["held"]}
    kept = [card for card in hand if card["uid"] in originals]
    created = [card for card in hand if card["uid"] not in originals]
    if key in ("c_sigil", "c_ouija"):
        field = "suit" if key == "c_sigil" else "rank"
        others = [{name: value for name, value in card.items() if name != field} for card in hand]
        return others, len({card[field] for card in hand}) <= 1
    if key == "c_aura":
        return [{**card, "edition": card["edition"] is not None} for card in hand]
    if key == "c_cryptid":
        return kept, [
            {name: value for name, value in card.items() if name != "uid"} for card in created
        ]
    if key in _CREATED_RANKS:
        valid = all(
            card["rank"] in _CREATED_RANKS[key]
            and card["enhancement"] not in (None, "m_stone")
            and card["edition"] is None
            and card["seal"] is None
            for card in created
        )
        return len(kept), len(created), valid
    if key == "c_immolate":
        return len(kept), len(created)
    return hand


def compare_consumable_use(
    scenario: dict[str, Any], ours: dict[str, Any], game: dict[str, Any]
) -> list[str]:
    """Differences between the engine's and the game's results of using a consumable."""
    if ours["usable"] != game["usable"]:
        return [f"usable {ours['usable']} vs {game['usable']}"]
    if not game["usable"]:
        return []
    key = scenario["consumable"]
    problems = [
        f"{field} {ours[field]} vs {game[field]}"
        for field in ("levels", "dollars", "hand_size", "last_tarot_planet")
        if ours[field] != game[field]
    ]
    if _hand_outline(key, scenario, ours["hand"]) != _hand_outline(key, scenario, game["hand"]):
        problems.append(f"hand {ours['hand']} vs {game['hand']}")
    random_creation = key in RANDOM_CREATORS

    def describe_joker(joker: dict[str, Any]) -> str:
        if key in _BY_RARITY:
            return f"rarity {CENTERS[joker['key']].rarity}"
        return "joker" if random_creation else joker["key"]

    for field, describe in (
        ("consumables", lambda item: CENTERS[item].set if random_creation else item),
        ("jokers", describe_joker),
    ):
        mine = sorted(map(describe, ours[field]))
        theirs = sorted(map(describe, game[field]))
        if mine != theirs:
            problems.append(f"{field} {mine} vs {theirs}")
    our_editions = [joker["edition"] for joker in ours["jokers"]]
    game_editions = [joker["edition"] for joker in game["jokers"]]
    if random_creation:
        # The Wheel of Fortune, Hex and Ectoplasm pick a random joker: compare how
        # many jokers have an edition. A created or copied joker's edition depends
        # on which joker it is.
        our_editions, game_editions = [sum(our_editions)], [sum(game_editions)]
        if key in _RANDOM_EDITIONS:
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


def engine_open_pack(scenario: dict[str, Any], game_cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Open a pack in the engine; report what ``rules.open_pack`` does.

    Which cards can be picked is checked on the cards the game created
    (``game_cards``), with the same hand, since the engine's random cards differ.
    """
    state = new_run(scenario["seed"])
    state.ante = scenario["ante"]
    state.money = scenario["dollars"]
    state.phase = Phase.SHOP
    state.used_vouchers = list(scenario.get("vouchers", []))
    for name, played in scenario.get("played_hands", {}).items():
        state.hands[HandType(name)].played = played
    for joker in scenario["jokers"]:
        state.jokers.append(new_item(state, joker["key"], edition=joker.get("edition")))
    state.shop_boosters = [new_item(state, scenario["pack"])]
    game = Game(state)
    game.step(BuyPack(0))
    pack = state.pack
    assert pack is not None
    result: dict[str, Any] = {
        "kind": pack_kind(pack.key),
        "choices": pack.choices_left,
        "size": pack.size,
        "dollars": state.money,
        "cards": [
            {"set": CENTERS[item.key].set, "key": item.key, "edition": item.edition}
            for item in pack.items
        ]
        + [
            {"set": "Playing", "enhancement": card.enhancement, "edition": card.edition}
            for card in pack.playing_cards
        ],
    }
    if state.hand:
        state.deck.extend(state.hand)
        state.hand = [_scenario_card(card) for card in scenario["held"]]
    if pack.items:
        pack.items = [new_item(state, card["key"], edition=card["edition"]) for card in game_cards]
    result["pickable"] = [game.is_legal(Pick(i)) for i in range(pack.size)]
    return result


def compare_open_pack(
    scenario: dict[str, Any], ours: dict[str, Any], game: dict[str, Any]
) -> list[str]:
    """Differences between opening a pack in the engine and in the game.

    The cards are random, so only their kinds are compared, with The Soul and
    Black Hole, which may stand in for any card of a pack, set aside; with Omen
    Globe an Arcana card may be a Spectral card. Playing cards must have a valid
    enhancement and no Negative edition. With Telescope, once a hand has been
    played, a Celestial pack's first card must be the same.
    """
    problems = [
        f"{field} {ours[field]} vs {game[field]}"
        for field in ("kind", "choices", "size", "dollars", "pickable")
        if ours[field] != game[field]
    ]
    vouchers = scenario.get("vouchers", [])
    omen_globe = "v_omen_globe" in vouchers and game["kind"] == "Arcana"

    def kinds(cards: list[dict[str, Any]]) -> list[str]:
        return sorted(
            "Tarot" if omen_globe and card["set"] == "Spectral" else card["set"]
            for card in cards
            if card.get("key") not in ("c_soul", "c_black_hole")
        )

    first_differs = ours["cards"][0].get("key") != game["cards"][0].get("key")
    telescope = "v_telescope" in vouchers and bool(scenario.get("played_hands"))
    if telescope and game["kind"] == "Celestial" and first_differs:
        problems.append(f"first card {ours['cards'][0]} vs {game['cards'][0]}")

    enhancements = {center.key for center in pool("Enhanced")}
    for side, cards in (("engine", ours["cards"]), ("game", game["cards"])):
        for card in cards:
            if card["set"] == "Playing" and (
                card["enhancement"] not in (None, *enhancements) or card["edition"] == "e_negative"
            ):
                problems.append(f"{side} made an invalid playing card {card}")
    souls = sum(card.get("key") in ("c_soul", "c_black_hole") for card in ours["cards"])
    game_souls = sum(card.get("key") in ("c_soul", "c_black_hole") for card in game["cards"])
    our_kinds, game_kinds = kinds(ours["cards"]), kinds(game["cards"])
    if souls == game_souls and our_kinds != game_kinds:
        problems.append(f"cards {our_kinds} vs {game_kinds}")
    return problems


def engine_redeem_voucher(scenario: dict[str, Any]) -> dict[str, Any]:
    """Redeem a voucher in the engine; report what ``rules.redeem_voucher`` does."""
    state = new_run("lua-scenario")
    state.deck.clear()
    state.phase = Phase.SHOP
    state.ante = scenario["ante"]
    state.money = scenario["dollars"]
    state.hands_per_round, state.hands_left = 4, scenario["hands_left"]
    state.discards_per_round, state.discards_left = 3, scenario["discards_left"]
    state.reroll_cost_increase = scenario["reroll_increase"]
    state.reroll_cost = state.base_reroll_cost + state.reroll_cost_increase
    state.used_vouchers = list(scenario["used"])
    jokers = []
    for joker in scenario["jokers"]:
        item = new_item(state, joker["key"], edition=joker.get("edition"))
        item.rental = joker.get("rental", False)
        item.values.update(joker.get("values", {}))
        set_cost(state, item)
        jokers.append(item)
    state.jokers = list(jokers)
    state.consumables = [new_item(state, key) for key in scenario["consumables"]]
    consumables = list(state.consumables)
    state.shop_joker_max = scenario["shop_slots"]
    state.shop_cards = [new_item(state, key) for key in scenario["shop"]]
    shop = list(state.shop_cards)
    state.shop_boosters = [new_item(state, scenario["pack"])]
    state.shop_vouchers = [new_item(state, scenario["voucher"])]
    state.shop_voucher = scenario["voucher"]
    game = Game(state)
    if not game.is_legal(RedeemVoucher(0)):
        return {"redeemable": False}
    game.step(RedeemVoucher(0))

    def prices(items: list[Item]) -> list[list[int]]:
        return [[item.cost, item.sell_value] for item in items]

    return {
        "redeemable": True,
        "dollars": state.money,
        "numbers": {
            "ante": state.ante,
            "hands_per_round": state.hands_per_round,
            "hands_left": state.hands_left,
            "discards_per_round": state.discards_per_round,
            "discards_left": state.discards_left,
            "hand_size": state.hand_size,
            "joker_slots": state.joker_slots,
            "consumable_slots": state.consumable_slots,
            "rates": [
                state.joker_rate,
                state.tarot_rate,
                state.planet_rate,
                state.playing_card_rate,
                state.spectral_rate,
                state.edition_rate,
            ],
            "interest_cap": state.interest_cap,
            "discount_percent": state.discount_percent,
            "base_reroll_cost": state.base_reroll_cost,
            "reroll_cost": state.reroll_cost,
            "shop_slots": state.shop_joker_max,
            "shop_cards": len(state.shop_cards),
        },
        "prices": {
            "jokers": prices(jokers),
            "consumables": prices(consumables),
            "shop": prices(shop),
            "pack": prices(state.shop_boosters),
        },
        "voucher_pool": sorted(
            center.key for center in pool("Voucher") if is_available(state, center)
        ),
        "shop_voucher": state.shop_voucher,
    }


def engine_reroll_boss(scenario: dict[str, Any], game: dict[str, Any]) -> dict[str, Any]:
    """Press the boss reroll in the engine; report what ``rules.reroll_boss`` does.

    The new boss is random, so the boss the game picked is reported back if the
    engine could have picked it too.
    """
    state = new_run("lua-scenario")
    state.ante = scenario["ante"]
    state.money = scenario["dollars"]
    state.bankrupt_at = scenario["bankrupt_at"]
    state.used_vouchers = list(scenario["used"])
    state.boss_rerolled = scenario["rerolled"]
    state.bosses_used = dict.fromkeys(state.bosses_used, 0) | scenario["bosses_used"]
    candidates = boss_candidates(state.ante, state.win_ante, state.bosses_used)
    run = Game(state)
    if not run.is_legal(RerollBoss()):
        return {"allowed": False}
    run.step(RerollBoss())
    picked = state.blind_choices[BlindSlot.BOSS]
    assert picked in candidates
    return {
        "allowed": True,
        "dollars": state.money,
        "rerolled": state.boss_rerolled,
        "boss": game.get("boss") if game.get("boss") in candidates else f"not in {candidates}",
    }


def _shop_kinds(scenario: dict[str, Any], cards: list[dict[str, Any]]) -> list[str]:
    """The kinds of shop cards seen: each card's set and, in shops that sell nothing
    but playing cards (where enough of them are made), whether a playing card
    has an edition and which editions occur."""
    joker_rate, tarot_rate, planet_rate, _, spectral_rate = scenario["rates"]
    only_playing = joker_rate + tarot_rate + planet_rate + spectral_rate == 0
    kinds = set()
    for card in cards:
        kinds.add(card["set"])
        if only_playing and card["edition"]:
            kinds.update((f"{card['set']} with an edition", card["edition"]))
    return sorted(kinds)


def engine_shop_playing_cards(scenario: dict[str, Any], game: dict[str, Any]) -> dict[str, Any]:
    """Fill a shop in the engine; report what ``rules.shop_playing_cards`` does.

    The cards are random, so each card's set (and a playing card's edition) is
    reported as a sorted list of the kinds seen; prices are checked on the
    playing cards the game created, and the purchase on the game's first one.
    """
    state = new_run(scenario["seed"])
    state.deck.clear()
    state.phase = Phase.SHOP
    state.ante = scenario["ante"]
    state.money = scenario["dollars"]
    state.discount_percent = scenario["discount"]
    state.used_vouchers = list(scenario["used"])
    (
        state.joker_rate,
        state.tarot_rate,
        state.planet_rate,
        state.playing_card_rate,
        state.spectral_rate,
    ) = scenario["rates"]
    jokers = [new_item(state, key) for key in scenario["jokers"]]
    state.jokers = list(jokers)
    scope = Scope(state)
    made = [create_card_for_shop(scope) for _ in range(scenario["count"])]
    ours = [
        {
            "set": CENTERS[item.key].set,
            "edition": item.playing_card.edition if item.playing_card else item.edition,
        }
        for item in made
    ]
    game_playing = [card for card in game["cards"] if card["set"] in ("Default", "Enhanced")]
    offers = [_shop_offer(state, card) for card in game_playing]
    result: dict[str, Any] = {
        "kinds": _shop_kinds(scenario, ours),
        "costs": [offer.cost for offer in offers],
    }
    if offers:
        state.shop_cards = [offers[0]]
        run = Game(state)
        if not run.is_legal(Buy(0)):
            result["bought"] = {"allowed": False}
        else:
            run.step(Buy(0))
            result["bought"] = {
                "allowed": True,
                "dollars": state.money,
                "deck": len(state.deck),
                "jokers": [dict(joker.values) for joker in jokers],
            }
    return result


def _shop_offer(state: RunState, card: dict[str, Any]) -> Item:
    """A playing card for sale, as the game described it."""
    playing_card = _scenario_card(card)
    item = Item(
        uid=playing_card.uid, key=card["enhancement"] or "c_base", playing_card=playing_card
    )
    set_cost(state, item)
    return item


def compare_shop_playing_cards(
    scenario: dict[str, Any], ours: dict[str, Any], game: dict[str, Any]
) -> list[str]:
    """Differences between filling a shop in the engine and in the game."""
    problems = []
    game_kinds = _shop_kinds(scenario, game["cards"])
    if ours["kinds"] != game_kinds:
        problems.append(f"kinds {ours['kinds']} vs {game_kinds}")
    game_costs = [card["cost"] for card in game["cards"] if card["set"] in ("Default", "Enhanced")]
    if ours["costs"] != game_costs:
        problems.append(f"costs {ours['costs']} vs {game_costs}")
    if ("bought" in ours) != ("bought" in game):
        problems.append("only one side bought a card")
    elif "bought" in ours:
        mine, theirs = ours["bought"], game["bought"]
        for field in ("allowed", "dollars", "deck"):
            if mine.get(field) != theirs.get(field):
                problems.append(f"bought {field} {mine.get(field)} vs {theirs.get(field)}")
        if mine.get("allowed"):
            keys = [{"key": key} for key in scenario["jokers"]]
            problems.extend(_compare_jokers(mine["jokers"], theirs["jokers"], keys))
    return problems


def engine_boss_round(scenario: dict[str, Any]) -> dict[str, Any]:
    """Play through a boss blind's effects in the engine; report what
    ``rules.boss_round`` does.

    Random picks (The Hook's cards, Cerulean Bell's card, Crimson Heart's joker,
    Amber Acorn's order) are reported as counts or sets, since the engine's
    random draws differ from the game's.
    """
    state = new_run(scenario["seed"])
    state.ante = scenario["ante"]
    state.money = scenario["dollars"]
    state.probabilities_normal = scenario["probability"]
    state.hands_per_round, state.discards_per_round = 4, 3
    state.hands_left, state.discards_left = 4, 3
    state.hands_played_this_round = scenario["hands_played"]
    state.discards_used_this_round = scenario["discards_used"]
    state.phase = Phase.BLIND_SELECT
    for joker in scenario["jokers"]:
        item = new_item(state, joker["key"], edition=joker.get("edition"))
        item.uid = joker["uid"]
        state.jokers.append(item)
    hand = [_scenario_card(card) for card in scenario["hand"]]
    deck = [_scenario_card(card) for card in scenario["deck"]]
    for card, description in zip(
        [*hand, *deck], [*scenario["hand"], *scenario["deck"]], strict=True
    ):
        card.played_this_ante = description.get("played_this_ante", False)
    state.hand, state.deck = list(hand), list(deck)
    scope = Scope(state)
    state.blind = set_blind(state, scenario["blind"], BlindSlot.BOSS)
    start_blind(scope)
    for joker in list(state.jokers):
        calculate_joker(scope, joker, JokerContext(ContextKind.SETTING_BLIND))
    scope.flush()

    def snapshot() -> dict[str, Any]:
        blind = state.blind
        return {
            "debuffed": sorted(card.uid for card in [*hand, *deck] if card.debuffed),
            "jokers_debuffed": sorted(joker.uid for joker in state.jokers if joker.debuffed),
            "jokers_face_down": sum(joker.face_down for joker in state.jokers),
            "jokers": sorted(joker.uid for joker in state.jokers),
            "hands_left": state.hands_left,
            "discards_left": state.discards_left,
            "hand_size": state.hand_size,
            "chips": blind.chips if blind is not None else 0.0,
            "disabled": blind.disabled if blind is not None else False,
            "dollars": state.money,
        }

    # Cards the engine draws (Chicot disabling The Manacle) are left out, since
    # the harness doesn't move cards.
    original = {card.uid for card in hand}

    def face_down() -> list[bool]:
        return [stays_face_down(scope, card) for card in state.hand if card.uid in original]

    result: dict[str, Any] = {"set": snapshot(), "face_down": face_down()}
    state.phase = Phase.SELECTING_HAND
    played = take_from_hand(state, [hand[i] for i in scenario["played"]])
    # Played cards are still the player's while they score (in the game's play area).
    state.discard.extend(played)
    hooked = press_play(scope, played)
    if hooked:
        discard_cards(scope, hooked)
    scope.flush()
    result["played"] = {**snapshot(), "hook_discarded": len(hooked), "face_down": face_down()}
    drawn_to_hand(scope)
    scope.flush()
    result["drawn"] = {
        **snapshot(),
        "forced": sum(card.forced_selection for card in state.hand if card.uid in original),
    }
    if scenario["disable"]:
        disable_blind(scope)
        scope.flush()
        result["disabled"] = snapshot()
    defeat_blind(scope)
    result["defeated"] = snapshot()
    return result


def compare_boss_round(
    scenario: dict[str, Any], ours: dict[str, Any], game: dict[str, Any]
) -> list[str]:
    """Differences in a boss round. Crimson Heart debuffs a random joker, so only
    how many it debuffs is compared."""
    problems = []
    for step in sorted({*ours, *game}):
        mine, theirs = ours.get(step), game.get(step)
        if scenario["blind"] == "bl_final_heart" and step == "drawn" and mine and theirs:
            mine = {**mine, "jokers_debuffed": len(mine["jokers_debuffed"])}
            theirs = {**theirs, "jokers_debuffed": len(theirs["jokers_debuffed"])}
        if mine != theirs:
            problems.append(f"{step}: {mine} vs {theirs}")
    return problems
