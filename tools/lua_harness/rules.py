"""Run the game's own rule code (scoring, jokers, consumables) under LuaJIT.

``LuaRules`` loads the game's source with the stand-ins in ``game_env.lua`` and
builds game situations from plain descriptions, so tests can put the same
situation into the original game and into the engine and compare the results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "CardDescription",
    "LuaRules",
    "discard",
    "end_of_round",
    "open_pack",
    "score",
    "shop",
    "use_consumable",
]

HERE = Path(__file__).parent

#: Game files loaded, in order. ``game.lua`` only defines ``Game`` methods here.
_SOURCES = (
    "functions/misc_functions.lua",
    "card.lua",
    "functions/common_events.lua",
    "functions/state_events.lua",
    "functions/button_callbacks.lua",
    "game.lua",
)

_PRELUDE = """
G = {FUNCS = {}}
UIBox = function() return {} end
"""

# Runs after engine/object.lua: the base class of cards, without any graphics.
_BASE_CLASSES = """
Moveable = Object:extend()
function Moveable:init(x, y, w, h)
  self.T = {x = x or 0, y = y or 0, w = w or 1, h = h or 1, r = 0, scale = 1}
  self.VT = {x = self.T.x, y = self.T.y, w = self.T.w, h = self.T.h, r = 0, scale = 1}
  self.states = {visible = true, collide = {}, hover = {}, drag = {is = false},
    click = {}, focus = {}}
  self.children = setmetatable({}, {__index = function() return harness_anything end})
  G.ID = (G.ID or 100000) + 1
  self.ID = G.ID
end
"""

_VALUES = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "Jack",
    12: "Queen",
    13: "King",
    14: "Ace",
}
_SUITS = {"S": "Spades", "H": "Hearts", "C": "Clubs", "D": "Diamonds"}
_EDITIONS = {
    "e_foil": "foil",
    "e_holo": "holo",
    "e_polychrome": "polychrome",
    "e_negative": "negative",
}

CardDescription = dict[str, Any]


class LuaRules:
    """The game's rules, loaded from its source and ready to be called."""

    def __init__(self, lua_dir: Path) -> None:
        from lupa import luajit21  # type: ignore[import-untyped]

        self.lua: Any = luajit21.LuaRuntime(unpack_returned_tuples=True)
        self.lua.execute(_PRELUDE)
        self.lua.execute((lua_dir / "engine/object.lua").read_text(encoding="utf-8"))
        self.lua.execute(_BASE_CLASSES)
        for relative in _SOURCES:
            self.lua.execute((lua_dir / relative).read_text(encoding="utf-8"))
        self.lua.execute((HERE / "game_env.lua").read_text(encoding="utf-8"))
        self._load_shop_card_creation(lua_dir)
        self._load_item_prototypes(lua_dir)
        self.globals: Any = self.lua.globals()

    def _load_shop_card_creation(self, lua_dir: Path) -> None:
        """Load ``create_card_for_shop`` from the UI definitions, without the UI.

        The rest of that file builds menus and screens, so only this function is
        taken; the price tags it attaches (``create_shop_card_ui``) are absorbed.
        """
        source = (lua_dir / "functions/UI_definitions.lua").read_text(encoding="utf-8")
        start = source.index("function create_card_for_shop(area)")
        end = source.index("function create_shop_card_ui", start)
        self.lua.execute(source[start:end] + "\nfunction create_shop_card_ui() end\n")

    def _load_item_prototypes(self, lua_dir: Path) -> None:
        """Build ``G.P_CENTERS`` and the item pools with the game's own code.

        ``Game:init_item_prototypes`` also reads the player's profile from disk, so
        only the parts that define items and build their pools are run.
        """
        source = (lua_dir / "game.lua").read_text(encoding="utf-8")
        start = source.index("function Game:init_item_prototypes()")
        definitions_end = source.index("self.P_CENTER_POOLS = {", start)
        pools_start = source.index("self.P_CENTER_POOLS = {", start)
        pools_end = source.index("self:save_progress()", pools_start)
        tail_start = source.index("for k, v in pairs(self.P_STAKES) do", pools_end)
        tail_end = source.index("function Game:load_profile", tail_start)
        body = (
            source[start:definitions_end]
            + source[pools_start:pools_end]
            + source[tail_start:tail_end]
        )
        self.lua.execute(
            body.replace("function Game:init_item_prototypes()", "function harness_items(self)", 1)
        )
        self.lua.execute("harness_items(G)")

    # Building situations.

    def table(self, value: Any) -> Any:
        """Convert nested Python dicts and lists into Lua tables."""
        if isinstance(value, dict):
            return self.lua.table_from({key: self.table(item) for key, item in value.items()})
        if isinstance(value, list | tuple):
            return self.lua.table_from([self.table(item) for item in value])
        return value

    def playing_card(
        self,
        uid: int,
        rank: int,
        suit: str,
        *,
        enhancement: str | None = None,
        edition: str | None = None,
        seal: str | None = None,
        debuffed: bool = False,
        perma_bonus: int = 0,
        position: int = 0,
    ) -> Any:
        spec: dict[str, Any] = {
            "id": uid,
            "x": position,
            "center": enhancement or "c_base",
            "value": _VALUES[rank],
            "suit": _SUITS[suit],
            "playing_card": uid,
            "debuff": debuffed,
        }
        if edition:
            spec["edition"] = _EDITIONS[edition]
        if seal:
            spec["seal"] = seal
        card = self.globals.harness_card(self.table(spec))
        card.ability.perma_bonus = perma_bonus
        return card

    def item(
        self,
        uid: int,
        key: str,
        *,
        edition: str | None = None,
        debuffed: bool = False,
        ability: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
        position: int = 0,
    ) -> Any:
        """A joker or consumable. ``ability`` and ``extra`` override its values."""
        spec: dict[str, Any] = {"id": uid, "x": position, "center": key, "debuff": debuffed}
        if edition:
            spec["edition"] = _EDITIONS[edition]
        if ability:
            spec["ability"] = ability
        if extra:
            spec["extra"] = extra
        return self.globals.harness_card(self.table(spec))

    def new_game(self, **setup: Any) -> None:
        self.globals.harness_new_game(self.table(setup))

    def add(self, area: str, card: Any) -> None:
        self.globals.harness_add(area, card)

    def run_events(self) -> None:
        self.globals.harness_run_events()

    @property
    def game(self) -> Any:
        return self.globals.G.GAME

    def area(self, name: str) -> list[Any]:
        cards = self.globals.G[name].cards
        return [cards[i] for i in range(1, len(cards) + 1)]


def _flatten_ability(ability: Any) -> dict[str, Any]:
    """The values the engine keeps for a joker, read from the game's ``ability`` table."""
    values: dict[str, Any] = {"mult": ability.mult, "x_mult": ability.x_mult}
    extra = ability.extra
    if hasattr(extra, "items"):
        for key, value in sorted(extra.items()):
            values[f"extra.{key}"] = value
    elif extra is not None:
        values["extra"] = extra
    if ability.extra_value:
        values["extra_value"] = ability.extra_value
    return values


def _apply_values(card: Any, values: dict[str, Any]) -> None:
    """Set engine-style values (``"extra.chips"``, ``"mult"``) on a game card."""
    for name, value in values.items():
        if name.startswith("extra."):
            card.ability.extra[name.removeprefix("extra.")] = value
        elif name == "extra":
            card.ability.extra = value
        else:
            card.ability[name] = value


def score(rules: LuaRules, scenario: dict[str, Any]) -> dict[str, Any]:
    """Play a scoring scenario through the game's ``evaluate_play``.

    Returns the hand's score, the money afterwards, every joker's values, and the
    uids of destroyed played cards.
    """
    rules.new_game(
        dollars=scenario["dollars"],
        probability=scenario["probability"],
        hands_left=scenario["hands_left"],
        discards_left=scenario["discards_left"],
        levels=scenario["levels"],
        seed=str(scenario.get("seed", "")),
    )
    tarots_used = scenario.get("tarots_used", 0)
    rules.globals.G.GAME.consumeable_usage_total = rules.table(
        {
            "tarot": tarots_used,
            "planet": 0,
            "spectral": 0,
            "tarot_planet": tarots_used,
            "all": tarots_used,
        }
    )
    played = []
    for position, card in enumerate(scenario["played"]):
        played.append(_playing_card(rules, card, position))
        rules.add("play", played[-1])
    for position, card in enumerate(scenario["held"]):
        rules.add("hand", _playing_card(rules, card, position))
    jokers = []
    for position, joker in enumerate(scenario["jokers"]):
        item = rules.item(
            joker["uid"], joker["key"], edition=joker.get("edition"), position=position
        )
        _apply_values(item, joker.get("values", {}))
        jokers.append(item)
        rules.add("jokers", item)
    rules.globals.G.FUNCS.evaluate_play()
    rules.run_events()
    return {
        "score": float(rules.game.chips),
        "dollars": int(rules.game.dollars),
        "jokers": [_flatten_ability(item.ability) for item in jokers],
        "destroyed": [
            card["uid"]
            for card, game_card in zip(scenario["played"], played, strict=True)
            if game_card.harness_destroyed
        ],
    }


def _playing_card(rules: LuaRules, card: dict[str, Any], position: int) -> Any:
    return rules.playing_card(
        card["uid"],
        card["rank"],
        card["suit"],
        enhancement=card.get("enhancement"),
        edition=card.get("edition"),
        seal=card.get("seal"),
        debuffed=card.get("debuffed", False),
        position=position,
    )


def end_of_round(rules: LuaRules, scenario: dict[str, Any]) -> dict[str, Any]:
    """Run the game's ``end_round`` and ``evaluate_round`` for a scenario.

    Returns the money afterwards, the cash-out total, the jokers left with their
    values, the consumables created, and the pool flags.
    """
    boss = scenario["boss"]
    rules.new_game(
        dollars=scenario["dollars"],
        probability=scenario["probability"],
        hands_left=scenario["hands_left"],
        discards_left=scenario["discards_left"],
        blind_type="Boss" if boss else "Small",
        blind_chips=100,
        blind_dollars=5 if boss else 3,
    )
    game = rules.globals.G.GAME
    game.chips = 100 if scenario["beaten"] else 50
    game.last_hand_played = scenario["last_hand"]
    for position, card in enumerate(scenario["held"]):
        rules.add("hand", _playing_card(rules, card, position))
    jokers = []
    for position, joker in enumerate(scenario["jokers"]):
        item = rules.item(
            joker["uid"], joker["key"], edition=joker.get("edition"), position=position
        )
        _apply_values(item, joker.get("values", {}))
        jokers.append(item)
        rules.add("jokers", item)
    rules.globals.end_round()
    rules.run_events()
    payout = None
    if scenario["beaten"]:
        rules.globals.G.FUNCS.evaluate_round()
        rules.run_events()
        rows = rules.globals.harness.rows
        payout = next(rows[i].dollars for i in range(1, len(rows) + 1) if rows[i].name == "bottom")
    remaining = {item.ID for item in rules.area("jokers")}
    return {
        "dollars": int(game.dollars),
        "payout": payout,
        "jokers": {
            joker["uid"]: _flatten_ability(item.ability)
            for joker, item in zip(scenario["jokers"], jokers, strict=True)
            if item.ID in remaining
        },
        "consumables": [item.config.center.key for item in rules.area("consumeables")],
        "pool_flags": {key: bool(value) for key, value in sorted(game.pool_flags.items())},
    }


def discard(rules: LuaRules, scenario: dict[str, Any]) -> dict[str, Any]:
    """Discard the selected hand cards with the game's own discard function.

    ``scenario["selected"]`` holds the hand positions of the discarded cards.
    Returns the money, the jokers' values and the consumables afterwards.
    """
    rules.new_game(
        dollars=scenario["dollars"],
        probability=scenario["probability"],
        hands_left=2,
        discards_left=scenario["discards_left"],
    )
    for position, card in enumerate(scenario["held"]):
        rules.add("hand", _playing_card(rules, card, position))
    hand = rules.globals.G.hand
    for position in scenario["selected"]:
        hand.highlighted[len(hand.highlighted) + 1] = hand.cards[position + 1]
    jokers = []
    for position, joker in enumerate(scenario["jokers"]):
        item = rules.item(
            joker["uid"], joker["key"], edition=joker.get("edition"), position=position
        )
        _apply_values(item, joker.get("values", {}))
        jokers.append(item)
        rules.add("jokers", item)
    rules.globals.G.FUNCS.discard_cards_from_highlighted()
    rules.run_events()
    return {
        "dollars": int(rules.game.dollars),
        "jokers": [_flatten_ability(item.ability) for item in jokers],
        # A Purple Seal creates a random Tarot: compare the kind, not the card.
        "consumables": [item.ability.set for item in rules.area("consumeables")],
    }


_STATES = {
    "selecting_hand": "SELECTING_HAND",
    "shop": "SHOP",
    "blind_select": "BLIND_SELECT",
    "pack_open": "SPECTRAL_PACK",
}
_RANK_IDS = {"Jack": 11, "Queen": 12, "King": 13, "Ace": 14}


def _describe_card(card: Any) -> dict[str, Any]:
    """A playing card's rank, suit and modifiers, read back from the game."""
    center = card.config.center.key
    edition = card.edition and card.edition.type
    return {
        "uid": int(card.ID),
        "rank": int(card.base.id),
        "suit": str(card.base.suit)[0],
        "enhancement": None if center == "c_base" else center,
        "edition": f"e_{edition}" if edition else None,
        "seal": card.seal,
    }


def use_consumable(rules: LuaRules, scenario: dict[str, Any]) -> dict[str, Any]:
    """Use a consumable with the game's own ``use_consumeable``.

    The scenario gives the hand and the selected positions, the consumable and
    any other consumables and jokers owned, the phase and the money. In the
    ``pack_open`` phase the consumable is picked from an open Spectral pack.
    Returns whether the game allows the use and, if it does, everything the use
    changed.
    """
    rules.new_game(
        dollars=scenario["dollars"],
        probability=scenario["probability"],
        hands_left=2,
        discards_left=2,
        levels=scenario.get("levels", {}),
    )
    globals_ = rules.globals
    game = globals_.G.GAME
    game.last_tarot_planet = scenario.get("last_tarot_planet")
    globals_.G.STATE = globals_.G.STATES[_STATES[scenario["phase"]]]
    for position, card in enumerate(scenario["held"]):
        rules.add("hand", _playing_card(rules, card, position))
    hand = globals_.G.hand
    for position in scenario["selected"]:
        hand.highlighted[len(hand.highlighted) + 1] = hand.cards[position + 1]
    jokers = []
    for position, joker in enumerate(scenario["jokers"]):
        item = rules.item(
            joker["uid"], joker["key"], edition=joker.get("edition"), position=position
        )
        _apply_values(item, joker.get("values", {}))
        jokers.append(item)
        rules.add("jokers", item)
    for position, other in enumerate(scenario.get("consumables", [])):
        rules.add("consumeables", rules.item(500 + position, other))
    used = rules.item(999, scenario["consumable"])
    from_pack = scenario["phase"] == "pack_open"
    if from_pack:
        globals_.G.pack_cards = globals_.harness.area("pack")
        rules.add("pack_cards", used)
    else:
        rules.add("consumeables", used)
    rules.run_events()
    # The game refreshes derived values (selection limits, Temperance's payout,
    # the Wheel of Fortune's targets) in its per-frame update.
    used.update(used, 0)
    # G.FUNCS.use_card also refuses a use that Card:check_use rejects (Ankh with
    # full joker slots).
    if not used.can_use_consumeable(used) or used.check_use(used):
        return {"usable": False}
    area = globals_.G.pack_cards if from_pack else globals_.G.consumeables
    area.remove_card(area, used)
    used.use_consumeable(used, area)
    for item in rules.area("jokers"):
        item.calculate_joker(item, rules.table({"using_consumeable": True, "consumeable": used}))
    rules.run_events()
    levels = game.hands
    return {
        "usable": True,
        "hand": [_describe_card(card) for card in rules.area("hand")],
        "levels": {name: int(levels[name].level) for name in sorted(levels)},
        "dollars": int(game.dollars),
        "hand_size": int(globals_.G.hand.config.card_limit),
        "consumables": [item.config.center.key for item in rules.area("consumeables")],
        "jokers": [
            {"key": item.config.center.key, "edition": bool(item.edition)}
            for item in rules.area("jokers")
        ],
        "joker_values": [_flatten_ability(item.ability) for item in jokers],
        "last_tarot_planet": game.last_tarot_planet,
    }


_POOLS = {"Joker1": ("Joker", 0.5), "Joker2": ("Joker", 0.8), "Joker3": ("Joker", 0.99)}


def shop(rules: LuaRules, scenario: dict[str, Any]) -> dict[str, Any]:
    """Check shop rules with the game's code: pools, prices, lasting joker effects,
    and selling.

    Every item counts as unlocked, the engine's default mode. Returns the keys each
    pool may offer, each joker's price and sell value, the run's numbers after the
    ``gained`` joker is added and after it is removed again, and the results of
    selling the joker at ``sell`` (money, jokers left, their values).
    """
    globals_ = rules.globals
    rules.lua.execute("for _, center in pairs(G.P_CENTERS) do center.unlocked = true end")
    rules.new_game(dollars=scenario["dollars"], probability=1, hands_left=3, discards_left=3)
    game = globals_.G.GAME
    game.inflation = scenario["inflation"]
    game.discount_percent = scenario["discount"]
    # Values a real run sets in Game:start_run.
    game.round_resets.discards = 3
    game.round_resets.reroll_cost = 5
    for name in scenario["played_hands"]:
        game.hands[name].played = 1
    for flag in scenario["pool_flags"]:
        game.pool_flags[flag] = True
    jokers = []
    for position, joker in enumerate(scenario["jokers"]):
        item = rules.item(
            joker["uid"], joker["key"], edition=joker.get("edition"), position=position
        )
        if joker.get("rental"):
            item.set_rental(item, True)
        _apply_values(item, joker.get("values", {}))
        item.set_cost(item)
        jokers.append(item)
        rules.add("jokers", item)
    for position, key in enumerate(scenario["consumables"]):
        rules.add("consumeables", rules.item(700 + position, key))

    pools = {}
    for name, (set_name, rarity) in {
        **_POOLS,
        "Tarot": ("Tarot", None),
        "Planet": ("Planet", None),
    }.items():
        pool, _ = globals_.get_current_pool(set_name, rarity)
        keys = [pool[i] for i in range(1, len(pool) + 1)]
        pools[name] = sorted(key for key in keys if key != "UNAVAILABLE")
    prices = [[int(item.cost), int(item.sell_cost)] for item in jokers]

    def run_numbers() -> dict[str, int]:
        current = game.current_round
        return {
            "hand_size": int(globals_.G.hand.config.card_limit),
            "discards_per_round": int(game.round_resets.discards),
            "discards_left": int(current.discards_left),
            "bankrupt_at": int(game.bankrupt_at),
            "free_rerolls": int(current.free_rerolls),
            "reroll_cost": int(current.reroll_cost),
        }

    gained = rules.item(800, scenario["gained"])
    gained.added_to_deck = False
    gained.add_to_deck(gained)
    rules.run_events()
    after_gain = run_numbers()
    gained.remove_from_deck(gained)
    rules.run_events()
    after_loss = run_numbers()

    globals_.G.shop_jokers = globals_.harness.area("shop")
    game.shop.joker_max = 2
    for _ in range(scenario["rerolls"]):
        button = rules.table({"config": {}})
        globals_.G.FUNCS.can_reroll(button)
        if button.config.button != "reroll_shop":
            break
        globals_.G.FUNCS.reroll_shop()
        rules.run_events()
    reroll_cost = int(game.current_round.reroll_cost)

    sold = rules.area("jokers")[scenario["sell"]] if scenario["sell"] is not None else None
    if sold is not None:
        globals_.G.FUNCS.sell_card(rules.table({"config": {"ref_table": sold}}))
        rules.run_events()
    remaining = {item.ID for item in rules.area("jokers")}
    return {
        "pools": pools,
        "prices": prices,
        "after_gain": after_gain,
        "after_loss": after_loss,
        "reroll_cost": reroll_cost,
        "dollars": int(game.dollars),
        "jokers": {
            joker["uid"]: _flatten_ability(item.ability)
            for joker, item in zip(scenario["jokers"], jokers, strict=True)
            if item.ID in remaining
        },
    }


_PACK_STATES = {
    "TAROT_PACK": "Arcana",
    "PLANET_PACK": "Celestial",
    "SPECTRAL_PACK": "Spectral",
    "STANDARD_PACK": "Standard",
    "BUFFOON_PACK": "Buffoon",
}


def open_pack(rules: LuaRules, scenario: dict[str, Any]) -> dict[str, Any]:
    """Open a booster pack in the shop with the game's own ``Card:open``.

    The scenario gives the pack, the ante, the money and the jokers owned. For
    Arcana and Spectral packs the hand the game draws is ``held``. Returns the
    pack phase, the number of choices, the money left, the cards created and
    whether the game lets each one be picked.
    """
    rules.new_game(
        dollars=scenario["dollars"],
        probability=1,
        hands_left=4,
        discards_left=3,
        ante=scenario["ante"],
        seed=scenario["seed"],
    )
    globals_ = rules.globals
    game = globals_.G
    game.STATE = game.STATES.SHOP
    game.pack_cards = globals_.harness.area("pack")
    for position, joker in enumerate(scenario["jokers"]):
        item = rules.item(100 + position, joker["key"], edition=joker.get("edition"))
        rules.add("jokers", item)
    pack = rules.item(900, scenario["pack"])
    pack.open(pack)
    rules.run_events()
    state_names = {int(number): name for name, number in game.STATES.items()}
    kind = _PACK_STATES[state_names[int(game.STATE)]]
    if kind in ("Arcana", "Spectral"):
        # Game:update_arcana_pack draws the hand once the pack is open.
        for position, card in enumerate(scenario["held"]):
            rules.add("hand", _playing_card(rules, card, position))
    cards = rules.area("pack_cards")
    pickable = []
    for card in cards:
        if card.ability.consumeable:
            # The per-frame update refreshes the lists of eligible jokers.
            card.update(card, 0)
            pickable.append(bool(card.can_use_consumeable(card)) and not card.check_use(card))
        else:
            button = rules.table({"config": {"ref_table": card}})
            game.FUNCS.can_select_card(button)
            pickable.append(button.config.button == "use_card")
    return {
        "kind": kind,
        "choices": int(game.GAME.pack_choices),
        "size": int(game.GAME.pack_size),
        "dollars": int(game.GAME.dollars),
        "cards": [_describe_pack_card(card) for card in cards],
        "pickable": pickable,
    }


def _describe_pack_card(card: Any) -> dict[str, Any]:
    if card.ability.set in ("Default", "Enhanced"):
        return {"set": "Playing", **_describe_card(card)}
    edition = card.edition and card.edition.type
    return {
        "set": str(card.ability.set),
        "key": str(card.config.center.key),
        "edition": f"e_{edition}" if edition else None,
    }
