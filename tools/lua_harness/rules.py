"""Run the game's own rule code (scoring, jokers, consumables) under LuaJIT.

``LuaRules`` loads the game's source with the stand-ins in ``game_env.lua`` and
builds game situations from plain descriptions, so tests can put the same
situation into the original game and into the engine and compare the results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["CardDescription", "LuaRules", "discard", "end_of_round", "score"]

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
        self._load_item_prototypes(lua_dir)
        self.globals: Any = self.lua.globals()

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
        "consumables": [item.config.center.key for item in rules.area("consumeables")],
    }
