"""Extract item data (costs, rarities, config values, pools) from the game's source.

Runs the game's own item definitions under LuaJIT and writes only the resulting
keys and numbers to ``src/balatro_gym/sim/data/content.json``. Sprite positions,
colours and all localized text are dropped.

Usage::

    BALATRO_LUA_DIR=/path/to/game/source uv run python tools/extract_content/extract.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parents[2] / "src" / "balatro_gym" / "sim" / "data" / "content.json"

# Presentation-only fields that are never exported.
DROPPED = {
    "pos",
    "soul_pos",
    "atlas",
    "alerted",
    "start_alerted",
    "colour",
    "boss_colour",
    "label",
    "vars",
    "debuff_text",
    "defeated",
}

STUBS = r"""
HEX = function() return {} end
localize = function(text) return text end
G = {C = setmetatable({}, {__index = function() return {} end})}
Game = {}
"""


def _to_python(lua: Any, value: Any) -> Any:
    if lua.lua_type(value) != "table":
        return value
    keys = list(value.keys())
    if (
        keys
        and all(isinstance(k, int) for k in keys)
        and sorted(keys) == list(range(1, len(keys) + 1))
    ):
        return [_to_python(lua, value[i]) for i in range(1, len(keys) + 1)]
    return {str(k): _to_python(lua, value[k]) for k in sorted(keys, key=str) if k not in DROPPED}


def main() -> int:
    lua_dir = os.environ.get("BALATRO_LUA_DIR")
    if not lua_dir or not (Path(lua_dir) / "game.lua").is_file():
        print("Set BALATRO_LUA_DIR to the directory that holds the game's Lua source.")
        return 1
    from lupa import luajit21  # type: ignore[import-untyped]

    source = (Path(lua_dir) / "game.lua").read_text(encoding="utf-8")
    start = source.index("function Game:init_item_prototypes()")
    stop = source.index("self.P_CENTER_POOLS = {", start)
    body = source[start:stop] + "\nend\n"

    runtime = luajit21.LuaRuntime(unpack_returned_tuples=True)
    runtime.execute(STUBS)
    runtime.execute(body)
    game = runtime.eval("{}")
    runtime.globals().Game.init_item_prototypes(game)

    def export(name: str) -> dict[str, Any]:
        table = _to_python(luajit21, game[name])
        # Entries without a set are placeholders for undiscovered items.
        return {
            key: table[key] for key in sorted(table) if name != "P_CENTERS" or "set" in table[key]
        }

    content = {
        "generated_by": "tools/extract_content/extract.py",
        "game_version": "1.0.1o",
        "centers": export("P_CENTERS"),
        "tags": export("P_TAGS"),
        "seals": export("P_SEALS"),
        "stakes": export("P_STAKES"),
        "blinds": export("P_BLINDS"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(content, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(content['centers'])} centers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
