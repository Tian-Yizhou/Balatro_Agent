# Content extraction

Writes `src/balatro_gym/sim/data/content.json` from the game's item definitions
(`Game:init_item_prototypes` in `game.lua`): costs, rarities, config values, pool
flags and unlock conditions for every joker, consumable, voucher, pack, edition,
enhancement and deck, plus tags, seals, stakes and blinds. Sprite positions,
colours and all localized text are left out.

```bash
BALATRO_LUA_DIR=/path/to/source uv run python tools/extract_content/extract.py
```

Re-run it only if the game version changes; the output is committed.
