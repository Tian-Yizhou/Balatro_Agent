# Lua differential harness

Runs functions from the original game's Lua source under LuaJIT (from the `lupa`
package) and compares them with the engine. See ADR 0002.

The game's source is never committed. Point `BALATRO_LUA_DIR` at your local copy,
the directory that contains `card.lua` and `functions/`.

```bash
# Compare the engine with the Lua directly, on many random inputs.
BALATRO_LUA_DIR=/path/to/source uv run pytest tests/sim/test_lua_live.py

# Regenerate the committed test vectors in tests/sim/data/.
BALATRO_LUA_DIR=/path/to/source uv run python tools/lua_harness/generate_vectors.py
```

Without `BALATRO_LUA_DIR`, the live tests are skipped and the suite checks the
committed vectors instead.

Covered so far:

- poker hand evaluation (`evaluate_poker_hand` with Four Fingers, Shortcut and
  Smeared Joker; Wild, Stone and debuffed cards) and blind score targets
  (`get_blind_amount` for every scaling level), in `harness.py`;
- the game's own scoring (`G.FUNCS.evaluate_play`), round end (`end_round` and
  `G.FUNCS.evaluate_round`) and discards (`G.FUNCS.discard_cards_from_highlighted`),
  run on random situations with jokers, enhancements, editions and seals;
- using every Planet and Tarot card (`Card:can_use_consumeable` and
  `Card:use_consumeable`), and the shop: item pools (`get_current_pool`), prices,
  lasting joker effects, rerolls and selling.

The rule checks live in `rules.py`, with the stand-ins in `game_env.lua` and the
random situations in `scoring_scenarios.py` and `consumable_scenarios.py`. Where
the game draws a random card (the Emperor, a Purple Seal), the kind and number of
cards are compared, since the engine's random draws differ from the game's.

`game_env.lua` replaces the game's graphics, sound, input and save code with
stand-ins and keeps its event queue, so the rule code runs unchanged.
