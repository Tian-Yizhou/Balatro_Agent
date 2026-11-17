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
- using every Planet, Tarot and Spectral card (`Card:can_use_consumeable`,
  `Card:check_use` and `Card:use_consumeable`), from the consumable slots, the shop
  or an open pack;
- the shop: item pools (`get_current_pool`), prices, lasting joker effects, rerolls
  and selling;
- opening every booster pack (`Card:open`): its size, choices and price, the kinds
  of cards it holds, and which of them can be picked (`G.FUNCS.can_select_card`).

The rule checks live in `rules.py`, with the stand-ins in `game_env.lua` and the
random situations in `scoring_scenarios.py` and `consumable_scenarios.py`. Where
the game draws something random (the Emperor's cards, Sigil's suit, the cards in a
pack), what must hold whatever the draw is compared instead: the kind and number of
cards, one suit for the whole hand, and so on, since the engine's random draws
differ from the game's.

`game_env.lua` replaces the game's graphics, sound, input and save code with
stand-ins and keeps its event queue, so the rule code runs unchanged.
