# 0002. Lua differential tests run on lupa, with committed test vectors

- Status: accepted
- Date: 2026-11-10

## Context

Differential tests compare the engine with the original game's Lua functions. The
plan used LuaJIT from `apt`, which needs root access on each machine. The game's
source is never committed (D16), so CI cannot run the original code at all.

## Decision

- The harness in `tools/lua_harness/` runs the game's Lua under LuaJIT 2.1 from the
  `lupa` package, a pinned development dependency. It reads the maintainer's local
  copy of the source from `BALATRO_LUA_DIR`.
- `tools/lua_harness/generate_vectors.py` writes test vectors to `tests/sim/data/`.
  A vector holds only inputs and the outputs the game computes for them; no game code
  or text.
- The vectors are committed, and the test suite checks the engine against them on
  every run, including in CI.
- Live tests (`tests/sim/test_lua_live.py`) compare against the Lua directly on many
  more random inputs. They run only where `BALATRO_LUA_DIR` is set and skip
  elsewhere.

## Consequences

- No system packages are needed; `uv sync` installs everything.
- CI checks parity with the game without having its source.
- When the harness covers a new function, its vectors must be regenerated on the
  maintainer's machine and committed.
