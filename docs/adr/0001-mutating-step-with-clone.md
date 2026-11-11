# 0001. Steps change the state in place; copies are explicit

- Status: accepted
- Date: 2026-11-10

## Context

The plan described the engine as a pure function `(state, action) → (state, events)`.
Taken literally, every step would copy the whole state, which costs time on every
step of every game, while only search and "what if" analysis need copies. Both
designs are equally deterministic and replayable, because all randomness lives in
the state's own RNG.

## Decision

- `step(state, action)` changes `state` in place and returns the list of events.
  An illegal action raises `IllegalActionError` and leaves the state unchanged.
- `clone(state)` returns an exact, independent copy, RNG included. It is defined as
  restoring a snapshot of the state, so a clone and a restored snapshot are always
  the same thing.
- `determinize(state, seed)` returns a copy that keeps everything the player can see
  and re-randomizes hidden information (draw pile order, future random draws).
- `Game` wraps a state with these operations for convenience.
- Events hold plain values (uids, keys, numbers), never references into the state,
  so they stay valid after later steps.

## Consequences

- No copy per step; copies are paid for only by the callers that need them, such
  as MCTS or replay branching.
- Code that must keep an earlier state clones it first.
- The plan's description of the engine as pure now means "deterministic, with no
  hidden inputs", not "never mutates".
