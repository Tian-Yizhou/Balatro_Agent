# 0003. Items without an implemented effect are unavailable

- Status: accepted
- Date: 2026-11-10

## Context

Content arrives over several milestones: a starter set of jokers in M2, the rest
in M4. Until then, a shop that offers every item would sell jokers that silently
do nothing, and agents would learn from a game that differs from the real one in
ways no observation reveals.

The game already has a mechanism for items that may not appear in a run, such as
locked or banned items: they stay in their pool as unavailable, and a draw that
lands on one is re-rolled.

## Decision

- An item whose effect is not implemented is unavailable, exactly as if it were
  locked: pools keep it as unavailable and re-roll draws that land on it.
- Rarity and card-type rolls are unchanged, so their probabilities stay exact.
- The registry of implemented effects (`sim.effects.JOKERS`, and the consumable
  registry when it exists) is the single source of truth for availability.
- Implementing an item makes it available. That changes shops, so it bumps
  `RULES_VERSION`.

## Consequences

- Every item an agent can obtain works as in the game.
- Until all content is implemented, the odds of each *specific* item differ from
  the game's, because its pool is smaller.
- Content can be added one item at a time without special cases in the shop.
