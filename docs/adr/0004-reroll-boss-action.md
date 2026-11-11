# 0004. Rerolling the boss blind is an action

- Status: accepted
- Date: 2026-11-11

## Context

The vouchers Director's Cut and Retcon add a button at blind selection that
replaces the ante's boss blind for $10. Director's Cut allows one reroll per ante,
and Retcon allows any number. The action table in plan §4.3 has no action for
this button, so these two vouchers could not be used.

The Boss Tag also rerolls the boss, but it does so on its own when the tag
triggers, with no click and no cost.

## Decision

- A new atomic action, `reroll_boss`, is legal at blind selection when a redeemed
  voucher allows a reroll and the player can pay for it.
- It follows the game's `G.FUNCS.reroll_boss`: it pays $10 and picks a new boss
  with the usual boss selection rule.
- The Boss Tag uses the same rule internally without being an action, and it
  costs nothing.
- Replays record `reroll_boss` like any other action.

## Consequences

- Director's Cut and Retcon work as in the game, and agents can learn when a
  reroll is worth $10.
- The action set grows by one, which the action codec (M7) includes.
