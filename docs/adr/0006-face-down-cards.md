# 0006. Face-down cards are hidden from planning copies

- Status: accepted
- Date: 2026-11-16

## Context

Some boss blinds hide information. The Wheel, The House, The Mark and The Fish
deal cards into the hand face down, and Amber Acorn turns the jokers face down
and shuffles them. The player can still select, play and move these cards, but
cannot see which cards they are, or which joker is where.

The engine knows every card. `engine.determinize` makes the copy that planning
agents search, so that they learn nothing the player could not know; so far it
only reshuffles the draw pile.

## Decision

- `PlayingCard.face_down` and `Item.face_down` record what the player cannot
  see. Played cards are revealed, and cards leaving the hand are face up.
- `determinize` deals every face-down card in hand again from the unseen cards
  (the draw pile and the face-down cards together), keeping each slot's
  selection, and shuffles the order of the face-down jokers.
- Observations (M7) will hide face-down cards and jokers. The replay viewer,
  which is for people inspecting a run, shows them with a `~` mark.

## Consequences

- A planner searching a determinized copy can't see through face-down cards,
  matching what a player knows.
- Without face-down cards `determinize` behaves exactly as before.
