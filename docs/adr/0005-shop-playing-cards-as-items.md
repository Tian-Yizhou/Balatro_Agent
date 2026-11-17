# 0005. A playing card for sale is an item that holds the card

- Status: accepted
- Date: 2026-11-16

## Context

With the vouchers Magic Trick and Illusion, the shop's card slots can hold
playing cards as well as jokers and consumables. The slots are a single ordered
list (`RunState.shop_cards`), and `buy(i)` names a position in it, so a playing
card has to sit in that list alongside the other cards.

Until now the list held only `Item`s, which carry a price, while a
`PlayingCard` has no price. In the game, every card in the shop is the same kind
of object, with a center (`c_base` or an enhancement), a front, and a cost.

## Decision

- `Item` gains an optional `playing_card` field. A playing card for sale is an
  `Item` whose `key` is the card's center (`c_base` or its enhancement), whose
  `playing_card` is the card itself, edition included, and whose `uid` is the
  card's uid.
- Its price follows the game's `Card:set_cost`: $1 before its edition, with the
  run's discount, like any other item.
- Buying it puts the card under the deck and needs no free slot.

## Consequences

- `shop_cards` stays a list of `Item`s, so prices, discounts and later price
  changes (the Coupon Tag) work the same for every card in the shop.
- Code that reads shop items must check `playing_card` before treating an item
  as a joker or consumable.
- The bought card keeps its uid, so a viewer can follow it from the shop into the
  deck.
