-- A minimal stand-in for the running game, enough to call the game's own rule
-- functions (evaluate_play, Card:calculate_joker, Card:use_consumeable, ...).
--
-- Everything visual is a no-op. The game's event queue is kept, and
-- harness_run_events() runs it to completion the way the game's update loop would.
-- This file is our code; it only defines stand-ins and glue.

harness = {}

function harness.noop() end

-- Stand-ins for UI, sound, saving and progress tracking.
for _, name in ipairs({
  "play_sound", "delay", "update_hand_text", "card_eval_status_text", "juice_card",
  "juice_card_until", "attention_text", "check_for_unlock", "check_and_set_high_score",
  "inc_career_stat", "set_hand_usage", "play_area_status_text", "highlight_card",
  "discover_card", "unlock_card", "save_run", "stop_use", "set_joker_usage",
  "ease_background_colour_blind", "set_consumeable_usage_profile",
  "alert_no_space", "draw_card",
}) do _G[name] = harness.noop end

-- Cash-out rows are recorded so tests can read the payout.
harness.rows = {}
function add_round_eval_row(row) table.insert(harness.rows, row) end

function localize(args) return type(args) == "table" and (args.key or "") or args end
function HEX() return {} end


-- The event queue: events are queued and run later, in order.
harness.queue = {}
Event = function(config) return config end

local function run_event(event)
  if event.trigger == "ease" then
    local value = event.ease_to
    if event.func then value = event.func(value) end
    event.ref_table[event.ref_value] = value
    return true
  end
  if event.func then return event.func() end
  return true
end

function harness_run_events()
  local guard = 0
  while #harness.queue > 0 do
    guard = guard + 1
    if guard > 100000 then error("event queue does not drain") end
    local event = table.remove(harness.queue, 1)
    local done = run_event(event)
    if done == false and not event.blocking_wait then
      -- An event that returns false waits for a condition; retry it later.
      event.retries = (event.retries or 0) + 1
      if event.retries < 50 then table.insert(harness.queue, event) end
    end
  end
end

-- A card area: just an ordered list of cards.
local Area = {}
Area.__index = Area
function harness.area(kind, limit)
  return setmetatable({cards = {}, highlighted = {}, T = {x = 0, y = 0, w = 1, h = 1},
    config = {type = kind, card_limit = limit or 1000}}, Area)
end
function Area:emplace(card) table.insert(self.cards, card); card.area = self end
function Area:remove_card(card)
  for i = #self.cards, 1, -1 do
    if self.cards[i] == card then table.remove(self.cards, i) end
  end
  for i = #self.highlighted, 1, -1 do
    if self.highlighted[i] == card then table.remove(self.highlighted, i) end
  end
  return card
end
function Area:unhighlight_all() self.highlighted = {} end
function Area:change_size(delta) self.config.card_limit = self.config.card_limit + delta end
function Area:sort() end
function Area:set_ranks() end
function Area:align_cards() end
function Area:shuffle() end

G = G or {}
G.E_MANAGER = {}
function G.E_MANAGER:add_event(event, queue, front)
  if front then table.insert(harness.queue, 1, event) else table.insert(harness.queue, event) end
end
function G.E_MANAGER:clear_queue() harness.queue = {} end
G.SETTINGS = {GAMESPEED = 1, profile = 1, tutorial_complete = true, SOUND = {volume = 0}}
G.PROFILES = {{career_stats = {}, high_scores = {}, consumeable_usage = {}}}
-- The input controller is UI; everything about it is absorbed (see harness.anything).
G.C = setmetatable({}, {__index = function() return setmetatable({}, {__index = function() return {} end}) end})
G.FUNCS = G.FUNCS or {}
G.STATES = {SELECTING_HAND = 1, HAND_PLAYED = 2, DRAW_TO_HAND = 3, SHOP = 5, PLAY_TAROT = 6,
  BLIND_SELECT = 7, ROUND_EVAL = 8, TAROT_PACK = 9, PLANET_PACK = 10, SPECTRAL_PACK = 15,
  STANDARD_PACK = 17, BUFFOON_PACK = 18, GAME_OVER = 4, NEW_ROUND = 19}
G.STAGES = {RUN = 1}
G.CARD_W, G.CARD_H = 1, 1.4
G.I = {CARD = {}}
G.ARGS = {}
G.STAGE = G.STAGES.RUN

-- A blind with no effect, as the plain blinds have.
harness.blind = {}
function harness.blind:debuff_hand() return false end
function harness.blind:modify_hand(cards, hands, text, mult, chips) return mult, chips, false end
function harness.blind:press_play() return false end
function harness.blind:debuff_card() end
function harness.blind:stay_flipped() return false end
function harness.blind:get_type() return self.type or "Small" end
function harness.blind:juice_up() end
for _, name in ipairs({"set_blind", "disable", "defeat", "drawn_to_hand", "set_text",
  "change_colour", "alert_debuff", "wiggle"}) do
  harness.blind[name] = harness.noop
end

harness.back = {}
function harness.back:trigger_effect() return nil end

-- Build a card from a plain description (see harness.py LuaRules.card).
function harness_card(spec)
  local card = setmetatable({}, Card)
  card.T = {x = spec.x or 0, y = 0, w = 1, h = 1, r = 0}
  card.VT, card.CT = card.T, card.T
  card.config = {card = {}}
  card.params = {playing_card = spec.playing_card}
  -- Sprites and buttons are UI; any access to them is absorbed.
  card.children = setmetatable({}, {__index = function() return harness.anything end})
  card.states = {visible = true, drag = {is = false}, collide = {}, hover = {}, click = {}, focus = {}}
  card.ID, card.sort_id = spec.id, spec.id
  card.unique_val = 1 - spec.id / 1603301
  card.base_cost, card.extra_cost, card.cost, card.sell_cost = 0, 0, 0, 0
  card:set_ability(G.P_CENTERS[spec.center], true)
  if spec.value then
    card:set_base({value = spec.value, suit = spec.suit, name = ""}, true)
  else
    card:set_base({}, true)
  end
  if spec.edition then card:set_edition({[spec.edition] = true}, true, true) end
  if spec.seal then card.seal = spec.seal end
  if spec.ability then
    for key, value in pairs(spec.ability) do card.ability[key] = value end
  end
  if spec.extra then
    if type(card.ability.extra) == "table" then
      for key, value in pairs(spec.extra) do card.ability.extra[key] = value end
    else
      card.ability.extra = spec.extra.value
    end
  end
  card.debuff = spec.debuff or false
  card.base_cost = G.P_CENTERS[spec.center].cost or 0
  card:set_cost()
  card.playing_card = spec.playing_card
  card.added_to_deck = true
  return card
end

-- Fresh run state with the given areas and numbers.
function harness_new_game(setup)
  harness.queue = {}
  harness.rows = {}
  G.GAME = Game.init_game_object(G)
  G.GAME.selected_back = harness.back
  if setup.seed then
    G.GAME.pseudorandom.seed = setup.seed
    G.GAME.pseudorandom.hashed_seed = pseudohash(setup.seed)
  end
  G.GAME.blind = setmetatable({type = setup.blind_type or "Small", chips = setup.blind_chips or 1,
    dollars = setup.blind_dollars or 0,
    name = "", boss = setup.blind_type == "Boss", disabled = false, config = {blind = {}}},
    {__index = harness.blind})
  G.GAME.dollars = setup.dollars
  G.GAME.probabilities.normal = setup.probability
  G.GAME.current_round.hands_left = setup.hands_left
  G.GAME.current_round.discards_left = setup.discards_left
  G.GAME.current_round.hands_played = setup.hands_played or 0
  G.GAME.current_round.discards_used = setup.discards_used or 0
  G.GAME.round_resets.ante = setup.ante or 1
  for name, level in pairs(setup.levels or {}) do
    local hand = G.GAME.hands[name]
    hand.level = level
    hand.mult = math.max(hand.s_mult + hand.l_mult * (level - 1), 1)
    hand.chips = math.max(hand.s_chips + hand.l_chips * (level - 1), 0)
  end
  G.play = harness.area("play")
  G.hand = harness.area("hand", 8)
  G.jokers = harness.area("joker", setup.joker_slots or 5)
  G.consumeables = harness.area("joker", setup.consumable_slots or 2)
  G.deck = harness.area("deck")
  G.discard = harness.area("discard")
  G.playing_cards = {}
  G.STATE = G.STATES.SELECTING_HAND
end

function harness_add(area_name, card)
  G[area_name]:emplace(card)
  if card.playing_card then table.insert(G.playing_cards, card) end
end

-- Visual card methods. Destroying a card only marks it, as the game's own
-- removal happens through dissolve animations.
function Card:set_sprites()
  self.children.center = self.children.center or harness.anything
  self.children.back = self.children.back or harness.anything
end
for _, name in ipairs({"juice_up", "update_alert", "start_materialize", "flip", "hard_set_T",
  "calculate_parrallax", "highlight", "set_card_area"}) do
  Card[name] = harness.noop
end
function Card:start_dissolve() self.harness_destroyed = true; if self.area then self.area:remove_card(self) end end
function Card:shatter() self.harness_destroyed = true; if self.area then self.area:remove_card(self) end end
function Card:remove() self.harness_removed = true; if self.area then self.area:remove_card(self) end end

-- The game's ease_* functions update the HUD as well; these keep only the state
-- change and its timing (queued, unless ``instant``).
local function eased(apply, instant)
  if instant then
    apply()
  else
    G.E_MANAGER:add_event(Event({trigger = "immediate", func = function() apply(); return true end}))
  end
end
function ease_dollars(mod, instant)
  eased(function() G.GAME.dollars = G.GAME.dollars + (mod or 0) end, instant)
end
function ease_hands_played(mod, instant)
  eased(function() G.GAME.current_round.hands_left = G.GAME.current_round.hands_left + mod end, instant)
end
function ease_discard(mod, instant)
  eased(function()
    local current = G.GAME.current_round.discards_left
    G.GAME.current_round.discards_left = current + math.max(-current, mod)
  end, instant)
end
function ease_chips(mod) G.GAME.chips = mod end

-- An object that accepts any field access or call, for UI elements the rule code
-- pokes at (``G.HUD_blind:get_UIE_by_ID(...):juice_up()``).
harness.anything = setmetatable({}, {})
harness_anything = harness.anything
getmetatable(harness.anything).__newindex = function() end
getmetatable(harness.anything).__index = function() return harness.anything end
getmetatable(harness.anything).__call = function() return harness.anything end
G.HUD, G.HUD_blind, G.ROOM = harness.anything, harness.anything, harness.anything
G.CONTROLLER = setmetatable({locked = false, locks = {}, interrupt = {}},
  {__index = function() return harness.anything end})

-- Profile statistics the round code updates; they don't affect the run.
G.PROFILES = {{
  career_stats = setmetatable({}, {__index = function() return 0 end}),
  high_scores = setmetatable({}, {__index = function(table, key) table[key] = {amt = 0}; return table[key] end}),
  consumeable_usage = {},
}}
function G:save_settings() end
function G:save_progress() end
G.FILE_HANDLER = {}
function ease_ante(mod)
  G.E_MANAGER:add_event(Event({trigger = "immediate", func = function()
    G.GAME.round_resets.ante = G.GAME.round_resets.ante + mod
    return true
  end}))
end
