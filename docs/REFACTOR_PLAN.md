# Balatro-Agent v2: Refactor Plan

> **Status:** agreed, ready for M0 · **Branch:** `refactor/v2` · **Last updated:** 2026-11-10
>
> Decisions are listed in §2. Any new open questions go in §12, and nothing open is
> built until it is agreed.

---

## 1. Goal and scope

Build a simulation engine that reproduces the **rules** of Balatro 1.0.1o, using the
original Lua source as the specification. Wrap it as a research environment for RL
and other agents. The environment must:

1. run at scale across many processes, nodes and GPUs;
2. let any simulated game be restored and replayed step by step, so a human can see
   exactly what the agent did.

**In scope:** every non-cosmetic rule of a standard run:
- 150 jokers;
- 22 tarots, 12 planets and 18 spectrals;
- enhancements, editions and seals;
- 5 booster pack kinds, in every size;
- 32 vouchers and 24 tags;
- 28 boss blinds;
- 15 decks;
- 8 stakes, plus the eternal / perishable / rental stickers;
- endless mode;
- unlock progression, as a mode that can be switched on or off.

**Later:** challenges, modded content, and a bridge that lets agents play the real game (§9).

**Out of scope:** graphics, animation, sound, menus, Steam and achievements.

**Non-goal:** seed parity with the real game. The same seed does not have to produce
the same run as in the real game. Rules must match exactly. Random outcomes only need
the same probabilities as the game, such as shop rarity weights and the odds on cards
like Lucky.

---

## 2. Decisions so far

| # | Decision | Status |
|---|----------|--------|
| D1 | No seed parity with the real game. We use our own deterministic RNG, whose state is part of every snapshot (§4.5). | decided |
| D2 | The engine is written in Python. A Rust core is reconsidered only if the M2 benchmark shows it is needed. | decided |
| D3 | Distribute **games**, not steps (§6). | decided |
| D4 | The agent interface uses UI-like atomic actions (§4.3). | decided |
| D5 | Reordering jokers is an action. | decided |
| D6 | Unlock state is a switchable mode: everything unlocked, or progression (§4.7). | decided |
| D7 | Endless mode is in scope. | decided |
| D8 | Challenges and modded content come later. | decided |
| D9 | The refactor happens on a new branch: first the repo and development rules, then the engine. | decided |
| D10 | Replays store the initial state, the actions, and a keyframe snapshot each round (§7). | decided |
| D11 | Moving hand cards and sorting the hand are actions (§4.3). | decided |
| D12 | The profile tracks discovered items as well as unlocks; discovery affects gameplay (§4.7). | decided |
| D13 | Repository rules: short-lived branches into a protected `main`; one `balatro_gym` package with subpackages; `pyright`; legacy code removed from the v2 tree; CI required with zero approvals, the owner merges; CHANGELOG edited by hand (§10). | decided |
| D14 | Progression uses careers with a `max_concurrent_runs` setting. The default is 1, a sequential career (§4.7). | decided |
| D15 | Tooling: `uv` for Python, Python 3.11 or newer; `rustup` + `maturin` if Rust is added later (§10.6). | decided |
| D16 | The Lua source stays on the owner's machine only, ignored through `.git/info/exclude`. It has been removed from the public repo's entire history (§10.1). | done 2026-11-10 |

The ADR index (`docs/adr/README.md`) lists these decisions. Later decisions get their own ADRs.

---

## 3. Architecture

```
      agents: RL policies · LLM agents · search (MCTS) · humans
          │                      │                       │
   balatro_gym.env        balatro_gym.serve       balatro_gym.replay
   obs encoders,          dispatcher ("lobby"),   record, re-simulate,
   action codec,          remote step server      CLI + web viewer
   Gymnasium, vec env
          └──────────────────────┼───────────────────────┘
                          balatro_gym.sim
        state · rules (phases, scoring) · content · RNG · events · profiles
```

| Package | Responsibility | Runtime deps |
|---|---|---|
| `sim` | The pure engine: `(state, action) → (state, events)` | standard library only |
| `replay` | Replay file format, recorder, re-simulation, text renderer | `sim` (+ zstd, optional) |
| `env` | Observation encoders, action codec, Gymnasium env, vectorized env, rewards, difficulty presets | `sim`, `replay`, numpy, gymnasium |
| `serve` | Dispatcher, remote step server, Ray integration | `sim`, `env`, `replay` (+ ray / grpc, optional) |
| `agents/` | Baselines and training code (later, ported from `dev`) | everything above + torch / ray |

**Dependency rule:** `sim` imports nothing from the other packages, and there are no
import cycles. Tests enforce this.

All four are subpackages of a single `balatro_gym` package (D13).

---

## 4. Engine (`balatro_gym.sim`)

### 4.1 State model
State is plain dataclasses with no stored functions and no reference cycles. Every
card and item has a stable `uid`.

- **Run:**
  - seed and RNG counters;
  - ante, round, money;
  - stake, deck, redeemed vouchers, tags held;
  - hand levels and how often each hand has been played;
  - pool flags (for example, Gros Michel has gone extinct);
  - the probability multiplier changed by Oops! All 6s;
  - endless flag;
  - the unlock set this run uses.
- **Areas (ordered lists):**
  - the draw pile (its order is hidden from the agent), hand, played cards and discard pile;
  - jokers and consumables;
  - shop slots for cards, vouchers and boosters;
  - the contents of an open pack.
- **Playing card:**
  - rank and suit;
  - enhancement, edition and seal;
  - stickers and debuff flag;
  - permanent bonuses (for example, Hiker's chips);
  - whether it is currently selected.
- **Joker or consumable:**
  - its key (for example, `j_blueprint`);
  - edition and stickers;
  - its changing internal values (for example, the Ride the Bus counter);
  - sell value.
- **Phase:** the current phase plus phase-specific data (for example, how many picks
  remain in an open pack).

Serialization uses `to_dict()` / `from_dict()`: JSON for people to read, and
msgpack + zstd for storage. The schema carries a version number. A snapshot is a few KB.

### 4.2 Phases
The phases mirror the game's `G.STATES`, with animations removed:

```
BLIND_SELECT ─select→ SELECTING_HAND ─(play|discard)→ … ─blind beaten→ ROUND_EVAL ─cash_out→ SHOP ⇄ PACK_OPEN
     ↑  └─skip (tag)                                    └─out of hands→ GAME_OVER          │
     └─────────────────────────────────────── end_shop ────────────────────────────────────┘
ante-8 boss beaten → WIN ─continue→ endless (BLIND_SELECT) | end_run
```

A pack can also open outside the shop, for example from Charm Tag during blind
selection. The Lua code runs effects through an animation event queue. We flatten
that queue into synchronous steps but keep the order in which effects happen.

### 4.3 Actions: UI-like atomic actions (D4, D5)
These are the canonical actions. Agents send them, and replays record them.

| Where | Actions |
|---|---|
| Wherever the game allows | `toggle_card(i)`, `move_card(i, j)`, `move_joker(i, j)`, `sell_joker(i)`, `sell_consumable(i)`, `use_consumable(i)` (the selected cards are its targets) |
| Blind select | `select_blind`, `skip_blind` |
| Selecting hand | `play`, `discard`, `sort_by_rank`, `sort_by_suit` |
| Round evaluation | `cash_out` |
| Shop | `buy(i)`, `buy_and_use(i)`, `redeem_voucher(i)`, `buy_pack(i)`, `reroll`, `end_shop` |
| Pack open | `toggle_card(i)` (selects targets for tarots and spectrals), `pick(i)`, `skip_pack` |
| Win | `continue_endless`, `end_run` |

- **Legal actions:** `legal_actions(state)` lists every valid action. The engine
  rejects anything else, and the env layer masks it out.
- **Macro helpers:** shortcuts such as `play_cards([1, 3, 5])` and
  `use_on(slot, targets)` expand into atomic actions in a fixed way. They are for
  tests, scripted agents and LLM tools. Replays always store the atomic actions.
- **Card order matters for scoring.** Played cards score from left to right by their
  position in hand. This affects Photograph, Hanging Chad, and whether ×mult comes
  before or after +mult. That is why `move_card` and the sort actions exist (D11).
- **Hand size is not fixed.** It can grow well past 8, so the engine allows any hand
  size. The env layer sets a configurable cap and masks the unused positions.

### 4.4 Scoring and content
- **Scoring:** a faithful port of `G.FUNCS.evaluate_play`, step by step:
  1. Detect the poker hand (`evaluate_poker_hand`, including the Four Fingers,
     Shortcut and Smeared Joker variants).
  2. Take base chips and mult from the hand's level.
  3. Score each played card, including retriggers.
  4. Apply effects from cards held in hand.
  5. Apply joker effects, including joker editions, from left to right.
  6. Apply boss-blind changes.
  7. Destroy cards where needed (for example, Glass).
  8. Run after-scoring effects.
- **Effect dispatch:** effects run on the same contexts that `Card:calculate_joker`
  uses: `setting_blind`, `first_hand_drawn`, `before`, `individual`, `repetition`,
  `other_joker`, `joker_main`, `after`, `pre_discard`, `discard`,
  `destroying_card`, `cards_destroyed`, `remove_playing_cards`, `end_of_round`,
  `buying_card`, `selling_card`, `selling_self`, `reroll_shop`, `ending_shop`,
  `open_booster`, `skipping_booster`, `skip_blind`, `using_consumeable`,
  `playing_card_added`, `debuffed_hand` and `game_over`. Blueprint and Brainstorm
  copy effects by running the copied joker's effect within these same contexts.
- **Content is data plus effect functions:**
  - **Data** (costs, rarities, config values, pools, unlock conditions) is extracted
    once from `game.lua` by a script in `tools/extract_content/`. Only the resulting
    numbers and keys are committed. Lua code and the game's localization text are
    never copied; we write our own short descriptions.
  - **Effects:** one Python function per item, grouped by mechanism. Each docstring
    cites its Lua source (for example, `Lua: card.lua Card:calculate_joker "Blueprint"`).

### 4.5 RNG and determinism (D1)
- **Keyed, counter-based streams.** Each draw is `mix(seed, stable_hash(stream_key), counter[stream_key])`,
  and then that stream's counter goes up by 1. `mix` is a 64-bit integer mixer such
  as splitmix64. `stable_hash` must give the same value on every machine. Python's
  built-in `hash()` is salted per process, so it is never used.
- **Stream keys** follow the game's own scheme: `shuffle_<ante>`, `shop_type_<ante>`,
  `rarity_<ante>`, `boss`, `lucky_mult`, `wheel_of_fortune`, and so on.
  - Because each stream is separate, shop rolls don't depend on how many cards were drawn.
  - So two agents on the same seed see largely the same shops, bosses and tags.
  - That makes comparisons between agents, and "what if" branches, much cleaner.
- **RNG state is tiny:** `{seed, {stream_key: counter}}`. It is stored in every
  snapshot, so restoring a snapshot reproduces the game's future exactly.
- **Wall clock and unseeded randomness** are never used anywhere in `sim`.
- **Two ways to copy a game for planning agents:**
  - `clone(state)` is an exact copy. It knows the real future, so it is for oracle
    baselines and debugging only.
  - `determinize(state, seed)` keeps everything the agent can see but re-randomizes
    hidden information (deck order, future rolls). Planning agents such as MCTS or an
    LLM looking ahead use this, so they don't see the true future.

### 4.6 Events
Every step emits typed events, for example:
- `CardsDrawn`
- `HandScored(type, chips, mult, total)`
- `EffectApplied(source=j_blueprint→j_cavendish, x_mult=3)`
- `MoneyChanged(+4, reason=interest)`
- `ItemCreated`, `CardDestroyed`
- `UnlockAchieved`

They feed the replay viewer, reward shaping, LLM narration, unlock tracking and
debugging. Events are derived data: a state can be restored without them.

### 4.7 Profiles and unlock progression (D6)
- **`Profile`** holds the unlocked items, discovered items and career stats (the
  equivalents of the game's `G.PROFILES[...]`).
- **Mode:** `all_unlocked` (the default) or `progression`. Progression starts from a
  fresh profile or from any saved one.
- **How a run uses the profile:**
  - It reads the profile at the start, which decides the item pools and which decks
    and stakes are available.
  - Locked items are skipped and re-rolled exactly the way the game does it, so
    progression really changes what the agent sees.
  - **Discovery also affects gameplay (D12).** Some skip tags enter the pool only once
    a linked item has been discovered: Rare Tag needs Blueprint, and the Foil / Holo /
    Polychrome / Negative tags need their edition (21 such `requires` links in
    `game.lua`). Several unlock conditions also count discoveries. A run marks items
    as discovered at the same moments the game does.
- **How the profile is updated:**
  - When a run ends, a pure function computes the new profile:
    `update_profile(profile, run_events) → profile'`.
  - The engine never saves anything itself. The caller, meaning the dispatcher or the
    agent code, decides where profiles are stored.
- **Unlock conditions:**
  - The Lua code defines 77 of them, of about 30 kinds: career counters, winning with
    a given deck or stake, reaching an ante, discovering N items, and so on.
  - A few depend on things outside a run, such as `continue_game` (resuming a saved
    game). An ADR will decide each of these: unlock automatically, never unlock, or
    handle specially.
- **Careers (D14).** A *career* is a chain of runs that share one profile, like one
  human player's save file. Each run starts from the profile as it stands when the
  run begins.
  - The dispatcher stores careers and assigns every run to one of them.
  - A career has one setting, `max_concurrent_runs`: how many of its runs may be in
    progress at once.
  - When several runs overlap, each one's changes are merged back when it ends. Merging
    can never conflict: unlock and discovery sets only grow, counters add up, and
    "best" stats take the maximum. An overlapping run may start from a slightly
    out-of-date profile.
  - **Default: `max_concurrent_runs = 1`, a sequential career.** Runs follow one after
    another, exactly like a human. Parallelism then comes from running many careers
    side by side; for example, 256 workers advance 256 careers.
  - The policy network is shared across all careers; only the profiles differ.
  - Two choices can wait until M5: whether the agent observes its own profile, and
    when careers reset.

### 4.8 Endless mode (D7)
- Score targets above ante 8 use the game's formula from `get_blind_amount`.
- After winning, the agent chooses `continue_endless` or `end_run`.
- Run config options: `max_ante` (stop early, for example for a curriculum) and
  `endless` on/off.

### 4.9 What the agent can see
`observe(state) → Observation` returns only what a human player can see:
- hand, jokers, consumables;
- money, hands and discards left, hand levels;
- the current blind, the upcoming blinds and boss, and the tags offered;
- the shop and vouchers;
- the remaining deck **composition**: counts, not order.

It never includes deck order, RNG state or future shop contents. An `oracle=True`
flag exists for research baselines that should see everything.

---

## 5. Environment layer (`balatro_gym.env`)
- **Observation encoders:**
  - (a) An entity/token encoder: sets of cards, jokers and shop items as feature
    rows, plus global features. It handles variable sizes and suits attention-based
    policies.
  - (b) A flat vector, as the MLP baseline.
  - (c) Text/JSON for LLM agents.
- **Action codec:** maps the atomic actions to `Discrete(N)` plus a mask. N follows
  from the caps (maximum hand size, joker slots, shop slots, …). Moves may be encoded
  as two steps ("pick up", then "drop"), so the number of actions grows linearly with
  hand size rather than quadratically. The exact layout is defined in M7.
- **Gymnasium env and vectorized env:** many games run in one process, with
  auto-reset.
- **Reward functions are pluggable,** building on `dev`'s `RewardContext` idea but fed
  by engine events.
- **Difficulty presets** set the content pools, ante limit, stake and deck (keeping
  `dev`'s plug-in idea).
- **Every env records a replay by default** (it's cheap), and has a hook so the agent
  can attach annotations.

---

## 6. Distribution (`balatro_gym.serve`) (D3)
- **`GameSpec`** describes one game:
  `{game_id, rules_version, config, seed, profile_ref?, start_snapshot?}`.
- **The dispatcher, or "lobby":**
  - hands out specs: training seed streams, curriculum schedules, fixed evaluation sets;
  - collects results and replays (or pointers to them);
  - stores progression careers and their profiles (§4.7).
  - It exists in two forms: a Ray actor (RLlib env runners ask it for a spec on each
    `reset()`), and a small HTTP/gRPC service when Ray isn't used.
- **Workers** run games in-process, usually many at once, and report back. A game
  never makes a network call per step.
- **The remote step server** is an optional adapter for clients where latency doesn't
  matter: LLM agents, humans, the web viewer and non-Python clients. It uses the same
  engine and protocol.
- **Games can move between nodes.** A game can be snapshotted and resumed on any node,
  for example after the cluster preempts a job.
- **Storage:** replays go to a shared filesystem or object store, indexed by `game_id`.
  Any game played anywhere can be rebuilt from its id.

---

## 7. Replay and inspection (`balatro_gym.replay`)
- **A replay file contains:**
  - a header: `game_id`, `rules_version`, schema version, config, seed, profile;
  - the initial state;
  - the atomic actions;
  - **keyframe** snapshots at the start of each round (the interval is configurable,
    or keyframes can be turned off);
  - optional per-step **annotations**: the agent's top-k action probabilities, its
    value estimate, or an LLM's reasoning text.
- **Using a replay:**
  - `Replay.state_at(step)` loads the nearest keyframe and re-applies the actions
    after it.
  - `Replay.branch(step)` returns a live game you can keep playing ("what if it had
    discarded here?").
  - `Game.from_snapshot(...)` resumes from any saved state.
- **Drift check:** on load, the engine re-simulates the game and checks each keyframe
  still matches. A mismatch means the engine's behavior changed since recording.
- **Viewers:** a CLI text renderer comes first (M1); a web viewer comes later (M9).
  The web viewer steps through rounds and shows the hand, jokers, shop, a scoring
  breakdown built from events, and the agent's annotations.
- **Golden replays** are committed under `tests/golden/` and lock in behavior: any
  change in their outcome fails CI.

This covers both things asked for: replaying from any recorded state, and seeing
every state of a finished game.

---

## 8. Validation
1. **Lua differential tests** for functions that stand alone (`evaluate_poker_hand`,
   `get_blind_amount`, …). These run the original Lua under LuaJIT, with the game
   functions it calls replaced by simple stand-ins (`tools/lua_harness/`). The harness
   reads the local Lua copy, which is never committed.
2. **Scenario tests:** at least one per content item (each joker, consumable, voucher,
   tag, boss, deck and stake). Each sets up a specific state and checks the score,
   money and side effects that the Lua code implies.
3. **Invariant fuzzing:** random and heuristic agents play millions of steps nightly.
   Checks:
   - no exceptions;
   - snapshots survive a save/load round trip unchanged;
   - replays are deterministic;
   - values stay within sane bounds;
   - every legal action executes, and every illegal one is rejected.
4. **Statistical checks:** shop rarity, pack contents and the odds of probabilistic
   effects match the game's stated probabilities (chi-square tests over many draws).
5. **Golden replays** guard against regressions.
6. **Later, through the real-game bridge:** load states from the real game and compare
   the deterministic outcomes (scores, money).

---

## 9. Future: real-game bridge (not scheduled)
A mod for the real game, built on the Lovely mod loader, would:
- export the real game's visible state in our `Observation` schema;
- accept our atomic actions.

That would let trained agents play the real game and give us ground-truth checks.

It shapes two choices now:
- **Observations describe what a player sees,** not the engine's internals.
- **Atomic actions mirror UI clicks.**

---

## 10. Repository refactor and development rules (M0)

### 10.1 Layout
```
.github/
  workflows/ci.yml            repository guards · lint · type-check · tests
  workflows/pr-title.yml      PR-title format check
  workflows/nightly.yml       fuzzing · benchmarks
  ISSUE_TEMPLATE/             bug_report · feature_request · content_port
  pull_request_template.md
  dependabot.yml
docs/
  REFACTOR_PLAN.md            this file
  adr/                        0001-…, one file per decision
src/balatro_gym/
  sim/  replay/  env/  serve/
tests/
  sim/  replay/  env/  serve/  golden/
tools/
  extract_content/  lua_harness/  bench/
agents/                       later, ported from dev
pyproject.toml · lockfile · .pre-commit-config.yaml
README.md · CONTRIBUTING.md · CLAUDE.md · CHANGELOG.md · LICENSE
```
- **Legacy code (D13):** v1 code and docs are removed from the new tree. They stay
  available in git history, on the `dev` branch, and under a `v1-legacy` tag.
- **Lua source (D16):**
  - It lives only on the owner's machine, in the working directory, ignored through
    `.git/info/exclude` (not `.gitignore`). The tools find it through
    `BALATRO_LUA_DIR`.
  - On 2026-11-10 it was removed from every commit of the public repo with
    `git filter-repo`, and all branches were force-pushed.
  - **Guard:** CI fails if any `.lua` file is tracked outside an allow-listed
    directory. The allow-list is empty until the real-game bridge (§9) needs one.

### 10.2 Branches (D13)
- **`main` is protected:** changes arrive only through PRs, CI must pass, and
  force-pushes are blocked.
- **`refactor/v2`** carries M0 only. Once M0 merges, `main` *is* v2, and all later
  work follows the normal short-lived-branch flow.
- **`dev` is retired:** kept as an archive, with no new work.

### 10.3 Commits and pull requests
The rules live in [`CONTRIBUTING.md`](../CONTRIBUTING.md), which is the single source
of truth. In short:
- Commit headers and PR titles follow Conventional Commits with a required scope.
  CI checks PR titles.
- Messages and descriptions state what changed and why, precisely and briefly. They
  leave out unfinished work, TODOs and open concerns.
- PRs are squash-merged. The PR title and description become the commit on `main`.

### 10.4 GitHub settings
Applied in M0 through the GitHub API:
- **Merging:** squash only. The squash commit takes the PR title and description.
  Branches are deleted after merge.
- **`main` ruleset:** PRs required with 0 approvals (one maintainer); the CI and
  PR-title checks required; force-pushes and deletion blocked. Admin bypass is for
  emergencies only.
- **Labels:** created to match the issue templates (`type/*`, `area/*`).

### 10.5 Issues
- **Templates:**
  - *Bug report:* must attach a replay or snapshot that reproduces the bug.
  - *Feature request.*
  - *Content port:* the item key, its Lua reference, the expected behavior, and test cases.
- **Labels:** `area/*`, `type/*`, `content/*`, `good-first-issue`.
- **Milestones:** GitHub milestones mirror M0–M9 below.

### 10.6 Code standards
- **Tooling:**
  - Python 3.11 or newer, managed with `uv` and a lockfile (D15). CI runs on
    3.11–3.13. LuaJIT for the test harness comes from `apt`.
  - Full type hints in `sim`.
  - `ruff` for linting and formatting.
  - Strict type checking in `sim` with `pyright` (D13).
- **Inside `sim`:**
  - no third-party runtime dependencies;
  - no global mutable state;
  - no wall clock;
  - no unseeded randomness;
  - no built-in `hash()` for anything that is saved.
- **Ported rules:** cite the Lua source. Never copy Lua code or the game's text word
  for word.
- **Tests:**
  - every content item has scenario tests;
  - every bug fix adds a regression test or a golden replay;
  - coverage in `sim` stays at 90% or higher.
- **Decisions:** recorded as ADRs (context, decision, consequences).
- **`CLAUDE.md`** summarizes these rules for AI assistants.

### 10.7 Versioning
- **The package uses SemVer.** Releases get `vX.Y.Z` tags and GitHub Releases.
- **Every replay records two versions:**
  - `RULES_VERSION`, bumped whenever game behavior changes;
  - `STATE_SCHEMA_VERSION`, bumped whenever the state format changes.
- **CHANGELOG.md** follows the Keep a Changelog format and is edited by hand in each PR (D13).

---

## 11. Milestones

| # | Milestone | Done when |
|---|---|---|
| **M0** | Repo refactor and development rules (§10): layout, `pyproject`, tooling, `.github/`, CONTRIBUTING, CLAUDE.md, ADR index, legacy removed, Lua untracked | CI is green on the empty skeleton; branch protection is on; merged to `main` |
| **M1** | Engine foundations: state model, keyed RNG, snapshot/restore, events, the atomic and macro action framework, replay format, recorder and text renderer, Lua harness | Snapshots round-trip; replays are deterministic; Lua differential tests pass for hand evaluation and blind amounts |
| **M2** | Core loop with plain cards: blind select, round, the scoring pipeline with full context dispatch, cash out, a weighted shop (jokers, tarots, planets), game over, win, endless | Random agents finish full games; **throughput benchmark → Python/Rust checkpoint** |
| **M3** | Card modifiers, consumables, packs and the pack phase, vouchers, tags with skipping, all bosses, decks, stakes and stickers | Scenario tests for every item |
| **M4** | All 150 jokers, in batches by mechanism: passive → conditional → scaling → retrigger → copy → economy → create/destroy → probability | Scenario tests for every joker |
| **M5** | Profiles and unlock progression (the all-unlocked default exists from M2) | Progression runs from a fresh profile; the profile-merge rules are tested |
| **M6** | Validation hardening: nightly fuzzing, statistical checks, more golden replays | Nightly job is green |
| **M7** | Env layer: encoders, action codec, Gymnasium env, vectorized env, rewards, difficulty presets | Benchmark results; random and heuristic baselines |
| **M8** | Distribution: dispatcher, remote server, Ray integration, progression careers | Demo across several processes, with per-game replays retrievable |
| **M9** | Web replay viewer | Steps through any stored replay |

M7 can start in parallel once M2 has fixed the engine API. After M9: port the agents
from `dev`, then challenges, mods and the real-game bridge.

---

## 12. Open questions

None. Every question raised during planning was settled on 2026-11-10; the
decisions are D1 and D10–D16 in §2. Later questions get their own ADRs.
