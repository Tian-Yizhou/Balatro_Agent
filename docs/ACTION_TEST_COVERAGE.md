# Action Test Coverage and Remaining Gaps

Status: based on `dev`, verified May 26, 2026.

This document tracks action coverage across the three interfaces involved in
playing or training against the project:

1. User-visible input and environment construction.
2. Agent output and training-wrapper forwarding.
3. Environment/core execution and state changes.

The current public environment action space is `Discrete(446)`. All 446
currently exposed action IDs have an end-to-end execution test. Actions that
exist only in core state, or are described as future agent functionality, are
listed separately as implementation gaps.

## Verified Test Run

Full suite, including the RLlib PPO smoke test:

```bash
python -m pytest -q
```

Result:

```text
1758 passed, 1 warning
```

The warning is emitted by Ray about a future change to accelerator environment
variable behavior when `num_gpus=0`; it is not a project test failure.

The PPO smoke test starts Ray and may require execution outside a restricted
sandbox because Ray inspects local process/CPU information.

## Layer 1: User-Visible Inputs

### Environment Creation and Session Controls

| User operation | Implementation | Coverage | Test file |
| --- | --- | --- | --- |
| Create preset environment (`easy`, `medium`, `hard`) | `balatro_gym.make()` / registered Gym envs | Covered | `test_public_api.py`, `test_env.py` |
| Create with explicit `GameConfig` | `balatro_gym.make(config=...)` | Covered | `test_public_api.py` |
| Create from YAML config | `GameConfig.from_file()`, `balatro_gym.make(config_path=...)` | Covered | `test_public_api.py` |
| Save YAML config | `GameConfig.to_yaml()` | Covered by round trip | `test_public_api.py` |
| Create deterministic seeded session | `balatro_gym.make(seed=...)`, `env.reset(seed=...)` | Covered | `test_public_api.py`, `test_env.py` |
| Create vector environments and step them | `balatro_gym.make_vec()` | Covered for synchronous and asynchronous vectorization | `test_public_api.py` |
| Save/resume a session | `save_state()`, `load_state()` | Covered previously | `test_seed_and_state.py` |
| Replay a shared seed ID | `from_seed_id()` | Covered previously | `test_seed_and_state.py` |

### User-Visible Game Choices

`GameStateRenderer` provides structured user/LLM-readable action choices. The
tests require visible action IDs to be accepted by the environment, so the
displayed choice and executed action cannot silently diverge.

| Visible user choice | Action IDs | Count | Coverage |
| --- | ---: | ---: | --- |
| Play selected cards | `0-217` | 218 | Every action ID rendered/forwarded/executed |
| Discard selected cards | `218-435` | 218 | Every action ID rendered/forwarded/executed |
| Buy item from shop | `436-438` | 3 | Both Joker slots and consumable slot executed |
| Sell owned Joker | `439-443` | 5 | Every exposed Joker slot executed |
| Reroll shop | `444` | 1 | Executed and cost checked |
| Skip shop | `445` | 1 | Executed and next blind transition checked |

Relevant test file: `tests/test_user_action_journeys.py`.

### Rejected User Choices

The user-facing action path also tests actions that must not take effect:

| Invalid choice | Expected behavior | Coverage |
| --- | --- | --- |
| Submit shop action during play phase | No state change; invalid-action penalty | Covered |
| Submit discard after discards are exhausted | No state change; invalid-action penalty | Covered |
| Buy item without enough money | No purchase; invalid-action penalty | Covered |
| Reroll without enough money | No reroll; invalid-action penalty | Covered |
| Sell from an empty Joker slot | No sale; invalid-action penalty | Covered |

## Layer 2: Agent Output and Training Paths

### Baseline Agents

| Agent path | Action behavior | Coverage |
| --- | --- | --- |
| `RandomAgent.act()` | Can choose any action allowed by the mask | All 446 public action IDs forwarded through agent to environment |
| `HeuristicAgent.act()` play | Plays when no useful/available discard path remains | Covered |
| `HeuristicAgent.act()` discard | Discards weak cards while searching for score | Covered |
| `HeuristicAgent.act()` buy | Buys an affordable shop item | Covered |
| `HeuristicAgent.act()` skip | Skips shop when nothing is affordable | Covered |
| `RandomAgent.run_episode()` | Own end-to-end episode runner | Covered |
| `HeuristicAgent.run_episode()` | Own end-to-end episode runner | Covered |

`HeuristicAgent` does not currently contain a policy branch for voluntarily
choosing `sell` or `reroll`. Those environment actions are covered through
`RandomAgent`, but they are not behavior of the heuristic strategy.

Relevant test file: `tests/test_user_action_journeys.py`.

### Shared Agent API

| Agent helper | Purpose | Coverage |
| --- | --- | --- |
| `Agent` protocol | Common `act()` / `reset()` contract | Baseline protocol conformance covered |
| `run_episode()` | Reset agent, loop `act -> step`, return final stats | Covered |
| `evaluate_agent()` | Run seeded episodes and aggregate metrics | Covered |

Relevant test file: `tests/test_agent_interfaces.py`.

### Structured LLM/User Rendering

| Component | Current role | Coverage |
| --- | --- | --- |
| `GameStateRenderer.render()` | Produce structured state and valid action IDs | Covered for initial state, play/shop actions, and pre-reset error |
| `GameStateRenderer.render_json()` | Serialize readable state for an LLM prompt | Covered |
| `ModelBackend` | Protocol for future model backend | No executable model behavior implemented |

Relevant test files: `tests/test_agent_interfaces.py`,
`tests/test_user_action_journeys.py`.

### RLlib Training Wrapper

| Training path | Coverage |
| --- | --- |
| `BalatroRLlibEnv.reset()` Dict observation and mask | Covered |
| `BalatroRLlibEnv.step()` forwarding | All 446 public action IDs forwarded through wrapper |
| PPO action-mask module construction | Covered |
| PPO algorithm build and one training iteration | Covered by slow smoke test |
| CLI config parsing/building | Covered |

Relevant test file: `tests/test_rllib.py`.

## Layer 3: Environment and Core Execution

### Public Environment Actions

| Environment action | Core execution target | Coverage |
| --- | --- | --- |
| `play` | `GameState.play_hand()` | Per action ID end-to-end plus scoring/core tests |
| `discard` | `GameState.discard()` | Per action ID end-to-end plus core tests |
| `buy` Joker | `GameState.shop_buy()` | Per public slot end-to-end |
| `buy` consumable | `GameState.shop_buy()` | Public consumable slot end-to-end |
| `sell` Joker | `GameState.shop_sell()` | Per public slot end-to-end |
| `reroll` | `GameState.shop_reroll()` | End-to-end including money cost |
| `skip` | `GameState.shop_skip()` | End-to-end including phase transition |

### Reward Result Path

| Reward behavior | Coverage |
| --- | --- |
| Default invalid-action penalty | Covered |
| Default score progress reward and clipping | Covered |
| Default blind completion and terminal rewards | Covered |
| Sparse reward behavior | Covered |
| Custom callable reward receives action validity context from `env.step()` | Covered |

Relevant test file: `tests/test_rewards.py`.

### Existing Core Mechanic Coverage

The pre-existing suite also covers cards/decks, poker evaluation, hand levels,
jokers, blinds, shop internals, consumable effects, enhanced scoring, state
serialization, seed replay, and data-recording wrappers:

```text
tests/test_card.py
tests/test_hand_evaluator.py
tests/test_hand_levels.py
tests/test_joker.py
tests/test_blind.py
tests/test_shop.py
tests/test_consumable.py
tests/test_game_state.py
tests/test_integration.py
tests/test_seed_and_state.py
tests/test_wrappers.py
```

## Fixes Required by Testing

Two environment defects were identified and fixed while building this
coverage:

1. Configured seed was ignored by `env.reset()` unless callers repeated the
   seed as a `reset(seed=...)` argument. `balatro_gym.make(seed=N)` now creates
   reproducible initial states when the user follows the documented
   `env.reset()` call pattern. An explicit reset seed still overrides the
   configured seed.
2. Masked shop actions inside a valid numeric action range were reported as
   successful. Unaffordable `buy`/`reroll` and invalid `sell` now return
   invalid-action behavior and receive the configured penalty.

Implementation file: `balatro_gym/envs/balatro_env.py`.

## Not Yet Exposed or Not Yet Implemented

These items cannot be completed as user-to-agent-to-environment action tests
without adding product functionality.

| Missing path | Existing partial implementation | Required implementation |
| --- | --- | --- |
| Use a purchased consumable through the environment action space | `GameState.use_consumable()` exists and consumable effects have core tests | Add consumable-use actions and target-card encoding to `BalatroEnv.action_space`, mask, observation/rendering, baseline/RL agent paths |
| Human text command to action ID | Renderer displays structured choices only | Add `HumanAgent` or command parser and validation/error handling |
| LLM response to action ID | `ModelBackend` protocol and state renderer exist | Add `LLMAgent`, prompt format, response parser, invalid-response fallback |
| Heuristic intentional `sell` decision | Environment supports sell | Add strategy branch if desired behavior is part of baseline design |
| Heuristic intentional `reroll` decision | Environment supports reroll | Add strategy branch if desired behavior is part of baseline design |

### Consumable Gap Details

The environment permits buying consumables (`action 438`) and observations
include owned consumables, but no action currently lets an agent use them.
This means a complete gameplay action model is still missing:

```text
user sees owned consumable
-> user/agent selects use + target cards
-> environment applies consumable
-> observation/reward reflects result
```

Core tests verify individual consumable mechanics, but those tests are not a
replacement for this absent environment/agent action path.

## Waiting for Future Tests

These tests should be added after the corresponding missing functionality is
implemented. They are not executable against the current interfaces.

| Future test area | Depends on | Required test scenarios |
| --- | --- | --- |
| Consumable action end-to-end path | Environment actions for `use_consumable` and card targets | User sees each valid consumable action; agent forwards it; environment applies it; invalid target/empty slot/full-slot cases are rejected |
| Human command path | `HumanAgent` or command parser | Valid command for every public action type; invalid card/slot/phase input; displayed choice maps to the executed action |
| LLM decision path | `LLMAgent` and response parser | Valid parsed action; malformed response fallback; masked invalid response rejection; full episode execution |
| Heuristic sell strategy | A heuristic `sell` policy decision | Affordability/capacity scenario triggers sell; money and inventory change correctly |
| Heuristic reroll strategy | A heuristic `reroll` policy decision | Affordable/no-useful-offer scenario triggers reroll; cost and new offerings are observed |

## Test Files Added or Extended

| File | Purpose |
| --- | --- |
| `tests/test_user_action_journeys.py` | User-readable choice -> agent -> environment action coverage and rejected actions |
| `tests/test_agent_interfaces.py` | Public agent protocol, episode/evaluation helpers, renderer |
| `tests/test_public_api.py` | Environment factories, config YAML, seed behavior, vector environment |
| `tests/test_rewards.py` | Default/sparse/custom reward pathways |
| `tests/test_rllib.py` | Extended with all-action RLlib wrapper forwarding |
| `pytest.ini` | Registers the existing `slow` PPO smoke test marker |
