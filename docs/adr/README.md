# Architecture decision records

An architecture decision record (ADR) captures one significant design decision:
the context, the decision, and its consequences. Write one whenever a decision
changes the architecture, a public interface, a data format, or the development
process.

## Format

File name: `NNNN-short-title.md`, numbered in order from `0001`.

```markdown
# NNNN. Title

- Status: proposed | accepted | superseded by NNNN
- Date: YYYY-MM-DD

## Context
The forces at play and the problem to solve.

## Decision
What we decided, stated precisely.

## Consequences
What becomes easier or harder as a result.
```

An accepted ADR is not edited. To change a decision, write a new ADR that
supersedes it.

## Baseline decisions

The decisions taken while planning v2 are recorded in
[REFACTOR_PLAN.md §2](../REFACTOR_PLAN.md#2-decisions-so-far):

| ID | Decision | Details |
|---|---|---|
| D1 | Own deterministic RNG; no seed parity with the real game | Plan §4.5 |
| D2 | Engine written in Python | Plan §2 |
| D3 | Distribute games, not steps | Plan §6 |
| D4 | UI-like atomic actions | Plan §4.3 |
| D5 | Reordering jokers is an action | Plan §4.3 |
| D6 | Switchable unlock mode | Plan §4.7 |
| D7 | Endless mode in scope | Plan §4.8 |
| D8 | Challenges and modded content later | Plan §1 |
| D9 | Refactor on a new branch, repository rules first | Plan §10 |
| D10 | Replays store initial state, actions and keyframes | Plan §7 |
| D11 | Moving and sorting hand cards are actions | Plan §4.3 |
| D12 | Profiles track discovered items | Plan §4.7 |
| D13 | Repository rules | Plan §10, CONTRIBUTING.md |
| D14 | Careers with `max_concurrent_runs` | Plan §4.7 |
| D15 | uv for Python tooling | Plan §10.6 |
| D16 | Original game source never tracked | Plan §10.1 |

## Records

| ADR | Decision |
|---|---|
| [0001](0001-mutating-step-with-clone.md) | Steps change the state in place; copies are explicit |
| [0002](0002-lua-harness-with-lupa.md) | Lua differential tests run on lupa, with committed test vectors |
| [0003](0003-unimplemented-items-unavailable.md) | Items without an implemented effect are unavailable |
