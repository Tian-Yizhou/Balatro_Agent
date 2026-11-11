# Contributing

This document sets the rules for branches, commits and pull requests (PRs).
They apply to every change, whether a person or an AI assistant writes it.

**The guiding principle:** each commit and PR states *what changed and why*,
precisely and briefly. Nothing else.

---

## 1. Workflow

There are two paths to `main`. Pick by the size and risk of the change.

**Pull request: for significant changes.** A new feature or capability, a change to
game behavior, a public interface or a data format, or a large refactor.

1. Create a branch from an up-to-date `main` (§2).
2. Commit as each logical unit of work is completed (§3.8).
3. When the work is complete and tested, open one PR into `main`. CI must pass.
4. The maintainer reviews and **squash-merges** the PR, and the branch is deleted.

**Direct commit: for small, low-risk changes.** Documentation fixes, configuration
tweaks, and similar housekeeping. Commit to `main` and push; CI runs on the push.
If a branch for related work is open, put the change on that branch instead.

`main` is never force-pushed.

---

## 2. Branches

### 2.1 When to open a branch

Branches carry work that will become a PR (§1). Opening one is an engineering
judgment. A branch holds one **coherent change**: a piece of work that makes sense
to review, test and merge as a whole.

- Related small tasks can share a branch, for example several jokers with the same
  mechanism.
- Unrelated changes go on separate branches, so each PR stays focused.
- Large work is split into a series of branches that each leave `main` working.
- New work starts on a new branch from `main`. A branch is never reused after it
  has been merged.
- There is no long-lived integration branch: `main` is always releasable, and
  tags mark stable snapshots.

### 2.2 Naming and upkeep

Format: `<type>/<short-description>`

- `<type>` is the commit type (§3.2) that best describes the branch as a whole.
- `<short-description>` is 2–5 lowercase words joined by hyphens.
- Examples: `feat/sim-scoring-pipeline`, `fix/shop-reroll-cost`, `docs/adr-rng`.
- To catch up with `main`, rebase onto it. After rebasing a pushed branch, push with
  `git push --force-with-lease`.

---

## 3. Commit messages

### 3.1 Format

We follow [Conventional Commits 1.0](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>(<scope>): <subject>

[body]

[trailers]
```

### 3.2 Types

| Type | Use for |
|---|---|
| `feat` | A new capability: an engine feature, a content item, an env option |
| `fix` | Correcting wrong behavior |
| `perf` | The same behavior, faster or with less memory |
| `refactor` | Restructuring code without changing behavior |
| `test` | Adding or changing tests only |
| `docs` | Documentation only |
| `build` | Packaging, dependencies, the lockfile |
| `ci` | CI workflows and GitHub configuration |
| `chore` | Maintenance that fits no other type |
| `revert` | Reverting an earlier commit |

### 3.3 Scopes

Every commit names a scope: the area it is about. The scope names the area, never
the kind of change, so documentation for the engine is `docs(sim)`, not `docs(docs)`.

| Scope | Area |
|---|---|
| `sim` | `balatro_gym.sim`, the engine |
| `replay` | `balatro_gym.replay` |
| `env` | `balatro_gym.env` |
| `serve` | `balatro_gym.serve` |
| `agents` | `agents/` |
| `tools` | `tools/` |
| `github` | `.github/`: workflows, templates, Dependabot |
| `repo` | The whole repository: layout, root config files, and project documents (README, this file, the plan, ADRs) |
| `deps` | Dependency version updates only, e.g. from Dependabot |

### 3.4 Subject line

- Imperative mood: "add", "fix", "remove". Not "added" or "adds".
- Lowercase after the colon, and no full stop at the end.
- The whole header line, including type and scope, is at most **72 characters**.
- Name the specific thing that changed: "add Blueprint copy effect", not "update jokers".

### 3.5 Body

- Optional. Add one only when the subject alone doesn't explain *why*.
- Explain the reason, and the approach if it isn't obvious. Don't list the files
  changed; the diff shows them.
- At most about 5 lines, wrapped at 72 characters. Bullets are fine.
- Only state facts about this change.

### 3.6 Trailers

| Trailer | Use |
|---|---|
| `Closes #<n>` | The commit resolves issue *n* |
| `Refs #<n>` | The commit relates to issue *n* without resolving it |
| `BREAKING CHANGE: <what changes for callers>` | Required for a breaking change (§3.7) |
| `Co-Authored-By: <name> <email>` | Credits a human co-author |

### 3.7 Breaking changes

A change is breaking when it changes any of:
- the public Python API;
- the state schema;
- the replay format;
- the observation or action layout.

A breaking change marks the header with `!`, as in `feat(env)!: …`, and adds a
`BREAKING CHANGE:` trailer saying what callers must do differently.

### 3.8 Commit content

- Commit when a logical unit of work is complete and the tests pass, not after
  every edit. Exploratory or temporary work is not committed.
- One logical change per commit. Don't mix reformatting or renaming with changes in
  behavior.
- Every commit on `main` must build and pass the tests.
- Never commit secrets, local data, large generated files, or any part of the
  original game (code, assets, text).

### 3.9 Examples

Good:

```
feat(sim): add Blueprint and Brainstorm copy effects
```

```
fix(sim): apply Glass destruction after joker effects

Destroying Glass cards inside the per-card loop dropped their x2 mult.
The game removes destroyed cards only after all effects resolve.

Closes #41
```

```
refactor(env)!: replace flat action layout with atomic action codec

BREAKING CHANGE: action indices changed; retrain policies built on the
old layout.
```

Bad:

| Message | Problem |
|---|---|
| `update code` | No type or scope, and says nothing specific |
| `fix: bug` | No scope, and doesn't say which bug |
| `feat(sim): added jokers, fixed shop, docs` | Several changes in one commit, and not imperative |
| `fix(sim): fix scoring (Mime still broken, TODO)` | Mentions unfinished work |
| `wip` | Work in progress belongs on a branch, squashed before merge |

---

## 4. Pull requests

### 4.1 Title

The title uses the same format as a commit header (§3.1–3.4). When the PR is
squash-merged, its title becomes the commit header on `main`. CI checks the format.

### 4.2 Description

The template has three sections. Keep each one short:

| Section | Content |
|---|---|
| **Summary** | 1–3 sentences: what changes and why |
| **Changes** | Up to about 5 bullets on the notable changes. Not a file list. |
| **Testing** | How it was checked: tests added, commands run |

Add `Closes #<n>` on the last line when the PR resolves an issue.

### 4.3 What a description leaves out

PR descriptions and commit messages describe the change as it is merged. They
don't contain:

- work that remains to be done, TODOs, or plans for later;
- known limitations, risks, or open concerns;
- the story of how the change was developed;
- a restatement of the diff.

Follow-up work is tracked in issues.

### 4.4 Size and scope

- A PR delivers one feature or capability, complete and tested.
- Size is a guideline, not a limit: aim for at most about 2,000 changed lines of
  non-test code. Tests, lockfiles, generated data and fixtures don't count. Split
  larger work where it divides naturally.
- Open the PR when the work is done. Use a draft PR only to get early feedback.

### 4.5 Before opening a PR

The author confirms all of these. They don't go in the description.

- [ ] CI passes.
- [ ] Changes in behavior come with tests.
- [ ] Rules ported from the original game cite their source location in docstrings.
- [ ] If game behavior changed, `RULES_VERSION` is bumped and golden replays are
      updated.
- [ ] Changes users can see have an entry under `## [Unreleased]` in `CHANGELOG.md`.
- [ ] Documentation and ADRs reflect the change.
- [ ] The author has read the full diff.

### 4.6 Review and merge

- The maintainer reviews and merges. The only merge method is **squash**.
- The squash commit uses the PR title as its header and the PR description as its
  body. Keeping descriptions brief keeps the history on `main` clean.
- The branch is deleted after merging.

---

## 5. Naming

Code is read far more often than it is written. Every name says what the thing
is, in full words, so a reader never has to decode it.

- Use whole words: `context`, `joker`, `scoring_cards`, `randomizer`; not `ctx`,
  `j`, `sc`, `rnd`.
- No single-letter names, except `i` for an integer index in a loop of a few lines.
  Name loop variables after what they hold: `for card in hand`, not `for c in hand`.
- Abbreviations are allowed only when they are better known than the full words:
  `id`, `uid`, `rng` (random number generator), `min`, `max`, `json`; and the
  game's own terms, such as `ante`, `chips`, `mult` and `x_mult`.
- Import names, not short module aliases: `from balatro_gym.sim.events import
  HandScored`, not `from balatro_gym.sim import events as ev`.
- Functions are named for what they do (`score_held_card`), and booleans read as
  statements (`is_face`, `has_room`).
- Keys from the game's data, such as the config field `t_mult` or the random
  stream `"cdt1"`, keep the game's spelling so they can be found in its source. The
  code around them uses descriptive names, with a comment where a key is cryptic.

---

## 6. AI-assisted changes

- All the rules above apply unchanged.
- An AI assistant may open PRs and make direct commits as §1 allows. It never merges
  PRs and never force-pushes `main`.
- AI tools are never listed as co-authors. Commits and PRs carry no AI attribution
  of any kind.
