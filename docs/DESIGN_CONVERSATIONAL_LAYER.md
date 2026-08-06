# Design: conversational interaction layer

Status: proposed. Written before implementation, per the handoff's instruction to
settle "incremental task semantics" and "optional acceptance" before touching code.

## 0. Corrections to the handoff's premises

Three claims in the handoff brief do not match the code. They matter because two
of them change the design.

### 0.1 The baseline is not clean

The handoff states "424 tests 全绿, ruff / mypy 零告警". Measured on `clean-main`
at `bd93aec`:

| Check | Handoff claim | Actual |
|---|---|---|
| pytest | 424 green | **424 green** ✅ |
| ruff | 0 warnings | **13 errors** (9 auto-fixable) |
| mypy | 0 warnings | **8 errors** in 4 files |

So the "keep it green" constraint is only true of pytest. I will hold pytest at
424+ green and *not* increase ruff/mypy error counts, but I am not silently
adopting a false baseline. One mypy error is a real defect (§0.2).

### 0.2 `full_agent.py` defines its dataclasses twice

`FullAgentInput` and `FullAgentResult` are each declared twice — lines 33–57 and
again at 161–186 (`mypy: Name "FullAgentInput" already defined`). The second pair
silently wins at import time. They are currently identical, so there is no
behavioral bug *today*; it is a live trap, because editing the first copy (the one
a reader finds first, directly above `execute_full_agent`) changes nothing. Any
incremental-turn work adds fields here, so this gets fixed first.

### 0.3 `WorktreeWorkspace` silently discards uncommitted work — this is the key finding

The handoff says the shadow workspace starts "from clean baseline". It is stronger
and worse than that: `_try_git_worktree` runs `git worktree add --detach <root> HEAD`
(worktree.py:116-130), so the shadow is seeded from **the last commit**, not the
working tree. Verified directly:

```
real repo f.txt   -> DIRTY-UNCOMMITTED
worktree shadow   -> committed
```

There is no dirty-repo guard anywhere in `qqcode/` or `tests/` (grepped for
`is_dirty`, `status --porcelain`, `diff --quiet`). Consequences:

- **Today, single-shot:** run against a dirty repo and `finalize()` copies the
  shadow over the target, **destroying uncommitted changes**. `finalize` preserves
  `.git` but nothing else. This is a pre-existing data-loss bug, not one I am
  introducing.
- **For conversation, it is fatal.** Turn 2's shadow is seeded from `HEAD`. If
  turn 1 finalized without committing, turn 2 starts from a tree that *does not
  contain turn 1's work* and then finalizes over it — silently reverting the
  previous turn. Multi-turn cannot be built on `HEAD`-seeded worktrees.

This single fact drives §2.

## 1. Session state and storage

Session records live in a **separate SQLite file**, `.qqcode/sessions.db`, not in
`trace.db`. Rationale: `trace.db`'s schema is deliberately one flat wide row per
run for ad-hoc calibration queries (trace.py:23-53), and `ReplayEngine` scans
`traces` wholesale. Session rows have different lifetime and cardinality (one row
per session, many turns per row). Both are covered by the existing `.qqcode/`
gitignore rule.

Proposed schema (one row per session):

| Field | Type | Note |
|---|---|---|
| `session_id` | TEXT PK | uuid4 |
| `repo` | TEXT | resolved absolute path |
| `created_at` / `updated_at` | TEXT | ISO-8601 UTC, matching `trace.py` |
| `base_commit` | TEXT | 40-char sha captured at session start |
| `turns_json` | TEXT | ordered array of turn records |

Each turn record: `{task, mode, finish_reason, changed_files[], accepted, tokens}`.

`--resume <id>` loads by id; `--continue` loads the most recent session for the
resolved repo.

### 1.1 LangGraph checkpointer: explicitly out of scope for conversation state

The handoff suggests "把 checkpointer 真正接上". I am deliberately *not* using
LangGraph's checkpointer as the conversation store, for three reasons:

1. `langgraph.checkpoint.sqlite` **is not installed** (verified: `ModuleNotFoundError`;
   only `.memory` and `.base` resolve). Using it means a new dependency.
2. `MemorySaver` is instantiated *per compiled graph* inside `_compile_graph`
   (graph.py:221) with a fresh `uuid4` thread id per run (full_agent.py:91). It is
   process-local and dies with the process. It was never a persistence layer.
3. Its checkpoints hold the **ReAct message list** — the intra-task tool loop.
   That is the wrong granularity. Conversation state is "what did the user ask
   across turns, and what was accepted", which survives independently of whether
   any individual ReAct loop is replayable.

Reusing it would couple session durability to LangGraph's internal state format
for no gain. Sessions get their own store; the checkpointer keeps doing its
current job of in-run state.

## 2. Incremental task semantics (the hard decision)

Options considered:

- **(A) Keep one long-lived workspace across turns.** Rejected: a workspace that
  survives N turns accumulates unreviewed drift, and its `finalize` writes
  everything at once. Also holds a git worktree registration open indefinitely.
- **(B) Rebuild per turn from `HEAD`, replay accepted patches.** Rejected: replay
  is a diff-reapplication engine with conflict semantics — large, and it
  reintroduces the §0.3 bug for any work not yet committed.
- **(C) Rebuild per turn, seeded from the real working tree. ← chosen**

**Decision: each turn gets a fresh shadow, seeded from the repo's current working
tree state, not `HEAD`.** Turn N automatically builds on turn N−1 because turn
N−1's accepted output is *in the working tree*. No replay engine, no long-lived
workspace, and it fixes §0.3 for the single-shot path at the same time.

Implementation: add `seed: Literal["head", "worktree"] = "head"` to
`WorktreeWorkspace`. Default `"head"` preserves every existing test's behavior.
`"worktree"` overlays tracked-but-modified and untracked files onto the shadow
after `git worktree add` (or just uses the existing `copytree` fallback, which
*already* copies the live tree — the non-git path was always correct here). The
conversational layer passes `seed="worktree"`.

Additionally: a **dirty-repo guard** on the single-shot path. Before a
non-`dry_run` finalize seeded from `HEAD`, if the target has uncommitted changes,
refuse and tell the user to commit or stash. That closes the data-loss bug without
changing the default seeding semantics that 424 tests encode.

## 3. Acceptance becomes optional — two verdict sources

The design conflict is real and the handoff frames it correctly. Resolution: the
gate's *third* condition changes from "acceptance passed" to "**a verdict was
obtained**", with two possible sources:

| Situation | Verdict source | Behavior |
|---|---|---|
| Harness supplied, non-empty | `objective` | Unchanged from today: tests decide, auto-finalize. |
| No harness (interactive) | `user` | Present diff + summary, finalize only on explicit approval. |
| No harness, non-interactive | `none` | Current behavior: conditions 1–2 only. Unchanged. |

Critically, conditions 1 (valid finish state) and 2 (diff ⊆ expected files) are
**untouched in all three cases**. Only the third condition gains a human variant.
`all_passed([])` already returns `True` for the empty-harness case
(harness.py:237-243), so the objective path needs no change at all.

Mechanically this is a new optional `confirm: ConfirmCallback | None` parameter on
`run_task`. `None` (every existing caller, all 424 tests) = today's behavior
exactly. The REPL passes a callback that renders the diff and prompts. This keeps
batch and conversational modes on one code path rather than forking
`orchestrator.py`.

## 4. Interruption and rollback

- **Ctrl-C mid-turn:** raises into `run_task`; the `with WorktreeWorkspace(...)`
  context manager (orchestrator.py:178, 244) already guarantees `cleanup()`. The
  shadow is discarded, the real repo is untouched, the REPL survives and the turn
  is logged as `interrupted`. Because finalize is the *only* path to the real repo
  and it runs last, interruption is inherently safe — nothing partial escapes.
- **User rejects a turn's result:** if not yet finalized, discard the shadow —
  free.

### 4.1 Undo — resolved, and not the way this section originally proposed

**Superseded.** This section originally planned to anchor undo on the session's
`base_commit` and therefore to *require a clean working tree at session start*.
That was abandoned during implementation, because git cannot do the job in the
case that matters most: `finalize` writes the working tree without committing, so
for an applied-but-uncommitted turn there is no commit to return to, and checking
out `base_commit` would additionally discard whatever the user had done by hand
before the session began.

`/undo` is instead **snapshot-based** (`qqcode/undo.py`). Accepting a turn keeps
what `build_review` already read to produce the diff — each changed file's
content on both sides. Undo replays that, so it is exact, scoped to one turn, and
independent of version control.

Consequences, all better than the original plan:

- **No clean-tree requirement.** A session can start in a dirty repository.
- **Per-turn, not per-session.** `/undo` reverses the last applied turn, which is
  what people mean by "undo", rather than resetting the whole conversation.
- **Refuses instead of merging.** If a file changed after the turn was applied,
  restoring it would silently discard that later edit, so the REPL reports the
  conflict and asks before proceeding.

Remaining limits, stated plainly:

- Binary files are reported unrestorable and left in place; their prior bytes
  were never captured.
- Undo history is per-process and deliberately not persisted — snapshots hold
  full file contents, so writing them to `sessions.db` would grow it with the
  size of the repository. A resumed session has nothing to undo and says so.

## 5. Build order

1. Fix §0.2 duplicate dataclasses; fix the 8 mypy / 13 ruff errors. Baseline honest.
2. `seed=` on `WorktreeWorkspace` + dirty-repo guard (§0.3, §2). Tests for both.
3. Session store `.qqcode/sessions.db` (§1). Tests.
4. `confirm=` on `run_task` (§3). Tests for all three verdict sources.
5. `qqcode/repl.py` + `--resume` / `--continue` wiring (§1). Tests.

Each step keeps pytest green and does not raise ruff/mypy counts.

## 6. Open defect: L0 routes to FastPath without file hints

Found while auditing routing behaviour after the conversational layer landed.
**This predates the conversational work** and affects batch mode identically —
conversation only made it frequent enough to notice.

**Symptom.** Every FastPath attempt in live conversation traces declined and
escalated. Across four repositories: 5 `declined`, 8 `model_error` (the latter
were a dead Anthropic key, not a routing problem).

**Root cause.** All five declines had `files_hint_count = 0`. Every L0 branch in
`router.py` returns `files_hint=()` — L0 is a static rule, it decides *that* a
task is simple without identifying *which* files it touches. FastPath prefetches
file contents by `files_hint`, so it prefetched nothing, while its system prompt
told the model "Current file contents are provided below the task description"
and required whole-file replacement content. The model could not see the function
it was asked to change, so it took the prompt's documented exit and returned an
empty patch.

**Measured, same task and model, only the hint differing:**

| `files_hint` | outcome | tokens |
|---|---|---|
| `()` | declined → escalates to Full Agent | 745 wasted, then ~6,900 |
| `("calc.py",)` | success | 786 |

41 extra tokens for the hint avoids roughly 6,900. The waste lands precisely on
the simple, single-file tasks FastPath exists to win.

**Not fixed here** — a fix belongs with the routing layer and its own
measurement, not tacked onto the conversation work. Options worth evaluating:
have L0 extract filenames mentioned in the task, fall back to L1 for hints when
L0 fires, or let FastPath prefetch a bounded set of recently-changed files when
it has no hint. Whichever is chosen should be validated by `trace replay` against
recorded traces rather than by intuition.
