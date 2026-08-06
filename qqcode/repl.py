"""Conversational REPL: multi-turn task execution against one repository.

Each turn is a full `run_task` — routing, FastPath or Full Agent, then the gate.
What makes it conversational rather than a loop of independent runs is two things:

- Shadow workspaces are seeded from the *working tree* (`seed="worktree"`), so
  turn N sees turn N-1's output even when it was never committed.
- With no hidden acceptance suite to consult, the verdict comes from the person
  at the keyboard: every change is shown as a diff and finalized only on
  approval.

Ctrl-C interrupts the current turn without killing the session. That is safe by
construction, not by cleanup: `finalize` is the only path to the real repository
and it runs last, so an interrupted turn leaves the shadow discarded and the repo
untouched.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from rich.console import Console
from rich.syntax import Syntax

from qqcode.config import Config
from qqcode.memory.session import SessionRecord, SessionStore, TurnRecord
from qqcode.memory.trace import TraceStore
from qqcode.orchestrator import Mode, RunResult, run_task
from qqcode.review import ChangeReview, ConfirmCallback

# Typed at the prompt to leave the session.
EXIT_COMMANDS = frozenset({"/exit", "/quit", "exit", "quit"})

# Diff lines above this count are collapsed in the terminal. The full text stays
# in the review object; this only bounds what is printed at once.
MAX_PRINTED_DIFF_LINES = 120

# Turns replayed in the header when resuming. Enough to recall where the
# conversation stood without redisplaying an entire history.
RESUME_CONTEXT_TURNS = 5


def current_commit(repo: Path) -> str:
    """HEAD of `repo`, or "" when it is not a git repository.

    Recorded at session start so an already-finalized turn has an undo anchor.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def render_review(review: ChangeReview, console: Console) -> None:
    """Print a change for human inspection."""
    console.print()
    if review.reasoning:
        console.print(f"[bold]Summary:[/bold] {review.reasoning}")

    if review.is_empty():
        console.print(
            "[yellow]The agent reported success but changed no files.[/yellow]"
        )
        return

    console.print(f"[bold]{len(review.diffs)} file(s) changed:[/bold]")
    for d in review.diffs:
        marker = {"added": "+", "deleted": "-", "modified": "~"}.get(d.status, "?")
        console.print(f"  [cyan]{marker}[/cyan] {d.path} [dim]({d.status})[/dim]")

    for d in review.diffs:
        console.print()
        console.print(f"[bold]{d.path}[/bold]")
        if not d.diff_text.strip():
            console.print("[dim](no textual diff — binary or unreadable)[/dim]")
            continue
        lines = d.diff_text.splitlines()
        shown = lines[:MAX_PRINTED_DIFF_LINES]
        console.print(Syntax("\n".join(shown), "diff", theme="ansi_dark", word_wrap=True))
        hidden = len(lines) - len(shown)
        if hidden > 0:
            console.print(f"[dim]... {hidden} more lines ...[/dim]")


def make_confirm(console: Console) -> ConfirmCallback:
    """Build the confirm callback that asks the person to accept a change."""

    def confirm(review: ChangeReview) -> bool:
        render_review(review, console)
        if review.is_empty():
            # Nothing to apply; finalizing an empty change is a no-op that would
            # still rewrite the target directory.
            return False
        console.print()
        answer = console.input(
            "[bold]Apply these changes?[/bold] [dim]([green]y[/green]/[red]n[/red])[/dim] "
        )
        return answer.strip().lower() in {"y", "yes"}

    return confirm


def _describe(result: RunResult) -> str:
    if result.rejected:
        return "rejected"
    return "accepted" if result.success else "failed"


def run_repl(
    repo: Path,
    config: Config,
    *,
    mode: Mode = "auto",
    max_turns: int = 30,
    console: Console | None = None,
    trace_store: TraceStore | None = None,
    provider: str | None = None,
    model: str | None = None,
    session_store: SessionStore | None = None,
    resume: SessionRecord | None = None,
) -> SessionRecord:
    """Run the interactive loop until the user exits. Returns the session log.

    Args:
        session_store: Where turns are persisted, after each one rather than at
            exit. `None` keeps the session in memory only.
        resume: A previously stored session to continue. Its turn log is
            replayed in the header and appended to, and its `base_commit` is
            kept so the undo anchor still points at the original start.
    """
    console = console or Console()
    confirm = make_confirm(console)

    session = resume or SessionRecord(
        repo=str(repo.resolve()), base_commit=current_commit(repo)
    )

    if resume is not None:
        _print_resume_header(session, console)
    else:
        console.print(f"[bold]qqcode[/bold] — conversational mode  [dim]({repo})[/dim]")
        console.print(f"[dim]session {session.short_id}[/dim]")
    console.print(
        "[dim]Describe a task, or /exit to leave. Ctrl-C cancels the current turn.[/dim]"
    )

    def record(turn: TurnRecord) -> None:
        """Append a turn and persist immediately.

        Saving per turn rather than at exit means a crash costs the turn in
        flight, not the whole conversation.
        """
        session.turns.append(turn)
        if session_store is not None:
            session_store.save(session)

    while True:
        try:
            task = console.input("\n[bold cyan]›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            break

        if not task:
            continue
        if task.lower() in EXIT_COMMANDS:
            break

        try:
            result = run_task(
                task,
                repo,
                config,
                mode=mode,
                dry_run=False,
                max_turns=max_turns,
                trace_store=trace_store,
                provider=provider,
                model=model,
                confirm=confirm,
                seed="worktree",      # turn N builds on turn N-1
            )
        except KeyboardInterrupt:
            # The shadow was discarded by the workspace context manager and
            # finalize never ran, so the repository is untouched.
            console.print("\n[yellow]Turn cancelled. Repository unchanged.[/yellow]")
            record(TurnRecord(task=task, outcome="interrupted"))
            continue
        except Exception as exc:
            # Surfaced, never swallowed: a silent failure here costs far more
            # time to diagnose than a loud one.
            console.print(f"[red]Error:[/red] {type(exc).__name__}: {exc}")
            record(TurnRecord(task=task, outcome="failed"))
            continue

        outcome = _describe(result)
        record(
            TurnRecord(
                task=task,
                outcome=outcome,
                mode_used=result.mode_used,
                changed_files=tuple(sorted(result.changed_files)),
                tokens=result.ledger.summary().get("automatic_total", 0),
            )
        )
        _report_turn(result, outcome, console)

    _report_session(session, console)
    return session


def _report_turn(result: RunResult, outcome: str, console: Console) -> None:
    if outcome == "accepted":
        console.print(
            f"[green]✅ Applied[/green] [dim]({result.mode_used}, "
            f"{len(result.changed_files)} file(s))[/dim]"
        )
    elif outcome == "rejected":
        console.print("[yellow]Discarded. Repository unchanged.[/yellow]")
    else:
        console.print(f"[red]❌ Failed[/red] [dim]({result.finish_reason})[/dim]")
        if result.error:
            console.print(f"[dim]{result.error[:300]}[/dim]")


def _print_resume_header(session: SessionRecord, console: Console) -> None:
    """Recall where the conversation stood, without redisplaying everything."""
    console.print(
        f"[bold]qqcode[/bold] — resumed session [cyan]{session.short_id}[/cyan]  "
        f"[dim]({session.repo})[/dim]"
    )
    if not session.turns:
        console.print("[dim]No turns recorded yet.[/dim]")
        return

    hidden = len(session.turns) - RESUME_CONTEXT_TURNS
    if hidden > 0:
        console.print(f"[dim]... {hidden} earlier turn(s) ...[/dim]")

    icons = {"accepted": "[green]✓[/green]", "rejected": "[yellow]✗[/yellow]",
             "failed": "[red]![/red]", "interrupted": "[dim]~[/dim]"}
    for i, turn in enumerate(session.turns[-RESUME_CONTEXT_TURNS:], start=max(1, hidden + 1)):
        icon = icons.get(turn.outcome, "?")
        console.print(f"  {icon} [dim]{i}.[/dim] {turn.task[:70]}")

    console.print(
        f"[dim]{len(session.turns)} turn(s) so far, {session.accepted_count} applied.[/dim]"
    )


def _report_session(session: SessionRecord, console: Console) -> None:
    if not session.turns:
        return
    console.print()
    console.print(
        f"[dim]{len(session.turns)} turn(s), {session.accepted_count} applied, "
        f"{session.total_tokens:,} tokens.[/dim]"
    )
    console.print(
        f"[dim]Resume with:[/dim] qqcode --chat --resume {session.short_id}"
    )
