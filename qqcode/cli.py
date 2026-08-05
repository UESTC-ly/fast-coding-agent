"""QQCode CLI entry point.

Usage:
    qqcode --task "fix the parse_args docstring" --repo .
    qqcode --task "..." --mode full --dry-run
    qqcode --task "..." --provider openai --model gpt-5.6-luna
    qqcode trace replay --repo .

There is no `run` subcommand: typer promotes a single registered command to the
app's default, so the options attach directly to `qqcode`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from qqcode.config import Config
from qqcode.memory.replay import ReplayEngine
from qqcode.memory.trace import TraceStore
from qqcode.orchestrator import RunResult, run_task

app = typer.Typer(name="qqcode", help="Coding agent with intelligent routing.")
trace_app = typer.Typer(name="trace", help="Routing trace and calibration commands.")
app.add_typer(trace_app, name="trace")
console = Console()


@app.callback(invoke_without_command=True)
def run(
    ctx: typer.Context,
    task: Annotated[str, typer.Option("--task", "-t", help="Task description")] = "",
    repo: Annotated[Path, typer.Option("--repo", "-r", help="Repo root")] = Path("."),
    mode: Annotated[str, typer.Option("--mode", "-m", help="auto|fast|full")] = "auto",
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Don't write back to repo")] = False,
    provider: Annotated[str, typer.Option("--provider", help="anthropic|openai")] = "",
    model: Annotated[str, typer.Option("--model", help="Pin a specific model id")] = "",
    max_turns: Annotated[int, typer.Option("--max-turns", help="Full Agent turn limit")] = 30,
) -> None:
    """Run a coding task, or use 'qqcode trace' for calibration commands."""
    if ctx.invoked_subcommand is not None:
        return  # a trace/other subcommand was invoked; let it run
    if not task:
        console.print("[red]Error:[/red] --task is required")
        raise typer.Exit(1)

    if mode not in {"auto", "fast", "full"}:
        console.print(f"[red]Error:[/red] --mode must be auto, fast, or full (got {mode!r})")
        raise typer.Exit(1)

    repo = repo.resolve()
    if not repo.is_dir():
        console.print(f"[red]Error:[/red] repo path not found: {repo}")
        raise typer.Exit(1)

    config = Config.from_env()
    if config.anthropic is None and config.openai is None:
        console.print("[red]Error:[/red] No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")
        raise typer.Exit(1)

    console.print(f"[bold]Task:[/bold] {task}")
    console.print(f"[dim]repo={repo}  mode={mode}  dry_run={dry_run}[/dim]")
    console.print()

    try:
        store = TraceStore.for_repo(repo)
        result = run_task(
            task,
            repo,
            config,
            mode=mode,  # type: ignore[arg-type]
            dry_run=dry_run,
            provider=provider or None,
            model=model or None,
            max_turns=max_turns,
            trace_store=store,
        )
        store.close()
    except Exception as exc:
        console.print(f"[red]Fatal error:[/red] {exc}")
        raise typer.Exit(2) from exc

    _display_result(result)
    raise typer.Exit(0 if result.success else 1)


def _display_result(result: RunResult) -> None:
    if result.success:
        icon = "✅" if not result.dry_run else "🔍"
        label = "dry run" if result.dry_run else "committed to repo"
        console.print(f"\n{icon} [bold green]Success[/bold green] ({result.mode_used} · {label})")
    else:
        console.print(f"\n❌ [bold red]Failed[/bold red] ({result.mode_used} · {result.finish_reason})")
        if result.error:
            console.print(f"[dim]{result.error[:300]}[/dim]")

    if result.reasoning:
        console.print(f"\n[bold]Summary:[/bold] {result.reasoning[:400]}")

    if result.changed_files:
        console.print(f"\n[bold]Changed files ({len(result.changed_files)}):[/bold]")
        for f in sorted(result.changed_files):
            console.print(f"  {f}")

    if result.turns_used:
        console.print(f"\n[dim]Turns used: {result.turns_used}[/dim]")

    s = result.ledger.summary()
    table = Table(title="Token cost", show_header=True, header_style="bold")
    table.add_column("Phase")
    table.add_column("Tokens", justify="right")
    for phase, tokens in s["by_phase"].items():
        if tokens:
            table.add_row(phase, f"{tokens:,}")
    table.add_row("[bold]total[/bold]", f"[bold]{s['automatic_total']:,}[/bold]")
    console.print()
    console.print(table)
    if s["retried_calls"]:
        console.print(f"[dim]Retried calls: {s['retried_calls']}[/dim]")


# --------------------------------------------------------------------------
# trace subcommands
# --------------------------------------------------------------------------


@trace_app.command("replay")
def trace_replay(
    repo: Annotated[Path, typer.Option("--repo", "-r", help="Repo root")] = Path("."),
    sweep: Annotated[str, typer.Option("--sweep", help="tau|length|files")] = "tau",
) -> None:
    """Replay routing decisions to calibrate thresholds (zero model calls).

    Reads traces from <repo>/.qqcode/trace.db and prints a calibration table
    showing how FastPath routing rates and estimated costs would change under
    different threshold settings.
    """
    repo = repo.resolve()
    store = TraceStore.for_repo(repo)
    traces = store.all()
    store.close()

    if not traces:
        console.print("[yellow]No traces found.[/yellow] Run some tasks first (trace_store is enabled by default).")
        raise typer.Exit(0)

    console.print(f"[bold]{len(traces)} traces loaded[/bold] from {repo / '.qqcode/trace.db'}\n")

    engine = ReplayEngine(traces)

    if sweep == "length":
        rows = engine.calibrate_task_length()
        title = "Calibration: max task length (L)"
        key_label = "max_length"
        key_fn = lambda r: str(r.thresholds.max_task_length)  # noqa: E731
    elif sweep == "files":
        rows = engine.calibrate_max_files()
        title = "Calibration: max files hint (K)"
        key_label = "max_files"
        key_fn = lambda r: str(r.thresholds.max_files)  # noqa: E731
    else:
        rows = engine.calibrate_tau()
        title = "Calibration: confidence threshold (τ)"
        key_label = "τ"
        key_fn = lambda r: f"{r.thresholds.confidence:.2f}"  # noqa: E731

    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column(key_label, justify="right")
    table.add_column("FP routed", justify="right")
    table.add_column("FA routed", justify="right")
    table.add_column("Indet.", justify="right")
    table.add_column("FP precision", justify="right")
    table.add_column("Avg tokens", justify="right")
    table.add_column("Δ cost", justify="right")

    baseline_tau = f"{engine.baseline.confidence:.2f}"
    baseline_L = str(engine.baseline.max_task_length)
    baseline_K = str(engine.baseline.max_files)

    for row in rows:
        key_val = key_fn(row)
        is_baseline = (
            (sweep == "tau" and key_val == baseline_tau)
            or (sweep == "length" and key_val == baseline_L)
            or (sweep == "files" and key_val == baseline_K)
        )
        prefix = "* " if is_baseline else "  "
        prec = f"{row.fp_precision:.0%}" if row.fp_precision is not None else "n/a"
        delta = f"{row.delta_vs_baseline:+.1%}" if row.delta_vs_baseline != 0.0 else "baseline"
        style = "bold cyan" if is_baseline else ""
        table.add_row(
            f"{prefix}{key_val}",
            str(row.fp_routed),
            str(row.fa_routed),
            str(row.indeterminate),
            prec,
            f"{row.mean_tokens_per_task:,.0f}",
            delta,
            style=style,
        )

    console.print(table)
    console.print("[dim]* = current production setting[/dim]")

    # Skill impact
    skill_rows = engine.skill_impact()
    if skill_rows:
        console.print()
        sk_table = Table(title="Skill impact on FastPath", show_header=True, header_style="bold")
        sk_table.add_column("Skill")
        sk_table.add_column("Tasks", justify="right")
        sk_table.add_column("FP attempted", justify="right")
        sk_table.add_column("FP succeeded", justify="right")
        sk_table.add_column("FP hit rate", justify="right")
        for sr in skill_rows:
            sk_table.add_row(
                sr.skill_name,
                str(sr.trace_count),
                str(sr.fp_routed_count),
                str(sr.fp_success_count),
                f"{sr.fp_hit_rate:.0%}",
            )
        console.print(sk_table)


@trace_app.command("stats")
def trace_stats(
    repo: Annotated[Path, typer.Option("--repo", "-r", help="Repo root")] = Path("."),
) -> None:
    """Show summary statistics and token-savings breakdown for all recorded traces."""
    repo = repo.resolve()
    store = TraceStore.for_repo(repo)
    traces = store.all()
    store.close()

    if not traces:
        console.print("[yellow]No traces found.[/yellow]")
        raise typer.Exit(0)

    total = len(traces)
    fp_runs = [t for t in traces if t.fastpath_attempted]
    fp_ok   = [t for t in traces if t.fastpath_success]
    fa_runs = [t for t in traces if t.mode_used == "fullagent"]
    overall_success = sum(1 for t in traces if t.final_success)

    # --- summary table --------------------------------------------------------
    table = Table(title=f"Trace summary ({total} runs)", show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total runs", str(total))
    table.add_row("FastPath attempted", str(len(fp_runs)))
    table.add_row("FastPath succeeded", str(len(fp_ok)))
    table.add_row(
        "FastPath precision",
        f"{len(fp_ok)/len(fp_runs):.0%}" if fp_runs else "n/a",
    )
    table.add_row("FullAgent runs", str(len(fa_runs)))
    table.add_row("Overall success rate", f"{overall_success/total:.0%}")
    avg_tok = sum(t.tokens_total for t in traces) / total
    table.add_row("Avg tokens/run (all)", f"{avg_tok:,.0f}")
    console.print(table)

    # --- token savings breakdown ----------------------------------------------
    # --- token breakdown (honest: different task populations, noted) ----------
    fp_tokens = [t.tokens_total for t in fp_ok if t.tokens_total > 0]
    fa_tokens = [t.tokens_total for t in fa_runs if t.tokens_total > 0]

    # Escalation overhead: tasks that tried FP and failed → paid FP + FA
    escalated = [t for t in traces if t.fastpath_attempted and not t.fastpath_success]
    esc_overhead_pct: float | None = None
    if escalated:
        esc_fp_tok = sum(t.tokens_fastpath for t in escalated if t.tokens_fastpath > 0)
        esc_fa_tok = sum(t.tokens_fullagent for t in escalated if t.tokens_fullagent > 0)
        esc_total = esc_fp_tok + esc_fa_tok
        esc_overhead_pct = esc_fp_tok / esc_total if esc_total > 0 else None

    savings_table = Table(
        title="Cost breakdown",
        show_header=True,
        header_style="bold",
    )
    savings_table.add_column("Metric")
    savings_table.add_column("Value", justify="right")

    if fp_tokens:
        savings_table.add_row("Avg tokens — FastPath success", f"{sum(fp_tokens)/len(fp_tokens):,.0f}")
    if fa_tokens:
        savings_table.add_row("Avg tokens — Full Agent", f"{sum(fa_tokens)/len(fa_tokens):,.0f}")
    if fp_tokens and fa_tokens:
        savings_table.add_row(
            "[dim]Note: different task populations (simpler vs harder)[/dim]",
            "[dim]not directly comparable[/dim]",
        )

    if escalated:
        savings_table.add_row("Escalated runs (FP failed → FA)", str(len(escalated)))
        if esc_overhead_pct is not None:
            savings_table.add_row(
                "[bold]Escalation overhead[/bold]",
                f"[bold yellow]{esc_overhead_pct:.0%}[/bold yellow] of escalated-run tokens wasted on failed FP",
            )
    else:
        savings_table.add_row("Escalation overhead", "n/a (no escalated runs yet)")

    # Loop savings — only for escalated tasks (have both paths observed)
    fa_turns = [t.turns_used for t in fa_runs if t.turns_used > 0]
    if fa_turns:
        avg_fa_turns = sum(fa_turns) / len(fa_turns)
        savings_table.add_row("Avg loops — Full Agent", f"{avg_fa_turns:.1f}")
        savings_table.add_row(
            "[dim]FastPath loops[/dim]",
            "[dim]1 (single-shot, no loop)[/dim]",
        )

    console.print()
    console.print(savings_table)
    console.print(
        "[dim]Escalation overhead = wasted FP tokens / (FP + FA tokens) on failed FP runs.[/dim]\n"
        "[dim]True savings require A/B: same tasks routed both ways.[/dim]"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
