from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from speculative_decoding.generate import GenerationStats

_ACCEPTED_STYLE = "green3"
_REJECTED_STYLE = "red3"


def print_draft_entropy(stats: GenerationStats, console: Console) -> None:
    """Per-proposed-position draft entropy alongside accept/reject outcome.

    Draft entropy comes from RoundResult.entropy_trace (captured in draft_step
    during proposal), not from a separate entropy_tracker pass over the
    verifier's final emissions -- this is the draft's uncertainty at
    proposal time, correlated against whether the verifier accepted it.
    """
    table = Table(title="Draft entropy vs. acceptance")
    table.add_column("Round")
    table.add_column("Pos")
    table.add_column("Entropy (norm.)")
    table.add_column("Outcome")

    for round_idx, r in enumerate(stats.rounds):
        for pos, entropy in enumerate(r.entropy_trace):
            accepted = pos < r.num_accepted
            style = _ACCEPTED_STYLE if accepted else _REJECTED_STYLE
            outcome = "accepted" if accepted else "rejected"
            table.add_row(
                str(round_idx),
                str(pos),
                f"{entropy:.3f}",
                Text(outcome, style=style),
            )
    console.print(table)
    console.print()


def print_combined_summary(health: float, stats: GenerationStats, console: Console) -> None:
    health_color = "green3" if health >= 7 else "yellow3" if health >= 4 else "red3"
    accept_color = "green3"

    table = Table(title="Combined summary", show_header=False)
    table.add_row("Context health", Text(f"{health:.1f}/10", style=f"bold {health_color}"))
    table.add_row(
        "Acceptance rate", Text(f"{stats.acceptance_rate:.1%}", style=f"bold {accept_color}")
    )
    table.add_row(
        "Draft tokens", f"{stats.total_accepted} accepted / {stats.total_proposed} proposed"
    )
    console.print(Panel.fit(table))
