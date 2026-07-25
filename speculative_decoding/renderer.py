from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from transformers import PreTrainedTokenizerBase

from speculative_decoding.generate import GenerationStats

_ACCEPTED_STYLE = "green3"
_BONUS_STYLE = "cyan"  # correction token / free bonus token -- came from the verifier, not the draft
_ENTROPY_STOP_STYLE = "underline"  # marks the token right before the draft cut its own round short


def replay(
    stats: GenerationStats, output_ids, tokenizer: PreTrainedTokenizerBase, console: Console
) -> None:
    """Print generated text, coloring each token by how it was produced:
    green = draft token the verifier accepted, cyan = verifier's own token
    (either a rejection correction or a free bonus token). The last accepted
    draft token in a round is underlined if the draft stopped proposing early
    because its own entropy crossed ENTROPY_STOP_THRESHOLD there.
    """
    pos = 0
    for round_result in stats.rounds:
        num_accepted = len(round_result.accepted_tokens)
        for i, _ in enumerate(round_result.accepted_tokens):
            if pos >= len(output_ids):
                break
            style = _ACCEPTED_STYLE
            if round_result.stopped_early_on_entropy and num_accepted > 0 and i == num_accepted - 1:
                style = f"{_ACCEPTED_STYLE} {_ENTROPY_STOP_STYLE}"
            console.print(Text(tokenizer.decode([output_ids[pos]]), style=style), end="")
            pos += 1
        if pos >= len(output_ids):
            break
        style = _BONUS_STYLE
        if round_result.stopped_early_on_entropy and num_accepted == 0:
            style = f"{_BONUS_STYLE} {_ENTROPY_STOP_STYLE}"
        console.print(Text(tokenizer.decode([output_ids[pos]]), style=style), end="")
        pos += 1
    console.print()
    console.print()

    legend = Text()
    legend.append("  ■  ", style=_ACCEPTED_STYLE)
    legend.append(" accepted draft token   ")
    legend.append("  ■  ", style=_BONUS_STYLE)
    legend.append(" verifier correction / bonus token   ")
    legend.append("  ■  ", style=f"{_ACCEPTED_STYLE} {_ENTROPY_STOP_STYLE}")
    legend.append(" draft stopped early (high entropy)")
    console.print(legend)
    console.print()


def print_summary(stats: GenerationStats, elapsed_s: float, console: Console) -> None:
    console.print(
        Panel.fit(
            Text(f"{stats.acceptance_rate:.1%}", style="bold green3"),
            title="Acceptance rate (sampling-mode)",
        )
    )
    console.print(
        f"{stats.total_accepted} draft tokens accepted / {stats.total_proposed} proposed "
        f"across {len(stats.rounds)} rounds in {elapsed_s:.1f}s\n"
    )

    table = Table(title="Per-round acceptance")
    table.add_column("Round")
    table.add_column("Proposed")
    table.add_column("Accepted")
    table.add_column("Outcome")
    for i, r in enumerate(stats.rounds):
        outcome = "all accepted + bonus" if r.num_accepted == r.num_proposed else "rejected mid-round"
        if r.stopped_early_on_entropy:
            outcome += f" (entropy stop, gamma {r.requested_gamma}→{r.num_proposed})"
        table.add_row(str(i), str(r.num_proposed), str(r.num_accepted), outcome)
    console.print(table)
