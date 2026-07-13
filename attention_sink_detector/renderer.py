from __future__ import annotations

from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from transformers import PreTrainedTokenizerBase

from .attention_sink import TokenAttentionRecord

_STRUCTURAL_STYLE = "bold white on grey37"   # expected (e.g. BOS) — muted, not alarming
_SINK_STYLE = "bold white on red3"            # anomalous — the actually interesting finding
_NORMAL_STYLE = "grey70"


def highlight_prompt(
    records: List[TokenAttentionRecord],
    tokenizer: PreTrainedTokenizerBase,
    console: Console,
) -> None:
    line = Text()
    for record in records:
        token_text = tokenizer.decode([record.token_id])
        if record.is_structural:
            style = _STRUCTURAL_STYLE
        elif record.is_sink:
            style = _SINK_STYLE
        else:
            style = _NORMAL_STYLE
        line.append(token_text, style=style)
    console.print(line)
    console.print()

    legend = Text()
    legend.append("  ■  ", style=_STRUCTURAL_STYLE)
    legend.append(" structural (expected)   ")
    legend.append("  ■  ", style=_SINK_STYLE)
    legend.append(" anomalous sink")
    console.print(legend)
    console.print()


def print_health_summary(
    health: float,
    records: List[TokenAttentionRecord],
    tokenizer: PreTrainedTokenizerBase,
    console: Console,
) -> None:
    color = "green3" if health >= 7 else "yellow3" if health >= 4 else "red3"
    console.print(
        Panel.fit(Text(f"{health:.1f}/10", style=f"bold {color}"), title="Context health")
    )

    structural = [r for r in records if r.is_structural]
    anomalous = [r for r in records if r.is_sink and not r.is_structural]

    if structural:
        table = Table(title="Structural sinks (expected, excluded from score)")
        table.add_column("Position")
        table.add_column("Token")
        table.add_column("Mean attn.")
        table.add_column("Z-score")
        for r in structural:
            table.add_row(
                str(r.position),
                repr(tokenizer.decode([r.token_id])),
                f"{r.mean_attention:.4f}",
                f"{r.z_score:.2f}",
            )
        console.print(table)
        console.print()

    if not anomalous:
        console.print("No anomalous sink tokens detected.\n")
        return

    table = Table(title="Anomalous sink tokens")
    table.add_column("Position")
    table.add_column("Token")
    table.add_column("Mean attn.")
    table.add_column("Z-score")
    for r in anomalous:
        table.add_row(
            str(r.position),
            repr(tokenizer.decode([r.token_id])),
            f"{r.mean_attention:.4f}",
            f"{r.z_score:.2f}",
        )
    console.print(table)