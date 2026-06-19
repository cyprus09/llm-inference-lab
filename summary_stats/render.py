from __future__ import annotations

import time
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from transformers import PreTrainedTokenizerBase

from llm_lab.entropy_tracker import StepRecord, entropy_bucket

_COLOR = {"green": "green3", "yellow": "yellow3", "red": "red3"}


def replay(
    records: List[StepRecord], tokenizer: PreTrainedTokenizerBase, console: Console
) -> None:
    for record in records:
        console.print(
            Text(
                tokenizer.decode([record.chosen_token_id]),
                style=_COLOR[entropy_bucket(record.normalized_entropy)],
            ),
            end="",
        )
        time.sleep(0.03)
    console.print()


def print_summary(summary: dict, console: Console) -> None:
    trace = Text()
    for bucket in summary["entropy_trace"]:
        trace.append("█", style=_COLOR[bucket])
    console.print(Panel.fit(trace, title="Entropy over time"))
    console.print(f"Mean entropy: {summary['mean_entropy']:.3f} nats\n")

    table = Table(title="Most uncertain tokens")
    table.add_column("Step")
    table.add_column("Token")
    table.add_column("Norm. entropy")
    table.add_column("Top alternatives")
    for item in summary["most_uncertain"]:
        alts = ", ".join(f"{tok!r} ({p:.2f})" for tok, p in item["top_k"][:3])
        table.add_row(
            str(item["step"]),
            repr(item["token"]),
            f"{item['normalized_entropy']:.2f}",
            alts,
        )
    console.print(table)
