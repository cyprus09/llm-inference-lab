from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, PreTrainedTokenizerBase


@dataclass
class StepRecord:
    step: int
    entropy: float  # raw, in nats
    normalized_entropy: float  # entropy / log(vocab_size), range [0, 1]
    top_k: List[Tuple[int, float]]  # (token_id, probability), sorted desc
    chosen_token_id: Optional[int] = None  # attached after generation


class EntropyLogitsProcessor(LogitsProcessor):
    """Observes (never modifies) logits at each decode step. batch_size=1 only."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.records: List[StepRecord] = []
        self._step = 0

    def reset(self) -> None:
        self.records = []
        self._step = 0

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        vocab_size = scores.shape[-1]
        # float32 avoids fp16 underflow; log_softmax+exp more stable than softmax().log()
        log_probs = F.log_softmax(scores[0].float(), dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum().item()
        top_probs, top_indices = torch.topk(probs, k=self.top_k)
        self.records.append(
            StepRecord(
                step=self._step,
                entropy=entropy,
                normalized_entropy=entropy / math.log(vocab_size),
                top_k=list(zip(top_indices.tolist(), top_probs.tolist())),
            )
        )
        self._step += 1
        return scores


def attach_chosen_tokens(
    records: List[StepRecord], generated_ids: torch.LongTensor, prompt_len: int
) -> None:
    for record, token_id in zip(records, generated_ids[prompt_len:].tolist()):
        record.chosen_token_id = token_id


def entropy_bucket(ne: float) -> str:
    if ne < 0.3:
        return "green"
    if ne < 0.6:
        return "yellow"
    return "red"


def summarize(records: List[StepRecord], tokenizer: PreTrainedTokenizerBase) -> dict:
    top5 = sorted(records, key=lambda r: r.normalized_entropy, reverse=True)[:5]
    return {
        "mean_entropy": sum(r.entropy for r in records) / len(records),
        "entropy_trace": [entropy_bucket(r.normalized_entropy) for r in records],
        "most_uncertain": [
            {
                "step": r.step,
                "token": tokenizer.decode([r.chosen_token_id]),
                "normalized_entropy": r.normalized_entropy,
                "top_k": [(tokenizer.decode([tid]), prob) for tid, prob in r.top_k],
            }
            for r in top5
        ],
    }
