from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
from transformers.cache_utils import Cache

from speculative_decoding.speculative_decoding import RoundResult, draft_step, verify_and_accept
from utils.constants import TEMPERATURE, TOP_P, GAMMA


@dataclass
class GenerationStats:
    rounds: List[RoundResult] = field(default_factory=list)

    @property
    def total_proposed(self) -> int:
        return sum(r.num_proposed for r in self.rounds)

    @property
    def total_accepted(self) -> int:
        return sum(r.num_accepted for r in self.rounds)

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted / self.total_proposed if self.total_proposed else 0.0


@torch.no_grad()
def generate(
    draft_model,
    verifier_model,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    eos_token_id: int,
    gamma: int = GAMMA,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> tuple[torch.LongTensor, GenerationStats]:
    """Sampling-mode speculative decoding main loop. Output distribution matches
    sampling from verifier_model alone (Leviathan et al., 2023) -- non-deterministic
    run to run unless the caller fixes torch's RNG seed beforehand.
    """
    stats = GenerationStats()
    generated = input_ids
    n_new = 0
    draft_cache: Cache | None = None
    verify_cache: Cache | None = None

    while n_new < max_new_tokens:
        remaining = max_new_tokens - n_new
        round_gamma = min(gamma, remaining)

        draft_ids, draft_probs, draft_dists, entropy_trace, draft_cache = draft_step(
            draft_model, generated, gamma=round_gamma, temperature=temperature, top_p=top_p,
            cache=draft_cache,
        )
        result, verify_cache = verify_and_accept(
            verifier_model, generated, draft_ids, draft_probs, draft_dists, entropy_trace,
            round_gamma,
            temperature=temperature, top_p=top_p, cache=verify_cache,
        )
        stats.rounds.append(result)

        new_tokens = result.accepted_tokens + [result.bonus_token]
        new_tokens = new_tokens[: max_new_tokens - n_new]
        generated = torch.cat(
            [generated, torch.tensor([new_tokens], device=generated.device)], dim=1
        )
        n_new += len(new_tokens)

        # The verifier's cache already covers `generated` exactly: the bonus/correction
        # token's KV entries came for free from the same forward pass that scored the
        # draft tokens (logits[gamma] in verify_and_accept). Just crop off any rejected
        # draft tokens past num_accepted.
        verify_cache.crop(generated.shape[1])

        # The draft model never saw the bonus/correction token -- it only generated
        # `num_accepted` of the `gamma` tokens itself. Crop to one *before* the last
        # token of `generated` (whatever that token is -- an accepted draft token, a
        # bonus token, or a correction) so next round's draft_step re-ingests it via
        # its normal cache is not None branch, rather than re-deriving from a stale
        # position.
        draft_cache.crop(generated.shape[1] - 1)

        if eos_token_id in new_tokens:
            break

    return generated, stats
