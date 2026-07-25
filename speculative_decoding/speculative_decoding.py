from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache

from utils.constants import ENTROPY_STOP_THRESHOLD, GAMMA, TEMPERATURE, TOP_P

from entropy_tracker.entropy_tracker import compute_entropy
from speculative_decoding.sampling import sample_token, top_p_filter


# Draft proposal length per round. Sampling-mode speculative decoding
# guarantees output == verifier's own sampling distribution only when
# both draft and verifier sample stochastically (see sampling.py for the shared temperature/top_p that both sides use.)


@dataclass
class RoundResult:
    accepted_tokens: List[int]  # tokens accepted this round, in order (draft-sampled)
    bonus_token: (
        int  # either the verifier's correction, or an extra verifier-sampled token
    )
    num_proposed: int  # actual proposed count this round, for acceptance-rate bookkeeping
    num_accepted: int  # len(accepted_tokens)
    entropy_trace: List[float] = field(default_factory=list)  # draft's normalized entropy per proposed position
    requested_gamma: int = 0  # gamma asked for this round, before any entropy-based early stop

    @property
    def stopped_early_on_entropy(self) -> bool:
        return self.num_proposed < self.requested_gamma


@torch.no_grad()
def draft_step(
    draft_model,
    input_ids: torch.Tensor,
    gamma: int = GAMMA,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    cache: Cache | None = None,
    entropy_stop_threshold: float = ENTROPY_STOP_THRESHOLD,
) -> tuple[torch.Tensor, List[float], List[torch.Tensor], List[float], Cache]:
    """Autoregressively sample up to gamma tokens from the draft model using KV cache.
    `cache` holds prior rounds' state (already covering everything up to input_ids[:, -1]);
    when None this is the first round and a full prefill builds it. Every subsequent round
    only needs input_ids[:, -1:] since the rest is already cached from generate.py's bookkeeping.

    Stops proposing early (before gamma) if the draft's own normalized entropy at a position
    is >= entropy_stop_threshold: a draft token the draft itself is very unsure about is likely
    to be rejected anyway, so continuing to propose further tokens conditioned on it tends to be
    wasted work. Always proposes at least one token per round.

    Returns (draft_ids [1, <=gamma], p_draft per sampled token, the full filtered draft
    distribution per proposed position (needed for a correct residual on rejection),
    the draft's normalized entropy per proposed position (for correlating against
    acceptance in Phase 4), updated cache covering the full prefix + all proposed tokens,
    caller must crop it back if any are rejected).
    """
    device = input_ids.device
    draft_probs: List[float] = []
    draft_dists: List[torch.Tensor] = []
    draft_ids_list: List[int] = []
    entropy_trace: List[float] = []

    prefix_len = input_ids.shape[1]

    if cache is None:
        # First round ever: no cache exists yet, prefill the whole prompt
        output = draft_model(input_ids, use_cache=True)
    else:
        # Cache already covers input_ids[:, :-1]; extend it with just the newest token
        output = draft_model(input_ids[:, -1:], past_key_values=cache, use_cache=True)

    assert output.past_key_values is not None
    cache = output.past_key_values

    # Sample up to gamma tokens autoregressively with cache reuse
    for i in range(gamma):
        # Current position in the sequence: where the next token will be predicted
        current_pos = prefix_len + i

        if i == 0:
            logits = output.logits[0, -1]
        else:
            model_input = torch.tensor(
                [[draft_ids_list[-1]]], device=device, dtype=torch.long
            )
            position_ids = torch.tensor([[current_pos - 1]], device=device, dtype=torch.long)
            output = draft_model(
                model_input, past_key_values=cache, position_ids=position_ids, use_cache=True
            )
            cache = output.past_key_values
            logits = output.logits[0, -1]

        _, normalized_entropy = compute_entropy(logits)
        if i > 0 and normalized_entropy >= entropy_stop_threshold:
            break

        token_id, p, dist = sample_token(logits, temperature, top_p)
        draft_ids_list.append(token_id)
        draft_probs.append(p)
        draft_dists.append(dist)
        entropy_trace.append(normalized_entropy)

    draft_ids = torch.tensor([draft_ids_list], device=device, dtype=torch.long)
    return draft_ids, draft_probs, draft_dists, entropy_trace, cache


@torch.no_grad()
def verify_and_accept(
    verifier_model,
    input_ids: torch.Tensor,
    draft_ids: torch.Tensor,
    draft_probs: List[float],
    draft_dists: List[torch.Tensor],
    entropy_trace: List[float],
    requested_gamma: int,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    cache: Cache | None = None,
) -> tuple[RoundResult, Cache]:
    """Single batched forward pass over [last accepted token + draft tokens], reusing
    `cache` (already covering input_ids[:, :-1]) so the verifier never re-prefills the
    whole generated sequence. Then sampling-mode accept/reject: accept token i with
    probability min(1, p_verify(x_i)/p_draft(x_i)); on first rejection, resample from
    the residual max(0, p_verify - p_draft) and stop. If all gamma tokens are accepted,
    sample one bonus token from the verifier's next position for free (the whole point
    of speculative decoding: gamma+1 tokens for the cost of one verifier forward pass
    when acceptance is perfect).

    Returns (result, cache) where cache covers input_ids + all gamma draft tokens --
    caller must crop it back to the accepted length before the next round.
    """
    gamma = draft_ids.shape[1]
    if cache is None:
        # First round ever: no cache exists yet, prefill the whole prompt + draft tokens
        new_ids = torch.cat([input_ids, draft_ids], dim=1)
    else:
        # Cache already covers input_ids[:, :-1]; extend it with the newest token + draft tokens
        new_ids = torch.cat([input_ids[:, -1:], draft_ids], dim=1)
    output = verifier_model(new_ids, past_key_values=cache, use_cache=True)
    # Only the last gamma+1 positions matter: logits[-(gamma+1)] predicts the first
    # draft token, ..., logits[-1] predicts the free bonus token. On round 1, new_ids
    # includes the full prompt, so the score positions must be taken from the tail,
    # not from index 0.
    logits = output.logits[0, -(gamma + 1):]  # [gamma + 1, vocab], logits[i] predicts draft_ids[i] (i<gamma) or bonus (i==gamma)
    assert output.past_key_values is not None
    cache = output.past_key_values

    accepted: List[int] = []
    for i in range(gamma):
        step_logits = logits[i]
        verify_probs = F.softmax(step_logits.float() / temperature, dim=-1)
        verify_probs = top_p_filter(verify_probs, top_p)

        token_id = int(draft_ids[0, i].item())
        p_verify = verify_probs[token_id].item()
        p_draft = draft_probs[i]

        accept_prob = min(1.0, p_verify / p_draft) if p_draft > 0 else 0.0
        if torch.rand(1).item() < accept_prob:
            accepted.append(token_id)
            continue

        # rejection: resample from the residual distribution and stop this round.
        # Uses the draft's full filtered distribution at this position (not just
        # p_draft(token_id)) -- subtracting only the sampled token's mass would
        # leave every other token's verifier probability untouched, overstating
        # residual mass on tokens the draft actually assigned real probability to.
        residual = torch.clamp(verify_probs - draft_dists[i], min=0.0)
        if residual.sum() <= 0:
            correction = int(torch.argmax(verify_probs).item())
        else:
            residual = residual / residual.sum()
            correction = int(torch.multinomial(residual, num_samples=1).item())
        return RoundResult(
            accepted_tokens=accepted,
            bonus_token=correction,
            num_proposed=gamma,
            num_accepted=len(accepted),
            entropy_trace=entropy_trace,
            requested_gamma=requested_gamma,
        ), cache

    # all gamma tokens accepted -- sample a free bonus token from the verifier
    bonus_logits = logits[gamma]
    bonus_probs = top_p_filter(
        F.softmax(bonus_logits.float() / temperature, dim=-1), top_p
    )
    bonus_token = int(torch.multinomial(bonus_probs, num_samples=1).item())
    return RoundResult(
        accepted_tokens=accepted,
        bonus_token=bonus_token,
        num_proposed=gamma,
        num_accepted=len(accepted),
        entropy_trace=entropy_trace,
    ), cache
